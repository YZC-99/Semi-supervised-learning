# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook,DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.loss.supconloss import SupConLoss

@ALGORITHMS.register('clinfixmatch')
class ClinFixMatch(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.clinical_mode = args.clinical
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.supconloss = SupConLoss()
    
    def set_hooks(self):
        lb_class_dist = [0 for _ in range(self.num_classes)]
        print(self.dataset_dict['train_lb'].targets.shape) # (370,16)
        # 计算总和，得到(n,16)
        target_sum = self.dataset_dict['train_lb'].targets.sum(axis=0)
        lb_class_dist = target_sum
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        # self.register_hook(
        #     DistAlignEMAHook(num_classes=self.num_classes, p_target_type='gt', p_target=lb_class_dist),
        #     "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb_w,x_lb_s, y_lb,in_clinical, x_ulb_w, x_ulb_s,ex_clinical):
        num_lb = y_lb.shape[0]
        clinical = torch.cat([in_clinical,ex_clinical],dim=0)
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb_w,x_lb_s, x_ulb_w, x_ulb_s))
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits'][:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            feats_x_lb = outputs['feat'][:num_lb]
            feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)

            clinical_feats = outputs['clinical_feat']
            clinical_feats_lb_w,clinical_feats_lb_s = clinical_feats[num_lb*2:].chunk(2)
            clinical_feats_ulb_w,clinical_feats_ulb_s = clinical_feats[:num_lb*2].chunk(2)

            clinical_feats_w = torch.cat([clinical_feats_lb_w,clinical_feats_ulb_w],dim=0)
            clinical_feats_s = torch.cat([clinical_feats_lb_s,clinical_feats_ulb_s],dim=0)

            clinical_supcon_feats = torch.cat([clinical_feats_w.unsqueeze(1),clinical_feats_s.unsqueeze(1)],dim=1)

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            # print(clinical_feats_x_lb_w.shape)
            # print(clinical_feats_x_lb_s.shape)
            # print(clinical_feats_x_ulb_w.shape)
            # print(clinical_feats_x_ulb_s.shape)
            self.supconloss.device = feats_x_ulb_w.device
            if 'eyeid' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[:][:,-4])
            elif 'bcva' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[:][:,-3])
            elif 'cst' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[:][:,-2])
            elif 'patientid' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[:][:,-1])
            elif 'simclr' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats)
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST

            # probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w)

            multi_label = True if self.args.loss == 'bce' else False

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False,multi_label=multi_label)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False,
                                          multi_label=multi_label,
                                          )
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               self.args.loss,
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + sup_con_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]