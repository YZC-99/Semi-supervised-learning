# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook,PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather



class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)


class Confidence_Estimator(nn.Module):
    def __init__(self, base, num_classes=16):
        super(Confidence_Estimator, self).__init__()
        self.backbone = base
        self.num_features = base.num_features

        self.proj_feat = nn.Sequential(nn.Linear(self.num_features, 64),
                      nn.ReLU())

        self.proj_cls = nn.Sequential(nn.Linear(num_classes, 64),
                                      nn.ReLU())

        self.con_estimator = nn.Sequential(nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64,32),
                                            nn.ReLU()) # sigmoid
        self.last_layer = nn.Linear(32, num_classes)

        torch.nn.init.zeros_(self.last_layer.weight)
        torch.nn.init.zeros_(self.last_layer.bias)

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        feat = self.l2norm(feat)
        class_probs = torch.sigmoid(logits)

        projected_feat = self.proj_feat(feat) # 64
        projected_class_prob = self.proj_cls(class_probs) # 64
        est_input = torch.cat((projected_feat, projected_class_prob), dim=1) # (176,128)
        # print(est_input.shape)
        con_output = self.con_estimator(est_input)
        con_output = self.last_layer(con_output)
        con_output = torch.sigmoid(con_output)

        return {'logits': logits, 'feat': con_output}



    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher




# TODO: move this to criterions
def comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=0.2):
    # embedding similarity
    sim = torch.exp(torch.mm(feats_x_ulb_s_0, feats_x_ulb_s_1.t()) / T)
    sim_probs = sim / sim.sum(1, keepdim=True)
    # contrastive loss
    loss = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
    loss = loss.mean()
    return loss


@ALGORITHMS.register('conmatch')
class ConMatch(AlgorithmBase):
    """
        CoMatch algorithm (https://arxiv.org/abs/2011.11183).
        Reference implementation (https://github.com/salesforce/CoMatch/).

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
            - contrast_p_cutoff (`float`):
                Confidence threshold for contrastive loss. Samples with similarity lower than a threshold are not connected.
            - queue_batch (`int`, *optional*, default to 128):
                Length of the memory bank to store class probabilities and embeddings of the past weakly augmented samples
            - smoothing_alpha (`float`, *optional*, default to 0.999):
                Weight for a smoothness constraint which encourages taking a similar value as its nearby samples’ class probabilities
            - da_len (`int`, *optional*, default to 256):
                Length of the memory bank for distribution alignment.
            - contrast_loss_ratio (`float`, *optional*, default to 1.0):
                Loss weight for contrastive loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # comatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff,
                  contrast_p_cutoff=args.contrast_p_cutoff, hard_label=args.hard_label,
                  queue_batch=args.queue_batch, smoothing_alpha=args.smoothing_alpha, da_len=args.da_len)
        self.lambda_c = args.contrast_loss_ratio

    def init(self, T, p_cutoff, contrast_p_cutoff, hard_label=True, queue_batch=128, smoothing_alpha=0.999, da_len=256):
        self.T = T
        self.p_cutoff = p_cutoff
        self.contrast_p_cutoff = contrast_p_cutoff
        self.use_hard_label = hard_label
        self.queue_batch = queue_batch
        self.smoothing_alpha = smoothing_alpha
        self.da_len = da_len

        # TODO: put this part into a hook
        # memory smoothing
        self.queue_size = int(queue_batch * (
                    self.args.uratio + 1) * self.args.batch_size) if self.args.dataset != 'imagenet' else self.args.K
        self.queue_feats = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_ptr = 0

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = Confidence_Estimator(model, num_classes=self.num_classes)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = Confidence_Estimator(ema_model, num_classes=self.num_classes)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
            outputs = self.model(inputs)
            logits, feats = outputs['logits'], outputs['feat']

            con_loss = torch.tensor(0.0, device=logits.device)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s1, logits_x_ulb_s2 = logits[num_lb:].chunk(3)
            con_lb = feats[:num_lb]
            con_x_ulb_w, con_x_ulb_s1, con_x_ulb_s2 = feats[num_lb:].chunk(3)

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_ulb_s1 = self.compute_prob(logits_x_ulb_s1.detach())
            probs_x_ulb_s2 = self.compute_prob(logits_x_ulb_s2.detach())
            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False,
                                  multi_label=True)
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False,
                                          multi_label=True,
                                          )

            pseudo_label_ulb_s1 = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                            logits=probs_x_ulb_s1,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False,
                                            multi_label=True,
                                            )
            pseudo_label_ulb_s2 = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                            logits=probs_x_ulb_s2,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False,
                                            multi_label=True,
                                            )


            # 对 logits 应用 sigmoid 激活函数
            sm_pred = torch.sigmoid(logits_x_lb)
            # 根据阈值（通常为 0.5）将概率转换为二进制预测
            predicted_label = (sm_pred > 0.5).long()
            # 将预测结果与真实标签进行比较
            con_true_l = torch.eq(predicted_label, y_lb).float() # (8,5)

            # version 1
            # con_loss += self.ce_loss(con_lb, con_true_l, reduction='mean')
            # reg_con_loss_1 = torch.mean(torch.log(1 / con_x_ulb_s1))
            # loss_ours_1 = (con_x_ulb_s1 * self.ce_loss(logits_x_ulb_s1.detach(), logits_x_ulb_w, reduction='none')).mean()
            # reg_con_loss_2 = torch.mean(torch.log(1 / con_x_ulb_s2))
            # loss_ours_2 = (con_x_ulb_s2 * self.ce_loss(logits_x_ulb_s2.detach(), logits_x_ulb_w, reduction='none')).mean()
            #
            # loss_ours_ss_1 = (con_x_ulb_s1.detach() * self.ce_loss(logits_x_ulb_s2, logits_x_ulb_s1, reduction='none')).mean()
            # loss_ours_ss_2 = (con_x_ulb_s2.detach() * self.ce_loss(logits_x_ulb_s1, logits_x_ulb_s2, reduction='none')).mean()
            # version 1

            # version 2
            conf_loss = (logits_x_ulb_s1 * self.ce_loss(logits_x_ulb_s1.detach(), logits_x_ulb_w.detach(),reduction='none')).mean() + \
                        torch.mean(torch.log(1 / logits_x_ulb_s1))

            conf_sup_loss = self.ce_loss(con_x_ulb_w,pseudo_label,reduction='mean')

            ccr_loss = (con_x_ulb_s1 * self.ce_loss(logits_x_ulb_s2,pseudo_label_ulb_s1, reduction='none')).mean() + \
                          (con_x_ulb_s2 * self.ce_loss(logits_x_ulb_s1,pseudo_label_ulb_s2, reduction='none')).mean()





            fixmatch_loss_1 = self.consistency_loss(logits_x_ulb_s1,
                                               pseudo_label,
                                               'bce',
                                               mask=mask)
            fixmatch_loss_2 = self.consistency_loss(logits_x_ulb_s2,
                                                  pseudo_label,
                                                    'bce',
                                                    mask=mask)

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            feat_dict = {}

            unsup_loss = fixmatch_loss_1 + fixmatch_loss_2

            # version1
            # reg_loss_ratio = 0.5
            # total_loss = sup_loss + self.lambda_u * unsup_loss \
            #              + con_loss + reg_loss_ratio * (reg_con_loss_1 + reg_con_loss_2) \
            #              + loss_ours_1 + loss_ours_2 + loss_ours_ss_1 + loss_ours_ss_2
            # version2
            total_loss = sup_loss + self.lambda_u * unsup_loss  + conf_loss + conf_sup_loss + ccr_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['queue_feats'] = self.queue_feats.cpu()
        save_dict['queue_probs'] = self.queue_probs.cpu()
        save_dict['queue_size'] = self.queue_size
        save_dict['queue_ptr'] = self.queue_ptr
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        # save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu() 
        # save_dict['p_target_ptr'] = self.hooks_dict['DistAlignHook'].p_target_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_feats = checkpoint['queue_feats'].cuda(self.gpu)
        self.queue_probs = checkpoint['queue_probs'].cuda(self.gpu)
        self.queue_size = checkpoint['queue_size']
        self.queue_ptr = checkpoint['queue_ptr']
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_target_ptr = checkpoint['p_target_ptr'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--contrast_p_cutoff', float, 0.8),
            SSL_Argument('--contrast_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--queue_batch', int, 128),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
        ]
