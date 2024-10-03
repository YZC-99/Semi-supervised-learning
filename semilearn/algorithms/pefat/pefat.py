# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from sklearn.mixture import GaussianMixture

def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x

@ALGORITHMS.register('pefat')
class PEFAT(AlgorithmBase):
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

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.ema_gmm =None

    def set_hooks(self):
        lb_class_dist = [0 for _ in range(self.num_classes)]
        print(self.dataset_dict['train_lb'].targets.shape)  # (370,16)
        # 计算总和，得到(n,16)
        target_sum = self.dataset_dict['train_lb'].targets.sum(axis=0)
        lb_class_dist = target_sum
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, p_target_type='gt', p_target=lb_class_dist),
            "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def compute_ema_gmm(self,x_type,device):
        if self.ema_gmm is None:
            with torch.no_grad():
                all_losses = torch.tensor([]).cuda()
                for id, data in enumerate(self.dataset_dict['train_lb']):
                    if id > 21:
                        break
                    x_lb, y_lb = data['x_lb'].to(x_type).to(device), torch.tensor(data['y_lb']).to(x_type).to(device)
                    outputs = self.model(x_lb)['logits']
                    loss_cls_sup = F.cross_entropy(outputs, y_lb, reduction='none')
                    all_losses = torch.cat((all_losses, loss_cls_sup), -1)

            all_losses = all_losses.view(-1)
            all_losses = (all_losses-all_losses.min())/(all_losses.max()-all_losses.min())
            all_losses = all_losses.view(-1,1)
            self.ema_gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)
            self.ema_gmm.fit(all_losses.cpu())

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        self.compute_ema_gmm(x_lb.dtype, x_lb.device)
        # designed for unlabeled data
        with torch.no_grad():
            un_output1 = self.model(x_ulb_w)['logits']
            un_output2 = self.model(x_ulb_s)['logits']
            un_output1_softmax = F.softmax(un_output1, dim=1)
            un_output2_softmax = F.softmax(un_output2, dim=1)

            un_prob1, un_pred1 = torch.max(un_output1_softmax, dim=1)
            un_prob2, un_pred2 = torch.max(un_output2_softmax, dim=1)
            un_prob, un_pred = torch.max(0.5 * (un_output1_softmax + un_output2_softmax), dim=1)
            un_prob1 = un_prob1.cpu().numpy()
            un_prob2 = un_prob2.cpu().numpy()
            un_prob = un_prob.cpu().numpy()

            los_12 = torch.nn.CrossEntropyLoss()(un_output2, un_pred1)
            los_21 = torch.nn.CrossEntropyLoss()(un_output1, un_pred2)

            b_size = x_ulb_w.shape[0]
            loss_tmp12 = torch.zeros(b_size)
            loss_tmp21 = torch.zeros(b_size)

            for r in range(b_size):
                loss_tmp12[r] = los_12[r]
                loss_tmp21[r] = los_21[r]

            loss_tmp12 = (loss_tmp12 - loss_tmp12.min()) / (loss_tmp12.max() - loss_tmp12.min())
            loss_tmp21 = (loss_tmp21 - loss_tmp21.min()) / (loss_tmp21.max() - loss_tmp21.min())

            loss_tmp12 = loss_tmp12.view(-1, 1)
            loss_tmp21 = loss_tmp21.view(-1, 1)

            prob12 = self.ema_gmm.predict_proba(loss_tmp12)
            prob12 = prob12[:, self.ema_gmm.means_.argmin()]

            prob21 = self.ema_gmm.predict_proba(loss_tmp21)
            prob21 = prob21[:, self.ema_gmm.means_.argmin()]

            prob_comb = self.ema_gmm.predict_proba(0.5 * loss_tmp12 + 0.5 * loss_tmp21)
            prob_comb = prob_comb[:, self.ema_gmm.means_.argmin()]

        trust_idx = prob_comb > 0.70  # 0.5
        unc_idx = trust_idx == False
        ltru = np.sum(trust_idx)
        lunc = np.sum(unc_idx)
        # divide into two categories based on gmm prediction
        trust_img, trust_lab = torch.cat([x_lb, x_ulb_w[trust_idx], x_ulb_s[trust_idx]]), \
                               torch.cat([y_lb, un_pred1[trust_idx], un_pred2[trust_idx]])
        uncer_img1,uncer_img2,mps_lab1,mps_lab2 = x_ulb_w[unc_idx],x_ulb_s[unc_idx],un_pred1[unc_idx],un_pred2[unc_idx]


        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outputs = self.model(trust_img)
            trust_feat, trust_output = outputs['feat'], outputs['logits']
            t_o_s = torch.softmax(trust_output, dim=1)
            #add perturbation on trustworthy images
            d = np.random.normal(size=trust_feat.shape)
            d = normalize_l2(d)
            for iter_num in range(8):
                x_d = torch.tensor(trust_feat.clone().detach().cpu().data.numpy() + d.astype(np.float32),
                                   requires_grad=True)
                t_d_logit = self.model.classifier(x_d.cuda())
                cls_loss_td = F.cross_entropy(t_d_logit, trust_lab, reduction='mean')
                cls_loss_td.backward(retain_graph=True)
                d = x_d.grad
                d = d.numpy()
            trust_feat_at = trust_feat.clone().detach().cpu().data.numpy() + d
            trust_feat_at = torch.tensor(trust_feat_at).cuda()
            t_outputs_at = self.model.classifier(trust_feat_at)
            los_t_ce = F.cross_entropy(trust_output, trust_lab) + 0.3 * F.cross_entropy(t_outputs_at, trust_lab)

            # los for the rest uncertainty image
            feat_u1, unc_output1 = self.model(uncer_img1)
            feat_u2, unc_output2 = self.model(uncer_img2)
            unc_o_s1 = torch.softmax(unc_output1, dim=1)
            unc_o_s2 = torch.softmax(unc_output2, dim=1)

            du1 = np.random.normal(size=feat_u1.shape)
            du2 = np.random.normal(size=feat_u2.shape)
            for iter_num in range(8):
                x_du1 = torch.tensor(feat_u1.clone().detach().cpu().data.numpy() + 1e-3 * du1.astype(np.float32),
                                     requires_grad=True)
                u_d_logit1 = self.model.classifier(x_du1.cuda())
                u_d1_s = torch.softmax(u_d_logit1, dim=1)
                x_du2 = torch.tensor(feat_u2.clone().detach().cpu().data.numpy() + 1e-3 * du2.astype(np.float32),
                                     requires_grad=True)
                u_d_logit2 = self.model.classifier(x_du2.cuda())
                u_d2_s = torch.softmax(u_d_logit2, dim=1)

                cls_loss_ud = F.kl_div(u_d2_s.log(), unc_o_s1.detach(), reduction='batchmean') + F.kl_div(u_d1_s.log(),
                                                                                                          unc_o_s2.detach(),
                                                                                                          reduction='batchmean')
                cls_loss_ud.backward(retain_graph=True)
                du1 = x_du1.grad
                du2 = x_du2.grad
                du1 = du1.numpy()
                du2 = du2.numpy()
                du1 = normalize_l2(du1)
                du2 = normalize_l2(du2)

            uncer_feat_at1 = feat_u1.clone().detach().cpu().data.numpy() + du1
            uncer_feat_at2 = feat_u2.clone().detach().cpu().data.numpy() + du2
            uncer_feat_at1 = torch.tensor(uncer_feat_at1).cuda()
            uncer_feat_at2 = torch.tensor(uncer_feat_at2).cuda()
            unc_output_at1 = self.model.classifier(uncer_feat_at1)
            unc_output_at2 = self.model.classifier(uncer_feat_at2)
            unc_o_ats1 = torch.softmax(unc_output_at1, dim=1)
            unc_o_ats2 = torch.softmax(unc_output_at2, dim=1)
            los_unc_at = F.kl_div(unc_o_ats2.log(), unc_o_s1, reduction='batchmean') + F.kl_div(unc_o_ats1.log(),
                                                                                                unc_o_s2,
                                                                                                reduction='batchmean')


            total_loss = los_t_ce + 0.1*los_unc_at

        out_dict = self.process_out_dict(loss=total_loss, feat=None)
        log_dict = self.process_log_dict(sup_loss=total_loss.item(),
                                         unsup_loss=total_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=None)
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]