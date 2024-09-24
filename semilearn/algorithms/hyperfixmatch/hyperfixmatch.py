# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook,DistAlignQueueHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.loss.supconloss import SupConLoss

def KL(out1,out2,eps=1e-5):
    kl = (out1 * (out1 + eps).log() - out1 * (out2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl

def construct_hypergraph(ex_clinical, pred_classes, feats):
    """
    构建超图 H 和对应的节点嵌入

    Args:
        ex_clinical: [N]，每个样本的 ex_clinical 值
        pred_classes: [N, C]，每个样本的预测类别（多标签，二值化）
        feats: [N, F]，每个样本的特征嵌入

    Returns:
        H: [num_nodes, num_classes]，超图的关联矩阵
        node_embeddings: [num_nodes, F]，节点的嵌入
    """
    N = ex_clinical.size(0)
    device = ex_clinical.device
    num_classes = pred_classes.size(1)  # C

    # 获取 ex_clinical 的唯一值及其在节点中的索引
    unique_ex_clinical, node_indices = torch.unique(ex_clinical, return_inverse=True)
    num_nodes = unique_ex_clinical.size(0)

    # 计算每个节点的特征（对于相同 ex_clinical 值的样本，取特征平均）
    F = feats.size(1)
    node_features = torch.zeros((num_nodes, F), device=device)
    counts = torch.zeros(num_nodes, 1, device=device)

    # 使用 index_add 累加特征和计数
    node_features = node_features.index_add(0, node_indices, feats)
    counts = counts.index_add(0, node_indices, torch.ones((N, 1), device=device))

    # 计算平均特征
    node_features = node_features / counts  # [num_nodes, F]

    # 构建超图关联矩阵 H
    # 获取 pred_classes 中值为 1 的索引
    indices = torch.nonzero(pred_classes)  # [K, 2], K 是值为 1 的元素个数
    sample_indices = indices[:, 0]  # [K]
    class_indices = indices[:, 1]   # [K]

    # 获取对应的节点索引
    node_indices_for_samples = node_indices[sample_indices]  # [K]

    # 创建超图关联矩阵 H
    H = torch.zeros((num_nodes, num_classes), device=device)
    H[node_indices_for_samples, class_indices] = 1  # 设置关联矩阵中的值为 1

    return H, node_features


class HGNN(nn.Module):
    def __init__(self, nb_classes, sz_embed, hidden):
        super(HGNN, self).__init__()

        self.theta1 = nn.Linear(sz_embed, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lrelu = nn.LeakyReLU(0.1)

        self.theta2 = nn.Linear(hidden, nb_classes)

    def compute_G(self, H):
        # the number of hyperedge
        n_edge = H.size(1)
        # the weight of the hyperedge
        we = torch.ones(n_edge).cuda()
        # the degree of the node
        Dv = (H * we).sum(dim=1)
        # the degree of the hyperedge
        De = H.sum(dim=0)

        We = torch.diag(we)
        inv_Dv_half = torch.diag(torch.pow(Dv, -0.5))
        inv_Dv_half[torch.isinf(inv_Dv_half)] = 0
        inv_De = torch.diag(torch.pow(De, -1))
        inv_De[torch.isinf(inv_De)] = 0
        H_T = torch.t(H)

        # propagation matrix
        G = torch.chain_matmul(inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half)

        return G

    def forward(self, X, H):
        # 这里的X是每个节点的embedding, H是超图的邻接矩阵
        G = self.compute_G(H)

        # 1st layer
        X = G.matmul(self.theta1(X))
        X = self.bn1(X)
        X = self.lrelu(X)

        # 2nd layer
        out = G.matmul(self.theta2(X))

        return out


class HyperMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128, epass=False, num_classes=5):
        super(HyperMatch_Net, self).__init__()
        self.backbone = base
        self.epass = epass
        self.num_features = base.num_features

        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])

        if self.epass:
            self.mlp_proj_2 = nn.Sequential(*[
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(inplace=False),
                nn.Linear(self.num_features, proj_size)
            ])

            self.mlp_proj_3 = nn.Sequential(*[
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(inplace=False),
                nn.Linear(self.num_features, proj_size)
            ])
        self.hgnn = HGNN(nb_classes=num_classes, sz_embed=proj_size, hidden=proj_size)
        self.hgnn2 = HGNN(nb_classes=num_classes, sz_embed=proj_size, hidden=proj_size)

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def stage1_forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        feat_proj1 = F.normalize(self.mlp_proj(feat), p=2, dim=1)
        feat_proj2 = F.normalize(self.mlp_proj_2(feat), p=2, dim=1)
        feat_proj3 = F.normalize(self.mlp_proj_3(feat), p=2, dim=1)

        if self.epass:
            feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat)) / 3)
        else:
            feat_proj = self.l2norm(self.mlp_proj(feat))

        return {'logits': logits, 'feat': feat_proj, 'feat_proj1': feat_proj1, 'feat_proj2': feat_proj2,
                'feat_proj3': feat_proj3}
    def stage2_forward(self, embeddings,H,embeddings2,H2, **kwargs):
        logits = self.hgnn(embeddings, H)
        logits2 = self.hgnn2(embeddings2, H2)
        return {
            'hyper_logits': logits,
            'hyper_logits2': logits2,
                }

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        feat_proj1 = F.normalize(self.mlp_proj(feat), p=2, dim=1)
        feat_proj2 = F.normalize(self.mlp_proj_2(feat), p=2, dim=1)
        feat_proj3 = F.normalize(self.mlp_proj_3(feat), p=2, dim=1)

        if self.epass:
            feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat)) / 3)
        else:
            feat_proj = self.l2norm(self.mlp_proj(feat))

        return {'logits': logits, 'feat': feat_proj, 'feat_proj1': feat_proj1, 'feat_proj2': feat_proj2,
                'feat_proj3': feat_proj3}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('hyperfixmatch')
class HyperFixMatch(AlgorithmBase):

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
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = HyperMatch_Net(model, proj_size=self.args.proj_size, epass=self.args.use_epass)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = HyperMatch_Net(ema_model, proj_size=self.args.proj_size, epass=self.args.use_epass)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model
    def train_step(self, x_lb, y_lb,in_clinical, x_ulb_w, x_ulb_s_0, x_ulb_s_1,ex_clinical):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1)) #
            clinical = torch.cat([in_clinical, ex_clinical,ex_clinical,ex_clinical], dim=0)
            clinical2 = clinical
            clinical3 = clinical
            if 'eyeid' == self.clinical_mode:
                clinical = clinical[:][:,-4]
            elif 'bcva' == self.clinical_mode:
                clinical = clinical[:][:,-3]
            elif 'cst' == self.clinical_mode:
                clinical = clinical[:][:,-2]
            elif 'patientid' == self.clinical_mode:
                clinical = clinical[:][:,-1]
            elif 'eyeid-cst' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -2]
            elif 'eyeid-bcva' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -3]
            elif 'bcva-cst' == self.clinical_mode:
                clinical = clinical[:][:, -3]
                clinical2 = clinical2[:][:, -2]
            elif 'eyeid-bcva-cst' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -3]
                clinical3 = clinical3[:][:, -2]
            else:
                clinical = clinical[:][:,-1]
            outputs = self.model.stage1_forward((inputs))
            logits, feats,feat_proj1,feat_proj2,feat_proj3 = outputs['logits'], outputs['feat'],outputs['feat_proj1'],outputs['feat_proj2'],outputs['feat_proj3']
            logits_x_lb, feats_x_lb = logits[:num_lb], feats[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s_0, _ = logits[num_lb:].chunk(3)
            feats_x_ulb_w, feats_x_ulb_s_0, feats_x_ulb_s_1 = feats[num_lb:].chunk(3)
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s':[feats_x_ulb_s_0, feats_x_ulb_s_1]}
            # 获取未标记数据的 ex_clinical 和类别预测
            ex_clinical_ulb_w = clinical[num_lb:num_lb + num_ulb]
            ex_clinical_ulb_s0 = clinical[num_lb + num_ulb:num_lb + 2 * num_ulb]
            # 获取类别预测
            pred_classes_ulb_w = (torch.sigmoid(logits_x_ulb_w) > 0.5).float()
            pred_classes_ulb_s0 = (torch.sigmoid(logits_x_ulb_s_0) > 0.5).float()
            # 构建超图 H1（针对 x_ulb_w）
            H1, node_embeddings1 = construct_hypergraph(ex_clinical_ulb_w, pred_classes_ulb_w, feats_x_ulb_w)
            # print(H1)
            # 构建超图 H2（针对 x_ulb_s_0）
            H2, node_embeddings2 = construct_hypergraph(ex_clinical_ulb_s0, pred_classes_ulb_s0, feats_x_ulb_s_0)
            hyper_out = self.model.stage2_forward(node_embeddings1, H1,node_embeddings2,H2)
            hyper_weak_out,hyper_strong_out = hyper_out['hyper_logits'],hyper_out['hyper_logits2']
            # print(hyper_weak_out.shape,hyper_strong_out.shape)
            # hyper_loss = F.mse_loss(hyper_weak_out.detach(),hyper_strong_out)
            hyper_weak_out = nn.functional.softmax(hyper_weak_out)
            hyper_strong_out = nn.functional.softmax(hyper_strong_out)
            # hyper_loss = KL(hyper_weak_out,hyper_strong_out) + KL(hyper_strong_out,hyper_weak_out)
            hyper_loss = nn.functional.kl_div(hyper_weak_out,hyper_strong_out) + nn.functional.kl_div(hyper_strong_out,hyper_weak_out)
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # supcon loss
            self.supconloss.device = feats_x_ulb_w.device
            clinical_supcon_feats = torch.cat([feats_x_ulb_w.unsqueeze(1),feats_x_ulb_s_0.unsqueeze(1),feats_x_ulb_s_1.unsqueeze(1)],dim=1)
            # print(clinical_supcon_feats.shape,clinical[num_lb:].shape)
            if 'simclr' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats)
            elif 'eyeid-cst' == self.clinical_mode or 'eyeid-bcva' == self.clinical_mode or 'bcva-cst' == self.clinical_mode :
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[num_lb:num_lb + num_ulb]) + self.supconloss(clinical_supcon_feats,clinical2[num_lb:num_lb + num_ulb])
            elif 'eyeid-bcva-cst' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[num_lb:num_lb + num_ulb]) + self.supconloss(clinical_supcon_feats,clinical2[num_lb:num_lb + num_ulb]) + self.supconloss(clinical_supcon_feats,clinical3[num_lb:num_lb + num_ulb])
            else:
                sup_con_loss = self.supconloss(clinical_supcon_feats,clinical[num_lb:num_lb + num_ulb])

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

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
            unsup_loss = self.consistency_loss(logits_x_ulb_s_0,
                                               pseudo_label,
                                               self.args.loss,
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + hyper_loss + sup_con_loss

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
            SSL_Argument('--contrast_p_cutoff', float, 0.8),
            SSL_Argument('--contrast_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--queue_batch', int, 128),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
        ]