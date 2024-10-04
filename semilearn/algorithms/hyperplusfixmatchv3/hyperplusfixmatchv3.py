# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignQueueHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.loss.supconloss import SupConLoss
from sklearn.mixture import GaussianMixture

from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import numpy as np

def fit_gmm_on_DDL(DDL, n_components=5):
    """
    使用GMM对DDL进行建模。

    Args:
        DDL: [num_classes, num_classes] 分布差异矩阵
        n_components: int GMM的混合成分数

    Returns:
        gmm: 训练好的GaussianMixture模型
    """
    DDL_flat = DDL.view(-1).cpu().detach().numpy().reshape(-1, 1)  # [num_classes * num_classes, 1]
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(DDL_flat)
    return gmm

def compute_DDU_likelihood(gmm, DDU):
    """
    计算未标记样本的DDU在GMM模型下的似然概率。

    Args:
        gmm: 训练好的GaussianMixture模型
        DDU: [num_ulb, num_classes] 未标记数据的分布差异

    Returns:
        likelihoods: [num_ulb] 每个未标记样本的总似然概率
    """
    num_ulb = DDU.size(0)
    # 假设 DDU 是 [num_ulb, num_classes]
    # 需要将其展平为与DDL相同的维度
    DDU_flat = DDU.view(num_ulb, -1).cpu().detach().numpy()  # [num_ulb, num_classes * num_classes]
    log_probs = gmm.score_samples(DDU_flat)  # [num_ulb]
    likelihoods = np.exp(log_probs)  # 转换为概率
    return torch.tensor(likelihoods, device=DDU.device)

def select_high_confidence_samples(likelihoods, threshold=0.95):
    """
    选择高可信度的未标记样本，并返回与原始形状相同的掩码。

    Args:
        likelihoods (torch.Tensor): [num_ulb] 每个未标记样本的似然概率
        threshold (float): 置信度阈值

    Returns:
        mask (torch.Tensor): [num_ulb] 选择的未标记样本掩码，符合条件的为1，否则为0
    """
    mask = (likelihoods > threshold).float()
    return mask

def KL(out1, out2, eps=1e-5):
    kl = (out1 * (out1 + eps).log() - out1 * (out2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl


def construct_hypergraph_multilabel(ex_clinical, pred_classes, feats):
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
    class_indices = indices[:, 1]  # [K]

    # 获取对应的节点索引
    node_indices_for_samples = node_indices[sample_indices]  # [K]

    # 创建超图关联矩阵 H
    H = torch.zeros((num_nodes, num_classes), device=device)
    H[node_indices_for_samples, class_indices] = 1  # 设置关联矩阵中的值为 1

    return H, node_features




def construct_hypergraph_single_label_whole(ex_clinical, pred_classes, feats):
    """
    构建超图 H 和对应的节点嵌入（单标签分类），每个样本都是一个节点。

    Args:
        ex_clinical: [N]，每个样本的 ex_clinical 值
        pred_classes: [N]，每个样本的预测类别（单标签，类别索引）
        feats: [N, F]，每个样本的特征嵌入

    Returns:
        H: [N, num_hyperedges]，超图的关联矩阵
        node_features: [N, F]，节点的嵌入（与 feats 相同）
        node_indices_map: [N]，每个节点对应的原始样本索引（0 到 N-1）
    """
    N = ex_clinical.size(0)
    device = ex_clinical.device
    F = feats.size(1)

    # 每个样本都是一个节点
    node_features = feats  # [N, F]
    node_indices_map = torch.arange(N, device=device)  # [N]

    # 构建超边
    # 可以根据 ex_clinical 值和 pred_classes 构建两种类型的超边

    # 1. 基于 ex_clinical 构建超边：将具有相同 ex_clinical 值的样本连接在一起
    unique_ex_clinical, inverse_indices = torch.unique(ex_clinical, sorted=True, return_inverse=True)
    num_ex_clinical = unique_ex_clinical.size(0)

    # 2. 基于 pred_classes 构建超边：将预测类别相同的样本连接在一起
    unique_classes = torch.arange(pred_classes.max().item() + 1, device=device)
    num_classes = unique_classes.size(0)

    # 总的超边数量
    num_hyperedges = num_ex_clinical + num_classes

    # 构建关联矩阵 H，形状为 [N, num_hyperedges]
    H = torch.zeros((N, num_hyperedges), device=device)

    # 添加基于 ex_clinical 的超边
    for idx in range(num_ex_clinical):
        # 获取 ex_clinical 值为 unique_ex_clinical[idx] 的样本索引
        samples_in_hyperedge = torch.nonzero(inverse_indices == idx).squeeze()
        H[samples_in_hyperedge, idx] = 1  # 超边索引从 0 开始

    # 添加基于 pred_classes 的超边
    for idx in range(num_classes):
        # 获取 pred_classes 值为 idx 的样本索引
        samples_in_hyperedge = torch.nonzero(pred_classes == idx).squeeze()
        H[samples_in_hyperedge, num_ex_clinical + idx] = 1  # 超边索引从 num_ex_clinical 开始

    return H, node_features, node_indices_map

def construct_hypergraph_multi_label_whole(ex_clinical, pred_classes, feats):
    """
    构建超图 H 和对应的节点嵌入（多标签分类），每个样本都是一个节点。

    Args:
        ex_clinical: [N]，每个样本的 ex_clinical 值
        pred_classes: [N, C]，每个样本的预测类别（多标签，二值化）
        feats: [N, F]，每个样本的特征嵌入

    Returns:
        H: [N, num_hyperedges]，超图的关联矩阵
        node_features: [N, F]，节点的嵌入（与 feats 相同）
        node_indices_map: [N]，每个节点对应的原始样本索引（0 到 N-1）
    """
    N = ex_clinical.size(0)
    device = ex_clinical.device
    F = feats.size(1)

    # 每个样本都是一个节点
    node_features = feats  # [N, F]
    node_indices_map = torch.arange(N, device=device)  # [N]

    # 构建超边
    # 可以根据 ex_clinical 值和 pred_classes 构建两种类型的超边

    # 1. 基于 ex_clinical 构建超边：将具有相同 ex_clinical 值的样本连接在一起
    unique_ex_clinical, inverse_indices = torch.unique(ex_clinical, sorted=True, return_inverse=True)
    num_ex_clinical = unique_ex_clinical.size(0)

    # 2. 基于 pred_classes 构建超边：将包含相同标签的样本连接在一起（多标签情况）
    num_classes = pred_classes.size(1)

    # 总的超边数量
    num_hyperedges = num_ex_clinical + num_classes

    # 构建关联矩阵 H，形状为 [N, num_hyperedges]
    H = torch.zeros((N, num_hyperedges), device=device)

    # 添加基于 ex_clinical 的超边
    for idx in range(num_ex_clinical):
        # 获取 ex_clinical 值为 unique_ex_clinical[idx] 的样本索引
        samples_in_hyperedge = torch.nonzero(inverse_indices == idx).squeeze()
        H[samples_in_hyperedge, idx] = 1  # 超边索引从 0 开始

    # 添加基于 pred_classes 的超边（多标签处理）
    for idx in range(num_classes):
        # 获取 pred_classes 中类别 idx 对应的标签为 1 的样本索引
        samples_in_hyperedge = torch.nonzero(pred_classes[:, idx] == 1).squeeze()
        H[samples_in_hyperedge, num_ex_clinical + idx] = 1  # 超边索引从 num_ex_clinical 开始

    return H, node_features, node_indices_map





def comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=0.5):
    # embedding similarity
    sim = torch.exp(torch.mm(feats_x_ulb_s_0, feats_x_ulb_s_1.t())/ T) # 构建相似性
    sim_probs = sim / sim.sum(1, keepdim=True) # 归一化
    # contrastive loss
    loss = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
    loss = loss.mean()
    return loss

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

        # self.mlp_proj = nn.Sequential(*[
        #     nn.Linear(self.num_features, self.num_features),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(self.num_features, proj_size)
        # ])
        #
        # if self.epass:
        #     self.mlp_proj_2 = nn.Sequential(*[
        #         nn.Linear(self.num_features, self.num_features),
        #         nn.ReLU(inplace=False),
        #         nn.Linear(self.num_features, proj_size)
        #     ])
        #
        #     self.mlp_proj_3 = nn.Sequential(*[
        #         nn.Linear(self.num_features, self.num_features),
        #         nn.ReLU(inplace=False),
        #         nn.Linear(self.num_features, proj_size)
        #     ])
        self.hgnn = HGNN(nb_classes=base.num_features, sz_embed=base.num_features, hidden=base.num_features)
        self.hgnn2 = HGNN(nb_classes=base.num_features, sz_embed=base.num_features, hidden=base.num_features)
        self.hgnn3 = HGNN(nb_classes=base.num_features, sz_embed=base.num_features, hidden=base.num_features)

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def stage1_forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        # feat_proj1 = F.normalize(self.mlp_proj(feat), p=2, dim=1)
        # feat_proj2 = F.normalize(self.mlp_proj_2(feat), p=2, dim=1)
        # feat_proj3 = F.normalize(self.mlp_proj_3(feat), p=2, dim=1)
        #
        # if self.epass:
        #     feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat)) / 3)
        # else:
        #     feat_proj = self.l2norm(self.mlp_proj(feat))

        return {'logits': logits, 'feat': feat, 'feat_proj1': feat, 'feat_proj2': feat,
                'feat_proj3': feat}

    def stage2_forward(self, embeddings, H, embeddings2, H2, embeddings3, H3, **kwargs):
        logits = self.hgnn(embeddings, H)
        logits2 = self.hgnn2(embeddings2, H2)
        logits3 = self.hgnn3(embeddings3, H3)
        return {
            'hyper_logits1': logits,
            'hyper_logits2': logits2,
            'hyper_logits3': logits3,
        }

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        feat_proj1 = feat
        feat_proj2 = feat
        feat_proj3 = feat



        return {'logits': logits, 'feat': feat, 'feat_proj1': feat, 'feat_proj2': feat,
                'feat_proj3': feat_proj3}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('hyperplusfixmatchv3')
class HyperPlusFixMatchV3(AlgorithmBase):
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
        #memory moothing
        self.global_post_means = torch.zeros(self.args.num_classes).cuda(self.gpu)
        self.global_neg_means = torch.zeros(self.args.num_classes).cuda(self.gpu)

    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = HyperMatch_Net(model, proj_size=self.args.proj_size, epass=self.args.use_epass,
                               num_classes=self.num_classes)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = HyperMatch_Net(ema_model, proj_size=self.args.proj_size, epass=self.args.use_epass,
                                   num_classes=self.num_classes)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    def train_step(self, x_lb, y_lb, in_clinical, x_ulb_w, x_ulb_s_0, x_ulb_s_1, ex_clinical):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0))  #
            clinical = torch.cat([in_clinical, ex_clinical, ex_clinical], dim=0)
            clinical2 = clinical
            clinical3 = clinical
            if 'eyeid' == self.clinical_mode or 'localization' == self.clinical_mode or 'view_position' == self.clinical_mode:
                clinical = clinical[:][:, -4]
            elif 'bcva' == self.clinical_mode or 'sex' == self.clinical_mode or 'patient_gender' == self.clinical_mode:
                clinical = clinical[:][:, -3]
            elif 'cst' == self.clinical_mode or 'age' == self.clinical_mode or 'patient_age' == self.clinical_mode:
                clinical = clinical[:][:, -2]
            elif 'patientid' == self.clinical_mode or 'lesion_id' == self.clinical_mode or 'patient_id' == self.clinical_mode:
                clinical = clinical[:][:, -1]
            elif 'eyeid-bcva' == self.clinical_mode or 'localization-sex' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -3]
            elif 'eyeid-cst' == self.clinical_mode or 'localization-age' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -2]
            elif 'localization-leison_id' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -1]
            elif 'bcva-cst' == self.clinical_mode or 'sex-age' == self.clinical_mode:
                clinical = clinical[:][:, -3]
                clinical2 = clinical2[:][:, -2]
            elif 'sex-lesion_id' in self.clinical_mode:
                clinical = clinical[:][:, -3]
                clinical2 = clinical2[:][:, -1]
            elif 'eyeid-bcva-cst' == self.clinical_mode:
                clinical = clinical[:][:, -4]
                clinical2 = clinical2[:][:, -3]
                clinical3 = clinical3[:][:, -2]
            else:
                clinical = clinical[:][:, -1]
            outputs = self.model.stage1_forward((inputs))
            logits, feats, feat_proj1, feat_proj2, feat_proj3 = outputs['logits'], outputs['feat'], outputs[
                'feat_proj1'], outputs['feat_proj2'], outputs['feat_proj3']
            logits_x_lb, feats_x_lb = logits[:num_lb], feats[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s_0 = logits[num_lb:].chunk(2)
            feats_x_ulb_w, feats_x_ulb_s_0 = feats[num_lb:].chunk(2)
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': [feats_x_ulb_s_0]}
            if self.ce_loss == 'ce':
                probs = F.softmax(logits_x_ulb_w, dim=1)
            else:
                probs = F.sigmoid(logits_x_ulb_w)
            probs = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs.detach())

            # 获取未标记数据的 ex_clinical 和类别预测
            ex_clinical_ulb_w = clinical[num_lb:num_lb + num_ulb]
            ex_clinical_ulb_s0 = clinical[num_lb + num_ulb:num_lb + 2 * num_ulb]
            # 获取类别预测
            if self.ce_loss == 'ce':
                # 构建超图,针对标记数据.这个超图是绝对的，不会改变
                H_lb, node_embeddings_lb, _ = construct_hypergraph_single_label_whole(clinical[:num_lb],
                                                                                torch.argmax(y_lb, dim=1),
                                                                                feats[:num_lb])
                # 构建超图 针对 x_ulb_s_0）
                pred_classes_ulb_s0 = logits_x_ulb_s_0.argmax(dim=1)
                H_ulb, node_embeddings_ulb, _ = construct_hypergraph_single_label_whole(ex_clinical_ulb_s0,
                                                                                  pred_classes_ulb_s0,
                                                                                  feats_x_ulb_s_0)
            else:
                H_lb, node_embeddings_lb, _ = construct_hypergraph_multi_label_whole(clinical[:num_lb], y_lb, feats[:num_lb])
                pred_classes_ulb_s0 = (torch.sigmoid(logits_x_ulb_s_0) > 0.5).float()
                H_ulb, node_embeddings_ulb, _ = construct_hypergraph_multi_label_whole(ex_clinical_ulb_s0, pred_classes_ulb_s0, feats_x_ulb_s_0)


            # 构建全部的超图
            # 这里使用伪标签的目的主要是为了 ，从而实现样本之间的信息传递，但这里的伪标签带有极大的噪声
            if self.ce_loss == 'ce':
                pred_classes_ulb_w = logits_x_ulb_w.argmax(dim=1)
                label_classes = torch.cat((torch.argmax(y_lb, dim=1), pred_classes_ulb_w), dim=0)
                H_whole, node_embeddings_whole, indices_map = construct_hypergraph_single_label_whole(
                    clinical[:num_lb + num_ulb],
                    label_classes,
                    feats[:num_lb + num_ulb])
            else:
                pred_classes_ulb_w = (torch.sigmoid(logits_x_ulb_w) > 0.5).float()
                label_classes = torch.cat((y_lb, pred_classes_ulb_w), dim=0)
                H_whole, node_embeddings_whole, indices_map = construct_hypergraph_multi_label_whole(
                    clinical[:num_lb + num_ulb],
                    label_classes,
                    feats[:num_lb + num_ulb])



            hyper_out = self.model.stage2_forward(node_embeddings_lb, H_lb, node_embeddings_ulb, H_ulb, node_embeddings_whole, H_whole)

            # 构建了三个超图,并且有了三个超图的输出.
            # 现在希望三个超图的features特征维度和backbone的特征维度是一样的,从而能够让超图的特征和原始分类头共享参数

            hyper_lb_out, hyper_ulb_out, hyper_whole_out = hyper_out['hyper_logits1'], hyper_out['hyper_logits2'], hyper_out['hyper_logits3']
            # 获得了三个超图的features,并且和原始的分类头的features是一样的,注意,这里的features是没有归一化的
            # 归一化特征
            hyper_lb_out_norm = F.normalize(hyper_lb_out, p=2, dim=1)
            hyper_ulb_out_norm = F.normalize(hyper_ulb_out, p=2, dim=1)
            hyper_whole_out_norm = F.normalize(hyper_whole_out, p=2, dim=1)
            # 常规的视图
            # 特征归一化
            feats_x_lb = F.normalize(feats_x_lb, p=2, dim=1)
            normal_sup_feats_lb = torch.cat([feats_x_lb.unsqueeze(1), feats_x_lb.unsqueeze(1)], dim=1)
            feats_x_ulb_w = F.normalize(feats_x_ulb_w, p=2, dim=1)
            feats_x_ulb_s_0 = F.normalize(feats_x_ulb_s_0, p=2, dim=1)
            normal_sup_feats_ulb = torch.cat([feats_x_ulb_w.unsqueeze(1), feats_x_ulb_s_0.unsqueeze(1)], dim=1)
            normal_sup_feats = torch.cat([normal_sup_feats_lb, normal_sup_feats_ulb], dim=0)
            # 超图的视图
            independent_sup_feats = torch.cat([hyper_lb_out_norm, hyper_ulb_out_norm], dim=0)
            hyper_sup_feats = torch.cat([independent_sup_feats.unsqueeze(1), hyper_whole_out_norm.unsqueeze(1)], dim=1)
            clinical_supcon_feats = torch.cat([normal_sup_feats, hyper_sup_feats], dim=1)
            # supcon loss
            self.supconloss.device = feats_x_ulb_w.device
            # 计算对比损失
            sup_con_loss = self.supconloss(clinical_supcon_feats, clinical[:num_lb + num_ulb])



            # hyper_mean_lb_logits = (hyper_lb_logits + hyper_whole_logits[:num_lb]) / 2
            # hyper_mean_ulb_logits = (hyper_ulb_logits + hyper_whole_logits[num_lb:]) / 2
            # 不计算梯度
            with torch.no_grad():
                hyper_lb_logits = self.model.backbone(hyper_lb_out, only_fc=True)
                # 单独预测ulb的logits
                hyper_ulb_logits = self.model.backbone(hyper_ulb_out, only_fc=True)
                # 单独预测whole的logits
                hyper_whole_logits = self.model.backbone(hyper_whole_out, only_fc=True)
                if self.ce_loss == 'ce':
                    # 计算probs_lb的两个版本
                    probs_lb_fc = F.softmax(logits_x_lb, dim=1)
                    probs_lb_hyper = F.softmax(hyper_whole_logits[:num_lb], dim=1)

                    # 计算probs_ulb的两个版本
                    probs_ulb_fc = F.softmax(logits_x_ulb_w, dim=1)
                    probs_ulb_hyper = F.softmax(hyper_whole_logits[num_lb:], dim=1)
                    # 直觉上,lb和ulb的diff_probs在每个类别上的差异性至少在方向上应该保持一致
                    # 但是,现在还没有一个条件来保证:同方向差异性的lb在本次预测中是否是正确的,因此需要一个条件来约束,首先想到的是,lb的probs应当是正确的
                    # probs_lb_hyper还应该具有纠正错误的作用,比如:即使lb的probs是错误的,但是probs_lb_hyper是正确的,那么probs_lb_hyper应当具有纠正错误的作用
                    # 获得本次的probs_lb_fc里面预测正确的样本索引,
                    correct_in_fc_probs_index = torch.argmax(probs_lb_fc, dim=1) == torch.argmax(y_lb, dim=1)
                    # 获得本次probs_lb_hyper里面预测正确的样本索引
                    correct_in_hyper_probs_index = torch.argmax(probs_lb_hyper, dim=1) == torch.argmax(y_lb, dim=1)
                else:
                    probs_lb_fc = torch.sigmoid(logits_x_lb)
                    probs_lb_hyper = torch.sigmoid(hyper_whole_logits[:num_lb])
                    probs_ulb_fc = torch.sigmoid(logits_x_ulb_w)
                    probs_ulb_hyper = torch.sigmoid(hyper_whole_logits[num_lb:])
                    # 获得本次的probs_lb_fc里面预测正确的样本索引,
                    correct_in_fc_probs_index = (probs_lb_fc > 0.5).float() == y_lb
                    # 获得本次probs_lb_hyper里面预测正确的样本索引
                    correct_in_hyper_probs_index = (probs_lb_hyper > 0.5).float() == y_lb
                # 获得两次都预测正确的样本索引
                correct_in_both_probs_index = correct_in_fc_probs_index & correct_in_hyper_probs_index
                # 获得在fc上预测正确,但是在hyper上预测错误的样本索引
                correct_in_fc_wrong_in_hyper_probs_index = correct_in_fc_probs_index & ~correct_in_hyper_probs_index

                # 计算类别变化均值,我称为postitive_means和negative_means
                # positive_means是指在fc上预测正确,且在hyper上预测正确的样本的probs_lb_fc和probs_lb_hyper的差异性
                # negative_means是指在fc上预测正确,但是在hyper上预测错误的样本的probs_lb_fc和probs_lb_hyper的差异性
                positive_means = torch.mean(probs_lb_fc[correct_in_both_probs_index] - probs_lb_hyper[correct_in_both_probs_index], dim=0)
                negative_means = torch.mean(probs_lb_fc[correct_in_fc_wrong_in_hyper_probs_index] - probs_lb_hyper[correct_in_fc_wrong_in_hyper_probs_index], dim=0)
                # 如果probs_lb_fc或者probs_lb_hyper有nan,则打印:
                if torch.isnan(positive_means).sum() > 0 or torch.isnan(negative_means).sum() > 0:
                    pass
                else:
                    # 结合global_post_means和global_neg_means进行全局的均值计算
                    if self.global_post_means.sum() != 0:
                        self.global_post_means = self.global_post_means * 0.5 + positive_means * 0.5
                        self.global_neg_means = self.global_neg_means * 0.5 + negative_means * 0.5
                    else:
                        self.global_post_means = positive_means
                        self.global_neg_means = negative_means


                # 在probs_ulb_fc上加positive_means
                probs_ulb_fc_add_positive_means = probs_ulb_fc + self.global_post_means
                probs_ulb_fc_add_negative_means = probs_ulb_fc + self.global_neg_means
                # 找出签后两次的概率最大值索引没有变化且概率值大于0.9的样本索引
                if self.ce_loss == 'ce':
                    post_correct_in_fc_probs_index = (
                            (torch.argmax(probs_ulb_fc, dim=1) == torch.argmax(probs_ulb_fc_add_positive_means, dim=1)) &
                            (torch.max(probs_ulb_fc_add_positive_means, dim=1).values > 0.95)
                    )
                    # 在probs_ulb_fc上加negative_means

                    # 找出probs_ulb_fc和probs_ulb_fc_add_negative_means前后两次的概率最大值索引发生变化的样本且变化后概率值大于0.9的样本
                    neg_correct_in_fc_probs_index = (
                            (torch.argmax(probs_ulb_fc, dim=1) != torch.argmax(probs_ulb_fc_add_negative_means, dim=1)) &
                            (torch.max(probs_ulb_fc_add_negative_means, dim=1).values > 0.95)
                    )
                else:
                    post_correct_in_fc_probs_index = (
                            (probs_ulb_fc > 0.5) == (probs_ulb_fc_add_positive_means > 0.95)
                    )
                    neg_correct_in_fc_probs_index = (
                            (probs_ulb_fc > 0.5) != (probs_ulb_fc_add_negative_means > 0.95)
                    )
            multi_label = True if self.args.loss == 'bce' else False
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs, softmax_x_ulb=False,multi_label=multi_label)
            mask = mask.int() #形状就是batch_size
            # post_correct_in_fc_probs_index和neg_correct_in_fc_probs_index需要返回最大值索引
            if self.args.loss == 'ce':
                post_correct_in_fc_probs_index = torch.argmax(post_correct_in_fc_probs_index, dim=1)
                neg_correct_in_fc_probs_index = torch.argmax(neg_correct_in_fc_probs_index, dim=1)
            final_mask = (mask | post_correct_in_fc_probs_index) & ~neg_correct_in_fc_probs_index

            hyper_class_loss = self.consistency_loss(logits_x_ulb_s_0,pred_classes_ulb_w,self.args.loss,final_mask)



            # 监督损失
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')





            total_loss = sup_loss + self.lambda_u * hyper_class_loss  + sup_con_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=hyper_class_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=final_mask.float().mean().item())
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