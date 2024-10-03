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
    DDL_flat = DDL.view(-1).cpu().numpy().reshape(-1, 1)  # [num_classes * num_classes, 1]
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
    num_classes = DDU.size(1)
    # 假设 DDU 是 [num_ulb, num_classes]
    # 需要将其展平为与DDL相同的维度
    DDU_flat = DDU.view(num_ulb, -1).cpu().numpy()  # [num_ulb, num_classes * num_classes]
    log_probs = gmm.score_samples(DDU_flat)  # [num_ulb]
    likelihoods = np.exp(log_probs)  # 转换为概率
    return torch.tensor(likelihoods, device=DDU.device)

def select_high_confidence_samples(likelihoods, threshold=0.95):
    """
    选择高可信度的未标记样本。

    Args:
        likelihoods: [num_ulb] 每个未标记样本的似然概率
        threshold: float 置信度阈值

    Returns:
        selected_indices: [num_selected] 选择的未标记样本索引
    """
    high_confidence_mask = likelihoods > threshold
    selected_indices = high_confidence_mask.nonzero(as_tuple=True)[0]
    return selected_indices


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


def construct_hypergraph_single_label(ex_clinical, pred_classes, feats):
    """
    构建超图 H 和对应的节点嵌入（单标签分类），并返回节点对应的原始样本索引。

    Args:
        ex_clinical: [N]，每个样本的 ex_clinical 值
        pred_classes: [N]，每个样本的预测类别（单标签，类别索引）
        feats: [N, F]，每个样本的特征嵌入

    Returns:
        H: [num_nodes, num_classes]，超图的关联矩阵
        node_features: [num_nodes, F]，节点的嵌入
        node_indices_map: [num_nodes]，每个节点对应的原始样本索引
    """
    N = ex_clinical.size(0)
    device = ex_clinical.device
    num_classes = pred_classes.max().item() + 1  # 假设类别索引从 0 开始

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
    class_indices = pred_classes  # [N]
    node_indices_for_samples = node_indices  # [N]

    H = torch.zeros((num_nodes, num_classes), device=device)
    H[node_indices_for_samples, class_indices] = 1  # 设置关联矩阵中的值为 1

    # 处理节点索引映射
    node_indices_map = torch.zeros(num_nodes, dtype=torch.long, device=device)

    for node_idx in range(num_nodes):
        # 获取对应于该节点的样本索引
        sample_indices_for_node = torch.nonzero(node_indices == node_idx).squeeze()
        num_samples = sample_indices_for_node.numel()

        if num_samples == 1:
            # 节点没有参与平均特征计算，直接返回该样本的索引
            node_indices_map[node_idx] = sample_indices_for_node.item()
        else:
            # 节点参与了平均特征计算，选择与平均特征 MSE 最小的样本索引
            feats_for_node = feats[sample_indices_for_node]  # [num_samples, F]
            node_feature = node_features[node_idx].unsqueeze(0)  # [1, F]

            # 计算每个样本特征与平均特征之间的 MSE
            mse = torch.mean((feats_for_node - node_feature) ** 2, dim=1)  # [num_samples]

            # 获取具有最小 MSE 的样本索引
            min_mse_idx = torch.argmin(mse)
            selected_sample_idx = sample_indices_for_node[min_mse_idx]

            node_indices_map[node_idx] = selected_sample_idx.item()

    return H, node_features, node_indices_map


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


def construct_hypergraph(ex_clinical, num_lb):
    """
    构建超图 H，包括已标记和未标记数据。

    Args:
        ex_clinical: [N]，每个样本的 ex_clinical 值（已标记和未标记）
        num_lb: int，已标记数据的数量

    Returns:
        H: [N, num_hyperedges]，超图的关联矩阵
    """
    N = ex_clinical.size(0)
    device = ex_clinical.device

    # 构建基于临床标签的超边
    unique_ex_clinical, inverse_indices = torch.unique(ex_clinical, sorted=True, return_inverse=True)
    num_ex_clinical = unique_ex_clinical.size(0)

    # 构建关联矩阵 H，形状为 [N, num_hyperedges]
    H = torch.zeros((N, num_ex_clinical), device=device)

    # 添加基于 ex_clinical 的超边
    for idx in range(num_ex_clinical):
        # 获取 ex_clinical 值为 unique_ex_clinical[idx] 的样本索引
        samples_in_hyperedge = torch.nonzero(inverse_indices == idx).squeeze()
        H[samples_in_hyperedge, idx] = 1

    return H

def compute_neighbor_class_distribution(H, num_lb, y_lb, num_classes):
    """
    计算未标记样本的邻域类别分布。

    Args:
        H: [N, num_hyperedges]，超图的关联矩阵
        num_lb: 已标记数据的数量
        y_lb: [num_lb]，已标记数据的真实标签
        num_classes: 类别数量

    Returns:
        neighbor_probs: [N - num_lb, num_classes]，未标记样本的邻域类别概率分布
    """
    N = H.size(0)
    device = H.device
    # 确保 y_lb 是整数类型
    if y_lb.dtype != torch.long:
        y_lb = y_lb.long()
    # 初始化未标记样本的邻域类别分布
    neighbor_probs = torch.zeros(N - num_lb, num_classes, device=device)

    # H_ulb: 未标记样本对应的行
    H_ulb = H[num_lb:]  # [N_ulb, num_hyperedges]

    # H_lb: 已标记样本对应的行
    H_lb = H[:num_lb]   # [num_lb, num_hyperedges]

    # 计算每个未标记样本的邻域类别分布
    for i in range(H_ulb.size(0)):
        # 找到未标记样本的超边
        hyperedges = (H_ulb[i] > 0).nonzero(as_tuple=True)[0]
        if hyperedges.numel() == 0:
            # 如果未标记样本未连接任何超边，使用均匀分布
            neighbor_probs[i] = torch.full((num_classes,), 1.0 / num_classes, device=device)
            continue
        # 找到这些超边连接的已标记样本
        neighbor_lb = (H_lb[:, hyperedges] > 0).nonzero(as_tuple=True)[0]
        if neighbor_lb.numel() > 0:
            # 收集已标记邻域样本的标签
            neighbor_labels = y_lb[neighbor_lb]
            # 统计类别分布
            neighbor_labels = torch.argmax(neighbor_labels, dim=1)
            class_counts = torch.bincount(neighbor_labels, minlength=num_classes)
            neighbor_probs[i] = class_counts.float() / class_counts.sum()
        else:
            # 如果没有已标记邻域样本，使用均匀分布
            neighbor_probs[i] = torch.full((num_classes,), 1.0 / num_classes, device=device)
    return neighbor_probs

def combine_predictions(model_probs, neighbor_probs, beta=0.5):
    """
    融合模型预测和邻域类别分布。

    Args:
        model_probs: [N_ulb, num_classes]，模型对未标记样本的预测概率
        neighbor_probs: [N_ulb, num_classes]，未标记样本的邻域类别概率分布
        beta: 融合权重

    Returns:
        combined_probs: [N_ulb, num_classes]，融合后的概率分布
    """
    combined_probs = beta * model_probs + (1 - beta) * neighbor_probs
    return combined_probs

def select_high_confidence_pseudo_labels(combined_probs, p_cutoff):
    """
    选择高置信度的伪标签。

    Args:
        combined_probs: [N_ulb, num_classes]，融合后的概率分布
        p_cutoff: 置信度阈值

    Returns:
        pseudo_labels: [N_ulb]，伪标签
        mask_ulb: [N_ulb]，高置信度样本的掩码
    """
    probs, pseudo_labels = torch.max(combined_probs, dim=1)
    mask_ulb = probs > p_cutoff
    return pseudo_labels, mask_ulb


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
        self.hgnn3 = HGNN(nb_classes=num_classes, sz_embed=proj_size, hidden=proj_size)

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


@ALGORITHMS.register('hyperplusfixmatchv2')
class HyperPlusFixMatchV2(AlgorithmBase):
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
            if 'eyeid' == self.clinical_mode or 'localization' == self.clinical_mode:
                clinical = clinical[:][:, -4]
            elif 'bcva' == self.clinical_mode or 'sex' == self.clinical_mode:
                clinical = clinical[:][:, -3]
            elif 'cst' == self.clinical_mode or 'age' == self.clinical_mode:
                clinical = clinical[:][:, -2]
            elif 'patientid' == self.clinical_mode or 'lesion_id' == self.clinical_mode:
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
            # 获取未标记数据的 ex_clinical 和类别预测
            ex_clinical_ulb_w = clinical[num_lb:num_lb + num_ulb]
            ex_clinical_ulb_s0 = clinical[num_lb + num_ulb:num_lb + 2 * num_ulb]
            # 获取类别预测

            # 构建超图,针对标记数据.这个超图是绝对的，不会改变
            H_lb, node_embeddings_lb, _ = construct_hypergraph_single_label_whole(clinical[:num_lb],
                                                                            torch.argmax(y_lb, dim=1),
                                                                            feats[:num_lb])

            # 构建超图 针对 x_ulb_s_0）
            pred_classes_ulb_s0 = logits_x_ulb_s_0.argmax(dim=1)
            H_ulb, node_embeddings_ulb, _ = construct_hypergraph_single_label_whole(ex_clinical_ulb_s0,
                                                                              pred_classes_ulb_s0,
                                                                              feats_x_ulb_s_0)

            # 构建全部的超图
            # 这里使用伪标签的目的主要是为了 ，从而实现样本之间的信息传递，但这里的伪标签带有极大的噪声
            pred_classes_ulb_w = logits_x_ulb_w.argmax(dim=1)
            label_classes = torch.cat((torch.argmax(y_lb, dim=1), pred_classes_ulb_w), dim=0)
            H_whole, node_embeddings_whole, indices_map = construct_hypergraph_single_label_whole(clinical[:num_lb + num_ulb],
                                                                                        label_classes,
                                                                                        feats[:num_lb + num_ulb])


            hyper_out = self.model.stage2_forward(node_embeddings_lb, H_lb, node_embeddings_ulb, H_ulb, node_embeddings_whole, H_whole)


            hyper_lb_out, hyper_ulb_out, hyper_whole_out = hyper_out['hyper_logits1'], hyper_out['hyper_logits2'], hyper_out['hyper_logits3']
            part_hyper_whole_out = hyper_whole_out[num_lb:, :]
            part_hyper_whole_out = nn.functional.softmax(part_hyper_whole_out)
            hyper_strong_out = nn.functional.softmax(hyper_ulb_out)
            # 保持图的一致性
            hyper_consistency_loss = nn.functional.kl_div(part_hyper_whole_out.log(), hyper_strong_out,
                                                          reduction='batchmean') + nn.functional.kl_div(
                hyper_strong_out.log(),
                part_hyper_whole_out, reduction='batchmean')

            DL1,DL2 = F.softmax(logits_x_lb,dim=1), F.softmax(hyper_lb_out,dim=1)
            DDL = F.kl_div(DL1.log(), DL2, reduction='none') + F.kl_div(DL2.log(), DL1, reduction='none')

            DL_whole1, DL_whole2 = F.softmax(logits_x_lb,dim=1), F.softmax(hyper_whole_out[:num_lb],dim=1)
            DDL_whole = F.kl_div(DL_whole1.log(), DL_whole2, reduction='none') + F.kl_div(DL_whole2.log(), DL_whole1, reduction='none')

            # 计算DDU1_s(distribution difference for unlabeled data)
            DU1, DU2 = F.softmax(logits_x_ulb_s_0,dim=1), F.softmax(hyper_strong_out,dim=1)
            # 计算每个未标记样本的 DDU_s
            DDU = F.kl_div(DU1.log(), DU2, reduction='none').sum(dim=1) + \
                    F.kl_div(DU2.log(), DU1, reduction='none').sum(dim=1)

            # 计算DDU1_w(distribution difference for unlabeled data)
            DU1_whole, DU2_whole = F.softmax(logits_x_ulb_w,dim=1), F.softmax(hyper_whole_out[num_lb:],dim=1)
            # 计算每个未标记样本的 DDU_w
            DDU_whole = F.kl_div(DU1_whole.log(), DU2_whole, reduction='none').sum(dim=1) + \
                    F.kl_div(DU2_whole.log(), DU1_whole, reduction='none').sum(dim=1)



            """
            所有的超图构建都基于以下原则:
            1.每个样本就是一个节点. 2.超边有两种类型: 一种是基于类别的超边，另一种是基于临床标签的超边
            
            DDL: 标记数据在普通分类器和超图分类器之间的分布差异,超图分类器里面只有标记数据
            DDL_whole: 标记数据在普通分类器和超图分类器之间的分布差异,超图分类器里面有标记数据和未标记数据
            DDU: 未标记数据在普通分类器和超图分类器之间的分布差异,超图分类器里面只有标记数据
            DDU_whole: 未标记数据在普通分类器和超图分类器之间的分布差异,超图分类器里面有标记数据和未标记数据
            值得注意的是
            1. DDL_whole和DDU_whole是出自同一个超图分类器，但是DDL和DDU是出自不同的分类器. 换句话说,总共有3个超图分类器.
            2. DDL的构建很特殊:他的超边信息是基于真实标签的，而不是基于预测标签的. 这是因为我们希望DDL能够反映出真实标签和预测标签之间的分布差异.
            
            客观事实: DDL肯定是有的,因为标记数据在超图分类器的logits分布肯定和普通分类器的logits分布不一样. 同样,在未标记数据上也有同样的情况,我们称之为DDU.
            1. 对于DDL我们有两个视角: 独立的DDL和整体的DDL_whole. 他们的计算方式是一样的,只是DDL是基于标记数据,而DDL_whole是基于标记数据和未标记数据.
            DDL的存在反映了标记数据纳入临床标签信息后的分布变化. DDL_whole的存在反映了标记数据和未标记数据纳入临床标签信息后的分布变化.
            DDL_whole里面的标记样本不仅通过临床标签与其他的标记样本交互,而且还通过临床标签和预测标签与未标记样本交互.
            2.对于DDU,我们也有两个视角: 独立的DDU和整体的DDU_whole. 他们的计算方式是一样的,只是DDU是基于未标记数据,而DDU_whole是基于标记数据和未标记数据.
            
            我的猜想:能否通过某种手段建模DDL,DDL_whole. 
            因为标记数据和未标记数据他们都会经过普通分类器和超图分类器.并且他们直接还共享了同一个超图分类器:hyper_whole_out.这表现了他们的共性.
            同时,他们还有各自的特性,具体表现在DDL和DDU上. 也就是说,DDL和DDU是标记数据和未标记数据的特性. 他们的存在是因为标记数据和未标记数据的不同.
            最后,DDL_whole和DDU_whole是标记数据和未标记数据的共性. 因为这两个的计算是基于同一个超图,只不过是各自取标记数据和未标记数据来分别计算.
            
            我期望得到的结果:
            通过某种手段建模DD(distribution difference).从而可以从DD的视角去挑选可信度更高的伪标签数据.并且通过DD的视角去挑选更有代表性的标记数据.
            """

            # 将DDL展平为二维数据，每个类别视为一个样本
            gmm = fit_gmm_on_DDL(DDL,self.num_classes)  # [num_lb, num_classes]
            # 计算未标记样本的DDU似然概率
            likelihoods = compute_DDU_likelihood(gmm, DDU)  # [num_ulb]

            # 选择高可信度样本
            selected_indices = select_high_confidence_samples(likelihoods, threshold=0.9)  # 根据需要调整阈值
            pseudo_labels = torch.argmax(F.softmax(logits_x_ulb_w[selected_indices], dim=1), dim=1)  # [num_selected]
            selected_logits = hyper_whole_out[num_lb:][selected_indices]  # [num_se

            hyper_class_loss = self.ce_loss(selected_logits, pseudo_labels, reduction='mean')

            # 监督损失
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # supcon loss
            self.supconloss.device = feats_x_ulb_w.device
            clinical_supcon_feats = torch.cat([feats_x_ulb_w.unsqueeze(1), feats_x_ulb_s_0.unsqueeze(1)], dim=1)
            # print(clinical_supcon_feats.shape,clinical[num_lb:].shape)
            if 'simclr' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats)
            elif 'eyeid-cst' == self.clinical_mode or 'eyeid-bcva' == self.clinical_mode or 'bcva-cst' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,
                                               clinical[num_lb:num_lb + num_ulb]) + self.supconloss(
                    clinical_supcon_feats, clinical2[num_lb:num_lb + num_ulb])
            elif 'eyeid-bcva-cst' == self.clinical_mode:
                sup_con_loss = self.supconloss(clinical_supcon_feats,
                                               clinical[num_lb:num_lb + num_ulb]) + self.supconloss(
                    clinical_supcon_feats, clinical2[num_lb:num_lb + num_ulb]) + self.supconloss(clinical_supcon_feats,
                                                                                                 clinical3[
                                                                                                 num_lb:num_lb + num_ulb])
            else:
                sup_con_loss = self.supconloss(clinical_supcon_feats, clinical[num_lb:num_lb + num_ulb])


            total_loss = sup_loss + self.lambda_u * hyper_class_loss + hyper_consistency_loss + sup_con_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=hyper_class_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=0)
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