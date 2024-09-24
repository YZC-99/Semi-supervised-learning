
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from semilearn.loss.supconloss import SupConLoss

def construct_hypergraph(ex_clinical, pred_classes, feats):
    """
    构建超图 H 和对应的节点嵌入

    Args:
        ex_clinical: [N]，每个样本的 ex_clinical 值
        pred_classes: [N]，每个样本的预测类别
        feats: [N, F]，每个样本的特征嵌入

    Returns:
        H: [N, E]，超图的关联矩阵
        node_embeddings: [N, F]，节点的嵌入
    """
    N = ex_clinical.size(0)
    num_classes = pred_classes.max().item() + 1

    # 获取 ex_clinical 的唯一值
    unique_ex_clinical, ex_indices = torch.unique(ex_clinical, return_inverse=True)
    num_ex_clinical = unique_ex_clinical.size(0)

    # 超边数目：ex_clinical 的唯一值数目 + 类别数目
    E = num_ex_clinical + num_classes

    # 初始化超图 H
    H = torch.zeros((N, E)).cuda()

    # 连接样本到 ex_clinical 超边
    H[torch.arange(N), ex_indices] = 1

    # 连接样本到类别预测超边
    class_edge_indices = num_ex_clinical + pred_classes
    H[torch.arange(N), class_edge_indices] = 1

    # 节点的嵌入就是样本的特征
    node_embeddings = feats

    return H, node_embeddings

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
        inv_De = torch.diag(torch.pow(De, -1))
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

class HyperCoMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128, epass=False,num_classes=5):
        super(HyperCoMatch_Net, self).__init__()
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
        
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)

        feat_proj1 = F.normalize(self.mlp_proj(feat), p=2, dim=1)
        feat_proj2 = F.normalize(self.mlp_proj_2(feat), p=2, dim=1)
        feat_proj3 = F.normalize(self.mlp_proj_3(feat), p=2, dim=1)

        if self.epass:
            feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat))/3)
        else:
            feat_proj = self.l2norm(self.mlp_proj(feat))

        return {'logits':logits, 'feat':feat_proj,'feat_proj1':feat_proj1,'feat_proj2':feat_proj2,'feat_proj3':feat_proj3}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


# TODO: move this to criterions
def comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=0.2):
    # embedding similarity
    sim = torch.exp(torch.mm(feats_x_ulb_s_0, feats_x_ulb_s_1.t())/ T) # 构建相似性
    sim_probs = sim / sim.sum(1, keepdim=True) # 归一化
    # contrastive loss
    loss = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
    loss = loss.mean()  
    return loss


@ALGORITHMS.register('hypercomatchv2')
class HyperCoMatchV2(AlgorithmBase):
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
        self.clinical_mode = args.clinical

    def init(self, T, p_cutoff, contrast_p_cutoff, hard_label=True, queue_batch=128, smoothing_alpha=0.999, da_len=256):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.contrast_p_cutoff = contrast_p_cutoff
        self.use_hard_label = hard_label
        self.queue_batch = queue_batch
        self.smoothing_alpha = smoothing_alpha
        self.da_len = da_len
        self.supconloss = SupConLoss()

        # TODO: put this part into a hook
        # memory smoothing
        self.queue_size = int(queue_batch * (self.args.uratio + 1) * self.args.batch_size) if self.args.dataset != 'imagenet' else self.args.K
        self.queue_feats = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_clinical = torch.zeros(self.queue_size, 1).cuda(self.gpu)
        self.queue_ptr = 0

        # init clinical labels
        self.clinical_memory = {'c1':[],'c2':[],'c3':[],'c4':[],'c5':[]}
        
    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'), 
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = CoMatch_Net(model, proj_size=self.args.proj_size, epass=self.args.use_epass)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = CoMatch_Net(ema_model, proj_size=self.args.proj_size, epass=self.args.use_epass)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model


    @torch.no_grad()
    def update_bank(self, feats, probs,clinical):
        if self.distributed and self.world_size > 1:
            feats = concat_all_gather(feats)
            probs = concat_all_gather(probs)
            clinical = concat_all_gather(clinical)
        # update memory bank
        length = feats.shape[0]
        if (self.queue_ptr + length) > self.queue_size:
            queue_ptr = self.queue_size - self.queue_ptr
            feats = feats[:queue_ptr]
            probs = probs[:queue_ptr]
            clinical = clinical[:queue_ptr]
        self.queue_feats[self.queue_ptr:self.queue_ptr + length, :] = feats
        self.queue_probs[self.queue_ptr:self.queue_ptr + length, :] = probs      
        self.queue_clinical[self.queue_ptr:self.queue_ptr + length, :] = clinical
        self.queue_ptr = (self.queue_ptr + length) % self.queue_size

    def train_step(self, x_lb, y_lb,in_clinical, x_ulb_w, x_ulb_s_0, x_ulb_s_1,ex_clinical):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1)) #
            clinical = torch.cat([in_clinical, ex_clinical,ex_clinical,ex_clinical], dim=0)
            if 'eyeid' == self.clinical_mode:
                clinical = clinical[:][:,-4]
            elif 'bcva' == self.clinical_mode:
                clinical = clinical[:][:,-3]
            elif 'cst' == self.clinical_mode:
                clinical = clinical[:][:,-2]
            elif 'patientid' == self.clinical_mode:
                clinical = clinical[:][:,-1]
            outputs = self.model(inputs)
            logits, feats,feat_proj1,feat_proj2,feat_proj3 = outputs['logits'], outputs['feat'],outputs['feat_proj1'],outputs['feat_proj2'],outputs['feat_proj3']
            logits_x_lb, feats_x_lb = logits[:num_lb], feats[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s_0, _ = logits[num_lb:].chunk(3)
            feats_x_ulb_w, feats_x_ulb_s_0, feats_x_ulb_s_1 = feats[num_lb:].chunk(3)

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s':[feats_x_ulb_s_0, feats_x_ulb_s_1]}

            # supervised loss
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # 获取未标记数据的 ex_clinical 和类别预测
            ex_clinical_ulb_w = clinical[num_lb:num_lb + num_ulb]
            ex_clinical_ulb_s0 = clinical[num_lb + num_ulb:num_lb + 2 * num_ulb]
            # 获取类别预测
            pred_classes_ulb_w = (torch.sigmoid(logits_x_ulb_w) > 0.5).float()
            pred_classes_ulb_s0 = (torch.sigmoid(logits_x_ulb_s_0) > 0.5).float()

            # 构建超图 H1（针对 x_ulb_w）
            H1, node_embeddings1 = construct_hypergraph(ex_clinical_ulb_w, pred_classes_ulb_w, feats_x_ulb_w)

            # 构建超图 H2（针对 x_ulb_s_0）
            H2, node_embeddings2 = construct_hypergraph(ex_clinical_ulb_s0, pred_classes_ulb_s0, feats_x_ulb_s_0)

            # 使用 HGNN 进行传播
            hgnn = HGNN(nb_classes=self.num_classes, sz_embed=feats_x_ulb_w.size(1), hidden=512).cuda()
            out1 = hgnn(node_embeddings1, H1)
            out2 = hgnn(node_embeddings2, H2)

            with torch.no_grad():
                logits_x_ulb_w = logits_x_ulb_w.detach()
                feats_x_lb = feats_x_lb.detach()
                feats_x_ulb_w = feats_x_ulb_w.detach()

                # probs = torch.softmax(logits_x_ulb_w, dim=1)            
                probs = self.compute_prob(logits_x_ulb_w)
                # distribution alignment
                probs = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs.detach())

                probs_orig = probs.clone()
                # memory-smoothing 
                if self.epoch >0 and self.it > self.queue_batch:
                    A = torch.exp(torch.mm(feats_x_ulb_w, self.queue_feats.t()) / self.T)
                    A = A / A.sum(1,keepdim=True)                    
                    probs = self.smoothing_alpha * probs + (1 - self.smoothing_alpha) * torch.mm(A, self.queue_probs)    
                
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs, softmax_x_ulb=False)    
                
                feats_w = torch.cat([feats_x_ulb_w, feats_x_lb],dim=0)   
                # probs_w = torch.cat([probs_orig, F.one_hot(y_lb, num_classes=self.num_classes)],dim=0)
                probs_w = torch.cat([probs_orig, y_lb],dim=0)

                self.update_bank(feats_w, probs_w)

            unsup_loss = self.consistency_loss(logits_x_ulb_s_0,
                                          probs,
                                          'bce',
                                          mask=mask)

            # pseudo-label graph with self-loop
            Q = torch.mm(probs, probs.t())# 构建图
            Q.fill_diagonal_(1) # 自连接
            pos_mask = (Q >= self.contrast_p_cutoff).to(mask.dtype) # 大于某个值的才保留
            Q = Q * pos_mask
            Q = Q / Q.sum(1, keepdim=True) # 归一化
            contrast_loss = comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=self.T)



        total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * contrast_loss + sup_con_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         contrast_loss=contrast_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict =  super().get_save_dict()
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
