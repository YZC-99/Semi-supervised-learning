import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
import numpy as np




import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np


class Evaluator(object):
    def __init__(self, num_classes,multilabel=True):
        self.num_classes = num_classes
        self.reset()
        self.multilabel = multilabel

    def reset(self):
        self.outputs = []
        self.labels = []

    def compute_single_label(self,outputs,labels):
        probs = outputs
        THRESH = 0.18
        AUROCs, Accs, Senss, Recas, Specs, F1s = [], [], [], [], [], []
        for i in range(self.num_classes):
            try:
                AUROCs.append(roc_auc_score(labels == i, probs[:, i]))
            except ValueError:
                AUROCs.append(0)
            try:
                Accs.append(accuracy_score(labels == i, probs[:, i] > THRESH))
            except ValueError:
                Accs.append(0)
            try:
                Senss.append(sensitivity_score(labels == i, probs[:, i] > THRESH))
            except ValueError:
                Senss.append(0)
            try:
                Recas.append(recall_score(labels == i, probs[:, i] > THRESH))
            except ValueError:
                Recas.append(0)
            try:
                Specs.append(specificity_score(labels == i, probs[:, i] > THRESH))
            except ValueError:
                Specs.append(0)
            try:
                F1s.append(f1_score(labels == i, probs[:, i] > THRESH))
            except ValueError:
                F1s.append(0)
        # 计算平均值
        acc = np.mean(Accs) * 100
        auc = np.mean(AUROCs) * 100
        F1 = np.mean(F1s) * 100
        sens = np.mean(Senss) * 100
        spec = np.mean(Specs) * 100
        return {
            'ACC': acc,
            'OF1': F1,
            'CF1': F1,
            'AUC': auc,
            'SENS': sens,
            'SPEC': spec,
            'AUPRC': auc,
        }

    def compute_multi_label(self, outputs, labels):
        outputs = torch.cat(outputs, dim=0).float().cpu()
        labels = torch.cat(labels, dim=0).float().cpu()
        THRESH = 0.18
        AUROCs, Accs, Senss, Recas, Specs, F1s = [], [], [], [], [], []
        for i in range(self.num_classes):
            try:
                AUROCs.append(roc_auc_score(labels[:, i], outputs[:, i]))
            except ValueError:
                AUROCs.append(0)
            try:
                Accs.append(accuracy_score(labels[:, i], outputs[:, i] > THRESH))
            except ValueError:
                Accs.append(0)
            try:
                Senss.append(sensitivity_score(labels[:, i], outputs[:, i] > THRESH))
            except ValueError:
                Senss.append(0)
            try:
                Recas.append(recall_score(labels[:, i], outputs[:, i] > THRESH))
            except ValueError:
                Recas.append(0)
            try:
                Specs.append(specificity_score(labels[:, i], outputs[:, i] > THRESH))
            except ValueError:
                Specs.append(0)
            try:
                F1s.append(f1_score(labels[:, i], outputs[:, i] > THRESH))
            except ValueError:
                F1s.append(0)
        # 计算平均值
        acc = np.mean(Accs) * 100
        auc = np.mean(AUROCs) * 100
        F1 = np.mean(F1s) * 100
        sens = np.mean(Senss) * 100
        spec = np.mean(Specs) * 100
        return {
            'ACC': acc,
            'OF1': F1,
            'CF1': F1,
            'AUC': auc,
            'SENS': sens,
            'SPEC': spec,
            'AUPRC': auc,
        }



    def compute(self, outputs, labels):
       return self.compute_single_label(outputs,labels) if not self.multilabel else self.compute_multi_label(outputs,labels)

    # def compute(self, outputs, labels):
    #     if not self.multilabel:
    #
    #         # 单标签多类别任务，使用 softmax 来处理每个样本的概率分布
    #         outputs = [torch.softmax(output, dim=-1) for output in outputs]
    #         # 获取每个样本的预测类别（取概率最大的类别）
    #         outputs = torch.cat([torch.argmax(output, dim=-1, keepdim=True) for output in outputs], dim=0)
    #
    #         # 确保 labels 是一维或更高维的 Tensor，转换标量为一维张量
    #         labels = [torch.tensor([label]) if torch.is_tensor(label) and label.dim() == 0 else label for label in
    #                   labels]
    #
    #         # 合并 labels 列表，并调整维度
    #         labels = torch.cat(labels, dim=0).view(-1, 1)  # 将 labels 调整为 (batch_size, 1) 形式
    #
    #         # 确保 outputs 是一维张量，代表类别索引
    #         outputs = outputs.view(-1)  # 确保 outputs 是一维，(batch_size,)
    #
    #         # 将预测和标签转换为 one-hot 格式以便后续计算
    #         outputs = torch.zeros(outputs.size(0), self.num_classes).to(outputs.device).scatter_(1, outputs.unsqueeze(1), 1)
    #         labels = torch.zeros(labels.size(0), self.num_classes).to(outputs.device).scatter_(1, labels, 1)
    #     else:
    #         # 多标签多分类任务，假设已经使用 sigmoid 作为激活函数
    #         outputs = torch.cat(outputs, dim=0)
    #         labels = torch.cat(labels, dim=0)
    #     # 确保 outputs 和 labels 是浮点型（如有必要，可根据实际情况调整数据类型）
    #     outputs = outputs.float()
    #     labels = labels.float()
    #
    #     # 计算 ACC：只有当一个样本的所有标签都被正确预测时，才计为正确
    #     correct_per_sample = (outputs == labels).all(dim=1)
    #     ACC = correct_per_sample.sum().item() / outputs.size(0)
    #
    #     # True positives, false positives, and false negatives
    #     tp = (outputs * labels).sum(dim=0).float()
    #     fp = (outputs * (1 - labels)).sum(dim=0).float()
    #     fn = ((1 - outputs) * labels).sum(dim=0).float()
    #
    #     # Overall Precision (OP)
    #     OP = tp.sum().item() / (tp + fp).sum().item()
    #
    #     # Overall Recall (OR)
    #     OR = tp.sum().item() / (tp + fn).sum().item()
    #
    #     # Overall F1 Score (OF1)
    #     OF1 = 2 * OP * OR / (OP + OR) if (OP + OR) != 0 else 0
    #
    #     # Per-class Precision (CP)
    #     with torch.no_grad():
    #         per_class_precision = tp / (tp + fp)
    #         per_class_precision[torch.isnan(per_class_precision)] = 0
    #     CP = per_class_precision.mean().item()
    #
    #     # Per-class Recall (CR)
    #     with torch.no_grad():
    #         per_class_recall = tp / (tp + fn)
    #         per_class_recall[torch.isnan(per_class_recall)] = 0
    #     CR = per_class_recall.mean().item()
    #
    #     # Per-class F1 Score (CF1)
    #     CF1 = 2 * CP * CR / (CP + CR) if (CP + CR) != 0 else 0
    #
    #     # AUC and AUPRC for each class
    #     auc_scores = []
    #     auprc_scores = []
    #     for i in range(labels.shape[1]):
    #         try:
    #             # AUC
    #             auc_score = roc_auc_score(labels[:, i].cpu().numpy(), outputs[:, i].cpu().numpy())
    #             auc_scores.append(auc_score)
    #
    #             # AUPRC
    #             precision, recall, _ = precision_recall_curve(labels[:, i].cpu().numpy(), outputs[:, i].cpu().numpy())
    #             auprc_score = auc(recall, precision)
    #             auprc_scores.append(auprc_score)
    #         except ValueError as e:
    #             print(f"Error for class {i}: {str(e)}")
    #             # Handle the case where AUC or AUPRC cannot be computed
    #             auc_scores.append(np.nan)
    #             auprc_scores.append(np.nan)
    #             continue
    #
    #     # Filter out NaN values before averaging
    #     valid_auc_scores = [score for score in auc_scores if not np.isnan(score)]
    #     valid_auprc_scores = [score for score in auprc_scores if not np.isnan(score)]
    #
    #     avg_auc = np.mean(valid_auc_scores) if valid_auc_scores else 0
    #     avg_auprc = np.mean(valid_auprc_scores) if valid_auprc_scores else 0
    #
    #     self.reset()
    #
    #     return {
    #         'ACC': ACC,
    #         'OP': OP,
    #         'OR': OR,
    #         'OF1': OF1,
    #         'CP': CP,
    #         'CR': CR,
    #         'CF1': CF1,
    #         'mAP': per_class_precision.mean().item(),
    #         'AP': per_class_precision.tolist(),
    #         'AUC': avg_auc,
    #         'AUPRC': avg_auprc,
    #         'AUCs': auc_scores,
    #         'AUPRCs': auprc_scores,
    #     }



