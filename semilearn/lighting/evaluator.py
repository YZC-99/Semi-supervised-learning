import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np


class Evaluator(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.outputs = []
        self.labels = []

    def compute(self, outputs, labels):
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)

        # True positives, false positives, and false negatives
        tp = (outputs * labels).sum(dim=0).float()
        fp = (outputs * (1 - labels)).sum(dim=0).float()
        fn = ((1 - outputs) * labels).sum(dim=0).float()

        # Overall Precision (OP)
        OP = tp.sum().item() / (tp + fp).sum().item()

        # Overall Recall (OR)
        OR = tp.sum().item() / (tp + fn).sum().item()

        # Overall F1 Score (OF1)
        OF1 = 2 * OP * OR / (OP + OR)

        # Per-class Precision (CP)
        with torch.no_grad():
            per_class_precision = tp / (tp + fp)
            per_class_precision[torch.isnan(per_class_precision)] = 0
        CP = per_class_precision.mean().item()

        # Per-class Recall (CR)
        with torch.no_grad():
            per_class_recall = tp / (tp + fn)
            per_class_recall[torch.isnan(per_class_recall)] = 0
        CR = per_class_recall.mean().item()

        # Per-class F1 Score (CF1)
        CF1 = 2 * CP * CR / (CP + CR) if (CP + CR) != 0 else 0

        # AUC and AUPRC for each class
        auc_scores = []
        auprc_scores = []
        for i in range(labels.shape[1]):
            try:
                # AUC
                auc_score = roc_auc_score(labels[:, i].cpu().numpy(), outputs[:, i].cpu().numpy())
                auc_scores.append(auc_score)

                # AUPRC
                precision, recall, _ = precision_recall_curve(labels[:, i].cpu().numpy(), outputs[:, i].cpu().numpy())
                auprc_score = auc(recall, precision)
                auprc_scores.append(auprc_score)
            except ValueError as e:
                print(f"Error for class {i}: {str(e)}")
                # Handle the case where AUC or AUPRC cannot be computed
                auc_scores.append(np.nan)  # Use NaN or another placeholder to indicate the error
                auprc_scores.append(np.nan)

        # Filter out NaN values before averaging
        valid_auc_scores = [score for score in auc_scores if not np.isnan(score)]
        valid_auprc_scores = [score for score in auprc_scores if not np.isnan(score)]

        avg_auc = np.mean(valid_auc_scores) if valid_auc_scores else 0
        avg_auprc = np.mean(valid_auprc_scores) if valid_auprc_scores else 0

        # Average AUC and AUPRC
        # avg_auc = np.mean(auc_scores)
        # avg_auprc = np.mean(auprc_scores)

        self.reset()

        return {
            'OP': OP,
            'OR': OR,
            'OF1': OF1,
            'CP': CP,
            'CR': CR,
            'CF1': CF1,
            'mAP': per_class_precision.mean().item(),
            'AP': per_class_precision.tolist(),
            'AUC': avg_auc,
            'AUPRC': avg_auprc,
            'AUCs': auc_scores,
            'AUPRCs': auprc_scores,
        }

    # def compute(self, outputs, labels):
    #     outputs = torch.cat(outputs, dim=0)
    #     labels = torch.cat(labels, dim=0)
    #     # outputs = torch.from_numpy(outputs)
    #     # labels = torch.from_numpy(labels)
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
    #     OF1 = 2 * OP * OR / (OP + OR)
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
    #     # Average Precision (AP) per class
    #     AP = per_class_precision
    #     mAP = AP.mean().item()
    #     self.reset()
    #
    #     return {
    #         'OP': OP,
    #         'OR': OR,
    #         'OF1': OF1,
    #         'CP': CP,
    #         'CR': CR,
    #         'CF1': CF1,
    #         'mAP': mAP,
    #         'AP': AP.tolist(),
    #     }
