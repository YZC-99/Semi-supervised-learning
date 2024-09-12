import torch


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
        # outputs = torch.from_numpy(outputs)
        # labels = torch.from_numpy(labels)
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

        # Average Precision (AP) per class
        AP = per_class_precision
        mAP = AP.mean().item()
        self.reset()

        return {
            'OP': OP,
            'OR': OR,
            'OF1': OF1,
            'CP': CP,
            'CR': CR,
            'CF1': CF1,
            'mAP': mAP,
            'AP': AP.tolist(),
        }
