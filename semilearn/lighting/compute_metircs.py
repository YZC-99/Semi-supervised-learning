import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import roc_auc_score, precision_recall_curve

def compute_metrics(targets : np.array, pred_probs : np.array, num_classes=7,multi_label=False):
    THRESH = 0.18
    # print(target_labels_ordered.shape, pred_probs.shape) # (N,) (N, num_classes)
    AUROCs, Accs, Senss, Recas, Specs, F1s = [], [], [], [], [], []
    for i in range(num_classes):
        if multi_label:
            try:
                AUROCs.append(roc_auc_score(targets[:,i], pred_probs[:, i]))
            except ValueError:
                AUROCs.append(0)
            try:
                Accs.append(accuracy_score(targets[:,i], pred_probs[:, i] > THRESH))
            except ValueError:
                Accs.append(0)
            try:
                Senss.append(sensitivity_score(targets[:,i], pred_probs[:, i] > THRESH))
            except ValueError:
                Senss.append(0)
            try:
                Recas.append(recall_score(targets[:,i], pred_probs[:, i] > THRESH))
            except ValueError:
                Recas.append(0)
            try:
                Specs.append(specificity_score(targets[:,i], pred_probs[:, i] > THRESH))
            except ValueError:
                Specs.append(0)
            try:
                F1s.append(f1_score(targets[:,i], pred_probs[:, i] > THRESH))
            except ValueError:
                F1s.append(0)
        else:
            try:
                AUROCs.append(roc_auc_score(targets == i, pred_probs[:, i]))
            except ValueError:
                AUROCs.append(0)
            try:
                Accs.append(accuracy_score(targets == i, pred_probs[:, i] > THRESH))
            except ValueError:
                Accs.append(0)
            try:
                Senss.append(sensitivity_score(targets == i, pred_probs[:, i] > THRESH))
            except ValueError:
                Senss.append(0)
            try:
                Recas.append(recall_score(targets == i, pred_probs[:, i] > THRESH))
            except ValueError:
                Recas.append(0)
            try:
                Specs.append(specificity_score(targets == i, pred_probs[:, i] > THRESH))
            except ValueError:
                Specs.append(0)
            try:
                F1s.append(f1_score(targets == i, pred_probs[:, i] > THRESH))
            except ValueError:
                F1s.append(0)
        AUC = np.mean(AUROCs) * 100
        SENS = np.mean(Senss) * 100
        SPEC = np.mean(Specs) * 100
        ACC = np.mean(Accs) * 100
        F1 = np.mean(F1s) * 100

    return {
        'AUC': AUC,
        'SENS': SENS,
        'SPEC': SPEC,
        'ACC': ACC,
        'F1': F1
    }