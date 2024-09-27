import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score

import os
import re
import torch

# def get_target_names_labels():
#     # 读取目标标签
#     target_csv = "/home/gu721/yzc/Semi-supervised-learning/data/olives_5/test_dataset.csv"
#     target_all_info = pd.read_csv(target_csv)
#     target_all_info = target_all_info.fillna(0)
#     target_names = target_all_info.iloc[:, 0].values
#     target_labels = target_all_info.iloc[:, 2:7].values  # 假设标签在第3到第7列（索引从0开始）
#     return target_names, target_labels

def get_target_names_labels():
    # 读取目标标签
    target_csv = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/test_dataset.csv"
    target_all_info = pd.read_csv(target_csv)
    target_all_info = target_all_info.fillna(0)
    target_names = target_all_info.iloc[:, 0].values
    target_labels = target_all_info.iloc[:, 1:8].values  # 假设标签在第3到第7列（索引从0开始）
    return target_names, target_labels


def compute_metrics_single_classes(pred_logits_dir):
    # 读取真实的目标名称和标签
    target_names, target_labels = get_target_names_labels()

    # 读取预测的 logits
    pred_logits_csv = pred_logits_dir + "/test_logits.csv"
    pred_logits_df = pd.read_csv(pred_logits_csv)
    pred_names = pred_logits_df.iloc[:, 0].values
    pred_probs = pred_logits_df.iloc[:, 1:].values

    # 去除预测名称中的前缀
    prefix = "/home/gu721/yzc/data/ISIC2018/images/"
    pred_names = [name.replace(prefix, '') for name in pred_names]
    pred_names = [name.replace('.jpg', '') for name in pred_names]

    # 确保预测和标签的样本顺序一致
    name_to_index = {name: idx for idx, name in enumerate(target_names)}
    indices = [name_to_index[name] for name in pred_names]
    target_labels_ordered = target_labels[indices]

    # 对预测 logits 应用 softmax 并选取最大值作为预测类别
    pred_probs = torch.softmax(torch.tensor(pred_probs), dim=-1).numpy()
    predictions = np.argmax(pred_probs, axis=1)  # 预测的类别
    target_labels_ordered = np.argmax(target_labels_ordered, axis=1)  # 真实类别转换为一维

    # 计算平均准确率
    avg_ACC = accuracy_score(target_labels_ordered, predictions)

    # 计算每类的 AUC 和 AUPRC
    per_class_AUCs = []
    per_class_Specificity = []
    mAP = 0
    for i in range(target_labels_ordered.max() + 1):  # 针对每个类别
        true_binary = (target_labels_ordered == i).astype(int)  # 真实标签二值化
        pred_binary_probs = pred_probs[:, i]  # 预测概率

        try:
            auc_score = roc_auc_score(true_binary, pred_binary_probs)
        except ValueError:
            auc_score = np.nan  # 当某个类别没有正例时会报错，设置为 NaN

        precision, recall, _ = precision_recall_curve(true_binary, pred_binary_probs)
        auprc_score = auc(recall, precision)

        per_class_AUCs.append(auc_score)
        mAP += auc(recall, precision)

        # 计算 Specificity (特异性)
        tn = ((predictions != i) & (target_labels_ordered != i)).sum()  # True negatives
        fp = ((predictions == i) & (target_labels_ordered != i)).sum()  # False positives
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        per_class_Specificity.append(specificity)

    # 计算 F1 分数
    OF1 = f1_score(target_labels_ordered, predictions, average='macro')  # Overall F1
    CF1 = f1_score(target_labels_ordered, predictions, average='weighted')  # Class F1

    # 计算敏感性（Sensitivity）和精确度（Precision）
    Sensitivity = recall_score(target_labels_ordered, predictions, average='weighted')  # 敏感性
    Precision = precision_score(target_labels_ordered, predictions, average='micro')  # 精确度

    # 计算整体准确率（ACC）
    ACC = accuracy_score(target_labels_ordered, predictions)

    # 平均准确率（mAP）
    mAP = mAP / len(per_class_AUCs)

    # 返回所有计算好的指标
    return {
        'avg_ACC': avg_ACC * 100,
        'ACC': ACC * 100,
        'OF1': OF1 * 100,
        'CF1': CF1 * 100,
        'AUC': np.nanmean(per_class_AUCs) * 100,
        'AUPRC': np.nanmean(auprc_score) * 100,
        'mAP': mAP * 100,
        'Sensitivity': Sensitivity * 100,
        'Specificity': np.nanmean(per_class_Specificity) * 100,
        'Precision': Precision * 100,
        'per_class_AUCs': per_class_AUCs * 100,
        'per_class_Specificity': per_class_Specificity * 100
    }

def compute_metrics(pred_logits_dir):
    target_names, target_labels = get_target_names_labels()
    # 读取预测的 logits
    pred_logits_csv = pred_logits_dir + "/test_logits.csv"
    pred_logits_df = pd.read_csv(pred_logits_csv)
    pred_names = pred_logits_df.iloc[:, 0].values
    pred_probs = pred_logits_df.iloc[:, 1:].values


    # 去除预测名称中的前缀
    # prefix = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES"
    # pred_names = [name.replace(prefix, '') for name in pred_names]

    prefix = "/home/gu721/yzc/data/ISIC2018/images/"
    pred_names = [name.replace(prefix, '') for name in pred_names]
    pred_names = [name.replace('.jpg', '') for name in pred_names]

    # 确保预测和标签的样本顺序一致
    # 创建名称到索引的映射
    name_to_index = {name: idx for idx, name in enumerate(target_names)}
    # 获取预测结果中名称对应的索引
    indices = [name_to_index[name] for name in pred_names]
    # 重新排列目标标签，使其与预测结果的顺序一致
    target_labels_ordered = target_labels[indices]

    # 将概率阈值化为二进制预测，阈值为0.5
    predictions = (pred_probs >= 0.5).astype(int)

    # 计算 ACC：当且仅当所有标签都预测正确时，才算预测正确
    correct_per_sample = np.all(predictions == target_labels_ordered, axis=1)
    ACC = np.mean(correct_per_sample) * 100

    # 计算平均每样本准确率（Average Accuracy）
    per_sample_accuracy = np.mean(predictions == target_labels_ordered, axis=1)
    average_accuracy = np.mean(per_sample_accuracy) * 100

    # 计算 Overall Precision (OP) 和 Overall Recall (OR)
    tp = np.sum(predictions * target_labels_ordered)
    fp = np.sum(predictions * (1 - target_labels_ordered))
    fn = np.sum((1 - predictions) * target_labels_ordered)

    OP = tp / (tp + fp) if (tp + fp) > 0 else 0
    OR = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算 Overall F1 Score (OF1)
    OF1 = (2 * OP * OR / (OP + OR)) * 100 if (OP + OR) > 0 else 0

    # 计算 Per-class Precision (CP) 和 Per-class Recall (CR)
    per_class_tp = np.sum(predictions * target_labels_ordered, axis=0)
    per_class_fp = np.sum(predictions * (1 - target_labels_ordered), axis=0)
    per_class_fn = np.sum((1 - predictions) * target_labels_ordered, axis=0)

    per_class_precision = per_class_tp / (per_class_tp + per_class_fp)
    per_class_recall = per_class_tp / (per_class_tp + per_class_fn)

    # 处理可能出现的 NaN 值
    per_class_precision = np.nan_to_num(per_class_precision)
    per_class_recall = np.nan_to_num(per_class_recall)

    CP = np.mean(per_class_precision)
    CR = np.mean(per_class_recall)

    # 计算 Per-class F1 Score (CF1)
    CF1 = (2 * CP * CR / (CP + CR)) * 100 if (CP + CR) > 0 else 0

    # 计算 Sensitivity 和 Precision
    Sensitivity = CR * 100
    Precision = CP * 100

    # 计算每个类别的 TN
    per_class_tn = np.sum((1 - predictions) * (1 - target_labels_ordered), axis=0)

    # 计算每个类别的 Specificity（特异性）
    per_class_specificity = per_class_tn / (per_class_tn + per_class_fp)
    per_class_specificity = np.nan_to_num(per_class_specificity)

    # 计算平均特异性
    Specificity = np.mean(per_class_specificity) * 100

    # 将每个类别的特异性乘以 100
    per_class_specificity *= 100

    # 计算 AUC 和 AUPRC
    auc_scores = []
    auprc_scores = []
    for i in range(target_labels_ordered.shape[1]):
        try:
            # 计算 AUC
            auc_score = roc_auc_score(target_labels_ordered[:, i], pred_probs[:, i])
            auc_scores.append(auc_score)
            # 计算 AUPRC
            precision, recall, _ = precision_recall_curve(target_labels_ordered[:, i], pred_probs[:, i])
            auprc = auc(recall, precision)
            auprc_scores.append(auprc)
        except ValueError:
            # 当标签只有正类或负类时，无法计算 AUC 或 AUPRC
            auc_scores.append(np.nan)
            auprc_scores.append(np.nan)

    # 计算平均 AUC 和 AUPRC，排除 NaN 值
    valid_auc_scores = [score for score in auc_scores if not np.isnan(score)]
    valid_auprc_scores = [score for score in auprc_scores if not np.isnan(score)]

    avg_auc = np.mean(valid_auc_scores) * 100 if valid_auc_scores else 0
    avg_auprc = np.mean(valid_auprc_scores) * 100 if valid_auprc_scores else 0

    # 计算 mAP（Mean Average Precision）
    ap_scores = []
    for i in range(target_labels_ordered.shape[1]):
        try:
            ap = average_precision_score(target_labels_ordered[:, i], pred_probs[:, i])
            ap_scores.append(ap)
        except ValueError:
            ap_scores.append(np.nan)

    # 计算平均 AP，排除 NaN 值
    valid_ap_scores = [score for score in ap_scores if not np.isnan(score)]
    mAP = np.mean(valid_ap_scores) * 100 if valid_ap_scores else 0

    # 将每个类别的 AUC 也乘以 100，并添加到返回结果中
    per_class_AUCs = [score * 100 if not np.isnan(score) else score for score in auc_scores]

    return {
        'avg_ACC': average_accuracy,
        'ACC': ACC,
        'OF1': OF1,
        'CF1': CF1,
        'AUC': avg_auc,
        'AUPRC': avg_auprc,
        'mAP': mAP,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'Precision': Precision,
        'per_class_AUCs': per_class_AUCs,
        'per_class_Specificity': per_class_specificity.tolist()
    }

# 从 'pred_logits_dir' 中提取 'N' 后的数值，用于排序
def extract_N_value(path):
    match = re.search(r'N(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return None  # 如果未找到，返回 None

if __name__ == '__main__':
    import os
    # exp_root = "/dk1/oct_exp/"
    exp_root = "/dk1/isic2018-exp/"
    save_dir = "/home/gu721/yzc/Semi-supervised-learning/results/"
    save_path = os.path.join(save_dir, 'isic2018-results.csv')
    multi_label=False

    pred_list = []
    for root, dirs, files in os.walk(exp_root):
        # 如果files包含test_logits.csv
        if 'test_logits.csv' in files:
            pred_list.append(root)

    # 如果保存目录不存在，创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化一个空的 DataFrame，用于保存所有结果
    all_results_df = pd.DataFrame()

    for pred_logits_dir in pred_list:
        test_logits_path = os.path.join(pred_logits_dir, 'test_logits.csv')
        # 检查 test_logits.csv 是否存在
        if not os.path.exists(test_logits_path):
            continue
        if multi_label:
            results_dict = compute_metrics(pred_logits_dir)
        else:
            results_dict = compute_metrics_single_classes(pred_logits_dir)


        # 包装标量值为列表
        data = {
            'pred_logits_dir': [pred_logits_dir.replace('/dk1/oct_exp/resnet50_/', '')],
            'avg_ACC': [results_dict['avg_ACC']],
            'ACC': [results_dict['ACC']],
            'OF1': [results_dict['OF1']],
            'CF1': [results_dict['CF1']],
            'AUC': [results_dict['AUC']],
            'AUPRC': [results_dict['AUPRC']],
            'mAP': [results_dict['mAP']],
            'Sensitivity': [results_dict['Sensitivity']],
            'Precision': [results_dict['Precision']],
            'Specificity': [results_dict['Specificity']]
        }

        # 获取每个类别的 AUC，并添加到数据中
        per_class_AUCs = results_dict['per_class_AUCs']
        num_classes = len(per_class_AUCs)
        for i in range(num_classes):
            data[f'AUC_Class_{i + 1}'] = [per_class_AUCs[i]]

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data)

        # 将当前结果添加到所有结果的 DataFrame 中
        all_results_df = pd.concat([all_results_df, df], ignore_index=True)

    # # 将所有结果保存到 CSV 文件中
    # # 为 DataFrame 添加一列，用于存储提取的 'N' 值
    # all_results_df['N_value'] = all_results_df['pred_logits_dir'].apply(extract_N_value)
    #
    # # 删除 'N_value' 为 None 的行（如果有）
    # all_results_df = all_results_df.dropna(subset=['N_value'])
    #
    # # 根据 'N_value' 对 DataFrame 进行排序
    # all_results_df = all_results_df.sort_values(by='N_value')

    # 保存排序后的结果到 CSV 文件
    all_results_df.to_csv(save_path, index=False)
    print(f"结果已保存到 {save_path}")



