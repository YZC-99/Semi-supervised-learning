import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
import os
import re
import torch
from semilearn.lighting.compute_metircs import compute_metrics
# 自动检测当前文件所在的目录的绝对路径
ABS_FLAG = True
if 'gu721' not in os.path.dirname(os.path.abspath(__file__)):
    ABS_FLAG = False



def get_target_names_labels():
    # 读取目标标签

    target_csv = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/test_dataset.csv"
    if not ABS_FLAG:
        target_csv = "/root/Semi-supervised-learning/data/ISIC2018/train_dataset.csv"
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
    if not ABS_FLAG:
        prefix = "/root/autodl-tmp/ISIC2018/images/"
    pred_names = [name.replace(prefix, '') for name in pred_names]
    pred_names = [name.replace('.jpg', '') for name in pred_names]

    # 确保预测和标签的样本顺序一致
    name_to_index = {name: idx for idx, name in enumerate(target_names)}
    indices = [name_to_index[name] for name in pred_names]
    target_labels_ordered = target_labels[indices]

    # 对预测 logits 应用 softmax 并选取最大值作为预测类别
    pred_probs = torch.softmax(torch.tensor(pred_probs), dim=-1).numpy() # (N,num_classes)
    target_labels_ordered = np.argmax(target_labels_ordered, axis=1)  # (N,)
    # print(target_labels_ordered.shape, pred_probs.shape)
    return compute_metrics(target_labels_ordered, pred_probs,num_classes=7)




# 从 'pred_logits_dir' 中提取 'N' 后的数值，用于排序
def extract_N_value(path):
    match = re.search(r'nlratio([0-9\.]+)', path)

    if match:
        return match.group(1)
    else:
        return None  # 如果未找到，返回 None

if __name__ == '__main__':
    import os
    # exp_root = "/dk1/oct_exp/"
    exp_root = "/dk1/isic2018-exp/"
    save_dir = "/home/gu721/yzc/Semi-supervised-learning/results/"
    if not ABS_FLAG:
        exp_root = "/root/autodl-tmp/isic2018-exp/"
        save_dir = "/root/Semi-supervised-learning/results/"
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

        results_dict = compute_metrics_single_classes(pred_logits_dir)


        # 包装标量值为列表
        data = {
            'pred_logits_dir': [pred_logits_dir.replace('/dk1/oct_exp/resnet50_/', '')],
            'AUC': [results_dict['AUC']],
            'SENS': [results_dict['SENS']],
            'SPEC': [results_dict['SPEC']],
            'ACC': [results_dict['ACC']],
            'F1': [results_dict['F1']]
        }

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data)

        # 将当前结果添加到所有结果的 DataFrame 中
        all_results_df = pd.concat([all_results_df, df], ignore_index=True)

    # 将所有结果保存到 CSV 文件中
    # all_results_df['ratio_value'] = all_results_df['pred_logits_dir'].apply(extract_N_value)
    #
    # # 删除 'N_value' 为 None 的行（如果有）
    # all_results_df = all_results_df.dropna(subset=['ratio_value'])
    #
    # # 根据 'N_value' 对 DataFrame 进行排序
    # all_results_df = all_results_df.sort_values(by='ratio_value')
    # print(all_results_df)
    # # 保存排序后的结果到 CSV 文件
    # all_results_df.to_csv(save_path, index=False)
    # print(f"结果已保存到 {save_path}")

    # 将所有结果保存到 CSV 文件中
    all_results_df['ratio_value'] = all_results_df['pred_logits_dir'].apply(extract_N_value)

    # 删除 'ratio_value' 为 None 的行（如果有）
    all_results_df = all_results_df.dropna(subset=['ratio_value'])

    # 根据 'ratio_value' 对 DataFrame 进行排序
    all_results_df = all_results_df.sort_values(by='ratio_value')
    print(all_results_df)

    # 保存排序后的结果到 CSV 文件，使用 utf-8-sig 编码
    all_results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {save_path}")




