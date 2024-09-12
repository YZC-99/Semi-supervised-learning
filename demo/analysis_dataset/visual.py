import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

label_to_number = {
    "IR HRF": 1,
    "Fully attached vitreous face": 2,
    "Fluid (IRF)": 3,
    "DRT/ME": 4,
    "Partially attached vitreous face": 5,
    "Vitreous debris": 6,
    "Preretinal tissue/hemorrhage": 7,
    "Disruption of EZ": 8,
    "IR hemorrhages": 9,
    "Fluid (SRF)": 10,
    "Atrophy / thinning of retinal layers": 11,
    "SHRM": 12,
    "DRIL": 13,
    "PED (serous)": 14,
    "Disruption of RPE": 15,
    "VMT": 16
}

def visual_biomarker_bar_on_dataset(csv_path,sigmoid=False):
    dataset_name = csv_path.split("/")[-1].split(".")[0]
    dataset = pd.read_csv(csv_path)
    # 从第2列到第18列是标签
    labels = dataset.iloc[:,2:18]
    if sigmoid:
        labels = labels.applymap(lambda x: 1 if x>0.5 else 0)
    # 统计每一列的总和
    labels = labels.sum()
    # 按照数量排序，由大到小
    labels = labels.sort_values(ascending=False)
    # 创建一个从1开始的数字标签映射
    # 打印原标签和转化后的标签的对应关系
    for label, number in label_to_number.items():
        print(f"原标签：{label} -> 转化后的数字标签：{number}")
    labels = pd.Series(labels.values, index=[label_to_number[label] for label in labels.index])
    # 使用plt绘制柱状图，标签竖直显示
    plt.figure(figsize=(12, 10))
    # 设置坐标轴标签字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("The distribution of labels in {}".format(dataset_name), fontsize=20)
    labels.plot(kind='bar')
    plt.savefig("{}_bar.png".format(dataset_name))

def visual_clinical_bar_on_dataset(csv_path, clinical):
    dataset_name = csv_path.split("/")[-1].split(".")[0]
    dataset = pd.read_csv(csv_path)
    # 根据clinical获取指定列
    labels = dataset[clinical]
    # 首先计算labels的唯一值，然后统计每个唯一值的数量，随后按照数量排序，由大到小
    label_counts = labels.value_counts().sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    # 绘制柱状图
    label_counts.plot(kind='bar')
    plt.xticks(fontsize=10, rotation=45)  # 旋转标签以便于阅读
    plt.yticks(fontsize=20)
    plt.title("The distribution of clinical labels in {}".format(dataset_name), fontsize=24)
    plt.xlabel(clinical, fontsize=22)  # 添加x轴标签
    plt.ylabel('Counts', fontsize=22)  # 添加y轴标签
    plt.savefig("{}_{}_bar.png".format(dataset_name,clinical))
    plt.show()  # 显示图表
if __name__ == '__main__':
    # visual_biomarker_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/data/olives/train_dataset.csv")
    # visual_biomarker_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/data/olives/val_dataset.csv")
    # visual_biomarker_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/data/olives/test_dataset.csv")

    # visual_biomarker_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/demo/analysis_dataset/selection-bais/fixdamatch_SGD_extr0.0_uratio3_nlratio0.05__lr0.01_num_train_iter100000_bs64_seed42.csv",sigmoid=True)
    # visual_biomarker_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/demo/analysis_dataset/selection-bais/fixmatch_SGD_extr0.0_uratio3_nlratio0.05__lr0.01_num_train_iter100000_bs64_seed42.csv",sigmoid=True)

    visual_clinical_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/data/olives/train_dataset.csv",'BCVA')
    # visual_clinical_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/data/olives/val_dataset.csv")
    # visual_clinical_bar_on_dataset("/home/gu721/yzc/Semi-supervised-learning/data/olives/test_dataset.csv")