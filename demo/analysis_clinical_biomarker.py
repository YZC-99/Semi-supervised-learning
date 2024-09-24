import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
csv_data_path = "D:/Dev_projects/SSL/Semi-supervised-learning/data/olives_5/train_dataset.csv"
data = pd.read_csv(csv_data_path)

# 准备标签和cst值
y = data.iloc[:, 2:7]  # 假设这些是分类标签
cst = data.iloc[:, -2]  # 指定cst列

# 创建一个字典来保存每个类别的唯一cst值
unique_csts_per_category = {}

# 1. 打印每个类别所包含的cst唯一值
for column in y.columns:
    indices = y[y[column] == 1].index
    unique_csts = cst.iloc[indices].unique()
    unique_csts_per_category[column] = unique_csts
    print(f"Category '{column}' has the following unique 'cst' values: {unique_csts}")

# 创建一个DataFrame来记录每个cst值在每个类别中的出现次数
cst_counts = pd.DataFrame(index=pd.unique(cst), columns=y.columns)

# 初始化cst_counts DataFrame
for column in cst_counts.columns:
    cst_counts[column] = 0

# 更新cst_counts DataFrame以记录每个cst的计数
for column in y.columns:
    class_csts = cst[y[column] == 1]
    cst_counts[column] += class_csts.value_counts()


