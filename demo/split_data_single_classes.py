import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# 读取数据
csv_data_path = "D:/Dev_projects/SSL/Semi-supervised-learning/data/ISIC2018/all_dataset.csv"
data = pd.read_csv(csv_data_path)

# 准备特征和标签
X = data.iloc[:, 0].values.reshape(-1, 1)  # 图片名

# 合并第2列到第7列为单一标签，用于分层
y = data.iloc[:, 1:7].astype(str).agg('-'.join, axis=1).values

# 第一步：分出20%作为测试集，80%留作训练和验证集
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(sss.split(X, y))

# 第二步：在剩下的80%数据中，分出1/8作为验证集，7/8作为训练集
sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)  # 1/8 = 0.125
train_idx, val_idx = next(sss_train_val.split(X[train_val_idx], y[train_val_idx]))

# 初始化数据集
train_data = data.iloc[train_val_idx[train_idx]]
val_data = data.iloc[train_val_idx[val_idx]]
test_data = data.iloc[test_idx]

# 获取csv文件的保存路径
output_dir = os.path.dirname(csv_data_path)

# 保存数据集到同级目录
train_data.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
val_data.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)

# 打印数据集大小以确认分割正确
print("Train set size:", train_data.shape)
print("Validation set size:", val_data.shape)
print("Test set size:", test_data.shape)
