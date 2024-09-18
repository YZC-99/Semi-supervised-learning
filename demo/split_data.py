import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 读取数据
csv_data_path = "D:/Dev_projects/SSL/Semi-supervised-learning/data/olives_5/Biomarker_5_Clinical_Data_Images.csv"
data = pd.read_csv(csv_data_path)

# 准备特征和标签
X = data.iloc[:, 0].values.reshape(-1, 1)  # 图片名
y = data.iloc[:, 2:17].astype(str).agg('-'.join, axis=1).values  # 合并多标签形成一个唯一的标签，用于分层

# 分层采样
# 总共分成10份，其中2份为测试集
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 初始化数据集容器
train_data = pd.DataFrame()
val_data = pd.DataFrame()
test_data = pd.DataFrame()

# 使用 StratifiedKFold 对数据集进行分割
for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
    if fold < 2:  # 分配 20% 数据作为测试集
        test_data = test_data.append(data.iloc[test_idx])
    else:  # 剩余的数据中，进一步将1/8作为验证集
        train_idx, val_idx = next(StratifiedKFold(n_splits=8, shuffle=True, random_state=42).split(X[train_val_idx], y[train_val_idx]))
        train_data = train_data.append(data.iloc[train_val_idx[train_idx]])
        val_data = val_data.append(data.iloc[train_val_idx[val_idx]])
        break  # 只需要处理一次分割，因此使用break终止循环

# 保存到csv文件
train_data.to_csv('D:/Dev_projects/SSL/Semi-supervised-learning/data/olives_5/train_dataset.csv', index=False)
val_data.to_csv('D:/Dev_projects/SSL/Semi-supervised-learning/data/olives_5/val_dataset.csv', index=False)
test_data.to_csv('D:/Dev_projects/SSL/Semi-supervised-learning/data/olives_5/test_dataset.csv', index=False)

# 打印数据集大小以确认分割正确
train_data.shape, val_data.shape, test_data.shape
