import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# 读取数据
csv_data_path = "D:/Dev_projects/SSL/Semi-supervised-learning/data/cxr8/all_dataset.csv"
data = pd.read_csv(csv_data_path)

# 准备特征和标签
X = data.iloc[:, 0].values  # 图片名（image_id）
y = data.iloc[:, 1:15].values  # 多标签（14 个疾病的 one-hot 编码）

# 定义保存路径
save_dir = os.path.dirname(csv_data_path)

# 第一步：使用 MultilabelStratifiedShuffleSplit 将数据分为训练验证集 (80%) 和 测试集 (20%)
msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_val_index, test_index in msss1.split(X, y):
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    train_val_data = data.iloc[train_val_index]
    test_data = data.iloc[test_index]

# 第二步：在训练验证集 (80%) 中，再次使用 MultilabelStratifiedShuffleSplit 将其分为训练集 (70%) 和 验证集 (10%)
# 由于训练验证集占 80%，验证集需要占总数据的 10%，所以验证集占训练验证集的比例为 10% / 80% = 0.125
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)

for train_index, val_index in msss2.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    train_data = train_val_data.iloc[train_index]
    val_data = train_val_data.iloc[val_index]

# 保存到 CSV 文件
train_data.to_csv(f'{save_dir}/train_dataset.csv', index=False)
val_data.to_csv(f'{save_dir}/val_dataset.csv', index=False)
test_data.to_csv(f'{save_dir}/test_dataset.csv', index=False)

# 打印数据集大小以确认分割正确
print(f'Train dataset size: {len(train_data)}')
print(f'Validation dataset size: {len(val_data)}')
print(f'Test dataset size: {len(test_data)}')




# import os.path
#
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
#
# # 读取数据
# csv_data_path = "D:/Dev_projects/SSL/Semi-supervised-learning/data/cxr8/all_dataset.csv"
# data = pd.read_csv(csv_data_path)
#
# # 准备特征和标签
# X = data.iloc[:, 0].values.reshape(-1, 1)  # 图片名
# y = data.iloc[:, 2:17].astype(str).agg('-'.join, axis=1).values  # 合并多标签形成一个唯一的标签，用于分层
#
# # 分层采样
# # 总共分成10份，其中2份为测试集
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# # 初始化数据集容器
# train_data = pd.DataFrame()
# val_data = pd.DataFrame()
# test_data = pd.DataFrame()
#
# # 使用 StratifiedKFold 对数据集进行分割
# for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
#     if fold < 2:  # 分配 20% 数据作为测试集
#         test_data = test_data.append(data.iloc[test_idx])
#     else:  # 剩余的数据中，进一步将1/8作为验证集
#         train_idx, val_idx = next(StratifiedKFold(n_splits=8, shuffle=True, random_state=42).split(X[train_val_idx], y[train_val_idx]))
#         train_data = train_data.append(data.iloc[train_val_idx[train_idx]])
#         val_data = val_data.append(data.iloc[train_val_idx[val_idx]])
#         break  # 只需要处理一次分割，因此使用break终止循环
#
# # 保存到csv文件
# save_dir = os.path.dirname(csv_data_path)
# train_data.to_csv(f'{save_dir}/train_dataset.csv', index=False)
# val_data.to_csv(f'{save_dir}/val_dataset.csv', index=False)
# test_data.to_csv(f'{save_dir}/test_dataset.csv', index=False)
#
# # 打印数据集大小以确认分割正确
# train_data.shape, val_data.shape, test_data.shape