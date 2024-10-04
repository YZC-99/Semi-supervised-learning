import pandas as pd

# 读取 CSV 文件
train_refer = pd.read_csv("D:/Dev_projects/SSL/Semi-supervised-learning/data/cxr8/refer/train.csv")
current_train = pd.read_csv("D:/Dev_projects/SSL/Semi-supervised-learning/data/cxr8/train_dataset.csv")

# 提取 train_refer 中的 image 顺序
train_refer_images = train_refer['image'].tolist()

# 按照 train_refer 中的顺序对 current_train 进行排序
current_train_sorted = current_train.set_index('image_id').loc[train_refer_images].reset_index()

# 保存排序后的结果
current_train_sorted.to_csv("D:/Dev_projects/SSL/Semi-supervised-learning/data/cxr8/train_dataset_sorted.csv", index=False)
