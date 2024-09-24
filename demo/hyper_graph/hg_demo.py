import torch
from hypergraph import binarize

targets = torch.tensor([2,2,4,6,8])
num_classes = 100

P_one_hot = binarize(targets,num_classes) # (4,100)
print(P_one_hot)
class_within_batch = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1) # (2,4,6,8)
H = P_one_hot[:, class_within_batch]

# 主要是看看HGNN的输入格式是什么
# print(H)
