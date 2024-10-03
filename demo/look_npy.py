import numpy as np

npy_path3407 = "/home/gu721/yzc/Semi-supervised-learning/data/isic2018/singlelabel_idx/lb_singlelabel_350_seed3407_idx.npy"
npy3407 = np.load(npy_path3407)
npy_path42 = "/home/gu721/yzc/Semi-supervised-learning/data/isic2018/singlelabel_idx/lb_singlelabel_350_seed42_idx.npy"
npy42 = np.load(npy_path42)
# 比较两者是否相同
print(np.all(npy3407 == npy42))
print(npy42)
print(npy3407)