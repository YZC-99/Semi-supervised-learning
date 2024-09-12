import os
import random
import numpy as np
import torch

def seed_everything(seed=2024):
    random.seed(seed)  # 设置Python内置random模块的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境变量，影响Python的hash函数
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch的CUDA随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有的CUDA设备设置随机种子
    # 一些cudnn的方法即使在固定种子后也可能是随机的，除非你告诉它要确定性的
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的基准测试模式
    torch.backends.cudnn.deterministic = True  # 开启cudnn的确定性模式
