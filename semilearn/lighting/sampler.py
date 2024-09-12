from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np

class Memory_NoReplacement_Sampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = np.arange(len(data_source))
        self.current_position = 0

    def __iter__(self):
        np.random.shuffle(self.indices)  # 每个 epoch 开始时进行 shuffle
        start = self.current_position
        end = len(self.indices)
        indices = self.indices[start:end].tolist()
        self.current_position = 0  # Reset for next epoch
        return iter(indices)

    def __len__(self):
        return len(self.indices) - self.current_position

    def set_current_position(self, position):
        self.current_position = position