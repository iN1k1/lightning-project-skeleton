import numpy as np
import random
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self,
                 size:Tuple[int, int, int] = (224, 224, 3),
                 num_samples:int = 1000):
        self._size = size
        self._num_samples = num_samples

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # Generate random sample with data of size self._size and a random label
        data = np.random.rand(*self._size)
        target = random.randint(0, 1)
        return {'data': data.astype(np.float32),
                'target': target }