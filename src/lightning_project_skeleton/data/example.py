import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from lightning_project_skeleton.build.from_config import instantiate_from_config


class ExampleDataset(Dataset):
    def __init__(self, phase: str, transform: dict):
        self._cifar = datasets.CIFAR10(
            './dataset/cifar10', train=phase == 'train', download=True
        )

        tx = {
            'transforms': [
                instantiate_from_config(t) for t in transform['params']
            ]
        }
        self._transform = instantiate_from_config(
            {'target': transform['target'], 'params': tx}
        )

    def __len__(self):
        return len(self._cifar)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        img, target = self._cifar[index]
        img = np.array(img).transpose((2, 0, 1)).astype(np.float32)

        return {
            'data': img,
            'target': target,
        }
