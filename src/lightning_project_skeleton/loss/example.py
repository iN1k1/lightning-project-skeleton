from .base import BaseLoss
from torch.nn import functional as F


class DummyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.cross_entropy(pred, target)