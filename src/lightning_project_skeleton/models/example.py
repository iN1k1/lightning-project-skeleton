import torch
import torch.nn as nn
import math
from timm.layers import trunc_normal_


def _common_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class DummyModel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_classes: int = 1,
    ):

        super(DummyModel, self).__init__()

        # Model
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, num_classes)

        # Initialize weights
        self.apply(_common_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {""}
        for n, _ in self.named_parameters():
            if "_layer_or_par_to_look_for" in n:
                nwd.add(n)
        return nwd

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
