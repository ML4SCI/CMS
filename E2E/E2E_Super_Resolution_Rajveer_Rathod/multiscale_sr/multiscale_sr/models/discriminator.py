from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """PatchGAN discriminator with spectral norm.

    Accepts (lr, hr) pair: upsamples lr to hr spatial size, concatenates,
    and classifies patches as real/fake. Input spatial size is derived
    dynamically from hr, so any LR scale works without changes. Spectral
    norm on every conv constrains the Lipschitz constant and prevents the
    discriminator from collapsing the adversarial signal.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, lr: Tensor, hr: Tensor) -> Tensor:
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        return self.net(torch.cat([lr_up, hr], dim=1))
