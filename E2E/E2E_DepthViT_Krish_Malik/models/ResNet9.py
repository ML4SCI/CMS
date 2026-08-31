"""
ResNet9.py — matched-size CNN baseline for DepthViT on HLS4ML LHC jets.
"""
from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = ConvBlock(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class ResNet9(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 5,
        widths: Tuple[int, int, int, int] = (14, 24, 44, 78),
    ):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.conv1 = ConvBlock(in_channels, w1)
        self.conv2 = ConvBlock(w1, w2, pool=True)
        self.res1  = ResidualBlock(w2)
        self.conv3 = ConvBlock(w2, w3, pool=True)
        self.conv4 = ConvBlock(w3, w4, pool=True)
        self.res2  = ResidualBlock(w4)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(w4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
