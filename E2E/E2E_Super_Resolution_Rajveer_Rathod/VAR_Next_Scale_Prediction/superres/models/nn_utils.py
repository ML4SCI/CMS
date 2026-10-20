"""Dimension-agnostic (2-D jets / 3-D showers) conv building blocks.
Axes listed in circular_dims (e.g. the calorimeter's angular axis) are padded
periodically; all others are zero padded."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNd(nn.Module):
    def __init__(self, ndim, cin, cout, k=3, stride=1, circular_dims=(), bias=True):
        super().__init__()
        self.ndim, self.p, self.circ = ndim, k // 2, set(circular_dims)
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.conv = conv(cin, cout, k, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        p = self.p
        if p:
            pads = []
            for d in reversed(range(self.ndim)):
                q = 0 if d in self.circ else p
                pads += [q, q]
            x = F.pad(x, pads)
            for d in self.circ:
                ax = 2 + d
                n = x.size(ax)
                x = torch.cat([x.narrow(ax, n - p, p), x, x.narrow(ax, 0, p)], dim=ax)
        return self.conv(x)


def norm(c):
    g = 32 if c % 32 == 0 else (8 if c % 8 == 0 else 1)
    return nn.GroupNorm(g, c, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, ndim, cin, cout, circ=(), dropout=0.0):
        super().__init__()
        self.body = nn.Sequential(
            norm(cin), nn.SiLU(), ConvNd(ndim, cin, cout, 3, circular_dims=circ),
            norm(cout), nn.SiLU(), nn.Dropout(dropout), ConvNd(ndim, cout, cout, 3, circular_dims=circ))
        self.skip = nn.Identity() if cin == cout else ConvNd(ndim, cin, cout, 1)

    def forward(self, x):
        return self.skip(x) + self.body(x)


class Upsample(nn.Module):
    def __init__(self, ndim, c, factor, circ=()):
        super().__init__()
        self.factor = tuple(float(f) for f in factor)
        self.conv = ConvNd(ndim, c, c, 3, circular_dims=circ)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.factor, mode="nearest"))


def conv_out(size, stride):
    """Output size of a k=3, pad=1 convolution."""
    return (size - 1) // stride + 1
