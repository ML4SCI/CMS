"""Taggers. ResNet18Tagger: the qg jet tagger. Tagger: small residual CNN, used as jet tagger with
tagger.arch=cnn, as energy regressor (calo) and as classifier two-sample test. Both adapt their down-sampling
to the input grid, so the same architecture runs on HR, every LR grid and every SR output."""
import torch
import torch.nn as nn

from .nn_utils import ConvNd, ResBlock, conv_out, norm


class Tagger(nn.Module):
    def __init__(self, ndim, in_ch, input_shape, widths, out_dim=1, dropout=0.1, circ=(), n_global=0):
        super().__init__()
        layers, c, cur = [ConvNd(ndim, in_ch, widths[0], circular_dims=circ)], widths[0], list(input_shape)
        for w in widths:
            layers.append(ResBlock(ndim, c, w, circ))
            c = w
            stride = tuple(2 if s >= 8 else 1 for s in cur)
            if any(st > 1 for st in stride):
                layers.append(ConvNd(ndim, c, c, 3, stride=stride, circular_dims=circ))
                cur = [conv_out(s, st) for s, st in zip(cur, stride)]
        pool = nn.AdaptiveAvgPool2d(1) if ndim == 2 else nn.AdaptiveAvgPool3d(1)
        layers += [norm(c), nn.SiLU(), pool, nn.Flatten()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c + n_global, c), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(c, out_dim))

    def forward(self, x, g=None):
        h = self.body(x)
        if g is not None:
            h = torch.cat([h, g], 1)
        return self.head(h)


class BasicBlock(nn.Module):
    """ResNet basic block: conv3x3-BN-ReLU-conv3x3-BN plus skip (1x1 conv + BN when the shape changes)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        nn.init.zeros_(self.body[-1].weight)  # each block starts as the identity
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.skip(x) + self.body(x))


class ResNet18Tagger(nn.Module):
    """ResNet-18 (blocks [2, 2, 2, 2], widths 64-512) for 2-D jet images, with a 3x3 stride-1 stem and no
    max-pool so single-cell track hits survive. Each stage halves the map while it is >= 8 px (as Tagger),
    so HR 128 -> 8 and LR 64 -> 4. Same signature as Tagger; `widths` is ignored."""

    def __init__(self, ndim, in_ch, input_shape, widths=None, out_dim=1, dropout=0.1, circ=(), n_global=0):
        super().__init__()
        if ndim != 2 or circ:
            raise ValueError("ResNet18Tagger is 2-D only, without circular axes (use tagger.arch=cnn for calo)")
        layers, c, cur = [nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)], \
            64, list(input_shape)
        for w in (64, 128, 256, 512):
            stride = 2 if min(cur) >= 8 else 1
            layers += [BasicBlock(c, w, stride), BasicBlock(w, w)]
            c, cur = w, [conv_out(s, stride) for s in cur]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c + n_global, c), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(c, out_dim))
        for m in self.body.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    forward = Tagger.forward


TAGGERS = {"cnn": Tagger, "resnet18": ResNet18Tagger}
