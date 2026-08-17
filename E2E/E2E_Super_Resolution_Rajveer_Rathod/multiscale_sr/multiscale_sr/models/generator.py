from __future__ import annotations

import math
import os

import torch.nn.functional as F
from torch import Tensor, nn

_DEBUG = os.environ.get("MULTISCALE_SR_DEBUG", "").lower() in ("1", "true", "yes")


class ResidualBlock(nn.Module):
    """Residual block with scaled identity skip (EDSR-style ×0.1 damping).

    InstanceNorm rather than BatchNorm: calorimeter images are sparse
    (~2-3% nonzero) and batches can be small, where BatchNorm statistics
    collapse. The 0.1 output scale keeps early-training gradients stable.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1 * self.block(x)


class PixelShuffleUpBlock(nn.Module):
    """Learned 2x upsample: conv to 4*C channels, PixelShuffle, then a residual block.

    Sub-pixel convolution (Shi et al. 2016) lets the network choose its own
    upsampling kernel instead of being locked into a fixed bicubic prior —
    important for sparse calorimeter images, where bicubic pre-blurring a
    coarse grid destroys structure before any learned layer sees it.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)
        self.refine = ResidualBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.shuffle(self.conv(x)))
        return self.refine(x)


class Generator(nn.Module):
    """Progressive-residual generator.

    Upsamples LR -> target_size in learned 2x stages (sub-pixel convolution +
    residual refinement), one stage per power-of-two the scale factor spans.
    HR is zero-padded 125->128 upstream (see ``multiscale.pad_to_size``) so
    target_size is 128x128 for parquet data — a clean power of two — meaning
    16/32/64 -> 128 all land exactly on an integer number of learned doubling
    stages with no bicubic remainder. The trailing ``F.interpolate(..., size=
    target_size)`` calls are then true no-ops for parquet and only matter as a
    safety net for non-power-of-two targets (e.g. an HDF5 hr_size override).

    At small input resolutions (e.g. 16x16 or 32x32) this staged approach
    builds up structure gradually instead of bicubic-upsampling straight to
    full HR resolution before any learned layer runs, which was smearing
    sparse energy deposits into blurry blobs the residual CNN then had no way
    to recover (16x/32x had near-zero or negative val_psnr_norm vs. 64x's ~9
    under the old single-shot bicubic-residual design).

    ``forward(lr, target_size)`` keeps the same signature as before: the
    number of learned stages is inferred per call from how many times 2x
    fits between the input size and target_size, so one checkpoint still
    works across LR scales.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
        lr_skip: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        # Add a bicubic-upsampled LR skip (global residual) to the learned
        # output. Helpful when LR already carries most of the low-frequency
        # energy, but for very sparse deposits it can inject a blurred prior the
        # residual head then has to subtract back out; expose it so it can be
        # disabled per experiment.
        self.lr_skip = lr_skip
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )
        # Enough stages to cover the coarsest scale used in this project (16x -> 125,
        # ceil(log2(125/16)) = 3); forward() only uses as many as a given call needs.
        self._max_up_stages = 3
        self.up_stages = nn.ModuleList(
            [PixelShuffleUpBlock(base_channels) for _ in range(self._max_up_stages)]
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.to_image = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3),
        )

    def forward(self, lr: Tensor, target_size: tuple[int, int]) -> Tensor:
        in_h = lr.shape[-2]
        target_h = target_size[0]
        n_stages = min(self._max_up_stages, max(0, math.ceil(math.log2(max(target_h / in_h, 1.0)))))
        if _DEBUG:
            print(f"[gen] in={in_h} target={target_h} n_stages={n_stages}")

        x = self.stem(lr)
        for stage in self.up_stages[:n_stages]:
            x = stage(x)

        # Nearest for the (usually no-op) remainder resize: bicubic overshoots
        # and rings on sparse deposits, pre-blurring peaks the residual head
        # then cannot recover. Nearest preserves peak magnitude and location.
        if x.shape[-2:] != tuple(target_size):
            x = F.interpolate(x, size=target_size, mode="nearest")
        x = self.body(x)
        out = self.to_image(x)

        if self.lr_skip:
            lr_up = F.interpolate(lr, size=target_size, mode="bicubic", align_corners=False)
            out = out + lr_up
        return out
