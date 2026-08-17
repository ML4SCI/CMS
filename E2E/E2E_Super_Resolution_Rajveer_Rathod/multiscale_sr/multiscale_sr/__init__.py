"""Multi-scale super-resolution (Option A): independent GAN per downsample scale.

Reuses the SRGAN bicubic-residual architecture but trains a separate model for
each low-resolution scale (e.g. 64, 32, 16) against a fixed HR ground truth.
The LR input at each scale is produced by area-downsampling the HR image, so the
same HR target is reconstructed from progressively coarser inputs. This isolates
how reconstruction quality and physics fidelity degrade as input resolution drops.
"""
from __future__ import annotations

__version__ = "0.1.0"
