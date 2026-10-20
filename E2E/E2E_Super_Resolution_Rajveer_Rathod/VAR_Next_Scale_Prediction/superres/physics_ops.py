"""Energy-conserving resolution changes. Work on numpy arrays or torch tensors
shaped (N, C, *spatial) with 2 or 3 spatial dims.

* sum_pool          : merge f_1 x f_2 (x f_3) cells into one; total energy unchanged.
                      This is what a detector with larger cells would record.
* uniform_upsample  : spread each coarse cell's energy evenly over its fine cells
                      (the energy-preserving "no learning" baseline).
* project_to_lr     : rescale an SR prediction so each block of fine cells sums
                      exactly to the measured coarse cell (hard energy consistency).
"""
import math

import numpy as np
import torch


def _is_torch(x):
    return isinstance(x, torch.Tensor)


def sum_pool(x, f):
    f = tuple(int(v) for v in f)
    if all(v == 1 for v in f):
        return x
    spatial = x.shape[2:]
    shape, axes = list(x.shape[:2]), []
    for d, (s, fi) in enumerate(zip(spatial, f)):
        if s % fi:
            raise ValueError(f"size {s} not divisible by pool factor {fi}")
        shape += [s // fi, fi]
        axes.append(3 + 2 * d)
    x = x.reshape(shape)
    return x.sum(dim=tuple(axes)) if _is_torch(x) else x.sum(axis=tuple(axes))


def repeat_upsample(x, f):
    for d, fi in enumerate(f):
        if fi > 1:
            x = x.repeat_interleave(int(fi), dim=2 + d) if _is_torch(x) else np.repeat(x, int(fi), axis=2 + d)
    return x


def uniform_upsample(x, f):
    return repeat_upsample(x, f) / math.prod(f)


def project_to_lr(sr, lr, f, eps=1e-8):
    """Rescale sr blockwise so sum_pool(out, f) == lr (up to eps)."""
    was_np = not _is_torch(sr)
    if was_np:
        sr, lr = torch.as_tensor(sr, dtype=torch.float32), torch.as_tensor(lr, dtype=torch.float32)
    sr = sr.clamp_min(0).float()
    lr = lr.clamp_min(0).float()
    pooled = sum_pool(sr, f)
    has = pooled > eps
    ratio = torch.where(has, lr / pooled.clamp_min(eps), torch.zeros_like(lr))
    out = sr * repeat_upsample(ratio, f)
    # coarse cell has energy but the prediction put none there: spread uniformly
    missing = torch.where((~has) & (lr > eps), lr, torch.zeros_like(lr))
    out = out + uniform_upsample(missing, f)
    return out.numpy() if was_np else out
