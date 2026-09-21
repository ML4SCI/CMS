"""Helpers to make LGATr more compile-friendly.

These patches are intentionally lightweight and opt-in:

- avoid cached device->device copies in invariant helper tensors
- replace a few dynamic ``cached_einsum`` call sites with static formulas

They are meant for benchmarking / experimentation and do not change model
math.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


_INNER_PRODUCT_FACTORS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]

_IP_TENSOR: torch.Tensor | None = None
_MG_TENSOR: torch.Tensor | None = None
_IP_DTYPE: torch.dtype | None = None


def _ensure_tensors(device, dtype):
    """Build the constant tensors once. Called before torch.compile."""
    global _IP_TENSOR, _MG_TENSOR, _IP_DTYPE
    if _IP_TENSOR is not None and _IP_TENSOR.device == torch.device(device) \
            and _IP_DTYPE == dtype:
        return
    _IP_TENSOR = torch.tensor(_INNER_PRODUCT_FACTORS, device=device, dtype=dtype)
    _IP_DTYPE = dtype
    m_grades = torch.zeros(5, 16, device=device, dtype=dtype)
    offset = 0
    for grade in range(5):
        width = math.comb(4, grade)
        m_grades[grade, offset : offset + width] = _IP_TENSOR[offset : offset + width]
        offset += width
    _MG_TENSOR = m_grades


def _fast_load_inner_product_factors(device, dtype) -> torch.Tensor:
    if _IP_TENSOR is not None and _IP_TENSOR.device == torch.device(device) \
            and _IP_TENSOR.dtype == dtype:
        return _IP_TENSOR
    return torch.tensor(_INNER_PRODUCT_FACTORS, device=device, dtype=dtype)


def _fast_load_metric_grades(device, dtype) -> torch.Tensor:
    if _MG_TENSOR is not None and _MG_TENSOR.device == torch.device(device) \
            and _MG_TENSOR.dtype == dtype:
        return _MG_TENSOR
    m = _fast_load_inner_product_factors(device, dtype)
    m_grades = torch.zeros(5, 16, device=device, dtype=dtype)
    offset = 0
    for grade in range(5):
        width = math.comb(4, grade)
        m_grades[grade, offset : offset + width] = m[offset : offset + width]
        offset += width
    return m_grades


def _fast_cached_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Static replacements for the most common LGATr einsums.

    Falls back to ``torch.einsum`` for anything outside the common hot path.
    """
    if equation == "... i, ... i -> ...":
        x, y = operands
        return (x * y).sum(dim=-1)

    if equation == "... i, ... i, g i -> ... g":
        x, y, g = operands
        return ((x * y).unsqueeze(-2) * g).sum(dim=-1)

    if equation == "g i j, ... j -> ... g i":
        basis, x = operands
        return torch.einsum("gij,...j->...gi", basis, x)

    return torch.einsum(equation, *operands)


def patch_lgatr_compile(device="cuda", dtype=torch.float32):
    """Monkey-patch a few LGATr helpers to be more compile-friendly.

    Pre-builds constant tensors on *device* so that ``torch.compile``
    never sees a dynamic cache lookup in the hot path.
    """
    _ensure_tensors(device, dtype)

    import lgatr.primitives.invariants as invariants_mod
    import lgatr.primitives.linear as linear_mod
    import lgatr.primitives.bilinear as bilinear_mod
    import lgatr.utils.einsum as einsum_mod

    invariants_mod._load_inner_product_factors = _fast_load_inner_product_factors
    invariants_mod._load_metric_grades = _fast_load_metric_grades

    einsum_mod.cached_einsum = _fast_cached_einsum
    linear_mod.cached_einsum = _fast_cached_einsum
    bilinear_mod.cached_einsum = _fast_cached_einsum
    invariants_mod.cached_einsum = _fast_cached_einsum

    print("[lgatr_kernels] Patched compile helpers (+invariant cache, +einsum fast paths)")
