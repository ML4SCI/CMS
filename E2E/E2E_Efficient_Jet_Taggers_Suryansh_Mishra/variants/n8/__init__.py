"""N8 — factorized pair bias for the Particle Transformer.

Replaces ParT's dense ``PairEmbed`` with extra query/key channels:

* rank-4 Minkowski ``d_ij / (E_i E_j)`` folded into Q/K,
* tensor-product rotary ``R(k φ) ⊗ M(a u)`` on ``(Δy, Δφ)`` with learned
  per-block ``λ_rotary``,
* optional degree-2 (``r = 14``) quadratic monomials.

The K6 screen is Minkowski + rotary, ``pair_embed_dims=None``.
"""

from .attention import FactorizedAttentionBlock, inv_softplus
from .features import (
    QUADRATIC_DIM,
    ROTARY_ANGLE_CLAMP,
    azimuth,
    jet_invariant_mass,
    minkowski_pair_matrix,
    minkowski_angle_pair_matrix,
    minkowski_qk_extras,
    quadratic_monomials,
    relative_rapidity,
    rotation_2d,
    rotary_qk_extras,
    tensor_product_basis,
    tensor_product_matrix,
)
from .part import N8ParT

__all__ = [
    "N8ParT",
    "FactorizedAttentionBlock",
    "inv_softplus",
    "QUADRATIC_DIM",
    "ROTARY_ANGLE_CLAMP",
    "azimuth",
    "jet_invariant_mass",
    "minkowski_pair_matrix",
    "minkowski_angle_pair_matrix",
    "minkowski_qk_extras",
    "quadratic_monomials",
    "relative_rapidity",
    "rotation_2d",
    "rotary_qk_extras",
    "tensor_product_basis",
    "tensor_product_matrix",
]
