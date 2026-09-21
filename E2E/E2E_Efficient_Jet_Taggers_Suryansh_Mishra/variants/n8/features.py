"""Kinematic extras for the N8 factorized pair bias.

N8 replaces ParT's dense ``PairEmbed`` (an MLP over ``[ln kT, ln z, ln ΔR, ln m²]``)
with extra query/key channels whose inner product *is* the pairwise structure:

* **Minkowski / opening-angle** (rank 4) — ``d_ij = E_i E_j - p_i · p_j``
  folded as ``Q_i · K_j = d_ij / (E_i E_j)``.  Per-particle four-vectors are
  scaled by ``1/E`` so typical collimated pairs are ``O(10⁻²–10⁻¹)`` (massless
  limit ``≈ 1 - cos θ``) rather than ``O(10⁻³)`` under ``d_ij / m_J²``.  This
  trades Lorentz invariance for a softmax-safe scale (PairEmbed's ``ΔR`` is
  also frame-dependent).
* **Tensor-product rotary** — four-dimensional blocks
  ``R(k φ) ⊗ M(a u)`` with integer azimuth harmonics ``k = 1 … K``,
  ``φ = atan2(py, px)``, and ``u = y - y_J``.  Block ``k`` scores
  ``cos(k Δφ) cos(a Δu)``, which is ``2π``-periodic in ``Δφ`` and *not*
  additive in ``Δy`` and ``Δφ`` (the reason ``ln ΔR`` is a nonlinear pair
  feature rather than two independent RoPEs).  A learned ``λ_rotary`` scales
  each block; ``a_k`` stays a rapidity frequency, not an amplitude gate.
* **Degree-2** — the ten symmetric quadratic monomials ``p^μ p^ν``,
  appended as Euclidean extras.  Combined with the rank-4 Minkowski
  term this is the ``r = 14`` follow-up from the T0.1 rank audit.

All functions accept weaver-order ``v`` of shape ``(B, 4, P)``.  Pairwise
products are never materialized here; the attention block does
``Q_extra K_extra^T`` as a separate logit term so the content scale
``1/sqrt(d_head)`` is not confounded by the extra width.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "ROTARY_ANGLE_CLAMP",
    "jet_invariant_mass",
    "minkowski_pair_matrix",
    "minkowski_angle_pair_matrix",
    "minkowski_qk_extras",
    "azimuth",
    "relative_rapidity",
    "rotation_2d",
    "tensor_product_matrix",
    "tensor_product_basis",
    "rotary_qk_extras",
    "quadratic_monomials",
    "QUADRATIC_DIM",
]

#: Clamp on ``a_k u`` so ``sin``/``cos`` stay well-defined in bf16.
ROTARY_ANGLE_CLAMP = 16.0

#: Unique ``p^μ p^ν`` components (symmetric 4×4, weaver channel order).
QUADRATIC_DIM = 10

_EPS = 1e-8


def jet_invariant_mass(v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Lorentz-invariant mass of the masked jet, shape ``(B,)``.

    ``m_J = sqrt(E_J² - |p_J|²)`` from the masked sum of ``v``.  Padded
    slots are ``v = 0`` under the loader contract, so omitting ``mask`` is
    equivalent when pads are already zeroed.
    """
    if mask is None:
        jet = v.sum(dim=-1)
    else:
        real = mask.to(dtype=v.dtype).reshape(v.shape[0], v.shape[-1])
        jet = (v * real.unsqueeze(1)).sum(dim=-1)
    mass_sq = (
        jet[:, 3].square()
        - jet[:, 0].square()
        - jet[:, 1].square()
        - jet[:, 2].square()
    )
    return mass_sq.clamp_min(_EPS).sqrt()


def minkowski_pair_matrix(v: Tensor) -> Tensor:
    """Pairwise Minkowski product ``d_ij = E_i E_j - p_i · p_j``.

    Parameters
    ----------
    v : Tensor
        ``(B, 4, P)`` weaver four-vectors ``[px, py, pz, E]``.

    Returns
    -------
    Tensor
        ``(B, P, P)``.
    """
    px, py, pz, energy = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    return (
        energy.unsqueeze(-1) * energy.unsqueeze(-2)
        - px.unsqueeze(-1) * px.unsqueeze(-2)
        - py.unsqueeze(-1) * py.unsqueeze(-2)
        - pz.unsqueeze(-1) * pz.unsqueeze(-2)
    )


def minkowski_angle_pair_matrix(v: Tensor) -> Tensor:
    """Opening-angle kernel ``d_ij / (E_i E_j)`` (massless limit ``≈ 1 - cos θ``).

    Parameters
    ----------
    v : Tensor
        ``(B, 4, P)`` weaver four-vectors ``[px, py, pz, E]``.

    Returns
    -------
    Tensor
        ``(B, P, P)``.
    """
    energy = v[:, 3].clamp_min(_EPS)
    denom = energy.unsqueeze(-1) * energy.unsqueeze(-2)
    return minkowski_pair_matrix(v) / denom.clamp_min(_EPS)


def minkowski_qk_extras(
    v: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Folded four-momenta such that ``Q_i · K_j = d_ij / (E_i E_j)``.

    Parameters
    ----------
    v : Tensor
        ``(B, 4, P)`` weaver four-vectors.
    mask : Tensor or None
        Unused (kept for API compatibility with :meth:`set_momenta` callers).
    normalize : bool
        Divide each four-vector by its energy so the extra logit is the
        opening-angle kernel ``d_ij / (E_i E_j)``.  Leave ``False`` only for
        the raw ``d_ij`` identity test.

    Returns
    -------
    q, k : Tensor
        Each ``(B, P, 4)``.  Scale by ``sqrt(λ_h)`` per head in the block.
    """
    del mask
    px, py, pz, energy = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    q = torch.stack([px, py, pz, energy], dim=-1)
    k = torch.stack([-px, -py, -pz, energy], dim=-1)
    if normalize:
        inv_e = (1.0 / energy.clamp_min(_EPS)).unsqueeze(-1)
        q = q * inv_e
        k = k * inv_e
    return q, k


def azimuth(v: Tensor) -> Tensor:
    """``φ = atan2(py, px)``, shape ``(B, P)``."""
    return torch.atan2(v[:, 1], v[:, 0])


def relative_rapidity(v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Beam-axis rapidity relative to the jet, ``u_i = y_i - y_J``.

    ``y = (1/2) log((E + pz) / (E - pz))``.  The jet four-momentum is the
    masked sum; padded slots (``v = 0``) contribute nothing and are
    zeroed in the output.

    Parameters
    ----------
    v : Tensor
        ``(B, 4, P)``.
    mask : Tensor or None
        ``(B, 1, P)`` or ``(B, P)``, 1/True for real particles.

    Returns
    -------
    Tensor
        ``(B, P)``.
    """
    pz, energy = v[:, 2], v[:, 3]
    y = 0.5 * torch.log(
        (energy + pz).clamp_min(_EPS) / (energy - pz).clamp_min(_EPS)
    )
    if mask is None:
        real = None
        jet = v.sum(dim=-1)
    else:
        real = mask.to(dtype=v.dtype).reshape(v.shape[0], v.shape[-1])
        jet = (v * real.unsqueeze(1)).sum(dim=-1)
    jet_e, jet_pz = jet[:, 3], jet[:, 2]
    y_jet = 0.5 * torch.log(
        (jet_e + jet_pz).clamp_min(_EPS) / (jet_e - jet_pz).clamp_min(_EPS)
    )
    u = y - y_jet.unsqueeze(-1)
    if real is not None:
        u = u * real
    return u


def rotation_2d(angle: Tensor) -> Tensor:
    """``SO(2)`` matrix ``[[cos, -sin], [sin, cos]]``.

    Parameters
    ----------
    angle : Tensor
        Arbitrary shape.

    Returns
    -------
    Tensor
        ``angle.shape + (2, 2)``.
    """
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    row0 = torch.stack([cosine, -sine], dim=-1)
    row1 = torch.stack([sine, cosine], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def tensor_product_matrix(alpha: Tensor, beta: Tensor) -> Tensor:
    """Kronecker product ``R(α) ⊗ M(β)`` as a ``(..., 4, 4)`` matrix."""
    rotation = rotation_2d(alpha)
    rapidity = rotation_2d(beta)
    kron = rotation[..., :, None, :, None] * rapidity[..., None, :, None, :]
    return kron.reshape(alpha.shape + (4, 4))


def tensor_product_basis(alpha: Tensor, beta: Tensor) -> Tensor:
    """``(R(α) ⊗ M(β)) e_0`` with ``e_0 = (1, 0, 0, 0)``.

    Equals ``(cos α cos β, cos α sin β, sin α cos β, sin α sin β)``.  The
    Euclidean product of two such vectors is ``cos(Δα) cos(Δβ)``.
    """
    c_a, s_a = torch.cos(alpha), torch.sin(alpha)
    c_b, s_b = torch.cos(beta), torch.sin(beta)
    return torch.stack(
        [c_a * c_b, c_a * s_b, s_a * c_b, s_a * s_b], dim=-1
    )


def rotary_qk_extras(
    phi: Tensor,
    u: Tensor,
    scale: Tensor,
    num_pairs: int,
    clamp: float = ROTARY_ANGLE_CLAMP,
) -> Tensor:
    """Tensor-product rotary extras, shared by Q and K.

    Block ``k`` (``k = 1 … num_pairs``) uses the integer azimuth harmonic
    ``α = k φ`` so the pairwise score is ``cos(k Δφ) cos(a_k Δu)`` — wrap
    safe on ``φ ∈ (-π, π]``.  The four channels per block are
    ``(cos kφ, sin kφ) ⊗ (cos a_k u, sin a_k u)``.

    Parameters
    ----------
    phi, u : Tensor
        ``(B, P)`` azimuth and relative rapidity.
    scale : Tensor
        Learnable per-head rapidity frequencies ``a_k``, shape ``(H, K)``.
    num_pairs : int
        Number of ``4``-dim blocks (``K`` above).
    clamp : float
        Absolute clamp on ``a_k u`` (bf16-safe ``sin``/``cos``).

    Returns
    -------
    Tensor
        ``(B, H, P, 4 K)``.
    """
    harmonics = torch.arange(1, num_pairs + 1, device=phi.device, dtype=phi.dtype)
    alpha = phi[:, None, :, None] * harmonics.view(1, 1, 1, num_pairs)
    beta = (scale[None, :, None, :] * u[:, None, :, None]).clamp(-clamp, clamp)
    return tensor_product_basis(alpha, beta).flatten(-2, -1)


def quadratic_monomials(v: Tensor) -> Tensor:
    """Symmetric degree-2 monomials ``p^μ p^ν``, weaver channel order.

    Column order matches the rank-audit helper: non-decreasing index pairs
    ``(0,0), (0,1), …, (3,3)`` over ``[px, py, pz, E]``.  Divided by
    ``E²`` so the extras stay O(1) in bf16.

    Returns
    -------
    Tensor
        ``(B, P, 10)``.
    """
    p = v.transpose(1, 2)  # (B, P, 4)
    columns = [
        p[..., i] * p[..., j]
        for i in range(4)
        for j in range(i, 4)
    ]
    phi = torch.stack(columns, dim=-1)
    energy_sq = p[..., 3:4].square().clamp_min(_EPS)
    return phi / energy_sq
