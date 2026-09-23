"""Lorentz Local Canonicalization (LLoCa) — local reference frame construction.

Implements the frame-prediction half of LLoCa: an equivariant prediction of
three four-vectors per particle, followed by the deterministic polar-decomposition
algorithm that turns them into a local Lorentz frame.

Reference
---------
Spinner, Favaro, Lippmann, Pitz, Gerhartz, Plehn, Hamprecht,
"Lorentz Local Canonicalization: How to Make Any Network Lorentz-Equivariant",
arXiv:2505.20280.  Companion paper arXiv:2508.14898.
Upstream reference implementation: https://github.com/heidelberg-hepml/lloca

Equation numbers below refer to arXiv:2505.20280v2.

Conventions
-----------
Four-vectors in this module are **time-first**, ``p = (E, px, py, pz)``, matching
the paper's ``x = (x^0, vec x)``.  The project's data loader emits
``v = [px, py, pz, E]`` (weaver's convention), so callers must reorder before
entering this module — :func:`to_time_first` does that.

The Minkowski metric is ``g = diag(+1, -1, -1, -1)``.

A local frame ``L`` is a ``4x4`` matrix satisfying

* ``L g L^T = g``           (it is a Lorentz transformation), and
* ``L -> L Lambda^-1``      under a global Lorentz transformation ``Lambda``,

which is exactly what makes ``x_L = L x`` invariant (Eq. 6).

Numerical precision
-------------------
The paper evaluates the frame construction in double precision and only the
scalar MLP ``phi`` in single precision (App. D.1).  This module follows that:
:class:`FramesNet` runs ``phi`` in the incoming dtype and promotes to
``float64`` for the vector combination and the orthonormalization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "MINKOWSKI_SIGNATURE",
    "to_time_first",
    "to_weaver_order",
    "minkowski_dot",
    "minkowski_norm",
    "apply_metric",
    "boost_matrix",
    "gram_schmidt_3d",
    "frames_from_vectors",
    "invert_frames",
    "REFERENCE_PARTICLES",
    "FramesNet",
]

#: ``g = diag(+1, -1, -1, -1)`` as a signature vector (time-first convention).
MINKOWSKI_SIGNATURE: Tuple[float, float, float, float] = (1.0, -1.0, -1.0, -1.0)

#: ``epsilon`` used in the normalizations of Eq. (17) / Eq. (25).
_EPS_NORM = 1e-15

#: Collinearity threshold for the 3D Gram-Schmidt guard (App. D.1).
_EPS_COLLINEAR = 1e-16

#: Symmetry-breaking reference particles (App. D.3), time-first ``(E, px, py, pz)``:
#: the global time direction and the two beam directions.
REFERENCE_PARTICLES: Tuple[Tuple[float, float, float, float], ...] = (
    (1.0, 0.0, 0.0, 0.0),   # time direction
    (1.0, 0.0, 0.0, 1.0),   # beam +z
    (1.0, 0.0, 0.0, -1.0),  # beam -z
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def to_time_first(v: Tensor) -> Tensor:
    """Reorder weaver four-vectors ``[px, py, pz, E]`` to ``[E, px, py, pz]``.

    Parameters
    ----------
    v : Tensor
        Shape ``(..., 4)`` with the weaver channel order.

    Returns
    -------
    Tensor
        Shape ``(..., 4)`` time-first.
    """
    return torch.cat([v[..., 3:4], v[..., 0:3]], dim=-1)


def to_weaver_order(p: Tensor) -> Tensor:
    """Inverse of :func:`to_time_first`: ``[E, px, py, pz] -> [px, py, pz, E]``."""
    return torch.cat([p[..., 1:4], p[..., 0:1]], dim=-1)


# ---------------------------------------------------------------------------
# Minkowski algebra
# ---------------------------------------------------------------------------

def minkowski_dot(a: Tensor, b: Tensor, keepdim: bool = False) -> Tensor:
    """Minkowski product ``<a, b> = a0 b0 - vec a . vec b`` (Eq. 1).

    Parameters
    ----------
    a, b : Tensor
        Broadcastable, shape ``(..., 4)``, time-first.
    keepdim : bool
        Keep the contracted trailing axis.

    Returns
    -------
    Tensor
        Shape ``(...)`` or ``(..., 1)``.
    """
    out = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(dim=-1)
    return out.unsqueeze(-1) if keepdim else out


def minkowski_norm(a: Tensor, keepdim: bool = False) -> Tensor:
    """``||a|| = sqrt(|<a, a>|)`` — the norm used throughout App. C.1.

    The absolute value keeps the norm real for space-like vectors; LLoCa only
    ever feeds it forward-time-like vectors, where ``<a, a> > 0``.
    """
    sq = minkowski_dot(a, a, keepdim=keepdim)
    return torch.sqrt(sq.abs().clamp_min(0.0))


def apply_metric(a: Tensor) -> Tensor:
    """Return ``g a``, i.e. negate the spatial part of a time-first four-vector.

    Used to fold the metric into an otherwise Euclidean dot product so that
    fused attention kernels can evaluate Minkowski products (Sec. 4.3).
    """
    g = a.new_tensor(MINKOWSKI_SIGNATURE)
    return a * g


# ---------------------------------------------------------------------------
# Polar decomposition: boost + rotation
# ---------------------------------------------------------------------------

def boost_matrix(v0: Tensor) -> Tensor:
    """Build the boost ``B(v0)`` that carries ``v0`` to its rest frame (Eq. 2).

    With ``beta = vec v0 / v0^0`` and ``gamma = (1 - beta^2)^(-1/2)``::

        B = [[ gamma,        -gamma beta^T                      ],
             [-gamma beta,    I3 + (gamma - 1) beta beta^T / beta^2]]

    Applying it gives ``B(v0) v0 = (||v0||, 0, 0, 0)``.

    Parameters
    ----------
    v0 : Tensor
        Shape ``(..., 4)``, time-first, forward time-like
        (``<v0, v0> > 0`` and ``v0^0 > 0``).

    Returns
    -------
    Tensor
        Shape ``(..., 4, 4)``.

    Notes
    -----
    ``(gamma - 1) / beta^2`` is evaluated as the algebraically identical
    ``gamma^2 / (gamma + 1)``, which stays finite as ``beta -> 0`` (where the
    literal form is ``0 / 0``) and avoids catastrophic cancellation for small
    boosts.
    """
    energy = v0[..., 0:1]
    beta = v0[..., 1:] / energy.clamp_min(torch.finfo(v0.dtype).tiny)  # (..., 3)

    beta_sq = (beta * beta).sum(dim=-1, keepdim=True)  # (..., 1)
    # A forward time-like v0 has beta^2 < 1; clamp guards against round-off
    # pushing a near-null vector onto or past the light cone.
    beta_sq = beta_sq.clamp(max=1.0 - 1e-9)
    gamma = torch.rsqrt(1.0 - beta_sq)  # (..., 1)

    # (gamma - 1) / beta^2 == gamma^2 / (gamma + 1)
    coeff = gamma * gamma / (gamma + 1.0)  # (..., 1)

    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)  # (..., 3, 3)
    eye3 = torch.eye(3, dtype=v0.dtype, device=v0.device).expand_as(outer)
    spatial = eye3 + coeff.unsqueeze(-1) * outer  # (..., 3, 3)

    gamma_beta = gamma * beta  # (..., 3)

    top = torch.cat([gamma, -gamma_beta], dim=-1).unsqueeze(-2)  # (..., 1, 4)
    bottom = torch.cat([-gamma_beta.unsqueeze(-1), spatial], dim=-1)  # (..., 3, 4)
    return torch.cat([top, bottom], dim=-2)  # (..., 4, 4)


def gram_schmidt_3d(
    w1: Tensor, w2: Tensor, eps_collinear: float = _EPS_COLLINEAR
) -> Tuple[Tensor, Tensor, Tensor]:
    """3D Gram-Schmidt orthonormalization ``GS^3`` of Eq. (25).

    Returns an orthonormal right-handed triad ``(u1, u2, u3)`` with
    ``u1 || w1`` and ``u3 = u1 x u2``.

    Parameters
    ----------
    w1, w2 : Tensor
        Shape ``(..., 3)`` spatial vectors.
    eps_collinear : float
        If ``||w1 x w2||`` falls below this, both inputs are nudged by random
        normal directions scaled by ``eps_collinear`` (App. D.1).  Without a
        guard, collinear inputs make ``u2`` ill-defined.

    Returns
    -------
    tuple of Tensor
        Three tensors of shape ``(..., 3)``.
    """
    cross_norm = torch.linalg.cross(w1, w2, dim=-1).norm(dim=-1, keepdim=True)
    degenerate = cross_norm < eps_collinear
    if degenerate.any():
        w1 = torch.where(degenerate, w1 + eps_collinear * torch.randn_like(w1), w1)
        w2 = torch.where(degenerate, w2 + eps_collinear * torch.randn_like(w2), w2)

    u1 = w1 / (w1.norm(dim=-1, keepdim=True) + _EPS_NORM)
    w2_perp = w2 - u1 * (w2 * u1).sum(dim=-1, keepdim=True)
    u2 = w2_perp / (w2_perp.norm(dim=-1, keepdim=True) + _EPS_NORM)
    u3 = torch.linalg.cross(u1, u2, dim=-1)
    return u1, u2, u3


def frames_from_vectors(v0: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
    """Local reference frames via polar decomposition — Algorithm 1.

    ``L = R B`` where ``B = B(v0)`` boosts into the frame picked out by ``v0``
    and ``R`` is the pure rotation that aligns the (boosted) ``v1``, ``v2``
    with the local spatial axes.

    Parameters
    ----------
    v0, v1, v2 : Tensor
        Shape ``(..., 4)``, time-first, equivariantly predicted four-vectors.
        ``v0`` must be forward time-like.

    Returns
    -------
    Tensor
        Shape ``(..., 4, 4)`` satisfying ``L g L^T = g`` and transforming as
        ``L -> L Lambda^-1``.
    """
    boost = boost_matrix(v0)  # (..., 4, 4)

    # w_k = B v_k  for k = 1, 2
    w1 = torch.matmul(boost, v1.unsqueeze(-1)).squeeze(-1)
    w2 = torch.matmul(boost, v2.unsqueeze(-1)).squeeze(-1)

    u1, u2, u3 = gram_schmidt_3d(w1[..., 1:], w2[..., 1:])

    # R = blockdiag(1, R~) with R~ = (u1, u2, u3)^T (the triad as rows).
    r_spatial = torch.stack([u1, u2, u3], dim=-2)  # (..., 3, 3)
    zeros3 = r_spatial.new_zeros(r_spatial.shape[:-2] + (3,))
    ones1 = r_spatial.new_ones(r_spatial.shape[:-2] + (1,))

    top = torch.cat([ones1, zeros3], dim=-1).unsqueeze(-2)  # (..., 1, 4)
    bottom = torch.cat([zeros3.unsqueeze(-1), r_spatial], dim=-1)  # (..., 3, 4)
    rotation = torch.cat([top, bottom], dim=-2)  # (..., 4, 4)

    return torch.matmul(rotation, boost)


def invert_frames(frames: Tensor) -> Tensor:
    """Return ``L^-1 = g L^T g`` — exact inverse for any Lorentz transformation.

    Cheaper and far better conditioned than a generic matrix inverse.

    Parameters
    ----------
    frames : Tensor
        Shape ``(..., 4, 4)``.
    """
    g = frames.new_tensor(MINKOWSKI_SIGNATURE)
    # g L^T g  ==  (row scaling by g) . L^T . (column scaling by g)
    return g.unsqueeze(-1) * frames.transpose(-1, -2) * g.unsqueeze(-2)


# ---------------------------------------------------------------------------
# Equivariant frame prediction
# ---------------------------------------------------------------------------

class FramesNet(nn.Module):
    """Equivariantly predict one local Lorentz frame per particle.

    Implements Eq. (13): three four-vectors per particle are formed as a
    softmax-weighted combination of normalized pair sums, with weights produced
    by an MLP ``phi`` acting only on Lorentz scalars::

        v_{i,k} = sum_j softmax_j( phi_k(s_i, s_j, <p_i, p_j>) )
                  * (p_i + p_j) / (||p_i + p_j|| + eps),     k = 0, 1, 2

    followed by the set-level rescaling of Eq. (24).  Because the coefficients
    are scalars and the summands are four-vectors, ``v_{i,k}`` transforms in the
    vector representation, which is what makes Algorithm 1 produce a frame with
    the required ``L -> L Lambda^-1`` behaviour.

    The first layer of ``phi`` is evaluated in factorized form —
    ``W_i s_i + W_j s_j + W_d <p_i, p_j> + b`` is algebraically identical to
    applying one ``Linear`` to ``concat(s_i, s_j, <p_i, p_j>)`` but never
    materializes the ``(B, P, P, 2 S + 1)`` concatenation.

    Parameters
    ----------
    scalar_dim : int
        Width of the per-particle scalar features ``s_i``.  For jet tagging
        these are the (non-Lorentz-invariant) token features, which is one of
        the two symmetry-breaking channels of App. D.3 ("NIS").
    hidden_dim : int
        Hidden width of ``phi``.  The paper's default is 128; App. E shows 16
        is nearly as good and cheaper.  ``phi`` allocates a
        ``(B, P, P + n_ref, hidden_dim)`` activation, so this dominates the
        module's memory.
    symmetry_breaking : bool
        Append the three :data:`REFERENCE_PARTICLES` to the sender set
        ("RV" in App. D.3).  Set ``False`` for an exactly Lorentz-equivariant
        model — used by the equivariance tests.
    dropout : float
        Dropout inside ``phi``.  App. E recommends 0.2 in the low-data regime
        and 0 otherwise.
    min_mass : float
        Mass regulator ``m_eps``: input energies are raised to
        ``sqrt(m_eps^2 + E^2)`` so that numerically massless particles cannot
        produce a null ``v0`` and hence a divergent boost (App. D.1).  The
        paper uses ``5e-3`` for tagging.
    """

    def __init__(
        self,
        scalar_dim: int,
        hidden_dim: int = 64,
        symmetry_breaking: bool = True,
        dropout: float = 0.0,
        min_mass: float = 5e-3,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.hidden_dim = hidden_dim
        self.symmetry_breaking = symmetry_breaking
        self.min_mass = min_mass

        self.num_reference = len(REFERENCE_PARTICLES) if symmetry_breaking else 0
        # Reference particles are distinguished from real ones by a one-hot tag
        # appended to the scalar features (real particles get zeros).
        augmented_dim = scalar_dim + self.num_reference

        self.lin_receiver = nn.Linear(augmented_dim, hidden_dim)
        self.lin_sender = nn.Linear(augmented_dim, hidden_dim, bias=False)
        self.lin_pair = nn.Linear(1, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Three output channels: one weight field per predicted four-vector.
        self.lin_out = nn.Linear(hidden_dim, 3)

        if symmetry_breaking:
            self.register_buffer(
                "reference_momenta",
                torch.tensor(REFERENCE_PARTICLES, dtype=torch.float64),
                persistent=False,
            )
        else:
            self.reference_momenta = None

    def _regulate_mass(self, p: Tensor) -> Tensor:
        """Raise energies to ``sqrt(m_eps^2 + E^2)`` (App. D.1)."""
        if self.min_mass <= 0.0:
            return p
        energy = torch.sqrt(p[..., 0:1] ** 2 + self.min_mass**2)
        return torch.cat([energy, p[..., 1:]], dim=-1)

    def _build_sender_set(
        self, p: Tensor, scalars: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Augment scalars with reference tags and append reference particles.

        Returns ``(p_send, s_send, mask_send)`` with ``P + num_reference``
        entries along the particle axis.
        """
        batch, num_particles, _ = scalars.shape
        pad = scalars.new_zeros(batch, num_particles, self.num_reference)
        s_real = torch.cat([scalars, pad], dim=-1)

        if not self.symmetry_breaking:
            return p, s_real, mask

        ref_p = self.reference_momenta.to(dtype=p.dtype).expand(batch, -1, -1)
        # scalar block zero, reference tag one-hot
        ref_s = torch.cat(
            [
                scalars.new_zeros(batch, self.num_reference, self.scalar_dim),
                torch.eye(
                    self.num_reference, dtype=scalars.dtype, device=scalars.device
                ).expand(batch, -1, -1),
            ],
            dim=-1,
        )
        ref_mask = mask.new_ones(batch, self.num_reference)

        return (
            torch.cat([p, ref_p], dim=1),
            torch.cat([s_real, ref_s], dim=1),
            torch.cat([mask, ref_mask], dim=1),
        )

    def predict_vectors(
        self, p: Tensor, scalars: Tensor, mask: Tensor
    ) -> Tensor:
        """Predict the three four-vectors per particle (Eq. 13 + Eq. 24).

        Parameters
        ----------
        p : Tensor
            ``(B, P, 4)`` float64 four-momenta, time-first.
        scalars : Tensor
            ``(B, P, scalar_dim)`` per-particle scalar features.
        mask : Tensor
            ``(B, P)`` bool/float, ``True``/1 for real particles.

        Returns
        -------
        Tensor
            ``(B, P, 3, 4)`` float64 predicted four-vectors.
        """
        mask_bool = mask.bool()
        p = self._regulate_mass(p)
        p_send, s_send, mask_send = self._build_sender_set(p, scalars, mask_bool)

        # --- phi on Lorentz scalars, in the incoming (single) precision ------
        pair_dot = minkowski_dot(
            p.unsqueeze(2), p_send.unsqueeze(1), keepdim=True
        )  # (B, P, P + R, 1)

        compute_dtype = scalars.dtype
        hidden = (
            self.lin_receiver(s_send[:, : p.shape[1]]).unsqueeze(2)
            + self.lin_sender(s_send).unsqueeze(1)
            + self.lin_pair(pair_dot.to(compute_dtype))
        )  # (B, P, P + R, H)
        logits = self.lin_out(self.dropout(self.act(hidden)))  # (B, P, P + R, 3)

        # Padded senders must not receive attention weight.
        logits = logits.masked_fill(
            ~mask_send[:, None, :, None], torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=2).to(p.dtype)  # (B, P, P + R, 3)

        # --- equivariant combination of normalized pair sums ----------------
        pair_sum = p.unsqueeze(2) + p_send.unsqueeze(1)  # (B, P, P + R, 4)
        pair_sum = pair_sum / (minkowski_norm(pair_sum, keepdim=True) + _EPS_NORM)

        # (B, P, P+R, 3) x (B, P, P+R, 4) -> (B, P, 3, 4)
        vectors = torch.einsum("bijk,bijm->bikm", weights, pair_sum)

        # --- Eq. (24): rescale by the set-level RMS norm --------------------
        # A global rescale leaves the frame invariant (both B(v0) and
        # Gram-Schmidt are scale-invariant); it only keeps magnitudes in range.
        norms_sq = minkowski_dot(vectors, vectors).abs()  # (B, P, 3)
        norms_sq = norms_sq * mask_bool.unsqueeze(-1)
        scale = torch.sqrt(norms_sq.sum(dim=1, keepdim=True).clamp_min(_EPS_NORM))
        return vectors / (scale.unsqueeze(-1) + _EPS_NORM)

    def forward(
        self, p: Tensor, scalars: Tensor, mask: Tensor
    ) -> Tensor:
        """Predict one local frame per particle.

        Parameters
        ----------
        p : Tensor
            ``(B, P, 4)`` four-momenta, **time-first**.
        scalars : Tensor
            ``(B, P, scalar_dim)`` per-particle scalar features.
        mask : Tensor
            ``(B, P)`` real-particle mask.

        Returns
        -------
        Tensor
            ``(B, P, 4, 4)`` local frames, in the dtype of ``p``.
        """
        # Precision-critical section in float64 (App. D.1).
        p64 = p.to(torch.float64)
        vectors = self.predict_vectors(p64, scalars, mask)
        frames = frames_from_vectors(
            vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        )
        # Padded slots get the identity frame so downstream matmuls stay finite.
        identity = torch.eye(4, dtype=frames.dtype, device=frames.device)
        frames = torch.where(
            mask.bool()[:, :, None, None], frames, identity.expand_as(frames)
        )
        return frames
