"""Pairwise rotary attention driven by PairEmbed ``U``, not an additive logit.

K6 asked whether extras from pairwise products (RoPE + Minkowski) can
*replace* dense ``U``.  They cannot (not-U).  This block asks the inverse:
keep PairEmbed, but consume ``U`` as a **rotation** of content Q/K instead
of adding it to the logits.

Two application styles:

**Pairwise fused** (``rope_apply=False``, the in-flight ``urot`` job):
``θ_ij = π tanh(α U_ij)`` enters the score as

    (Q_i · K_j) cos θ_ij + (Q_i ⋆ K_j) sin θ_ij

That ``θ`` is pairwise, so Q cannot be rotated once then matmul'd.

**RoPE apply** (``rope_apply=True``, ``urot_rope``): pool ``U`` to a
per-token phase ``ψ_i``, rotate ``Q_i`` by ``R(ψ_i)`` and ``K_j`` by
``R(ψ_j)`` with LLaMA geometric frequencies, then ordinary ``Q Kᵀ``.
Relative structure is ``ψ_j − ψ_i`` (the RoPE identity).  This is a
rank-1 factorization of ``U``; T0.1 p90=20 is why it may lose pairwise
information — that is the screen.

``θ = 0`` / ``ψ = 0`` recovers content attention.  ``U`` is never added
as a scalar.  This is not K2 and not N8: PairEmbed stays on.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from .feedforward import Feedforward

__all__ = [
    "URotaryAttentionBlock",
    "ANGLE_SCALE_INIT",
    "ROPE_BASE",
    "ROPE_PHASE_SCALE_INIT",
    "apply_rope",
    "pairwise_rotary_logits",
    "pairwise_rotary_logits_per_plane",
    "rotate_half",
    "u_to_angles",
    "u_to_token_phases",
]

#: ``θ = π tanh(α U)`` for the pairwise fused path.  α = 0.1 keeps early
#: training near content-only until PairEmbed and α grow.
ANGLE_SCALE_INIT = 0.1

#: LLaMA RoPE base.  ``ω_p = base^{-2p/d}``.
ROPE_BASE = 10000.0

#: ``ψ_i = α_h · mean_j U_ij`` for the RoPE path.  α = 1 so pooled U is
#: used as a real-valued position (RoPE positions are unbounded; tanh
#: would squash them into a range where high-frequency planes die).
ROPE_PHASE_SCALE_INIT = 1.0


def u_to_angles(U: Tensor, angle_scale: Tensor) -> Tensor:
    """Map pair-bias ``U`` to bounded rotation angles.

    Parameters
    ----------
    U : Tensor
        ``(B, H, N, N)``.
    angle_scale : Tensor
        Per-head ``α``, shape ``(H,)``.

    Returns
    -------
    Tensor
        ``θ ∈ (−π, π)``, same shape as ``U``.
    """
    if angle_scale.ndim != 1:
        raise ValueError(
            f"u_to_angles expects α of shape (H,), got {tuple(angle_scale.shape)}"
        )
    scale = angle_scale.to(dtype=U.dtype).view(1, -1, 1, 1)
    return math.pi * torch.tanh(scale * U)


def u_to_token_phases(
    U: Tensor,
    angle_scale: Tensor,
    padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """Pool pairwise ``U`` to a per-token RoPE phase.

    ``ψ_i = α_h · mean_{j valid} U_ij``.  Padded keys are excluded.
    Shape ``(B, H, N)``.  Unbounded: this is a position, not an angle.
    """
    if angle_scale.ndim != 1:
        raise ValueError(
            f"u_to_token_phases expects α of shape (H,), got {tuple(angle_scale.shape)}"
        )
    if padding_mask is None:
        pooled = U.mean(dim=-1)
    else:
        key_valid = ~padding_mask.bool()
        mask = key_valid[:, None, None, :]
        masked = U.masked_fill(~mask, 0.0)
        denom = mask.to(dtype=U.dtype).sum(dim=-1).clamp(min=1.0)
        pooled = masked.sum(dim=-1) / denom
    return angle_scale.to(dtype=U.dtype).view(1, -1, 1) * pooled


def rotate_half(x: Tensor) -> Tensor:
    """LLaMA ``rotate_half``: ``[..., x1, x2] → [..., −x2, x1]``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    query: Tensor,
    key: Tensor,
    psi: Tensor,
    inv_freq: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Rotate Q and K like LLaMA RoPE, then the caller does ``Q Kᵀ``.

    ``query`` / ``key`` are ``(B, H, N, d)``.  ``psi`` is the per-token
    phase ``(B, H, N)``.  ``inv_freq`` is ``(d/2,)`` with
    ``ω_p = base^{-2p/d}``.
    """
    freqs = psi.unsqueeze(-1) * inv_freq.to(dtype=psi.dtype)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key


def pairwise_rotary_logits(
    query: Tensor,
    key: Tensor,
    theta: Tensor,
    scale: float,
) -> Tensor:
    """Content Q/K mixed by a pairwise rotation ``R(θ_ij)``.

    ``query`` / ``key`` are ``(B, H, N, d)`` with even ``d``.  ``theta`` is
    ``(B, H, N, N)`` and is **shared across all 2-planes**.  When ``theta``
    is 0 this is exactly ``Q Kᵀ / √d``.
    """
    head_dim = query.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"pairwise rotary needs even head_dim, got {head_dim}")
    dot = torch.matmul(query, key.transpose(-2, -1))
    q_even = query[..., 0::2]
    q_odd = query[..., 1::2]
    k_even = key[..., 0::2]
    k_odd = key[..., 1::2]
    cross = torch.matmul(q_even, k_odd.transpose(-2, -1)) - torch.matmul(
        q_odd, k_even.transpose(-2, -1)
    )
    return (dot * theta.cos() + cross * theta.sin()) * scale


def pairwise_rotary_logits_per_plane(
    query: Tensor,
    key: Tensor,
    U: Tensor,
    angle_scale: Tensor,
    scale: float,
    checkpoint_planes: bool = True,
) -> Tensor:
    """Independent 2-plane rotations; ``α`` is ``(H, P)`` with ``P = d/2``.

    Planes are accumulated in a Python loop so we never materialize a
    ``(B, H, P, N, N)`` tensor.  That bounds **forward** memory but not
    backward: a Python loop does not bound the autograd tape, and each plane
    retains ~5 ``(B, H, N, N)`` intermediates for its own backward — the
    ``tanh`` output inside :func:`u_to_angles`, ``θ.cos()``, ``θ.sin()``,
    ``dot`` and ``cross``.  Retention therefore grew *linearly in*
    ``n_planes``: measured 111× ``(B, H, N, N)`` at ``head_dim=32`` against 21×
    on the shared-θ path, which at the shipped ``urot_mf`` config (16 planes,
    8 layers, ``batch_size=256``) is tens of GB of activations and OOMs an
    80 GB A100.

    ``checkpoint_planes`` recomputes each plane during backward instead of
    storing it, taking retention back to ``O(1)`` in ``n_planes`` — ``U`` is
    shared across planes, so the checkpoints hold one reference to it rather
    than a copy each.  The cost is one extra forward per plane in backward.
    ``θ`` is a deterministic ``tanh`` of ``U`` with no dropout or RNG, so the
    replay is bit-identical and the result does not depend on this flag.

    When every ``α_{h,p}`` is equal, this matches :func:`pairwise_rotary_logits`.
    """
    head_dim = query.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"pairwise rotary needs even head_dim, got {head_dim}")
    n_planes = head_dim // 2
    if angle_scale.shape[-1] != n_planes:
        raise ValueError(
            f"per-plane α has {angle_scale.shape[-1]} frequencies, "
            f"head_dim={head_dim} expects {n_planes}"
        )

    def one_plane(q: Tensor, k: Tensor, u: Tensor, alpha: Tensor) -> Tensor:
        return pairwise_rotary_logits(q, k, u_to_angles(u, alpha), scale=1.0)

    # Nothing to recompute when no graph is being built (inference / no_grad),
    # and checkpointing there would only add a redundant forward.
    use_checkpoint = (
        checkpoint_planes
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in (query, key, U, angle_scale))
    )

    logits = None
    for p in range(n_planes):
        q_p = query[..., 2 * p : 2 * p + 2]
        k_p = key[..., 2 * p : 2 * p + 2]
        alpha_p = angle_scale[:, p]
        if use_checkpoint:
            plane = torch.utils.checkpoint.checkpoint(
                one_plane, q_p, k_p, U, alpha_p, use_reentrant=False
            )
        else:
            plane = one_plane(q_p, k_p, U, alpha_p)
        logits = plane if logits is None else logits + plane
    return logits * scale


class URotaryAttentionBlock(nn.Module):
    """Softmax attention that reads PairEmbed ``U`` as a rotation generator.

    Same pre-LN residual wiring as :class:`SoftmaxAttentionBlock`.  The
    single delta is how ``U`` enters the logits.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        ffn: nn.Module | None = None,
        angle_scale_init: float | None = None,
        per_plane: bool = False,
        rope_apply: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"U-rotary needs even head_dim so 2-planes tile; got {head_dim}"
            )
        if rope_apply and per_plane:
            raise ValueError(
                "rope_apply and per_plane cannot both be True: RoPE apply "
                "uses geometric frequencies on a per-token phase, not a "
                "pairwise θ per plane"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_planes = head_dim // 2
        self.per_plane = bool(per_plane)
        self.rope_apply = bool(rope_apply)
        #: Recompute per-plane logits in backward rather than retaining them.
        #: Without it, backward activations scale with ``n_planes`` and the
        #: ``urot_mf`` config OOMs an 80 GB A100 — see
        #: :func:`pairwise_rotary_logits_per_plane`.  Plain attribute so it can
        #: be flipped for a memory/speed experiment without a config change.
        self.checkpoint_planes = True

        if angle_scale_init is None:
            angle_scale_init = (
                ROPE_PHASE_SCALE_INIT if self.rope_apply else ANGLE_SCALE_INIT
            )

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.feedforward = ffn if ffn is not None else Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self._ffn_accepts_mask = bool(
            getattr(self.feedforward, "accepts_padding_mask", False)
        )
        if self.per_plane:
            # Geometric frequencies, same idea as N8 rotary extras / RoPE:
            # plane 0 is slow (α ≈ init), higher planes faster (α / 2^p).
            freqs = 1.0 / (2.0 ** torch.arange(self.n_planes, dtype=torch.float32))
            self.angle_scale = nn.Parameter(
                (float(angle_scale_init) * freqs).repeat(num_heads, 1)
            )
        else:
            self.angle_scale = nn.Parameter(
                torch.full((num_heads,), float(angle_scale_init))
            )
        if self.rope_apply:
            inv_freq = 1.0 / (
                ROPE_BASE
                ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
                )
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, int, int]:
        x_norm = self.layernorm1(x)
        batch, seq_len, _ = x_norm.shape
        query = self.q_proj(x_norm).view(
            batch, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = self.k_proj(x_norm).view(
            batch, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = self.v_proj(x_norm).view(
            batch, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        return query, key, value, batch, seq_len

    def _scores(
        self,
        query: Tensor,
        key: Tensor,
        U: Optional[Tensor],
        batch: int,
        seq_len: int,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        if U is None:
            return torch.matmul(query, key.transpose(-2, -1)) * scale
        U_reshaped = U.view(batch, self.num_heads, seq_len, seq_len)
        if self.rope_apply:
            psi = u_to_token_phases(U_reshaped, self.angle_scale, padding_mask)
            query, key = apply_rope(query, key, psi, self.rope_inv_freq)
            return torch.matmul(query, key.transpose(-2, -1)) * scale
        if self.per_plane:
            return pairwise_rotary_logits_per_plane(
                query,
                key,
                U_reshaped,
                self.angle_scale,
                scale,
                checkpoint_planes=self.checkpoint_planes,
            )
        theta = u_to_angles(U_reshaped, self.angle_scale)
        return pairwise_rotary_logits(query, key, theta, scale)

    def attention_logits(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
        U: Optional[Tensor],
    ) -> Tensor:
        """Pre-softmax logits ``(B, H, N, N)``.  Does not consume residual."""
        query, key, _, batch, seq_len = self._qkv(x)
        logits = self._scores(query, key, U, batch, seq_len, padding_mask)
        if padding_mask is not None:
            fill = padding_mask.bool().unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(fill, -1e9)
        return logits

    def forward(
        self, x: Tensor, padding_mask: Optional[Tensor], U: Optional[Tensor] = None
    ) -> Tensor:
        residual = x
        query, key, value, batch, seq_len = self._qkv(x)
        channels = x.shape[-1]
        attn_scores = self._scores(
            query, key, U, batch, seq_len, padding_mask
        )

        if padding_mask is not None:
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, value)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, channels)
        out = self.out_proj(out)

        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        if self._ffn_accepts_mask:
            x_out = self.feedforward(x_out, padding_mask)
        else:
            x_out = self.feedforward(x_out)
        return x_out
