"""Factorized attention: content SDPA plus extra Minkowski / rotary logits.

Structurally this is
:class:`~variants.blocks.attention_block.SoftmaxAttentionBlock` with one
delta: pairwise structure arrives as extra Q/K channels rather than as a
dense ``PairEmbed`` bias ``U``.  The content scale stays ``1/sqrt(d_head)``
— extras are a *separate* logit term, so a comparison against baseline /
``baseline_nopair`` is not confounded by changing the content temperature.

The ParT pair-embedding ``attn_mask`` is ignored (N8 runs with
``pair_embed_dims=None``).  Padding uses the same ``-1e9`` fill as the
other variant blocks.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..blocks.feedforward import Feedforward
from .features import (
    QUADRATIC_DIM,
    ROTARY_ANGLE_CLAMP,
    azimuth,
    minkowski_qk_extras,
    quadratic_monomials,
    relative_rapidity,
    rotary_qk_extras,
)

__all__ = ["FactorizedAttentionBlock", "inv_softplus"]


def inv_softplus(y: float) -> float:
    """Inverse of ``softplus``, for initializing a positive parameter at ``y``."""
    if y <= 0.0:
        raise ValueError(f"softplus inverse requires y > 0, got {y}")
    return math.log(math.expm1(y))


#: Minkowski extras are ``d_ij / (E_i E_j) ∈ [0, O(1)]`` (collimated ~ 10⁻²,
#: two-prong cores ~ 0.5).  Init ``λ = 1`` so the term starts near content scale.
_LAMBDA_INIT = 1.0
_ROTARY_A_INIT = 0.1
_ROTARY_LAMBDA_INIT = 0.1


class FactorizedAttentionBlock(nn.Module):
    """Transformer block with content attention plus factorized pair extras.

    Parameters
    ----------
    embed_dim, num_heads, dropout, expansion_factor
        Same as :class:`~variants.blocks.attention_block.SoftmaxAttentionBlock`.
    minkowski : bool
        Rank-4 Minkowski extras (K1 / the K6 Minkowski half).
    rotary : bool
        Tensor-product rotary extras (the K6 RoPE half).
    rotary_pairs : int
        Number of 4-dim rotary blocks (K6 uses 4 → 16 extra dims).
    degree : int
        ``1`` = Minkowski four-momenta only.  ``2`` additionally appends
        the ten quadratic monomials (``r = 14`` with Minkowski on).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        minkowski: bool = True,
        rotary: bool = True,
        rotary_pairs: int = 4,
        degree: int = 1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if degree not in (1, 2):
            raise ValueError(f"n8_degree must be 1 or 2, got {degree}")
        if rotary and rotary_pairs < 1:
            raise ValueError(f"rotary_pairs must be >= 1, got {rotary_pairs}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.minkowski = bool(minkowski)
        self.rotary = bool(rotary)
        self.rotary_pairs = int(rotary_pairs) if rotary else 0
        self.degree = int(degree)
        self.quadratic = degree >= 2

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.feedforward = Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        init = inv_softplus(_LAMBDA_INIT)
        if self.minkowski:
            self.lambda_raw = nn.Parameter(torch.full((num_heads,), init))
        else:
            self.register_parameter("lambda_raw", None)

        if self.quadratic:
            self.quad_raw = nn.Parameter(
                torch.full((num_heads, QUADRATIC_DIM), init)
            )
        else:
            self.register_parameter("quad_raw", None)

        if self.rotary:
            freqs = 1.0 / (2.0 ** torch.arange(self.rotary_pairs, dtype=torch.float32))
            self.rotary_a = nn.Parameter(
                (_ROTARY_A_INIT * freqs).repeat(num_heads, 1)
            )
            self.rotary_lambda_raw = nn.Parameter(
                torch.full(
                    (num_heads, self.rotary_pairs),
                    inv_softplus(_ROTARY_LAMBDA_INIT),
                )
            )
        else:
            self.register_parameter("rotary_a", None)
            self.register_parameter("rotary_lambda_raw", None)

        self._v: Optional[Tensor] = None
        self._mask: Optional[Tensor] = None

    def extra_dim(self) -> int:
        """Width of the concatenated extra Q/K channels (not content ``head_dim``)."""
        width = 0
        if self.minkowski:
            width += 4
        if self.quadratic:
            width += QUADRATIC_DIM
        if self.rotary:
            width += 4 * self.rotary_pairs
        return width

    def set_momenta(self, v: Tensor, mask: Optional[Tensor] = None) -> None:
        """Stash post-trim four-momenta for the next :meth:`forward`.

        Weaver blocks have no extra-tensor slot, so momenta are handed
        out-of-band (same pattern as LLoCa frames).  They are cleared on
        use so a missing call fails loudly instead of reusing a stale jet.
        """
        self._v = v
        self._mask = mask

    def _take_momenta(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if self._v is None:
            raise RuntimeError(
                "FactorizedAttentionBlock.forward() called without momenta; "
                "call set_momenta(v, mask) before every forward pass "
                "(N8ParT does this for all blocks)"
            )
        v, mask = self._v, self._mask
        if v.shape[-1] != x.shape[1]:
            raise RuntimeError(
                f"momenta/token length mismatch: v has {v.shape[-1]} "
                f"particles but x has {x.shape[1]} tokens — momenta must be "
                "set after sequence trimming"
            )
        return v, mask

    def minkowski_scale(self) -> Tensor:
        """Positive per-head ``λ_h``, shape ``(H,)``."""
        if self.lambda_raw is None:
            raise RuntimeError("minkowski extras are disabled")
        return F.softplus(self.lambda_raw)

    def rotary_scale(self) -> Tensor:
        """Positive per-head per-block ``λ_rotary``, shape ``(H, K)``."""
        if self.rotary_lambda_raw is None:
            raise RuntimeError("rotary extras are disabled")
        return F.softplus(self.rotary_lambda_raw)

    def _extra_qk(self, v: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """Build extra Q/K of shape ``(B, H, P, extra_dim)``."""
        chunks_q = []
        chunks_k = []
        heads = self.num_heads

        if self.minkowski:
            q_m, k_m = minkowski_qk_extras(v, mask)
            scale = self.minkowski_scale().sqrt().view(1, -1, 1, 1)
            chunks_q.append(q_m.unsqueeze(1) * scale)
            chunks_k.append(k_m.unsqueeze(1) * scale)

        if self.quadratic:
            phi = quadratic_monomials(v)  # (B, P, 10)
            scale = F.softplus(self.quad_raw).sqrt().view(1, heads, 1, QUADRATIC_DIM)
            qk = phi.unsqueeze(1) * scale
            chunks_q.append(qk)
            chunks_k.append(qk)

        if self.rotary:
            phi = azimuth(v)
            u = relative_rapidity(v, mask)
            if mask is not None:
                real = mask.to(dtype=phi.dtype).reshape(phi.shape[0], phi.shape[-1])
                phi = phi * real
            rot = rotary_qk_extras(
                phi,
                u,
                self.rotary_a,
                self.rotary_pairs,
                clamp=ROTARY_ANGLE_CLAMP,
            )
            lam = self.rotary_scale().sqrt().unsqueeze(0).unsqueeze(2)
            rot = rot * lam.repeat_interleave(4, dim=-1)
            chunks_q.append(rot)
            chunks_k.append(rot)

        return torch.cat(chunks_q, dim=-1), torch.cat(chunks_k, dim=-1)

    def _content_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Project ``x`` to ``(B, H, N, d)`` queries, keys and values."""
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
        return query, key, value

    def _scores(
        self,
        query: Tensor,
        key: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Content ``QK^T / sqrt(d_head)`` plus extra logit terms."""
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        if self.extra_dim() > 0:
            extra_q, extra_k = self._extra_qk(v.to(dtype=query.dtype), mask)
            logits = logits + torch.matmul(extra_q, extra_k.transpose(-2, -1))
        if padding_mask is not None:
            fill = padding_mask.bool().unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(fill, -1e9)
        return logits

    def attention_logits(
        self, x: Tensor, padding_mask: Optional[Tensor]
    ) -> Tensor:
        """Pre-softmax logits ``(B, H, N, N)``.  Consumes stashed momenta."""
        v, mask = self._take_momenta(x)
        self._v = self._mask = None
        query, key, _ = self._content_qkv(x)
        return self._scores(query, key, v, mask, padding_mask)

    def forward(
        self, x: Tensor, padding_mask: Optional[Tensor], U: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``(B, N, embed_dim)``.
        padding_mask : Tensor or None
            ``(B, N)``, ``True`` marks padded keys.
        U : Tensor or None
            Ignored.  N8 does not use a dense pair bias.
        """
        del U  # pair_embed is off; a leftover attn_mask must not leak back in
        v, mask = self._take_momenta(x)
        self._v = self._mask = None

        residual = x
        batch, seq_len, channels = x.shape
        query, key, value = self._content_qkv(x)
        logits = self._scores(query, key, v, mask, padding_mask)
        weights = self.dropout(F.softmax(logits, dim=-1))

        out = torch.matmul(weights, value)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, channels)
        out = self.out_proj(out)

        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        x_out = self.feedforward(x_out)
        return x_out
