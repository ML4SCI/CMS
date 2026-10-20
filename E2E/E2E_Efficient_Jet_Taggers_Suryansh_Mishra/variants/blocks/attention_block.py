"""Softmax attention block with an injectable feed-forward network.

NEW user-authored block (design decision DD5), derived from the user's
``SparsemaxAttentionBlock`` with exactly two changes:

1. ``F.softmax`` replaces ``sparsemax`` for the attention weights.
2. The feed-forward network is injectable (``ffn: nn.Module | None``),
   defaulting to the standard ``Feedforward``.

This block hosts the MoE ablation arm:
``SoftmaxAttentionBlock(ffn=MoEFeedforward(...))`` — so all four variant
arms share identical block wiring (pre-LN attention → post-LN → dropout →
residual → FFN) and differ in exactly one component.

Same interface as the other variant blocks:
``forward(x, padding_mask, U) -> Tensor``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .feedforward import Feedforward


class SoftmaxAttentionBlock(nn.Module):
    """Transformer block with standard softmax attention and pluggable FFN.

    Architecture is identical to ``SparsemaxAttentionBlock`` except:
    - ``F.softmax`` replaces ``sparsemax`` for attention weights
    - The FFN is injectable via ``ffn`` (defaults to ``Feedforward``)

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    dropout : float
    expansion_factor : int
        Used only when ``ffn`` is None to build the default ``Feedforward``.
    ffn : nn.Module | None
        Optional feed-forward module replacing the default ``Feedforward``.
        Must map ``(B, N, embed_dim) -> (B, N, embed_dim)``. The MoE arm
        passes ``MoEFeedforward(...)`` here.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        ffn: nn.Module | None = None,
    ):
        super(SoftmaxAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Manual Q/K/V projections (kept identical to SparsemaxAttentionBlock
        # so cross-variant comparisons stay single-delta)
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
        # Duck-typed: an FFN advertising ``accepts_padding_mask`` is handed the
        # mask so it can exclude padded tokens (the MoE router needs this to
        # keep its load-balancing statistics from being dominated by padding).
        # Checked by attribute rather than isinstance to avoid importing the MoE
        # module here.
        self._ffn_accepts_mask = bool(
            getattr(self.feedforward, "accepts_padding_mask", False)
        )

    def forward(
        self, x: Tensor, padding_mask: Tensor, U: Optional[Tensor] = None
    ) -> Tensor:
        residual = x
        x_norm = self.layernorm1(x)
        B, N, C = x_norm.shape

        # Q, K, V projections
        Q = self.q_proj(x_norm)  # (B, N, embed_dim)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        # Reshape to per-head views: (B, N, H, d) -> (B, H, N, d)
        Q = Q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Add interaction bias U (B * num_heads, N, N) -> (B, num_heads, N, N)
        if U is not None:
            U_reshaped = U.view(B, self.num_heads, N, N)
            attn_scores = attn_scores + U_reshaped

        # Apply padding mask BEFORE softmax
        # Set padded positions to large negative so softmax maps them to ~0
        if padding_mask is not None:
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        # SOFTMAX (the single attention-side delta vs SparsemaxAttentionBlock)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, N, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)
        out = self.out_proj(out)

        # Post-attention: same as SparsemaxAttentionBlock
        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        if self._ffn_accepts_mask:
            x_out = self.feedforward(x_out, padding_mask)
        else:
            x_out = self.feedforward(x_out)

        return x_out
