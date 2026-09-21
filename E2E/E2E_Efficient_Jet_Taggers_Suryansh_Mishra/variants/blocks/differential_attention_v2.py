"""Differential Attention V2 block for Particle Transformer.

Provenance: lifted verbatim from the project's earlier hybrid-transformer
tree (``src/models/differential_attention_v2.py``) so the variants package
is self-contained. Adapts the Differential Transformer design,
arXiv:2410.05258 — Diff-Transformer-V2 variant (upstream repo:
https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2).

Implements Differential Transformer V2 (https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2).

Key differences from V1 (DifferentialAttentionBlock):

1. **Separate Q2 projection**: Q1 and Q2 use independent weight matrices
   instead of splitting a doubled projection. This allows matching baseline
   Transformer decoding speed and using FlashAttention without custom kernels.

2. **No per-head RMSNorm**: The per-head normalization after differential
   attention is removed (causes instability in large-scale pretraining).

3. **Token-specific, head-wise projected lambda**: Instead of globally shared
   scalar lambda parameters, lambda is projected from the input:
       lambda = sigmoid(x @ W_lambda)
   This eliminates the exponential reparameterization and simplifies init.

4. **No output rescaling**: The (1 - lambda_init) scaling factor is removed.

This is a drop-in replacement for `ParticleAttentionBlock` with the same
interface: forward(x, padding_mask, U) -> Tensor.

References
----------
- Differential Transformer V2: https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2
- Differential Transformer V1: arXiv:2410.05258
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .feedforward import Feedforward


class DifferentialAttentionV2Block(nn.Module):
    """
    Transformer block with Differential Attention V2.

    Each head computes two attention maps from separate Q1/Q2 projections
    (shared K), then subtracts the second weighted by a token-specific,
    head-wise lambda projected from the input.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding space.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    expansion_factor : int
        Expansion factor for the feedforward layers.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        super(DifferentialAttentionV2Block, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Layer norms (same as ParticleAttentionBlock)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # V2: Separate Q1 and Q2 projections (not split from doubled projection)
        self.q1_proj = nn.Linear(embed_dim, embed_dim)
        self.q2_proj = nn.Linear(embed_dim, embed_dim)  # independent Q2
        self.k_proj = nn.Linear(embed_dim, embed_dim)    # shared K
        self.v_proj = nn.Linear(embed_dim, embed_dim)    # shared V
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # V2: Token-specific, head-wise projected lambda
        # Projects from embed_dim -> num_heads, then sigmoid to get lambda in [0, 1]
        self.lambda_proj = nn.Linear(embed_dim, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.feedforward = Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(
        self, x: Tensor, padding_mask: Tensor, U: Optional[Tensor] = None
    ) -> Tensor:
        residual = x
        x_norm = self.layernorm1(x)
        B, N, C = x_norm.shape

        # --- Project Q1, Q2 (separate), K (shared), V (shared) ---
        Q1 = self.q1_proj(x_norm)  # (B, N, embed_dim)
        Q2 = self.q2_proj(x_norm)  # (B, N, embed_dim)
        K = self.k_proj(x_norm)    # (B, N, embed_dim)
        V = self.v_proj(x_norm)    # (B, N, embed_dim)

        # Reshape to per-head views: (B, N, H, head_dim) -> (B, H, N, head_dim)
        Q1 = Q1.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Q2 = Q2.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Compute attention scores
        attn1 = torch.matmul(Q1, K.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn2 = torch.matmul(Q2, K.transpose(-2, -1)) * scale

        # Add interaction bias U (B * num_heads, N, N) -> (B, num_heads, N, N)
        if U is not None:
            U_reshaped = U.view(B, self.num_heads, N, N)
            attn1 = attn1 + U_reshaped
            attn2 = attn2 + U_reshaped

        # Apply padding mask (True = padded / should be ignored)
        if padding_mask is not None:
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn1 = attn1.masked_fill(mask, -1e9)
            attn2 = attn2.masked_fill(mask, -1e9)

        # Softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.dropout(attn1)
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)

        # V2: Token-specific lambda projected from input
        # (B, N, embed_dim) -> (B, N, num_heads) -> sigmoid -> (B, H, N, 1)
        lambda_weight = torch.sigmoid(self.lambda_proj(x_norm))  # (B, N, num_heads)
        lambda_weight = lambda_weight.permute(0, 2, 1).unsqueeze(-1)  # (B, H, N, 1)

        # Differential combination (V2: NO (1 - lambda_init) rescaling)
        diff_attn = attn1 - lambda_weight * attn2  # (B, H, N, N)

        # Apply to V (V2: NO per-head RMSNorm)
        out = torch.matmul(diff_attn, V)  # (B, H, N, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, embed_dim)
        out = self.out_proj(out)

        # Post-attention residual + FFN (same as ParticleAttentionBlock)
        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        x_out = self.feedforward(x_out)

        return x_out
