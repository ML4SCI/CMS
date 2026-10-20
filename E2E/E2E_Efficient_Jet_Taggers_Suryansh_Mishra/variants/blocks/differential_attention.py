"""Differential attention block for Particle Transformer.

Provenance: lifted verbatim from the project's earlier hybrid-transformer
tree (``src/models/differential_attention.py``) so the variants package is
self-contained. Adapts the Differential Transformer design, arXiv:2410.05258
(upstream repo: https://github.com/microsoft/unilm/tree/master/Diff-Transformer).

Implements differential attention (arXiv:2410.05258): each head computes two
separate attention maps via doubled Q/K projections, then subtracts the second
weighted by a per-head learnable lambda parameter.

    diff_attn = softmax(Q1 @ K1^T / sqrt(d) + bias) - lambda * softmax(Q2 @ K2^T / sqrt(d) + bias)
    output = (diff_attn @ V) * (1 - lambda_init)

This is a drop-in replacement for `ParticleAttentionBlock` with the same
interface: forward(x, padding_mask, U) -> Tensor.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .feedforward import Feedforward


class DifferentialAttentionBlock(nn.Module):
    """
    Transformer block with Differential Attention.

    Replaces standard multi-head attention with a differential mechanism
    where each head computes two attention maps from doubled Q/K projections.
    The second map is subtracted from the first, weighted by a per-head
    learnable lambda parameter (sigmoid-gated to stay in [0, 1]).

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
    lambda_init : float
        Initial value for the per-head lambda parameters. Default 0.8.
    lambda_sigmoid : bool
        Whether to gate lambda through a sigmoid. Default True.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        lambda_init: float = 0.8,
        lambda_sigmoid: bool = True,
    ):
        super(DifferentialAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lambda_init = lambda_init
        self.lambda_sigmoid = lambda_sigmoid

        # Layer norms (same as ParticleAttentionBlock)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Q, K: doubled for two groups; V: single (shared)
        self.q_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.k_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.feedforward = Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        # Per-head learnable lambda
        initial_lambda = torch.full((num_heads,), lambda_init)
        self.lambdas = nn.Parameter(initial_lambda)

    def forward(
        self, x: Tensor, padding_mask: Tensor, U: Optional[Tensor] = None
    ) -> Tensor:
        residual = x
        x_norm = self.layernorm1(x)
        B, N, C = x_norm.shape

        # --- Project Q, K (doubled), V (shared) ---
        Q = self.q_proj(x_norm)  # (B, N, 2 * embed_dim)
        K = self.k_proj(x_norm)  # (B, N, 2 * embed_dim)
        V = self.v_proj(x_norm)  # (B, N, embed_dim)

        # Reshape to per-head views with doubled head_dim for Q/K
        # Q, K: (B, N, num_heads, 2 * head_dim) -> (B, num_heads, N, 2 * head_dim)
        Q = Q.view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        # V: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Split into two groups along the last dim
        # Each: (B, num_heads, N, head_dim)
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Compute attention scores
        attn1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scale

        # Add interaction bias U (B * num_heads, N, N) -> (B, num_heads, N, N)
        if U is not None:
            U_reshaped = U.view(B, self.num_heads, N, N)
            attn1 = attn1 + U_reshaped
            attn2 = attn2 + U_reshaped

        # Apply padding mask (True = padded / should be ignored)
        if padding_mask is not None:
            # padding_mask: (B, N) with 1.0 for padded positions
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn1 = attn1.masked_fill(mask, -1e9)
            attn2 = attn2.masked_fill(mask, -1e9)

        # Softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.dropout(attn1)
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)

        # Differential combination
        if self.lambda_sigmoid:
            lambda_weight = torch.sigmoid(self.lambdas)  # (num_heads,)
        else:
            lambda_weight = self.lambdas

        lambda_weight = lambda_weight.view(1, -1, 1, 1)  # (1, H, 1, 1)
        diff_attn = attn1 - lambda_weight * attn2
        diff_attn = diff_attn * (1.0 - self.lambda_init)  # rescale output

        # Apply to V
        out = torch.matmul(diff_attn, V)  # (B, H, N, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, embed_dim)
        out = self.out_proj(out)

        # Post-attention residual + FFN (same as ParticleAttentionBlock)
        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        x_out = self.feedforward(x_out)

        return x_out
