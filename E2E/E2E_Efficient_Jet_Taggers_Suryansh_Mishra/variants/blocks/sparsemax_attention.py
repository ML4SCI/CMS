"""Sparsemax attention block for Particle Transformer.

Provenance: user-authored; lifted verbatim from the project's earlier
hybrid-transformer tree (``src/models/sparsemax_attention.py``) so the
variants package is self-contained.

Replaces ``softmax`` in standard multi-head attention with ``sparsemax``
(Martins & Astudillo, 2016).  Sparsemax projects attention scores onto the
probability simplex via Euclidean projection, producing **exact zeros** —
structurally sparse attention without arbitrary Top-K cutoffs.

This is a new block class (does NOT modify ParticleAttentionBlock).
Same interface: ``forward(x, padding_mask, U) -> Tensor``.

Properties of sparsemax vs softmax:
- Output sums to 1 ✓
- Contains exact zeros ✓ (unlike softmax which only asymptotically → 0)
- Well-defined gradients (piecewise linear Jacobian) ✓
- No overflow risk from large scores (no exp) ✓
- Same parameter count as standard attention ✓

References
----------
- From Softmax to Sparsemax (Martins & Astudillo, ICML 2016)
- Adaptively Sparse Transformers (Correia et al., EMNLP 2019)
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .feedforward import Feedforward


# ---------------------------------------------------------------------------
# Sparsemax function
# ---------------------------------------------------------------------------

def sparsemax(scores: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax: Euclidean projection onto the probability simplex.

    Replaces ``F.softmax(scores, dim=dim)`` with a sparse alternative.
    Output sums to 1, contains exact zeros, and gradients are well-defined.

    Parameters
    ----------
    scores : Tensor
        Raw attention logits, any shape.
    dim : int
        Dimension along which to apply sparsemax.  Default -1.

    Returns
    -------
    Tensor
        Same shape as ``scores``, with values in [0, 1] summing to 1 along ``dim``.
    """
    # Move target dim to last position for convenience
    scores = scores.transpose(dim, -1)
    original_shape = scores.shape
    scores = scores.contiguous().view(-1, scores.size(-1))  # (batch, K)

    K = scores.size(-1)

    # Sort descending
    sorted_scores, _ = torch.sort(scores, descending=True, dim=-1)

    # Cumulative sum
    cumsum = torch.cumsum(sorted_scores, dim=-1)  # (batch, K)

    # Find support: k_z = max{k : 1 + k * z_k > cumsum_k}
    arange = torch.arange(1, K + 1, device=scores.device, dtype=scores.dtype)
    condition = 1.0 + arange * sorted_scores > cumsum  # (batch, K)

    # k_z: number of elements in the support (at least 1)
    k_z = condition.sum(dim=-1, keepdim=True).clamp(min=1)  # (batch, 1)

    # Threshold tau
    # tau = (cumsum at k_z - 1) / k_z
    # Gather the cumsum value at position k_z - 1
    tau_idx = (k_z - 1).long()  # (batch, 1)
    tau_cumsum = cumsum.gather(-1, tau_idx)  # (batch, 1)
    tau = (tau_cumsum - 1.0) / k_z.float()  # (batch, 1)

    # Project: max(0, scores - tau)
    output = torch.clamp(scores - tau, min=0.0)

    # Reshape back
    output = output.view(original_shape)
    output = output.transpose(dim, -1)

    return output


# ---------------------------------------------------------------------------
# Sparsemax Attention Block
# ---------------------------------------------------------------------------

class SparsemaxAttentionBlock(nn.Module):
    """Transformer block using sparsemax instead of softmax in attention.

    Architecture is identical to ``ParticleAttentionBlock`` except:
    - Manual Q/K/V projections (not ``nn.MultiheadAttention``, which hardcodes softmax)
    - ``sparsemax`` replaces ``F.softmax`` for attention weights

    Zero additional parameters vs baseline.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    dropout : float
    expansion_factor : int
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        super(SparsemaxAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Manual Q/K/V projections (nn.MultiheadAttention hardcodes softmax)
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

        # Apply padding mask BEFORE sparsemax
        # Set padded positions to large negative so sparsemax maps them to 0
        if padding_mask is not None:
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        # SPARSEMAX instead of softmax
        attn_weights = sparsemax(attn_scores, dim=-1)  # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, N, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)
        out = self.out_proj(out)

        # Post-attention: same as ParticleAttentionBlock
        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        x_out = self.feedforward(x_out)

        return x_out
