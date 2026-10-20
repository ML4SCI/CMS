"""Interchangeable transformer sub-blocks for the ParT ablation.

Every module here is a drop-in replacement for one component of a ParT encoder
block, so an ablation arm is defined by swapping exactly one of them.

Attention blocks share the signature ``forward(x, padding_mask, U) -> Tensor``
with ``x (B, N, C)``, ``padding_mask (B, N)`` where ``True`` marks padding, and
``U`` the ParT pair-embedding bias:

===============================  ==================================================
:class:`SoftmaxAttentionBlock`   standard softmax attention; also the host for the
                                 MoE arm, since its FFN is injectable
:class:`SparsemaxAttentionBlock` sparsemax instead of softmax (hard zeros)
:class:`DifferentialAttentionBlock`    two attention maps, subtracted
:class:`DifferentialAttentionV2Block`  V2: independent Q1/Q2, projected lambda
:class:`URotaryAttentionBlock`   PairEmbed ``U`` as rotation (pairwise fused or RoPE apply), not a logit
===============================  ==================================================

Feed-forward networks map ``(B, N, C) -> (B, N, C)`` and include their own
residual connection:

=============================  ====================================================
:class:`Feedforward`           the dense pre-LN baseline
:class:`MoEFeedforward`        Mixture-of-Experts, matched to the dense baseline on
                               either parameters or FLOPs
=============================  ====================================================

The LLoCa arm's attention block lives in :mod:`variants.lloca` instead, because
it additionally requires per-particle local reference frames and so is not
interchangeable with these without that extra input.
"""

from .attention_block import SoftmaxAttentionBlock
from .differential_attention import DifferentialAttentionBlock
from .differential_attention_v2 import DifferentialAttentionV2Block
from .feedforward import Feedforward
from .moe_feedforward import GATE_MODES, HIDDEN_MODES, MoEFeedforward
from .sparsemax_attention import SparsemaxAttentionBlock, sparsemax
from .u_rotary_attention import URotaryAttentionBlock

__all__ = [
    # feed-forward
    "Feedforward",
    "MoEFeedforward",
    "HIDDEN_MODES",
    "GATE_MODES",
    # attention
    "SoftmaxAttentionBlock",
    "SparsemaxAttentionBlock",
    "DifferentialAttentionBlock",
    "DifferentialAttentionV2Block",
    "URotaryAttentionBlock",
    "sparsemax",
]
