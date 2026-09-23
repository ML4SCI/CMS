"""
Eval-mode-only FusedPairMLP module.

Replaces PairEmbed with a single fused kernel launch that computes
pairwise Lorentz features AND the full Conv1d MLP in one pass, keeping
all MLP intermediates in GPU registers.
"""
import math

import torch
import torch.nn as nn
import triton

from ..triton.fused_pair_mlp_kernel import (
    _fused_pair_mlp_kernel,
    pack_pair_embed_weights,
)


class FusedPairMLP(nn.Module):
    """Eval-mode-only module that replaces PairEmbed with a single fused
    kernel launch (pairwise features + full Conv1d MLP)."""

    def __init__(self, embed_seq: nn.Sequential, eps: float = 1e-8,
                 remove_self_pair: bool = False):
        super().__init__()
        self.eps = eps
        self.remove_self_pair = remove_self_pair
        bn0_s, bn0_b, Ws, bs = pack_pair_embed_weights(embed_seq)
        self.register_buffer('bn0_s', bn0_s)
        self.register_buffer('bn0_b', bn0_b)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            self.register_buffer(f'W{i+1}', W)
            self.register_buffer(f'b{i+1}', b)
        self.out_dim = Ws[-1].shape[0]
        self.hid_dim = Ws[0].shape[0]

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (N, 4, P) particle 4-vectors
        Returns:
            (N, out_dim, P, P) attention bias
        """
        v = v.contiguous().float()
        N, _, P = v.shape
        out = torch.empty(N, self.out_dim, P, P, device=v.device, dtype=torch.float32)

        PP = P * P
        BLOCK = min(128, triton.next_power_of_2(PP))
        num_blocks = triton.cdiv(PP, BLOCK)
        grid = (N, num_blocks)

        _fused_pair_mlp_kernel[grid](
            v, out,
            self.bn0_s, self.bn0_b,
            self.W1, self.b1,
            self.W2, self.b2,
            self.W3, self.b3,
            self.W4, self.b4,
            P,
            v.stride(0), v.stride(1), v.stride(2),
            EPS=self.eps,
            PI=math.pi,
            TWO_PI=2.0 * math.pi,
            HID=self.hid_dim,
            OUT=self.out_dim,
            BLOCK=BLOCK,
            num_warps=4,
        )

        if self.remove_self_pair:
            idx = torch.arange(P, device=v.device)
            out[:, :, idx, idx] = 0.0

        return out
