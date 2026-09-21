"""
torch.autograd.Function wrapper for the fused attention kernel.

Provides ``fused_attention_with_bias`` -- the public entry point used by
OptimizedBlock to replace nn.MultiheadAttention in the self-attention path.

Import safety (Req 2.6, 8.3): triton and the Triton kernels are imported
inside the ``Function`` dispatch (forward/backward), not at module scope,
so importing this module on a CPU-only machine works. Actually running the
CUDA path imports triton on first use exactly as before.
"""
from __future__ import annotations

import torch


class _FusedAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, bias, pad_mask, scale, num_heads):
        import ml4sci_26.part_kernels.triton as triton

        from ..triton.attention_kernel import _fused_attn_fwd

        NH, P, D = Q.shape
        BLOCK_M = triton.next_power_of_2(min(P, 64))
        BLOCK_N = triton.next_power_of_2(min(P, 128))
        D_pow2 = triton.next_power_of_2(D)

        Out = torch.empty_like(Q)
        LSE = torch.empty(NH, P, device=Q.device, dtype=torch.float32)

        has_bias = bias is not None
        has_pad = pad_mask is not None

        if not has_bias:
            bias = torch.empty(0, device=Q.device)
        if not has_pad:
            pad_mask = torch.empty(0, device=Q.device, dtype=torch.bool)

        grid = (NH, triton.cdiv(P, BLOCK_M))
        _fused_attn_fwd[grid](
            Q, K, V, bias, pad_mask, Out, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            bias.stride(0) if has_bias else 0,
            bias.stride(1) if has_bias else 0,
            bias.stride(2) if has_bias else 0,
            pad_mask.stride(0) if has_pad else 0,
            pad_mask.stride(1) if has_pad else 0,
            Out.stride(0), Out.stride(1), Out.stride(2),
            LSE.stride(0), LSE.stride(1),
            sm_scale=scale,
            P=P, D=D_pow2,
            HAS_BIAS=has_bias, HAS_PAD=has_pad,
            NUM_HEADS=num_heads,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(Q, K, V,
                              bias if has_bias else torch.empty(0, device=Q.device),
                              pad_mask if has_pad else torch.empty(0, device=Q.device, dtype=torch.bool),
                              Out, LSE)
        ctx.scale = scale
        ctx.num_heads = num_heads
        ctx.has_bias = has_bias
        ctx.has_pad = has_pad
        ctx.P = P
        ctx.D = D
        ctx.D_pow2 = D_pow2
        return Out

    @staticmethod
    def backward(ctx, dOut):
        import ml4sci_26.part_kernels.triton as triton

        from ..triton.attention_kernel import _fused_attn_bwd

        Q, K, V, bias, pad_mask, Out, LSE = ctx.saved_tensors
        NH, P, D = Q.shape

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dBias = torch.zeros_like(bias) if ctx.has_bias and bias.requires_grad else torch.empty(0, device=Q.device)
        need_dbias = ctx.has_bias and bias.requires_grad

        BLOCK_M = triton.next_power_of_2(min(P, 64))
        BLOCK_N = triton.next_power_of_2(min(P, 128))

        grid = (NH, triton.cdiv(P, BLOCK_M))
        _fused_attn_bwd[grid](
            Q, K, V, bias, pad_mask, Out, LSE, dOut.contiguous(),
            dQ, dK, dV, dBias,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            bias.stride(0) if ctx.has_bias else 0,
            bias.stride(1) if ctx.has_bias else 0,
            bias.stride(2) if ctx.has_bias else 0,
            pad_mask.stride(0) if ctx.has_pad else 0,
            pad_mask.stride(1) if ctx.has_pad else 0,
            Out.stride(0), Out.stride(1), Out.stride(2),
            LSE.stride(0), LSE.stride(1),
            sm_scale=ctx.scale,
            P=P, D=ctx.D_pow2,
            HAS_BIAS=ctx.has_bias, HAS_PAD=ctx.has_pad,
            NUM_HEADS=ctx.num_heads,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NEED_DBIAS=need_dbias,
        )

        return dQ, dK, dV, dBias if need_dbias else None, None, None, None


def fused_attention_with_bias(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor | None,
    pad_mask: torch.Tensor | None,
    scale: float,
    num_heads: int,
) -> torch.Tensor:
    """Fused attention with additive pair bias and padding mask.

    Args:
        Q, K, V : (N*H, P, D)  pre-projected per-head tensors
        bias    : (N*H, P, P)  additive attention bias, or None
        pad_mask: (N, P)       bool, True=padded, or None
        scale   : 1/sqrt(head_dim)
        num_heads: H

    Returns:
        (N*H, P, D)  attention output
    """
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return _FusedAttnFunc.apply(Q, K, V, bias, pad_mask, scale, num_heads)
