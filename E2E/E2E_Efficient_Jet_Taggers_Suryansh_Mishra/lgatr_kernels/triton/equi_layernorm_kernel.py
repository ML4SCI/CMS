"""Fused Triton kernel for equivariant LayerNorm in Cl(1,3).

Fuses: grade-wise metric product -> abs -> sum -> mean -> clamp -> rsqrt -> scale
into a single kernel. Grade metric signs are compile-time constants.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _equi_layernorm_fwd_kernel(
    X_ptr, OUT_ptr, N_items, C: tl.constexpr,
    gain, epsilon,
    stride_n, stride_c, stride_16,
    stride_on, stride_oc, stride_o16,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N_items:
        return
    total_norm = 0.0
    for c in range(C):
        base = pid * stride_n + c * stride_c
        x0=tl.load(X_ptr+base+0*stride_16).to(tl.float32);x1=tl.load(X_ptr+base+1*stride_16).to(tl.float32)
        x2=tl.load(X_ptr+base+2*stride_16).to(tl.float32);x3=tl.load(X_ptr+base+3*stride_16).to(tl.float32)
        x4=tl.load(X_ptr+base+4*stride_16).to(tl.float32);x5=tl.load(X_ptr+base+5*stride_16).to(tl.float32)
        x6=tl.load(X_ptr+base+6*stride_16).to(tl.float32);x7=tl.load(X_ptr+base+7*stride_16).to(tl.float32)
        x8=tl.load(X_ptr+base+8*stride_16).to(tl.float32);x9=tl.load(X_ptr+base+9*stride_16).to(tl.float32)
        x10=tl.load(X_ptr+base+10*stride_16).to(tl.float32);x11=tl.load(X_ptr+base+11*stride_16).to(tl.float32)
        x12=tl.load(X_ptr+base+12*stride_16).to(tl.float32);x13=tl.load(X_ptr+base+13*stride_16).to(tl.float32)
        x14=tl.load(X_ptr+base+14*stride_16).to(tl.float32);x15=tl.load(X_ptr+base+15*stride_16).to(tl.float32)
        g0 = x0*x0
        g1 = x1*x1 - x2*x2 - x3*x3 - x4*x4
        g2 = -x5*x5 - x6*x6 - x7*x7 + x8*x8 + x9*x9 + x10*x10
        g3 = x11*x11 + x12*x12 + x13*x13 - x14*x14
        g4 = -x15*x15
        total_norm += tl.abs(g0) + tl.abs(g1) + tl.abs(g2) + tl.abs(g3) + tl.abs(g4)
    inv_c = 1.0 / C
    mean_norm = total_norm * inv_c
    mean_norm = tl.where(mean_norm < epsilon, epsilon, mean_norm)
    scale = gain * tl.rsqrt(mean_norm)
    for c in range(C):
        base_in = pid * stride_n + c * stride_c
        base_out = pid * stride_on + c * stride_oc
        for j in tl.static_range(16):
            xj = tl.load(X_ptr + base_in + j * stride_16).to(tl.float32)
            tl.store(OUT_ptr + base_out + j * stride_o16, xj * scale)


def equi_layernorm_forward(x: torch.Tensor, channel_dim: int = -2,
                            gain: float = 1.0, epsilon: float = 0.01) -> torch.Tensor:
    assert channel_dim == -2
    C = x.shape[-2]
    N = x[..., 0, 0].numel()
    x_flat = x.reshape(N, C, 16).contiguous()
    out = torch.empty_like(x_flat)
    _equi_layernorm_fwd_kernel[(N,)](
        x_flat, out, N, C, gain, epsilon,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        out.stride(0), out.stride(1), out.stride(2), BLOCK_C=C,
    )
    return out.reshape(x.shape)
