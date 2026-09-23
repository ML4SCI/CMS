"""Fused Triton kernel for scalar-gated GELU nonlinearity.

GatedGELU(x) = GELU(x[..., 0]) * x   where x has shape (..., 16).
Uses erf-based GELU matching PyTorch's default.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch


@triton.jit
def _gated_gelu_fwd_kernel(
    X_ptr, OUT_ptr, N,
    stride_n, stride_16, stride_on, stride_o16,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    xb = offs * stride_n
    x0 = tl.load(X_ptr + xb + 0 * stride_16, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    gate = 0.5 * x0 * (1.0 + libdevice.erf(x0 * inv_sqrt2))
    ob = offs * stride_on
    for j in tl.static_range(16):
        xj = tl.load(X_ptr + xb + j * stride_16, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT_ptr + ob + j * stride_o16, gate * xj, mask=mask)


@triton.jit
def _gated_gelu_bwd_kernel(
    GRAD_OUT_ptr, X_ptr, GRAD_X_ptr, N,
    stride_gon, stride_go16, stride_xn, stride_x16, stride_gxn, stride_gx16,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    xb = offs * stride_xn
    gob = offs * stride_gon
    gxb = offs * stride_gxn
    x0 = tl.load(X_ptr + xb + 0 * stride_x16, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    inv_sqrt_2pi = 0.3989422804014327
    erf_val = libdevice.erf(x0 * inv_sqrt2)
    gelu_val = 0.5 * x0 * (1.0 + erf_val)
    gelu_grad = 0.5 * (1.0 + erf_val) + x0 * inv_sqrt_2pi * libdevice.exp(-0.5 * x0 * x0)
    dot = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in tl.static_range(16):
        go_j = tl.load(GRAD_OUT_ptr + gob + j * stride_go16, mask=mask, other=0.0).to(tl.float32)
        x_j = tl.load(X_ptr + xb + j * stride_x16, mask=mask, other=0.0).to(tl.float32)
        dot += go_j * x_j
    go_0 = tl.load(GRAD_OUT_ptr + gob + 0 * stride_go16, mask=mask, other=0.0).to(tl.float32)
    gx_0 = gelu_val * go_0 + gelu_grad * dot
    tl.store(GRAD_X_ptr + gxb + 0 * stride_gx16, gx_0, mask=mask)
    for j in tl.static_range(1, 16):
        go_j = tl.load(GRAD_OUT_ptr + gob + j * stride_go16, mask=mask, other=0.0).to(tl.float32)
        tl.store(GRAD_X_ptr + gxb + j * stride_gx16, gelu_val * go_j, mask=mask)


def gated_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    N = x[..., 0].numel()
    x_flat = x.reshape(N, 16).contiguous()
    out = torch.empty_like(x_flat)
    BLOCK_N = min(256, triton.next_power_of_2(N))
    _gated_gelu_fwd_kernel[(triton.cdiv(N, BLOCK_N),)](
        x_flat, out, N, x_flat.stride(0), x_flat.stride(1),
        out.stride(0), out.stride(1), BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


def gated_gelu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    N = x[..., 0].numel()
    go_flat = grad_output.reshape(N, 16).contiguous()
    x_flat = x.reshape(N, 16).contiguous()
    grad_x = torch.empty_like(x_flat)
    BLOCK_N = min(256, triton.next_power_of_2(N))
    _gated_gelu_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
        go_flat, x_flat, grad_x, N,
        go_flat.stride(0), go_flat.stride(1), x_flat.stride(0), x_flat.stride(1),
        grad_x.stride(0), grad_x.stride(1), BLOCK_N=BLOCK_N,
    )
    return grad_x.reshape(orig_shape)
