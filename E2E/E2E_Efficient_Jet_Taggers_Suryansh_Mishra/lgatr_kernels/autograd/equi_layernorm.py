"""Custom autograd Function for the fused equivariant LayerNorm.

Uses Triton kernel for forward, PyTorch autograd for backward (the backward
involves d|x|/dx = sign(x) which is non-differentiable at zero).
"""

import torch
from torch.autograd import Function
from ..triton.equi_layernorm_kernel import equi_layernorm_forward

INNER_PRODUCT_SIGNS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
GRADE_RANGES = [(0, 1), (1, 5), (5, 11), (11, 15), (15, 16)]


def _pytorch_equi_layer_norm(x, gain=1.0, epsilon=0.01):
    m = torch.tensor(INNER_PRODUCT_SIGNS, device=x.device, dtype=x.dtype)
    total = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
    for start, end in GRADE_RANGES:
        grade_slice = x[..., start:end]
        metric_slice = m[start:end]
        sq = (grade_slice * grade_slice * metric_slice).sum(dim=-1, keepdim=True)
        total = total + sq.abs()
    mean_norm = total.mean(dim=-2, keepdim=True)
    mean_norm = mean_norm.clamp(min=epsilon)
    return gain * x * torch.rsqrt(mean_norm)


class EquiLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, gain, epsilon):
        ctx.save_for_backward(x)
        ctx.gain = gain
        ctx.epsilon = epsilon
        return equi_layernorm_forward(x, gain=gain, epsilon=epsilon)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x_detached = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = _pytorch_equi_layer_norm(x_detached, ctx.gain, ctx.epsilon)
            y.backward(grad_output)
        return x_detached.grad, None, None


def triton_equi_layer_norm(x, channel_dim=-2, gain=1.0, epsilon=0.01):
    assert channel_dim == -2
    return EquiLayerNormFunction.apply(x, gain, epsilon)
