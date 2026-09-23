"""Custom autograd Function for the fused scalar-gated GELU."""

import torch
from torch.autograd import Function
from ..triton.gated_gelu_kernel import gated_gelu_forward, gated_gelu_backward


class GatedGELUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return gated_gelu_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return gated_gelu_backward(grad_output, x)


def triton_gated_gelu(x):
    return GatedGELUFunction.apply(x)
