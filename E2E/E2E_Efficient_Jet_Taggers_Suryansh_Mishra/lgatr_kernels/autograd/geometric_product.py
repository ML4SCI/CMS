"""Custom autograd Function for the fused geometric product."""

import torch
from torch.autograd import Function
from ..triton.geometric_product_kernel import geometric_product_forward, geometric_product_backward


class GeometricProductFunction(Function):
    @staticmethod
    def forward(ctx, x, y, zero_bivector=False):
        ctx.save_for_backward(x, y)
        ctx.zero_bivector = zero_bivector
        return geometric_product_forward(x, y, zero_bivector=zero_bivector)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        if ctx.zero_bivector:
            grad_output = grad_output.clone()
            grad_output[..., 5:11] = 0.0
        grad_x, grad_y = geometric_product_backward(grad_output, x, y)
        return grad_x, grad_y, None


def triton_geometric_product(x, y):
    return GeometricProductFunction.apply(x, y, False)
