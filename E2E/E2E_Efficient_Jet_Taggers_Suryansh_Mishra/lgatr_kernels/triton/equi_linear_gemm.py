"""GEMM-based equivariant linear map.

Decomposes the equivariant linear into two standard matmuls:
  1. Weight expansion: W_eff = weight_flat @ basis_flat  (tiny, ~512x10 @ 10x256)
  2. Forward GEMM:     out   = x_flat @ W_eff.T          (main, Nx256 @ 256x512)
"""

import torch
from torch.autograd import Function
from functools import lru_cache


@lru_cache(maxsize=4)
def _get_basis_flat(device, dtype):
    try:
        from lgatr.primitives.linear import _compute_pin_equi_linear_basis
        from lgatr.primitives.config import gatr_config
        basis = _compute_pin_equi_linear_basis(
            use_fully_connected_subgroup=gatr_config.use_fully_connected_subgroup,
            device=device, dtype=dtype,
        )
    except ImportError:
        from ..codegen.cayley_table import compute_linear_basis_sparse
        bases = compute_linear_basis_sparse()
        basis = torch.zeros(10, 16, 16, device=device, dtype=dtype)
        for b_idx, mapping in enumerate(bases):
            for (inp, out), weight in mapping.items():
                basis[b_idx, out, inp] = weight
    return basis.reshape(10, 256).contiguous()


def _expand_weight(weight, basis_flat):
    C_out, C_in, _ = weight.shape
    w_flat = weight.reshape(C_out * C_in, 10)
    W_eff = w_flat @ basis_flat
    W_eff = W_eff.reshape(C_out, C_in, 16, 16)
    return W_eff.permute(0, 2, 1, 3).reshape(C_out * 16, C_in * 16)


class GemmEquiLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight):
        basis_flat = _get_basis_flat(x.device, x.dtype)
        batch_shape = x.shape[:-2]
        C_in = x.shape[-2]
        C_out = weight.shape[0]
        N = x[..., 0, 0].numel()
        W_mat = _expand_weight(weight, basis_flat)
        x_flat = x.reshape(N, C_in * 16).contiguous()
        out_flat = x_flat @ W_mat.t()
        ctx.save_for_backward(x, weight, W_mat, basis_flat)
        ctx.shapes = (batch_shape, C_in, C_out, N)
        return out_flat.reshape(*batch_shape, C_out, 16)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, W_mat, basis_flat = ctx.saved_tensors
        batch_shape, C_in, C_out, N = ctx.shapes
        go_flat = grad_output.reshape(N, C_out * 16).contiguous()
        x_flat = x.reshape(N, C_in * 16).contiguous()
        grad_x_flat = go_flat @ W_mat
        grad_W_mat = go_flat.t() @ x_flat
        grad_W_eff = grad_W_mat.reshape(C_out, 16, C_in, 16).permute(0, 2, 1, 3).reshape(C_out * C_in, 256)
        grad_weight = (grad_W_eff @ basis_flat.t()).reshape(C_out, C_in, 10)
        return grad_x_flat.reshape(x.shape), grad_weight


def gemm_equi_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for equi_linear using GEMM decomposition."""
    return GemmEquiLinearFunction.apply(x, weight)
