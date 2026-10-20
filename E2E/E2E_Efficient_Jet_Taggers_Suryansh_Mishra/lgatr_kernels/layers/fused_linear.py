"""Fused EquiLinear layer: replaces the entire lgatr EquiLinear module.

Fuses MV equivariant linear + scalar mixing + bias into a single GEMM by
constructing an expanded weight matrix operating on concatenated [mv_flat, scalars].

Reduces each EquiLinear from ~12 CUDA kernels to 3 (1 addmm fwd, 2 mm bwd).
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from functools import lru_cache


@lru_cache(maxsize=4)
def _get_basis_flat(device, dtype):
    try:
        from lgatr.primitives.linear import _compute_pin_equi_linear_basis
        try:
            from lgatr.primitives.config import gatr_config
            basis = _compute_pin_equi_linear_basis(
                use_fully_connected_subgroup=gatr_config.use_fully_connected_subgroup,
                device=device, dtype=dtype,
            )
        except TypeError:
            basis = _compute_pin_equi_linear_basis(device=device, dtype=dtype)
    except ImportError:
        from ..codegen.cayley_table import compute_linear_basis_sparse
        bases = compute_linear_basis_sparse()
        basis = torch.zeros(10, 16, 16, device=device, dtype=dtype)
        for b_idx, mapping in enumerate(bases):
            for (inp, out), weight in mapping.items():
                basis[b_idx, out, inp] = weight
    return basis.reshape(10, 256).contiguous()


def _build_fused_weight(weight, bias, s2mvs_w, s2mvs_b, mvs2s_w, mvs2s_b, s2s_w,
                         C_in, C_out, S_in, S_out, basis_flat):
    D_in = C_in * 16 + S_in
    D_out = C_out * 16 + S_out
    W = torch.zeros(D_out, D_in, device=weight.device, dtype=weight.dtype)
    b = torch.zeros(D_out, device=weight.device, dtype=weight.dtype)

    # MV -> MV block
    w_flat = weight.reshape(C_out * C_in, 10)
    W_mv = w_flat @ basis_flat
    W_mv = W_mv.reshape(C_out, C_in, 16, 16)
    W[:C_out * 16, :C_in * 16] = W_mv.permute(0, 2, 1, 3).reshape(C_out * 16, C_in * 16)

    # Bias on scalar component
    if bias is not None:
        for co in range(C_out):
            b[co * 16] += bias[co, 0]

    # S_in -> MV scalar/pseudoscalar (s2mvs): interleaved layout
    if s2mvs_w is not None:
        col_start = C_in * 16
        for co in range(C_out):
            W[co * 16,      col_start:col_start + S_in] += s2mvs_w[co * 2]
            W[co * 16 + 15, col_start:col_start + S_in] += s2mvs_w[co * 2 + 1]
        if s2mvs_b is not None:
            for co in range(C_out):
                b[co * 16]      += s2mvs_b[co * 2]
                b[co * 16 + 15] += s2mvs_b[co * 2 + 1]

    # MV scalar/pseudoscalar -> S_out (mvs2s): interleaved input layout
    if mvs2s_w is not None:
        row_start = C_out * 16
        for ci in range(C_in):
            W[row_start:row_start + S_out, ci * 16]      += mvs2s_w[:, ci * 2]
            W[row_start:row_start + S_out, ci * 16 + 15] += mvs2s_w[:, ci * 2 + 1]
        if mvs2s_b is not None:
            b[row_start:row_start + S_out] += mvs2s_b

    # S_in -> S_out (s2s)
    if s2s_w is not None:
        col_start = C_in * 16
        row_start = C_out * 16
        W[row_start:row_start + S_out, col_start:col_start + S_in] += s2s_w

    return W, b


class FusedEquiLinearFunction(Function):
    @staticmethod
    def forward(ctx, x_concat, W_fused, b_fused):
        out = torch.addmm(b_fused.unsqueeze(0), x_concat, W_fused.t())
        ctx.save_for_backward(x_concat, W_fused)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_concat, W_fused = ctx.saved_tensors
        grad_output = grad_output.to(W_fused.dtype)
        grad_x = grad_output @ W_fused
        grad_W = grad_output.t() @ x_concat
        grad_b = grad_output.sum(0)
        return grad_x, grad_W, grad_b


class FusedEquiLinear(nn.Module):
    """Drop-in replacement for lgatr.layers.linear.EquiLinear.

    Fuses the MV equivariant linear, scalar mixing, and bias into a
    single GEMM operating on concatenated [mv_flat, scalars] vectors.
    """

    def __init__(self, original_layer):
        super().__init__()
        # MV channels: always reliable from weight shape (out_mv, in_mv, 10).
        w = original_layer.weight
        self._out_mv = w.shape[0]
        self._in_mv = w.shape[1]

        # Scalar channels: use the stored attributes (work across lgatr versions).
        # These can be None meaning "no scalars on this side".
        in_s = getattr(original_layer, '_in_s_channels', None)
        if in_s is None:
            in_s = getattr(original_layer, 'in_s_channels', None)
        out_s = getattr(original_layer, '_out_s_channels', None)
        if out_s is None:
            out_s = getattr(original_layer, 'out_s_channels', None)
        self._in_s = in_s or 0
        self._out_s = out_s or 0

        D_in = self._in_mv * 16 + self._in_s
        D_out = self._out_mv * 16 + self._out_s

        self.W_fused = nn.Parameter(torch.zeros(D_out, D_in))
        self.b_fused = nn.Parameter(torch.zeros(D_out))
        self._rebuild_fused(original_layer)

    def _rebuild_fused(self, layer):
        basis_flat = _get_basis_flat(layer.weight.device, layer.weight.dtype)
        W, b = _build_fused_weight(
            weight=layer.weight.data,
            bias=layer.bias.data if layer.bias is not None else None,
            s2mvs_w=layer.s2mvs.weight.data if layer.s2mvs is not None else None,
            s2mvs_b=layer.s2mvs.bias.data if (layer.s2mvs is not None and layer.s2mvs.bias is not None) else None,
            mvs2s_w=layer.mvs2s.weight.data if layer.mvs2s is not None else None,
            mvs2s_b=layer.mvs2s.bias.data if (layer.mvs2s is not None and layer.mvs2s.bias is not None) else None,
            s2s_w=layer.s2s.weight.data if layer.s2s is not None else None,
            C_in=self._in_mv, C_out=self._out_mv,
            S_in=self._in_s, S_out=self._out_s,
            basis_flat=basis_flat,
        )
        self.W_fused.data.copy_(W)
        self.b_fused.data.copy_(b)

    def forward(self, multivectors, scalars=None):
        batch_shape = multivectors.shape[:-2]
        N = multivectors[..., 0, 0].numel()
        mv_flat = multivectors.reshape(N, self._in_mv * 16)
        if scalars is not None and self._in_s > 0:
            s_flat = scalars.reshape(N, self._in_s)
            x_concat = torch.cat([mv_flat, s_flat], dim=-1)
        else:
            x_concat = mv_flat
        out = FusedEquiLinearFunction.apply(x_concat, self.W_fused, self.b_fused)
        out_mv = out[:, :self._out_mv * 16].reshape(*batch_shape, self._out_mv, 16)
        out_s = out[:, self._out_mv * 16:].reshape(*batch_shape, self._out_s) if self._out_s > 0 else None
        return out_mv, out_s


def fuse_equi_linear_layers(model):
    """Replace all EquiLinear layers in a model with FusedEquiLinear. Returns count."""
    from lgatr.layers.linear import EquiLinear
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, EquiLinear):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            fused = FusedEquiLinear(module).to(device=module.weight.device, dtype=module.weight.dtype)
            setattr(parent, parts[-1], fused)
            count += 1
    return count
