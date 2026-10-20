"""Drop-in replacement primitives for lgatr.

Usage:
    from lgatr_kernels.primitives import equi_linear, geometric_product, equi_layer_norm
    # Or monkey-patch: from lgatr_kernels.primitives import patch_lgatr; patch_lgatr()
"""

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as _torch_sdpa

from .triton.equi_linear_gemm import gemm_equi_linear
from .autograd.geometric_product import triton_geometric_product
from .autograd.equi_layernorm import triton_equi_layer_norm
from .autograd.gated_gelu import triton_gated_gelu
from .triton.attention_prep_kernel import fused_attn_qkv_prep, fused_attn_output_split

equi_linear = gemm_equi_linear
geometric_product = triton_geometric_product
equi_layer_norm = triton_equi_layer_norm
gated_gelu = triton_gated_gelu


def _triton_gated_gelu_compat(x: Tensor, gates: Tensor) -> Tensor:
    """Compat wrapper matching upstream ``gated_gelu(x, gates)`` signature.

    The Triton kernel extracts the scalar gate from ``x[..., 0]`` internally,
    so *gates* is unused.
    """
    return triton_gated_gelu(x)


def _fused_sdp_attention(
    q_mv: Tensor, k_mv: Tensor, v_mv: Tensor,
    q_s: Tensor, k_s: Tensor, v_s: Tensor,
    **attn_kwargs,
) -> tuple[Tensor, Tensor]:
    """Drop-in replacement for ``lgatr.primitives.attention.sdp_attention``.

    Avoids ``einops.rearrange`` overhead and uses the Triton QKV-prep kernel
    when running without gradients (inference).  Falls back to fast reshape
    ops otherwise so autograd still works for training.
    """
    from lgatr.primitives.invariants import _load_inner_product_factors
    from lgatr.primitives.attention_backends import get_attention_backend

    num_channels_out = v_mv.shape[-2]

    if not torch.is_grad_enabled():
        q, k, v = fused_attn_qkv_prep(q_mv, k_mv, v_mv, q_s, k_s, v_s)
    else:
        factors = _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype)
        q = torch.cat([(q_mv * factors).flatten(-2), q_s], -1)
        k = torch.cat([k_mv.flatten(-2), k_s], -1)
        v = torch.cat([v_mv.flatten(-2), v_s], -1)

    attention_backend = get_attention_backend(**attn_kwargs)
    v_out = attention_backend(q, k, v, **attn_kwargs)

    v_out_mv = v_out[..., :num_channels_out * 16].unflatten(-1, (num_channels_out, 16))
    v_out_s = v_out[..., num_channels_out * 16:]
    return v_out_mv, v_out_s


def patch_lgatr():
    """Monkey-patch lgatr to use the proven-faster kernels only.

    Patches:
      - equi_linear   -> GEMM-based (used by FusedEquiLinear internally)
      - geometric_product -> Triton hardcoded Cayley arithmetic

    Does NOT patch LayerNorm or Gated GELU because isolated benchmarks
    show they regress at the default LGATr shape (B=128, items=128, mv=8).
    Those are left to the upstream lgatr implementations, which are faster
    in eager mode.  If you also enable torch.compile, the compiler handles
    them better than either version.
    """
    import lgatr.primitives.linear as linear_mod
    import lgatr.primitives.bilinear as bilinear_mod

    linear_mod.equi_linear = gemm_equi_linear
    bilinear_mod.geometric_product = triton_geometric_product

    print("[lgatr_kernels] Patched lgatr primitives (equi_linear, geometric_product)")
