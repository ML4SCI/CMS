"""
Fused Triton kernel for pairwise Lorentz-invariant feature computation.

Replaces ~20 intermediate tensor allocations in the original
pairwise_lv_fts + PairEmbed triangle-gather-expand pipeline with a
single kernel that reads particle 4-vectors and writes pair features
directly into a dense (N, 4, P, P) output.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pairwise_lv_kernel(
    V_ptr,
    Out_ptr,
    P,
    stride_vn, stride_vc, stride_vp,
    stride_on, stride_oc, stride_oi, stride_oj,
    EPS: tl.constexpr,
    PI: tl.constexpr,
    TWO_PI: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i = offs // P
    j = offs % P
    valid = (i < P) & (j < P)

    base = pid_n * stride_vn

    px_i = tl.load(V_ptr + base + 0 * stride_vc + i * stride_vp, mask=valid, other=0.0)
    py_i = tl.load(V_ptr + base + 1 * stride_vc + i * stride_vp, mask=valid, other=0.0)
    pz_i = tl.load(V_ptr + base + 2 * stride_vc + i * stride_vp, mask=valid, other=0.0)
    e_i  = tl.load(V_ptr + base + 3 * stride_vc + i * stride_vp, mask=valid, other=1.0)

    px_j = tl.load(V_ptr + base + 0 * stride_vc + j * stride_vp, mask=valid, other=0.0)
    py_j = tl.load(V_ptr + base + 1 * stride_vc + j * stride_vp, mask=valid, other=0.0)
    pz_j = tl.load(V_ptr + base + 2 * stride_vc + j * stride_vp, mask=valid, other=0.0)
    e_j  = tl.load(V_ptr + base + 3 * stride_vc + j * stride_vp, mask=valid, other=1.0)

    pt_i = tl.sqrt(px_i * px_i + py_i * py_i)
    pt_j = tl.sqrt(px_j * px_j + py_j * py_j)

    rap_i = 0.5 * tl.log(1.0 + 2.0 * pz_i / tl.maximum(e_i - pz_i, 1e-20))
    rap_j = 0.5 * tl.log(1.0 + 2.0 * pz_j / tl.maximum(e_j - pz_j, 1e-20))

    phi_i = libdevice.atan2(py_i, px_i)
    phi_j = libdevice.atan2(py_j, px_j)

    diff = phi_i - phi_j + PI
    wrapped = diff - tl.math.floor(diff / TWO_PI) * TWO_PI
    dphi = wrapped - PI

    drap = rap_i - rap_j
    delta = tl.sqrt(drap * drap + dphi * dphi)

    ptmin = tl.minimum(pt_i, pt_j)
    lnkt    = tl.log(tl.maximum(ptmin * delta, EPS))
    lnz     = tl.log(tl.maximum(ptmin / tl.maximum(pt_i + pt_j, EPS), EPS))
    lndelta = tl.log(tl.maximum(delta, EPS))

    e_s  = e_i + e_j
    px_s = px_i + px_j
    py_s = py_i + py_j
    pz_s = pz_i + pz_j
    m2   = tl.maximum(e_s * e_s - px_s * px_s - py_s * py_s - pz_s * pz_s, EPS)
    lnm2 = tl.log(m2)

    base_out = pid_n * stride_on + i * stride_oi + j * stride_oj
    tl.store(Out_ptr + base_out + 0 * stride_oc, lnkt,    mask=valid)
    tl.store(Out_ptr + base_out + 1 * stride_oc, lnz,     mask=valid)
    tl.store(Out_ptr + base_out + 2 * stride_oc, lndelta, mask=valid)
    tl.store(Out_ptr + base_out + 3 * stride_oc, lnm2,    mask=valid)


def fused_pairwise_lv_fts(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise Lorentz features for all particle pairs.

    Args:
        v:   (N, 4, P)  particle 4-vectors [px, py, pz, E]
        eps: clamping constant for log

    Returns:
        (N, 4, P, P)  pair features [lnkt, lnz, lndelta, lnm2]
    """
    assert v.ndim == 3 and v.shape[1] == 4, f"Expected (N,4,P), got {v.shape}"
    v = v.contiguous().float()
    N, _, P = v.shape
    out = torch.empty(N, 4, P, P, device=v.device, dtype=torch.float32)

    total_pairs = P * P
    BLOCK_SIZE = min(1024, triton.next_power_of_2(total_pairs))
    num_blocks = triton.cdiv(total_pairs, BLOCK_SIZE)

    grid = (N, num_blocks)
    _pairwise_lv_kernel[grid](
        v, out, P,
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=eps,
        PI=math.pi,
        TWO_PI=2.0 * math.pi,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
