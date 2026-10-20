"""Fused Triton kernel for attention QKV preparation in L-GATr.

Fuses sign-flip (for queries) + reshape + concatenate for q/k/v into one kernel.
Also provides the inverse split operation for the attention output.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _attn_qkv_prep_kernel(
    QMV_ptr, QS_ptr, Q_OUT_ptr, KMV_ptr, KS_ptr, K_OUT_ptr,
    VMV_ptr, VS_ptr, V_OUT_ptr, N,
    mv_ch: tl.constexpr, s_ch: tl.constexpr,
    stride_mvn, stride_mvc, stride_mv16, stride_sn, stride_sc, stride_on, stride_od,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    for idx in range(mv_ch):
        mv_base_q = offs * stride_mvn + idx * stride_mvc
        out_offset = idx * 16
        for j in tl.static_range(16):
            qval = tl.load(QMV_ptr + mv_base_q + j * stride_mv16, mask=mask, other=0.0)
            kval = tl.load(KMV_ptr + offs * stride_mvn + idx * stride_mvc + j * stride_mv16, mask=mask, other=0.0)
            vval = tl.load(VMV_ptr + offs * stride_mvn + idx * stride_mvc + j * stride_mv16, mask=mask, other=0.0)
            if j == 2 or j == 3 or j == 4 or j == 5 or j == 6 or j == 7 or j == 14 or j == 15:
                qval = -qval
            out_d = out_offset + j
            tl.store(Q_OUT_ptr + offs * stride_on + out_d * stride_od, qval, mask=mask)
            tl.store(K_OUT_ptr + offs * stride_on + out_d * stride_od, kval, mask=mask)
            tl.store(V_OUT_ptr + offs * stride_on + out_d * stride_od, vval, mask=mask)
    s_start = mv_ch * 16
    for j in tl.static_range(s_ch):
        qs = tl.load(QS_ptr + offs * stride_sn + j * stride_sc, mask=mask, other=0.0)
        ks = tl.load(KS_ptr + offs * stride_sn + j * stride_sc, mask=mask, other=0.0)
        vs = tl.load(VS_ptr + offs * stride_sn + j * stride_sc, mask=mask, other=0.0)
        out_d = s_start + j
        tl.store(Q_OUT_ptr + offs * stride_on + out_d * stride_od, qs, mask=mask)
        tl.store(K_OUT_ptr + offs * stride_on + out_d * stride_od, ks, mask=mask)
        tl.store(V_OUT_ptr + offs * stride_on + out_d * stride_od, vs, mask=mask)


def fused_attn_qkv_prep(q_mv, k_mv, v_mv, q_s, k_s, v_s):
    batch_shape = q_mv.shape[:-2]
    mv_ch = q_mv.shape[-2]
    s_ch = q_s.shape[-1]
    N = q_mv[..., 0, 0].numel()
    total_d = mv_ch * 16 + s_ch
    q_mv_flat = q_mv.reshape(N, mv_ch, 16).contiguous()
    q_s_flat = q_s.reshape(N, s_ch).contiguous()
    k_mv_flat = k_mv.reshape(N, mv_ch, 16).contiguous()
    k_s_flat = k_s.reshape(N, s_ch).contiguous()
    v_mv_flat = v_mv.reshape(N, mv_ch, 16).contiguous()
    v_s_flat = v_s.reshape(N, s_ch).contiguous()
    q_out = torch.empty(N, total_d, device=q_mv.device, dtype=q_mv.dtype)
    k_out = torch.empty(N, total_d, device=q_mv.device, dtype=q_mv.dtype)
    v_out = torch.empty(N, total_d, device=q_mv.device, dtype=q_mv.dtype)
    BLOCK_N = min(128, triton.next_power_of_2(N))
    _attn_qkv_prep_kernel[(triton.cdiv(N, BLOCK_N),)](
        q_mv_flat, q_s_flat, q_out, k_mv_flat, k_s_flat, k_out,
        v_mv_flat, v_s_flat, v_out, N, mv_ch, s_ch,
        q_mv_flat.stride(0), q_mv_flat.stride(1), q_mv_flat.stride(2),
        q_s_flat.stride(0), q_s_flat.stride(1), q_out.stride(0), q_out.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return (q_out.reshape(*batch_shape, total_d),
            k_out.reshape(*batch_shape, total_d),
            v_out.reshape(*batch_shape, total_d))


def fused_attn_output_split(v_out, mv_channels, s_channels):
    batch_shape = v_out.shape[:-1]
    out_mv = v_out[..., :mv_channels * 16].reshape(*batch_shape, mv_channels, 16)
    out_s = v_out[..., mv_channels * 16:]
    return out_mv, out_s
