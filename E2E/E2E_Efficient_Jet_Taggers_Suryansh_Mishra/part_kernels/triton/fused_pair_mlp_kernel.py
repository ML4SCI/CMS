"""
Fused Triton kernel: pairwise Lorentz features + full Conv1d MLP.

Computes pair features AND the entire PairEmbed MLP in one kernel launch,
keeping all MLP intermediates (64-wide hidden state) in registers.

Eval-mode only (BatchNorm folded into linear weights).
"""
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# ---- BN folding utilities -------------------------------------------------

def _fold_bn(bn: nn.BatchNorm1d):
    sigma = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / sigma
    shift = bn.bias - bn.running_mean * scale
    return scale.detach(), shift.detach()


def _fold_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    sigma = torch.sqrt(bn.running_var + bn.eps)
    s = bn.weight / sigma
    W = conv.weight.squeeze(-1) * s[:, None]
    b = (conv.bias - bn.running_mean) * s + bn.bias
    return W.detach().contiguous(), b.detach().contiguous()


def pack_pair_embed_weights(embed_seq: nn.Sequential):
    """Extract and fold all weights from PairEmbed's Sequential for eval."""
    layers = list(embed_seq.children())
    input_bn = layers[0]
    bn0_s, bn0_b = _fold_bn(input_bn)

    convs, bns = [], []
    for m in layers[1:]:
        if isinstance(m, nn.Conv1d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm1d):
            bns.append(m)

    Ws, bs = [], []
    for c, b in zip(convs, bns):
        W, bias = _fold_conv_bn(c, b)
        Ws.append(W)
        bs.append(bias)

    return bn0_s, bn0_b, Ws, bs


# ---- fused kernel ----------------------------------------------------------

@triton.jit
def _fused_pair_mlp_kernel(
    V_ptr, Out_ptr,
    BN0s_ptr, BN0b_ptr,
    W1_ptr, B1_ptr,
    W2_ptr, B2_ptr,
    W3_ptr, B3_ptr,
    W4_ptr, B4_ptr,
    P,
    stride_vn, stride_vc, stride_vp,
    EPS: tl.constexpr,
    PI: tl.constexpr,
    TWO_PI: tl.constexpr,
    HID: tl.constexpr,
    OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    PP = P * P

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    ii = offs // P
    jj = offs % P
    valid = (ii < P) & (jj < P)

    base = pid_n * stride_vn
    px_i = tl.load(V_ptr + base + 0 * stride_vc + ii * stride_vp, mask=valid, other=0.0)
    py_i = tl.load(V_ptr + base + 1 * stride_vc + ii * stride_vp, mask=valid, other=0.0)
    pz_i = tl.load(V_ptr + base + 2 * stride_vc + ii * stride_vp, mask=valid, other=0.0)
    e_i  = tl.load(V_ptr + base + 3 * stride_vc + ii * stride_vp, mask=valid, other=1.0)
    px_j = tl.load(V_ptr + base + 0 * stride_vc + jj * stride_vp, mask=valid, other=0.0)
    py_j = tl.load(V_ptr + base + 1 * stride_vc + jj * stride_vp, mask=valid, other=0.0)
    pz_j = tl.load(V_ptr + base + 2 * stride_vc + jj * stride_vp, mask=valid, other=0.0)
    e_j  = tl.load(V_ptr + base + 3 * stride_vc + jj * stride_vp, mask=valid, other=1.0)

    pt_i = tl.sqrt(px_i * px_i + py_i * py_i)
    pt_j = tl.sqrt(px_j * px_j + py_j * py_j)
    rap_i = 0.5 * tl.log(1.0 + 2.0 * pz_i / tl.maximum(e_i - pz_i, 1e-20))
    rap_j = 0.5 * tl.log(1.0 + 2.0 * pz_j / tl.maximum(e_j - pz_j, 1e-20))
    phi_i = libdevice.atan2(py_i, px_i)
    phi_j = libdevice.atan2(py_j, px_j)
    diff = phi_i - phi_j + PI
    dphi = diff - tl.math.floor(diff / TWO_PI) * TWO_PI - PI
    drap = rap_i - rap_j
    delta = tl.sqrt(drap * drap + dphi * dphi)
    ptmin = tl.minimum(pt_i, pt_j)

    f0 = tl.log(tl.maximum(ptmin * delta, EPS))
    f1 = tl.log(tl.maximum(ptmin / tl.maximum(pt_i + pt_j, EPS), EPS))
    f2 = tl.log(tl.maximum(delta, EPS))
    es = e_i + e_j; pxs = px_i + px_j; pys = py_i + py_j; pzs = pz_i + pz_j
    f3 = tl.log(tl.maximum(es * es - pxs * pxs - pys * pys - pzs * pzs, EPS))

    # ---- input BN (affine) ----
    s_0 = tl.load(BN0s_ptr + 0); b_0 = tl.load(BN0b_ptr + 0)
    s_1 = tl.load(BN0s_ptr + 1); b_1 = tl.load(BN0b_ptr + 1)
    s_2 = tl.load(BN0s_ptr + 2); b_2 = tl.load(BN0b_ptr + 2)
    s_3 = tl.load(BN0s_ptr + 3); b_3 = tl.load(BN0b_ptr + 3)
    f0 = f0 * s_0 + b_0
    f1 = f1 * s_1 + b_1
    f2 = f2 * s_2 + b_2
    f3 = f3 * s_3 + b_3

    # ---- layer 1: 4 -> HID  (outer-product accumulation) ----
    hid_range = tl.arange(0, HID)
    w1c0 = tl.load(W1_ptr + hid_range * 4 + 0)
    w1c1 = tl.load(W1_ptr + hid_range * 4 + 1)
    w1c2 = tl.load(W1_ptr + hid_range * 4 + 2)
    w1c3 = tl.load(W1_ptr + hid_range * 4 + 3)
    bias1 = tl.load(B1_ptr + hid_range)

    h = (tl.expand_dims(f0, 1) * tl.expand_dims(w1c0, 0) +
         tl.expand_dims(f1, 1) * tl.expand_dims(w1c1, 0) +
         tl.expand_dims(f2, 1) * tl.expand_dims(w1c2, 0) +
         tl.expand_dims(f3, 1) * tl.expand_dims(w1c3, 0) +
         tl.expand_dims(bias1, 0))

    # ---- GELU + layer 2: HID -> HID ----
    h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    W2 = tl.load(W2_ptr + hid_range[:, None] * HID + hid_range[None, :])
    bias2 = tl.load(B2_ptr + hid_range)
    h = tl.dot(h, tl.trans(W2)) + tl.expand_dims(bias2, 0)

    # ---- GELU + layer 3: HID -> HID ----
    h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    W3 = tl.load(W3_ptr + hid_range[:, None] * HID + hid_range[None, :])
    bias3 = tl.load(B3_ptr + hid_range)
    h = tl.dot(h, tl.trans(W3)) + tl.expand_dims(bias3, 0)

    # ---- GELU + layer 4: HID -> OUT  (per-channel reduction) ----
    h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    out_base = Out_ptr + pid_n * OUT * PP
    for c in tl.static_range(OUT):
        w4_row = tl.load(W4_ptr + c * HID + hid_range)
        b4_c = tl.load(B4_ptr + c)
        out_c = tl.sum(h * tl.expand_dims(w4_row, 0), axis=1) + b4_c
        tl.store(out_base + c * PP + offs, out_c, mask=valid)
