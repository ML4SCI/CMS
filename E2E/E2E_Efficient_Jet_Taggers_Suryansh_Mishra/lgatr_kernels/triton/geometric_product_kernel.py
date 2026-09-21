"""Fused Triton kernel for the geometric product in Cl(1,3).

Replaces einsum("i j k, ... j, ... k -> ... i", gp_tensor, x, y) with
hardcoded arithmetic from the Cayley table. All 256 nonzero entries are
compiled directly into the kernel -- no runtime tensor lookup.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _geometric_product_fwd_kernel(
    X_ptr, Y_ptr, OUT_ptr, N,
    stride_xn, stride_x16, stride_yn, stride_y16, stride_on, stride_o16,
    ZERO_BIVECTOR: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    xb = offs * stride_xn
    x0  = tl.load(X_ptr + xb + 0  * stride_x16, mask=mask, other=0.0)
    x1  = tl.load(X_ptr + xb + 1  * stride_x16, mask=mask, other=0.0)
    x2  = tl.load(X_ptr + xb + 2  * stride_x16, mask=mask, other=0.0)
    x3  = tl.load(X_ptr + xb + 3  * stride_x16, mask=mask, other=0.0)
    x4  = tl.load(X_ptr + xb + 4  * stride_x16, mask=mask, other=0.0)
    x5  = tl.load(X_ptr + xb + 5  * stride_x16, mask=mask, other=0.0)
    x6  = tl.load(X_ptr + xb + 6  * stride_x16, mask=mask, other=0.0)
    x7  = tl.load(X_ptr + xb + 7  * stride_x16, mask=mask, other=0.0)
    x8  = tl.load(X_ptr + xb + 8  * stride_x16, mask=mask, other=0.0)
    x9  = tl.load(X_ptr + xb + 9  * stride_x16, mask=mask, other=0.0)
    x10 = tl.load(X_ptr + xb + 10 * stride_x16, mask=mask, other=0.0)
    x11 = tl.load(X_ptr + xb + 11 * stride_x16, mask=mask, other=0.0)
    x12 = tl.load(X_ptr + xb + 12 * stride_x16, mask=mask, other=0.0)
    x13 = tl.load(X_ptr + xb + 13 * stride_x16, mask=mask, other=0.0)
    x14 = tl.load(X_ptr + xb + 14 * stride_x16, mask=mask, other=0.0)
    x15 = tl.load(X_ptr + xb + 15 * stride_x16, mask=mask, other=0.0)

    yb = offs * stride_yn
    y0  = tl.load(Y_ptr + yb + 0  * stride_y16, mask=mask, other=0.0)
    y1  = tl.load(Y_ptr + yb + 1  * stride_y16, mask=mask, other=0.0)
    y2  = tl.load(Y_ptr + yb + 2  * stride_y16, mask=mask, other=0.0)
    y3  = tl.load(Y_ptr + yb + 3  * stride_y16, mask=mask, other=0.0)
    y4  = tl.load(Y_ptr + yb + 4  * stride_y16, mask=mask, other=0.0)
    y5  = tl.load(Y_ptr + yb + 5  * stride_y16, mask=mask, other=0.0)
    y6  = tl.load(Y_ptr + yb + 6  * stride_y16, mask=mask, other=0.0)
    y7  = tl.load(Y_ptr + yb + 7  * stride_y16, mask=mask, other=0.0)
    y8  = tl.load(Y_ptr + yb + 8  * stride_y16, mask=mask, other=0.0)
    y9  = tl.load(Y_ptr + yb + 9  * stride_y16, mask=mask, other=0.0)
    y10 = tl.load(Y_ptr + yb + 10 * stride_y16, mask=mask, other=0.0)
    y11 = tl.load(Y_ptr + yb + 11 * stride_y16, mask=mask, other=0.0)
    y12 = tl.load(Y_ptr + yb + 12 * stride_y16, mask=mask, other=0.0)
    y13 = tl.load(Y_ptr + yb + 13 * stride_y16, mask=mask, other=0.0)
    y14 = tl.load(Y_ptr + yb + 14 * stride_y16, mask=mask, other=0.0)
    y15 = tl.load(Y_ptr + yb + 15 * stride_y16, mask=mask, other=0.0)

    o0 = (x0*y0 + x1*y1 - x2*y2 - x3*y3 - x4*y4 + x5*y5 + x6*y6 + x7*y7 - x8*y8 - x9*y9 - x10*y10 - x11*y11 - x12*y12 - x13*y13 + x14*y14 - x15*y15)
    o1 = (x0*y1 + x1*y0 + x2*y5 + x3*y6 + x4*y7 - x5*y2 - x6*y3 - x7*y4 - x8*y11 - x9*y12 - x10*y13 - x11*y8 - x12*y9 - x13*y10 - x14*y15 + x15*y14)
    o2 = (x0*y2 + x1*y5 + x2*y0 + x3*y8 + x4*y9 - x5*y1 - x6*y11 - x7*y12 - x8*y3 - x9*y4 - x10*y14 - x11*y6 - x12*y7 - x13*y15 - x14*y10 + x15*y13)
    o3 = (x0*y3 + x1*y6 - x2*y8 + x3*y0 + x4*y10 + x5*y11 - x6*y1 - x7*y13 + x8*y2 + x9*y14 - x10*y4 + x11*y5 + x12*y15 - x13*y7 + x14*y9 - x15*y12)
    o4 = (x0*y4 + x1*y7 - x2*y9 - x3*y10 + x4*y0 + x5*y12 + x6*y13 - x7*y1 - x8*y14 + x9*y2 + x10*y3 - x11*y15 + x12*y5 + x13*y6 - x14*y8 + x15*y11)

    if ZERO_BIVECTOR:
        o5 = tl.zeros_like(o0); o6 = tl.zeros_like(o0); o7 = tl.zeros_like(o0)
        o8 = tl.zeros_like(o0); o9 = tl.zeros_like(o0); o10 = tl.zeros_like(o0)
    else:
        o5 = (x0*y5 + x1*y2 - x2*y1 - x3*y11 - x4*y12 + x5*y0 + x6*y8 + x7*y9 - x8*y6 - x9*y7 - x10*y15 - x11*y3 - x12*y4 - x13*y14 + x14*y13 - x15*y10)
        o6 = (x0*y6 + x1*y3 + x2*y11 - x3*y1 - x4*y13 - x5*y8 + x6*y0 + x7*y10 + x8*y5 + x9*y15 - x10*y7 + x11*y2 + x12*y14 - x13*y4 - x14*y12 + x15*y9)
        o7 = (x0*y7 + x1*y4 + x2*y12 + x3*y13 - x4*y1 - x5*y9 - x6*y10 + x7*y0 - x8*y15 + x9*y5 + x10*y6 - x11*y14 + x12*y2 + x13*y3 + x14*y11 - x15*y8)
        o8 = (x0*y8 + x1*y11 + x2*y3 - x3*y2 - x4*y14 - x5*y6 + x6*y5 + x7*y15 + x8*y0 + x9*y10 - x10*y9 + x11*y1 + x12*y13 - x13*y12 - x14*y4 + x15*y7)
        o9 = (x0*y9 + x1*y12 + x2*y4 + x3*y14 - x4*y2 - x5*y7 - x6*y15 + x7*y5 - x8*y10 + x9*y0 + x10*y8 - x11*y13 + x12*y1 + x13*y11 + x14*y3 - x15*y6)
        o10 = (x0*y10 + x1*y13 - x2*y14 + x3*y4 - x4*y3 + x5*y15 - x6*y7 + x7*y6 + x8*y9 - x9*y8 + x10*y0 + x11*y12 - x12*y11 + x13*y1 - x14*y2 + x15*y5)

    o11 = (x0*y11 + x1*y8 - x2*y6 + x3*y5 + x4*y15 + x5*y3 - x6*y2 - x7*y14 + x8*y1 + x9*y13 - x10*y12 + x11*y0 + x12*y10 - x13*y9 + x14*y7 - x15*y4)
    o12 = (x0*y12 + x1*y9 - x2*y7 - x3*y15 + x4*y5 + x5*y4 + x6*y14 - x7*y2 - x8*y13 + x9*y1 + x10*y11 - x11*y10 + x12*y0 + x13*y8 - x14*y6 + x15*y3)
    o13 = (x0*y13 + x1*y10 + x2*y15 - x3*y7 + x4*y6 - x5*y14 + x6*y4 - x7*y3 + x8*y12 - x9*y11 + x10*y1 + x11*y9 - x12*y8 + x13*y0 + x14*y5 - x15*y2)
    o14 = (x0*y14 + x1*y15 + x2*y10 - x3*y9 + x4*y8 - x5*y13 + x6*y12 - x7*y11 + x8*y4 - x9*y3 + x10*y2 + x11*y7 - x12*y6 + x13*y5 + x14*y0 - x15*y1)
    o15 = (x0*y15 + x1*y14 - x2*y13 + x3*y12 - x4*y11 + x5*y10 - x6*y9 + x7*y8 + x8*y7 - x9*y6 + x10*y5 + x11*y4 - x12*y3 + x13*y2 - x14*y1 + x15*y0)

    ob = offs * stride_on
    tl.store(OUT_ptr + ob + 0  * stride_o16, o0,  mask=mask)
    tl.store(OUT_ptr + ob + 1  * stride_o16, o1,  mask=mask)
    tl.store(OUT_ptr + ob + 2  * stride_o16, o2,  mask=mask)
    tl.store(OUT_ptr + ob + 3  * stride_o16, o3,  mask=mask)
    tl.store(OUT_ptr + ob + 4  * stride_o16, o4,  mask=mask)
    tl.store(OUT_ptr + ob + 5  * stride_o16, o5,  mask=mask)
    tl.store(OUT_ptr + ob + 6  * stride_o16, o6,  mask=mask)
    tl.store(OUT_ptr + ob + 7  * stride_o16, o7,  mask=mask)
    tl.store(OUT_ptr + ob + 8  * stride_o16, o8,  mask=mask)
    tl.store(OUT_ptr + ob + 9  * stride_o16, o9,  mask=mask)
    tl.store(OUT_ptr + ob + 10 * stride_o16, o10, mask=mask)
    tl.store(OUT_ptr + ob + 11 * stride_o16, o11, mask=mask)
    tl.store(OUT_ptr + ob + 12 * stride_o16, o12, mask=mask)
    tl.store(OUT_ptr + ob + 13 * stride_o16, o13, mask=mask)
    tl.store(OUT_ptr + ob + 14 * stride_o16, o14, mask=mask)
    tl.store(OUT_ptr + ob + 15 * stride_o16, o15, mask=mask)


@triton.jit
def _geometric_product_bwd_kernel(
    GRAD_OUT_ptr, X_ptr, Y_ptr, GRAD_X_ptr, GRAD_Y_ptr, N,
    stride_gon, stride_go16, stride_xn, stride_x16, stride_yn, stride_y16,
    stride_gxn, stride_gx16, stride_gyn, stride_gy16,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    gb = offs * stride_gon
    g0=tl.load(GRAD_OUT_ptr+gb+0*stride_go16,mask=mask,other=0.0);g1=tl.load(GRAD_OUT_ptr+gb+1*stride_go16,mask=mask,other=0.0)
    g2=tl.load(GRAD_OUT_ptr+gb+2*stride_go16,mask=mask,other=0.0);g3=tl.load(GRAD_OUT_ptr+gb+3*stride_go16,mask=mask,other=0.0)
    g4=tl.load(GRAD_OUT_ptr+gb+4*stride_go16,mask=mask,other=0.0);g5=tl.load(GRAD_OUT_ptr+gb+5*stride_go16,mask=mask,other=0.0)
    g6=tl.load(GRAD_OUT_ptr+gb+6*stride_go16,mask=mask,other=0.0);g7=tl.load(GRAD_OUT_ptr+gb+7*stride_go16,mask=mask,other=0.0)
    g8=tl.load(GRAD_OUT_ptr+gb+8*stride_go16,mask=mask,other=0.0);g9=tl.load(GRAD_OUT_ptr+gb+9*stride_go16,mask=mask,other=0.0)
    g10=tl.load(GRAD_OUT_ptr+gb+10*stride_go16,mask=mask,other=0.0);g11=tl.load(GRAD_OUT_ptr+gb+11*stride_go16,mask=mask,other=0.0)
    g12=tl.load(GRAD_OUT_ptr+gb+12*stride_go16,mask=mask,other=0.0);g13=tl.load(GRAD_OUT_ptr+gb+13*stride_go16,mask=mask,other=0.0)
    g14=tl.load(GRAD_OUT_ptr+gb+14*stride_go16,mask=mask,other=0.0);g15=tl.load(GRAD_OUT_ptr+gb+15*stride_go16,mask=mask,other=0.0)

    xb = offs * stride_xn
    x0=tl.load(X_ptr+xb+0*stride_x16,mask=mask,other=0.0);x1=tl.load(X_ptr+xb+1*stride_x16,mask=mask,other=0.0)
    x2=tl.load(X_ptr+xb+2*stride_x16,mask=mask,other=0.0);x3=tl.load(X_ptr+xb+3*stride_x16,mask=mask,other=0.0)
    x4=tl.load(X_ptr+xb+4*stride_x16,mask=mask,other=0.0);x5=tl.load(X_ptr+xb+5*stride_x16,mask=mask,other=0.0)
    x6=tl.load(X_ptr+xb+6*stride_x16,mask=mask,other=0.0);x7=tl.load(X_ptr+xb+7*stride_x16,mask=mask,other=0.0)
    x8=tl.load(X_ptr+xb+8*stride_x16,mask=mask,other=0.0);x9=tl.load(X_ptr+xb+9*stride_x16,mask=mask,other=0.0)
    x10=tl.load(X_ptr+xb+10*stride_x16,mask=mask,other=0.0);x11=tl.load(X_ptr+xb+11*stride_x16,mask=mask,other=0.0)
    x12=tl.load(X_ptr+xb+12*stride_x16,mask=mask,other=0.0);x13=tl.load(X_ptr+xb+13*stride_x16,mask=mask,other=0.0)
    x14=tl.load(X_ptr+xb+14*stride_x16,mask=mask,other=0.0);x15=tl.load(X_ptr+xb+15*stride_x16,mask=mask,other=0.0)

    yb = offs * stride_yn
    y0=tl.load(Y_ptr+yb+0*stride_y16,mask=mask,other=0.0);y1=tl.load(Y_ptr+yb+1*stride_y16,mask=mask,other=0.0)
    y2=tl.load(Y_ptr+yb+2*stride_y16,mask=mask,other=0.0);y3=tl.load(Y_ptr+yb+3*stride_y16,mask=mask,other=0.0)
    y4=tl.load(Y_ptr+yb+4*stride_y16,mask=mask,other=0.0);y5=tl.load(Y_ptr+yb+5*stride_y16,mask=mask,other=0.0)
    y6=tl.load(Y_ptr+yb+6*stride_y16,mask=mask,other=0.0);y7=tl.load(Y_ptr+yb+7*stride_y16,mask=mask,other=0.0)
    y8=tl.load(Y_ptr+yb+8*stride_y16,mask=mask,other=0.0);y9=tl.load(Y_ptr+yb+9*stride_y16,mask=mask,other=0.0)
    y10=tl.load(Y_ptr+yb+10*stride_y16,mask=mask,other=0.0);y11=tl.load(Y_ptr+yb+11*stride_y16,mask=mask,other=0.0)
    y12=tl.load(Y_ptr+yb+12*stride_y16,mask=mask,other=0.0);y13=tl.load(Y_ptr+yb+13*stride_y16,mask=mask,other=0.0)
    y14=tl.load(Y_ptr+yb+14*stride_y16,mask=mask,other=0.0);y15=tl.load(Y_ptr+yb+15*stride_y16,mask=mask,other=0.0)

    # grad_x: code-generated from Cayley table
    gx0 = (g0*y0 + g1*y1 + g2*y2 + g3*y3 + g4*y4 + g5*y5 + g6*y6 + g7*y7 + g8*y8 + g9*y9 + g10*y10 + g11*y11 + g12*y12 + g13*y13 + g14*y14 + g15*y15)
    gx1 = (g0*y1 + g1*y0 + g2*y5 + g3*y6 + g4*y7 + g5*y2 + g6*y3 + g7*y4 + g8*y11 + g9*y12 + g10*y13 + g11*y8 + g12*y9 + g13*y10 + g14*y15 + g15*y14)
    gx2 = (-g0*y2 + g1*y5 + g2*y0 - g3*y8 - g4*y9 - g5*y1 + g6*y11 + g7*y12 + g8*y3 + g9*y4 - g10*y14 - g11*y6 - g12*y7 + g13*y15 + g14*y10 - g15*y13)
    gx3 = (-g0*y3 + g1*y6 + g2*y8 + g3*y0 - g4*y10 - g5*y11 - g6*y1 + g7*y13 - g8*y2 + g9*y14 + g10*y4 + g11*y5 - g12*y15 - g13*y7 - g14*y9 + g15*y12)
    gx4 = (-g0*y4 + g1*y7 + g2*y9 + g3*y10 + g4*y0 - g5*y12 - g6*y13 - g7*y1 - g8*y14 - g9*y2 - g10*y3 + g11*y15 + g12*y5 + g13*y6 + g14*y8 - g15*y11)
    gx5 = (g0*y5 - g1*y2 - g2*y1 + g3*y11 + g4*y12 + g5*y0 - g6*y8 - g7*y9 - g8*y6 - g9*y7 + g10*y15 + g11*y3 + g12*y4 - g13*y14 - g14*y13 + g15*y10)
    gx6 = (g0*y6 - g1*y3 - g2*y11 - g3*y1 + g4*y13 + g5*y8 + g6*y0 - g7*y10 + g8*y5 - g9*y15 - g10*y7 - g11*y2 + g12*y14 + g13*y4 + g14*y12 - g15*y9)
    gx7 = (g0*y7 - g1*y4 - g2*y12 - g3*y13 - g4*y1 + g5*y9 + g6*y10 + g7*y0 + g8*y15 + g9*y5 + g10*y6 - g11*y14 - g12*y2 - g13*y3 - g14*y11 + g15*y8)
    gx8 = (-g0*y8 - g1*y11 - g2*y3 + g3*y2 - g4*y14 - g5*y6 + g6*y5 - g7*y15 + g8*y0 - g9*y10 + g10*y9 + g11*y1 - g12*y13 + g13*y12 + g14*y4 + g15*y7)
    gx9 = (-g0*y9 - g1*y12 - g2*y4 + g3*y14 + g4*y2 - g5*y7 + g6*y15 + g7*y5 + g8*y10 + g9*y0 - g10*y8 + g11*y13 + g12*y1 - g13*y11 - g14*y3 - g15*y6)
    gx10 = (-g0*y10 - g1*y13 - g2*y14 - g3*y4 + g4*y3 - g5*y15 - g6*y7 + g7*y6 - g8*y9 + g9*y8 + g10*y0 - g11*y12 + g12*y11 + g13*y1 + g14*y2 + g15*y5)
    gx11 = (-g0*y11 - g1*y8 - g2*y6 + g3*y5 - g4*y15 - g5*y3 + g6*y2 - g7*y14 + g8*y1 - g9*y13 + g10*y12 + g11*y0 - g12*y10 + g13*y9 + g14*y7 + g15*y4)
    gx12 = (-g0*y12 - g1*y9 - g2*y7 + g3*y15 + g4*y5 - g5*y4 + g6*y14 + g7*y2 + g8*y13 + g9*y1 - g10*y11 + g11*y10 + g12*y0 - g13*y8 - g14*y6 - g15*y3)
    gx13 = (-g0*y13 - g1*y10 - g2*y15 - g3*y7 + g4*y6 - g5*y14 - g6*y4 + g7*y3 - g8*y12 + g9*y11 + g10*y1 - g11*y9 + g12*y8 + g13*y0 + g14*y5 + g15*y2)
    gx14 = (g0*y14 - g1*y15 - g2*y10 + g3*y9 - g4*y8 + g5*y13 - g6*y12 + g7*y11 - g8*y4 + g9*y3 - g10*y2 + g11*y7 - g12*y6 + g13*y5 + g14*y0 - g15*y1)
    gx15 = (-g0*y15 + g1*y14 + g2*y13 - g3*y12 + g4*y11 - g5*y10 + g6*y9 - g7*y8 + g8*y7 - g9*y6 + g10*y5 - g11*y4 + g12*y3 - g13*y2 - g14*y1 + g15*y0)

    # grad_y: code-generated from Cayley table
    gy0 = (g0*x0 + g1*x1 + g2*x2 + g3*x3 + g4*x4 + g5*x5 + g6*x6 + g7*x7 + g8*x8 + g9*x9 + g10*x10 + g11*x11 + g12*x12 + g13*x13 + g14*x14 + g15*x15)
    gy1 = (g0*x1 + g1*x0 - g2*x5 - g3*x6 - g4*x7 - g5*x2 - g6*x3 - g7*x4 + g8*x11 + g9*x12 + g10*x13 + g11*x8 + g12*x9 + g13*x10 - g14*x15 - g15*x14)
    gy2 = (-g0*x2 - g1*x5 + g2*x0 + g3*x8 + g4*x9 + g5*x1 + g6*x11 + g7*x12 - g8*x3 - g9*x4 - g10*x14 - g11*x6 - g12*x7 - g13*x15 + g14*x10 + g15*x13)
    gy3 = (-g0*x3 - g1*x6 - g2*x8 + g3*x0 + g4*x10 - g5*x11 + g6*x1 + g7*x13 + g8*x2 + g9*x14 - g10*x4 + g11*x5 + g12*x15 - g13*x7 - g14*x9 - g15*x12)
    gy4 = (-g0*x4 - g1*x7 - g2*x9 - g3*x10 + g4*x0 - g5*x12 - g6*x13 + g7*x1 - g8*x14 + g9*x2 + g10*x3 - g11*x15 + g12*x5 + g13*x6 + g14*x8 + g15*x11)
    gy5 = (g0*x5 + g1*x2 + g2*x1 + g3*x11 + g4*x12 + g5*x0 + g6*x8 + g7*x9 + g8*x6 + g9*x7 + g10*x15 + g11*x3 + g12*x4 + g13*x14 + g14*x13 + g15*x10)
    gy6 = (g0*x6 + g1*x3 - g2*x11 + g3*x1 + g4*x13 - g5*x8 + g6*x0 + g7*x10 - g8*x5 - g9*x15 + g10*x7 - g11*x2 - g12*x14 + g13*x4 - g14*x12 - g15*x9)
    gy7 = (g0*x7 + g1*x4 - g2*x12 - g3*x13 + g4*x1 - g5*x9 - g6*x10 + g7*x0 + g8*x15 - g9*x5 - g10*x6 + g11*x14 - g12*x2 - g13*x3 + g14*x11 + g15*x8)
    gy8 = (-g0*x8 - g1*x11 + g2*x3 - g3*x2 - g4*x14 + g5*x6 - g6*x5 - g7*x15 + g8*x0 + g9*x10 - g10*x9 + g11*x1 + g12*x13 - g13*x12 + g14*x4 + g15*x7)
    gy9 = (-g0*x9 - g1*x12 + g2*x4 + g3*x14 - g4*x2 + g5*x7 + g6*x15 - g7*x5 - g8*x10 + g9*x0 + g10*x8 - g11*x13 + g12*x1 + g13*x11 - g14*x3 - g15*x6)
    gy10 = (-g0*x10 - g1*x13 - g2*x14 + g3*x4 - g4*x3 - g5*x15 + g6*x7 - g7*x6 + g8*x9 - g9*x8 + g10*x0 + g11*x12 - g12*x11 + g13*x1 + g14*x2 + g15*x5)
    gy11 = (-g0*x11 - g1*x8 - g2*x6 + g3*x5 + g4*x15 - g5*x3 + g6*x2 + g7*x14 + g8*x1 + g9*x13 - g10*x12 + g11*x0 + g12*x10 - g13*x9 - g14*x7 - g15*x4)
    gy12 = (-g0*x12 - g1*x9 - g2*x7 - g3*x15 + g4*x5 - g5*x4 - g6*x14 + g7*x2 - g8*x13 + g9*x1 + g10*x11 - g11*x10 + g12*x0 + g13*x8 + g14*x6 + g15*x3)
    gy13 = (-g0*x13 - g1*x10 + g2*x15 - g3*x7 + g4*x6 + g5*x14 - g6*x4 + g7*x3 + g8*x12 - g9*x11 + g10*x1 + g11*x9 - g12*x8 + g13*x0 - g14*x5 - g15*x2)
    gy14 = (g0*x14 + g1*x15 - g2*x10 + g3*x9 - g4*x8 - g5*x13 + g6*x12 - g7*x11 - g8*x4 + g9*x3 - g10*x2 - g11*x7 + g12*x6 - g13*x5 + g14*x0 + g15*x1)
    gy15 = (-g0*x15 - g1*x14 - g2*x13 + g3*x12 - g4*x11 - g5*x10 + g6*x9 - g7*x8 + g8*x7 - g9*x6 + g10*x5 + g11*x4 - g12*x3 + g13*x2 + g14*x1 + g15*x0)

    gxb = offs * stride_gxn
    tl.store(GRAD_X_ptr + gxb + 0  * stride_gx16, gx0,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 1  * stride_gx16, gx1,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 2  * stride_gx16, gx2,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 3  * stride_gx16, gx3,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 4  * stride_gx16, gx4,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 5  * stride_gx16, gx5,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 6  * stride_gx16, gx6,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 7  * stride_gx16, gx7,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 8  * stride_gx16, gx8,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 9  * stride_gx16, gx9,  mask=mask)
    tl.store(GRAD_X_ptr + gxb + 10 * stride_gx16, gx10, mask=mask)
    tl.store(GRAD_X_ptr + gxb + 11 * stride_gx16, gx11, mask=mask)
    tl.store(GRAD_X_ptr + gxb + 12 * stride_gx16, gx12, mask=mask)
    tl.store(GRAD_X_ptr + gxb + 13 * stride_gx16, gx13, mask=mask)
    tl.store(GRAD_X_ptr + gxb + 14 * stride_gx16, gx14, mask=mask)
    tl.store(GRAD_X_ptr + gxb + 15 * stride_gx16, gx15, mask=mask)
    gyb = offs * stride_gyn
    tl.store(GRAD_Y_ptr + gyb + 0  * stride_gy16, gy0,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 1  * stride_gy16, gy1,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 2  * stride_gy16, gy2,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 3  * stride_gy16, gy3,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 4  * stride_gy16, gy4,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 5  * stride_gy16, gy5,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 6  * stride_gy16, gy6,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 7  * stride_gy16, gy7,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 8  * stride_gy16, gy8,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 9  * stride_gy16, gy9,  mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 10 * stride_gy16, gy10, mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 11 * stride_gy16, gy11, mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 12 * stride_gy16, gy12, mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 13 * stride_gy16, gy13, mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 14 * stride_gy16, gy14, mask=mask)
    tl.store(GRAD_Y_ptr + gyb + 15 * stride_gy16, gy15, mask=mask)


def geometric_product_forward(x: torch.Tensor, y: torch.Tensor,
                               zero_bivector: bool = False) -> torch.Tensor:
    orig_shape = x.shape
    N = x[..., 0].numel()
    x_flat = x.reshape(N, 16).contiguous()
    y_flat = y.reshape(N, 16).contiguous()
    out = torch.empty_like(x_flat)
    BLOCK_N = min(256, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BLOCK_N),)
    _geometric_product_fwd_kernel[grid](
        x_flat, y_flat, out, N,
        x_flat.stride(0), x_flat.stride(1), y_flat.stride(0), y_flat.stride(1),
        out.stride(0), out.stride(1), ZERO_BIVECTOR=zero_bivector, BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


def geometric_product_backward(grad_output: torch.Tensor, x: torch.Tensor,
                                y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    N = x[..., 0].numel()
    go_flat = grad_output.reshape(N, 16).contiguous()
    x_flat = x.reshape(N, 16).contiguous()
    y_flat = y.reshape(N, 16).contiguous()
    grad_x = torch.empty_like(x_flat)
    grad_y = torch.empty_like(y_flat)
    BLOCK_N = min(256, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BLOCK_N),)
    _geometric_product_bwd_kernel[grid](
        go_flat, x_flat, y_flat, grad_x, grad_y, N,
        go_flat.stride(0), go_flat.stride(1), x_flat.stride(0), x_flat.stride(1),
        y_flat.stride(0), y_flat.stride(1), grad_x.stride(0), grad_x.stride(1),
        grad_y.stride(0), grad_y.stride(1), BLOCK_N=BLOCK_N,
    )
    return grad_x.reshape(orig_shape), grad_y.reshape(orig_shape)
