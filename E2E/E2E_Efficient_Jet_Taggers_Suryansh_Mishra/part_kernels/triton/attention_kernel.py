"""
Raw Triton JIT kernels for fused attention with additive pair-bias.

Forward: online softmax (FlashAttention-style) avoiding full (P,P) alloc.
Backward: recomputes attention weights from saved LSE; uses tl.atomic_add
for dK/dV accumulation across query tiles.

Optimised for ParT's regime:  P <= 128, D = 16, H = 8.
"""
import triton
import triton.language as tl


@triton.jit
def _fused_attn_fwd(
    Q_ptr, K_ptr, V_ptr, Bias_ptr, Pad_ptr, Out_ptr, LSE_ptr,
    stride_qb, stride_qp, stride_qd,
    stride_kb, stride_kp, stride_kd,
    stride_vb, stride_vp, stride_vd,
    stride_bb, stride_bi, stride_bj,
    stride_pb, stride_pp,
    stride_ob, stride_op, stride_od,
    stride_lb, stride_lp,
    sm_scale,
    P: tl.constexpr,
    D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_PAD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offs < P

    q_ptrs = Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qp + tl.arange(0, D)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    if HAS_PAD:
        batch_idx = pid_b // NUM_HEADS

    for start_n in range(0, P, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < P

        k_ptrs = K_ptr + pid_b * stride_kb + n_offs[:, None] * stride_kp + tl.arange(0, D)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            b_ptrs = Bias_ptr + pid_b * stride_bb + q_offs[:, None] * stride_bi + n_offs[None, :] * stride_bj
            bias = tl.load(b_ptrs, mask=q_mask[:, None] & n_mask[None, :], other=0.0)
            qk += bias

        if HAS_PAD:
            pad_ptrs = Pad_ptr + batch_idx * stride_pb + n_offs * stride_pp
            pad = tl.load(pad_ptrs, mask=n_mask, other=False)
            qk = tl.where(pad[None, :], float('-inf'), qk)

        qk = tl.where(q_mask[:, None] & n_mask[None, :], qk, float('-inf'))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + pid_b * stride_vb + n_offs[:, None] * stride_vp + tl.arange(0, D)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = Out_ptr + pid_b * stride_ob + q_offs[:, None] * stride_op + tl.arange(0, D)[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask[:, None])

    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE_ptr + pid_b * stride_lb + q_offs * stride_lp
    tl.store(lse_ptrs, lse, mask=q_mask)


@triton.jit
def _fused_attn_bwd(
    Q_ptr, K_ptr, V_ptr, Bias_ptr, Pad_ptr, Out_ptr, LSE_ptr, dOut_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dBias_ptr,
    stride_qb, stride_qp, stride_qd,
    stride_kb, stride_kp, stride_kd,
    stride_vb, stride_vp, stride_vd,
    stride_bb, stride_bi, stride_bj,
    stride_pb, stride_pp,
    stride_ob, stride_op, stride_od,
    stride_lb, stride_lp,
    sm_scale,
    P: tl.constexpr,
    D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_PAD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_DBIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offs < P

    base_q = pid_b * stride_qb
    base_k = pid_b * stride_kb
    base_v = pid_b * stride_vb

    d_range = tl.arange(0, D)
    q = tl.load(Q_ptr + base_q + q_offs[:, None] * stride_qp + d_range[None, :] * stride_qd, mask=q_mask[:, None], other=0.0)
    do = tl.load(dOut_ptr + pid_b * stride_ob + q_offs[:, None] * stride_op + d_range[None, :] * stride_od, mask=q_mask[:, None], other=0.0)
    o = tl.load(Out_ptr + pid_b * stride_ob + q_offs[:, None] * stride_op + d_range[None, :] * stride_od, mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE_ptr + pid_b * stride_lb + q_offs * stride_lp, mask=q_mask, other=0.0)

    D_i = tl.sum(do * o, axis=1)

    if HAS_PAD:
        batch_idx = pid_b // NUM_HEADS

    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, P, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < P

        k = tl.load(K_ptr + base_k + n_offs[:, None] * stride_kp + d_range[None, :] * stride_kd, mask=n_mask[:, None], other=0.0)
        v = tl.load(V_ptr + base_v + n_offs[:, None] * stride_vp + d_range[None, :] * stride_vd, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_BIAS:
            b_ptrs = Bias_ptr + pid_b * stride_bb + q_offs[:, None] * stride_bi + n_offs[None, :] * stride_bj
            bias = tl.load(b_ptrs, mask=q_mask[:, None] & n_mask[None, :], other=0.0)
            qk += bias
        if HAS_PAD:
            pad = tl.load(Pad_ptr + batch_idx * stride_pb + n_offs * stride_pp, mask=n_mask, other=False)
            qk = tl.where(pad[None, :], float('-inf'), qk)
        qk = tl.where(q_mask[:, None] & n_mask[None, :], qk, float('-inf'))

        p = tl.exp(qk - lse[:, None])

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - D_i[:, None]) * sm_scale

        dq_acc += tl.dot(ds.to(k.dtype), k)

        dk_chunk = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dk_ptrs = dK_ptr + pid_b * stride_kb + n_offs[:, None] * stride_kp + d_range[None, :] * stride_kd
        tl.atomic_add(dk_ptrs, dk_chunk, mask=n_mask[:, None])

        dv_chunk = tl.dot(tl.trans(p.to(do.dtype)), do)
        dv_ptrs = dV_ptr + pid_b * stride_vb + n_offs[:, None] * stride_vp + d_range[None, :] * stride_vd
        tl.atomic_add(dv_ptrs, dv_chunk, mask=n_mask[:, None])

        if NEED_DBIAS:
            db_ptrs = dBias_ptr + pid_b * stride_bb + q_offs[:, None] * stride_bi + n_offs[None, :] * stride_bj
            db_val = p * (dp - D_i[:, None])
            tl.store(db_ptrs, db_val, mask=q_mask[:, None] & n_mask[None, :])

    dq_ptrs = dQ_ptr + base_q + q_offs[:, None] * stride_qp + d_range[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc, mask=q_mask[:, None])
