# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Forward-only KDA subset vendored from flash-linear-attention commit
# bccaf2d3cf4d9badc8be050a2c71616220b246d7 under the repository MIT license.
# Krasis compiles these kernels into architecture-specific CUDA sidecars at
# build time; Python/Triton is never present in the model runtime hot path.

import triton
import triton.language as tl

from fla.ops.utils.op import exp, exp2, gather

SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")

@triton.jit
def softplus(x):
    return tl.log(1.0 + tl.exp(x))

@triton.jit(do_not_specialize=['T'])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    p_s = s + (bos * H + i_h) * S + o_t[:, None] * (H*S) + o_s[None, :]
    p_o = o + (bos * H + i_h) * S + o_t[:, None] * (H*S) + o_s[None, :]
    # [BT, BS]
    b_s = tl.load(p_s, mask=m_s, other=0.0).to(tl.float32)

    # Apply dt_bias if exists
    if HAS_BIAS:
        b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32) if HAS_A else 1.0
    if not USE_LOWER_BOUND:
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_s)

    # Apply chunk local cumsum
    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_s)


@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = m_c[:, None] & m_k[None, :]
    p_q = q + o_c[:, None] * (H*K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H*K) + o_k[None, :]
    p_g = g + o_c[:, None] * (HV*K) + o_k[None, :]

    p_beta = beta + o_c * HV

    b_q = tl.load(p_q, mask=m_ck, other=0.0)
    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_g = tl.load(p_g, mask=m_ck, other=0.0)
    b_beta = tl.load(p_beta, mask=m_c, other=0.0)

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        # caculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    m_Aqk_st = m_c[:, None] & (o_i[None, :] < BT)
    m_Akk_st = m_c[:, None] & (o_i[None, :] < BC)
    p_Aqk = Aqk + o_c[:, None] * (HV*BT) + (i_i * BC + o_i)[None, :]
    p_Akk = Akk + o_c[:, None] * (HV*BC) + o_i[None, :]
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=m_Aqk_st)
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), mask=m_Akk_st)

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * HV*BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk_st)

@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc1 + o_i
    o_c2 = i_tc2 + o_i
    o_c3 = i_tc3 + o_i
    m_tc0 = o_c0 < T
    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        m_ck0 = m_tc0[:, None] & m_k[None, :]
        p_k0 = k + o_c0[:, None] * (H*K) + o_k[None, :]
        p_g0 = g + o_c0[:, None] * (HV*K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_ck0, other=0.0).to(tl.float32)
        b_g0 = tl.load(p_g0, mask=m_ck0, other=0.0).to(tl.float32)

        if i_tc1 < T:
            m_ck1 = m_tc1[:, None] & m_k[None, :]
            p_q1 = q + o_c1[:, None] * (H*K) + o_k[None, :]
            p_k1 = k + o_c1[:, None] * (H*K) + o_k[None, :]
            p_g1 = g + o_c1[:, None] * (HV*K) + o_k[None, :]
            # [BC, BK]
            b_q1 = tl.load(p_q1, mask=m_ck1, other=0.0).to(tl.float32)
            b_k1 = tl.load(p_k1, mask=m_ck1, other=0.0).to(tl.float32)
            b_g1 = tl.load(p_g1, mask=m_ck1, other=0.0).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if NC >= 3 and i_tc2 < T:
                m_ck2 = m_tc2[:, None] & m_k[None, :]
                p_q2 = q + o_c2[:, None] * (H*K) + o_k[None, :]
                p_k2 = k + o_c2[:, None] * (H*K) + o_k[None, :]
                p_g2 = g + o_c2[:, None] * (HV*K) + o_k[None, :]
                # [BC, BK]
                b_q2 = tl.load(p_q2, mask=m_ck2, other=0.0).to(tl.float32)
                b_k2 = tl.load(p_k2, mask=m_ck2, other=0.0).to(tl.float32)
                b_g2 = tl.load(p_g2, mask=m_ck2, other=0.0).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if NC >= 4 and i_tc3 < T:
                    m_ck3 = m_tc3[:, None] & m_k[None, :]
                    p_q3 = q + o_c3[:, None] * (H*K) + o_k[None, :]
                    p_k3 = k + o_c3[:, None] * (H*K) + o_k[None, :]
                    p_g3 = g + o_c3[:, None] * (HV*K) + o_k[None, :]
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_k3 = tl.load(p_k3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_g3 = tl.load(p_g3, mask=m_ck3, other=0.0).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = Aqk + o_c1[:, None] * (HV*BT) + o_i[None, :]
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_A1)

        p_b1 = beta + bos * HV + i_hv + o_c1 * HV
        b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3 and i_tc2 < T:
        p_Aqk20 = Aqk + o_c2[:, None] * (HV*BT) + o_i[None, :]
        p_Aqk21 = Aqk + o_c2[:, None] * (HV*BT) + (o_i + BC)[None, :]
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), mask=m_A2)
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), mask=m_A2)

        p_b2 = beta + bos * HV + i_hv + o_c2 * HV
        b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if NC >= 4 and i_tc3 < T:
        p_Aqk30 = Aqk + o_c3[:, None] * (HV*BT) + o_i[None, :]
        p_Aqk31 = Aqk + o_c3[:, None] * (HV*BT) + (o_i + BC)[None, :]
        p_Aqk32 = Aqk + o_c3[:, None] * (HV*BT) + (o_i + 2*BC)[None, :]
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), mask=m_A3)

        p_b3 = beta + bos * HV + i_hv + o_c3 * HV
        b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    p_Akk00 = Akkd + o_c0[:, None] * (HV*BC) + o_i[None, :]
    p_Akk11 = Akkd + o_c1[:, None] * (HV*BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_A0, other=0.0).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, mask=m_A1, other=0.0).to(tl.float32)
    if NC >= 3:
        p_Akk22 = Akkd + o_c2[:, None] * (HV*BC) + o_i[None, :]
        b_Ai22 = tl.load(p_Akk22, mask=m_A2, other=0.0).to(tl.float32)
    if NC >= 4:
        p_Akk33 = Akkd + o_c3[:, None] * (HV*BC) + o_i[None, :]
        b_Ai33 = tl.load(p_Akk33, mask=m_A3, other=0.0).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    ################################################################################

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        if NC >= 3:
            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        if NC >= 4:
            b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2*BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NC >= 3:
            for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
                b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
                b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
                b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
        if NC >= 4:
            for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
                b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
                b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
                b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        if NC >= 3:
            b_Ai22 += m_I
        if NC >= 4:
            b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai11,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai22,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )

    ################################################################################
    # store full Akk_inv to Akk
    ################################################################################

    p_Akk00 = Akk + o_c0[:, None] * (HV*BT) + o_i[None, :]
    p_Akk10 = Akk + o_c1[:, None] * (HV*BT) + o_i[None, :]
    p_Akk11 = Akk + o_c1[:, None] * (HV*BT) + (o_i + BC)[None, :]

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), mask=m_A0)
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), mask=m_A1)
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), mask=m_A1)
    if NC >= 3:
        p_Akk20 = Akk + o_c2[:, None] * (HV*BT) + o_i[None, :]
        p_Akk21 = Akk + o_c2[:, None] * (HV*BT) + (o_i + BC)[None, :]
        p_Akk22 = Akk + o_c2[:, None] * (HV*BT) + (o_i + 2*BC)[None, :]
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), mask=m_A2)
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), mask=m_A2)
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), mask=m_A2)
    if NC >= 4:
        p_Akk30 = Akk + o_c3[:, None] * (HV*BT) + o_i[None, :]
        p_Akk31 = Akk + o_c3[:, None] * (HV*BT) + (o_i + BC)[None, :]
        p_Akk32 = Akk + o_c3[:, None] * (HV*BT) + (o_i + 2*BC)[None, :]
        p_Akk33 = Akk + o_c3[:, None] * (HV*BT) + (o_i + 3*BC)[None, :]
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), mask=m_A3)

@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kda_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = beta + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        p_u = u + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_w = w + o_t[:, None] * (HV*K) + o_k[None, :]
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kb = b_k * b_b[:, None]

        p_gk = gk + o_t[:, None] * (HV*K) + o_k[None, :]
        b_gk = tl.load(p_gk, mask=m_tk, other=0.0).to(tl.float32)
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
            p_qg = qg + o_t[:, None] * (HV*K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_tk, other=0.0)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), mask=m_tk)
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            b_gn = tl.load(gk + last_idx * HV*K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = kg + o_t[:, None] * (HV*K) + o_k[None, :]
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_tk)

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_tk)

@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV*K) + o_k[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        # [BT, BK]
        b_g = tl.load(p_g, mask=m_qk, other=0.0).to(tl.float32)
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        if i_k >= 0:
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV*V) + o_v[None, :]
    p_A = A + o_t[:, None] * (HV*BT) + o_i[None, :]
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    # [BT, BT]
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)
