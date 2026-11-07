#pragma once

// GLM-5.3 KDA chronological recurrence, relay organisation.
//
// Written from the recurrence equations and the ordinary prepared-input
// contract (normalised BF16 Q/K, sigmoid BF16 beta, FP32 [head][key][value]
// state):
//     s[key]  *= decay[key]
//     recall   = sum_key s[key] * k[key]
//     delta    = beta * (v - recall)
//     s[key]  += k[key] * delta
//     out      = sum_key s[key] * q[key]
//
// Measured constraints on the target cards that shape the layout:
//   * a shared-memory read is cheap only when every lane of a warp reads the
//     same address, so lanes are value columns and key operands are staged
//     once per token and read uniformly;
//   * lane exchanges and per-lane-distinct shared or global accesses cost
//     several arithmetic instructions each, so hand-offs are kept to one
//     value per column wherever the summation order allows it;
//   * a warp running alone is limited by its operand reads, so the key axis
//     is divided between PARTS warps per 32-column group.
//
// A 32-column group is served by PARTS warps; warp `part` keeps keys
// [part*128/PARTS, (part+1)*128/PARTS) of its columns in registers for the
// whole launch. Per token:
//   pass 1   every warp decays its keys and forms its sixteen-term chains.
//            The first warp hands on one running sum, middle warps their
//            chain values.                                        -- barrier
//   close    the last warp of the group completes the ordered sum, forms
//            delta and publishes it; it also completes and stores the
//            previous token's readout. Meanwhile staging threads convert
//            the next token's operands into the alternate operand buffer
//            and request the raw operands of the token after it.  -- barrier
//   pass 2   every warp applies delta and forms its readout chains, handed
//            on in the same way for the following close step.
// Two block barriers per token; operand conversion happens once per block,
// and a block may hold one, two or all four column groups of a head.
//
// Summation order: sixteen-term chains over ascending keys, the eight chain
// results added in ascending order from zero. This is the order used by the
// existing register-state recurrence, so the two can be compared bit for bit.
//
// Launch: block (GROUPS * PARTS * 32) threads with GROUPS in {1, 2, 4};
// grid (heads, 4 / GROUPS); no dynamic shared memory.
extern "C" __global__ void glm53_kda_prepared_relay(
    float* __restrict__ state,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ gate_logits,
    const __nv_bfloat16* __restrict__ beta_values,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    int tokens, int heads, int dim,
    float gate_lower_bound, float q_scale, float norm_eps)
{
    constexpr int PARTS = 2;
    constexpr bool HOLD = true;
    enum { KEYS = 128 / (PARTS), CHAINS = 8 / (PARTS) };
    (void)norm_eps;
    const int warps = blockDim.x >> 5;
    const int groups = warps / (PARTS);
    if (dim != 128 || tokens <= 0 || (blockDim.x & 31) || warps != groups * (PARTS)
        || !(groups == 1 || groups == 2 || groups == 4)
        || blockIdx.x >= (unsigned)heads || (blockIdx.y + 1) * groups > 4) return;
    __shared__ float operand[2][3][128];
    __shared__ float recall_part[4][8][32], readout_part[4][8][32], delta_of[4][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int group = warp / (PARTS), part = warp % (PARTS);
    const bool closer = part == (PARTS) - 1;
    const int head = blockIdx.x;
    const int col = ((int)blockIdx.y * groups + group) * 32 + lane;
    const int first_key = part * KEYS;
    float* const column = state + (long long)head * 128 * 128 + col;
    float s[KEYS];
    _Pragma("unroll")
    for (int i = 0; i < KEYS; ++i) s[i] = column[(first_key + i) * 128];
    /* Staging duty is spread evenly: thread t converts keys t, t + blockDim, */
    /* ... below 128 (two keys per thread in the smallest block).           */
    const int stage_step = blockDim.x;
    const int stage_slot = tid;
    const float rate = __expf(A_log[head]);
    float stage_bias[2];
    __nv_bfloat16 raw_q[2], raw_k[2], raw_g[2], raw_v, raw_beta;
    _Pragma("unroll")
    for (int j = 0; j < 2; ++j) {
        const int key = stage_slot + j * stage_step;
        if (key < 128) {
            const long long row = (long long)head * 128;
            stage_bias[j] = dt_bias[head * 128 + key];
            const float gate = bf16_to_f32(gate_logits[row + key]) + stage_bias[j];
            operand[0][0][key] = __expf(gate_lower_bound / (1.0f + __expf(-rate * gate)));
            operand[0][1][key] = bf16_to_f32(k[row + key]);
            operand[0][2][key] = bf16_to_f32(q[row + key]) * q_scale;
            const long long next = ((long long)min(1, tokens - 1) * heads + head) * 128;
            raw_q[j] = q[next + key];
            raw_k[j] = k[next + key];
            raw_g[j] = gate_logits[next + key];
        }
    }
    if (closer) {
        raw_v = v[(long long)head * 128 + col];
        raw_beta = beta_values[head];
    }
    float held_readout[CHAINS];
    __syncthreads();
    for (int token = 0; token < tokens; ++token) {
        const float* const decay_of = operand[token & 1][0] + first_key;
        const float* const key_of = operand[token & 1][1] + first_key;
        const float* const query_of = operand[token & 1][2] + first_key;
        float held_key[(HOLD) ? KEYS : 1];
        float chain[CHAINS];
        _Pragma("unroll")
        for (int c = 0; c < CHAINS; ++c) chain[c] = 0.0f;
        _Pragma("unroll")
        for (int i = 0; i < 16; ++i) {
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c)
                s[c * 16 + i] = __fmul_rn(s[c * 16 + i], decay_of[c * 16 + i]);
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c) {
                const float key_value = key_of[c * 16 + i];
                if (HOLD) held_key[c * 16 + i] = key_value;
                chain[c] = __fmaf_rn(s[c * 16 + i], key_value, chain[c]);
            }
        }
        if (part == 0) {
            float running = 0.0f;
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c) running = __fadd_rn(running, chain[c]);
            recall_part[group][CHAINS - 1][lane] = running;
        } else if (!closer) {
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c)
                recall_part[group][part * CHAINS + c][lane] = chain[c];
        }
        __syncthreads();
        float delta = 0.0f;
        if (closer) {
            float recalled = recall_part[group][CHAINS - 1][lane];
            _Pragma("unroll")
            for (int j = CHAINS; j < ((PARTS) - 1) * CHAINS; ++j)
                recalled = __fadd_rn(recalled, recall_part[group][j][lane]);
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c) recalled = __fadd_rn(recalled, chain[c]);
            delta = __fmul_rn(bf16_to_f32(raw_beta),
                              __fsub_rn(bf16_to_f32(raw_v), recalled));
            delta_of[group][lane] = delta;
            if (token > 0) {
                float total = readout_part[group][CHAINS - 1][lane];
                _Pragma("unroll")
                for (int j = CHAINS; j < ((PARTS) - 1) * CHAINS; ++j)
                    total = __fadd_rn(total, readout_part[group][j][lane]);
                _Pragma("unroll")
                for (int c = 0; c < CHAINS; ++c) total = __fadd_rn(total, held_readout[c]);
                output[((long long)(token - 1) * heads + head) * 128 + col] = f32_to_bf16(total);
            }
            const int ahead = min(token + 1, tokens - 1);
            raw_v = v[((long long)ahead * heads + head) * 128 + col];
            raw_beta = beta_values[(long long)ahead * heads + head];
        }
        {
            float (* const next_operand)[128] = operand[(token + 1) & 1];
            const long long row = ((long long)min(token + 2, tokens - 1) * heads + head) * 128;
            _Pragma("unroll")
            for (int j = 0; j < 2; ++j) {
                const int key = stage_slot + j * stage_step;
                if (key < 128) {
                    const float gate = bf16_to_f32(raw_g[j]) + stage_bias[j];
                    next_operand[0][key] = __expf(gate_lower_bound / (1.0f + __expf(-rate * gate)));
                    next_operand[1][key] = bf16_to_f32(raw_k[j]);
                    next_operand[2][key] = bf16_to_f32(raw_q[j]) * q_scale;
                    raw_q[j] = q[row + key];
                    raw_k[j] = k[row + key];
                    raw_g[j] = gate_logits[row + key];
                }
            }
        }
        __syncthreads();
        if (!closer) delta = delta_of[group][lane];
        _Pragma("unroll")
        for (int c = 0; c < CHAINS; ++c) chain[c] = 0.0f;
        _Pragma("unroll")
        for (int i = 0; i < 16; ++i) {
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c)
                s[c * 16 + i] = __fmaf_rn((HOLD) ? held_key[c * 16 + i] : key_of[c * 16 + i],
                                          delta, s[c * 16 + i]);
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c)
                chain[c] = __fmaf_rn(s[c * 16 + i], query_of[c * 16 + i], chain[c]);
        }
        if (part == 0) {
            float running = 0.0f;
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c) running = __fadd_rn(running, chain[c]);
            readout_part[group][CHAINS - 1][lane] = running;
        } else if (!closer) {
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c)
                readout_part[group][part * CHAINS + c][lane] = chain[c];
        } else {
            _Pragma("unroll")
            for (int c = 0; c < CHAINS; ++c) held_readout[c] = chain[c];
        }
    }
    __syncthreads();
    if (closer) {
        float total = readout_part[group][CHAINS - 1][lane];
        _Pragma("unroll")
        for (int j = CHAINS; j < ((PARTS) - 1) * CHAINS; ++j)
            total = __fadd_rn(total, readout_part[group][j][lane]);
        _Pragma("unroll")
        for (int c = 0; c < CHAINS; ++c) total = __fadd_rn(total, held_readout[c]);
        output[((long long)(tokens - 1) * heads + head) * 128 + col] = f32_to_bf16(total);
    }
    _Pragma("unroll")
    for (int i = 0; i < KEYS; ++i) column[(first_key + i) * 128] = s[i];
}
