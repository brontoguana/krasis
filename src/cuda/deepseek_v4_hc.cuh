#pragma once

// Shared DeepSeek-V4 hyper-connection primitives used by both prefill and
// decode PTX modules. All geometry and iteration counts are runtime inputs.

__device__ __forceinline__ float dsv4_hc_bf16_to_f32(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 dsv4_hc_f32_to_bf16(float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ float dsv4_hc_sigmoid(float value) {
    return 1.0f / (1.0f + __expf(-value));
}

// Initialize one token's hyper-connection streams from its embedding. HC is
// depth-wise across transformer layers rather than recurrent across tokens, so
// every decode token starts from hc_mult identical copies of the embedding.
extern "C" __global__ void deepseek_v4_hc_replicate_kernel(
    __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ hidden,
    int hidden_size,
    int hc_mult)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)hidden_size * hc_mult;
    if (linear >= total || hidden_size <= 0 || hc_mult <= 0) return;
    state[linear] = hidden[linear % hidden_size];
}

// One block per token computes the inverse RMS of the complete [hc, hidden]
// residual state. Keeping only this scalar avoids a second full-size state
// buffer during prefill.
extern "C" __global__ void deepseek_v4_hc_inv_rms_kernel(
    float* __restrict__ inv_rms,
    const __nv_bfloat16* __restrict__ state,
    int hidden_size,
    int hc_mult,
    float rms_eps)
{
    int token = (int)blockIdx.x;
    int flat_size = hidden_size * hc_mult;
    const __nv_bfloat16* row = state + (int64_t)token * flat_size;
    float sum_sq = 0.0f;
    for (int i = (int)threadIdx.x; i < flat_size; i += (int)blockDim.x) {
        float value = dsv4_hc_bf16_to_f32(row[i]);
        sum_sq += value * value;
    }
    extern __shared__ float reduce[];
    reduce[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        inv_rms[token] = rsqrtf(reduce[0] / (float)flat_size + rms_eps);
    }
}

// Grid [mix_width, tokens]. Each block computes one FP32 projection output
// from the BF16 residual state normalized by the token's inverse RMS.
extern "C" __global__ void deepseek_v4_hc_project_kernel(
    float* __restrict__ mixes,
    const __nv_bfloat16* __restrict__ state,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int hidden_size,
    int hc_mult,
    int mix_width)
{
    int output = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (output >= mix_width) return;
    int flat_size = hidden_size * hc_mult;
    const __nv_bfloat16* row = state + (int64_t)token * flat_size;
    const float* weight_row = weight + (int64_t)output * flat_size;
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < flat_size; i += (int)blockDim.x) {
        sum += dsv4_hc_bf16_to_f32(row[i]) * weight_row[i];
    }
    extern __shared__ float reduce[];
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mixes[(int64_t)token * mix_width + output] = reduce[0] * inv_rms[token];
    }
}

// Convert a contiguous token tile of BF16 residual state to normalized FP32.
// The output is consumed directly by the prefill tensor-core GEMM. Runtime
// geometry controls both the tile and row width; no model-specific dimensions
// are compiled into this kernel.
extern "C" __global__ void deepseek_v4_hc_normalize_f32_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ state,
    const float* __restrict__ inv_rms,
    int64_t elements,
    int flat_size)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= elements || flat_size <= 0) return;
    int token = (int)(linear / flat_size);
    output[linear] = dsv4_hc_bf16_to_f32(state[linear]) * inv_rms[token];
}

// Produces pre/post weights and the balanced [dst, src] combination matrix.
// The matrix layout is dst + src*hc, matching the shipped reshape/view order.
extern "C" __global__ void deepseek_v4_hc_prepare_kernel(
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    int hc_mult,
    int sinkhorn_iters,
    float hc_eps)
{
    int token = (int)blockIdx.x;
    if (hc_mult <= 0 || sinkhorn_iters <= 0) return;
    int mix_width = (2 + hc_mult) * hc_mult;
    const float* row = mixes + (int64_t)token * mix_width;
    float* pre_row = pre + (int64_t)token * hc_mult;
    float* post_row = post + (int64_t)token * hc_mult;
    float* comb_row = comb + (int64_t)token * hc_mult * hc_mult;
    const int comb_offset = 2 * hc_mult;

    // A one-thread launch preserves the established control path exactly.
    if (blockDim.x == 1) {
        if (threadIdx.x != 0) return;
        for (int stream = 0; stream < hc_mult; ++stream) {
            pre_row[stream] = dsv4_hc_sigmoid(
                row[stream] * scale[0] + base[stream]) + hc_eps;
            post_row[stream] = 2.0f * dsv4_hc_sigmoid(
                row[hc_mult + stream] * scale[1] + base[hc_mult + stream]);
        }

        // Initial softmax is over dst for each src, followed by epsilon.
        for (int src = 0; src < hc_mult; ++src) {
            float max_value = -INFINITY;
            for (int dst = 0; dst < hc_mult; ++dst) {
                int idx = dst + src * hc_mult;
                float value = row[comb_offset + idx] * scale[2] + base[comb_offset + idx];
                comb_row[idx] = value;
                max_value = fmaxf(max_value, value);
            }
            float sum = 0.0f;
            for (int dst = 0; dst < hc_mult; ++dst) {
                int idx = dst + src * hc_mult;
                float value = expf(comb_row[idx] - max_value);
                comb_row[idx] = value;
                sum += value;
            }
            for (int dst = 0; dst < hc_mult; ++dst) {
                int idx = dst + src * hc_mult;
                comb_row[idx] = comb_row[idx] / sum + hc_eps;
            }
        }

        for (int dst = 0; dst < hc_mult; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < hc_mult; ++src) sum += comb_row[dst + src * hc_mult];
            float denom = sum + hc_eps;
            for (int src = 0; src < hc_mult; ++src) comb_row[dst + src * hc_mult] /= denom;
        }
        for (int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
            for (int src = 0; src < hc_mult; ++src) {
                float sum = 0.0f;
                for (int dst = 0; dst < hc_mult; ++dst) sum += comb_row[dst + src * hc_mult];
                float denom = sum + hc_eps;
                for (int dst = 0; dst < hc_mult; ++dst) comb_row[dst + src * hc_mult] /= denom;
            }
            for (int dst = 0; dst < hc_mult; ++dst) {
                float sum = 0.0f;
                for (int src = 0; src < hc_mult; ++src) sum += comb_row[dst + src * hc_mult];
                float denom = sum + hc_eps;
                for (int src = 0; src < hc_mult; ++src) comb_row[dst + src * hc_mult] /= denom;
            }
        }
        return;
    }

    // Candidate path: independent streams, source rows, and destination
    // columns execute concurrently. Each thread retains the exact scalar
    // summation order of the control for its assigned row/column, so the
    // result is bit-identical while remaining valid for arbitrary hc_mult.
    for (int stream = (int)threadIdx.x; stream < hc_mult; stream += (int)blockDim.x) {
        pre_row[stream] = dsv4_hc_sigmoid(
            row[stream] * scale[0] + base[stream]) + hc_eps;
        post_row[stream] = 2.0f * dsv4_hc_sigmoid(
            row[hc_mult + stream] * scale[1] + base[hc_mult + stream]);
    }

    for (int src = (int)threadIdx.x; src < hc_mult; src += (int)blockDim.x) {
        float max_value = -INFINITY;
        for (int dst = 0; dst < hc_mult; ++dst) {
            int idx = dst + src * hc_mult;
            float value = row[comb_offset + idx] * scale[2] + base[comb_offset + idx];
            comb_row[idx] = value;
            max_value = fmaxf(max_value, value);
        }
        float sum = 0.0f;
        for (int dst = 0; dst < hc_mult; ++dst) {
            int idx = dst + src * hc_mult;
            float value = expf(comb_row[idx] - max_value);
            comb_row[idx] = value;
            sum += value;
        }
        for (int dst = 0; dst < hc_mult; ++dst) {
            int idx = dst + src * hc_mult;
            comb_row[idx] = comb_row[idx] / sum + hc_eps;
        }
    }
    __syncthreads();

    for (int dst = (int)threadIdx.x; dst < hc_mult; dst += (int)blockDim.x) {
        float sum = 0.0f;
        for (int src = 0; src < hc_mult; ++src) sum += comb_row[dst + src * hc_mult];
        float denom = sum + hc_eps;
        for (int src = 0; src < hc_mult; ++src) comb_row[dst + src * hc_mult] /= denom;
    }
    __syncthreads();
    for (int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
        for (int src = (int)threadIdx.x; src < hc_mult; src += (int)blockDim.x) {
            float sum = 0.0f;
            for (int dst = 0; dst < hc_mult; ++dst) sum += comb_row[dst + src * hc_mult];
            float denom = sum + hc_eps;
            for (int dst = 0; dst < hc_mult; ++dst) comb_row[dst + src * hc_mult] /= denom;
        }
        __syncthreads();
        for (int dst = (int)threadIdx.x; dst < hc_mult; dst += (int)blockDim.x) {
            float sum = 0.0f;
            for (int src = 0; src < hc_mult; ++src) sum += comb_row[dst + src * hc_mult];
            float denom = sum + hc_eps;
            for (int src = 0; src < hc_mult; ++src) comb_row[dst + src * hc_mult] /= denom;
        }
        __syncthreads();
    }
}

extern "C" __global__ void deepseek_v4_hc_reduce_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ state,
    const float* __restrict__ pre,
    int hidden_size,
    int hc_mult)
{
    int token = (int)blockIdx.x;
    const __nv_bfloat16* state_row = state + (int64_t)token * hc_mult * hidden_size;
    const float* pre_row = pre + (int64_t)token * hc_mult;
    __nv_bfloat16* output_row = output + (int64_t)token * hidden_size;
    for (int hidden = (int)threadIdx.x; hidden < hidden_size; hidden += (int)blockDim.x) {
        float sum = 0.0f;
        for (int stream = 0; stream < hc_mult; ++stream) {
            sum += dsv4_hc_bf16_to_f32(state_row[(int64_t)stream * hidden_size + hidden])
                * pre_row[stream];
        }
        output_row[hidden] = dsv4_hc_f32_to_bf16(sum);
    }
}

extern "C" __global__ void deepseek_v4_hc_post_kernel(
    __nv_bfloat16* __restrict__ output_state,
    const __nv_bfloat16* __restrict__ sublayer,
    const __nv_bfloat16* __restrict__ residual_state,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    int hidden_size,
    int hc_mult)
{
    int token = (int)blockIdx.x;
    const __nv_bfloat16* sublayer_row = sublayer + (int64_t)token * hidden_size;
    const __nv_bfloat16* residual_row = residual_state + (int64_t)token * hc_mult * hidden_size;
    const float* post_row = post + (int64_t)token * hc_mult;
    const float* comb_row = comb + (int64_t)token * hc_mult * hc_mult;
    __nv_bfloat16* output_row = output_state + (int64_t)token * hc_mult * hidden_size;
    for (int hidden = (int)threadIdx.x; hidden < hidden_size; hidden += (int)blockDim.x) {
        float sublayer_value = dsv4_hc_bf16_to_f32(sublayer_row[hidden]);
        for (int dst = 0; dst < hc_mult; ++dst) {
            float value = sublayer_value * post_row[dst];
            for (int src = 0; src < hc_mult; ++src) {
                value += dsv4_hc_bf16_to_f32(
                    residual_row[(int64_t)src * hidden_size + hidden])
                    * comb_row[dst + src * hc_mult];
            }
            output_row[(int64_t)dst * hidden_size + hidden] = dsv4_hc_f32_to_bf16(value);
        }
    }
}

extern "C" __global__ void deepseek_v4_hc_head_prepare_kernel(
    float* __restrict__ pre,
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    int hc_mult,
    float hc_eps)
{
    int token = (int)blockIdx.x;
    const float* row = mixes + (int64_t)token * hc_mult;
    float* pre_row = pre + (int64_t)token * hc_mult;
    for (int stream = (int)threadIdx.x; stream < hc_mult; stream += (int)blockDim.x) {
        pre_row[stream] = dsv4_hc_sigmoid(
            row[stream] * scale[0] + base[stream]) + hc_eps;
    }
}
