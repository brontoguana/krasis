#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// Source-faithful DeepSeek-V4 compressor/indexer primitives shared by full-GPU
// prefill and Rust decode. All geometry is supplied at runtime.

__device__ __forceinline__ float dsv4_pow2_scale(float amax, float format_max) {
    return exp2f(ceilf(log2f(amax / format_max)));
}

__device__ __forceinline__ float dsv4_e2m1_round(float value) {
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float magnitude = fminf(fabsf(value), 6.0f);
    int best = 0;
    float best_distance = fabsf(magnitude - levels[0]);
#pragma unroll
    for (int code = 1; code < 8; ++code) {
        float distance = fabsf(magnitude - levels[code]);
        if (distance < best_distance ||
            (distance == best_distance && (code & 1) == 0 && (best & 1) != 0)) {
            best = code;
            best_distance = distance;
        }
    }
    return copysignf(levels[best], value);
}

__device__ __forceinline__ unsigned char dsv4_e2m1_code(float value) {
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float magnitude = fminf(fabsf(value), 6.0f);
    int best = 0;
    float best_distance = fabsf(magnitude - levels[0]);
#pragma unroll
    for (int code = 1; code < 8; ++code) {
        float distance = fabsf(magnitude - levels[code]);
        if (distance < best_distance ||
            (distance == best_distance && (code & 1) == 0 && (best & 1) != 0)) {
            best = code;
            best_distance = distance;
        }
    }
    return (unsigned char)(best | (signbit(value) ? 8 : 0));
}

__device__ __forceinline__ float dsv4_e2m1_from_code(unsigned char code) {
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float value = levels[code & 7];
    return (code & 8) != 0 ? -value : value;
}

__device__ __forceinline__ signed char dsv4_pow2_scale_exponent(float scale) {
    return (signed char)ilogbf(scale);
}

__device__ __forceinline__ float dsv4_pow2_scale_from_exponent(signed char exponent) {
    return ldexpf(1.0f, (int)exponent);
}

// Pool complete compression windows. FP32 WKV/WGate projections and FP32 APE
// enter the exact per-dimension softmax equation. With overlap enabled, the
// first half comes from the preceding window and the second half from the
// current window, matching Compressor.overlap_transform.
extern "C" __global__ void deepseek_v4_compressor_pool_prefill_kernel(
    float* __restrict__ output,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const float* __restrict__ ape,
    int tokens,
    int head_dim,
    int ratio,
    int overlap)
{
    int dim = (int)blockIdx.x;
    int group = (int)blockIdx.y;
    int groups = ratio > 0 ? tokens / ratio : 0;
    if (dim >= head_dim || group >= groups || ratio <= 0) return;
    int candidates = overlap ? 2 * ratio : ratio;
    if (candidates > (int)blockDim.x) return;
    int projected_dim = overlap ? 2 * head_dim : head_dim;
    extern __shared__ float shared[];
    float* scores = shared;
    float* values = shared + blockDim.x;
    float* work = shared + 2 * blockDim.x;
    int lane = (int)threadIdx.x;
    float candidate_score = -INFINITY;
    float candidate_value = 0.0f;
    if (lane < candidates) {
        int slot = overlap && lane >= ratio ? lane - ratio : lane;
        int token;
        int column;
        if (overlap && lane < ratio) {
            token = (group - 1) * ratio + slot;
            column = dim;
        } else {
            token = group * ratio + slot;
            column = overlap ? head_dim + dim : dim;
        }
        if (token >= 0 && token < tokens) {
            int64_t offset = (int64_t)token * projected_dim + column;
            int64_t ape_offset = (int64_t)slot * projected_dim + column;
            candidate_score = score[offset] + ape[ape_offset];
            candidate_value = kv[offset];
        }
    }
    scores[lane] = candidate_score;
    values[lane] = candidate_value;
    work[lane] = candidate_score;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) work[lane] = fmaxf(work[lane], work[lane + stride]);
        __syncthreads();
    }
    float maximum = work[0];
    float weight = lane < candidates && isfinite(scores[lane])
        ? expf(scores[lane] - maximum)
        : 0.0f;
    scores[lane] = weight;
    work[lane] = weight;
    values[lane] *= weight;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            work[lane] += work[lane + stride];
            values[lane] += values[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[(int64_t)group * head_dim + dim] = values[0] / work[0];
    }
}

// Preserve the exact incremental state that Compressor.forward would retain
// after a full-prompt prefill. This is separate from pooling so the prefill
// path can process all complete groups in parallel without changing decode
// continuation semantics.
extern "C" __global__ void deepseek_v4_compressor_state_prefill_kernel(
    float* __restrict__ kv_state,
    float* __restrict__ score_state,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const float* __restrict__ ape,
    int tokens,
    int head_dim,
    int ratio,
    int overlap)
{
    int copies = overlap ? 2 : 1;
    int projected_dim = copies * head_dim;
    int state_rows = copies * ratio;
    int64_t total = (int64_t)state_rows * projected_dim;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total || ratio <= 0 || head_dim <= 0) return;
    int row = (int)(linear / projected_dim);
    int column = (int)(linear % projected_dim);
    int cutoff = tokens - tokens % ratio;
    int remainder = tokens - cutoff;
    int token = -1;
    int ape_slot = 0;
    if (overlap) {
        if (row < ratio) {
            if (cutoff >= ratio) token = cutoff - ratio + row;
            ape_slot = row;
        } else {
            int remainder_row = row - ratio;
            if (remainder_row < remainder) token = cutoff + remainder_row;
            ape_slot = remainder_row;
        }
    } else {
        if (row < remainder) token = cutoff + row;
        ape_slot = row;
    }
    if (token >= 0) {
        int64_t source = (int64_t)token * projected_dim + column;
        kv_state[linear] = kv[source];
        score_state[linear] = score[source] + ape[(int64_t)ape_slot * projected_dim + column];
    } else {
        kv_state[linear] = 0.0f;
        score_state[linear] = -INFINITY;
    }
}

// Update one-token compressor state. On ratio boundaries this also emits the
// pooled FP32 row; callers then run cast-before-RMSNorm, RoPE and QAT. One
// block owns one output dimension, so its state update and read are ordered
// without a grid-wide synchronization.
extern "C" __global__ void deepseek_v4_compressor_decode_kernel(
    float* __restrict__ output,
    float* __restrict__ kv_state,
    float* __restrict__ score_state,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const float* __restrict__ ape,
    const int* __restrict__ position_ptr,
    int head_dim,
    int ratio,
    int overlap)
{
    int dim = (int)blockIdx.x;
    if (dim >= head_dim || ratio <= 0 || position_ptr == nullptr) return;
    int position = *position_ptr;
    if (position < 0) return;
    int slot = position % ratio;
    int copies = overlap ? 2 : 1;
    int projected_dim = copies * head_dim;
    int write_row = overlap ? ratio + slot : slot;
    if (threadIdx.x == 0) {
        for (int copy = 0; copy < copies; ++copy) {
            int column = copy * head_dim + dim;
            int64_t state_offset = (int64_t)write_row * projected_dim + column;
            kv_state[state_offset] = kv[column];
            score_state[state_offset] = score[column] + ape[(int64_t)slot * projected_dim + column];
        }
    }
    __syncthreads();
    if ((position + 1) % ratio != 0) return;
    int candidates = copies * ratio;
    if (candidates > (int)blockDim.x) return;
    extern __shared__ float shared[];
    float* scores = shared;
    float* values = shared + blockDim.x;
    float* work = shared + 2 * blockDim.x;
    int lane = (int)threadIdx.x;
    float candidate_score = -INFINITY;
    float candidate_value = 0.0f;
    if (lane < candidates) {
        int state_row = lane;
        int column = overlap && lane >= ratio ? head_dim + dim : dim;
        int64_t state_offset = (int64_t)state_row * projected_dim + column;
        candidate_score = score_state[state_offset];
        candidate_value = kv_state[state_offset];
    }
    scores[lane] = candidate_score;
    values[lane] = candidate_value;
    work[lane] = candidate_score;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) work[lane] = fmaxf(work[lane], work[lane + stride]);
        __syncthreads();
    }
    float maximum = work[0];
    float weight = lane < candidates && isfinite(scores[lane])
        ? expf(scores[lane] - maximum)
        : 0.0f;
    work[lane] = weight;
    values[lane] *= weight;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            work[lane] += work[lane + stride];
            values[lane] += values[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) output[dim] = values[0] / work[0];
    __syncthreads();
    if (overlap && lane < ratio) {
        for (int copy = 0; copy < copies; ++copy) {
            int column = copy * head_dim + dim;
            int64_t destination = (int64_t)lane * projected_dim + column;
            int64_t source = (int64_t)(ratio + lane) * projected_dim + column;
            kv_state[destination] = kv_state[source];
            score_state[destination] = score_state[source];
        }
    }
}

// Graph-addressable learned-index score kernel. A fixed occupancy-sized grid
// walks only the live compressed prefix derived on GPU from the decode
// position. Each warp computes one head dot product; one key row is shared by
// all heads in the block. The query and key have already undergone adjacent
// tail RoPE, normalized Hadamard rotation, and FP4 QAT.
extern "C" __global__ void deepseek_v4_index_scores_decode_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ head_weights,
    const int* __restrict__ compressed_count,
    int score_capacity,
    int num_heads,
    int head_dim,
    int clear_inactive_tail)
{
    if (compressed_count == nullptr || score_capacity <= 0 || num_heads <= 0 ||
        head_dim <= 0 || (blockDim.x & 31) != 0) return;
    int context = min(max(*compressed_count, 0), score_capacity);
    int num_warps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    extern __shared__ float shared[];
    float* shared_key = shared;
    float* contributions = shared_key + head_dim;

    for (int token = (int)blockIdx.x; token < context; token += (int)gridDim.x) {
        for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
            shared_key[dim] = __bfloat162float(
                key_cache[(int64_t)token * head_dim + dim]);
        }
        __syncthreads();
        for (int head = warp; head < num_heads; head += num_warps) {
            const __nv_bfloat16* head_query = query + (int64_t)head * head_dim;
            float dot = 0.0f;
            for (int dim = lane; dim < head_dim; dim += 32) {
                dot += __bfloat162float(head_query[dim]) * shared_key[dim];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                contributions[head] = __bfloat162float(head_weights[head]) * fmaxf(dot, 0.0f);
            }
        }
        __syncthreads();
        if (warp == 0) {
            float score = 0.0f;
            for (int head = lane; head < num_heads; head += 32) score += contributions[head];
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
            if (lane == 0) output[token] = score;
        }
        __syncthreads();
    }
    if (clear_inactive_tail) {
        for (int token = context + (int)blockIdx.x; token < score_capacity;
             token += (int)gridDim.x) {
            output[token] = -INFINITY;
        }
    }
}

// Native-cache equivalent of the learned-index scorer. The persistent cache
// stores the model's own E2M1 QAT codes and power-of-two scale exponents; this
// reconstructs the same BF16-expanded values consumed by the reference path
// without retaining a second expanded cache.
extern "C" __global__ void deepseek_v4_index_scores_native_decode_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ key_codes,
    const signed char* __restrict__ key_scale_exponents,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ head_weights,
    const int* __restrict__ compressed_count,
    int score_capacity,
    int num_heads,
    int head_dim,
    int block_size,
    int clear_inactive_tail)
{
    if (compressed_count == nullptr || key_codes == nullptr ||
        key_scale_exponents == nullptr || score_capacity <= 0 || num_heads <= 0 ||
        head_dim <= 0 || block_size <= 0 || (blockDim.x & 31) != 0) return;
    int context = min(max(*compressed_count, 0), score_capacity);
    int num_warps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int code_stride = (head_dim + 1) >> 1;
    int scale_stride = (head_dim + block_size - 1) / block_size;
    extern __shared__ float shared[];
    float* shared_key = shared;
    float* contributions = shared_key + head_dim;

    for (int token = (int)blockIdx.x; token < context; token += (int)gridDim.x) {
        for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
            unsigned char packed = key_codes[(int64_t)token * code_stride + (dim >> 1)];
            unsigned char code = (dim & 1) == 0 ? packed & 0x0f : packed >> 4;
            float scale = dsv4_pow2_scale_from_exponent(
                key_scale_exponents[(int64_t)token * scale_stride + dim / block_size]);
            shared_key[dim] = dsv4_e2m1_from_code(code) * scale;
        }
        __syncthreads();
        for (int head = warp; head < num_heads; head += num_warps) {
            const __nv_bfloat16* head_query = query + (int64_t)head * head_dim;
            float dot = 0.0f;
            for (int dim = lane; dim < head_dim; dim += 32) {
                dot += __bfloat162float(head_query[dim]) * shared_key[dim];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                contributions[head] =
                    __bfloat162float(head_weights[head]) * fmaxf(dot, 0.0f);
            }
        }
        __syncthreads();
        if (warp == 0) {
            float score = 0.0f;
            for (int head = lane; head < num_heads; head += 32) {
                score += contributions[head];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
            if (lane == 0) output[token] = score;
        }
        __syncthreads();
    }
    if (clear_inactive_tail) {
        for (int token = context + (int)blockIdx.x; token < score_capacity;
             token += (int)gridDim.x) {
            output[token] = -INFINITY;
        }
    }
}

// Continue compressor prefill from an arbitrary absolute prompt position.
// One block owns one output dimension and walks the chunk in order, so state
// updates and ratio-boundary pooling are ordered without host or grid-wide
// synchronization. `output` is local to this chunk: row zero corresponds to
// `first_output_group` in the persistent compressed cache.
extern "C" __global__ void deepseek_v4_compressor_continue_prefill_kernel(
    float* __restrict__ output,
    float* __restrict__ kv_state,
    float* __restrict__ score_state,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const float* __restrict__ ape,
    int start_pos,
    int tokens,
    int head_dim,
    int ratio,
    int overlap,
    int first_output_group)
{
    int dim = (int)blockIdx.x;
    if (dim >= head_dim || ratio <= 0 || tokens <= 0) return;
    int copies = overlap ? 2 : 1;
    int projected_dim = copies * head_dim;
    int candidates = copies * ratio;
    if (candidates > (int)blockDim.x) return;
    extern __shared__ float shared[];
    float* weights = shared;
    float* values = shared + blockDim.x;
    float* work = shared + 2 * blockDim.x;
    int lane = (int)threadIdx.x;

    for (int local = 0; local < tokens; ++local) {
        int position = start_pos + local;
        int slot = position % ratio;
        int write_row = overlap ? ratio + slot : slot;
        if (lane == 0) {
            for (int copy = 0; copy < copies; ++copy) {
                int column = copy * head_dim + dim;
                int64_t source = (int64_t)local * projected_dim + column;
                int64_t state_offset = (int64_t)write_row * projected_dim + column;
                kv_state[state_offset] = kv[source];
                score_state[state_offset] =
                    score[source] + ape[(int64_t)slot * projected_dim + column];
            }
        }
        __syncthreads();
        if ((position + 1) % ratio == 0) {
            float candidate_score = -INFINITY;
            float candidate_value = 0.0f;
            if (lane < candidates) {
                int column = overlap && lane >= ratio ? head_dim + dim : dim;
                int64_t state_offset = (int64_t)lane * projected_dim + column;
                candidate_score = score_state[state_offset];
                candidate_value = kv_state[state_offset];
            }
            weights[lane] = candidate_score;
            values[lane] = candidate_value;
            work[lane] = candidate_score;
            __syncthreads();
            for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
                if (lane < stride) work[lane] = fmaxf(work[lane], work[lane + stride]);
                __syncthreads();
            }
            float maximum = work[0];
            float weight = lane < candidates && isfinite(weights[lane])
                ? expf(weights[lane] - maximum)
                : 0.0f;
            work[lane] = weight;
            values[lane] *= weight;
            __syncthreads();
            for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    work[lane] += work[lane + stride];
                    values[lane] += values[lane + stride];
                }
                __syncthreads();
            }
            if (lane == 0) {
                int output_group = position / ratio;
                int output_row = output_group - first_output_group;
                output[(int64_t)output_row * head_dim + dim] = values[0] / work[0];
            }
            __syncthreads();
            if (overlap && lane < ratio) {
                for (int copy = 0; copy < copies; ++copy) {
                    int column = copy * head_dim + dim;
                    int64_t destination = (int64_t)lane * projected_dim + column;
                    int64_t source = (int64_t)(ratio + lane) * projected_dim + column;
                    kv_state[destination] = kv_state[source];
                    score_state[destination] = score_state[source];
                }
            }
        }
        __syncthreads();
    }
}

// Cast pooled FP32 to BF16, then apply the checkpoint BF16-weighted RMSNorm.
// The cast-before-normalize ordering is required by Compressor.forward.
extern "C" __global__ void deepseek_v4_compressor_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int rows,
    int width,
    float eps)
{
    int row = (int)blockIdx.x;
    if (row >= rows || width <= 0) return;
    extern __shared__ float reduction[];
    int lane = (int)threadIdx.x;
    float sumsq = 0.0f;
    for (int dim = lane; dim < width; dim += (int)blockDim.x) {
        float value = __bfloat162float(__float2bfloat16(input[(int64_t)row * width + dim]));
        sumsq += value * value;
    }
    reduction[lane] = sumsq;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inv = rsqrtf(reduction[0] / (float)width + eps);
    for (int dim = lane; dim < width; dim += (int)blockDim.x) {
        float value = __bfloat162float(__float2bfloat16(input[(int64_t)row * width + dim]));
        output[(int64_t)row * width + dim] = __float2bfloat16(
            value * inv * __bfloat162float(weight[dim]));
    }
}

// Graph-safe one-token compressor finalization. The decode graph launches this
// every token, but it writes a cache row only when the current token closes a
// compression window. Keeping the boundary test on-device preserves a fixed
// launch graph while retaining the exact cast -> RMSNorm -> RoPE -> QAT order
// used by the shipped incremental implementation.
extern "C" __global__ void deepseek_v4_compressor_finalize_decode_kernel(
    __nv_bfloat16* __restrict__ cache,
    const float* __restrict__ pooled,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ position_ptr,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int cache_rows,
    int head_dim,
    int rope_dim,
    int rope_rows,
    int ratio,
    int indexer_mode,
    int qat_block_size,
    float eps)
{
    if (blockIdx.x != 0 || position_ptr == nullptr || cache == nullptr || pooled == nullptr ||
        norm_weight == nullptr || cos_table == nullptr || sin_table == nullptr ||
        cache_rows <= 0 || head_dim <= 0 || head_dim > (int)blockDim.x ||
        rope_dim <= 0 || rope_dim > head_dim || (rope_dim & 1) != 0 ||
        rope_rows <= 0 || ratio <= 0 || qat_block_size <= 0) return;
    int position = *position_ptr;
    if (position < 0 || (position + 1) % ratio != 0) return;
    int cache_row = (position + 1) / ratio - 1;
    if (cache_row < 0 || cache_row >= cache_rows) return;
    int compressed_position = cache_row * ratio;
    if (compressed_position < 0 || compressed_position >= rope_rows) return;

    extern __shared__ float shared[];
    int lane = (int)threadIdx.x;
    __nv_bfloat16* output = cache + (int64_t)cache_row * head_dim;

    float cast_value = lane < head_dim
        ? __bfloat162float(__float2bfloat16(pooled[lane]))
        : 0.0f;
    shared[lane] = cast_value * cast_value;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        __syncthreads();
    }
    float inv = rsqrtf(shared[0] / (float)head_dim + eps);
    if (lane < head_dim) {
        output[lane] = __float2bfloat16(
            cast_value * inv * __bfloat162float(norm_weight[lane]));
    }
    __syncthreads();

    int half_rope = rope_dim / 2;
    int tail = head_dim - rope_dim;
    if (lane < half_rope) {
        int real_index = tail + 2 * lane;
        float real = __bfloat162float(output[real_index]);
        float imag = __bfloat162float(output[real_index + 1]);
        float cosine = cos_table[(int64_t)compressed_position * half_rope + lane];
        float sine = sin_table[(int64_t)compressed_position * half_rope + lane];
        output[real_index] = __float2bfloat16(real * cosine - imag * sine);
        output[real_index + 1] = __float2bfloat16(imag * cosine + real * sine);
    }
    __syncthreads();

    if (indexer_mode != 0) {
        if ((head_dim & (head_dim - 1)) != 0) return;
        if (lane < head_dim) shared[lane] = __bfloat162float(output[lane]);
        __syncthreads();
        for (int stride = 1; stride < head_dim; stride <<= 1) {
            if (lane < head_dim / 2) {
                int base = (lane / stride) * (stride << 1);
                int offset = lane - (lane / stride) * stride;
                float left = shared[base + offset];
                float right = shared[base + offset + stride];
                shared[base + offset] = left + right;
                shared[base + offset + stride] = left - right;
            }
            __syncthreads();
        }
        if (lane < head_dim) {
            output[lane] = __float2bfloat16(shared[lane] * rsqrtf((float)head_dim));
        }
        __syncthreads();
    }

    int quant_cols = indexer_mode != 0 ? head_dim : head_dim - rope_dim;
    for (int block_start = 0; block_start < quant_cols; block_start += qat_block_size) {
        float amax = 0.0f;
        int column = block_start + lane;
        if (lane < qat_block_size && column < quant_cols) {
            amax = fabsf(__bfloat162float(output[column]));
        }
        shared[lane] = amax;
        __syncthreads();
        for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
            __syncthreads();
        }
        if (lane < qat_block_size && column < quant_cols) {
            float value = __bfloat162float(output[column]);
            if (indexer_mode != 0) {
                float scale = dsv4_pow2_scale(fmaxf(shared[0], 6.0f * 0x1p-126f), 6.0f);
                float scaled = fminf(fmaxf(value / scale, -6.0f), 6.0f);
                output[column] = __float2bfloat16(dsv4_e2m1_round(scaled) * scale);
            } else {
                float scale = dsv4_pow2_scale(fmaxf(shared[0], 1.0e-4f), 448.0f);
                float scaled = fminf(fmaxf(value / scale, -448.0f), 448.0f);
                __nv_fp8_e4m3 quantized(scaled);
                output[column] = __float2bfloat16((float)quantized * scale);
            }
        }
        __syncthreads();
    }
}

// Graph-safe native-cache finalizer. It preserves the reference operation and
// rounding order (FP32 pool -> BF16 cast -> RMSNorm -> BF16 -> RoPE -> BF16 ->
// optional Hadamard -> BF16 -> QAT) but publishes the final QAT codes and exact
// power-of-two scale exponents instead of an expanded BF16 cache row.
extern "C" __global__ void deepseek_v4_compressor_finalize_native_decode_kernel(
    unsigned char* __restrict__ codes,
    signed char* __restrict__ scale_exponents,
    __nv_bfloat16* __restrict__ tails,
    const float* __restrict__ pooled,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ position_ptr,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int cache_rows,
    int head_dim,
    int rope_dim,
    int rope_rows,
    int ratio,
    int code_bits,
    int qat_block_size,
    float eps)
{
    if (blockIdx.x != 0 || position_ptr == nullptr || codes == nullptr ||
        scale_exponents == nullptr || pooled == nullptr || norm_weight == nullptr ||
        cos_table == nullptr || sin_table == nullptr || cache_rows <= 0 ||
        head_dim <= 0 || head_dim > (int)blockDim.x || rope_dim <= 0 ||
        rope_dim > head_dim || (rope_dim & 1) != 0 || rope_rows <= 0 ||
        ratio <= 0 || qat_block_size <= 0 || !((code_bits == 8 && tails != nullptr) ||
        (code_bits == 4 && tails == nullptr))) return;
    int position = *position_ptr;
    if (position < 0 || (position + 1) % ratio != 0) return;
    int cache_row = (position + 1) / ratio - 1;
    if (cache_row < 0 || cache_row >= cache_rows) return;
    int compressed_position = cache_row * ratio;
    if (compressed_position < 0 || compressed_position >= rope_rows) return;

    extern __shared__ float shared[];
    float* reduction = shared;
    float* values = shared + blockDim.x;
    int lane = (int)threadIdx.x;
    float cast_value = lane < head_dim
        ? __bfloat162float(__float2bfloat16(pooled[lane]))
        : 0.0f;
    values[lane] = cast_value;
    reduction[lane] = cast_value * cast_value;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inv = rsqrtf(reduction[0] / (float)head_dim + eps);
    if (lane < head_dim) {
        values[lane] = __bfloat162float(__float2bfloat16(
            values[lane] * inv * __bfloat162float(norm_weight[lane])));
    }
    __syncthreads();

    int half_rope = rope_dim / 2;
    int tail_start = head_dim - rope_dim;
    if (lane < half_rope) {
        int real_index = tail_start + 2 * lane;
        float real = values[real_index];
        float imag = values[real_index + 1];
        float cosine = cos_table[(int64_t)compressed_position * half_rope + lane];
        float sine = sin_table[(int64_t)compressed_position * half_rope + lane];
        values[real_index] = __bfloat162float(__float2bfloat16(real * cosine - imag * sine));
        values[real_index + 1] = __bfloat162float(__float2bfloat16(imag * cosine + real * sine));
    }
    __syncthreads();

    if (code_bits == 4) {
        if ((head_dim & (head_dim - 1)) != 0) return;
        for (int stride = 1; stride < head_dim; stride <<= 1) {
            if (lane < head_dim / 2) {
                int base = (lane / stride) * (stride << 1);
                int offset = lane - (lane / stride) * stride;
                float left = values[base + offset];
                float right = values[base + offset + stride];
                values[base + offset] = left + right;
                values[base + offset + stride] = left - right;
            }
            __syncthreads();
        }
        if (lane < head_dim) {
            values[lane] = __bfloat162float(__float2bfloat16(
                values[lane] * rsqrtf((float)head_dim)));
        }
        __syncthreads();
    }

    int quant_cols = code_bits == 4 ? head_dim : head_dim - rope_dim;
    int blocks_per_row = (quant_cols + qat_block_size - 1) / qat_block_size;
    for (int block = 0; block < blocks_per_row; ++block) {
        int block_start = block * qat_block_size;
        float amax = 0.0f;
        int column = block_start + lane;
        if (lane < qat_block_size && column < quant_cols) amax = fabsf(values[column]);
        reduction[lane] = amax;
        __syncthreads();
        for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
            __syncthreads();
        }
        float scale = code_bits == 4
            ? dsv4_pow2_scale(fmaxf(reduction[0], 6.0f * 0x1p-126f), 6.0f)
            : dsv4_pow2_scale(fmaxf(reduction[0], 1.0e-4f), 448.0f);
        if (lane == 0) {
            scale_exponents[(int64_t)cache_row * blocks_per_row + block] =
                dsv4_pow2_scale_exponent(scale);
        }
        if (code_bits == 8) {
            if (lane < qat_block_size && column < quant_cols) {
                float scaled = fminf(fmaxf(values[column] / scale, -448.0f), 448.0f);
                __nv_fp8_e4m3 quantized(scaled);
                codes[(int64_t)cache_row * quant_cols + column] = quantized.__x;
            }
        } else {
            int pair_column = block_start + 2 * lane;
            if (2 * lane < qat_block_size && pair_column < quant_cols) {
                float scaled0 = fminf(fmaxf(values[pair_column] / scale, -6.0f), 6.0f);
                unsigned char code0 = dsv4_e2m1_code(scaled0);
                unsigned char code1 = 0;
                if (pair_column + 1 < quant_cols) {
                    float scaled1 = fminf(fmaxf(values[pair_column + 1] / scale, -6.0f), 6.0f);
                    code1 = dsv4_e2m1_code(scaled1);
                }
                codes[(int64_t)cache_row * ((quant_cols + 1) / 2) + pair_column / 2] =
                    (unsigned char)(code0 | (code1 << 4));
            }
        }
        __syncthreads();
    }
    if (code_bits == 8) {
        for (int offset = lane; offset < rope_dim; offset += (int)blockDim.x) {
            tails[(int64_t)cache_row * rope_dim + offset] =
                __float2bfloat16(values[quant_cols + offset]);
        }
    }
}

// In-place block FP8 E4M3 quantize/dequantize with power-of-two scales. Only
// quant_cols are transformed so the main compressor can preserve RoPE dims.
extern "C" __global__ void deepseek_v4_fp8_qat_inplace_kernel(
    __nv_bfloat16* __restrict__ values,
    int rows,
    int row_width,
    int quant_cols,
    int block_size)
{
    int block = (int)blockIdx.x;
    int row = (int)blockIdx.y;
    int block_start = block * block_size;
    if (row >= rows || block_start >= quant_cols || block_size <= 0) return;
    extern __shared__ float reduction[];
    int lane = (int)threadIdx.x;
    float amax = 0.0f;
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            amax = fmaxf(amax, fabsf(__bfloat162float(values[(int64_t)row * row_width + column])));
        }
    }
    reduction[lane] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float scale = dsv4_pow2_scale(fmaxf(reduction[0], 1.0e-4f), 448.0f);
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            int64_t index = (int64_t)row * row_width + column;
            float scaled = fminf(fmaxf(__bfloat162float(values[index]) / scale, -448.0f), 448.0f);
            __nv_fp8_e4m3 quantized(scaled);
            values[index] = __float2bfloat16((float)quantized * scale);
        }
    }
}

// Normalized Walsh-Hadamard rotation used by the learned indexer. Runtime
// validation requires a power-of-two width; each row is independent.
extern "C" __global__ void deepseek_v4_hadamard_inplace_kernel(
    __nv_bfloat16* __restrict__ values,
    int rows,
    int width)
{
    int row = (int)blockIdx.x;
    if (row >= rows || width <= 0 || (width & (width - 1)) != 0 ||
        width > (int)blockDim.x) return;
    extern __shared__ float shared[];
    int lane = (int)threadIdx.x;
    if (lane < width) shared[lane] = __bfloat162float(values[(int64_t)row * width + lane]);
    __syncthreads();
    for (int stride = 1; stride < width; stride <<= 1) {
        if (lane < width / 2) {
            int base = (lane / stride) * (stride << 1);
            int offset = lane - (lane / stride) * stride;
            float left = shared[base + offset];
            float right = shared[base + offset + stride];
            shared[base + offset] = left + right;
            shared[base + offset + stride] = left - right;
        }
        __syncthreads();
    }
    if (lane < width) {
        values[(int64_t)row * width + lane] = __float2bfloat16(
            shared[lane] * rsqrtf((float)width));
    }
}

// In-place block FP4 E2M1 quantize/dequantize with power-of-two UE8M0 scale.
extern "C" __global__ void deepseek_v4_fp4_qat_inplace_kernel(
    __nv_bfloat16* __restrict__ values,
    int rows,
    int width,
    int block_size)
{
    if (block_size <= 0 || width <= 0) return;
    int blocks_per_row = (width + block_size - 1) / block_size;
    int linear_block = (int)blockIdx.x;
    int row = linear_block / blocks_per_row;
    int block = linear_block - row * blocks_per_row;
    int block_start = block * block_size;
    if (row >= rows || block_start >= width) return;
    extern __shared__ float reduction[];
    int lane = (int)threadIdx.x;
    float amax = 0.0f;
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < width) {
            amax = fmaxf(amax, fabsf(__bfloat162float(values[(int64_t)row * width + column])));
        }
    }
    reduction[lane] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float scale = dsv4_pow2_scale(fmaxf(reduction[0], 6.0f * 0x1p-126f), 6.0f);
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < width) {
            int64_t index = (int64_t)row * width + column;
            float scaled = fminf(fmaxf(__bfloat162float(values[index]) / scale, -6.0f), 6.0f);
            values[index] = __float2bfloat16(dsv4_e2m1_round(scaled) * scale);
        }
    }
}

// Pack already-normalized rows into DeepSeek-V4's source-native persistent
// representation. The BF16 input is rewritten with the exact dequantized value
// so callers can compare/use it without changing the established QAT contract.
extern "C" __global__ void deepseek_v4_pack_fp8_native_kernel(
    __nv_bfloat16* __restrict__ values,
    unsigned char* __restrict__ codes,
    signed char* __restrict__ scale_exponents,
    __nv_bfloat16* __restrict__ tails,
    int rows,
    int row_width,
    int quant_cols,
    int block_size,
    int destination_row)
{
    int block = (int)blockIdx.x;
    int row = (int)blockIdx.y;
    int blocks_per_row = (quant_cols + block_size - 1) / block_size;
    if (row >= rows || row_width <= 0 || quant_cols <= 0 || quant_cols > row_width ||
        block_size <= 0 || codes == nullptr || scale_exponents == nullptr ||
        tails == nullptr || values == nullptr) return;
    int destination = destination_row + row;
    int lane = (int)threadIdx.x;
    if (block == blocks_per_row) {
        int tail_width = row_width - quant_cols;
        for (int offset = lane; offset < tail_width; offset += (int)blockDim.x) {
            tails[(int64_t)destination * tail_width + offset] =
                values[(int64_t)row * row_width + quant_cols + offset];
        }
        return;
    }
    if (block > blocks_per_row) return;
    int block_start = block * block_size;
    extern __shared__ float reduction[];
    float amax = 0.0f;
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            amax = fmaxf(amax, fabsf(__bfloat162float(
                values[(int64_t)row * row_width + column])));
        }
    }
    reduction[lane] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float scale = dsv4_pow2_scale(fmaxf(reduction[0], 1.0e-4f), 448.0f);
    if (lane == 0) {
        scale_exponents[(int64_t)destination * blocks_per_row + block] =
            dsv4_pow2_scale_exponent(scale);
    }
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            int64_t source = (int64_t)row * row_width + column;
            float scaled = fminf(fmaxf(__bfloat162float(values[source]) / scale, -448.0f), 448.0f);
            __nv_fp8_e4m3 quantized(scaled);
            codes[(int64_t)destination * quant_cols + column] = quantized.__x;
            values[source] = __float2bfloat16((float)quantized * scale);
        }
    }
}

extern "C" __global__ void deepseek_v4_unpack_fp8_native_kernel(
    __nv_bfloat16* __restrict__ output,
    const unsigned char* __restrict__ codes,
    const signed char* __restrict__ scale_exponents,
    const __nv_bfloat16* __restrict__ tails,
    int rows,
    int row_width,
    int quant_cols,
    int block_size,
    int source_row)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * row_width;
    if (linear >= total || output == nullptr || codes == nullptr ||
        scale_exponents == nullptr || tails == nullptr || block_size <= 0 ||
        quant_cols <= 0 || quant_cols > row_width) return;
    int row = (int)(linear / row_width);
    int column = (int)(linear - (int64_t)row * row_width);
    int source = source_row + row;
    if (column < quant_cols) {
        int blocks_per_row = (quant_cols + block_size - 1) / block_size;
        float scale = dsv4_pow2_scale_from_exponent(
            scale_exponents[(int64_t)source * blocks_per_row + column / block_size]);
        __nv_fp8_e4m3 quantized;
        quantized.__x = codes[(int64_t)source * quant_cols + column];
        output[linear] = __float2bfloat16((float)quantized * scale);
    } else {
        int tail_width = row_width - quant_cols;
        output[linear] = tails[(int64_t)source * tail_width + column - quant_cols];
    }
}

extern "C" __global__ void deepseek_v4_pack_fp4_native_kernel(
    __nv_bfloat16* __restrict__ values,
    unsigned char* __restrict__ packed_codes,
    signed char* __restrict__ scale_exponents,
    int rows,
    int width,
    int block_size,
    int destination_row)
{
    int linear_block = (int)blockIdx.x;
    int blocks_per_row = (width + block_size - 1) / block_size;
    int row = linear_block / blocks_per_row;
    int block = linear_block - row * blocks_per_row;
    int block_start = block * block_size;
    if (row >= rows || values == nullptr || packed_codes == nullptr ||
        scale_exponents == nullptr || width <= 0 || block_size <= 0) return;
    int lane = (int)threadIdx.x;
    extern __shared__ float reduction[];
    float amax = 0.0f;
    for (int offset = 2 * lane; offset < block_size; offset += 2 * (int)blockDim.x) {
        int column0 = block_start + offset;
        int column1 = column0 + 1;
        if (column0 < width) amax = fmaxf(amax, fabsf(__bfloat162float(values[(int64_t)row * width + column0])));
        if (column1 < width) amax = fmaxf(amax, fabsf(__bfloat162float(values[(int64_t)row * width + column1])));
    }
    reduction[lane] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float scale = dsv4_pow2_scale(fmaxf(reduction[0], 6.0f * 0x1p-126f), 6.0f);
    int destination = destination_row + row;
    if (lane == 0) {
        scale_exponents[(int64_t)destination * blocks_per_row + block] =
            dsv4_pow2_scale_exponent(scale);
    }
    int code_stride = (width + 1) / 2;
    for (int offset = 2 * lane; offset < block_size; offset += 2 * (int)blockDim.x) {
        int column0 = block_start + offset;
        int column1 = column0 + 1;
        if (column0 >= width) continue;
        int64_t source0 = (int64_t)row * width + column0;
        float scaled0 = fminf(fmaxf(__bfloat162float(values[source0]) / scale, -6.0f), 6.0f);
        unsigned char code0 = dsv4_e2m1_code(scaled0);
        unsigned char code1 = 0;
        values[source0] = __float2bfloat16(dsv4_e2m1_from_code(code0) * scale);
        if (column1 < width) {
            int64_t source1 = (int64_t)row * width + column1;
            float scaled1 = fminf(fmaxf(__bfloat162float(values[source1]) / scale, -6.0f), 6.0f);
            code1 = dsv4_e2m1_code(scaled1);
            values[source1] = __float2bfloat16(dsv4_e2m1_from_code(code1) * scale);
        }
        packed_codes[(int64_t)destination * code_stride + column0 / 2] =
            (unsigned char)(code0 | (code1 << 4));
    }
}

extern "C" __global__ void deepseek_v4_unpack_fp4_native_kernel(
    __nv_bfloat16* __restrict__ output,
    const unsigned char* __restrict__ packed_codes,
    const signed char* __restrict__ scale_exponents,
    int rows,
    int width,
    int block_size,
    int source_row)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * width;
    if (linear >= total || output == nullptr || packed_codes == nullptr ||
        scale_exponents == nullptr || width <= 0 || block_size <= 0) return;
    int row = (int)(linear / width);
    int column = (int)(linear - (int64_t)row * width);
    int source = source_row + row;
    int code_stride = (width + 1) / 2;
    int blocks_per_row = (width + block_size - 1) / block_size;
    unsigned char pair = packed_codes[(int64_t)source * code_stride + column / 2];
    unsigned char code = (column & 1) == 0 ? pair & 0x0f : pair >> 4;
    float scale = dsv4_pow2_scale_from_exponent(
        scale_exponents[(int64_t)source * blocks_per_row + column / block_size]);
    output[linear] = __float2bfloat16(dsv4_e2m1_from_code(code) * scale);
}

// The shipped indexer rounds weights_proj(x) *
// (head_dim^-0.5 * n_heads^-0.5) back to BF16 before score accumulation.
extern "C" __global__ void deepseek_v4_scale_index_weights_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int elements,
    float scale)
{
    int index = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16(__bfloat162float(input[index]) * scale);
    }
}

// Build the raw sliding-window portion of sparse-attention indices. Prefill
// uses absolute rows in its transient contiguous raw buffer; decode uses the
// physical ring slots. Both preserve chronological accumulation order.
extern "C" __global__ void deepseek_v4_window_indices_kernel(
    int* __restrict__ output,
    const int* __restrict__ positions,
    const int* __restrict__ visible_left,
    const int* __restrict__ visible_right,
    int rows,
    int window,
    int raw_width,
    int output_stride,
    int physical_ring)
{
    int column = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int)blockIdx.y;
    if (row >= rows || column >= raw_width || positions == nullptr ||
        output_stride < raw_width || window <= 0 || raw_width < window) return;
    int position = positions[row];
    int left = visible_left ? visible_left[position] : 0;
    int right = visible_right ? visible_right[position] : 0;
    int earliest = max(0, position + 1 - window);
    if (left > window - 1) earliest = position - left;
    int latest = position + right;
    int count = min(raw_width, latest - earliest + 1);
    int value = -1;
    if (column < count) {
        int absolute = earliest + column;
        value = physical_ring ? absolute % window : absolute;
    }
    output[(int64_t)row * output_stride + column] = value;
}

// Offset validated learned compressed-cache selections into the concatenated
// raw+compressed sparse-attention index space. Invalid/padded candidates stay
// -1 rather than becoming large positive offsets.
extern "C" __global__ void deepseek_v4_offset_index_selection_kernel(
    int* __restrict__ output,
    const int* __restrict__ selected,
    const int* __restrict__ valid_counts,
    int rows,
    int selected_stride,
    int output_stride,
    int output_column,
    int cache_offset)
{
    int column = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int)blockIdx.y;
    if (row >= rows || column >= selected_stride || valid_counts == nullptr ||
        output_stride < output_column + selected_stride) return;
    int index = selected[(int64_t)row * selected_stride + column];
    output[(int64_t)row * output_stride + output_column + column] =
        index >= 0 && index < valid_counts[row] ? index + cache_offset : -1;
}

// Positions assigned to compressed rows use the first token in each source
// window, matching `freqs_cis[:cutoff:ratio]` in the shipped implementation.
extern "C" __global__ void deepseek_v4_compressed_positions_kernel(
    int* __restrict__ output,
    int rows,
    int first_group,
    int ratio)
{
    int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && ratio > 0) output[row] = (first_group + row) * ratio;
}

// Convert absolute token positions to the compressed-cache causal boundary.
// `offset=-1` produces the pseudo-position consumed by the shared DSA scorer
// (which uses position+1 as its valid count); `offset=0` produces the literal
// valid-count vector consumed by V4's selection-offset validator.
extern "C" __global__ void deepseek_v4_compressed_causal_counts_kernel(
    int* __restrict__ output,
    const int* __restrict__ positions,
    int rows,
    int ratio,
    int offset)
{
    int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || positions == nullptr || ratio <= 0) return;
    output[row] = (positions[row] + 1) / ratio + offset;
}

// Build the causal static-compressor suffix of the attention index matrix.
// The raw prefix occupies [0, raw_offset), so valid compressed rows are
// shifted by raw_offset; incomplete future rows remain -1.
extern "C" __global__ void deepseek_v4_static_compressed_indices_kernel(
    int* __restrict__ output,
    const int* __restrict__ positions,
    int rows,
    int ratio,
    int output_stride,
    int output_column,
    int selected_stride,
    int raw_offset)
{
    int column = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int)blockIdx.y;
    if (row >= rows || column >= selected_stride || ratio <= 0 ||
        output_stride < output_column + selected_stride) return;
    int valid = (positions[row] + 1) / ratio;
    output[(int64_t)row * output_stride + output_column + column] =
        column < valid ? raw_offset + column : -1;
}

// Store a chunk's normalized/QAT KV both in absolute chronological prompt
// history and in the persistent modulo ring consumed by Rust decode.
extern "C" __global__ void deepseek_v4_store_raw_kv_kernel(
    __nv_bfloat16* __restrict__ history,
    __nv_bfloat16* __restrict__ ring,
    const __nv_bfloat16* __restrict__ input,
    int start_pos,
    int tokens,
    int head_dim,
    int window)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)tokens * head_dim;
    if (linear >= total || head_dim <= 0 || window <= 0) return;
    int token = (int)(linear / head_dim);
    int dim = (int)(linear % head_dim);
    int position = start_pos + token;
    __nv_bfloat16 value = input[linear];
    history[(int64_t)position * head_dim + dim] = value;
    // A prompt chunk may span the ring more than once. Only the last token in
    // this chunk for a physical slot may publish it; otherwise parallel writes
    // race and an older absolute row can win nondeterministically.
    if ((int64_t)token + window >= tokens) {
        ring[((int64_t)(position % window) * head_dim) + dim] = value;
    }
}

// Rebuild the request-scoped chronological prefix from the persistent modulo
// ring before a cached continuation. The ring is the canonical sequence state;
// chronological history is derived prefill scratch and is intentionally not
// included in host snapshots.
extern "C" __global__ void deepseek_v4_restore_raw_history_kernel(
    __nv_bfloat16* __restrict__ history,
    const __nv_bfloat16* __restrict__ ring,
    int end_pos,
    int head_dim,
    int window)
{
    int retained = min(end_pos, window);
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)retained * head_dim;
    if (linear >= total || end_pos <= 0 || head_dim <= 0 || window <= 0) return;
    int row = (int)(linear / head_dim);
    int dim = (int)(linear % head_dim);
    int position = end_pos - retained + row;
    history[(int64_t)position * head_dim + dim] =
        ring[((int64_t)(position % window) * head_dim) + dim];
}

// Decode writes one normalized/QAT KV row into the physical sliding ring. The
// absolute prompt history is prefill-only and is deliberately not addressed.
extern "C" __global__ void deepseek_v4_store_raw_kv_decode_kernel(
    __nv_bfloat16* __restrict__ ring,
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ position,
    int head_dim,
    int window)
{
    int dim = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= head_dim || position == nullptr || head_dim <= 0 || window <= 0) return;
    int pos = *position;
    if (pos < 0) return;
    ring[(int64_t)(pos % window) * head_dim + dim] = input[dim];
}

// Source-native raw-ring append for prefill. Quantized dimensions are rounded
// once, published both to chronological BF16 request scratch and to their exact
// E4M3 code/exponent planes. RoPE dimensions remain BF16. A chunk can wrap the
// ring repeatedly, so only its newest token for a physical slot is published.
extern "C" __global__ void deepseek_v4_store_raw_native_prefill_kernel(
    __nv_bfloat16* __restrict__ history,
    unsigned char* __restrict__ ring_codes,
    signed char* __restrict__ ring_scale_exponents,
    __nv_bfloat16* __restrict__ ring_tails,
    __nv_bfloat16* __restrict__ input,
    int start_pos,
    int tokens,
    int head_dim,
    int quant_cols,
    int block_size,
    int window)
{
    int block = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    int blocks_per_row = (quant_cols + block_size - 1) / block_size;
    if (token >= tokens || history == nullptr || ring_codes == nullptr ||
        ring_scale_exponents == nullptr || ring_tails == nullptr || input == nullptr ||
        head_dim <= 0 || quant_cols <= 0 || quant_cols > head_dim ||
        block_size <= 0 || window <= 0) return;
    int lane = (int)threadIdx.x;
    int position = start_pos + token;
    int ring_row = position % window;
    bool publish_ring = (int64_t)token + window >= tokens;
    if (block == blocks_per_row) {
        int tail_width = head_dim - quant_cols;
        for (int offset = lane; offset < tail_width; offset += (int)blockDim.x) {
            __nv_bfloat16 value = input[(int64_t)token * head_dim + quant_cols + offset];
            history[(int64_t)position * head_dim + quant_cols + offset] = value;
            if (publish_ring) ring_tails[(int64_t)ring_row * tail_width + offset] = value;
        }
        return;
    }
    if (block > blocks_per_row) return;
    int block_start = block * block_size;
    extern __shared__ float reduction[];
    float amax = 0.0f;
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            amax = fmaxf(amax, fabsf(__bfloat162float(input[(int64_t)token * head_dim + column])));
        }
    }
    reduction[lane] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float scale = dsv4_pow2_scale(fmaxf(reduction[0], 1.0e-4f), 448.0f);
    if (lane == 0 && publish_ring) {
        ring_scale_exponents[(int64_t)ring_row * blocks_per_row + block] =
            dsv4_pow2_scale_exponent(scale);
    }
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            int64_t input_index = (int64_t)token * head_dim + column;
            float scaled = fminf(fmaxf(__bfloat162float(input[input_index]) / scale, -448.0f), 448.0f);
            __nv_fp8_e4m3 quantized(scaled);
            __nv_bfloat16 value = __float2bfloat16((float)quantized * scale);
            input[input_index] = value;
            history[(int64_t)position * head_dim + column] = value;
            if (publish_ring) ring_codes[(int64_t)ring_row * quant_cols + column] = quantized.__x;
        }
    }
}

extern "C" __global__ void deepseek_v4_store_raw_native_decode_kernel(
    unsigned char* __restrict__ ring_codes,
    signed char* __restrict__ ring_scale_exponents,
    __nv_bfloat16* __restrict__ ring_tails,
    __nv_bfloat16* __restrict__ input,
    const int* __restrict__ position,
    int head_dim,
    int quant_cols,
    int block_size,
    int window)
{
    int block = (int)blockIdx.x;
    int blocks_per_row = (quant_cols + block_size - 1) / block_size;
    if (position == nullptr || ring_codes == nullptr || ring_scale_exponents == nullptr ||
        ring_tails == nullptr || input == nullptr || head_dim <= 0 || quant_cols <= 0 ||
        quant_cols > head_dim || block_size <= 0 || window <= 0) return;
    int pos = *position;
    if (pos < 0) return;
    int ring_row = pos % window;
    int lane = (int)threadIdx.x;
    if (block == blocks_per_row) {
        int tail_width = head_dim - quant_cols;
        for (int offset = lane; offset < tail_width; offset += (int)blockDim.x) {
            ring_tails[(int64_t)ring_row * tail_width + offset] = input[quant_cols + offset];
        }
        return;
    }
    if (block > blocks_per_row) return;
    int block_start = block * block_size;
    extern __shared__ float reduction[];
    float amax = 0.0f;
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) amax = fmaxf(amax, fabsf(__bfloat162float(input[column])));
    }
    reduction[lane] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float scale = dsv4_pow2_scale(fmaxf(reduction[0], 1.0e-4f), 448.0f);
    if (lane == 0) {
        ring_scale_exponents[(int64_t)ring_row * blocks_per_row + block] =
            dsv4_pow2_scale_exponent(scale);
    }
    for (int offset = lane; offset < block_size; offset += (int)blockDim.x) {
        int column = block_start + offset;
        if (column < quant_cols) {
            float scaled = fminf(fmaxf(__bfloat162float(input[column]) / scale, -448.0f), 448.0f);
            __nv_fp8_e4m3 quantized(scaled);
            ring_codes[(int64_t)ring_row * quant_cols + column] = quantized.__x;
            input[column] = __float2bfloat16((float)quantized * scale);
        }
    }
}

extern "C" __global__ void deepseek_v4_restore_raw_history_native_kernel(
    __nv_bfloat16* __restrict__ history,
    const unsigned char* __restrict__ ring_codes,
    const signed char* __restrict__ ring_scale_exponents,
    const __nv_bfloat16* __restrict__ ring_tails,
    int end_pos,
    int head_dim,
    int quant_cols,
    int block_size,
    int window)
{
    int retained = min(end_pos, window);
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)retained * head_dim;
    if (linear >= total || history == nullptr || ring_codes == nullptr ||
        ring_scale_exponents == nullptr || ring_tails == nullptr || end_pos <= 0 ||
        head_dim <= 0 || quant_cols <= 0 || quant_cols > head_dim ||
        block_size <= 0 || window <= 0) return;
    int row = (int)(linear / head_dim);
    int column = (int)(linear - (int64_t)row * head_dim);
    int position_value = end_pos - retained + row;
    int ring_row = position_value % window;
    if (column < quant_cols) {
        int blocks_per_row = (quant_cols + block_size - 1) / block_size;
        float scale = dsv4_pow2_scale_from_exponent(
            ring_scale_exponents[(int64_t)ring_row * blocks_per_row + column / block_size]);
        __nv_fp8_e4m3 quantized;
        quantized.__x = ring_codes[(int64_t)ring_row * quant_cols + column];
        history[(int64_t)position_value * head_dim + column] =
            __float2bfloat16((float)quantized * scale);
    } else {
        int tail_width = head_dim - quant_cols;
        history[(int64_t)position_value * head_dim + column] =
            ring_tails[(int64_t)ring_row * tail_width + column - quant_cols];
    }
}
