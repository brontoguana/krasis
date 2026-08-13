#pragma once

#include <math.h>
#include <stdint.h>

// Source-faithful DeepSeek-V4 sparse attention primitives shared by prefill
// and decode. Geometry and cache boundaries are runtime inputs.

__device__ __forceinline__ float dsv4_attn_bf16_to_f32(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 dsv4_attn_f32_to_bf16(float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ float dsv4_attn_native_main_value(
    const unsigned char* codes,
    const signed char* scale_exponents,
    const __nv_bfloat16* tails,
    int row,
    int column,
    int head_dim,
    int quant_cols,
    int block_size)
{
    if (column >= quant_cols) {
        return __bfloat162float(tails[(int64_t)row * (head_dim - quant_cols) + column - quant_cols]);
    }
    int blocks_per_row = (quant_cols + block_size - 1) / block_size;
    float scale = ldexpf(1.0f, (int)scale_exponents[
        (int64_t)row * blocks_per_row + column / block_size]);
    __nv_fp8_e4m3 quantized;
    quantized.__x = codes[(int64_t)row * quant_cols + column];
    return (float)quantized * scale;
}

// One block normalizes one BF16 row. DeepSeek-V4 applies this both to the
// per-head Q rows (without a learned weight) and the single KV row (with one).
// Keeping all rows in one launch avoids a per-head launch chain at decode M=1.
extern "C" __global__ void deepseek_v4_rmsnorm_rows_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int rows,
    int width,
    float scale,
    float eps)
{
    int row = (int)blockIdx.x;
    if (row >= rows || rows <= 0 || width <= 0) return;
    extern __shared__ float reduction[];
    int lane = (int)threadIdx.x;
    float sumsq = 0.0f;
    const __nv_bfloat16* input_row = input + (int64_t)row * width;
    __nv_bfloat16* output_row = output + (int64_t)row * width;
    for (int dim = lane; dim < width; dim += (int)blockDim.x) {
        float value = dsv4_attn_bf16_to_f32(input_row[dim]);
        sumsq += value * value;
    }
    reduction[lane] = sumsq;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inv = rsqrtf(reduction[0] / (float)width + eps) * scale;
    for (int dim = lane; dim < width; dim += (int)blockDim.x) {
        float value = dsv4_attn_bf16_to_f32(input_row[dim]) * inv;
        if (weight != nullptr) value *= dsv4_attn_bf16_to_f32(weight[dim]);
        output_row[dim] = dsv4_attn_f32_to_bf16(value);
    }
}

// Apply the shipped DeepSeek-V4 rotary contract to the tail of each head.
// V4 treats adjacent values as the real/imaginary parts of one complex value;
// inverse de-rotation multiplies by the conjugate. The non-RoPE prefix is
// deliberately untouched. All geometry and table bounds are runtime inputs.
extern "C" __global__ void deepseek_v4_tail_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ values,
    const int* __restrict__ positions,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int rope_rows,
    int inverse)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int64_t total_pairs = (int64_t)tokens * heads * half_rope;
    if (linear >= total_pairs || values == nullptr || positions == nullptr ||
        cos_table == nullptr || sin_table == nullptr || tokens <= 0 || heads <= 0 ||
        head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim || (rope_dim & 1) != 0 ||
        rope_rows <= 0) {
        return;
    }

    int pair = (int)(linear % half_rope);
    int64_t row = linear / half_rope;
    int token = (int)(row / heads);
    int position = positions[token];
    if (position < 0 || position >= rope_rows) return;

    int tail = head_dim - rope_dim;
    int64_t base = row * head_dim + tail + 2 * pair;
    float real = dsv4_attn_bf16_to_f32(values[base]);
    float imag = dsv4_attn_bf16_to_f32(values[base + 1]);
    float cosine = cos_table[(int64_t)position * half_rope + pair];
    float sine = sin_table[(int64_t)position * half_rope + pair];
    if (inverse != 0) sine = -sine;
    values[base] = dsv4_attn_f32_to_bf16(real * cosine - imag * sine);
    values[base + 1] = dsv4_attn_f32_to_bf16(imag * cosine + real * sine);
}

__device__ __forceinline__ const __nv_bfloat16* dsv4_attn_kv_row(
    int index,
    const __nv_bfloat16* raw,
    int raw_rows,
    const __nv_bfloat16* compressed,
    int compressed_rows,
    int head_dim)
{
    if (index < 0) return nullptr;
    if (index < raw_rows) {
        return raw + (int64_t)index * head_dim;
    }
    int compressed_index = index - raw_rows;
    if (compressed == nullptr || compressed_index >= compressed_rows) return nullptr;
    return compressed + (int64_t)compressed_index * head_dim;
}

// One block computes one (token, head, selected-position) dot product. Scores
// stay FP32 exactly as in the shipped online-softmax implementation.
extern "C" __global__ void deepseek_v4_sparse_scores_kernel(
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const int* __restrict__ indices,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows,
    float scale)
{
    int selected = (int)blockIdx.x;
    int head = (int)blockIdx.y;
    int token = (int)blockIdx.z;
    if (selected >= topk || head >= heads || token >= tokens) return;
    int index = indices[(int64_t)token * topk + selected];
    const __nv_bfloat16* kv = dsv4_attn_kv_row(
        index, raw, raw_rows, compressed, compressed_rows, head_dim);
    if (kv == nullptr) {
        if (threadIdx.x == 0) {
            scores[((int64_t)token * heads + head) * topk + selected] = -INFINITY;
        }
        return;
    }
    const __nv_bfloat16* q = query + ((int64_t)token * heads + head) * head_dim;
    float partial = 0.0f;
    for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
        partial += dsv4_attn_bf16_to_f32(q[dim]) * dsv4_attn_bf16_to_f32(kv[dim]);
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[((int64_t)token * heads + head) * topk + selected] = reduction[0] * scale;
    }
}

// Bit-equivalent query-cached form. One block owns a (token, head) row and
// walks every checkpoint-selected KV row. The per-score multiply and shared
// reduction order is identical to deepseek_v4_sparse_scores_kernel; only the
// immutable BF16 query load is shared across selected rows.
extern "C" __global__ void deepseek_v4_sparse_scores_query_cached_kernel(
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const int* __restrict__ indices,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows,
    float scale)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (head >= heads || token >= tokens || head_dim <= 0 || topk <= 0) return;

    extern __shared__ unsigned char shared_bytes[];
    __nv_bfloat16* cached_query =
        reinterpret_cast<__nv_bfloat16*>(shared_bytes);
    size_t query_bytes = (size_t)head_dim * sizeof(__nv_bfloat16);
    size_t reduction_offset = (query_bytes + alignof(float) - 1) &
        ~(size_t)(alignof(float) - 1);
    float* reduction = reinterpret_cast<float*>(shared_bytes + reduction_offset);
    const __nv_bfloat16* q =
        query + ((int64_t)token * heads + head) * head_dim;
    for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
        cached_query[dim] = q[dim];
    }
    __syncthreads();

    const int* token_indices = indices + (int64_t)token * topk;
    float* score_row = scores + ((int64_t)token * heads + head) * topk;
    for (int selected = 0; selected < topk; ++selected) {
        int index = token_indices[selected];
        const __nv_bfloat16* kv = dsv4_attn_kv_row(
            index, raw, raw_rows, compressed, compressed_rows, head_dim);
        if (kv == nullptr) {
            if (threadIdx.x == 0) score_row[selected] = -INFINITY;
            __syncthreads();
            continue;
        }

        float partial = 0.0f;
        for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
            partial += dsv4_attn_bf16_to_f32(cached_query[dim]) *
                dsv4_attn_bf16_to_f32(kv[dim]);
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) score_row[selected] = reduction[0] * scale;
        __syncthreads();
    }
}

// Gather the selected KV rows once for a dense score GEMM. The same launch
// initializes the FP32 score matrix for beta=1 GEMM accumulation: valid rows
// start at zero, while invalid rows start at -infinity and gather as zero so
// the established invalid-index result remains -infinity.
extern "C" __global__ void deepseek_v4_gather_selected_kv_scores_kernel(
    __nv_bfloat16* __restrict__ selected_kv,
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const int* __restrict__ indices,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t kv_elements = (int64_t)topk * head_dim;
    int64_t score_elements = (int64_t)heads * topk;
    int64_t total = kv_elements + score_elements;
    if (linear >= total || selected_kv == nullptr || scores == nullptr ||
        raw == nullptr || indices == nullptr || heads <= 0 || head_dim <= 0 ||
        topk <= 0 || raw_rows < 0 || compressed_rows < 0) {
        return;
    }

    if (linear < kv_elements) {
        int selected = (int)(linear / head_dim);
        int dim = (int)(linear - (int64_t)selected * head_dim);
        int index = indices[selected];
        const __nv_bfloat16* kv = dsv4_attn_kv_row(
            index, raw, raw_rows, compressed, compressed_rows, head_dim);
        selected_kv[linear] = kv == nullptr ? dsv4_attn_f32_to_bf16(0.0f) : kv[dim];
        return;
    }

    int64_t score_linear = linear - kv_elements;
    int selected = (int)(score_linear % topk);
    int index = indices[selected];
    bool valid = index >= 0 &&
        (index < raw_rows ||
         (compressed != nullptr && index - raw_rows >= 0 &&
          index - raw_rows < compressed_rows));
    scores[score_linear] = valid ? 0.0f : -INFINITY;
}

extern "C" __global__ void deepseek_v4_gather_selected_native_kv_scores_kernel(
    __nv_bfloat16* __restrict__ selected_kv,
    float* __restrict__ scores,
    const unsigned char* __restrict__ raw_codes,
    const signed char* __restrict__ raw_scale_exponents,
    const __nv_bfloat16* __restrict__ raw_tails,
    const unsigned char* __restrict__ compressed_codes,
    const signed char* __restrict__ compressed_scale_exponents,
    const __nv_bfloat16* __restrict__ compressed_tails,
    const int* __restrict__ indices,
    int heads,
    int head_dim,
    int quant_cols,
    int block_size,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t kv_elements = (int64_t)topk * head_dim;
    int64_t score_elements = (int64_t)heads * topk;
    int64_t total = kv_elements + score_elements;
    if (linear >= total || selected_kv == nullptr || scores == nullptr ||
        raw_codes == nullptr || raw_scale_exponents == nullptr || raw_tails == nullptr ||
        indices == nullptr || heads <= 0 || head_dim <= 0 || quant_cols <= 0 ||
        quant_cols >= head_dim || block_size <= 0 || topk <= 0 || raw_rows < 0 ||
        compressed_rows < 0) return;

    if (linear < kv_elements) {
        int selected = (int)(linear / head_dim);
        int dim = (int)(linear - (int64_t)selected * head_dim);
        int index = indices[selected];
        float value = 0.0f;
        if (index >= 0 && index < raw_rows) {
            value = dsv4_attn_native_main_value(
                raw_codes, raw_scale_exponents, raw_tails,
                index, dim, head_dim, quant_cols, block_size);
        } else {
            int compressed_row = index - raw_rows;
            if (compressed_row >= 0 && compressed_row < compressed_rows &&
                compressed_codes != nullptr && compressed_scale_exponents != nullptr &&
                compressed_tails != nullptr) {
                value = dsv4_attn_native_main_value(
                    compressed_codes, compressed_scale_exponents, compressed_tails,
                    compressed_row, dim, head_dim, quant_cols, block_size);
            }
        }
        selected_kv[linear] = dsv4_attn_f32_to_bf16(value);
        return;
    }

    int64_t score_linear = linear - kv_elements;
    int selected = (int)(score_linear % topk);
    int index = indices[selected];
    bool valid = index >= 0 &&
        (index < raw_rows ||
         (compressed_codes != nullptr && index - raw_rows >= 0 &&
          index - raw_rows < compressed_rows));
    scores[score_linear] = valid ? 0.0f : -INFINITY;
}

#if defined(KRASIS_DEEPSEEK_V4_PREFILL_ONLY_KERNELS)
// Multi-token gathered-score preparation for prefill. The BF16 form feeds the
// fast ordinary GEMM path; the FP32 form materializes both operands exactly
// before the pedantic GEMM. Both initialize invalid score columns to -infinity
// so beta=1 GEMM preserves the established invalid-index contract.
extern "C" __global__ void deepseek_v4_prefill_gather_scores_bf16_kernel(
    __nv_bfloat16* __restrict__ selected_kv,
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const int* __restrict__ indices,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t kv_per_token = (int64_t)topk * head_dim;
    int64_t kv_elements = (int64_t)tokens * kv_per_token;
    int64_t score_per_token = (int64_t)heads * topk;
    int64_t score_elements = (int64_t)tokens * score_per_token;
    int64_t total = kv_elements + score_elements;
    if (linear >= total || selected_kv == nullptr || scores == nullptr ||
        raw == nullptr || indices == nullptr || tokens <= 0 || heads <= 0 ||
        head_dim <= 0 || topk <= 0 || raw_rows < 0 || compressed_rows < 0) {
        return;
    }

    if (linear < kv_elements) {
        int token = (int)(linear / kv_per_token);
        int64_t within = linear - (int64_t)token * kv_per_token;
        int selected = (int)(within / head_dim);
        int dim = (int)(within - (int64_t)selected * head_dim);
        int index = indices[(int64_t)token * topk + selected];
        const __nv_bfloat16* kv = dsv4_attn_kv_row(
            index, raw, raw_rows, compressed, compressed_rows, head_dim);
        selected_kv[linear] =
            kv == nullptr ? dsv4_attn_f32_to_bf16(0.0f) : kv[dim];
        return;
    }

    int64_t score_linear = linear - kv_elements;
    int token = (int)(score_linear / score_per_token);
    int selected = (int)(score_linear % topk);
    int index = indices[(int64_t)token * topk + selected];
    bool valid = index >= 0 &&
        (index < raw_rows ||
         (compressed != nullptr && index - raw_rows >= 0 &&
          index - raw_rows < compressed_rows));
    scores[score_linear] = valid ? 0.0f : -INFINITY;
}

// Copy selected BF16 value rows in 16-byte chunks while retaining the scalar
// score-initialization contract. Runtime head width determines both the chunk
// count and the scalar tail; unaligned rows are copied element-by-element, so
// no model-specific alignment assumption or fallback path is required.
extern "C" __global__ void deepseek_v4_prefill_gather_scores_bf16x8_kernel(
    __nv_bfloat16* __restrict__ selected_kv,
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const int* __restrict__ indices,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    constexpr int values_per_thread = 8;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t selected_rows = (int64_t)tokens * topk;
    int64_t chunks_per_row = (head_dim + values_per_thread - 1) / values_per_thread;
    int64_t value_chunks = selected_rows * chunks_per_row;
    int64_t score_per_token = (int64_t)heads * topk;
    int64_t score_elements = (int64_t)tokens * score_per_token;
    int64_t total = value_chunks + score_elements;
    if (linear >= total || selected_kv == nullptr || scores == nullptr ||
        raw == nullptr || indices == nullptr || tokens <= 0 || heads <= 0 ||
        head_dim <= 0 || topk <= 0 || raw_rows < 0 || compressed_rows < 0) {
        return;
    }

    if (linear < value_chunks) {
        int64_t selected_row = linear / chunks_per_row;
        int chunk = (int)(linear - selected_row * chunks_per_row);
        int token = (int)(selected_row / topk);
        int selected = (int)(selected_row - (int64_t)token * topk);
        int dim = chunk * values_per_thread;
        int count = min(values_per_thread, head_dim - dim);
        int index = indices[(int64_t)token * topk + selected];
        const __nv_bfloat16* kv = dsv4_attn_kv_row(
            index, raw, raw_rows, compressed, compressed_rows, head_dim);
        __nv_bfloat16* destination = selected_kv + selected_row * head_dim + dim;
        if (count == values_per_thread &&
            (((uintptr_t)destination | (uintptr_t)(kv == nullptr ? destination : kv + dim)) &
             (alignof(uint4) - 1)) == 0) {
            *reinterpret_cast<uint4*>(destination) = kv == nullptr
                ? make_uint4(0, 0, 0, 0)
                : *reinterpret_cast<const uint4*>(kv + dim);
        } else {
            for (int offset = 0; offset < count; ++offset) {
                destination[offset] = kv == nullptr
                    ? dsv4_attn_f32_to_bf16(0.0f)
                    : kv[dim + offset];
            }
        }
        return;
    }

    int64_t score_linear = linear - value_chunks;
    int token = (int)(score_linear / score_per_token);
    int selected = (int)(score_linear % topk);
    int index = indices[(int64_t)token * topk + selected];
    bool valid = index >= 0 &&
        (index < raw_rows ||
         (compressed != nullptr && index - raw_rows >= 0 &&
          index - raw_rows < compressed_rows));
    scores[score_linear] = valid ? 0.0f : -INFINITY;
}

extern "C" __global__ void deepseek_v4_prefill_gather_scores_fp32_kernel(
    float* __restrict__ selected_kv,
    float* __restrict__ query_fp32,
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const int* __restrict__ indices,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t kv_per_token = (int64_t)topk * head_dim;
    int64_t kv_elements = (int64_t)tokens * kv_per_token;
    int64_t query_per_token = (int64_t)heads * head_dim;
    int64_t query_elements = (int64_t)tokens * query_per_token;
    int64_t score_per_token = (int64_t)heads * topk;
    int64_t score_elements = (int64_t)tokens * score_per_token;
    int64_t total = kv_elements + query_elements + score_elements;
    if (linear >= total || selected_kv == nullptr || query_fp32 == nullptr ||
        scores == nullptr || query == nullptr || raw == nullptr ||
        indices == nullptr || tokens <= 0 || heads <= 0 || head_dim <= 0 ||
        topk <= 0 || raw_rows < 0 || compressed_rows < 0) {
        return;
    }

    if (linear < kv_elements) {
        int token = (int)(linear / kv_per_token);
        int64_t within = linear - (int64_t)token * kv_per_token;
        int selected = (int)(within / head_dim);
        int dim = (int)(within - (int64_t)selected * head_dim);
        int index = indices[(int64_t)token * topk + selected];
        const __nv_bfloat16* kv = dsv4_attn_kv_row(
            index, raw, raw_rows, compressed, compressed_rows, head_dim);
        selected_kv[linear] =
            kv == nullptr ? 0.0f : dsv4_attn_bf16_to_f32(kv[dim]);
        return;
    }

    linear -= kv_elements;
    if (linear < query_elements) {
        query_fp32[linear] = dsv4_attn_bf16_to_f32(query[linear]);
        return;
    }

    int64_t score_linear = linear - query_elements;
    int token = (int)(score_linear / score_per_token);
    int selected = (int)(score_linear % topk);
    int index = indices[(int64_t)token * topk + selected];
    bool valid = index >= 0 &&
        (index < raw_rows ||
         (compressed != nullptr && index - raw_rows >= 0 &&
          index - raw_rows < compressed_rows));
    scores[score_linear] = valid ? 0.0f : -INFINITY;
}

extern "C" __global__ void deepseek_v4_prefill_scale_finite_scores_kernel(
    float* __restrict__ scores,
    int elements,
    float scale)
{
    int index = (int)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements && scores != nullptr && isfinite(scores[index])) {
        scores[index] *= scale;
    }
}

// Materialize each token's selected value rows once for the sparse-output
// GEMM. Invalid indices become zero rows, matching the scalar path's skipped
// contribution. All dimensions and cache boundaries are runtime inputs.
extern "C" __global__ void deepseek_v4_prefill_gather_sparse_values_bf16_kernel(
    __nv_bfloat16* __restrict__ selected_values,
    const int* __restrict__ indices,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    int tokens,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t per_token = (int64_t)topk * head_dim;
    int64_t total = (int64_t)tokens * per_token;
    if (linear >= total || selected_values == nullptr || indices == nullptr ||
        raw == nullptr || tokens <= 0 || head_dim <= 0 || topk <= 0 ||
        raw_rows < 0 || compressed_rows < 0) {
        return;
    }
    int token = (int)(linear / per_token);
    int64_t within = linear - (int64_t)token * per_token;
    int selected = (int)(within / head_dim);
    int dim = (int)(within - (int64_t)selected * head_dim);
    int index = indices[(int64_t)token * topk + selected];
    const __nv_bfloat16* kv = dsv4_attn_kv_row(
        index, raw, raw_rows, compressed, compressed_rows, head_dim);
    selected_values[linear] =
        kv == nullptr ? dsv4_attn_f32_to_bf16(0.0f) : kv[dim];
}

// Reproduce the scalar path's FP32 max and denominator reductions, including
// the learned attention sink, then materialize normalized BF16 weights for a
// tensor-core GEMM. Invalid (-infinity) scores become exact zero weights.
extern "C" __global__ void deepseek_v4_prefill_softmax_weights_bf16_kernel(
    __nv_bfloat16* __restrict__ weights,
    const float* __restrict__ scores,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int topk)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (weights == nullptr || scores == nullptr || attention_sink == nullptr ||
        head >= heads || token >= tokens || heads <= 0 || topk <= 0) {
        return;
    }
    const float* score_row = scores + ((int64_t)token * heads + head) * topk;
    float local_max = threadIdx.x == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = (int)threadIdx.x; selected < topk;
         selected += (int)blockDim.x) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = reduction[0];
    float local_sum = threadIdx.x == 0 ?
        expf(attention_sink[head] - row_max) : 0.0f;
    for (int selected = (int)threadIdx.x; selected < topk;
         selected += (int)blockDim.x) {
        float value = score_row[selected];
        if (isfinite(value)) local_sum += expf(value - row_max);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denominator = reduction[0];
    __nv_bfloat16* weight_row =
        weights + ((int64_t)token * heads + head) * topk;
    for (int selected = (int)threadIdx.x; selected < topk;
         selected += (int)blockDim.x) {
        float value = score_row[selected];
        float weight = isfinite(value) ? expf(value - row_max) / denominator : 0.0f;
        weight_row[selected] = dsv4_attn_f32_to_bf16(weight);
    }
}

// One warp computes one (token, head) row. The block-wide reference above
// launches 512 or 1024 threads for the production selected widths even though
// each thread touches at most two scores. Packing several rows into one block
// removes most launch/reduction overhead while preserving FP32 max/sum and the
// final BF16 weight contract. The reduction tree differs from the reference
// and is therefore selected only by the explicitly validated runtime mode.
extern "C" __global__ void deepseek_v4_prefill_softmax_weights_warp_bf16_kernel(
    __nv_bfloat16* __restrict__ weights,
    const float* __restrict__ scores,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int topk)
{
    if (weights == nullptr || scores == nullptr || attention_sink == nullptr ||
        tokens <= 0 || heads <= 0 || topk <= 0) {
        return;
    }
    int lane = (int)threadIdx.x & (warpSize - 1);
    int warp_in_block = (int)threadIdx.x / warpSize;
    int warps_per_block = (int)blockDim.x / warpSize;
    int64_t row = (int64_t)blockIdx.x * warps_per_block + warp_in_block;
    int64_t rows = (int64_t)tokens * heads;
    if (row >= rows) return;

    int head = (int)(row % heads);
    const float* score_row = scores + row * topk;
    float local_max = lane == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = lane; selected < topk; selected += warpSize) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    unsigned mask = 0xffffffffu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
    }
    float row_max = __shfl_sync(mask, local_max, 0);

    float local_sum = lane == 0 ? expf(attention_sink[head] - row_max) : 0.0f;
    for (int selected = lane; selected < topk; selected += warpSize) {
        float value = score_row[selected];
        if (isfinite(value)) local_sum += expf(value - row_max);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    float denominator = __shfl_sync(mask, local_sum, 0);

    __nv_bfloat16* weight_row = weights + row * topk;
    for (int selected = lane; selected < topk; selected += warpSize) {
        float value = score_row[selected];
        float weight = isfinite(value) ? expf(value - row_max) / denominator : 0.0f;
        weight_row[selected] = dsv4_attn_f32_to_bf16(weight);
    }
}
#endif  // KRASIS_DEEPSEEK_V4_PREFILL_ONLY_KERNELS


// One block computes one (token, head) output. The learned attention sink
// contributes to the softmax denominator only, matching the shipped kernel.
extern "C" __global__ void deepseek_v4_sparse_output_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ scores,
    const int* __restrict__ indices,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const float* score_row = scores + ((int64_t)token * heads + head) * topk;
    float local_max = threadIdx.x == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = reduction[0];
    float local_sum = threadIdx.x == 0 ? expf(attention_sink[head] - row_max) : 0.0f;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        float value = score_row[selected];
        if (isfinite(value)) local_sum += expf(value - row_max);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denominator = reduction[0];
    __nv_bfloat16* output_row = output + ((int64_t)token * heads + head) * head_dim;
    for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
        float accumulator = 0.0f;
        for (int selected = 0; selected < topk; ++selected) {
            float value = score_row[selected];
            if (!isfinite(value)) continue;
            int index = indices[(int64_t)token * topk + selected];
            const __nv_bfloat16* kv = dsv4_attn_kv_row(
                index, raw, raw_rows, compressed, compressed_rows, head_dim);
            if (kv != nullptr) {
                accumulator += expf(value - row_max) * dsv4_attn_bf16_to_f32(kv[dim]);
            }
        }
        output_row[dim] = dsv4_attn_f32_to_bf16(accumulator / denominator);
    }
}

// Preserve the reference softmax reductions and each output dimension's
// selected-position accumulation order while loading two adjacent BF16 values
// per instruction. Odd-width rows use the scalar equation unchanged.
extern "C" __global__ void deepseek_v4_sparse_output_bf16x2_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ scores,
    const int* __restrict__ indices,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const float* score_row = scores + ((int64_t)token * heads + head) * topk;
    float local_max = threadIdx.x == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = reduction[0];
    float local_sum = threadIdx.x == 0 ? expf(attention_sink[head] - row_max) : 0.0f;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        float value = score_row[selected];
        if (isfinite(value)) local_sum += expf(value - row_max);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denominator = reduction[0];
    __nv_bfloat16* output_row = output + ((int64_t)token * heads + head) * head_dim;
    if ((head_dim & 1) != 0) {
        for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
            float accumulator = 0.0f;
            for (int selected = 0; selected < topk; ++selected) {
                float value = score_row[selected];
                if (!isfinite(value)) continue;
                int index = indices[(int64_t)token * topk + selected];
                const __nv_bfloat16* kv = dsv4_attn_kv_row(
                    index, raw, raw_rows, compressed, compressed_rows, head_dim);
                if (kv != nullptr) {
                    accumulator += expf(value - row_max) * dsv4_attn_bf16_to_f32(kv[dim]);
                }
            }
            output_row[dim] = dsv4_attn_f32_to_bf16(accumulator / denominator);
        }
        return;
    }

    int pair_count = head_dim / 2;
    for (int pair = (int)threadIdx.x; pair < pair_count; pair += (int)blockDim.x) {
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;
        for (int selected = 0; selected < topk; ++selected) {
            float value = score_row[selected];
            if (!isfinite(value)) continue;
            int index = indices[(int64_t)token * topk + selected];
            const __nv_bfloat16* kv = dsv4_attn_kv_row(
                index, raw, raw_rows, compressed, compressed_rows, head_dim);
            if (kv != nullptr) {
                float exponential = expf(value - row_max);
                __nv_bfloat162 packed =
                    reinterpret_cast<const __nv_bfloat162*>(kv)[pair];
                accumulator0 += exponential * dsv4_attn_bf16_to_f32(packed.x);
                accumulator1 += exponential * dsv4_attn_bf16_to_f32(packed.y);
            }
        }
        output_row[2 * pair] = dsv4_attn_f32_to_bf16(accumulator0 / denominator);
        output_row[2 * pair + 1] = dsv4_attn_f32_to_bf16(accumulator1 / denominator);
    }
}

// Algebraically identical sparse output with selected-position exponentials
// cached in shared memory. The reference kernel above recomputes the same
// expf(score-row_max) for every output dimension.
extern "C" __global__ void deepseek_v4_sparse_output_cached_exp_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ scores,
    const int* __restrict__ indices,
    const __nv_bfloat16* __restrict__ raw,
    const __nv_bfloat16* __restrict__ compressed,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int head_dim,
    int topk,
    int raw_rows,
    int compressed_rows)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const float* score_row = scores + ((int64_t)token * heads + head) * topk;
    float local_max = threadIdx.x == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    extern __shared__ float sparse_output_shared[];
    float* reduction = sparse_output_shared;
    float* exponentials = reduction + blockDim.x;
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = reduction[0];
    float local_sum = threadIdx.x == 0 ? expf(attention_sink[head] - row_max) : 0.0f;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        float value = score_row[selected];
        float exponential = isfinite(value) ? expf(value - row_max) : 0.0f;
        exponentials[selected] = exponential;
        local_sum += exponential;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denominator = reduction[0];
    __nv_bfloat16* output_row = output + ((int64_t)token * heads + head) * head_dim;
    for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
        float accumulator = 0.0f;
        for (int selected = 0; selected < topk; ++selected) {
            float exponential = exponentials[selected];
            if (exponential == 0.0f) continue;
            int index = indices[(int64_t)token * topk + selected];
            const __nv_bfloat16* kv = dsv4_attn_kv_row(
                index, raw, raw_rows, compressed, compressed_rows, head_dim);
            if (kv != nullptr) {
                accumulator += exponential * dsv4_attn_bf16_to_f32(kv[dim]);
            }
        }
        output_row[dim] = dsv4_attn_f32_to_bf16(accumulator / denominator);
    }
}

// Native decode already reconstructed selected rows once for the score GEMM.
// Reuse that bounded BF16 scratch for value accumulation instead of decoding
// the persistent codes a second time.
extern "C" __global__ void deepseek_v4_sparse_output_selected_cached_exp_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ selected_kv,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int head_dim,
    int topk)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (head >= heads || token >= tokens || output == nullptr || scores == nullptr ||
        selected_kv == nullptr || attention_sink == nullptr) return;
    const float* score_row = scores + ((int64_t)token * heads + head) * topk;
    float local_max = threadIdx.x == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    extern __shared__ float shared[];
    float* reduction = shared;
    float* exponentials = reduction + blockDim.x;
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = reduction[0];
    float local_sum = threadIdx.x == 0 ? expf(attention_sink[head] - row_max) : 0.0f;
    for (int selected = (int)threadIdx.x; selected < topk; selected += (int)blockDim.x) {
        float value = score_row[selected];
        float exponential = isfinite(value) ? expf(value - row_max) : 0.0f;
        exponentials[selected] = exponential;
        local_sum += exponential;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float denominator = reduction[0];
    __nv_bfloat16* output_row = output + ((int64_t)token * heads + head) * head_dim;
    const __nv_bfloat16* selected_rows = selected_kv + (int64_t)token * topk * head_dim;
    for (int dim = (int)threadIdx.x; dim < head_dim; dim += (int)blockDim.x) {
        float accumulator = 0.0f;
        for (int selected = 0; selected < topk; ++selected) {
            accumulator += exponentials[selected] * __bfloat162float(
                selected_rows[(int64_t)selected * head_dim + dim]);
        }
        output_row[dim] = dsv4_attn_f32_to_bf16(accumulator / denominator);
    }
}

// Convert sparse attention scores into normalized BF16 probabilities for a
// following tensor-core value GEMM. The attention sink contributes to the
// denominator exactly as in the established sparse-output kernel, but has no
// value row and therefore is not written to the probability matrix.
extern "C" __global__ void deepseek_v4_sparse_softmax_bf16_kernel(
    __nv_bfloat16* __restrict__ probabilities,
    const float* __restrict__ scores,
    const float* __restrict__ attention_sink,
    int tokens,
    int heads,
    int topk)
{
    int head = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    if (head >= heads || token >= tokens || probabilities == nullptr ||
        scores == nullptr || attention_sink == nullptr || topk <= 0) return;
    const float* score_row = scores + ((int64_t)token * heads + head) * topk;
    __nv_bfloat16* probability_row =
        probabilities + ((int64_t)token * heads + head) * topk;
    extern __shared__ float reduction[];
    float local_max = threadIdx.x == 0 ? attention_sink[head] : -INFINITY;
    for (int selected = (int)threadIdx.x; selected < topk;
         selected += (int)blockDim.x) {
        local_max = fmaxf(local_max, score_row[selected]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = reduction[0];
    float local_sum = threadIdx.x == 0
        ? expf(attention_sink[head] - row_max)
        : 0.0f;
    for (int selected = (int)threadIdx.x; selected < topk;
         selected += (int)blockDim.x) {
        float value = score_row[selected];
        local_sum += isfinite(value) ? expf(value - row_max) : 0.0f;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse_denominator = 1.0f / reduction[0];
    for (int selected = (int)threadIdx.x; selected < topk;
         selected += (int)blockDim.x) {
        float value = score_row[selected];
        float probability = isfinite(value)
            ? expf(value - row_max) * inverse_denominator
            : 0.0f;
        probability_row[selected] = dsv4_attn_f32_to_bf16(probability);
    }
}
