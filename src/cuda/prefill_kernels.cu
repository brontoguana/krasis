/*
 * Krasis Prefill Kernels — GPU prefill without Python/PyTorch.
 *
 * Simple element-wise and reduction kernels compiled to PTX via nvcc.
 * Complex kernels (Marlin GEMM, attention, Mamba2 SSD) are in separate files.
 *
 * All functions follow the C API defined in prefill_shim.h.
 * BF16 = unsigned short (cuda_bf16.h nv_bfloat16).
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>

/* ── Helpers ───────────────────────────────────────────────────────────── */

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ void apply_swiglu_limit(float &silu_gate, float &up, float limit) {
    if (limit > 0.0f) {
        silu_gate = fminf(silu_gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
}

extern "C" __device__ float __nv_logf(float);
extern "C" __device__ float __nv_expf(float);

__device__ __forceinline__ float mamba2_ssd_exp_a_log(float a_log) {
    return __nv_expf(a_log);
}

__device__ __forceinline__ float mamba2_ssd_a_value(float a_log) {
    return -mamba2_ssd_exp_a_log(a_log);
}

__device__ __forceinline__ float mamba2_chunk_cumsum_softplus(float x) {
    return __nv_logf(1.0f + __expf(x));
}

__device__ __forceinline__ float mamba2_ssd_dt_value(
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ dt_bias,
    int t,
    int n_heads,
    int head,
    int use_softplus)
{
    float dt = bf16_to_float(dt_in[t * n_heads + head]);
    if (dt_bias != NULL) dt += dt_bias[head];
    if (use_softplus) dt = mamba2_chunk_cumsum_softplus(dt);
    return dt;
}

__device__ __forceinline__ float mamba2_ssd_cb_dot_reverse(
    const __nv_bfloat16* __restrict__ C_mat,
    const __nv_bfloat16* __restrict__ B_mat,
    int c_row_base_idx,
    int b_row_base_idx,
    int state_size)
{
    float cb = 0.0f;
    for (int s = state_size - 1; s >= 0; s--) {
        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
        float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
        cb += C_val * B_val;
    }
    return cb;
}

__device__ __forceinline__ void mamba2_ssd_timing_add(
    unsigned long long* __restrict__ timing,
    int idx,
    unsigned long long cycles)
{
    if (timing != NULL && cycles != 0ULL) {
        atomicAdd(&timing[idx], cycles);
    }
}

__device__ __forceinline__ int mamba2_ssd_lower_tri_pair_index(int t_pos, int u_pos)
{
    return (t_pos * (t_pos + 1)) / 2 + u_pos;
}

__device__ __forceinline__ void mamba2_ssd_lower_tri_pair_decode(
    int pair_idx,
    int* __restrict__ t_pos,
    int* __restrict__ u_pos)
{
    int row = (int)((sqrtf((float)(8 * pair_idx + 1)) - 1.0f) * 0.5f);
    while ((row * (row + 1)) / 2 > pair_idx) row--;
    while (((row + 1) * (row + 2)) / 2 <= pair_idx) row++;
    *t_pos = row;
    *u_pos = pair_idx - (row * (row + 1)) / 2;
}

__device__ __forceinline__ float* mamba2_ssd_build_da_prefix_scan(
    float* da_smem,
    int chunk_capacity,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ dt_bias,
    float A_val,
    int n_heads,
    int head,
    int chunk_start,
    int chunk_len,
    int use_softplus)
{
    float* src = da_smem;
    float* dst = da_smem + chunk_capacity;
    for (int i = threadIdx.x; i < chunk_len; i += blockDim.x) {
        float dt = mamba2_ssd_dt_value(dt_in, dt_bias, chunk_start + i, n_heads, head, use_softplus);
        src[i] = A_val * dt;
    }
    __syncthreads();

    for (int offset = 1; offset < chunk_len; offset <<= 1) {
        for (int i = threadIdx.x; i < chunk_len; i += blockDim.x) {
            float value = src[i];
            if (i >= offset) value += src[i - offset];
            dst[i] = value;
        }
        __syncthreads();
        float* tmp = src;
        src = dst;
        dst = tmp;
    }
    return src;
}

__device__ __forceinline__ __nv_fp8_e4m3 f32_to_fp8e4m3(float x) {
    return __nv_fp8_e4m3(x);
}

__device__ __forceinline__ __nv_fp8_e4m3 bf16_to_fp8e4m3(__nv_bfloat16 x) {
    return f32_to_fp8e4m3(bf16_to_float(x));
}

__device__ __forceinline__ float trace_sqrt_rn_f32(float x) {
    float y;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float trace_div_rn_f32(float a, float b) {
    float y;
    asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float trace_rcp_rn_f32(float x) {
    float y;
    asm volatile("rcp.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float trace_sqrt_approx_f32(float x) {
    float y;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float trace_div_approx_f32(float a, float b) {
    float y;
    asm volatile("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float trace_rsqrt_approx_f32(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float mamba2_gated_rmsnorm_triton_rstd(float x) {
    float sqrt_approx;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(sqrt_approx) : "f"(x));
    float y;
    asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(1.0f), "f"(sqrt_approx));
    return y;
}

__device__ __forceinline__ float trace_mul_rn_f32(float a, float b) {
    float y;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float trace_add_rn_f32(float a, float b) {
    float y;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float trace_fma_rn_f32(float a, float b, float c) {
    float y;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
    return y;
}

__device__ __forceinline__ unsigned long long trace_mix_u64(
    unsigned long long h,
    unsigned long long v)
{
    h ^= v;
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ void gated_rmsnorm_adjacent_pairwise_reduce(float* smem) {
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int span = stride << 1;
        if (((threadIdx.x & (span - 1)) == 0) && threadIdx.x + stride < blockDim.x) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

/* ── Batched RMSNorm ──────────────────────────────────────────────────── */

/* One block per token. Shared memory reduction for variance. */
extern "C" __global__ void rmsnorm_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;

    extern __shared__ float smem[];

    /* Compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    /* Tree reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    /* Normalize and scale */
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * bf16_to_float(weight[i]));
    }
}

extern "C" __global__ void rmsnorm_batched_fp32w_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;

    extern __shared__ float smem[];

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * weight[i]);
    }
}

/* Diagnostic candidate: contiguous per-thread chunks before the block
 * reduction. This preserves parallel output writes and avoids a serial row
 * loop while testing whether reduction order can match HF/index-order stores.
 */
extern "C" __global__ void rmsnorm_batched_contig_reduce_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;

    extern __shared__ float smem[];

    float local_ss = 0.0f;
    int elems_per_thread = (D + blockDim.x - 1) / blockDim.x;
    int start = threadIdx.x * elems_per_thread;
    int end = min(D, start + elems_per_thread);
    for (int i = start; i < end; ++i) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * bf16_to_float(weight[i]));
    }
}

extern "C" void krasis_rmsnorm_batched(
    void* out, const void* x, const void* weight,
    int M, int D, float eps, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, D);
    /* Round up to next warp */
    threads = ((threads + 31) / 32) * 32;
    int smem = threads * sizeof(float);
    rmsnorm_batched_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)weight, D, eps);
}

/* ── Fused Add + RMSNorm ──────────────────────────────────────────────── */

extern "C" __global__ void fused_add_rmsnorm_batched_kernel(
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    __nv_bfloat16* res_row = residual + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;

    extern __shared__ float smem[];

    /* First pass: add and compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float r = bf16_to_float(res_row[i]) + bf16_to_float(x_row[i]);
        __nv_bfloat16 rounded = float_to_bf16(r);
        res_row[i] = rounded;
        float v = bf16_to_float(rounded);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    /* Second pass: normalize */
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(res_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * bf16_to_float(weight[i]));
    }
}

extern "C" void krasis_fused_add_rmsnorm_batched(
    void* residual, void* out, const void* x, const void* weight,
    int M, int D, float eps, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, D);
    threads = ((threads + 31) / 32) * 32;
    int smem = threads * sizeof(float);
    fused_add_rmsnorm_batched_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)residual, (__nv_bfloat16*)out,
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)weight, D, eps);
}

extern "C" __global__ void add_bf16_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int D)
{
    int token = blockIdx.x;
    const __nv_bfloat16* a_row = a + (int64_t)token * D;
    const __nv_bfloat16* b_row = b + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        o_row[i] = float_to_bf16(bf16_to_float(a_row[i]) + bf16_to_float(b_row[i]));
    }
}

extern "C" __global__ void scale_bf16_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    float scale,
    int D)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        o_row[i] = float_to_bf16(bf16_to_float(x_row[i]) * scale);
    }
}

extern "C" __global__ void scale_bf16_by_ptr_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale_ptr,
    int D)
{
    int token = blockIdx.x;
    const float scale = scale_ptr ? bf16_to_float(scale_ptr[0]) : 1.0f;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        o_row[i] = float_to_bf16(bf16_to_float(x_row[i]) * scale);
    }
}

extern "C" __global__ void apply_topk_per_expert_scale_kernel(
    float* __restrict__ topk_weights,
    const int* __restrict__ topk_ids,
    const __nv_bfloat16* __restrict__ per_expert_scale,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total || per_expert_scale == nullptr) return;
    int eid = topk_ids[i];
    if (eid >= 0) {
        topk_weights[i] *= bf16_to_float(per_expert_scale[eid]);
    }
}

extern "C" __global__ void rmsnorm_scale_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale_weight,
    float scale,
    int D,
    float eps)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;

    extern __shared__ float smem[];

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms_inv * scale;
        if (scale_weight != nullptr) {
            v *= bf16_to_float(scale_weight[i]);
        }
        o_row[i] = float_to_bf16(v);
    }
}

/* ── Embedding Lookup ──────────────────────────────────────────────────── */

extern "C" __global__ void embedding_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ token_ids,
    int D,
    float scale)
{
    int token = blockIdx.x;
    int tid = token_ids[token];
    const __nv_bfloat16* src = table + (int64_t)tid * D;
    __nv_bfloat16* dst = out + (int64_t)token * D;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = float_to_bf16(bf16_to_float(src[i]) * scale);
    }
}

extern "C" void krasis_embedding_batched(
    void* out, const void* table, const void* token_ids,
    int M, int D, float scale, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, D);
    threads = ((threads + 31) / 32) * 32;
    embedding_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)table,
        (const int*)token_ids, D, scale);
}

/* ── HQQ4 Dequant ─────────────────────────────────────────────────────── */

extern "C" __global__ void hqq4_dequant_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int rows,
    int cols,
    int group_size)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx - row * cols;
    int groups = (cols + group_size - 1) / group_size;
    int packed_cols = (groups * group_size) / 2;
    int group = col / group_size;
    int packed_idx = row * packed_cols + (col >> 1);
    unsigned char byte = packed[packed_idx];
    int q = (col & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
    float scale = scales[row * groups + group];
    float zero = zeros[row * groups + group];
    out[idx] = float_to_bf16((float(q) - zero) * scale);
}

extern "C" void krasis_hqq4_dequant_bf16(
    void* out,
    const void* packed,
    const void* scales,
    const void* zeros,
    int rows,
    int cols,
    int group_size,
    void* stream)
{
    int total = rows * cols;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hqq4_dequant_bf16_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out,
        (const unsigned char*)packed,
        (const float*)scales,
        (const float*)zeros,
        rows,
        cols,
        group_size);
}

__device__ __forceinline__ int hqq_group_idx(int col, int group_size) {
    return group_size == 128 ? (col >> 7) : (col / group_size);
}

template<int NBITS>
__device__ __forceinline__ float hqq_load_weight(
    const unsigned char* __restrict__ row,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int col,
    int group_size)
{
    int q;
    if constexpr (NBITS == 4) {
        unsigned char packed = row[col >> 1];
        q = (col & 1) ? (int)(packed >> 4) : (int)(packed & 0x0F);
    } else if constexpr (NBITS == 6) {
        int group4 = col >> 2;
        int offset = col & 3;
        const unsigned char* tri = row + group4 * 3;
        unsigned int bits = ((unsigned int)tri[0]) | (((unsigned int)tri[1]) << 8) | (((unsigned int)tri[2]) << 16);
        q = (bits >> (offset * 6)) & 0x3F;
    } else {
        q = (int)row[col];
    }
    int group = hqq_group_idx(col, group_size);
    return ((float)q - zeros[group]) * scales[group];
}

template<int NBITS>
__device__ __forceinline__ void hqq_dequant_bf16_device(
    __nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx - row * cols;
    const unsigned char* w_row = packed + (long long)row * packed_row_stride_bytes;
    const float* s_row = (const float*)((const char*)scales + (long long)row * scales_row_stride_bytes);
    const float* z_row = (const float*)((const char*)zeros + (long long)row * zeros_row_stride_bytes);
    out[idx] = float_to_bf16(hqq_load_weight<NBITS>(w_row, s_row, z_row, col, group_size));
}

extern "C" __global__ void hqq_dequant_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes,
    int nbits)
{
    if (nbits == 4) {
        hqq_dequant_bf16_device<4>(
            out, packed, scales, zeros, rows, cols, group_size,
            packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
    } else if (nbits == 6) {
        hqq_dequant_bf16_device<6>(
            out, packed, scales, zeros, rows, cols, group_size,
            packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
    } else if (nbits == 8) {
        hqq_dequant_bf16_device<8>(
            out, packed, scales, zeros, rows, cols, group_size,
            packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
    }
}

template<int NBITS>
__device__ void hqq_quantized_prefill_gemm_bf16_device(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    __shared__ float partial[256];
    int tid = threadIdx.x;
    int lane = tid & 3;
    int out_lane = tid >> 2;
    int local_m = out_lane >> 3;
    int local_row = out_lane & 7;
    int token = blockIdx.y * 8 + local_m;
    int row = blockIdx.x * 8 + local_row;
    if (token >= M || row >= rows) return;

    const __nv_bfloat16* x_row = input + (long long)token * cols;
    const unsigned char* w_row = packed + (long long)row * packed_row_stride_bytes;
    const float* s_row = (const float*)((const char*)scales + (long long)row * scales_row_stride_bytes);
    const float* z_row = (const float*)((const char*)zeros + (long long)row * zeros_row_stride_bytes);

    float acc = 0.0f;
    for (int col = lane; col < cols; col += 4) {
        float w = hqq_load_weight<NBITS>(w_row, s_row, z_row, col, group_size);
        float x = bf16_to_float(x_row[col]);
        acc += w * x;
    }

    partial[tid] = acc;
    __syncthreads();
    if (lane == 0) {
        float sum = partial[tid] + partial[tid + 1] + partial[tid + 2] + partial[tid + 3];
        out[(long long)token * rows + row] = float_to_bf16(sum);
    }
}

extern "C" __global__ void hqq4_prefill_gemm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    hqq_quantized_prefill_gemm_bf16_device<4>(
        out, input, packed, scales, zeros, M, rows, cols, group_size,
        packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
}

extern "C" __global__ void hqq8_prefill_gemm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    hqq_quantized_prefill_gemm_bf16_device<8>(
        out, input, packed, scales, zeros, M, rows, cols, group_size,
        packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
}

extern "C" __global__ void hqq6_prefill_gemm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    hqq_quantized_prefill_gemm_bf16_device<6>(
        out, input, packed, scales, zeros, M, rows, cols, group_size,
        packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
}

extern "C" __global__ void hqq_prefill_group_sums_bf16_kernel(
    float* __restrict__ group_sums,
    const __nv_bfloat16* __restrict__ input,
    int M,
    int cols,
    int group_size,
    int groups)
{
    constexpr int GROUPS_PER_BLOCK = 8;
    int token = blockIdx.x;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = blockIdx.y * GROUPS_PER_BLOCK + warp;
    if (token >= M || group >= groups) return;

    int start = group * group_size;
    int end = min(start + group_size, cols);
    float acc = 0.0f;
    const __nv_bfloat16* x_row = input + (long long)token * cols;
    for (int col = start + lane; col < end; col += 32) {
        acc += bf16_to_float(x_row[col]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        group_sums[(long long)token * groups + group] = acc;
    }
}

extern "C" __global__ void hqq8_marlin_zero_correct_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ delta_out,
    const float* __restrict__ group_sums,
    const float* __restrict__ zero_correction,
    int M,
    int rows,
    int groups)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * rows;
    if (idx >= total) return;

    int row = (int)(idx % rows);
    int token = (int)(idx / rows);
    float acc = 0.0f;
    const float* sums = group_sums + (long long)token * groups;
    const float* corr = zero_correction + (long long)row * groups;
    for (int g = 0; g < groups; ++g) {
        acc += sums[g] * corr[g];
    }
    __nv_bfloat16 base = out[idx];
    __nv_bfloat16 delta = delta_out[idx];
    out[idx] = float_to_bf16(bf16_to_float(base) + bf16_to_float(delta) + acc);
}

extern "C" __global__ void hqq8_marlin_intercept_correct_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ group_sums,
    const float* __restrict__ intercept_correction,
    int M,
    int rows,
    int groups)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * rows;
    if (idx >= total) return;

    int row = (int)(idx % rows);
    int token = (int)(idx / rows);
    float acc = 0.0f;
    const float* sums = group_sums + (long long)token * groups;
    const float* corr = intercept_correction + (long long)row * groups;
    for (int g = 0; g < groups; ++g) {
        acc += sums[g] * corr[g];
    }
    out[idx] = float_to_bf16(bf16_to_float(out[idx]) + acc);
}

extern "C" __global__ void hqq_marlin_add_correction_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ correction,
    int total)
{
    int idx = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;
    out[idx] = float_to_bf16(bf16_to_float(out[idx]) + correction[idx]);
}

extern "C" __global__ void hqq_prefill_int8_exception_delta_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const signed char* __restrict__ exception_qint8,
    const float* __restrict__ exception_scales,
    const int* __restrict__ output_rows,
    const int* __restrict__ start_cols,
    const int* __restrict__ widths,
    const float* __restrict__ hqq_base_f32,
    const unsigned char* __restrict__ hqq_packed_w,
    const float* __restrict__ hqq_scales,
    const float* __restrict__ hqq_zeros,
    int M,
    int rows,
    int row_group_count,
    int cols,
    int group_size,
    int max_width,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes,
    int nbits)
{
    extern __shared__ float smem[];
    int entry = blockIdx.x;
    int token = blockIdx.y;
    int tid = threadIdx.x;
    if (entry >= row_group_count || token >= M) return;

    int row = output_rows[entry];
    int start_col = start_cols[entry];
    int width = widths[entry];
    const __nv_bfloat16* x_row = input + (long long)token * cols;
    float exc_scale = exception_scales[entry];

    float acc = 0.0f;
    for (int local = tid; local < width; local += blockDim.x) {
        int col = start_col + local;
        if (col >= cols) continue;
        float hqq_w;
        if (hqq_base_f32 != nullptr) {
            hqq_w = hqq_base_f32[entry * max_width + local];
        } else {
            const unsigned char* w_row = hqq_packed_w + (long long)row * packed_row_stride_bytes;
            const float* s_row = (const float*)((const char*)hqq_scales + (long long)row * scales_row_stride_bytes);
            const float* z_row = (const float*)((const char*)hqq_zeros + (long long)row * zeros_row_stride_bytes);
            int q;
            if (nbits == 4) {
                unsigned char packed_byte = w_row[col >> 1];
                q = (col & 1) ? (int)(packed_byte >> 4) : (int)(packed_byte & 0x0F);
            } else {
                q = (int)w_row[col];
            }
            int group = hqq_group_idx(col, group_size);
            hqq_w = ((float)q - z_row[group]) * s_row[group];
        }
        float int8_w = (float)exception_qint8[entry * max_width + local] * exc_scale;
        float x = bf16_to_float(x_row[col]);
        acc += (int8_w - hqq_w) * x;
    }

    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        __nv_bfloat16 base = out[(long long)token * rows + row];
        out[(long long)token * rows + row] = float_to_bf16(bf16_to_float(base) + smem[0]);
    }
}

extern "C" __global__ void hqq_apply_sidecar_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const signed char* __restrict__ correction_qint8,
    const __nv_bfloat16* __restrict__ correction_bf16,
    const float* __restrict__ scales,
    const int* __restrict__ output_rows,
    const int* __restrict__ start_cols,
    const int* __restrict__ widths,
    int row_group_count,
    int cols,
    int max_width,
    int mode)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = row_group_count * max_width;
    if (idx >= total) return;

    int entry = idx / max_width;
    int local = idx - entry * max_width;
    int width = widths[entry];
    if (local >= width) return;

    int row = output_rows[entry];
    int col = start_cols[entry] + local;
    float value = 0.0f;
    if (mode == 1) {
        value = (float)correction_qint8[idx] * scales[entry];
    } else if (mode == 2) {
        value = bf16_to_float(correction_bf16[idx]);
    } else if (mode == 3) {
        value = (float)correction_qint8[idx] * scales[entry];
    } else {
        return;
    }
    int out_idx = row * cols + col;
    if (mode == 3) {
        out[out_idx] = float_to_bf16(value);
    } else {
        float base = bf16_to_float(out[out_idx]);
        out[out_idx] = float_to_bf16(base + value);
    }
}

/* ── RoPE (Rotary Position Embedding) ─────────────────────────────────── */

/* Apply RoPE to Q and K tensors in-place.
 * Layout: [M, num_heads, head_dim] bf16
 * cos/sin cache: [max_pos, head_dim/2] bf16
 */
extern "C" __global__ void rope_batched_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int* __restrict__ positions,
    const float* __restrict__ cos_cache,   /* FP32 [max_seq, half_dim] */
    const float* __restrict__ sin_cache,   /* FP32 [max_seq, half_dim] */
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim)
{
    int token = blockIdx.x;
    int pos = positions[token];

    const float* cos_row = cos_cache + (int64_t)pos * half_dim;
    const float* sin_row = sin_cache + (int64_t)pos * half_dim;

    /* Apply to Q heads */
    int q_stride = num_q_heads * head_dim;
    __nv_bfloat16* q_row = q + (int64_t)token * q_stride;
    for (int h = 0; h < num_q_heads; h++) {
        __nv_bfloat16* qh = q_row + h * head_dim;
        for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float q0 = bf16_to_float(qh[i]);
            float q1 = bf16_to_float(qh[i + half_dim]);
            float c = cos_row[i];
            float s = sin_row[i];
            qh[i] = float_to_bf16(q0 * c - q1 * s);
            qh[i + half_dim] = float_to_bf16(q1 * c + q0 * s);
        }
    }

    /* Apply to K heads */
    int k_stride = num_kv_heads * head_dim;
    __nv_bfloat16* k_row = k + (int64_t)token * k_stride;
    for (int h = 0; h < num_kv_heads; h++) {
        __nv_bfloat16* kh = k_row + h * head_dim;
        for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float k0 = bf16_to_float(kh[i]);
            float k1 = bf16_to_float(kh[i + half_dim]);
            float c = cos_row[i];
            float s = sin_row[i];
            kh[i] = float_to_bf16(k0 * c - k1 * s);
            kh[i + half_dim] = float_to_bf16(k1 * c + k0 * s);
        }
    }
}

/* Apply RoPE with the HuggingFace rotate_half layout:
 * rotate_half([x0..xH-1, y0..yH-1]) = [-y0..-yH-1, x0..xH-1].
 * Gemma4 uses this layout for text RoPE. The existing rope_batched_kernel is
 * kept for models using pairwise rotary layout.
 */
extern "C" __global__ void rope_batched_half_split_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int* __restrict__ positions,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim)
{
    int token = blockIdx.x;
    int pos = positions[token];

    const float* cos_row = cos_cache + (int64_t)pos * half_dim;
    const float* sin_row = sin_cache + (int64_t)pos * half_dim;

    int q_stride = num_q_heads * head_dim;
    __nv_bfloat16* q_row = q + (int64_t)token * q_stride;
    int rotary_dim = half_dim * 2;
    if (rotary_dim > head_dim) rotary_dim = head_dim;
    for (int h = 0; h < num_q_heads; h++) {
        __nv_bfloat16* qh = q_row + h * head_dim;
        for (int i = threadIdx.x; i < rotary_dim; i += blockDim.x) {
            int src = i < half_dim ? i + half_dim : i - half_dim;
            float x = bf16_to_float(qh[i]);
            float r = bf16_to_float(qh[src]);
            if (i < half_dim) r = -r;
            int ci = i < half_dim ? i : i - half_dim;
            float c = cos_row[ci];
            float s = sin_row[ci];
            qh[i] = float_to_bf16(x * c + r * s);
        }
    }

    int k_stride = num_kv_heads * head_dim;
    __nv_bfloat16* k_row = k + (int64_t)token * k_stride;
    for (int h = 0; h < num_kv_heads; h++) {
        __nv_bfloat16* kh = k_row + h * head_dim;
        for (int i = threadIdx.x; i < rotary_dim; i += blockDim.x) {
            int src = i < half_dim ? i + half_dim : i - half_dim;
            float x = bf16_to_float(kh[i]);
            float r = bf16_to_float(kh[src]);
            if (i < half_dim) r = -r;
            int ci = i < half_dim ? i : i - half_dim;
            float c = cos_row[ci];
            float s = sin_row[ci];
            kh[i] = float_to_bf16(x * c + r * s);
        }
    }
}

/* Apply RoPE using per-token precomputed MRoPE rows.
 * Layout is identical to rope_batched_kernel, but cos/sin rows are already
 * expanded to [M, half_dim] for the current prompt chunk.
 */
extern "C" __global__ void rope_batched_mrope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ cos_rows,
    const float* __restrict__ sin_rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim)
{
    int token = blockIdx.x;
    const float* cos_row = cos_rows + (int64_t)token * half_dim;
    const float* sin_row = sin_rows + (int64_t)token * half_dim;

    int q_stride = num_q_heads * head_dim;
    __nv_bfloat16* q_row = q + (int64_t)token * q_stride;
    for (int h = 0; h < num_q_heads; h++) {
        __nv_bfloat16* qh = q_row + h * head_dim;
        for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float q0 = bf16_to_float(qh[i]);
            float q1 = bf16_to_float(qh[i + half_dim]);
            float c = cos_row[i];
            float s = sin_row[i];
            qh[i] = float_to_bf16(q0 * c - q1 * s);
            qh[i + half_dim] = float_to_bf16(q1 * c + q0 * s);
        }
    }

    int k_stride = num_kv_heads * head_dim;
    __nv_bfloat16* k_row = k + (int64_t)token * k_stride;
    for (int h = 0; h < num_kv_heads; h++) {
        __nv_bfloat16* kh = k_row + h * head_dim;
        for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float k0 = bf16_to_float(kh[i]);
            float k1 = bf16_to_float(kh[i + half_dim]);
            float c = cos_row[i];
            float s = sin_row[i];
            kh[i] = float_to_bf16(k0 * c - k1 * s);
            kh[i + half_dim] = float_to_bf16(k1 * c + k0 * s);
        }
    }
}

/* ── MLA prefill transforms ───────────────────────────────────────────── */

/*
 * Normalize only the compressed-latent prefix of each kv_a projection row.
 * The positional tail is deliberately preserved byte-for-byte for RoPE.
 *
 * x layout: [M, kv_lora_rank + qk_rope_dim] BF16
 * weight:   [kv_lora_rank] FP32
 */
extern "C" __global__ void mla_rmsnorm_prefix_bf16_fp32w_kernel(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    int row_stride,
    int norm_dim,
    float eps)
{
    int token = blockIdx.x;
    __nv_bfloat16* row = x + (int64_t)token * row_stride;
    extern __shared__ float smem[];

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < norm_dim; i += blockDim.x) {
        float v = bf16_to_float(row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)norm_dim + eps);
    for (int i = threadIdx.x; i < norm_dim; i += blockDim.x) {
        row[i] = float_to_bf16(bf16_to_float(row[i]) * rms_inv * weight[i]);
    }
}

// Build every prompt key row needed by one GLM DSA owner. The layout and
// BF16 boundaries match dsa_layernorm_rope_key_write in decode.
extern "C" __global__ void dsa_prefill_layernorm_rope_key_write_kernel(
    __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ raw_keys,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ norm_bias,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int token_count,
    int max_context,
    int head_dim,
    int rope_dim,
    float eps)
{
    int token = blockIdx.x;
    if (token < 0 || token >= token_count || head_dim <= 0 ||
        rope_dim < 0 || rope_dim > head_dim || (rope_dim & 1) != 0) {
        return;
    }
    int position = positions[token];
    if (position < 0 || position >= max_context) {
        return;
    }

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid & (warpSize - 1);
    int num_warps = (num_threads + warpSize - 1) / warpSize;
    const __nv_bfloat16* raw_key = raw_keys + (int64_t)token * head_dim;

    extern __shared__ float shared[];
    float* normalized = shared;
    float* warp_sum = normalized + head_dim;
    float* warp_sq = warp_sum + 32;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int i = tid; i < head_dim; i += num_threads) {
        float value = bf16_to_float(raw_key[i]);
        local_sum += value;
        local_sq += value * value;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
    }
    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sq[warp_id] = local_sq;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int warp = 0; warp < num_warps; warp++) {
            sum += warp_sum[warp];
            sum_sq += warp_sq[warp];
        }
        float mean = sum / (float)head_dim;
        float variance = fmaxf(sum_sq / (float)head_dim - mean * mean, 0.0f);
        warp_sum[0] = mean;
        warp_sq[0] = rsqrtf(variance + eps);
    }
    __syncthreads();

    float mean = warp_sum[0];
    float inv_std = warp_sq[0];
    for (int i = tid; i < head_dim; i += num_threads) {
        float value = (bf16_to_float(raw_key[i]) - mean) * inv_std;
        value = value * bf16_to_float(norm_weight[i]) + bf16_to_float(norm_bias[i]);
        normalized[i] = bf16_to_float(float_to_bf16(value));
    }
    __syncthreads();

    int half_rope = rope_dim / 2;
    __nv_bfloat16* row = key_cache + (int64_t)position * head_dim;
    for (int i = tid; i < head_dim; i += num_threads) {
        float value = normalized[i];
        if (i < rope_dim) {
            int table_idx = i < half_rope ? i : i - half_rope;
            float even = normalized[2 * table_idx];
            float odd = normalized[2 * table_idx + 1];
            float cos_value = bf16_to_float(float_to_bf16(
                cos_table[(int64_t)position * half_rope + table_idx]));
            float sin_value = bf16_to_float(float_to_bf16(
                sin_table[(int64_t)position * half_rope + table_idx]));
            float direct = bf16_to_float(float_to_bf16(
                (i < half_rope ? even : odd) * cos_value));
            float cross = bf16_to_float(float_to_bf16(
                (i < half_rope ? odd : even) * sin_value));
            value = bf16_to_float(float_to_bf16(
                i < half_rope ? direct - cross : direct + cross));
        }
        row[i] = float_to_bf16(value);
    }
}

// Apply the GLM DSA interleaved-input RoPE contract to a tile of owner
// queries. Rows remain independent so the same kernel can serve any
// runtime-derived prefill tile size.
extern "C" __global__ void dsa_prefill_rope_query_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int row_count,
    int num_heads,
    int head_dim,
    int rope_dim)
{
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row_width = (int64_t)num_heads * head_dim;
    int64_t total = (int64_t)row_count * row_width;
    if (linear >= total || positions == nullptr || head_dim <= 0 ||
        rope_dim < 0 || rope_dim > head_dim || (rope_dim & 1) != 0) {
        return;
    }
    int row = (int)(linear / row_width);
    int dim = (int)(linear % head_dim);
    int64_t head_base = linear - dim;
    int position = positions[row];
    float value = bf16_to_float(input[linear]);
    if (dim < rope_dim) {
        int half_rope = rope_dim / 2;
        int table_idx = dim < half_rope ? dim : dim - half_rope;
        float even = bf16_to_float(input[head_base + 2 * table_idx]);
        float odd = bf16_to_float(input[head_base + 2 * table_idx + 1]);
        float cos_value = bf16_to_float(float_to_bf16(
            cos_table[(int64_t)position * half_rope + table_idx]));
        float sin_value = bf16_to_float(float_to_bf16(
            sin_table[(int64_t)position * half_rope + table_idx]));
        float direct = bf16_to_float(float_to_bf16(
            (dim < half_rope ? even : odd) * cos_value));
        float cross = bf16_to_float(float_to_bf16(
            (dim < half_rope ? odd : even) * sin_value));
        value = bf16_to_float(float_to_bf16(
            dim < half_rope ? direct - cross : direct + cross));
    }
    output[linear] = float_to_bf16(value);
}

// Compute the exact owner score matrix for a bounded prefill row tile:
// sum_h(weight_h * relu(dot(query_h, key_token) * score_scale)).
// Each row uses its absolute position as the inclusive causal boundary.
extern "C" __global__ void dsa_prefill_fused_scores_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ queries,
    const __nv_bfloat16* __restrict__ head_weights,
    const int* __restrict__ positions,
    int row_count,
    int context_end,
    int num_heads,
    int head_dim,
    float score_scale)
{
    int row = blockIdx.y;
    if (row < 0 || row >= row_count || positions == nullptr ||
        context_end <= 0 || num_heads <= 0 || head_dim <= 0 ||
        (blockDim.x & 31) != 0) {
        return;
    }
    int context = min(max(positions[row] + 1, 0), context_end);
    int num_warps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    extern __shared__ unsigned char shared_raw[];
    float* shared_key = reinterpret_cast<float*>(shared_raw);
    float* shared_contributions = shared_key + head_dim;
    const __nv_bfloat16* row_queries =
        queries + (int64_t)row * num_heads * head_dim;
    const __nv_bfloat16* row_weights =
        head_weights + (int64_t)row * num_heads;
    float* row_output = output + (int64_t)row * context_end;

    for (int token = blockIdx.x; token < context; token += gridDim.x) {
        for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            shared_key[dim] = bf16_to_float(
                key_cache[(int64_t)token * head_dim + dim]);
        }
        __syncthreads();

        for (int head = warp; head < num_heads; head += num_warps) {
            float dot = 0.0f;
            const __nv_bfloat16* head_query =
                row_queries + (int64_t)head * head_dim;
            for (int dim = lane; dim < head_dim; dim += 32) {
                dot += bf16_to_float(head_query[dim]) * shared_key[dim];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                shared_contributions[head] =
                    bf16_to_float(row_weights[head]) *
                    fmaxf(dot * score_scale, 0.0f);
            }
        }
        __syncthreads();

        if (warp == 0) {
            float score = 0.0f;
            for (int head = lane; head < num_heads; head += 32) {
                score += shared_contributions[head];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
            if (lane == 0) row_output[token] = score;
        }
        __syncthreads();
    }
}

__device__ inline bool dsa_prefill_topk_precedes(
    float lhs_score,
    int lhs_index,
    float rhs_score,
    int rhs_index)
{
    if (lhs_score > rhs_score) return true;
    if (lhs_score < rhs_score) return false;
    return lhs_index < rhs_index;
}

__device__ inline void dsa_prefill_topk_bitonic_sort(
    float* scores,
    int* indices,
    int width)
{
    for (int sequence = 2; sequence <= width; sequence <<= 1) {
        for (int stride = sequence >> 1; stride > 0; stride >>= 1) {
            for (int left = threadIdx.x; left < width; left += blockDim.x) {
                int right = left ^ stride;
                if (right <= left) continue;
                bool descending = (left & sequence) == 0;
                bool right_precedes = dsa_prefill_topk_precedes(
                    scores[right], indices[right], scores[left], indices[left]);
                bool left_precedes = dsa_prefill_topk_precedes(
                    scores[left], indices[left], scores[right], indices[right]);
                bool swap = descending ? right_precedes : left_precedes;
                if (swap) {
                    float score = scores[left];
                    scores[left] = scores[right];
                    scores[right] = score;
                    int index = indices[left];
                    indices[left] = indices[right];
                    indices[right] = index;
                }
            }
            __syncthreads();
        }
    }
}

// Batched exact base sorts for chunk-local prefill rows. Candidate storage is
// row-strided at the maximum-context plan capacity. A one-run plan publishes
// directly to the retained-index matrix.
extern "C" __global__ void dsa_prefill_topk_sort_rows_kernel(
    const float* __restrict__ input_scores,
    float* __restrict__ candidate_scores,
    int* __restrict__ candidate_indices,
    int* __restrict__ selected_indices,
    const int* __restrict__ positions,
    int row_count,
    int context_end,
    int selected_stride,
    int candidate_stride,
    int initial_runs,
    int sort_width)
{
    int row = blockIdx.y;
    int run = blockIdx.x;
    if (row < 0 || row >= row_count || run < 0 || run >= initial_runs ||
        positions == nullptr || selected_indices == nullptr ||
        context_end <= 0 || selected_stride <= 0 || sort_width < selected_stride ||
        (sort_width & (sort_width - 1)) != 0) {
        return;
    }
    int context = min(max(positions[row] + 1, 0), context_end);
    extern __shared__ unsigned char shared_raw[];
    float* shared_scores = reinterpret_cast<float*>(shared_raw);
    int* shared_indices = reinterpret_cast<int*>(shared_scores + sort_width);
    int input_base = run * sort_width;
    const float* row_scores = input_scores + (int64_t)row * context_end;
    for (int item = threadIdx.x; item < sort_width; item += blockDim.x) {
        int token = input_base + item;
        shared_scores[item] =
            token < context ? row_scores[token] : -INFINITY;
        shared_indices[item] = token < context ? token : INT_MAX;
    }
    __syncthreads();
    dsa_prefill_topk_bitonic_sort(
        shared_scores, shared_indices, sort_width);

    int* row_selected =
        selected_indices + (int64_t)row * selected_stride;
    if (initial_runs == 1) {
        for (int item = threadIdx.x; item < selected_stride;
             item += blockDim.x) {
            row_selected[item] = shared_indices[item];
        }
        return;
    }
    if (candidate_scores == nullptr || candidate_indices == nullptr ||
        candidate_stride < initial_runs * selected_stride) {
        return;
    }
    int candidate_base =
        (int64_t)row * candidate_stride + run * selected_stride;
    for (int item = threadIdx.x; item < selected_stride;
         item += blockDim.x) {
        candidate_scores[candidate_base + item] = shared_scores[item];
        candidate_indices[candidate_base + item] = shared_indices[item];
    }
}

extern "C" __global__ void dsa_prefill_topk_merge_rows_kernel(
    const float* __restrict__ input_scores,
    const int* __restrict__ input_indices,
    float* __restrict__ output_scores,
    int* __restrict__ output_indices,
    int* __restrict__ selected_indices,
    int row_count,
    int selected_stride,
    int candidate_stride,
    int input_runs,
    int sort_width)
{
    int row = blockIdx.y;
    int output_run = blockIdx.x;
    int output_runs = (input_runs + 1) / 2;
    if (row < 0 || row >= row_count || output_run < 0 ||
        output_run >= output_runs || input_runs <= 1 ||
        selected_stride <= 0 || candidate_stride < input_runs * selected_stride ||
        sort_width < 2 * selected_stride ||
        (sort_width & (sort_width - 1)) != 0 ||
        input_scores == nullptr || input_indices == nullptr ||
        output_scores == nullptr || output_indices == nullptr ||
        selected_indices == nullptr) {
        return;
    }
    extern __shared__ unsigned char shared_raw[];
    float* shared_scores = reinterpret_cast<float*>(shared_raw);
    int* shared_indices = reinterpret_cast<int*>(shared_scores + sort_width);
    int first_run = output_run * 2;
    int second_run = first_run + 1;
    int64_t row_base = (int64_t)row * candidate_stride;
    for (int item = threadIdx.x; item < sort_width; item += blockDim.x) {
        int source_run = item < selected_stride ? first_run : second_run;
        int source_item =
            item < selected_stride ? item : item - selected_stride;
        bool valid =
            item < 2 * selected_stride && source_run < input_runs;
        int64_t source =
            row_base + (int64_t)source_run * selected_stride + source_item;
        shared_scores[item] = valid ? input_scores[source] : -INFINITY;
        shared_indices[item] = valid ? input_indices[source] : INT_MAX;
    }
    __syncthreads();
    dsa_prefill_topk_bitonic_sort(
        shared_scores, shared_indices, sort_width);

    int64_t output_base =
        row_base + (int64_t)output_run * selected_stride;
    int* row_selected =
        selected_indices + (int64_t)row * selected_stride;
    for (int item = threadIdx.x; item < selected_stride;
         item += blockDim.x) {
        output_scores[output_base + item] = shared_scores[item];
        output_indices[output_base + item] = shared_indices[item];
        if (output_runs == 1) row_selected[item] = shared_indices[item];
    }
}

/*
 * Pack head-major content K/V expansions into the token-major FA2 layout,
 * while applying RoPE to Q's positional tail and the shared positional K.
 *
 * q_in/q_out:     [M, H, nope+rope] BF16, separate to make the
 *                 interleaved-to-half-split transform race-free
 * k_out:          [M, H, nope+rope] BF16
 * v_out:          [M, H, v_dim] BF16
 * k_content:      [H, M, nope] BF16
 * v_content:      [H, M, v_dim] BF16
 * kv_a:           [M, kv_lora_rank+rope] BF16; positional tail is unrotated
 */
extern "C" __global__ void mla_pack_qkv_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q_out,
    const __nv_bfloat16* __restrict__ q_in,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    const __nv_bfloat16* __restrict__ k_content,
    const __nv_bfloat16* __restrict__ v_content,
    const __nv_bfloat16* __restrict__ kv_a,
    const int* __restrict__ positions,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int num_heads,
    int nope_dim,
    int rope_dim,
    int value_dim,
    int kv_row_stride,
    int token_count,
    int rope_interleave)
{
    int token = blockIdx.x;
    if (token >= token_count) return;

    int qk_dim = nope_dim + rope_dim;
    int half = rope_dim / 2;
    int pos = positions[token];
    const float* cos_row = cos_cache + (int64_t)pos * half;
    const float* sin_row = sin_cache + (int64_t)pos * half;
    const __nv_bfloat16* kv_row = kv_a + (int64_t)token * kv_row_stride;
    const __nv_bfloat16* kpe = kv_row + (kv_row_stride - rope_dim);

    for (int flat = threadIdx.x; flat < num_heads * nope_dim; flat += blockDim.x) {
        int head = flat / nope_dim;
        int d = flat % nope_dim;
        q_out[((int64_t)token * num_heads + head) * qk_dim + d] =
            q_in[((int64_t)token * num_heads + head) * qk_dim + d];
        k_out[((int64_t)token * num_heads + head) * qk_dim + d] =
            k_content[((int64_t)head * token_count + token) * nope_dim + d];
    }

    for (int flat = threadIdx.x; flat < num_heads * value_dim; flat += blockDim.x) {
        int head = flat / value_dim;
        int d = flat % value_dim;
        v_out[((int64_t)token * num_heads + head) * value_dim + d] =
            v_content[((int64_t)head * token_count + token) * value_dim + d];
    }

    for (int flat = threadIdx.x; flat < num_heads * half; flat += blockDim.x) {
        int head = flat / half;
        int d = flat % half;
        int q_base = ((int64_t)token * num_heads + head) * qk_dim + nope_dim;
        int q0_idx = rope_interleave ? 2 * d : d;
        int q1_idx = rope_interleave ? 2 * d + 1 : d + half;
        float q0 = bf16_to_float(q_in[q_base + q0_idx]);
        float q1 = bf16_to_float(q_in[q_base + q1_idx]);
        float c = cos_row[d];
        float s = sin_row[d];
        q_out[q_base + d] = float_to_bf16(q0 * c - q1 * s);
        q_out[q_base + half + d] = float_to_bf16(q1 * c + q0 * s);

        int k0_idx = rope_interleave ? 2 * d : d;
        int k1_idx = rope_interleave ? 2 * d + 1 : d + half;
        float k0 = bf16_to_float(kpe[k0_idx]);
        float k1 = bf16_to_float(kpe[k1_idx]);
        int k_base = ((int64_t)token * num_heads + head) * qk_dim + nope_dim;
        k_out[k_base + d] = float_to_bf16(k0 * c - k1 * s);
        k_out[k_base + half + d] = float_to_bf16(k1 * c + k0 * s);
    }
}

__device__ inline float mla_prefill_quantize_k4_one_pass_ls(
    const float* src,
    unsigned char* codes)
{
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) max_abs = fmaxf(max_abs, fabsf(src[i]));

    float scale = fmaxf(max_abs * (1.0f / 7.0f), 1e-8f);
    float inv_scale = 1.0f / scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-7, min(7, q));
        codes[i] = (unsigned char)(q + 8);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) scale = fmaxf(ls_num / ls_den, 1e-8f);
    return scale;
}

__device__ inline void mla_prefill_store_k4_block(
    unsigned char* dst,
    const float* values)
{
    unsigned char codes[16];
    float scale = mla_prefill_quantize_k4_one_pass_ls(values, codes);
    __nv_bfloat16 scale_bf16 = float_to_bf16(scale);
    *reinterpret_cast<unsigned short*>(dst) =
        *reinterpret_cast<unsigned short*>(&scale_bf16);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[2 + i] =
            (unsigned char)((codes[i * 2 + 1] << 4) | (codes[i * 2] & 0x0f));
    }
}

/*
 * Capture the normalized latent KV and the already-RoPE-transformed positional
 * K into the compact MLA signed-INT4 cache. The dense K input is the same BF16
 * tensor consumed by FlashAttention, so prefill and decode cache state share
 * an exact source.
 */
extern "C" __global__ void mla_cache_append_k4_kernel(
    unsigned char* __restrict__ ckv_cache,
    unsigned char* __restrict__ kpe_cache,
    const __nv_bfloat16* __restrict__ kv_a,
    const __nv_bfloat16* __restrict__ dense_k,
    int start_pos,
    int token_count,
    int kv_row_stride,
    int kv_lora_rank,
    int ckv_cache_dim,
    int num_heads,
    int qk_dim,
    int nope_dim,
    int rope_dim)
{
    int token = blockIdx.x;
    int cache_block = threadIdx.x;
    if (token >= token_count) return;

    int ckv_blocks = ckv_cache_dim / 16;
    int kpe_blocks = rope_dim / 16;
    int position = start_pos + token;

    if (cache_block < ckv_blocks) {
        float values[16];
        int base = cache_block * 16;
        const __nv_bfloat16* src = kv_a + (int64_t)token * kv_row_stride;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int d = base + i;
            values[i] = d < kv_lora_rank ? bf16_to_float(src[d]) : 0.0f;
        }
        unsigned char* dst =
            ckv_cache + ((int64_t)position * ckv_blocks + cache_block) * 10;
        mla_prefill_store_k4_block(dst, values);
    }

    if (cache_block < kpe_blocks) {
        float values[16];
        int base = cache_block * 16;
        const __nv_bfloat16* src =
            dense_k + ((int64_t)token * num_heads) * qk_dim + nope_dim;
        #pragma unroll
        for (int i = 0; i < 16; i++) values[i] = bf16_to_float(src[base + i]);
        unsigned char* dst =
            kpe_cache + ((int64_t)position * kpe_blocks + cache_block) * 10;
        mla_prefill_store_k4_block(dst, values);
    }
}

__device__ inline int mla_prefill_unpack_k4(
    const unsigned char* src,
    int index)
{
    unsigned char packed = src[index >> 1];
    return (index & 1) ? (int)(packed >> 4) : (int)(packed & 0x0f);
}

__device__ inline float mla_prefill_load_k4_value(
    const unsigned char* cache,
    int position,
    int logical_dim,
    int element)
{
    int blocks_per_row = logical_dim / 16;
    int block = element >> 4;
    const unsigned char* packed =
        cache + ((int64_t)position * blocks_per_row + block) * 10;
    float scale = bf16_to_float(
        *reinterpret_cast<const __nv_bfloat16*>(packed));
    return scale *
        (float)(mla_prefill_unpack_k4(packed + 2, element & 15) - 8);
}

__device__ inline float mla_prefill_dot_k4(
    const unsigned char* cache,
    int position,
    int logical_dim,
    const float* query)
{
    int blocks_per_row = logical_dim / 16;
    const unsigned char* row =
        cache + (int64_t)position * blocks_per_row * 10;
    float score = 0.0f;
    for (int block = 0; block < blocks_per_row; block++) {
        const unsigned char* packed = row + block * 10;
        float scale = bf16_to_float(
            *reinterpret_cast<const __nv_bfloat16*>(packed));
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            score += query[block * 16 + j] * scale *
                (float)(mla_prefill_unpack_k4(packed + 2, j) - 8);
        }
    }
    return score;
}

/*
 * Absorb the non-positional portion of a packed BF16 MLA prefill query into
 * the compressed latent width. Grid=(heads, rows); one block owns one
 * query-head row.
 */
extern "C" __global__ void mla_prefill_absorb_wkc_kernel(
    float* __restrict__ q_absorbed,
    const __nv_bfloat16* __restrict__ packed_q,
    const __nv_bfloat16* __restrict__ w_kc,
    int rows,
    int num_heads,
    int qk_dim,
    int nope_dim,
    int ckv_cache_dim)
{
    int head = blockIdx.x;
    int row = blockIdx.y;
    if (row >= rows || head >= num_heads) return;

    const __nv_bfloat16* query =
        packed_q + ((int64_t)row * num_heads + head) * qk_dim;
    const __nv_bfloat16* weight =
        w_kc + (int64_t)head * nope_dim * ckv_cache_dim;
    float* output =
        q_absorbed + ((int64_t)row * num_heads + head) * ckv_cache_dim;
    for (int dim = threadIdx.x; dim < ckv_cache_dim; dim += blockDim.x) {
        float value = 0.0f;
        for (int k = 0; k < nope_dim; k++) {
            value += bf16_to_float(query[k]) *
                bf16_to_float(weight[(int64_t)k * ckv_cache_dim + dim]);
        }
        output[dim] = value;
    }
}

/*
 * Sparse causal MLA prefill over one chunk-local selected-index matrix.
 * Grid=(heads, rows). Every row may have a different causal sequence length;
 * selected rows are fixed-stride and deterministic sentinel tails are ignored.
 * Dynamic shared memory:
 *   ckv_cache_dim + rope_dim + num_warps + selected_per_row floats.
 */
extern "C" __global__ void mla_prefill_sparse_attention_k4_kernel(
    float* __restrict__ output,
    const float* __restrict__ q_absorbed,
    const __nv_bfloat16* __restrict__ packed_q,
    const unsigned char* __restrict__ ckv_cache,
    const unsigned char* __restrict__ kpe_cache,
    const int* __restrict__ selected_indices,
    const int* __restrict__ positions,
    float sm_scale,
    int rows,
    int num_heads,
    int qk_dim,
    int nope_dim,
    int rope_dim,
    int ckv_cache_dim,
    int selected_per_row,
    int max_context)
{
    int head = blockIdx.x;
    int row = blockIdx.y;
    if (row >= rows || head >= num_heads) return;

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    int seq_len = min(max(positions[row] + 1, 0), max_context);
    int selected_count = min(seq_len, max(selected_per_row, 0));

    extern __shared__ float shared[];
    float* shared_q_absorbed = shared;
    float* shared_q_pe = shared_q_absorbed + ckv_cache_dim;
    float* shared_reduce = shared_q_pe + rope_dim;
    float* shared_weights = shared_reduce + num_warps;

    const float* q_absorbed_head =
        q_absorbed + ((int64_t)row * num_heads + head) * ckv_cache_dim;
    const __nv_bfloat16* packed_q_pe =
        packed_q + ((int64_t)row * num_heads + head) * qk_dim + nope_dim;
    const int* selected_row =
        selected_indices + (int64_t)row * selected_per_row;
    float* output_head =
        output + ((int64_t)row * num_heads + head) * ckv_cache_dim;

    for (int dim = tid; dim < ckv_cache_dim; dim += blockDim.x) {
        shared_q_absorbed[dim] = q_absorbed_head[dim];
    }
    for (int dim = tid; dim < rope_dim; dim += blockDim.x) {
        shared_q_pe[dim] = bf16_to_float(packed_q_pe[dim]);
    }
    __syncthreads();

    if (selected_count <= 0) {
        for (int dim = tid; dim < ckv_cache_dim; dim += blockDim.x) {
            output_head[dim] = 0.0f;
        }
        return;
    }

    float local_max = -1e30f;
    for (int slot = tid; slot < selected_count; slot += blockDim.x) {
        int position = selected_row[slot];
        float score = -1e30f;
        if (position >= 0 && position < seq_len) {
            score =
                mla_prefill_dot_k4(
                    ckv_cache, position, ckv_cache_dim, shared_q_absorbed) +
                mla_prefill_dot_k4(
                    kpe_cache, position, rope_dim, shared_q_pe);
            score *= sm_scale;
        }
        local_max = fmaxf(local_max, score);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(
            local_max,
            __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) shared_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float maximum = shared_reduce[0];
        for (int warp = 1; warp < num_warps; warp++) {
            maximum = fmaxf(maximum, shared_reduce[warp]);
        }
        shared_reduce[0] = maximum;
    }
    __syncthreads();
    float maximum = shared_reduce[0];

    float local_sum = 0.0f;
    for (int slot = tid; slot < selected_count; slot += blockDim.x) {
        int position = selected_row[slot];
        float weight = 0.0f;
        if (position >= 0 && position < seq_len) {
            float score =
                mla_prefill_dot_k4(
                    ckv_cache, position, ckv_cache_dim, shared_q_absorbed) +
                mla_prefill_dot_k4(
                    kpe_cache, position, rope_dim, shared_q_pe);
            weight = __expf(score * sm_scale - maximum);
        }
        shared_weights[slot] = weight;
        local_sum += weight;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) shared_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < num_warps; warp++) {
            sum += shared_reduce[warp];
        }
        shared_reduce[0] = sum;
    }
    __syncthreads();
    float inv_sum = shared_reduce[0] > 0.0f ? 1.0f / shared_reduce[0] : 0.0f;

    for (int dim = tid; dim < ckv_cache_dim; dim += blockDim.x) {
        float value = 0.0f;
        for (int slot = 0; slot < selected_count; slot++) {
            int position = selected_row[slot];
            if (position >= 0 && position < seq_len) {
                value += shared_weights[slot] *
                    mla_prefill_load_k4_value(
                        ckv_cache, position, ckv_cache_dim, dim);
            }
        }
        output_head[dim] = value * inv_sum;
    }
}

/*
 * Expand compressed sparse-attention output through w_vc and round to BF16
 * for the existing quantized output projection. Grid=(heads, rows).
 */
extern "C" __global__ void mla_prefill_apply_wvc_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ attention,
    const __nv_bfloat16* __restrict__ w_vc,
    int rows,
    int num_heads,
    int value_head_dim,
    int ckv_cache_dim)
{
    int head = blockIdx.x;
    int row = blockIdx.y;
    if (row >= rows || head >= num_heads) return;

    const float* attention_head =
        attention + ((int64_t)row * num_heads + head) * ckv_cache_dim;
    const __nv_bfloat16* weight =
        w_vc + (int64_t)head * value_head_dim * ckv_cache_dim;
    __nv_bfloat16* output_head =
        output + ((int64_t)row * num_heads + head) * value_head_dim;
    for (int dim = threadIdx.x; dim < value_head_dim; dim += blockDim.x) {
        float value = 0.0f;
        for (int k = 0; k < ckv_cache_dim; k++) {
            value += attention_head[k] *
                bf16_to_float(weight[(int64_t)dim * ckv_cache_dim + k]);
        }
        output_head[dim] = float_to_bf16(value);
    }
}

extern "C" void krasis_rope_batched(
    void* q, void* k, const void* positions,
    const void* cos_cache, const void* sin_cache,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    void* stream)
{
    if (M == 0) return;
    int half_dim = head_dim / 2;
    int threads = min(512, half_dim);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    rope_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)q, (__nv_bfloat16*)k,
        (const int*)positions,
        (const float*)cos_cache,
        (const float*)sin_cache,
        num_q_heads, num_kv_heads, head_dim, half_dim);
}

/* ── SiLU + Mul ────────────────────────────────────────────────────────── */

/* gate_up: [M, 2*N], out: [M, N]
 * out[i,j] = silu(gate_up[i,j]) * gate_up[i, N+j]
 */
extern "C" __global__ void silu_mul_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate_up,
    int N)
{
    int token = blockIdx.x;
    int two_N = 2 * N;
    const __nv_bfloat16* gu = gate_up + (int64_t)token * two_N;
    __nv_bfloat16* o = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float gate = bf16_to_float(gu[i]);
        float up = bf16_to_float(gu[N + i]);
        float silu_gate = gate / (1.0f + __expf(-gate));
        o[i] = float_to_bf16(silu_gate * up);
    }
}

extern "C" __global__ void silu_mul_limited_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate_up,
    int N,
    float limit)
{
    int token = blockIdx.x;
    int two_N = 2 * N;
    const __nv_bfloat16* gu = gate_up + (int64_t)token * two_N;
    __nv_bfloat16* o = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float gate = bf16_to_float(gu[i]);
        float up = bf16_to_float(gu[N + i]);
        float silu_gate = gate / (1.0f + __expf(-gate));
        apply_swiglu_limit(silu_gate, up, limit);
        o[i] = float_to_bf16(silu_gate * up);
    }
}

extern "C" void krasis_silu_mul_batched(
    void* out, const void* gate_up,
    int M, int N, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, N);
    threads = ((threads + 31) / 32) * 32;
    silu_mul_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)gate_up, N);
}

/* ── ReLU² ─────────────────────────────────────────────────────────────── */

/* out[i,j] = max(0, x[i,j])² */
extern "C" __global__ void relu2_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    int N)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * N;
    __nv_bfloat16* o_row = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        v = fmaxf(v, 0.0f);
        o_row[i] = float_to_bf16(v * v);
    }
}

extern "C" void krasis_relu2_batched(
    void* out, const void* x,
    int M, int N, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, N);
    threads = ((threads + 31) / 32) * 32;
    relu2_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x, N);
}

/* ── GELU(tanh approximation) + Mul ───────────────────────────────────── */

extern "C" __global__ void gelu_tanh_mul_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate_up,
    int N)
{
    int token = blockIdx.x;
    int two_N = 2 * N;
    const __nv_bfloat16* gu = gate_up + (int64_t)token * two_N;
    __nv_bfloat16* o = out + (int64_t)token * N;

    const float c = 0.7978845608028654f;
    const float k = 0.044715f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float gate = bf16_to_float(gu[i]);
        float up = bf16_to_float(gu[N + i]);
        float gelu = 0.5f * gate * (1.0f + tanhf(c * (gate + k * gate * gate * gate)));
        o[i] = float_to_bf16(gelu * up);
    }
}

extern "C" void krasis_gelu_tanh_mul_batched(
    void* out, const void* gate_up,
    int M, int N, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, N);
    threads = ((threads + 31) / 32) * 32;
    gelu_tanh_mul_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)gate_up, N);
}

extern "C" __global__ void gated_activation_split_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int activation,
    int N)
{
    int token = blockIdx.x;
    const __nv_bfloat16* g_row = gate + (int64_t)token * N;
    const __nv_bfloat16* u_row = up + (int64_t)token * N;
    __nv_bfloat16* o_row = out + (int64_t)token * N;

    const float c = 0.7978845608028654f;
    const float k = 0.044715f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float g = bf16_to_float(g_row[i]);
        float u = bf16_to_float(u_row[i]);
        float a;
        if (activation == 2) {
            a = 0.5f * g * (1.0f + tanhf(c * (g + k * g * g * g)));
        } else if (activation == 1) {
            float r = fmaxf(g, 0.0f);
            a = r * r;
        } else {
            a = g / (1.0f + __expf(-g));
        }
        o_row[i] = float_to_bf16(a * u);
    }
}

extern "C" void krasis_gated_activation_split_batched(
    void* out, const void* gate, const void* up,
    int activation, int M, int N, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, N);
    threads = ((threads + 31) / 32) * 32;
    gated_activation_split_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,
        activation,
        N);
}

/* ── Sigmoid-gated multiply (for gated GQA attention) ───────────────── */

/* out[i] = attn[i] * sigmoid(gate[i])
 * Used by QCN: q_proj outputs [query, gate], gate applied to attention output.
 * attn: [M, N] bf16 — attention output
 * gate: [M, N] bf16 — gate values (raw logits, sigmoid applied here)
 * out:  [M, N] bf16 — gated output (can be same buffer as attn)
 */
extern "C" __global__ void sigmoid_mul_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    int N)
{
    int token = blockIdx.x;
    const __nv_bfloat16* a_row = attn + (int64_t)token * N;
    const __nv_bfloat16* g_row = gate + (int64_t)token * N;
    __nv_bfloat16* o_row = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float a = bf16_to_float(a_row[i]);
        float g = bf16_to_float(g_row[i]);
        float sig = 1.0f / (1.0f + __expf(-g));
        o_row[i] = float_to_bf16(a * sig);
    }
}

/* Step head-wise GQA attention gate.
 * gate: [M, H] raw BF16 gate logits from self_attn.g_proj
 * attn/out: [M, H * D] BF16 attention output
 */
extern "C" __global__ void sigmoid_head_mul_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    int H,
    int D)
{
    int token = blockIdx.x;
    int total = H * D;
    const __nv_bfloat16* a_row = attn + (int64_t)token * total;
    const __nv_bfloat16* g_row = gate + (int64_t)token * H;
    __nv_bfloat16* o_row = out + (int64_t)token * total;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int head = idx / D;
        float a = bf16_to_float(a_row[idx]);
        float g = bf16_to_float(g_row[head]);
        float sig = 1.0f / (1.0f + __expf(-g));
        o_row[idx] = float_to_bf16(a * sig);
    }
}

/* ── BF16 ↔ FP32 conversion ──────────────────────────────────────────── */

extern "C" __global__ void bf16_to_fp32_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = bf16_to_float(in[idx]);
    }
}

extern "C" void krasis_bf16_to_fp32(
    void* out_fp32, const void* in_bf16, int count, void* stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)out_fp32, (const __nv_bfloat16*)in_bf16, count);
}

extern "C" __global__ void fp32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = float_to_bf16(in[idx]);
    }
}

extern "C" void krasis_fp32_to_bf16(
    void* out_bf16, const void* in_fp32, int count, void* stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out_bf16, (const float*)in_fp32, count);
}

/* ── Sigmoid Top-K routing ─────────────────────────────────────────────── */

/* One block per token. Computes sigmoid(gate_logits), then selects top-k.
 * Gate input is FP32 (from cuBLAS GEMM output).
 *
 * gate_bias is a pre-sigmoid logit bias. e_score_correction is a
 * post-sigmoid selection-only correction: selected weights are gathered from
 * the raw sigmoid scores, matching Step/Qwen-style router semantics.
 */
extern "C" __global__ void sigmoid_topk_kernel(
    float* __restrict__ topk_weights,       /* [M, topk] */
    int* __restrict__ topk_ids,             /* [M, topk] */
    const float* __restrict__ gate,         /* [M, E] FP32 */
    const float* __restrict__ gate_bias,    /* [E] or NULL */
    const float* __restrict__ e_score_corr, /* [E] or NULL */
    int E,
    int topk)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* tw = topk_weights + (int64_t)token * topk;
    int* ti = topk_ids + (int64_t)token * topk;

    /* Initialize top-k with -inf */
    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;       /* [E] selection scores */
    float* raw_scores = scores + E;         /* [E] routed weights */
    float* top_vals = raw_scores + E;       /* [topk] */
    int* top_idxs = (int*)(top_vals + topk); /* [topk] */

    /* Compute raw sigmoid weights and selection scores. */
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float x = g[i];
        if (gate_bias) x += gate_bias[i];
        float raw = 1.0f / (1.0f + __expf(-x));
        raw_scores[i] = raw;
        scores[i] = raw + (e_score_corr ? e_score_corr[i] : 0.0f);
    }
    __syncthreads();

    /* Single-threaded top-k selection (E is typically small, e.g. 128) */
    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            /* Find insertion point in sorted top-k */
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        for (int k = 0; k < topk; k++) {
            tw[k] = raw_scores[top_idxs[k]];
            ti[k] = top_idxs[k];
        }
    }
}

extern "C" void krasis_sigmoid_topk(
    void* topk_weights, void* topk_ids, const void* gate_logits,
    const void* gate_bias, const void* e_score_correction,
    int M, int num_experts, int topk, void* stream)
{
    if (M == 0) return;
    int threads = min(256, num_experts);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    int smem = 2 * num_experts * sizeof(float) + topk * (sizeof(float) + sizeof(int));
    sigmoid_topk_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (float*)topk_weights, (int*)topk_ids,
        (const float*)gate_logits,
        (const float*)gate_bias,
        (const float*)e_score_correction,
        num_experts, topk);
}

extern "C" __global__ void normalize_topk_weights_kernel(
    float* __restrict__ topk_weights,  /* [M, topk] */
    int M,
    int topk)
{
    int token = blockIdx.x;
    if (token >= M || threadIdx.x != 0) return;
    float* row = topk_weights + (int64_t)token * topk;
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
        sum += row[k];
    }
    float inv_sum = 1.0f / (sum + 1.0e-20f);
    for (int k = 0; k < topk; k++) {
        row[k] *= inv_sum;
    }
}

/* ── Softmax Top-K routing ─────────────────────────────────────────────── */

/* Gate input is FP32 (from cuBLAS GEMM output).
 * Uses warp-shuffle reduction instead of atomicMax to handle negative floats correctly. */
extern "C" __global__ void softmax_topk_kernel(
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids,
    const float* __restrict__ gate,
    int E,
    int topk)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* tw = topk_weights + (int64_t)token * topk;
    int* ti = topk_ids + (int64_t)token * topk;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    /* Compute exp(logit - max), then normalize only selected top-k values.
     * The full softmax denominator cancels during top-k renormalization, and
     * avoiding it removes atomicAdd reduction-order sensitivity. */
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = g[i];
        max_val = fmaxf(max_val, scores[i]);
    }
    /* Reduce max across threads using warp shuffle + shared memory
     * (atomicMax on float-as-int fails for negative values) */
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_warp_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) m = fmaxf(m, s_warp_max[w]);
        s_warp_max[0] = m;
    }
    __syncthreads();
    max_val = s_warp_max[0];

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = __expf(scores[i] - max_val);
    }
    __syncthreads();

    /* Top-k selection (single thread, E small) */
    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        /* Normalize top-k weights to sum to 1.0 */
        float wsum = 0.0f;
        for (int k = 0; k < topk; k++) wsum += top_vals[k];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (int k = 0; k < topk; k++) {
            tw[k] = top_vals[k] * inv_wsum;
            ti[k] = top_idxs[k];
        }
    }
}

extern "C" void krasis_softmax_topk(
    void* topk_weights, void* topk_ids, const void* gate_logits,
    int M, int num_experts, int topk, void* stream)
{
    if (M == 0) return;
    int threads = min(256, num_experts);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    /* Extra smem for warp-max reduction: 32 floats */
    int smem = num_experts * sizeof(float) + topk * (sizeof(float) + sizeof(int)) + 32 * sizeof(float);
    softmax_topk_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (float*)topk_weights, (int*)topk_ids,
        (const float*)gate_logits,
        num_experts, topk);
}

/* Diagnostic-only softmax top-k sum probe. This reproduces the denominator and
 * selected-weight path for observation, but writes only to a separate debug
 * buffer and is never used by normal routing. Output row layout:
 * [max, sum, inv_sum, topk_prob_sum, inv_topk_sum,
 *  topk ids as f32, topk softmax probabilities, topk renormalized weights].
 */
extern "C" __global__ void softmax_topk_sum_probe_kernel(
    float* __restrict__ probe_out,
    const float* __restrict__ gate,
    int E,
    int topk,
    int fields)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* out = probe_out + (int64_t)token * fields;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = g[i];
        max_val = fmaxf(max_val, scores[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_warp_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) m = fmaxf(m, s_warp_max[w]);
        s_warp_max[0] = m;
    }
    __syncthreads();
    max_val = s_warp_max[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = __expf(scores[i] - max_val);
        sum += scores[i];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        float wsum = 0.0f;
        for (int k = 0; k < topk; k++) wsum += top_vals[k];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;

        if (fields >= 5 + 3 * topk) {
            out[0] = max_val;
            out[1] = s_sum;
            out[2] = inv_sum;
            out[3] = wsum;
            out[4] = inv_wsum;
            for (int k = 0; k < topk; k++) {
                out[5 + k] = (float)top_idxs[k];
                out[5 + topk + k] = top_vals[k];
                out[5 + 2 * topk + k] = top_vals[k] * inv_wsum;
            }
        }
    }
}

/* Diagnostic-only selected-logit normalization probe. Unlike
 * softmax_topk_sum_probe_kernel, this never uses the full softmax denominator:
 * it selects top-k by exp(logit - max) and normalizes only the selected exp
 * values. It writes only to a separate debug buffer.
 *
 * Output row layout:
 * [max, selected_exp_sum, inv_selected_exp_sum,
 *  topk ids as f32, selected exp values, selected-only weights].
 */
extern "C" __global__ void softmax_topk_selected_probe_kernel(
    float* __restrict__ probe_out,
    const float* __restrict__ gate,
    int E,
    int topk,
    int fields)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* out = probe_out + (int64_t)token * fields;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = g[i];
        max_val = fmaxf(max_val, scores[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_warp_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) m = fmaxf(m, s_warp_max[w]);
        s_warp_max[0] = m;
    }
    __syncthreads();
    max_val = s_warp_max[0];

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = __expf(scores[i] - max_val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        float wsum = 0.0f;
        for (int k = 0; k < topk; k++) wsum += top_vals[k];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;

        if (fields >= 3 + 3 * topk) {
            out[0] = max_val;
            out[1] = wsum;
            out[2] = inv_wsum;
            for (int k = 0; k < topk; k++) {
                out[3 + k] = (float)top_idxs[k];
                out[3 + topk + k] = top_vals[k];
                out[3 + 2 * topk + k] = top_vals[k] * inv_wsum;
            }
        }
    }
}

/* ── MoE Sum Reduce ────────────────────────────────────────────────────── */

/* Reduce expert outputs weighted by topk_weights.
 * expert_outputs: [M*topk, K] viewed as [M, topk, K]
 * output: [M, K]
 */
extern "C" __global__ void moe_sum_reduce_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_outputs,
    const float* __restrict__ topk_weights,
    int K,
    int topk,
    float scale)
{
    int token = blockIdx.x;
    __nv_bfloat16* o = output + (int64_t)token * K;
    const float* tw = topk_weights + (int64_t)token * topk;

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) {
            const __nv_bfloat16* e = expert_outputs + ((int64_t)token * topk + k) * K;
            acc += bf16_to_float(e[i]) * tw[k];
        }
        o[i] = float_to_bf16(acc * scale);
    }
}

extern "C" void krasis_moe_sum_reduce(
    void* output, const void* expert_outputs, const void* topk_weights,
    int M, int K, int topk, float scale, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, K);
    threads = ((threads + 31) / 32) * 32;
    moe_sum_reduce_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)expert_outputs,
        (const float*)topk_weights, K, topk, scale);
}

/* ── MoE Align Block Size ──────────────────────────────────────────────── */

/* Sort tokens by expert assignment with block-size padding.
 * This is a CPU-side operation typically, but we implement on GPU for
 * the Rust prefill path (avoids D2H copy of topk_ids).
 *
 * Each block handles one expert. Outputs:
 *   sorted_token_ids: padded list of token indices sorted by expert
 *   expert_ids: which expert each block of tokens belongs to
 *   num_tokens_post_padded: total padded token count
 */
extern "C" __global__ void moe_align_block_size_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    int* __restrict__ expert_counts,     /* [E] scratch */
    int* __restrict__ expert_offsets,    /* [E+1] scratch */
    const int* __restrict__ topk_ids,   /* [M, topk] */
    int M,
    int topk,
    int block_size)
{
    /* Phase 1: count tokens per expert (single block, single pass) */
    int E = gridDim.x;  /* one block per expert initially, but use 1 block for counting */
    /* Actually this needs a 2-phase approach. Use a simple sequential kernel. */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int total = M * topk;
        /* Clear counts */
        for (int e = 0; e < E; e++) expert_counts[e] = 0;
        /* Count */
        for (int i = 0; i < total; i++) {
            int eid = topk_ids[i];
            if (eid >= 0 && eid < E) expert_counts[eid]++;
        }
        /* Compute offsets with padding */
        int offset = 0;
        for (int e = 0; e < E; e++) {
            expert_offsets[e] = offset;
            int padded = ((expert_counts[e] + block_size - 1) / block_size) * block_size;
            offset += padded;
            /* Fill expert_ids for each block */
            int num_blocks_for_expert = padded / block_size;
            for (int b = 0; b < num_blocks_for_expert; b++) {
                expert_ids[expert_offsets[e] / block_size + b] = e;
            }
        }
        expert_offsets[E] = offset;
        *num_tokens_post_padded = offset;

        /* Scatter token indices */
        /* First, fill sorted_token_ids with padding value (M*topk = invalid) */
        for (int i = 0; i < offset; i++) sorted_token_ids[i] = total;
        /* Reset counts as write cursors */
        for (int e = 0; e < E; e++) expert_counts[e] = 0;
        /* Scatter */
        for (int i = 0; i < total; i++) {
            int eid = topk_ids[i];
            if (eid >= 0 && eid < E) {
                int pos = expert_offsets[eid] + expert_counts[eid];
                sorted_token_ids[pos] = i;
                expert_counts[eid]++;
            }
        }
    }
}

extern "C" void krasis_moe_align_block_size(
    void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded,
    const void* topk_ids,
    int M, int num_experts, int topk, int block_size,
    void* stream)
{
    /* We need scratch space for expert_counts [E] and expert_offsets [E+1].
     * For simplicity, run on 1 block 1 thread (it's fast for small E).
     * TODO: optimize for large E with parallel counting. */
    /* The scratch pointers are after the main outputs.
     * Caller must allocate extra: (E + E + 1) * sizeof(int) after sorted_token_ids. */
    /* Actually, we'll use a different approach: caller provides scratch. */
    /* For now, simple single-thread kernel. E <= 512 typically. */

    /* Use global memory for scratch: allocate after sorted_token_ids.
     * Actually, let's just use a simple approach: the caller ensures the buffer
     * is large enough. We'll put scratch at sorted_token_ids + M*topk + padding. */

    /* Simple approach: use CUDA managed memory for scratch. Actually, just
     * allocate on the device. Let Rust handle this. For now, minimal kernel. */
    /* TODO: This needs scratch memory. For initial impl, use shared memory. */

    if (M == 0) return;
    /* E * 2 + 1 ints for scratch, E <= 1024, fits in shared memory */
    int smem = (num_experts * 2 + 1) * sizeof(int);
    /* Override: put scratch in shared memory instead of separate buffers */
    /* Actually the kernel above uses separate pointers. Let me simplify. */

    /* For initial implementation, run a single-thread sequential kernel.
     * This is fine because MoE routing is not the bottleneck (GEMM is). */
    /* Provide scratch via dynamic shared memory */

    /* Rewrite: use shared memory scratch */
    /* TODO: rewrite with shared memory. For now, this is a placeholder
     * that will be replaced by linking against sgl_kernel's moe_align_block_size. */
    (void)sorted_token_ids;
    (void)expert_ids;
    (void)num_tokens_post_padded;
    (void)topk_ids;
    (void)M;
    (void)num_experts;
    (void)topk;
    (void)block_size;
    (void)stream;
}

/* ── Memory helpers ────────────────────────────────────────────────────── */

extern "C" void krasis_zero_buffer(void* ptr, int64_t bytes, void* stream) {
    cudaMemsetAsync(ptr, 0, bytes, (cudaStream_t)stream);
}

extern "C" void krasis_memcpy_d2d(void* dst, const void* src, int64_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}

extern "C" void krasis_memcpy_h2d(void* dst, const void* src, int64_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

extern "C" void krasis_memcpy_d2h(void* dst, const void* src, int64_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

extern "C" void krasis_stream_sync(void* stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

/* ── Non-perturbing debug summaries ───────────────────────────────────── */

__device__ __forceinline__ unsigned long long trace_f32_bits(float v) {
    union {
        float f;
        unsigned int u;
    } cvt;
    cvt.f = v;
    return (unsigned long long)cvt.u;
}

extern "C" __global__ void prefill_trace_bf16_row_summary_kernel(
    const __nv_bfloat16* __restrict__ row,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width)
{
    if (width <= 0) return;

    __shared__ unsigned long long s_hash[256];
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_finite[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];

    const unsigned short* bits = (const unsigned short*)row;
    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(threadIdx.x + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        unsigned int raw = (unsigned int)bits[i];
        h ^= (unsigned long long)raw;
        h *= 1099511628211ULL;

        float v = bf16_to_float(row[i]);
        if (isfinite(v)) {
            finite_count++;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        } else if (isnan(v)) {
            nan_count++;
        } else {
            inf_count++;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    int tid = threadIdx.x;
    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    s_nan[tid] = nan_count;
    s_inf[tid] = inf_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
            s_nan[tid] += s_nan[tid + stride];
            s_inf[tid] += s_inf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)width;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = (unsigned long long)s_nan[0];
        trace[base + 14] = (unsigned long long)s_inf[0];
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_f32_slice_summary_kernel(
    const float* __restrict__ values,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width)
{
    if (width <= 0) return;

    __shared__ unsigned long long s_hash[256];
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_finite[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];

    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(threadIdx.x + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = values[i];
        unsigned long long raw = trace_f32_bits(v);
        h ^= raw;
        h *= 1099511628211ULL;

        if (isfinite(v)) {
            finite_count++;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        } else if (isnan(v)) {
            nan_count++;
        } else {
            inf_count++;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    int tid = threadIdx.x;
    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    s_nan[tid] = nan_count;
    s_inf[tid] = inf_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
            s_nan[tid] += s_nan[tid + stride];
            s_inf[tid] += s_inf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)width;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = (unsigned long long)s_nan[0];
        trace[base + 14] = (unsigned long long)s_inf[0];
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_i32_slice_summary_kernel(
    const int* __restrict__ values,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width)
{
    if (width <= 0) return;

    __shared__ unsigned long long s_hash[256];
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_finite[256];

    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(threadIdx.x + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        int raw_i = values[i];
        float v = (float)raw_i;
        unsigned int raw = (unsigned int)raw_i;
        h ^= (unsigned long long)raw;
        h *= 1099511628211ULL;
        finite_count++;
        sum += (double)v;
        sumsq += (double)v * (double)v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    int tid = threadIdx.x;
    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)width;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = 0ULL;
        trace[base + 14] = 0ULL;
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_router_row_candidate_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ hidden_row,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ logits_row,
    const float* __restrict__ topk_weights_row,
    int hidden_size,
    int num_experts,
    int topk,
    float routed_scale,
    int mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode == 0) {
        if (idx >= num_experts) return;
        const __nv_bfloat16* w_row = weight + (int64_t)idx * hidden_size;
        float acc = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            acc += bf16_to_float(hidden_row[i]) * bf16_to_float(w_row[i]);
        }
        out[idx] = acc;
    } else if (mode == 1) {
        if (idx >= num_experts) return;
        float v = logits_row[idx];
        out[idx] = 1.0f / (1.0f + __expf(-v));
    } else if (mode == 2) {
        if (idx >= topk) return;
        out[idx] = topk_weights_row[idx] * routed_scale;
    }
}

extern "C" __global__ void prefill_trace_router_logit_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ hidden_row,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ logits_row,
    int hidden_size,
    int num_experts)
{
    int expert = blockIdx.x;
    if (threadIdx.x != 0 || expert >= num_experts || hidden_size <= 0) return;

    const unsigned short* hidden_bits = reinterpret_cast<const unsigned short*>(hidden_row);
    const unsigned short* weight_bits = reinterpret_cast<const unsigned short*>(weight);
    const __nv_bfloat16* weight_row = weight + (int64_t)expert * hidden_size;
    const unsigned short* weight_row_bits = weight_bits + (int64_t)expert * hidden_size;

    unsigned long long input_hash = 14695981039346656037ULL;
    unsigned long long weight_hash = 14695981039346656037ULL;
    float acc = 0.0f;
    float max_abs_contrib = -1.0f;
    int max_abs_contrib_idx = 0;
    float max_abs_hidden = 0.0f;
    float max_abs_weight = 0.0f;

    for (int i = 0; i < hidden_size; i++) {
        unsigned int h_raw = (unsigned int)hidden_bits[i];
        unsigned int w_raw = (unsigned int)weight_row_bits[i];
        input_hash ^= (unsigned long long)(h_raw & 0xffU);
        input_hash *= 1099511628211ULL;
        input_hash ^= (unsigned long long)((h_raw >> 8) & 0xffU);
        input_hash *= 1099511628211ULL;
        weight_hash ^= (unsigned long long)(w_raw & 0xffU);
        weight_hash *= 1099511628211ULL;
        weight_hash ^= (unsigned long long)((w_raw >> 8) & 0xffU);
        weight_hash *= 1099511628211ULL;

        float h = bf16_to_float(hidden_row[i]);
        float w = bf16_to_float(weight_row[i]);
        float prod = h * w;
        acc += prod;
        float abs_prod = fabsf(prod);
        if (abs_prod > max_abs_contrib) {
            max_abs_contrib = abs_prod;
            max_abs_contrib_idx = i;
            max_abs_hidden = h;
            max_abs_weight = w;
        }
    }

    float production = logits_row[expert];
    int base = (entry_base_idx + expert) * 16;
    trace[base + 0] = (unsigned long long)stage_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)expert;
    trace[base + 6] = (unsigned long long)hidden_size;
    trace[base + 7] = input_hash;
    trace[base + 8] = weight_hash;
    trace[base + 9] = trace_f32_bits(acc);
    trace[base + 10] = trace_f32_bits(production);
    trace[base + 11] = trace_f32_bits(acc - production);
    trace[base + 12] = (unsigned long long)max_abs_contrib_idx;
    trace[base + 13] = trace_f32_bits(max_abs_hidden);
    trace[base + 14] = trace_f32_bits(max_abs_weight);
    trace[base + 15] = 1ULL;
}

extern "C" __global__ void prefill_trace_bf16_element_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ row,
    int width,
    int dim_index)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || width <= 0 || dim_index < 0 || dim_index >= width) return;

    const unsigned short* row_bits = reinterpret_cast<const unsigned short*>(row);
    unsigned long long row_hash = 14695981039346656037ULL;
    for (int i = 0; i < width; i++) {
        unsigned int raw = (unsigned int)row_bits[i];
        row_hash ^= (unsigned long long)(raw & 0xffU);
        row_hash *= 1099511628211ULL;
        row_hash ^= (unsigned long long)((raw >> 8) & 0xffU);
        row_hash *= 1099511628211ULL;
    }

    float value = bf16_to_float(row[dim_index]);
    int base = entry_idx * 16;
    trace[base + 0] = (unsigned long long)stage_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)dim_index;
    trace[base + 6] = (unsigned long long)width;
    trace[base + 7] = (unsigned long long)row_bits[dim_index];
    trace[base + 8] = row_hash;
    trace[base + 9] = trace_f32_bits(value);
    trace[base + 10] = 0ULL;
    trace[base + 11] = 0ULL;
    trace[base + 12] = 0ULL;
    trace[base + 13] = 0ULL;
    trace[base + 14] = 0ULL;
    trace[base + 15] = 1ULL;
}

extern "C" __global__ void prefill_trace_bf16_row_element_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ row,
    int width,
    int element_count)
{
    int dim_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_index >= element_count || dim_index >= width || width <= 0) return;

    const unsigned short* row_bits = reinterpret_cast<const unsigned short*>(row);
    float value = bf16_to_float(row[dim_index]);
    int base = (entry_base_idx + dim_index) * 16;
    trace[base + 0] = (unsigned long long)stage_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)dim_index;
    trace[base + 6] = (unsigned long long)width;
    trace[base + 7] = (unsigned long long)row_bits[dim_index];
    trace[base + 8] = 0ULL;
    trace[base + 9] = trace_f32_bits(value);
    trace[base + 10] = 0ULL;
    trace[base + 11] = 0ULL;
    trace[base + 12] = 0ULL;
    trace[base + 13] = 0ULL;
    trace[base + 14] = 0ULL;
    trace[base + 15] = 1ULL;
}

extern "C" __global__ void prefill_trace_bf16_projection_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ input_row,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ output_row,
    int input_width,
    int output_width,
    int output_dim)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || input_width <= 0 || output_width <= 0 ||
        output_dim < 0 || output_dim >= output_width) return;

    const unsigned short* input_bits = reinterpret_cast<const unsigned short*>(input_row);
    const __nv_bfloat16* weight_row = weight + (int64_t)output_dim * input_width;
    const unsigned short* weight_bits = reinterpret_cast<const unsigned short*>(weight_row);
    const unsigned short* output_bits = reinterpret_cast<const unsigned short*>(output_row);

    unsigned long long input_hash = 14695981039346656037ULL;
    unsigned long long weight_hash = 14695981039346656037ULL;
    float acc = 0.0f;
    float max_abs_contrib = -1.0f;
    int max_abs_contrib_idx = 0;
    float max_abs_input = 0.0f;
    float max_abs_weight = 0.0f;

    for (int i = 0; i < input_width; i++) {
        unsigned int h_raw = (unsigned int)input_bits[i];
        unsigned int w_raw = (unsigned int)weight_bits[i];
        input_hash ^= (unsigned long long)(h_raw & 0xffU);
        input_hash *= 1099511628211ULL;
        input_hash ^= (unsigned long long)((h_raw >> 8) & 0xffU);
        input_hash *= 1099511628211ULL;
        weight_hash ^= (unsigned long long)(w_raw & 0xffU);
        weight_hash *= 1099511628211ULL;
        weight_hash ^= (unsigned long long)((w_raw >> 8) & 0xffU);
        weight_hash *= 1099511628211ULL;

        float h = bf16_to_float(input_row[i]);
        float w = bf16_to_float(weight_row[i]);
        float prod = h * w;
        acc += prod;
        float abs_prod = fabsf(prod);
        if (abs_prod > max_abs_contrib) {
            max_abs_contrib = abs_prod;
            max_abs_contrib_idx = i;
            max_abs_input = h;
            max_abs_weight = w;
        }
    }

    float production = bf16_to_float(output_row[output_dim]);
    int base = entry_idx * 16;
    trace[base + 0] = (unsigned long long)stage_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)output_dim;
    trace[base + 6] = (unsigned long long)input_width;
    trace[base + 7] = input_hash;
    trace[base + 8] = weight_hash;
    trace[base + 9] = trace_f32_bits(acc);
    trace[base + 10] = trace_f32_bits(production);
    trace[base + 11] = trace_f32_bits(acc - production);
    trace[base + 12] = (unsigned long long)max_abs_contrib_idx;
    trace[base + 13] = trace_f32_bits(max_abs_input);
    trace[base + 14] = trace_f32_bits(max_abs_weight);
    trace[base + 15] = (unsigned long long)(output_bits[output_dim] | 0x10000U);
}

extern "C" __global__ void prefill_trace_mamba2_gated_norm_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int stage_input_id,
    int stage_output_id,
    int stage_rstd_candidate_id,
    int stage_ptx_candidate_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ x_row,
    const __nv_bfloat16* __restrict__ gate_row,
    const float* __restrict__ weight,
    const __nv_bfloat16* __restrict__ out_row,
    int d_inner,
    int n_groups,
    int group_size,
    int proj_dim,
    float eps,
    int dim_index)
{
    if (d_inner <= 0 || n_groups <= 0 || group_size <= 0 || proj_dim <= 0) return;
    if (dim_index < 0 || dim_index >= d_inner || dim_index >= proj_dim) return;

    int group = dim_index / group_size;
    if (group < 0 || group >= n_groups) return;
    int group_base = group * group_size;

    extern __shared__ float smem[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_base + i;
        float xv = bf16_to_float(x_row[idx]);
        float gv = bf16_to_float(gate_row[idx]);
        float silu_g = gv / (1.0f + __expf(-gv));
        float gated = xv * silu_g;
        local_ss += gated * gated;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    gated_rmsnorm_adjacent_pairwise_reduce(smem);

    if (threadIdx.x != 0) return;

    const unsigned short* x_bits = reinterpret_cast<const unsigned short*>(x_row);
    const unsigned short* gate_bits = reinterpret_cast<const unsigned short*>(gate_row);
    const unsigned short* out_bits = reinterpret_cast<const unsigned short*>(out_row);

    float xv = bf16_to_float(x_row[dim_index]);
    float gv = bf16_to_float(gate_row[dim_index]);
    float silu_g = gv / (1.0f + __expf(-gv));
    float gated = xv * silu_g;
    float mean_square = smem[0] / (float)group_size;
    float mean_square_plus_eps = mean_square + eps;
    float rms_inv = rsqrtf(mean_square_plus_eps);
    float sqrt_value = sqrtf(mean_square_plus_eps);
    float one_over_sqrtf = 1.0f / sqrt_value;
    float double_promoted_rstd = (float)(1.0 / sqrt((double)mean_square_plus_eps));
    float sqrt_rn = trace_sqrt_rn_f32(mean_square_plus_eps);
    float rstd_sqrt_rn_div_rn = trace_div_rn_f32(1.0f, sqrt_rn);
    float rstd_sqrt_rn_rcp_rn = trace_rcp_rn_f32(sqrt_rn);
    float sqrt_approx = trace_sqrt_approx_f32(mean_square_plus_eps);
    float rstd_sqrt_approx_div_rn = trace_div_rn_f32(1.0f, sqrt_approx);
    float rstd_sqrt_approx_div_approx = trace_div_approx_f32(1.0f, sqrt_approx);
    float rstd_rsqrt_approx = trace_rsqrt_approx_f32(mean_square_plus_eps);
    float normalized = gated * rms_inv;
    float weight_value = weight[dim_index];
    float pre_store = normalized * weight_value;
    float pre_store_one_over_sqrtf = (gated * one_over_sqrtf) * weight_value;
    float pre_store_double_promoted = (gated * double_promoted_rstd) * weight_value;
    float stored_value = bf16_to_float(out_row[dim_index]);

    int base = entry_base_idx * 16;
    trace[base + 0] = (unsigned long long)stage_input_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)dim_index;
    trace[base + 6] = (unsigned long long)d_inner;
    trace[base + 7] = (unsigned long long)x_bits[dim_index];
    trace[base + 8] = (unsigned long long)gate_bits[dim_index];
    trace[base + 9] = (unsigned long long)out_bits[dim_index];
    trace[base + 10] = trace_f32_bits(xv);
    trace[base + 11] = trace_f32_bits(gv);
    trace[base + 12] = trace_f32_bits(silu_g);
    trace[base + 13] = trace_f32_bits(gated);
    trace[base + 14] = (unsigned long long)group;
    trace[base + 15] = trace_f32_bits(eps);

    int base2 = (entry_base_idx + 1) * 16;
    trace[base2 + 0] = (unsigned long long)stage_output_id;
    trace[base2 + 1] = (unsigned long long)layer_idx;
    trace[base2 + 2] = (unsigned long long)chunk_idx;
    trace[base2 + 3] = (unsigned long long)absolute_position;
    trace[base2 + 4] = (unsigned long long)token_id;
    trace[base2 + 5] = (unsigned long long)dim_index;
    trace[base2 + 6] = (unsigned long long)group_size;
    trace[base2 + 7] = (unsigned long long)trace_f32_bits(weight_value);
    trace[base2 + 8] = (unsigned long long)group;
    trace[base2 + 9] = trace_f32_bits(mean_square);
    trace[base2 + 10] = trace_f32_bits(rms_inv);
    trace[base2 + 11] = trace_f32_bits(normalized);
    trace[base2 + 12] = trace_f32_bits(pre_store);
    trace[base2 + 13] = trace_f32_bits(stored_value);
    trace[base2 + 14] = (unsigned long long)out_bits[dim_index];
    trace[base2 + 15] = trace_f32_bits(mean_square_plus_eps);

    int base3 = (entry_base_idx + 2) * 16;
    trace[base3 + 0] = (unsigned long long)stage_rstd_candidate_id;
    trace[base3 + 1] = (unsigned long long)layer_idx;
    trace[base3 + 2] = (unsigned long long)chunk_idx;
    trace[base3 + 3] = (unsigned long long)absolute_position;
    trace[base3 + 4] = (unsigned long long)token_id;
    trace[base3 + 5] = (unsigned long long)dim_index;
    trace[base3 + 6] = (unsigned long long)group_size;
    trace[base3 + 7] = (unsigned long long)group;
    trace[base3 + 8] = trace_f32_bits(mean_square_plus_eps);
    trace[base3 + 9] = trace_f32_bits(rms_inv);
    trace[base3 + 10] = trace_f32_bits(sqrt_value);
    trace[base3 + 11] = trace_f32_bits(one_over_sqrtf);
    trace[base3 + 12] = trace_f32_bits(double_promoted_rstd);
    trace[base3 + 13] = trace_f32_bits(pre_store);
    trace[base3 + 14] = trace_f32_bits(pre_store_one_over_sqrtf);
    trace[base3 + 15] = trace_f32_bits(pre_store_double_promoted);

    int base4 = (entry_base_idx + 3) * 16;
    trace[base4 + 0] = (unsigned long long)stage_ptx_candidate_id;
    trace[base4 + 1] = (unsigned long long)layer_idx;
    trace[base4 + 2] = (unsigned long long)chunk_idx;
    trace[base4 + 3] = (unsigned long long)absolute_position;
    trace[base4 + 4] = (unsigned long long)token_id;
    trace[base4 + 5] = (unsigned long long)dim_index;
    trace[base4 + 6] = (unsigned long long)group_size;
    trace[base4 + 7] = (unsigned long long)group;
    trace[base4 + 8] = trace_f32_bits(mean_square_plus_eps);
    trace[base4 + 9] = trace_f32_bits(sqrt_rn);
    trace[base4 + 10] = trace_f32_bits(rstd_sqrt_rn_div_rn);
    trace[base4 + 11] = trace_f32_bits(rstd_sqrt_rn_rcp_rn);
    trace[base4 + 12] = trace_f32_bits(sqrt_approx);
    trace[base4 + 13] = trace_f32_bits(rstd_sqrt_approx_div_rn);
    trace[base4 + 14] = trace_f32_bits(rstd_sqrt_approx_div_approx);
    trace[base4 + 15] = trace_f32_bits(rstd_rsqrt_approx);
}

extern "C" __global__ void prefill_trace_mamba2_gated_norm_reduction_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int max_entries,
    int stage_term_id,
    int stage_reduction_id,
    int stage_summary_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ x_row,
    const __nv_bfloat16* __restrict__ gate_row,
    int d_inner,
    int n_groups,
    int group_size,
    int proj_dim,
    float eps,
    int dim_index)
{
    if (trace == NULL || max_entries <= 0 || d_inner <= 0 || n_groups <= 0 ||
        group_size <= 0 || proj_dim <= 0) return;
    if (dim_index < 0 || dim_index >= d_inner || dim_index >= proj_dim) return;

    int group = dim_index / group_size;
    if (group < 0 || group >= n_groups) return;
    int group_base = group * group_size;

    extern __shared__ float smem[];
    const unsigned short* x_bits = reinterpret_cast<const unsigned short*>(x_row);
    const unsigned short* gate_bits = reinterpret_cast<const unsigned short*>(gate_row);

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_base + i;
        float xv = bf16_to_float(x_row[idx]);
        float gv = bf16_to_float(gate_row[idx]);
        float silu_g = gv / (1.0f + __expf(-gv));
        float gated = xv * silu_g;
        float square = gated * gated;
        local_ss += square;

        int entry = entry_base_idx + i;
        if (entry >= 0 && entry < max_entries) {
            int base = entry * 16;
            unsigned long long packed_bf16 =
                ((unsigned long long)(gate_bits[idx] & 0xffffU) << 16) |
                (unsigned long long)(x_bits[idx] & 0xffffU);
            trace[base + 0] = (unsigned long long)stage_term_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)group;
            trace[base + 6] = (unsigned long long)group_size;
            trace[base + 7] = (unsigned long long)i;
            trace[base + 8] = (unsigned long long)idx;
            trace[base + 9] = (unsigned long long)threadIdx.x;
            trace[base + 10] = packed_bf16;
            trace[base + 11] = trace_f32_bits(silu_g);
            trace[base + 12] = trace_f32_bits(gated);
            trace[base + 13] = trace_f32_bits(square);
            trace[base + 14] = trace_f32_bits(local_ss);
            trace[base + 15] = 1ULL;
        }
    }

    smem[threadIdx.x] = local_ss;
    __syncthreads();

    int reduction_entry_base = entry_base_idx + group_size;
    int reduction_offset = 0;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int span = stride << 1;
        int active_count = blockDim.x / span;
        if (((threadIdx.x & (span - 1)) == 0) && threadIdx.x + stride < blockDim.x) {
            float left_before = smem[threadIdx.x];
            float right_before = smem[threadIdx.x + stride];
            float after = left_before + right_before;
            int entry = reduction_entry_base + reduction_offset + threadIdx.x / span;
            if (entry >= 0 && entry < max_entries) {
                int base = entry * 16;
                trace[base + 0] = (unsigned long long)stage_reduction_id;
                trace[base + 1] = (unsigned long long)layer_idx;
                trace[base + 2] = (unsigned long long)chunk_idx;
                trace[base + 3] = (unsigned long long)absolute_position;
                trace[base + 4] = (unsigned long long)token_id;
                trace[base + 5] = (unsigned long long)group;
                trace[base + 6] = (unsigned long long)group_size;
                trace[base + 7] = (unsigned long long)stride;
                trace[base + 8] = (unsigned long long)threadIdx.x;
                trace[base + 9] = (unsigned long long)(threadIdx.x + stride);
                trace[base + 10] = trace_f32_bits(left_before);
                trace[base + 11] = trace_f32_bits(right_before);
                trace[base + 12] = trace_f32_bits(after);
                trace[base + 13] = (unsigned long long)(reduction_offset + threadIdx.x / span);
                trace[base + 14] = (unsigned long long)blockDim.x;
                trace[base + 15] = 1ULL;
            }
            smem[threadIdx.x] = after;
        }
        __syncthreads();
        reduction_offset += active_count;
    }

    if (threadIdx.x == 0) {
        int entry = reduction_entry_base + reduction_offset;
        if (entry >= 0 && entry < max_entries) {
            float sum = smem[0];
            float mean_square = sum / (float)group_size;
            float mean_square_plus_eps = mean_square + eps;
            float rms_inv = rsqrtf(mean_square_plus_eps);
            int base = entry * 16;
            trace[base + 0] = (unsigned long long)stage_summary_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)group;
            trace[base + 6] = (unsigned long long)group_size;
            trace[base + 7] = (unsigned long long)blockDim.x;
            trace[base + 8] = trace_f32_bits(sum);
            trace[base + 9] = trace_f32_bits(mean_square);
            trace[base + 10] = trace_f32_bits(eps);
            trace[base + 11] = trace_f32_bits(mean_square_plus_eps);
            trace[base + 12] = trace_f32_bits(rms_inv);
            trace[base + 13] = (unsigned long long)group_size;
            trace[base + 14] = (unsigned long long)reduction_offset;
            trace[base + 15] = 1ULL;
        }
    }
}

extern "C" __global__ void prefill_trace_fused_rmsnorm_input_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ residual_row,
    const __nv_bfloat16* __restrict__ hidden_row,
    int width,
    int dim_start,
    int element_count)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dim_start + out_idx;
    if (out_idx >= element_count || idx >= width) return;

    const unsigned short* residual_bits = reinterpret_cast<const unsigned short*>(residual_row);
    const unsigned short* hidden_bits = reinterpret_cast<const unsigned short*>(hidden_row);
    float residual = bf16_to_float(residual_row[idx]);
    float hidden = bf16_to_float(hidden_row[idx]);
    float added = residual + hidden;
    __nv_bfloat16 rounded = float_to_bf16(added);
    unsigned short rounded_bits = *reinterpret_cast<unsigned short*>(&rounded);

    int base = (entry_base_idx + out_idx) * 16;
    trace[base + 0] = (unsigned long long)stage_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)idx;
    trace[base + 6] = (unsigned long long)width;
    trace[base + 7] = (unsigned long long)residual_bits[idx];
    trace[base + 8] = (unsigned long long)hidden_bits[idx];
    trace[base + 9] = (unsigned long long)rounded_bits;
    trace[base + 10] = trace_f32_bits(residual);
    trace[base + 11] = trace_f32_bits(hidden);
    trace[base + 12] = trace_f32_bits(added);
    trace[base + 13] = trace_f32_bits(bf16_to_float(rounded));
    trace[base + 14] = 0ULL;
    trace[base + 15] = 1ULL;
}

extern "C" __global__ void prefill_trace_fused_add_source_metadata_kernel(
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width,
    unsigned long long lhs_row_ptr,
    unsigned long long rhs_row_ptr,
    unsigned long long rounded_store_row_ptr,
    unsigned long long flags)
{
    int base = entry_idx * 16;
    trace[base + 0] = (unsigned long long)stage_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)row_idx;
    trace[base + 6] = (unsigned long long)width;
    trace[base + 7] = lhs_row_ptr;
    trace[base + 8] = rhs_row_ptr;
    trace[base + 9] = rounded_store_row_ptr;
    trace[base + 10] = flags;
    trace[base + 11] = (lhs_row_ptr == rhs_row_ptr) ? 1ULL : 0ULL;
    trace[base + 12] = (lhs_row_ptr == rounded_store_row_ptr) ? 1ULL : 0ULL;
    trace[base + 13] = (rhs_row_ptr == rounded_store_row_ptr) ? 1ULL : 0ULL;
    trace[base + 14] = 0ULL;
    trace[base + 15] = 1ULL;
}

extern "C" __global__ void prefill_trace_fused_rmsnorm_output_detail_kernel(
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    const __nv_bfloat16* __restrict__ norm_input_row,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ output_row,
    int width,
    float eps,
    int dim_start,
    int element_count)
{
    if (width <= 0) return;

    __shared__ float s_ms[1024];
    int tid = threadIdx.x;
    float local_ss = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        float v = bf16_to_float(norm_input_row[i]);
        local_ss += v * v;
    }
    s_ms[tid] = local_ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ms[tid] += s_ms[tid + stride];
        }
        __syncthreads();
    }
    float rms_inv = rsqrtf(s_ms[0] / (float)width + eps);

    const unsigned short* input_bits = reinterpret_cast<const unsigned short*>(norm_input_row);
    const unsigned short* weight_bits = reinterpret_cast<const unsigned short*>(weight);
    const unsigned short* output_bits = reinterpret_cast<const unsigned short*>(output_row);

    for (int out_idx = tid; out_idx < element_count; out_idx += blockDim.x) {
        int idx = dim_start + out_idx;
        if (idx >= width) continue;
        float norm_input = bf16_to_float(norm_input_row[idx]);
        float weight_value = bf16_to_float(weight[idx]);
        float pre_store = norm_input * rms_inv * weight_value;
        int base = (entry_base_idx + out_idx) * 16;
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)idx;
        trace[base + 6] = (unsigned long long)width;
        trace[base + 7] = (unsigned long long)input_bits[idx];
        trace[base + 8] = (unsigned long long)weight_bits[idx];
        trace[base + 9] = (unsigned long long)output_bits[idx];
        trace[base + 10] = trace_f32_bits(norm_input);
        trace[base + 11] = trace_f32_bits(weight_value);
        trace[base + 12] = trace_f32_bits(rms_inv);
        trace[base + 13] = trace_f32_bits(pre_store);
        trace[base + 14] = trace_f32_bits(bf16_to_float(output_row[idx]));
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_rmsnorm_summary_kernel(
    const __nv_bfloat16* __restrict__ x_row,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ out_row,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width,
    float eps,
    int mode)
{
    if (width <= 0) return;

    __shared__ float s_ms[1024];
    __shared__ unsigned long long s_hash[1024];
    __shared__ double s_sum[1024];
    __shared__ double s_sumsq[1024];
    __shared__ float s_min[1024];
    __shared__ float s_max[1024];
    __shared__ int s_finite[1024];
    __shared__ int s_nan[1024];
    __shared__ int s_inf[1024];

    int tid = threadIdx.x;
    if (tid >= 1024) return;

    float local_ss = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    s_ms[tid] = local_ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ms[tid] += s_ms[tid + stride];
        }
        __syncthreads();
    }

    float mean_square = s_ms[0] / (float)width;
    float rms_inv = rsqrtf(mean_square + eps);
    if (tid == 0) {
        float seq_ss = 0.0f;
        for (int i = 0; i < width; ++i) {
            float v = bf16_to_float(x_row[i]);
            seq_ss += v * v;
        }
        s_ms[0] = seq_ss / (float)width;
    }
    __syncthreads();
    float seq_mean_square = s_ms[0];
    float seq_rms_inv = rsqrtf(seq_mean_square + eps);

    float contig_local_ss = 0.0f;
    int elems_per_thread = (width + blockDim.x - 1) / blockDim.x;
    int start = tid * elems_per_thread;
    int end = min(width, start + elems_per_thread);
    for (int i = start; i < end; ++i) {
        float v = bf16_to_float(x_row[i]);
        contig_local_ss += v * v;
    }
    s_ms[tid] = contig_local_ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ms[tid] += s_ms[tid + stride];
        }
        __syncthreads();
    }
    float contig_mean_square = s_ms[0] / (float)width;
    float contig_rms_inv = rsqrtf(contig_mean_square + eps);

    int value_count = ((mode >= 2 && mode <= 4) || mode == 8 || mode == 9 || mode == 12 || mode == 13) ? 1 : width;

    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(tid + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = tid; i < value_count; i += blockDim.x) {
        float v;
        if (mode == 0) {
            v = bf16_to_float(x_row[i]);
        } else if (mode == 1) {
            v = bf16_to_float(weight[i]);
        } else if (mode == 2) {
            v = mean_square;
        } else if (mode == 3) {
            v = eps;
        } else if (mode == 4) {
            v = rms_inv;
        } else if (mode == 5) {
            v = bf16_to_float(x_row[i]) * rms_inv;
        } else if (mode == 6) {
            v = bf16_to_float(x_row[i]) * rms_inv * bf16_to_float(weight[i]);
        } else if (mode == 8) {
            v = seq_mean_square;
        } else if (mode == 9) {
            v = seq_rms_inv;
        } else if (mode == 10) {
            v = bf16_to_float(x_row[i]) * seq_rms_inv * bf16_to_float(weight[i]);
        } else if (mode == 11) {
            float out_v = bf16_to_float(x_row[i]) * seq_rms_inv * bf16_to_float(weight[i]);
            v = bf16_to_float(float_to_bf16(out_v));
        } else if (mode == 12) {
            v = contig_mean_square;
        } else if (mode == 13) {
            v = contig_rms_inv;
        } else if (mode == 14) {
            v = bf16_to_float(x_row[i]) * contig_rms_inv * bf16_to_float(weight[i]);
        } else if (mode == 15) {
            float out_v = bf16_to_float(x_row[i]) * contig_rms_inv * bf16_to_float(weight[i]);
            v = bf16_to_float(float_to_bf16(out_v));
        } else {
            v = bf16_to_float(out_row[i]);
        }

        unsigned long long raw = trace_f32_bits(v);
        h ^= raw;
        h *= 1099511628211ULL;

        if (isfinite(v)) {
            finite_count++;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        } else if (isnan(v)) {
            nan_count++;
        } else {
            inf_count++;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    s_nan[tid] = nan_count;
    s_inf[tid] = inf_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
            s_nan[tid] += s_nan[tid + stride];
            s_inf[tid] += s_inf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)value_count;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = (unsigned long long)s_nan[0];
        trace[base + 14] = (unsigned long long)s_inf[0];
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_mamba2_dt_softplus_summary_kernel(
    const __nv_bfloat16* __restrict__ dt_row,
    const float* __restrict__ dt_bias,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width,
    int mode)
{
    if (width <= 0) return;

    __shared__ unsigned long long s_hash[256];
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_finite[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];

    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(threadIdx.x + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float raw_dt = bf16_to_float(dt_row[i]);
        float bias = (dt_bias != NULL) ? dt_bias[i] : 0.0f;
        float biased_dt = raw_dt + bias;
        float v;
        if (mode == 0) {
            v = raw_dt;
        } else if (mode == 1) {
            v = bias;
        } else if (mode == 2) {
            v = biased_dt;
        } else {
            v = mamba2_chunk_cumsum_softplus(biased_dt);
        }
        unsigned long long raw = trace_f32_bits(v);
        h ^= raw;
        h *= 1099511628211ULL;

        if (isfinite(v)) {
            finite_count++;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        } else if (isnan(v)) {
            nan_count++;
        } else {
            inf_count++;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    int tid = threadIdx.x;
    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    s_nan[tid] = nan_count;
    s_inf[tid] = inf_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
            s_nan[tid] += s_nan[tid + stride];
            s_inf[tid] += s_inf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)width;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = (unsigned long long)s_nan[0];
        trace[base + 14] = (unsigned long long)s_inf[0];
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_mamba2_ssd_scan_summary_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ B_mat,
    const __nv_bfloat16* __restrict__ C_mat,
    const float* __restrict__ D_vec,
    const float* __restrict__ ssm_state,
    const float* __restrict__ dt_bias,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int mode)
{
    if (L <= 0 || n_heads <= 0 || head_dim <= 0 || state_size <= 0 || n_groups <= 0) return;
    int heads_per_group = n_heads / n_groups;
    if (heads_per_group <= 0) return;
    int t_last = row_idx;
    if (t_last < 0) t_last = L - 1;
    if (t_last >= L) t_last = L - 1;

    int d_inner = n_heads * head_dim;
    int value_count;
    if (mode <= 4) {
        value_count = n_heads;
    } else if (mode <= 7) {
        value_count = d_inner;
    } else if (mode <= 20) {
        value_count = d_inner * state_size;
    } else {
        value_count = d_inner;
    }
    if (value_count <= 0) return;

    __shared__ unsigned long long s_hash[256];
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_finite[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];

    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(threadIdx.x + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = threadIdx.x; i < value_count; i += blockDim.x) {
        int head;
        int d = 0;
        int state_idx = 0;
        if (mode <= 4) {
            head = i;
        } else if (mode <= 7) {
            head = i / head_dim;
            d = i - head * head_dim;
        } else if (mode <= 20) {
            int hd = i / state_size;
            state_idx = i - hd * state_size;
            head = hd / head_dim;
            d = hd - head * head_dim;
        } else {
            head = i / head_dim;
            d = i - head * head_dim;
        }
        int group = head / heads_per_group;
        float A_val = mamba2_ssd_a_value(A_log[head]);
        float raw_dt = bf16_to_float(dt_in[t_last * n_heads + head]);
        float dt = raw_dt + ((dt_bias != NULL) ? dt_bias[head] : 0.0f);
        dt = mamba2_chunk_cumsum_softplus(dt);
        float v = 0.0f;

        if (mode == 0) {
            v = A_val;
        } else if (mode == 1) {
            v = (D_vec != NULL) ? D_vec[head] : 0.0f;
        } else if (mode == 2) {
            v = A_val * dt;
        } else if (mode == 3) {
            float acc = 0.0f;
            for (int t = 0; t <= t_last; t++) {
                float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
                dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
                dt_t = mamba2_chunk_cumsum_softplus(dt_t);
                acc += A_val * dt_t;
            }
            v = acc;
        } else if (mode == 4) {
            v = __expf(A_val * dt);
        } else if (mode == 5) {
            float x_val = bf16_to_float(x[(t_last * n_heads + head) * head_dim + d]);
            float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
            v = D_val * x_val;
        } else if (mode == 6 || mode == 7) {
            float contrib = 0.0f;
            const float* h_state = ssm_state + ((int64_t)head * head_dim + d) * state_size;
            for (int s = 0; s < state_size; s++) {
                float C_val = bf16_to_float(C_mat[(t_last * n_groups + group) * state_size + s]);
                contrib += C_val * h_state[s];
            }
            if (mode == 6) {
                v = contrib;
            } else {
                float x_val = bf16_to_float(x[(t_last * n_heads + head) * head_dim + d]);
                float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
                v = D_val * x_val + contrib;
            }
        } else if (mode == 8) {
            float x_val = bf16_to_float(x[(t_last * n_heads + head) * head_dim + d]);
            float B_val = bf16_to_float(B_mat[(t_last * n_groups + group) * state_size + state_idx]);
            v = B_val * dt * x_val;
        } else if (mode == 9) {
            const float* h_state = ssm_state + ((int64_t)head * head_dim + d) * state_size;
            float C_val = bf16_to_float(C_mat[(t_last * n_groups + group) * state_size + state_idx]);
            v = C_val * h_state[state_idx];
        } else if (mode <= 20) {
            float pre_state = 0.0f;
            for (int t = 0; t < t_last; t++) {
                float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
                dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
                dt_t = mamba2_chunk_cumsum_softplus(dt_t);
                float A_bar_t = __expf(A_val * dt_t);
                float x_t = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
                float B_t = bf16_to_float(B_mat[(t * n_groups + group) * state_size + state_idx]);
                float B_bar_t = bf16_to_float(float_to_bf16(B_t * dt_t));
                pre_state = A_bar_t * pre_state + B_bar_t * x_t;
            }

            float A_bar = __expf(A_val * dt);
            float x_val = bf16_to_float(x[(t_last * n_heads + head) * head_dim + d]);
            float B_val = bf16_to_float(B_mat[(t_last * n_groups + group) * state_size + state_idx]);
            float post_decay = A_bar * pre_state;
            float update_fp32 = B_val * dt * x_val;
            float bdt_bf16 = bf16_to_float(float_to_bf16(B_val * dt));
            float update_bf16_bdt = bdt_bf16 * x_val;
            float post_fp32 = post_decay + update_fp32;
            float post_bf16_update = post_decay + update_bf16_bdt;

            float da_final = 0.0f;
            float da_at_selected = 0.0f;
            for (int t = 0; t < L; t++) {
                float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
                dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
                dt_t = mamba2_chunk_cumsum_softplus(dt_t);
                da_final += A_val * dt_t;
                if (t == t_last) {
                    da_at_selected = da_final;
                }
            }
            float selected_scale = __expf(fminf(da_final - da_at_selected, 0.0f)) * dt;
            float final_contrib_fp32 = B_val * selected_scale * x_val;
            float bscale_bf16 = bf16_to_float(float_to_bf16(B_val * selected_scale));
            float final_contrib_bf16_bscale = bscale_bf16 * x_val;

            float chunk_formula_final_fp32 = 0.0f;
            float chunk_formula_final_bf16_bscale = 0.0f;
            float da_running = 0.0f;
            for (int t = 0; t < L; t++) {
                float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
                dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
                dt_t = mamba2_chunk_cumsum_softplus(dt_t);
                da_running += A_val * dt_t;
                float scale_t = __expf(fminf(da_final - da_running, 0.0f)) * dt_t;
                float x_t = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
                float B_t = bf16_to_float(B_mat[(t * n_groups + group) * state_size + state_idx]);
                chunk_formula_final_fp32 += B_t * scale_t * x_t;
                float bscale_t_bf16 = bf16_to_float(float_to_bf16(B_t * scale_t));
                chunk_formula_final_bf16_bscale += bscale_t_bf16 * x_t;
            }

            if (mode == 10) {
                v = pre_state;
            } else if (mode == 11) {
                v = A_bar;
            } else if (mode == 12) {
                v = post_decay;
            } else if (mode == 13) {
                v = update_fp32;
            } else if (mode == 14) {
                v = update_bf16_bdt;
            } else if (mode == 15) {
                v = post_fp32;
            } else if (mode == 16) {
                v = post_bf16_update;
            } else if (mode == 17) {
                v = final_contrib_fp32;
            } else if (mode == 18) {
                v = final_contrib_bf16_bscale;
            } else if (mode == 19) {
                v = chunk_formula_final_fp32;
            } else if (mode == 20) {
                v = chunk_formula_final_bf16_bscale;
            }
        } else {
            float dA_target = 0.0f;
            for (int t = 0; t <= t_last; t++) {
                float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
                dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
                dt_t = mamba2_chunk_cumsum_softplus(dt_t);
                dA_target += A_val * dt_t;
            }

            float c_state_fp32_cbscale = 0.0f;
            float c_state_bf16_cbscale = 0.0f;
            float dA_running = 0.0f;
            for (int t = 0; t <= t_last; t++) {
                float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
                dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
                dt_t = mamba2_chunk_cumsum_softplus(dt_t);
                dA_running += A_val * dt_t;

                int c_row_base_idx = (t_last * n_groups + group) * state_size;
                int b_row_base_idx = (t * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                float scale = __expf(fminf(dA_target - dA_running, 0.0f)) * dt_t;
                float cb_scaled = cb * scale;
                float cb_scaled_bf16 = bf16_to_float(float_to_bf16(cb_scaled));
                float x_t = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
                c_state_fp32_cbscale += cb_scaled * x_t;
                c_state_bf16_cbscale += cb_scaled_bf16 * x_t;
            }

            float x_val = bf16_to_float(x[(t_last * n_heads + head) * head_dim + d]);
            float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
            float d_x = D_val * x_val;
            float y_fp32_cbscale = d_x + c_state_fp32_cbscale;
            float y_bf16_cbscale = d_x + c_state_bf16_cbscale;

            if (mode == 21) {
                v = c_state_fp32_cbscale;
            } else if (mode == 22) {
                v = c_state_bf16_cbscale;
            } else if (mode == 23) {
                v = y_fp32_cbscale;
            } else if (mode == 24) {
                v = y_bf16_cbscale;
            } else if (mode == 25) {
                v = bf16_to_float(float_to_bf16(y_bf16_cbscale));
            }
        }

        unsigned long long raw = trace_f32_bits(v);
        h ^= raw;
        h *= 1099511628211ULL;

        if (isfinite(v)) {
            finite_count++;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        } else if (isnan(v)) {
            nan_count++;
        } else {
            inf_count++;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    int tid = threadIdx.x;
    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    s_nan[tid] = nan_count;
    s_inf[tid] = inf_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
            s_nan[tid] += s_nan[tid + stride];
            s_inf[tid] += s_inf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)value_count;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = (unsigned long long)s_nan[0];
        trace[base + 14] = (unsigned long long)s_inf[0];
        trace[base + 15] = 1ULL;
    }
}

extern "C" __global__ void prefill_trace_mamba2_ssd_output_detail_kernel(
    const __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ B_mat,
    const __nv_bfloat16* __restrict__ C_mat,
    const float* __restrict__ D_vec,
    const float* __restrict__ dt_bias,
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int stage_value_id,
    int stage_component_id,
    int stage_source_id,
    int stage_hash_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus,
    int dim_index)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (L <= 0 || n_heads <= 0 || head_dim <= 0 || state_size <= 0 || n_groups <= 0) return;
    int d_inner = n_heads * head_dim;
    if (dim_index < 0 || dim_index >= d_inner) return;
    int heads_per_group = n_heads / n_groups;
    if (heads_per_group <= 0) return;

    int t_last = row_idx;
    if (t_last < 0) t_last = L - 1;
    if (t_last >= L) t_last = L - 1;

    int head = dim_index / head_dim;
    int d = dim_index - head * head_dim;
    int group = head / heads_per_group;
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    int chunk_start = (t_last / effective_chunk_size) * effective_chunk_size;

    float A_log_value = A_log[head];
    float exp_A_log = mamba2_ssd_exp_a_log(A_log_value);
    float A_val = -exp_A_log;
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    const unsigned short* out_bits = reinterpret_cast<const unsigned short*>(out);
    const unsigned short* x_bits = reinterpret_cast<const unsigned short*>(x);
    const unsigned short* dt_bits = reinterpret_cast<const unsigned short*>(dt_in);

    float dt_raw = bf16_to_float(dt_in[t_last * n_heads + head]);
    float dt_last = dt_raw + ((dt_bias != NULL) ? dt_bias[head] : 0.0f);
    if (use_softplus) dt_last = mamba2_chunk_cumsum_softplus(dt_last);

    float dA_target = 0.0f;
    float dA_chunk_base = 0.0f;
    for (int t = 0; t <= t_last; t++) {
        if (t == chunk_start) {
            dA_chunk_base = dA_target;
        }
        float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
        dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
        if (use_softplus) dt_t = mamba2_chunk_cumsum_softplus(dt_t);
        dA_target += A_val * dt_t;
    }

    float x_last = bf16_to_float(x[(t_last * n_heads + head) * head_dim + d]);
    float d_x = D_val * x_last;

    float c_state_total = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float h = 0.0f;
        for (int t = 0; t <= t_last; t++) {
            float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
            dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
            if (use_softplus) dt_t = mamba2_chunk_cumsum_softplus(dt_t);
            float A_bar = __expf(A_val * dt_t);
            float x_t = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
            float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);
            float B_bar = bf16_to_float(float_to_bf16(B_val * dt_t));
            h = A_bar * h + B_bar * x_t;
        }
        float C_val = bf16_to_float(C_mat[(t_last * n_groups + group) * state_size + s]);
        c_state_total += C_val * h;
    }

    float local_old_state = 0.0f;
    float local_hf_chunk_scan = 0.0f;
    float dA_running = dA_chunk_base;
    for (int u = chunk_start; u <= t_last; u++) {
        float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
        dt_u += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
        float dt_u_scale = dt_u;
        if (use_softplus) {
            dt_u = mamba2_chunk_cumsum_softplus(dt_u);
            dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
        }
        dA_running += A_val * dt_u;
        float decay = __expf(fminf(dA_target - dA_running, 0.0f));

        float old_state_source = 0.0f;
        int c_row_base_idx = (t_last * n_groups + group) * state_size;
        int b_row_base_idx = (u * n_groups + group) * state_size;
        float cb = mamba2_ssd_cb_dot_reverse(
            C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
        if (chunk_start != 0) {
            for (int s = 0; s < state_size; s++) {
                float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
                float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
                old_state_source += C_val * B_bar;
            }
        }

        float scale = decay * dt_u_scale;
        float cb_scaled_bf16 = bf16_to_float(float_to_bf16(cb * scale));
        float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
        local_old_state += old_state_source * decay * x_u;
        local_hf_chunk_scan += cb_scaled_bf16 * x_u;
    }

    float prior_chunk_state = (chunk_start == 0) ? 0.0f : (c_state_total - local_old_state);
    float y = d_x + prior_chunk_state + local_hf_chunk_scan;
    float stored = bf16_to_float(out[(t_last * n_heads + head) * head_dim + d]);
    unsigned int out_raw = out_bits[(t_last * n_heads + head) * head_dim + d];
    unsigned int x_raw = x_bits[(t_last * n_heads + head) * head_dim + d];
    unsigned int dt_raw_bits = dt_bits[t_last * n_heads + head];

    unsigned long long c_hash = 14695981039346656037ULL;
    for (int s = 0; s < state_size; s++) {
        unsigned int raw = reinterpret_cast<const unsigned short*>(C_mat)[(t_last * n_groups + group) * state_size + s];
        c_hash ^= (unsigned long long)(raw & 0xffU);
        c_hash *= 1099511628211ULL;
        c_hash ^= (unsigned long long)((raw >> 8) & 0xffU);
        c_hash *= 1099511628211ULL;
    }
    unsigned long long b_hash = 14695981039346656037ULL;
    unsigned long long x_chunk_hash = 14695981039346656037ULL;
    unsigned long long dt_chunk_hash = 14695981039346656037ULL;
    for (int u = chunk_start; u <= t_last; u++) {
        unsigned int x_u_raw = x_bits[(u * n_heads + head) * head_dim + d];
        x_chunk_hash ^= (unsigned long long)(x_u_raw & 0xffU);
        x_chunk_hash *= 1099511628211ULL;
        x_chunk_hash ^= (unsigned long long)((x_u_raw >> 8) & 0xffU);
        x_chunk_hash *= 1099511628211ULL;

        unsigned int dt_u_raw = dt_bits[u * n_heads + head];
        dt_chunk_hash ^= (unsigned long long)(dt_u_raw & 0xffU);
        dt_chunk_hash *= 1099511628211ULL;
        dt_chunk_hash ^= (unsigned long long)((dt_u_raw >> 8) & 0xffU);
        dt_chunk_hash *= 1099511628211ULL;

        for (int s = 0; s < state_size; s++) {
            unsigned int raw = reinterpret_cast<const unsigned short*>(B_mat)[(u * n_groups + group) * state_size + s];
            b_hash ^= (unsigned long long)(raw & 0xffU);
            b_hash *= 1099511628211ULL;
            b_hash ^= (unsigned long long)((raw >> 8) & 0xffU);
            b_hash *= 1099511628211ULL;
        }
    }

    int base = entry_base_idx * 16;
    trace[base + 0] = (unsigned long long)stage_value_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)dim_index;
    trace[base + 6] = (unsigned long long)d_inner;
    trace[base + 7] = (unsigned long long)out_raw;
    trace[base + 8] = (unsigned long long)x_raw;
    trace[base + 9] = (unsigned long long)dt_raw_bits;
    trace[base + 10] = trace_f32_bits(x_last);
    trace[base + 11] = trace_f32_bits(dt_raw);
    trace[base + 12] = trace_f32_bits(dt_last);
    trace[base + 13] = trace_f32_bits(y);
    trace[base + 14] = trace_f32_bits(stored);
    trace[base + 15] = (unsigned long long)head;

    int base2 = (entry_base_idx + 1) * 16;
    trace[base2 + 0] = (unsigned long long)stage_component_id;
    trace[base2 + 1] = (unsigned long long)layer_idx;
    trace[base2 + 2] = (unsigned long long)chunk_idx;
    trace[base2 + 3] = (unsigned long long)absolute_position;
    trace[base2 + 4] = (unsigned long long)token_id;
    trace[base2 + 5] = (unsigned long long)dim_index;
    trace[base2 + 6] = (unsigned long long)head;
    trace[base2 + 7] = (unsigned long long)d;
    trace[base2 + 8] = trace_f32_bits(d_x);
    trace[base2 + 9] = trace_f32_bits(prior_chunk_state);
    trace[base2 + 10] = trace_f32_bits(local_hf_chunk_scan);
    trace[base2 + 11] = trace_f32_bits(c_state_total);
    trace[base2 + 12] = trace_f32_bits(local_old_state);
    trace[base2 + 13] = trace_f32_bits(A_val);
    trace[base2 + 14] = trace_f32_bits(A_log_value);
    trace[base2 + 15] = trace_f32_bits(exp_A_log);

    int base3 = (entry_base_idx + 2) * 16;
    trace[base3 + 0] = (unsigned long long)stage_source_id;
    trace[base3 + 1] = (unsigned long long)layer_idx;
    trace[base3 + 2] = (unsigned long long)chunk_idx;
    trace[base3 + 3] = (unsigned long long)absolute_position;
    trace[base3 + 4] = (unsigned long long)token_id;
    trace[base3 + 5] = (unsigned long long)dim_index;
    trace[base3 + 6] = (unsigned long long)row_idx;
    trace[base3 + 7] = (unsigned long long)(x + (int64_t)t_last * d_inner);
    trace[base3 + 8] = (unsigned long long)(dt_in + (int64_t)t_last * n_heads);
    trace[base3 + 9] = (unsigned long long)(B_mat + ((int64_t)t_last * n_groups + group) * state_size);
    trace[base3 + 10] = (unsigned long long)(C_mat + ((int64_t)t_last * n_groups + group) * state_size);
    trace[base3 + 11] = (unsigned long long)(out + (int64_t)t_last * d_inner);
    trace[base3 + 12] = (unsigned long long)n_heads;
    trace[base3 + 13] = (unsigned long long)head_dim;
    trace[base3 + 14] = (unsigned long long)state_size;
    trace[base3 + 15] = (unsigned long long)n_groups;

    int base4 = (entry_base_idx + 3) * 16;
    trace[base4 + 0] = (unsigned long long)stage_hash_id;
    trace[base4 + 1] = (unsigned long long)layer_idx;
    trace[base4 + 2] = (unsigned long long)chunk_idx;
    trace[base4 + 3] = (unsigned long long)absolute_position;
    trace[base4 + 4] = (unsigned long long)token_id;
    trace[base4 + 5] = (unsigned long long)dim_index;
    trace[base4 + 6] = (unsigned long long)(chunk_start + 1);
    trace[base4 + 7] = (unsigned long long)t_last;
    trace[base4 + 8] = c_hash;
    trace[base4 + 9] = b_hash;
    trace[base4 + 10] = x_chunk_hash;
    trace[base4 + 11] = dt_chunk_hash;
    trace[base4 + 12] = trace_f32_bits(dA_chunk_base);
    trace[base4 + 13] = trace_f32_bits(dA_target);
    trace[base4 + 14] = (unsigned long long)effective_chunk_size;
    trace[base4 + 15] = 1ULL;

}

extern "C" __global__ void prefill_trace_mamba2_ssd_local_scan_detail_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ B_mat,
    const __nv_bfloat16* __restrict__ C_mat,
    const float* __restrict__ dt_bias,
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int max_entries,
    int stage_summary_id,
    int stage_token_id,
    int stage_cb_id,
    int stage_dt_id,
    int stage_scale_id,
    int stage_accum_candidate_id,
    int stage_cb_term_id,
    int stage_cb_term_summary_id,
    int selected_local_scan_token,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus,
    int dim_index)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (L <= 0 || n_heads <= 0 || head_dim <= 0 || state_size <= 0 || n_groups <= 0) return;
    if (entry_base_idx < 0 || entry_base_idx >= max_entries) return;
    int d_inner = n_heads * head_dim;
    if (dim_index < 0 || dim_index >= d_inner) return;
    int heads_per_group = n_heads / n_groups;
    if (heads_per_group <= 0) return;

    int t_last = row_idx;
    if (t_last < 0) t_last = L - 1;
    if (t_last >= L) t_last = L - 1;

    int head = dim_index / head_dim;
    int d = dim_index - head * head_dim;
    int group = head / heads_per_group;
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    int chunk_start = (t_last / effective_chunk_size) * effective_chunk_size;

    float A_val = mamba2_ssd_a_value(A_log[head]);
    const unsigned short* x_bits = reinterpret_cast<const unsigned short*>(x);
    const unsigned short* dt_bits = reinterpret_cast<const unsigned short*>(dt_in);

    float dA_target = 0.0f;
    float dA_chunk_base = 0.0f;
    for (int t = 0; t <= t_last; t++) {
        if (t == chunk_start) {
            dA_chunk_base = dA_target;
        }
        float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
        dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
        if (use_softplus) dt_t = mamba2_chunk_cumsum_softplus(dt_t);
        dA_target += A_val * dt_t;
    }

    float local_scan_forward = 0.0f;
    float local_scan_kahan = 0.0f;
    float local_scan_kahan_c = 0.0f;
    float local_scan_fp32_cbscale = 0.0f;
    float local_scan_mul_rn_add_rn = 0.0f;
    float local_scan_fma_rn = 0.0f;
    float dA_running = dA_chunk_base;
    int entry_idx = entry_base_idx + 1;
    int cb_term_entry_idx = entry_base_idx + 1 + (t_last - chunk_start + 1) * 5;
    int cb_term_count = 0;

    for (int u = chunk_start; u <= t_last; u++) {
        float raw_dt_u = bf16_to_float(dt_in[u * n_heads + head]);
        float dt_bias_value = (dt_bias != NULL) ? dt_bias[head] : 0.0f;
        float dt_plus_bias = raw_dt_u + dt_bias_value;
        __nv_bfloat16 dt_plus_bias_bf16_raw = float_to_bf16(dt_plus_bias);
        float dt_plus_bias_bf16 = bf16_to_float(dt_plus_bias_bf16_raw);
        unsigned int dt_plus_bias_bf16_bits =
            (unsigned int)(*reinterpret_cast<unsigned short*>(&dt_plus_bias_bf16_raw));
        float softplus_log_fast = dt_plus_bias;
        float softplus_log1p_exp = dt_plus_bias;
        float softplus_bf16_plus_log_fast = dt_plus_bias_bf16;
        if (use_softplus) {
            softplus_log_fast = mamba2_chunk_cumsum_softplus(dt_plus_bias);
            softplus_log1p_exp = log1pf(expf(dt_plus_bias));
            softplus_bf16_plus_log_fast = mamba2_chunk_cumsum_softplus(dt_plus_bias_bf16);
        }
        float dt_u = softplus_log_fast;
        float dt_u_scale = softplus_log_fast;
        dA_running += A_val * dt_u;
        float decay = __expf(fminf(dA_target - dA_running, 0.0f));

        float cb_forward = 0.0f;
        float cb_reverse = 0.0f;
        float top_state_abs = -1.0f;
        float top_state_contrib = 0.0f;
        int top_state_index = 0;
        bool capture_cb_terms = selected_local_scan_token >= chunk_start &&
            selected_local_scan_token <= t_last &&
            u == selected_local_scan_token &&
            stage_cb_term_id != 0;
        for (int s = 0; s < state_size; s++) {
            float C_val = bf16_to_float(C_mat[(t_last * n_groups + group) * state_size + s]);
            float B_val = bf16_to_float(B_mat[(u * n_groups + group) * state_size + s]);
            float contrib = C_val * B_val;
            float partial_before = cb_forward;
            cb_forward += contrib;
            float abs_contrib = fabsf(contrib);
            if (abs_contrib > top_state_abs) {
                top_state_abs = abs_contrib;
                top_state_contrib = contrib;
                top_state_index = s;
            }
            if (capture_cb_terms && cb_term_entry_idx < max_entries) {
                const unsigned short* c_bits = reinterpret_cast<const unsigned short*>(C_mat);
                const unsigned short* b_bits = reinterpret_cast<const unsigned short*>(B_mat);
                int c_idx = (t_last * n_groups + group) * state_size + s;
                int b_idx = (u * n_groups + group) * state_size + s;
                int base = cb_term_entry_idx * 16;
                trace[base + 0] = (unsigned long long)stage_cb_term_id;
                trace[base + 1] = (unsigned long long)layer_idx;
                trace[base + 2] = (unsigned long long)chunk_idx;
                trace[base + 3] = (unsigned long long)absolute_position;
                trace[base + 4] = (unsigned long long)token_id;
                trace[base + 5] = (unsigned long long)dim_index;
                trace[base + 6] = (unsigned long long)u;
                trace[base + 7] = (unsigned long long)s;
                trace[base + 8] = (unsigned long long)c_bits[c_idx];
                trace[base + 9] = (unsigned long long)b_bits[b_idx];
                trace[base + 10] = trace_f32_bits(C_val);
                trace[base + 11] = trace_f32_bits(B_val);
                trace[base + 12] = trace_f32_bits(contrib);
                trace[base + 13] = trace_f32_bits(partial_before);
                trace[base + 14] = trace_f32_bits(cb_forward);
                trace[base + 15] = 1ULL;
                cb_term_entry_idx++;
                cb_term_count++;
            }
        }

        for (int s = state_size - 1; s >= 0; s--) {
            float C_val = bf16_to_float(C_mat[(t_last * n_groups + group) * state_size + s]);
            float B_val = bf16_to_float(B_mat[(u * n_groups + group) * state_size + s]);
            cb_reverse += C_val * B_val;
        }
        if (capture_cb_terms && stage_cb_term_summary_id != 0 && cb_term_entry_idx < max_entries) {
            int base = cb_term_entry_idx * 16;
            trace[base + 0] = (unsigned long long)stage_cb_term_summary_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)u;
            trace[base + 7] = (unsigned long long)state_size;
            trace[base + 8] = trace_f32_bits(cb_forward);
            trace[base + 9] = trace_f32_bits(cb_reverse);
            trace[base + 10] = (unsigned long long)top_state_index;
            trace[base + 11] = trace_f32_bits(top_state_contrib);
            trace[base + 12] = (unsigned long long)selected_local_scan_token;
            trace[base + 13] = (unsigned long long)group;
            trace[base + 14] = (unsigned long long)cb_term_count;
            trace[base + 15] = 1ULL;
            cb_term_entry_idx++;
        }

        float scale = decay * dt_u_scale;
        float cb_scaled_fp32 = cb_forward * scale;
        __nv_bfloat16 cb_scaled_bf16_raw = float_to_bf16(cb_scaled_fp32);
        float cb_scaled_bf16 = bf16_to_float(cb_scaled_bf16_raw);
        unsigned int cb_scaled_bf16_bits =
            (unsigned int)(*reinterpret_cast<unsigned short*>(&cb_scaled_bf16_raw));
        float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
        float term_bf16 = cb_scaled_bf16 * x_u;
        float term_fp32 = cb_scaled_fp32 * x_u;
        float term_mul_rn = trace_mul_rn_f32(cb_scaled_bf16, x_u);

        local_scan_forward += term_bf16;
        local_scan_mul_rn_add_rn = trace_add_rn_f32(local_scan_mul_rn_add_rn, term_mul_rn);
        local_scan_fma_rn = trace_fma_rn_f32(cb_scaled_bf16, x_u, local_scan_fma_rn);
        float kahan_y = term_bf16 - local_scan_kahan_c;
        float kahan_t = local_scan_kahan + kahan_y;
        local_scan_kahan_c = (kahan_t - local_scan_kahan) - kahan_y;
        local_scan_kahan = kahan_t;
        local_scan_fp32_cbscale += term_fp32;

        if (entry_idx < max_entries) {
            int base = entry_idx * 16;
            trace[base + 0] = (unsigned long long)stage_token_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)(u + 1);
            trace[base + 7] = (unsigned long long)head;
            trace[base + 8] = (unsigned long long)x_bits[(u * n_heads + head) * head_dim + d];
            trace[base + 9] = (unsigned long long)dt_bits[u * n_heads + head];
            trace[base + 10] = trace_f32_bits(x_u);
            trace[base + 11] = trace_f32_bits(dt_u);
            trace[base + 12] = trace_f32_bits(dA_running);
            trace[base + 13] = trace_f32_bits(decay);
            trace[base + 14] = trace_f32_bits(term_bf16);
            trace[base + 15] = trace_f32_bits(local_scan_forward);
        }
        entry_idx++;

        if (entry_idx < max_entries) {
            int base = entry_idx * 16;
            trace[base + 0] = (unsigned long long)stage_cb_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)(u + 1);
            trace[base + 7] = (unsigned long long)top_state_index;
            trace[base + 8] = trace_f32_bits(cb_forward);
            trace[base + 9] = trace_f32_bits(cb_reverse);
            trace[base + 10] = trace_f32_bits(cb_scaled_fp32);
            trace[base + 11] = trace_f32_bits(cb_scaled_bf16);
            trace[base + 12] = trace_f32_bits(term_fp32);
            trace[base + 13] = trace_f32_bits(top_state_contrib);
        trace[base + 14] = (unsigned long long)cb_scaled_bf16_bits;
            trace[base + 15] = 1ULL;
        }
        entry_idx++;

        if (entry_idx < max_entries) {
            int base = entry_idx * 16;
            trace[base + 0] = (unsigned long long)stage_dt_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)(u + 1);
            trace[base + 7] = (unsigned long long)head;
            trace[base + 8] = (unsigned long long)dt_bits[u * n_heads + head];
            trace[base + 9] = trace_f32_bits(raw_dt_u);
            trace[base + 10] = trace_f32_bits(dt_bias_value);
            trace[base + 11] = trace_f32_bits(dt_plus_bias);
            trace[base + 12] = (unsigned long long)dt_plus_bias_bf16_bits;
            trace[base + 13] = trace_f32_bits(softplus_log_fast);
            trace[base + 14] = trace_f32_bits(softplus_log1p_exp);
            trace[base + 15] = trace_f32_bits(softplus_bf16_plus_log_fast);
        }
        entry_idx++;

        float scale_log1p_exp = decay * softplus_log1p_exp;
        float scale_bf16_plus_log_fast = decay * softplus_bf16_plus_log_fast;
        __nv_bfloat16 cb_scaled_log1p_exp_raw = float_to_bf16(cb_forward * scale_log1p_exp);
        __nv_bfloat16 cb_scaled_bf16_plus_raw =
            float_to_bf16(cb_forward * scale_bf16_plus_log_fast);
        unsigned int cb_scaled_log1p_exp_bits =
            (unsigned int)(*reinterpret_cast<unsigned short*>(&cb_scaled_log1p_exp_raw));
        unsigned int cb_scaled_bf16_plus_bits =
            (unsigned int)(*reinterpret_cast<unsigned short*>(&cb_scaled_bf16_plus_raw));
        if (entry_idx < max_entries) {
            int base = entry_idx * 16;
            trace[base + 0] = (unsigned long long)stage_scale_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)(u + 1);
            trace[base + 7] = (unsigned long long)head;
            trace[base + 8] = trace_f32_bits(decay);
            trace[base + 9] = trace_f32_bits(scale);
            trace[base + 10] = trace_f32_bits(scale_log1p_exp);
            trace[base + 11] = trace_f32_bits(scale_bf16_plus_log_fast);
            trace[base + 12] = (unsigned long long)cb_scaled_bf16_bits;
            trace[base + 13] = (unsigned long long)cb_scaled_log1p_exp_bits;
            trace[base + 14] = (unsigned long long)cb_scaled_bf16_plus_bits;
            trace[base + 15] = 1ULL;
        }
        entry_idx++;

        if (entry_idx < max_entries) {
            int base = entry_idx * 16;
            trace[base + 0] = (unsigned long long)stage_accum_candidate_id;
            trace[base + 1] = (unsigned long long)layer_idx;
            trace[base + 2] = (unsigned long long)chunk_idx;
            trace[base + 3] = (unsigned long long)absolute_position;
            trace[base + 4] = (unsigned long long)token_id;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)(u + 1);
            trace[base + 7] = (unsigned long long)head;
            trace[base + 8] = trace_f32_bits(term_bf16);
            trace[base + 9] = trace_f32_bits(local_scan_forward);
            trace[base + 10] = trace_f32_bits(term_mul_rn);
            trace[base + 11] = trace_f32_bits(local_scan_mul_rn_add_rn);
            trace[base + 12] = trace_f32_bits(local_scan_fma_rn);
            trace[base + 13] = trace_f32_bits(local_scan_kahan);
            trace[base + 14] = trace_f32_bits(local_scan_fp32_cbscale);
            trace[base + 15] = 1ULL;
        }
        entry_idx++;
    }

    int base = entry_base_idx * 16;
    trace[base + 0] = (unsigned long long)stage_summary_id;
    trace[base + 1] = (unsigned long long)layer_idx;
    trace[base + 2] = (unsigned long long)chunk_idx;
    trace[base + 3] = (unsigned long long)absolute_position;
    trace[base + 4] = (unsigned long long)token_id;
    trace[base + 5] = (unsigned long long)dim_index;
    trace[base + 6] = (unsigned long long)head;
    trace[base + 7] = (unsigned long long)d;
    trace[base + 8] = trace_f32_bits(local_scan_forward);
    trace[base + 9] = trace_f32_bits(local_scan_kahan);
    trace[base + 10] = trace_f32_bits(local_scan_fp32_cbscale);
    trace[base + 11] = trace_f32_bits(dA_chunk_base);
    trace[base + 12] = trace_f32_bits(dA_target);
    trace[base + 13] = trace_f32_bits(A_val);
    trace[base + 14] = (unsigned long long)(chunk_start + 1);
    trace[base + 15] = (unsigned long long)t_last;
}

extern "C" __global__ void prefill_trace_mamba2_ssd_shadow_context_detail_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ B_mat,
    const __nv_bfloat16* __restrict__ C_mat,
    const float* __restrict__ D_vec,
    const float* __restrict__ ssm_state,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ out,
    unsigned long long* __restrict__ trace,
    int entry_base_idx,
    int max_entries,
    int stage_pointer_id,
    int stage_index_id,
    int stage_sample_id,
    int stage_summary_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int row_idx,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus,
    int dim_index)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (L <= 0 || n_heads <= 0 || head_dim <= 0 || state_size <= 0 || n_groups <= 0) return;
    if (entry_base_idx < 0 || entry_base_idx + 5 >= max_entries) return;
    int d_inner = n_heads * head_dim;
    if (dim_index < 0 || dim_index >= d_inner) return;
    int heads_per_group = n_heads / n_groups;
    if (heads_per_group <= 0) return;

    int t_last = row_idx;
    if (t_last < 0) t_last = L - 1;
    if (t_last >= L) t_last = L - 1;

    int head = dim_index / head_dim;
    int d = dim_index - head * head_dim;
    int group = head / heads_per_group;
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    int chunk_start = (t_last / effective_chunk_size) * effective_chunk_size;
    int mid_u = chunk_start + (t_last - chunk_start) / 2;
    int out_flat_idx = (t_last * n_heads + head) * head_dim + d;
    int x_target_flat_idx = out_flat_idx;
    int dt_target_idx = t_last * n_heads + head;
    int c_row_base_idx = (t_last * n_groups + group) * state_size;
    int b_chunk_start_base_idx = (chunk_start * n_groups + group) * state_size;
    int b_target_base_idx = (t_last * n_groups + group) * state_size;

    int base_ptr = entry_base_idx * 16;
    trace[base_ptr + 0] = (unsigned long long)stage_pointer_id;
    trace[base_ptr + 1] = (unsigned long long)layer_idx;
    trace[base_ptr + 2] = (unsigned long long)chunk_idx;
    trace[base_ptr + 3] = (unsigned long long)absolute_position;
    trace[base_ptr + 4] = (unsigned long long)t_last;
    trace[base_ptr + 5] = (unsigned long long)dim_index;
    trace[base_ptr + 6] = (unsigned long long)x;
    trace[base_ptr + 7] = (unsigned long long)dt_in;
    trace[base_ptr + 8] = (unsigned long long)B_mat;
    trace[base_ptr + 9] = (unsigned long long)C_mat;
    trace[base_ptr + 10] = (unsigned long long)A_log;
    trace[base_ptr + 11] = (unsigned long long)D_vec;
    trace[base_ptr + 12] = (unsigned long long)dt_bias;
    trace[base_ptr + 13] = (unsigned long long)out;
    trace[base_ptr + 14] = (unsigned long long)ssm_state;
    trace[base_ptr + 15] = 1ULL;

    int base_idx = (entry_base_idx + 1) * 16;
    trace[base_idx + 0] = (unsigned long long)stage_index_id;
    trace[base_idx + 1] = (unsigned long long)layer_idx;
    trace[base_idx + 2] = (unsigned long long)chunk_idx;
    trace[base_idx + 3] = (unsigned long long)t_last;
    trace[base_idx + 4] = (unsigned long long)dim_index;
    trace[base_idx + 5] = (unsigned long long)head;
    trace[base_idx + 6] = (unsigned long long)d;
    trace[base_idx + 7] = (unsigned long long)group;
    trace[base_idx + 8] = (unsigned long long)out_flat_idx;
    trace[base_idx + 9] = (unsigned long long)x_target_flat_idx;
    trace[base_idx + 10] = (unsigned long long)dt_target_idx;
    trace[base_idx + 11] = (unsigned long long)c_row_base_idx;
    trace[base_idx + 12] = (unsigned long long)b_chunk_start_base_idx;
    trace[base_idx + 13] = (unsigned long long)b_target_base_idx;
    trace[base_idx + 14] = (unsigned long long)chunk_start;
    trace[base_idx + 15] = (unsigned long long)effective_chunk_size;

    float A_val = mamba2_ssd_a_value(A_log[head]);
    float dA_target = 0.0f;
    float dA_chunk_base = 0.0f;
    for (int t = 0; t <= t_last; t++) {
        if (t == chunk_start) {
            dA_chunk_base = dA_target;
        }
        float dt_t = bf16_to_float(dt_in[t * n_heads + head]);
        dt_t += (dt_bias != NULL) ? dt_bias[head] : 0.0f;
        if (use_softplus) dt_t = mamba2_chunk_cumsum_softplus(dt_t);
        dA_target += A_val * dt_t;
    }

    const unsigned short* x_bits = reinterpret_cast<const unsigned short*>(x);
    const unsigned short* dt_bits = reinterpret_cast<const unsigned short*>(dt_in);
    float dt_bias_value = (dt_bias != NULL) ? dt_bias[head] : 0.0f;
    float local_scan = 0.0f;
    float dA_running = dA_chunk_base;
    int sample_count = 0;

    for (int u = chunk_start; u <= t_last; u++) {
        float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
        if (dt_bias != NULL) dt_u += dt_bias[head];
        float dt_u_scale = dt_u;
        if (use_softplus) {
            dt_u = mamba2_chunk_cumsum_softplus(dt_u);
            dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
        }
        dA_running += A_val * dt_u;
        float decay = __expf(fminf(dA_target - dA_running, 0.0f));

        float cb_forward = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float C_val = bf16_to_float(C_mat[(t_last * n_groups + group) * state_size + s]);
            float B_val = bf16_to_float(B_mat[(u * n_groups + group) * state_size + s]);
            cb_forward += C_val * B_val;
        }

        float scale = decay * dt_u_scale;
        __nv_bfloat16 cb_scaled_bf16_raw = float_to_bf16(cb_forward * scale);
        float cb_scaled_bf16 = bf16_to_float(cb_scaled_bf16_raw);
        unsigned int cb_scaled_bf16_bits =
            (unsigned int)(*reinterpret_cast<unsigned short*>(&cb_scaled_bf16_raw));
        int x_flat_idx = (u * n_heads + head) * head_dim + d;
        int dt_idx = u * n_heads + head;
        int b_row_base_idx = (u * n_groups + group) * state_size;
        float x_u = bf16_to_float(x[x_flat_idx]);
        float term_bf16 = cb_scaled_bf16 * x_u;
        local_scan += term_bf16;

        if ((u == chunk_start || u == mid_u || u == t_last) && sample_count < 3) {
            int base_sample = (entry_base_idx + 2 + sample_count) * 16;
            trace[base_sample + 0] = (unsigned long long)stage_sample_id;
            trace[base_sample + 1] = (unsigned long long)layer_idx;
            trace[base_sample + 2] = (unsigned long long)chunk_idx;
            trace[base_sample + 3] = (unsigned long long)t_last;
            trace[base_sample + 4] = (unsigned long long)dim_index;
            trace[base_sample + 5] = (unsigned long long)u;
            trace[base_sample + 6] = (unsigned long long)x_flat_idx;
            trace[base_sample + 7] = (unsigned long long)dt_idx;
            trace[base_sample + 8] = (unsigned long long)b_row_base_idx;
            trace[base_sample + 9] = (unsigned long long)c_row_base_idx;
            trace[base_sample + 10] = (unsigned long long)x_bits[x_flat_idx];
            trace[base_sample + 11] = (unsigned long long)dt_bits[dt_idx];
            trace[base_sample + 12] = trace_f32_bits(decay);
            trace[base_sample + 13] = trace_f32_bits(cb_forward);
            trace[base_sample + 14] = (unsigned long long)cb_scaled_bf16_bits;
            trace[base_sample + 15] = trace_f32_bits(term_bf16);
            sample_count++;
        }
    }

    int base_summary = (entry_base_idx + 5) * 16;
    trace[base_summary + 0] = (unsigned long long)stage_summary_id;
    trace[base_summary + 1] = (unsigned long long)layer_idx;
    trace[base_summary + 2] = (unsigned long long)chunk_idx;
    trace[base_summary + 3] = (unsigned long long)t_last;
    trace[base_summary + 4] = (unsigned long long)dim_index;
    trace[base_summary + 5] = (unsigned long long)head;
    trace[base_summary + 6] = (unsigned long long)group;
    trace[base_summary + 7] = (unsigned long long)chunk_start;
    trace[base_summary + 8] = trace_f32_bits(local_scan);
    trace[base_summary + 9] = trace_f32_bits(dA_chunk_base);
    trace[base_summary + 10] = trace_f32_bits(dA_target);
    trace[base_summary + 11] = trace_f32_bits(A_val);
    trace[base_summary + 12] = trace_f32_bits(dt_bias_value);
    trace[base_summary + 13] = (unsigned long long)sample_count;
    trace[base_summary + 14] = (unsigned long long)out_flat_idx;
    trace[base_summary + 15] = 1ULL;
}

extern "C" __global__ void prefill_trace_bf16_pair_sum_row_summary_kernel(
    const __nv_bfloat16* __restrict__ lhs,
    const __nv_bfloat16* __restrict__ rhs,
    unsigned long long* __restrict__ trace,
    int entry_idx,
    int stage_id,
    int layer_idx,
    int chunk_idx,
    int absolute_position,
    int token_id,
    int row_idx,
    int width)
{
    if (width <= 0) return;

    __shared__ unsigned long long s_hash[256];
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_finite[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];

    unsigned long long h = 1469598103934665603ULL ^ (unsigned long long)(threadIdx.x + 1);
    double sum = 0.0;
    double sumsq = 0.0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = bf16_to_float(lhs[i]) + bf16_to_float(rhs[i]);
        unsigned long long raw = trace_f32_bits(v);
        h ^= raw;
        h *= 1099511628211ULL;

        if (isfinite(v)) {
            finite_count++;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        } else if (isnan(v)) {
            nan_count++;
        } else {
            inf_count++;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    int tid = threadIdx.x;
    s_hash[tid] = h;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_min[tid] = min_v;
    s_max[tid] = max_v;
    s_finite[tid] = finite_count;
    s_nan[tid] = nan_count;
    s_inf[tid] = inf_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
            s_hash[tid] *= 1099511628211ULL;
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_finite[tid] += s_finite[tid + stride];
            s_nan[tid] += s_nan[tid + stride];
            s_inf[tid] += s_inf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = entry_idx * 16;
        float mean = (s_finite[0] > 0) ? (float)(s_sum[0] / (double)s_finite[0]) : NAN;
        float l2 = sqrtf((float)s_sumsq[0]);
        trace[base + 0] = (unsigned long long)stage_id;
        trace[base + 1] = (unsigned long long)layer_idx;
        trace[base + 2] = (unsigned long long)chunk_idx;
        trace[base + 3] = (unsigned long long)absolute_position;
        trace[base + 4] = (unsigned long long)token_id;
        trace[base + 5] = (unsigned long long)row_idx;
        trace[base + 6] = (unsigned long long)width;
        trace[base + 7] = s_hash[0];
        trace[base + 8] = trace_f32_bits(mean);
        trace[base + 9] = trace_f32_bits(l2);
        trace[base + 10] = trace_f32_bits(s_min[0]);
        trace[base + 11] = trace_f32_bits(s_max[0]);
        trace[base + 12] = (unsigned long long)s_finite[0];
        trace[base + 13] = (unsigned long long)s_nan[0];
        trace[base + 14] = (unsigned long long)s_inf[0];
        trace[base + 15] = 1ULL;
    }
}

/* ── GQA Prefill Attention ─────────────────────────────────────────────── */

/* Basic tiled GQA attention for prefill. Not FlashAttention-optimized yet,
 * but correct and usable. One block per (query_head, tile) pair.
 *
 * q: [M, num_q_heads, head_dim] bf16
 * k: [M, num_kv_heads, head_dim] bf16 (before KV cache append)
 * v: [M, num_kv_heads, head_dim] bf16
 * out: [M, num_q_heads, head_dim] bf16
 *
 * Causal mask: position i can attend to positions 0..i+start_pos
 */
extern "C" __global__ void gqa_prefill_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    int start_pos)  /* number of previous tokens in KV cache */
{
    /* blockIdx.x = query position, blockIdx.y = query head */
    int qi = blockIdx.x;  /* query token index */
    int qh = blockIdx.y;  /* query head index */
    int kv_h = qh / (num_q_heads / num_kv_heads);  /* corresponding KV head */

    /* Query vector for this position and head */
    const __nv_bfloat16* q_vec = q + ((int64_t)qi * num_q_heads + qh) * head_dim;

    /* Output vector */
    __nv_bfloat16* o_vec = out + ((int64_t)qi * num_q_heads + qh) * head_dim;

    /* Causal attention: attend to positions 0..qi (within this prefill batch) */
    int num_attend = qi + 1;  /* causal mask: can attend to 0..qi inclusive */

    /* Online softmax: maintain running max and sum */
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    /* Accumulate output in fp32 */
    extern __shared__ float smem[];
    float* acc = smem;  /* [head_dim] */

    /* Initialize accumulator */
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }
    __syncthreads();

    /* Iterate over KV positions */
    for (int ki = 0; ki < num_attend; ki++) {
        const __nv_bfloat16* k_vec = k + ((int64_t)ki * num_kv_heads + kv_h) * head_dim;
        const __nv_bfloat16* v_vec = v + ((int64_t)ki * num_kv_heads + kv_h) * head_dim;

        /* Compute Q·K dot product */
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += bf16_to_float(q_vec[d]) * bf16_to_float(k_vec[d]);
        }
        /* Warp reduce the dot product */
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }
        /* Cross-warp reduce via shared memory */
        __shared__ float s_dots[32];  /* max 32 warps */
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (lane_id == 0) s_dots[warp_id] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            int num_warps = (blockDim.x + 31) / 32;
            for (int w = 0; w < num_warps; w++) total += s_dots[w];
            s_dots[0] = total * softmax_scale;
        }
        __syncthreads();
        float score = s_dots[0];

        /* Online softmax update */
        float old_max = max_score;
        if (score > max_score) max_score = score;
        float rescale = __expf(old_max - max_score);
        float new_exp = __expf(score - max_score);
        sum_exp = sum_exp * rescale + new_exp;

        /* Update accumulator: rescale old values and add new contribution */
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc[d] = acc[d] * rescale + new_exp * bf16_to_float(v_vec[d]);
        }
        __syncthreads();
    }

    /* Write output = acc / sum_exp */
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        o_vec[d] = float_to_bf16(acc[d] * inv_sum);
    }
}

extern "C" void krasis_gqa_prefill(
    void* out, const void* q, const void* k, const void* v,
    void* kv_cache, const void* page_table,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int page_size, int num_existing_tokens, float softmax_scale,
    int kv_dtype, void* stream)
{
    if (M == 0) return;
    /* This initial version does NOT use paged KV cache — it operates directly
     * on the Q, K, V tensors from the current prefill batch.
     * KV cache append is done separately via krasis_kv_cache_append. */
    (void)kv_cache;
    (void)page_table;
    (void)page_size;
    (void)kv_dtype;

    dim3 grid(M, num_q_heads);
    int threads = min(256, head_dim);
    threads = ((threads + 31) / 32) * 32;
    int smem = head_dim * sizeof(float);
    gqa_prefill_kernel<<<grid, threads, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        M, num_q_heads, num_kv_heads, head_dim,
        softmax_scale, num_existing_tokens);
}

/* ── KV Cache Append ───────────────────────────────────────────────────── */

/* Append M tokens of K and V to FP8 E4M3 KV caches.
 * k_cache, v_cache: separate [max_seq, kv_stride] FP8 E4M3 buffers
 * k, v: [M, kv_stride] BF16 input from GEMM projections
 * kv_stride = num_kv_heads * head_dim
 * Converts BF16 -> FP8 E4M3 on write, matching decode's cache format.
 */
extern "C" __global__ void kv_cache_append_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ k_cache,   /* [max_seq, kv_stride] */
    __nv_fp8_e4m3* __restrict__ v_cache,   /* [max_seq, kv_stride] */
    const __nv_bfloat16* __restrict__ k,   /* [M, kv_stride] */
    const __nv_bfloat16* __restrict__ v,   /* [M, kv_stride] */
    int M,
    int kv_stride,   /* num_kv_heads * head_dim */
    int max_seq,
    int start_pos)
{
    int ti = blockIdx.x;  /* token index 0..M-1 */
    int pos = start_pos + ti;
    if (pos >= max_seq) return;

    int64_t src_off = (int64_t)ti * kv_stride;
    int64_t dst_off = (int64_t)pos * kv_stride;

    for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
        k_cache[dst_off + d] = bf16_to_fp8e4m3(k[src_off + d]);
        v_cache[dst_off + d] = bf16_to_fp8e4m3(v[src_off + d]);
    }
}

/* PTX entry point — called from Rust via cuLaunchKernel.
 * Same signature as kv_cache_append_fp8_kernel, launched with grid=(M,1,1). */
extern "C" __global__ void kv_cache_append_kernel(
    __nv_fp8_e4m3* __restrict__ k_cache,
    __nv_fp8_e4m3* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos)
{
    int ti = blockIdx.x;
    int pos = start_pos + ti;
    if (pos >= max_seq) return;

    int64_t src_off = (int64_t)ti * kv_stride;
    int64_t dst_off = (int64_t)pos * kv_stride;

    for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
        k_cache[dst_off + d] = bf16_to_fp8e4m3(k[src_off + d]);
        v_cache[dst_off + d] = bf16_to_fp8e4m3(v[src_off + d]);
    }
}

extern "C" __global__ void kv_cache_append_bf16_kernel(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos)
{
    int ti = blockIdx.x;
    int pos = start_pos + ti;
    if (pos >= max_seq) return;

    int64_t src_off = (int64_t)ti * kv_stride;
    int64_t dst_off = (int64_t)pos * kv_stride;

    for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
        k_cache[dst_off + d] = k[src_off + d];
        v_cache[dst_off + d] = v[src_off + d];
    }
}

/* ── FP8 KV Cache Dequant + Concat for Cross-Chunk FA2 ─────────────────
 * Dequantizes FP8 E4M3 KV cache [0..cache_len] to BF16, then copies
 * current chunk BF16 K/V [0..m] into [cache_len..cache_len+m].
 * Result: contiguous BF16 [cache_len+m, kv_stride] buffer for FA2.
 * Grid: (cache_len + m, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void kv_cache_dequant_concat_kernel(
    __nv_bfloat16* __restrict__ out,            /* [cache_len+m, kv_stride] BF16 output */
    const __nv_fp8_e4m3* __restrict__ kv_cache, /* [max_seq, kv_stride] FP8 cache */
    const __nv_bfloat16* __restrict__ kv_new,   /* [m, kv_stride] BF16 current chunk */
    int cache_len,                               /* number of cached tokens */
    int m,                                       /* current chunk size */
    int kv_stride)                               /* num_kv_heads * head_dim */
{
    int ti = blockIdx.x;
    if (ti < cache_len) {
        /* Dequant FP8 -> BF16 from cache */
        int64_t off = (int64_t)ti * kv_stride;
        for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
            float val = float(kv_cache[off + d]);
            out[off + d] = __float2bfloat16(val);
        }
    } else {
        /* Copy BF16 from current chunk */
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_off = (int64_t)ci * kv_stride;
            int64_t dst_off = (int64_t)ti * kv_stride;
            for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
                out[dst_off + d] = kv_new[src_off + d];
            }
        }
    }
}

extern "C" __global__ void kv_cache_concat_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ kv_cache,
    const __nv_bfloat16* __restrict__ kv_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    if (ti < cache_len) {
        int64_t off = (int64_t)ti * kv_stride;
        for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
            out[off + d] = kv_cache[off + d];
        }
    } else {
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_off = (int64_t)ci * kv_stride;
            int64_t dst_off = (int64_t)ti * kv_stride;
            for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
                out[dst_off + d] = kv_new[src_off + d];
            }
        }
    }
}

/* Bounded FP8 KV window staging for ring-window sliding prefill.
 * Copies only the chronological cache tail [cache_start, cache_start+cache_len)
 * followed by the current chunk, producing [cache_len+m, kv_stride] BF16 for FA2.
 */
extern "C" __global__ void kv_cache_dequant_window_concat_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ kv_cache,
    const __nv_bfloat16* __restrict__ kv_new,
    int cache_start,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    if (ti < cache_len) {
        int64_t src_off = (int64_t)(cache_start + ti) * kv_stride;
        int64_t dst_off = (int64_t)ti * kv_stride;
        for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
            out[dst_off + d] = __float2bfloat16(float(kv_cache[src_off + d]));
        }
    } else {
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_off = (int64_t)ci * kv_stride;
            int64_t dst_off = (int64_t)ti * kv_stride;
            for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
                out[dst_off + d] = kv_new[src_off + d];
            }
        }
    }
}

/* ── Mamba2 Strided Extraction ──────────────────────────────────────────
 * Extract x, B, C, dt from in_proj output [M, proj_dim] BF16 into separate
 * contiguous buffers. Replaces M per-token memcpy loops.
 *
 * in_proj layout per row: [z(d_inner) | x(d_inner) | B(bc) | C(bc) | dt(n_heads)]
 * where bc = n_groups * d_state, proj_dim = 2*d_inner + 2*bc + n_heads
 *
 * Grid: (M, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void mamba2_extract_kernel(
    __nv_bfloat16* __restrict__ x_out,    /* [M, d_inner] */
    __nv_bfloat16* __restrict__ b_out,    /* [M, bc] */
    __nv_bfloat16* __restrict__ c_out,    /* [M, bc] */
    __nv_bfloat16* __restrict__ dt_out,   /* [M, n_heads] */
    const __nv_bfloat16* __restrict__ inp, /* [M, proj_dim] */
    int d_inner, int bc, int n_heads, int proj_dim)
{
    int t = blockIdx.x;
    const __nv_bfloat16* row = inp + (int64_t)t * proj_dim;

    /* x: offset d_inner, length d_inner */
    for (int d = threadIdx.x; d < d_inner; d += blockDim.x) {
        x_out[(int64_t)t * d_inner + d] = row[d_inner + d];
    }
    /* B: offset 2*d_inner, length bc */
    for (int d = threadIdx.x; d < bc; d += blockDim.x) {
        b_out[(int64_t)t * bc + d] = row[2 * d_inner + d];
    }
    /* C: offset 2*d_inner+bc, length bc */
    for (int d = threadIdx.x; d < bc; d += blockDim.x) {
        c_out[(int64_t)t * bc + d] = row[2 * d_inner + bc + d];
    }
    /* dt: offset 2*d_inner+2*bc, length n_heads */
    for (int d = threadIdx.x; d < n_heads; d += blockDim.x) {
        dt_out[(int64_t)t * n_heads + d] = row[2 * d_inner + 2 * bc + d];
    }
}

/* Causal conv1d + SiLU over the row-major Mamba2 xBC segment.
 *
 * in_proj row layout: [z(d_inner) | x(d_inner) | B(bc) | C(bc) | dt(n_heads)]
 * conv_dim = d_inner + 2*bc. The convolved x/B/C outputs stay row-major and
 * are split into the existing x_out/b_out/c_out buffers for SSD.
 */
extern "C" __global__ void mamba2_xbc_conv1d_silu_split_kernel(
    __nv_bfloat16* __restrict__ x_out,       /* [M, d_inner] */
    __nv_bfloat16* __restrict__ b_out,       /* [M, bc] */
    __nv_bfloat16* __restrict__ c_out,       /* [M, bc] */
    const float* __restrict__ conv_state,    /* [conv_dim, conv_kernel] FP32 or NULL */
    const __nv_bfloat16* __restrict__ inp,   /* [M, proj_dim] */
    const float* __restrict__ weight,        /* [conv_dim, conv_kernel] FP32 */
    const float* __restrict__ bias,          /* [conv_dim] FP32 or NULL */
    int M,
    int d_inner,
    int bc,
    int proj_dim,
    int conv_dim,
    int conv_kernel)
{
    int token = blockIdx.x;
    if (token >= M) return;

    for (int ch = threadIdx.x; ch < conv_dim; ch += blockDim.x) {
        const float* wt = weight + (int64_t)ch * conv_kernel;
        float acc = (bias != NULL) ? bias[ch] : 0.0f;
        for (int k = 0; k < conv_kernel; k++) {
            int src_pos = token + k - (conv_kernel - 1);
            float val = 0.0f;
            if (src_pos >= 0) {
                val = bf16_to_float(inp[(int64_t)src_pos * proj_dim + d_inner + ch]);
            } else if (conv_state != NULL) {
                int state_idx = conv_kernel + src_pos;
                if (state_idx >= 0 && state_idx < conv_kernel) {
                    val = conv_state[(int64_t)ch * conv_kernel + state_idx];
                }
            }
            acc += val * wt[k];
        }
        float silu = acc / (1.0f + __expf(-acc));
        if (ch < d_inner) {
            x_out[(int64_t)token * d_inner + ch] = float_to_bf16(silu);
        } else if (ch < d_inner + bc) {
            b_out[(int64_t)token * bc + (ch - d_inner)] = float_to_bf16(silu);
        } else {
            c_out[(int64_t)token * bc + (ch - d_inner - bc)] = float_to_bf16(silu);
        }
    }
}

extern "C" __global__ void mamba2_xbc_update_conv_state_kernel(
    float* __restrict__ conv_state,          /* [conv_dim, conv_kernel] FP32 */
    const __nv_bfloat16* __restrict__ inp,   /* [M, proj_dim] */
    int M,
    int d_inner,
    int proj_dim,
    int conv_dim,
    int conv_kernel)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim || conv_state == NULL || M <= 0) return;

    float* st = conv_state + (int64_t)ch * conv_kernel;
    for (int k = 0; k < conv_kernel; k++) {
        int src_pos = M - conv_kernel + k;
        float val = 0.0f;
        if (src_pos >= 0) {
            val = bf16_to_float(inp[(int64_t)src_pos * proj_dim + d_inner + ch]);
        } else {
            int state_idx = conv_kernel + src_pos;
            if (state_idx >= 0 && state_idx < conv_kernel) {
                val = st[state_idx];
            }
        }
        st[k] = val;
    }
}

/* Reference-compatible Mamba2 gated group RMSNorm.
 *
 * Python reference computes:
 *   gated = scan_output * silu(gate)
 *   out = norm_weight * group_rmsnorm(gated)
 *
 * group_size = d_inner / n_groups. Weight is full d_inner FP32, not a
 * per-head vector.
 */
extern "C" __global__ void mamba2_gated_group_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,          /* [M, d_inner] BF16 */
    const __nv_bfloat16* __restrict__ x,      /* [M, d_inner] BF16 */
    const __nv_bfloat16* __restrict__ gate,   /* [M, d_inner] BF16 */
    const float* __restrict__ weight,         /* [d_inner] FP32 */
    int n_groups,
    int group_size,
    int proj_dim,
    float eps)
{
    int token = blockIdx.x;
    int group = blockIdx.y;
    int group_base = group * group_size;
    const __nv_bfloat16* x_row = x + (int64_t)token * n_groups * group_size;
    const __nv_bfloat16* gate_row = gate + (int64_t)token * proj_dim;
    __nv_bfloat16* out_row = out + (int64_t)token * n_groups * group_size;

    extern __shared__ float smem[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_base + i;
        float xv = bf16_to_float(x_row[idx]);
        float gv = bf16_to_float(gate_row[idx]);
        float silu_g = gv / (1.0f + __expf(-gv));
        float gated = xv * silu_g;
        local_ss += gated * gated;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    gated_rmsnorm_adjacent_pairwise_reduce(smem);

    if (threadIdx.x == 0) {
        float mean_square_plus_eps = smem[0] / (float)group_size + eps;
        smem[0] = mamba2_gated_rmsnorm_triton_rstd(mean_square_plus_eps);
    }
    __syncthreads();

    float rms_inv = smem[0];
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_base + i;
        float xv = bf16_to_float(x_row[idx]);
        float gv = bf16_to_float(gate_row[idx]);
        float silu_g = gv / (1.0f + __expf(-gv));
        float gated = xv * silu_g;
        out_row[idx] = float_to_bf16(gated * rms_inv * weight[idx]);
    }
}

extern "C" __global__ void mamba2_gated_group_rmsnorm_sqrt_approx_div_rn_replay_kernel(
    __nv_bfloat16* __restrict__ out,          /* [M, d_inner] BF16 */
    const __nv_bfloat16* __restrict__ x,      /* [M, d_inner] BF16 */
    const __nv_bfloat16* __restrict__ gate,   /* [M, d_inner] BF16 */
    const float* __restrict__ weight,         /* [d_inner] FP32 */
    int row_index,
    int n_groups,
    int group_size,
    int proj_dim,
    float eps)
{
    if (row_index < 0 || n_groups <= 0 || group_size <= 0 || proj_dim <= 0) return;
    int group = blockIdx.x;
    if (group >= n_groups) return;
    int group_base = group * group_size;
    const __nv_bfloat16* x_row = x + (int64_t)row_index * n_groups * group_size;
    const __nv_bfloat16* gate_row = gate + (int64_t)row_index * proj_dim;
    __nv_bfloat16* out_row = out + (int64_t)row_index * n_groups * group_size;

    extern __shared__ float smem[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_base + i;
        float xv = bf16_to_float(x_row[idx]);
        float gv = bf16_to_float(gate_row[idx]);
        float silu_g = gv / (1.0f + __expf(-gv));
        float gated = xv * silu_g;
        local_ss += gated * gated;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    gated_rmsnorm_adjacent_pairwise_reduce(smem);

    if (threadIdx.x == 0) {
        float mean_square_plus_eps = smem[0] / (float)group_size + eps;
        smem[0] = mamba2_gated_rmsnorm_triton_rstd(mean_square_plus_eps);
    }
    __syncthreads();

    float rms_inv = smem[0];
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_base + i;
        float xv = bf16_to_float(x_row[idx]);
        float gv = bf16_to_float(gate_row[idx]);
        float silu_g = gv / (1.0f + __expf(-gv));
        float gated = xv * silu_g;
        out_row[idx] = float_to_bf16(gated * rms_inv * weight[idx]);
    }
}

/* ── Mamba2 Causal Conv1d (prefill) ────────────────────────────────────── */

/* Simple causal conv1d for Mamba2 prefill.
 * x: [1, D, L] bf16 (batch=1 for inference)
 * weight: [D, width] bf16
 * bias: [D] bf16 or NULL
 * out: [1, D, L] bf16
 * conv_state: [1, D, width-1] bf16 (updated with final state)
 *
 * Each thread block handles one channel dimension.
 */
extern "C" __global__ void causal_conv1d_fwd_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ conv_state,
    int D,
    int L,
    int width,
    int silu_act)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    /* Load weight for this channel */
    float w[8];  /* width <= 8 for Mamba2 (typically 4) */
    for (int j = 0; j < width; j++) {
        w[j] = bf16_to_float(weight[d * width + j]);
    }
    float b = (bias != NULL) ? bf16_to_float(bias[d]) : 0.0f;

    /* Process sequence positions */
    const __nv_bfloat16* x_d = x + (int64_t)d * L;  /* [L] for this channel */
    __nv_bfloat16* out_d = out + (int64_t)d * L;

    for (int t = 0; t < L; t++) {
        float acc = b;
        for (int j = 0; j < width; j++) {
            int src_t = t - (width - 1) + j;
            float xv = 0.0f;
            if (src_t >= 0) {
                xv = bf16_to_float(x_d[src_t]);
            }
            /* else: zero padding (causal, no initial state for prefill) */
            acc += xv * w[j];
        }
        if (silu_act) {
            acc = acc / (1.0f + __expf(-acc));
        }
        out_d[t] = float_to_bf16(acc);
    }

    /* Save final conv state: last (width-1) values of x for this channel */
    __nv_bfloat16* cs = conv_state + (int64_t)d * (width - 1);
    for (int j = 0; j < width - 1; j++) {
        int src_t = L - (width - 1) + j;
        cs[j] = (src_t >= 0) ? x_d[src_t] : float_to_bf16(0.0f);
    }
}

extern "C" void krasis_causal_conv1d_fwd(
    void* out, const void* x, const void* weight, const void* bias,
    void* conv_state,
    int B, int D, int L, int width, int silu_activation,
    void* stream)
{
    if (D == 0 || L == 0) return;
    /* B is always 1 for inference. Process all channels in parallel. */
    int threads = 256;
    int blocks = (D + threads - 1) / threads;
    causal_conv1d_fwd_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)conv_state,
        D, L, width, silu_activation);
}

/* ── Mamba2 SSD Forward ────────────────────────────────────────────────── */

/* Mamba2 SSD (Structured State-Space Duality) chunked scan.
 *
 * This implements the chunked SSD algorithm for prefill:
 * 1. Discretize: dt → A_bar = exp(A * softplus(dt + bias)),
 *    B_bar = bf16(B * softplus(dt + bias))
 * 2. Within each chunk of size C:
 *    - Compute chunk-level SSM: parallel O(C²) attention-like computation
 * 3. Between chunks: sequential state passing
 *
 * For initial correctness, we implement a simple sequential scan
 * (equivalent to running decode N times). Chunk-parallel version comes later.
 */
extern "C" __global__ void mamba2_ssd_sequential_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus)
{
    /* One block per (head, head_dim_idx) pair */
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;

    int group = head / (n_heads / n_groups);
    float A_val = mamba2_ssd_a_value(A_log[head]);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;

    /* SSM state for this (head, d): [state_size] floats */
    float* h = valid_d ? (ssm_state + ((int64_t)head * head_dim + d) * state_size) : NULL;

    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    float dA_chunk_base = 0.0f;
    extern __shared__ float mamba2_ssd_da_prefix_smem[];

    /* Sequential scan over time steps */
    for (int t = 0; t < L; t++) {
        int chunk_start = (t / effective_chunk_size) * effective_chunk_size;
        int chunk_pos = t - chunk_start;
        int chunk_len = min(effective_chunk_size, L - chunk_start);
        float* dA_prefix = NULL;
        if (chunk_pos == 0) {
            dA_prefix = mamba2_ssd_build_da_prefix_scan(
                mamba2_ssd_da_prefix_smem,
                effective_chunk_size,
                dt_in,
                dt_bias,
                A_val,
                n_heads,
                head,
                chunk_start,
                chunk_len,
                use_softplus);
        } else {
            dA_prefix = mamba2_ssd_da_prefix_smem;
            int rounds = 0;
            for (int offset = 1; offset < chunk_len; offset <<= 1) rounds++;
            if ((rounds & 1) != 0) dA_prefix += effective_chunk_size;
        }

        float dt = mamba2_ssd_dt_value(dt_in, dt_bias, t, n_heads, head, use_softplus);

        /* Discretize: A_bar = exp(A * dt) */
        float A_bar = __expf(A_val * dt);
        float dA_target = dA_chunk_base + dA_prefix[chunk_pos];

        /* Get x for this timestep */
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;

        /* Update final SSM state. */
        if (valid_d) {
            for (int s = 0; s < state_size; s++) {
                float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);

                /* HF chunk-state casts scaled B back to x dtype before FP32 accumulation. */
                float B_bar = bf16_to_float(float_to_bf16(B_val * dt));
                h[s] = A_bar * h[s] + B_bar * x_val;
            }
        }

        /*
         * HF chunk-scan emission is separate from chunk-state accumulation:
         * y[t] = D*x[t] + dot(BF16((C[t] @ B[u]) * decay(t,u) * dt[u]), x[u]).
         * Use the recurrent state for prior chunks, then replace only the
         * current chunk's local C*state contribution with HF's CB emission.
         */
        float c_state_total = 0.0f;
        if (valid_d) {
            for (int s = 0; s < state_size; s++) {
                float C_val = bf16_to_float(C_mat[(t * n_groups + group) * state_size + s]);
                c_state_total += C_val * h[s];
            }
        }

        float local_old_state = 0.0f;
        float local_hf_chunk_scan = 0.0f;
        if (valid_d) {
            for (int u = chunk_start; u <= t; u++) {
                int u_chunk_pos = u - chunk_start;
                float dA_running = dA_chunk_base + dA_prefix[u_chunk_pos];
                float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
                if (dt_bias != NULL) dt_u += dt_bias[head];
                float dt_u_scale = dt_u;
                if (use_softplus) {
                    dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                    dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                }
                float decay = (u == t) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));

                float old_state_source = 0.0f;
                int c_row_base_idx = (t * n_groups + group) * state_size;
                int b_row_base_idx = (u * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                if (chunk_start != 0) {
                    for (int s = 0; s < state_size; s++) {
                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                        float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
                        float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
                        old_state_source += C_val * B_bar;
                    }
                }

                float scale = decay * dt_u_scale;
                float cb_scaled_bf16 = bf16_to_float(float_to_bf16(cb * scale));
                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                local_old_state += old_state_source * decay * x_u;
                local_hf_chunk_scan += cb_scaled_bf16 * x_u;
            }
        }

        if (valid_d) {
            float prior_chunk_state = (chunk_start == 0) ? 0.0f : (c_state_total - local_old_state);
            float y = D_val * x_val + prior_chunk_state + local_hf_chunk_scan;
            int flat_idx = (t * n_heads + head) * head_dim + d;
            out[flat_idx] = float_to_bf16(y);
        }
        if (chunk_pos + 1 == chunk_len) {
            dA_chunk_base += dA_prefix[chunk_len - 1];
        }
        __syncthreads();
    }
}

extern "C" __global__ void mamba2_ssd_sequential_kernel_trace(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus,
    unsigned long long* __restrict__ trace,
    int trace_entry_idx,
    int trace_stage_id,
    int trace_component_stage_id,
    int trace_same_kernel_summary_stage_id,
    int trace_same_kernel_context_stage_id,
    int trace_same_kernel_final_sample_operand_stage_id,
    int trace_same_kernel_final_sample_index_stage_id,
    int trace_same_kernel_final_sample_decay_stage_id,
    int trace_target_row,
    int trace_target_dim,
    int trace_layer_idx,
    int trace_chunk_idx,
    int trace_start_abs_pos)
{
    /* One block per (head, head_dim_idx) pair */
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;
    int dim_index = head * head_dim + d;

    int group = head / (n_heads / n_groups);
    float A_val = mamba2_ssd_a_value(A_log[head]);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    int d_inner = n_heads * head_dim;
    unsigned short* out_bits = reinterpret_cast<unsigned short*>(out);

    /* SSM state for this (head, d): [state_size] floats */
    float* h = valid_d ? (ssm_state + ((int64_t)head * head_dim + d) * state_size) : NULL;

    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    float dA_chunk_base = 0.0f;
    extern __shared__ float mamba2_ssd_da_prefix_smem[];

    /* Sequential scan over time steps */
    for (int t = 0; t < L; t++) {
        int chunk_start = (t / effective_chunk_size) * effective_chunk_size;
        int chunk_pos = t - chunk_start;
        int chunk_len = min(effective_chunk_size, L - chunk_start);
        float* dA_prefix = NULL;
        if (chunk_pos == 0) {
            dA_prefix = mamba2_ssd_build_da_prefix_scan(
                mamba2_ssd_da_prefix_smem,
                effective_chunk_size,
                dt_in,
                dt_bias,
                A_val,
                n_heads,
                head,
                chunk_start,
                chunk_len,
                use_softplus);
        } else {
            dA_prefix = mamba2_ssd_da_prefix_smem;
            int rounds = 0;
            for (int offset = 1; offset < chunk_len; offset <<= 1) rounds++;
            if ((rounds & 1) != 0) dA_prefix += effective_chunk_size;
        }

        float dt = mamba2_ssd_dt_value(dt_in, dt_bias, t, n_heads, head, use_softplus);

        /* Discretize: A_bar = exp(A * dt) */
        float A_bar = __expf(A_val * dt);
        float dA_target = dA_chunk_base + dA_prefix[chunk_pos];

        /* Get x for this timestep */
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;

        /* Update final SSM state. */
        if (valid_d) {
            for (int s = 0; s < state_size; s++) {
                float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);

                /* HF chunk-state casts scaled B back to x dtype before FP32 accumulation. */
                float B_bar = bf16_to_float(float_to_bf16(B_val * dt));
                h[s] = A_bar * h[s] + B_bar * x_val;
            }
        }

        /*
         * HF chunk-scan emission is separate from chunk-state accumulation:
         * y[t] = D*x[t] + dot(BF16((C[t] @ B[u]) * decay(t,u) * dt[u]), x[u]).
         * Use the recurrent state for prior chunks, then replace only the
         * current chunk's local C*state contribution with HF's CB emission.
         */
        float c_state_total = 0.0f;
        if (valid_d) {
            for (int s = 0; s < state_size; s++) {
                float C_val = bf16_to_float(C_mat[(t * n_groups + group) * state_size + s]);
                c_state_total += C_val * h[s];
            }
        }

        float local_old_state = 0.0f;
        float local_hf_chunk_scan = 0.0f;
        if (valid_d) {
            for (int u = chunk_start; u <= t; u++) {
                int u_chunk_pos = u - chunk_start;
                float dA_running = dA_chunk_base + dA_prefix[u_chunk_pos];
                float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
                if (dt_bias != NULL) dt_u += dt_bias[head];
                float dt_u_scale = dt_u;
                if (use_softplus) {
                    dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                    dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                }
                float decay = (u == t) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));

                float old_state_source = 0.0f;
                int c_row_base_idx = (t * n_groups + group) * state_size;
                int b_row_base_idx = (u * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                if (chunk_start != 0) {
                    for (int s = 0; s < state_size; s++) {
                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                        float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
                        float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
                        old_state_source += C_val * B_bar;
                    }
                }

                float scale = decay * dt_u_scale;
                float cb_scaled_bf16 = bf16_to_float(float_to_bf16(cb * scale));
                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                local_old_state += old_state_source * decay * x_u;
                local_hf_chunk_scan += cb_scaled_bf16 * x_u;
            }
        }

        float prior_chunk_state = valid_d
            ? ((chunk_start == 0) ? 0.0f : (c_state_total - local_old_state))
            : 0.0f;
        float y = valid_d ? (D_val * x_val + prior_chunk_state + local_hf_chunk_scan) : 0.0f;
        int flat_idx = (t * n_heads + head) * head_dim + d;
        __nv_bfloat16 y_bf16 = float_to_bf16(y);
        bool capture_store = trace != NULL && trace_entry_idx >= 0 && trace_stage_id != 0 &&
            valid_d && t == trace_target_row && dim_index == trace_target_dim;
        unsigned int pre_existing_raw = capture_store ? (unsigned int)out_bits[flat_idx] : 0U;
        unsigned int candidate_raw = capture_store
            ? (unsigned int)(*reinterpret_cast<unsigned short*>(&y_bf16))
            : 0U;
        float duplicate_local_scan = 0.0f;
        float duplicate_y = y;
        unsigned long long input_checksum = 1469598103934665603ULL;
        unsigned long long term_checksum = 1469598103934665603ULL;
        unsigned int first_term_bits = 0U;
        unsigned int mid_term_bits = 0U;
        unsigned int last_term_bits = 0U;
        unsigned int duplicate_count = 0U;
        unsigned int final_x_bits = 0U;
        unsigned int final_dt_bits = 0U;
        unsigned int final_cb_scaled_bits = 0U;
        int final_x_flat_idx = -1;
        int final_dt_idx = -1;
        int final_b_row_base_idx = -1;
        int final_c_row_base_idx = -1;
        float final_dt_softplus = 0.0f;
        float final_scale = 0.0f;
        float final_cb = 0.0f;
        float final_cb_scaled_pre_bf16 = 0.0f;
        float final_dA_target = 0.0f;
        float final_dA_chunk_base = 0.0f;
        float final_dA_running_before = 0.0f;
        float final_dA_increment = 0.0f;
        float final_dA_running_after = 0.0f;
        float final_decay_arg = 0.0f;
        float final_decay = 0.0f;
        if (capture_store &&
            (trace_same_kernel_summary_stage_id != 0 ||
             trace_same_kernel_context_stage_id != 0 ||
             trace_same_kernel_final_sample_operand_stage_id != 0 ||
             trace_same_kernel_final_sample_index_stage_id != 0 ||
             trace_same_kernel_final_sample_decay_stage_id != 0)) {
            const unsigned short* x_bits = reinterpret_cast<const unsigned short*>(x);
            const unsigned short* dt_bits = reinterpret_cast<const unsigned short*>(dt_in);
            int mid_u = chunk_start + (t - chunk_start) / 2;
            for (int u = chunk_start; u <= t; u++) {
                float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
                if (dt_bias != NULL) dt_u += dt_bias[head];
                float dt_u_scale = dt_u;
                if (use_softplus) {
                    dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                    dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                }
                int u_chunk_pos = u - chunk_start;
                float dA_running_after = dA_chunk_base + dA_prefix[u_chunk_pos];
                float dA_running_before = (u_chunk_pos == 0)
                    ? dA_chunk_base
                    : (dA_chunk_base + dA_prefix[u_chunk_pos - 1]);
                float dA_increment = A_val * dt_u;
                float decay_arg = (u == t) ? 0.0f : fminf(dA_target - dA_running_after, 0.0f);
                float decay = (u == t) ? 1.0f : __expf(decay_arg);

                int c_row_base_idx = (t * n_groups + group) * state_size;
                int b_row_base_idx = (u * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);

                float scale = decay * dt_u_scale;
                __nv_bfloat16 cb_scaled_raw = float_to_bf16(cb * scale);
                unsigned int cb_scaled_bits =
                    (unsigned int)(*reinterpret_cast<unsigned short*>(&cb_scaled_raw));
                float cb_scaled_bf16 = bf16_to_float(cb_scaled_raw);
                int x_flat_idx = (u * n_heads + head) * head_dim + d;
                int dt_idx = u * n_heads + head;
                float x_u = bf16_to_float(x[x_flat_idx]);
                float term = cb_scaled_bf16 * x_u;
                duplicate_local_scan += term;

                if (u == t) {
                    final_x_bits = (unsigned int)x_bits[x_flat_idx];
                    final_dt_bits = (unsigned int)dt_bits[dt_idx];
                    final_cb_scaled_bits = cb_scaled_bits;
                    final_x_flat_idx = x_flat_idx;
                    final_dt_idx = dt_idx;
                    final_b_row_base_idx = b_row_base_idx;
                    final_c_row_base_idx = c_row_base_idx;
                    final_dt_softplus = dt_u_scale;
                    final_scale = scale;
                    final_cb = cb;
                    final_cb_scaled_pre_bf16 = cb * scale;
                    final_dA_target = dA_target;
                    final_dA_chunk_base = dA_chunk_base;
                    final_dA_running_before = dA_running_before;
                    final_dA_increment = dA_increment;
                    final_dA_running_after = dA_running_after;
                    final_decay_arg = decay_arg;
                    final_decay = decay;
                }

                unsigned int term_bits = (unsigned int)trace_f32_bits(term);
                if (u == chunk_start) first_term_bits = term_bits;
                if (u == mid_u) mid_term_bits = term_bits;
                if (u == t) last_term_bits = term_bits;
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)u);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)x_flat_idx);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)dt_idx);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)b_row_base_idx);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)c_row_base_idx);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)x_bits[x_flat_idx]);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)dt_bits[dt_idx]);
                input_checksum = trace_mix_u64(input_checksum, (unsigned long long)cb_scaled_bits);
                term_checksum = trace_mix_u64(term_checksum, (unsigned long long)term_bits);
                duplicate_count++;
            }
            duplicate_y = D_val * x_val + prior_chunk_state + duplicate_local_scan;
        }
        if (valid_d) {
            out[flat_idx] = y_bf16;
        }
        if (chunk_pos + 1 == chunk_len) {
            dA_chunk_base += dA_prefix[chunk_len - 1];
        }
        if (capture_store) {
            unsigned int post_store_raw = (unsigned int)out_bits[flat_idx];
            int base = trace_entry_idx * 16;
            trace[base + 0] = (unsigned long long)trace_stage_id;
            trace[base + 1] = (unsigned long long)trace_layer_idx;
            trace[base + 2] = (unsigned long long)trace_chunk_idx;
            trace[base + 3] = (unsigned long long)(trace_start_abs_pos + t);
            trace[base + 4] = (unsigned long long)t;
            trace[base + 5] = (unsigned long long)dim_index;
            trace[base + 6] = (unsigned long long)flat_idx;
            trace[base + 7] = (unsigned long long)pre_existing_raw;
            trace[base + 8] = trace_f32_bits(y);
            trace[base + 9] = (unsigned long long)candidate_raw;
            trace[base + 10] = (unsigned long long)post_store_raw;
            trace[base + 11] = (unsigned long long)(out + flat_idx);
            trace[base + 12] = (unsigned long long)head;
            trace[base + 13] = (unsigned long long)d;
            trace[base + 14] = (unsigned long long)d_inner;
            trace[base + 15] = 1ULL;
            if (trace_component_stage_id != 0) {
                int base2 = (trace_entry_idx + 1) * 16;
                float component_sum = D_val * x_val + prior_chunk_state + local_hf_chunk_scan;
                trace[base2 + 0] = (unsigned long long)trace_component_stage_id;
                trace[base2 + 1] = (unsigned long long)trace_layer_idx;
                trace[base2 + 2] = (unsigned long long)trace_chunk_idx;
                trace[base2 + 3] = (unsigned long long)(trace_start_abs_pos + t);
                trace[base2 + 4] = (unsigned long long)t;
                trace[base2 + 5] = (unsigned long long)dim_index;
                trace[base2 + 6] = (unsigned long long)flat_idx;
                trace[base2 + 7] = (unsigned long long)head;
                trace[base2 + 8] = (unsigned long long)d;
                trace[base2 + 9] = trace_f32_bits(D_val * x_val);
                trace[base2 + 10] = trace_f32_bits(prior_chunk_state);
                trace[base2 + 11] = trace_f32_bits(local_hf_chunk_scan);
                trace[base2 + 12] = trace_f32_bits(c_state_total);
                trace[base2 + 13] = trace_f32_bits(local_old_state);
                trace[base2 + 14] = trace_f32_bits(component_sum);
                trace[base2 + 15] = trace_f32_bits(y);
            }
            if (trace_same_kernel_summary_stage_id != 0) {
                int base3 = (trace_entry_idx + 2) * 16;
                trace[base3 + 0] = (unsigned long long)trace_same_kernel_summary_stage_id;
                trace[base3 + 1] = (unsigned long long)trace_layer_idx;
                trace[base3 + 2] = (unsigned long long)trace_chunk_idx;
                trace[base3 + 3] = (unsigned long long)(trace_start_abs_pos + t);
                trace[base3 + 4] = (unsigned long long)t;
                trace[base3 + 5] = (unsigned long long)dim_index;
                trace[base3 + 6] = (unsigned long long)head;
                trace[base3 + 7] = (unsigned long long)d;
                trace[base3 + 8] = trace_f32_bits(local_hf_chunk_scan);
                trace[base3 + 9] = trace_f32_bits(duplicate_local_scan);
                trace[base3 + 10] = trace_f32_bits(y);
                trace[base3 + 11] = trace_f32_bits(duplicate_y);
                trace[base3 + 12] = term_checksum;
                trace[base3 + 13] = input_checksum;
                trace[base3 + 14] =
                    ((unsigned long long)first_term_bits << 32) | (unsigned long long)mid_term_bits;
                trace[base3 + 15] =
                    ((unsigned long long)last_term_bits << 32) | (unsigned long long)duplicate_count;
            }
            if (trace_same_kernel_context_stage_id != 0) {
                int base4 = (trace_entry_idx + 3) * 16;
                int dt_target_idx = t * n_heads + head;
                int c_row_base_idx = (t * n_groups + group) * state_size;
                int b_target_base_idx = (t * n_groups + group) * state_size;
                trace[base4 + 0] = (unsigned long long)trace_same_kernel_context_stage_id;
                trace[base4 + 1] = (unsigned long long)trace_layer_idx;
                trace[base4 + 2] = (unsigned long long)trace_chunk_idx;
                trace[base4 + 3] = (unsigned long long)(trace_start_abs_pos + t);
                trace[base4 + 4] = (unsigned long long)t;
                trace[base4 + 5] = (unsigned long long)dim_index;
                trace[base4 + 6] = (unsigned long long)x;
                trace[base4 + 7] = (unsigned long long)dt_in;
                trace[base4 + 8] = (unsigned long long)B_mat;
                trace[base4 + 9] = (unsigned long long)C_mat;
                trace[base4 + 10] = (unsigned long long)out;
                trace[base4 + 11] = (unsigned long long)flat_idx;
                trace[base4 + 12] = (unsigned long long)dt_target_idx;
                trace[base4 + 13] = (unsigned long long)c_row_base_idx;
                trace[base4 + 14] = (unsigned long long)b_target_base_idx;
                trace[base4 + 15] =
                    ((unsigned long long)chunk_start << 32) | (unsigned long long)effective_chunk_size;
            }
            if (trace_same_kernel_final_sample_operand_stage_id != 0) {
                int base5 = (trace_entry_idx + 4) * 16;
                trace[base5 + 0] =
                    (unsigned long long)trace_same_kernel_final_sample_operand_stage_id;
                trace[base5 + 1] = (unsigned long long)trace_layer_idx;
                trace[base5 + 2] = (unsigned long long)trace_chunk_idx;
                trace[base5 + 3] = (unsigned long long)(trace_start_abs_pos + t);
                trace[base5 + 4] = (unsigned long long)t;
                trace[base5 + 5] = (unsigned long long)dim_index;
                trace[base5 + 6] = (unsigned long long)t;
                trace[base5 + 7] = (unsigned long long)head;
                trace[base5 + 8] = (unsigned long long)d;
                trace[base5 + 9] = (unsigned long long)final_x_bits;
                trace[base5 + 10] = (unsigned long long)final_dt_bits;
                trace[base5 + 11] = trace_f32_bits(final_dt_softplus);
                trace[base5 + 12] = trace_f32_bits(final_scale);
                trace[base5 + 13] = trace_f32_bits(final_cb);
                trace[base5 + 14] = trace_f32_bits(final_cb_scaled_pre_bf16);
                trace[base5 + 15] = (unsigned long long)final_cb_scaled_bits;
            }
            if (trace_same_kernel_final_sample_index_stage_id != 0) {
                int base6 = (trace_entry_idx + 5) * 16;
                trace[base6 + 0] =
                    (unsigned long long)trace_same_kernel_final_sample_index_stage_id;
                trace[base6 + 1] = (unsigned long long)trace_layer_idx;
                trace[base6 + 2] = (unsigned long long)trace_chunk_idx;
                trace[base6 + 3] = (unsigned long long)(trace_start_abs_pos + t);
                trace[base6 + 4] = (unsigned long long)t;
                trace[base6 + 5] = (unsigned long long)dim_index;
                trace[base6 + 6] = (unsigned long long)t;
                trace[base6 + 7] = (unsigned long long)final_x_flat_idx;
                trace[base6 + 8] = (unsigned long long)final_dt_idx;
                trace[base6 + 9] = (unsigned long long)final_b_row_base_idx;
                trace[base6 + 10] = (unsigned long long)final_c_row_base_idx;
                trace[base6 + 11] = (unsigned long long)x;
                trace[base6 + 12] = (unsigned long long)dt_in;
                trace[base6 + 13] = (unsigned long long)B_mat;
                trace[base6 + 14] = (unsigned long long)C_mat;
                trace[base6 + 15] = 1ULL;
            }
            if (trace_same_kernel_final_sample_decay_stage_id != 0) {
                int base7 = (trace_entry_idx + 6) * 16;
                trace[base7 + 0] =
                    (unsigned long long)trace_same_kernel_final_sample_decay_stage_id;
                trace[base7 + 1] = (unsigned long long)trace_layer_idx;
                trace[base7 + 2] = (unsigned long long)trace_chunk_idx;
                trace[base7 + 3] = (unsigned long long)(trace_start_abs_pos + t);
                trace[base7 + 4] = (unsigned long long)t;
                trace[base7 + 5] = (unsigned long long)dim_index;
                trace[base7 + 6] = (unsigned long long)t;
                trace[base7 + 7] = (unsigned long long)head;
                trace[base7 + 8] = (unsigned long long)d;
                trace[base7 + 9] = trace_f32_bits(final_dA_target);
                trace[base7 + 10] = trace_f32_bits(final_dA_chunk_base);
                trace[base7 + 11] = trace_f32_bits(final_dA_running_before);
                trace[base7 + 12] = trace_f32_bits(final_dA_increment);
                trace[base7 + 13] = trace_f32_bits(final_dA_running_after);
                trace[base7 + 14] = trace_f32_bits(final_decay_arg);
                trace[base7 + 15] = trace_f32_bits(final_decay);
            }
        }
        __syncthreads();
    }
}

/*
 * Opt-in v5 correctness candidate. This intentionally preserves the current
 * sequential per-token FP32 recurrence for state propagation while writing to
 * separate candidate buffers. It exists only behind Rust-side oracle gates.
 */
extern "C" __global__ void mamba2_ssd_v5_recurrent_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;

    int group = head / (n_heads / n_groups);
    float A_val = mamba2_ssd_a_value(A_log[head]);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    float* h = valid_d ? (ssm_state + ((int64_t)head * head_dim + d) * state_size) : NULL;

    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    float dA_chunk_base = 0.0f;
    extern __shared__ float mamba2_ssd_v5_da_prefix_smem[];

    for (int t = 0; t < L; t++) {
        int chunk_start = (t / effective_chunk_size) * effective_chunk_size;
        int chunk_pos = t - chunk_start;
        int chunk_len = min(effective_chunk_size, L - chunk_start);
        float* dA_prefix = NULL;
        if (chunk_pos == 0) {
            dA_prefix = mamba2_ssd_build_da_prefix_scan(
                mamba2_ssd_v5_da_prefix_smem,
                effective_chunk_size,
                dt_in,
                dt_bias,
                A_val,
                n_heads,
                head,
                chunk_start,
                chunk_len,
                use_softplus);
        } else {
            dA_prefix = mamba2_ssd_v5_da_prefix_smem;
            int rounds = 0;
            for (int offset = 1; offset < chunk_len; offset <<= 1) rounds++;
            if ((rounds & 1) != 0) dA_prefix += effective_chunk_size;
        }

        float dt = mamba2_ssd_dt_value(dt_in, dt_bias, t, n_heads, head, use_softplus);
        float A_bar = __expf(A_val * dt);
        float dA_target = dA_chunk_base + dA_prefix[chunk_pos];
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;

        if (valid_d) {
            for (int s = 0; s < state_size; s++) {
                float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);
                float B_bar = bf16_to_float(float_to_bf16(B_val * dt));
                h[s] = A_bar * h[s] + B_bar * x_val;
            }
        }

        float c_state_total = 0.0f;
        if (valid_d) {
            for (int s = 0; s < state_size; s++) {
                float C_val = bf16_to_float(C_mat[(t * n_groups + group) * state_size + s]);
                c_state_total += C_val * h[s];
            }
        }

        float local_old_state = 0.0f;
        float local_hf_chunk_scan = 0.0f;
        if (valid_d) {
            for (int u = chunk_start; u <= t; u++) {
                int u_chunk_pos = u - chunk_start;
                float dA_running = dA_chunk_base + dA_prefix[u_chunk_pos];
                float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
                if (dt_bias != NULL) dt_u += dt_bias[head];
                float dt_u_scale = dt_u;
                if (use_softplus) {
                    dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                    dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                }
                float decay = (u == t) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));

                float old_state_source = 0.0f;
                int c_row_base_idx = (t * n_groups + group) * state_size;
                int b_row_base_idx = (u * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                if (chunk_start != 0) {
                    for (int s = 0; s < state_size; s++) {
                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                        float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
                        float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
                        old_state_source += C_val * B_bar;
                    }
                }

                float scale = decay * dt_u_scale;
                float cb_scaled_bf16 = bf16_to_float(float_to_bf16(cb * scale));
                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                local_old_state += old_state_source * decay * x_u;
                local_hf_chunk_scan += cb_scaled_bf16 * x_u;
            }
        }

        if (valid_d) {
            float prior_chunk_state = (chunk_start == 0) ? 0.0f : (c_state_total - local_old_state);
            float y = D_val * x_val + prior_chunk_state + local_hf_chunk_scan;
            int flat_idx = (t * n_heads + head) * head_dim + d;
            out[flat_idx] = float_to_bf16(y);
        }
        if (chunk_pos + 1 == chunk_len) {
            dA_chunk_base += dA_prefix[chunk_len - 1];
        }
        __syncthreads();
    }
}

/*
 * Opt-in block-scan correctness seed, phase 1. This preserves the exact
 * token-order FP32 recurrence as the accepted source for chunk entries and
 * final state, while writing per-chunk entry snapshots for a separate output
 * assembly phase.
 */
extern "C" __global__ void mamba2_ssd_block_scan_recurrent_kernel(
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
	    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
	    float* __restrict__ entry_state,        /* [n_chunks, n_heads, head_dim, state_size] */
	    float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
	    unsigned long long* __restrict__ recurrent_subloop_timing,/* optional [11], NULL when disabled */
	    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
	    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;
    if (!valid_d) return;

    int group = head / (n_heads / n_groups);
	    float A_val = mamba2_ssd_a_value(A_log[head]);
			    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
			    int state_base = ((int64_t)head * head_dim + d) * state_size;
			    float* h = ssm_state + state_base;
	    unsigned long long thread_total_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
	    unsigned long long entry_snapshot_cycles = 0ULL;
	    unsigned long long dt_a_x_setup_cycles = 0ULL;
	    unsigned long long state_update_cycles = 0ULL;
	    unsigned long long c_dot_cycles = 0ULL;
	    unsigned long long c_state_store_cycles = 0ULL;
	    unsigned long long row_lanes = 0ULL;
	    unsigned long long entry_snapshot_elements = 0ULL;
	    unsigned long long state_update_elements = 0ULL;
	    unsigned long long c_dot_elements = 0ULL;
	    unsigned long long c_state_stores = 0ULL;

		    for (int t = 0; t < L; t++) {
			        int chunk_idx = t / effective_chunk_size;
		        int chunk_pos = t - chunk_idx * effective_chunk_size;
		        if (chunk_pos == 0) {
	            unsigned long long entry_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
			            int64_t entry_base = ((int64_t)chunk_idx * n_heads * head_dim + (int64_t)head * head_dim + d) * state_size;
			            for (int s = 0; s < state_size; s++) {
			                entry_state[entry_base + s] = h[s];
			            }
	            if (recurrent_subloop_timing != NULL) {
	                entry_snapshot_cycles += clock64() - entry_t0;
	                entry_snapshot_elements += (unsigned long long)state_size;
	            }
			        }

	        unsigned long long setup_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
		        float dt = mamba2_ssd_dt_value(dt_in, dt_bias, t, n_heads, head, use_softplus);
		        float A_bar = __expf(A_val * dt);
			        float x_val = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
	        if (recurrent_subloop_timing != NULL) {
	            dt_a_x_setup_cycles += clock64() - setup_t0;
	        }
	        unsigned long long update_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
			        for (int s = 0; s < state_size; s++) {
			            float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);
			            float B_bar = bf16_to_float(float_to_bf16(B_val * dt));
			            h[s] = A_bar * h[s] + B_bar * x_val;
			        }
	        if (recurrent_subloop_timing != NULL) {
	            state_update_cycles += clock64() - update_t0;
	            state_update_elements += (unsigned long long)state_size;
	        }
			        float c_state_total = 0.0f;
			        int c_row_base_idx = (t * n_groups + group) * state_size;
	        unsigned long long cdot_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
			        for (int s = 0; s < state_size; s++) {
			            float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
			            c_state_total += C_val * h[s];
			        }
	        if (recurrent_subloop_timing != NULL) {
	            c_dot_cycles += clock64() - cdot_t0;
	            c_dot_elements += (unsigned long long)state_size;
	        }
	        unsigned long long store_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
			        c_state_total_exact[(t * n_heads + head) * head_dim + d] = c_state_total;
	        if (recurrent_subloop_timing != NULL) {
	            c_state_store_cycles += clock64() - store_t0;
	            c_state_stores += 1ULL;
	            row_lanes += 1ULL;
	        }
			    }
	    if (recurrent_subloop_timing != NULL) {
	        mamba2_ssd_timing_add(recurrent_subloop_timing, 0, entry_snapshot_cycles);
	        mamba2_ssd_timing_add(recurrent_subloop_timing, 1, dt_a_x_setup_cycles);
	        mamba2_ssd_timing_add(recurrent_subloop_timing, 2, state_update_cycles);
	        mamba2_ssd_timing_add(recurrent_subloop_timing, 3, c_dot_cycles);
	        mamba2_ssd_timing_add(recurrent_subloop_timing, 4, c_state_store_cycles);
	        mamba2_ssd_timing_add(recurrent_subloop_timing, 5, clock64() - thread_total_t0);
	        atomicAdd(&recurrent_subloop_timing[6], row_lanes);
	        atomicAdd(&recurrent_subloop_timing[7], entry_snapshot_elements);
	        atomicAdd(&recurrent_subloop_timing[8], state_update_elements);
	        atomicAdd(&recurrent_subloop_timing[9], c_dot_elements);
	        atomicAdd(&recurrent_subloop_timing[10], c_state_stores);
		    }
				}

/*
 * Opt-in state-parallel recurrent prototype. Token order is unchanged for each
 * (head,d,s) recurrence. Independent state slots are updated in parallel, then
 * one thread per d lane computes c_state_total_exact with the same serial
 * ascending-s accumulation order as the accepted recurrent kernel.
 */
extern "C" __global__ void mamba2_ssd_block_scan_recurrent_state_parallel_kernel(
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
    float* __restrict__ entry_state,        /* [n_chunks, n_heads, head_dim, state_size] */
    float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
    unsigned long long* __restrict__ recurrent_subloop_timing,/* optional [11], NULL when disabled */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus)
{
    int head = blockIdx.x;
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    int d_tile = blockDim.x / state_size;
    if (d_tile <= 0) return;

    int lane = threadIdx.x / state_size;
    int state_idx = threadIdx.x - lane * state_size;
    int d = blockIdx.y * d_tile + lane;
    bool valid_lane = lane < d_tile && state_idx < state_size && d < head_dim;
    int group = head / (n_heads / n_groups);
    float A_val = mamba2_ssd_a_value(A_log[head]);

    extern __shared__ float mamba2_ssd_state_parallel_smem[];
    float* h_tile = mamba2_ssd_state_parallel_smem;
    float* b_bar_tile = h_tile + d_tile * state_size;
    float* c_tile = b_bar_tile + state_size;
    float* x_tile = c_tile + state_size;
    float* scalar_tile = x_tile + d_tile;

    unsigned long long thread_total_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
    unsigned long long entry_snapshot_cycles = 0ULL;
    unsigned long long dt_a_x_setup_cycles = 0ULL;
    unsigned long long state_update_cycles = 0ULL;
    unsigned long long c_dot_cycles = 0ULL;
    unsigned long long c_state_store_cycles = 0ULL;
    unsigned long long row_lanes = 0ULL;
    unsigned long long entry_snapshot_elements = 0ULL;
    unsigned long long state_update_elements = 0ULL;
    unsigned long long c_dot_elements = 0ULL;
    unsigned long long c_state_stores = 0ULL;

    if (valid_lane) {
        int64_t state_base = ((int64_t)head * head_dim + d) * state_size;
        h_tile[lane * state_size + state_idx] = ssm_state[state_base + state_idx];
    }
    __syncthreads();

    for (int t = 0; t < L; t++) {
        int chunk_idx = t / effective_chunk_size;
        int chunk_pos = t - chunk_idx * effective_chunk_size;
        if (chunk_pos == 0) {
            unsigned long long entry_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
            if (valid_lane) {
                int64_t entry_base =
                    ((int64_t)chunk_idx * n_heads * head_dim + (int64_t)head * head_dim + d)
                    * state_size;
                entry_state[entry_base + state_idx] = h_tile[lane * state_size + state_idx];
            }
            if (recurrent_subloop_timing != NULL) {
                entry_snapshot_cycles += clock64() - entry_t0;
                if (valid_lane) entry_snapshot_elements += 1ULL;
            }
        }

        unsigned long long setup_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
        if (threadIdx.x == 0) {
            float dt = mamba2_ssd_dt_value(dt_in, dt_bias, t, n_heads, head, use_softplus);
            scalar_tile[0] = dt;
            scalar_tile[1] = __expf(A_val * dt);
        }
        __syncthreads();
        float dt = scalar_tile[0];
        if (lane == 0 && state_idx < state_size) {
            int row_base = (t * n_groups + group) * state_size;
            float B_val = bf16_to_float(B_mat[row_base + state_idx]);
            b_bar_tile[state_idx] = bf16_to_float(float_to_bf16(B_val * dt));
            c_tile[state_idx] = bf16_to_float(C_mat[row_base + state_idx]);
        }
        if (state_idx == 0 && d < head_dim) {
            x_tile[lane] = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
        }
        __syncthreads();
        if (recurrent_subloop_timing != NULL) {
            dt_a_x_setup_cycles += clock64() - setup_t0;
        }

        unsigned long long update_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
        if (valid_lane) {
            int h_idx = lane * state_size + state_idx;
            h_tile[h_idx] = scalar_tile[1] * h_tile[h_idx] + b_bar_tile[state_idx] * x_tile[lane];
        }
        if (recurrent_subloop_timing != NULL) {
            state_update_cycles += clock64() - update_t0;
            if (valid_lane) state_update_elements += 1ULL;
        }
        __syncthreads();

        if (state_idx == 0 && d < head_dim) {
            float c_state_total = 0.0f;
            unsigned long long cdot_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
            for (int s = 0; s < state_size; s++) {
                c_state_total += c_tile[s] * h_tile[lane * state_size + s];
            }
            if (recurrent_subloop_timing != NULL) {
                c_dot_cycles += clock64() - cdot_t0;
                c_dot_elements += (unsigned long long)state_size;
            }
            unsigned long long store_t0 = (recurrent_subloop_timing != NULL) ? clock64() : 0ULL;
            c_state_total_exact[(t * n_heads + head) * head_dim + d] = c_state_total;
            if (recurrent_subloop_timing != NULL) {
                c_state_store_cycles += clock64() - store_t0;
                c_state_stores += 1ULL;
                row_lanes += 1ULL;
            }
        }
        __syncthreads();
    }

    if (valid_lane) {
        int64_t state_base = ((int64_t)head * head_dim + d) * state_size;
        ssm_state[state_base + state_idx] = h_tile[lane * state_size + state_idx];
    }

    if (recurrent_subloop_timing != NULL) {
        mamba2_ssd_timing_add(recurrent_subloop_timing, 0, entry_snapshot_cycles);
        mamba2_ssd_timing_add(recurrent_subloop_timing, 1, dt_a_x_setup_cycles);
        mamba2_ssd_timing_add(recurrent_subloop_timing, 2, state_update_cycles);
        mamba2_ssd_timing_add(recurrent_subloop_timing, 3, c_dot_cycles);
        mamba2_ssd_timing_add(recurrent_subloop_timing, 4, c_state_store_cycles);
        mamba2_ssd_timing_add(recurrent_subloop_timing, 5, clock64() - thread_total_t0);
        atomicAdd(&recurrent_subloop_timing[6], row_lanes);
        atomicAdd(&recurrent_subloop_timing[7], entry_snapshot_elements);
        atomicAdd(&recurrent_subloop_timing[8], state_update_elements);
        atomicAdd(&recurrent_subloop_timing[9], c_dot_elements);
        atomicAdd(&recurrent_subloop_timing[10], c_state_stores);
    }
}

/*
 * Opt-in block-scan correctness seed, phase 2. Output assembly is separated
 * from accepted final-state propagation. For each row, replay the same
 * token-order recurrence from the exact chunk-entry snapshot up to that row,
 * then run the existing local/output assembly math into candidate output.
 */
extern "C" __global__ void mamba2_ssd_block_scan_output_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
		    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
	    const float* __restrict__ entry_state,  /* [n_chunks, n_heads, head_dim, state_size] */
	    const float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
		    float* __restrict__ term_probe,         /* optional [16], NULL when disabled */
		    unsigned long long* __restrict__ subloop_timing,/* optional [8], NULL when disabled */
	    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int probe_row,
    int probe_head,
    int probe_d,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;

    int group = head / (n_heads / n_groups);
    float A_val = mamba2_ssd_a_value(A_log[head]);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    float dA_chunk_base = 0.0f;
    extern __shared__ float mamba2_ssd_block_scan_da_prefix_smem[];
    unsigned long long thread_total_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;

    for (int t = 0; t < L; t++) {
        unsigned long long setup_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
        int chunk_idx = t / effective_chunk_size;
        int chunk_start = chunk_idx * effective_chunk_size;
        int chunk_pos = t - chunk_start;
        int chunk_len = min(effective_chunk_size, L - chunk_start);
        float* dA_prefix = NULL;
        if (chunk_pos == 0) {
            dA_prefix = mamba2_ssd_build_da_prefix_scan(
                mamba2_ssd_block_scan_da_prefix_smem,
                effective_chunk_size,
                dt_in,
                dt_bias,
                A_val,
                n_heads,
                head,
                chunk_start,
                chunk_len,
                use_softplus);
        } else {
            dA_prefix = mamba2_ssd_block_scan_da_prefix_smem;
            int rounds = 0;
            for (int offset = 1; offset < chunk_len; offset <<= 1) rounds++;
            if ((rounds & 1) != 0) dA_prefix += effective_chunk_size;
        }

        float dA_target = dA_chunk_base + dA_prefix[chunk_pos];
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;

        float c_state_total = valid_d
            ? c_state_total_exact[(t * n_heads + head) * head_dim + d]
            : 0.0f;
        if (subloop_timing != NULL) {
            mamba2_ssd_timing_add(subloop_timing, 0, clock64() - setup_t0);
        }

		        float local_old_state = 0.0f;
		        float local_hf_chunk_scan = 0.0f;
		        if (valid_d) {
		            if (subloop_timing != NULL) {
		                atomicAdd(&subloop_timing[5], 1ULL);
		                atomicAdd(&subloop_timing[6], (unsigned long long)(t - chunk_start + 1));
		            }
		            for (int u = chunk_start; u <= t; u++) {
                int u_chunk_pos = u - chunk_start;
                float dA_running = dA_chunk_base + dA_prefix[u_chunk_pos];
                float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
                if (dt_bias != NULL) dt_u += dt_bias[head];
                float dt_u_scale = dt_u;
                if (use_softplus) {
                    dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                    dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                }
		                float decay = (u == t) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));

		                float old_state_source = 0.0f;
		                int c_row_base_idx = (t * n_groups + group) * state_size;
		                int b_row_base_idx = (u * n_groups + group) * state_size;
		                unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
		                float cb = mamba2_ssd_cb_dot_reverse(
		                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
		                if (subloop_timing != NULL) {
		                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
		                }
		                unsigned long long old_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
		                if (chunk_start != 0) {
		                    for (int s = 0; s < state_size; s++) {
		                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
		                        float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
		                        float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
		                        old_state_source += C_val * B_bar;
		                    }
		                }
		                if (subloop_timing != NULL) {
		                    mamba2_ssd_timing_add(subloop_timing, 1, clock64() - old_t0);
		                }

		                tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
		                float scale = decay * dt_u_scale;
		                float cb_scaled_bf16 = bf16_to_float(float_to_bf16(cb * scale));
		                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
		                if (subloop_timing != NULL) {
		                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
		                }
		                old_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
		                local_old_state += old_state_source * decay * x_u;
		                if (subloop_timing != NULL) {
		                    mamba2_ssd_timing_add(subloop_timing, 1, clock64() - old_t0);
		                }
		                tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
		                local_hf_chunk_scan += cb_scaled_bf16 * x_u;
		                if (subloop_timing != NULL) {
		                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
		                }
		            }
	        }

        if (valid_d) {
            unsigned long long d_prior_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            float prior_chunk_state = (chunk_start == 0) ? 0.0f : (c_state_total - local_old_state);
            float d_skip = D_val * x_val;
            float y = d_skip + prior_chunk_state + local_hf_chunk_scan;
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - d_prior_t0);
            }
            int flat_idx = (t * n_heads + head) * head_dim + d;
            unsigned long long cast_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
	            out[flat_idx] = float_to_bf16(y);
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 3, clock64() - cast_t0);
            }
	            if (term_probe != NULL && t == probe_row && head == probe_head && d == probe_d) {
		                float c_state_total_replayed = 0.0f;
		                int64_t entry_base = ((int64_t)chunk_idx * n_heads * head_dim + (int64_t)head * head_dim + d) * state_size;
		                int c_row_base_idx = (t * n_groups + group) * state_size;
	                for (int s = 0; s < state_size; s++) {
                    float h_s = entry_state[entry_base + s];
                    for (int v = chunk_start; v <= t; v++) {
                        float dt_v = mamba2_ssd_dt_value(dt_in, dt_bias, v, n_heads, head, use_softplus);
                        float A_bar_v = __expf(A_val * dt_v);
                        float B_val = bf16_to_float(B_mat[(v * n_groups + group) * state_size + s]);
                        float B_bar = bf16_to_float(float_to_bf16(B_val * dt_v));
                        float x_v = bf16_to_float(x[(v * n_heads + head) * head_dim + d]);
                        h_s = A_bar_v * h_s + B_bar * x_v;
                    }
		                    float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
		                    c_state_total_replayed += C_val * h_s;
		                }
		                term_probe[0] = c_state_total;
		                term_probe[1] = c_state_total_replayed;
		                term_probe[2] = local_old_state;
                term_probe[3] = prior_chunk_state;
                term_probe[4] = local_hf_chunk_scan;
                term_probe[5] = D_val * x_val;
                term_probe[6] = y;
                term_probe[7] = c_state_total - c_state_total_replayed;
                term_probe[8] = (float)chunk_idx;
                term_probe[9] = (float)chunk_start;
                term_probe[10] = (float)chunk_pos;
                term_probe[11] = (float)group;
                term_probe[12] = dA_target;
	                term_probe[13] = x_val;
		                term_probe[14] = D_val;
		                term_probe[15] = 1.0f;
		            }
	        }
        if (chunk_pos + 1 == chunk_len) {
            dA_chunk_base += dA_prefix[chunk_len - 1];
        }
        __syncthreads();
    }
	    if (subloop_timing != NULL) {
	        mamba2_ssd_timing_add(subloop_timing, 4, clock64() - thread_total_t0);
	    }
	}

/*
 * Opt-in coefficient-tiled block-scan output assembly. This keeps the
 * accepted recurrent state and c_state_total_exact path unchanged, but
 * computes per-(t,u) local scalar coefficients once per block before applying
 * them in the same increasing-u output accumulation order as the accepted
 * block-scan output kernel.
 */
extern "C" __global__ void mamba2_ssd_block_scan_output_coeff_tile_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    const float* __restrict__ entry_state,  /* [n_chunks, n_heads, head_dim, state_size] */
    const float* __restrict__ c_state_total_exact,/* unused in fast chunked path */
    float* __restrict__ term_probe,         /* optional [32], NULL when disabled */
    unsigned long long* __restrict__ subloop_timing,/* optional [10], NULL when disabled */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int probe_row,
    int probe_head,
    int probe_d,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int use_softplus)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;

    int group = head / (n_heads / n_groups);
    float A_val = mamba2_ssd_a_value(A_log[head]);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;

    extern __shared__ unsigned char mamba2_ssd_coeff_tile_smem[];
    float* dA_smem = reinterpret_cast<float*>(mamba2_ssd_coeff_tile_smem);
    int max_pair_count = (effective_chunk_size * (effective_chunk_size + 1)) / 2;
    size_t prefix_bytes = (size_t)effective_chunk_size * 2 * sizeof(float);
    size_t tri_offset = prefix_bytes;
    size_t tri_bytes = (size_t)max_pair_count * sizeof(unsigned short);
    size_t old_offset = (tri_offset + tri_bytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    unsigned short* tri_coeff_bits = reinterpret_cast<unsigned short*>(mamba2_ssd_coeff_tile_smem + tri_offset);
    float* old_coeff_tile = reinterpret_cast<float*>(mamba2_ssd_coeff_tile_smem + old_offset);
    unsigned long long thread_total_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;

    float dA_chunk_base = 0.0f;
    for (int chunk_start = 0; chunk_start < L; chunk_start += effective_chunk_size) {
        int chunk_len = min(effective_chunk_size, L - chunk_start);
        float* dA_prefix = mamba2_ssd_build_da_prefix_scan(
            dA_smem,
            effective_chunk_size,
            dt_in,
            dt_bias,
            A_val,
            n_heads,
            head,
            chunk_start,
            chunk_len,
            use_softplus);

        unsigned long long coeff_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
        for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
            float dA_target = dA_chunk_base + dA_prefix[t_pos];
            int t_abs = chunk_start + t_pos;
            int c_row_base_idx = (t_abs * n_groups + group) * state_size;
            for (int u_pos = threadIdx.x; u_pos <= t_pos; u_pos += blockDim.x) {
                int u_abs = chunk_start + u_pos;
                int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                float dA_running = dA_chunk_base + dA_prefix[u_pos];
                float dt_u = bf16_to_float(dt_in[u_abs * n_heads + head]);
                if (dt_bias != NULL) dt_u += dt_bias[head];
                float dt_u_scale = dt_u;
                if (use_softplus) {
                    dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                    dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                }
                float decay = (u_pos == t_pos) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));
                int b_row_base_idx = (u_abs * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                float scale = decay * dt_u_scale;
                __nv_bfloat16 tri_bf16 = float_to_bf16(cb * scale);
                tri_coeff_bits[pair_idx] = *reinterpret_cast<unsigned short*>(&tri_bf16);

                float old_state_source = 0.0f;
                if (chunk_start != 0) {
                    for (int s = 0; s < state_size; s++) {
                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                        float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
                        float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
                        old_state_source += C_val * B_bar;
                    }
                }
                old_coeff_tile[pair_idx] = old_state_source * decay;
            }
        }
        if (subloop_timing != NULL) {
            mamba2_ssd_timing_add(subloop_timing, 7, clock64() - coeff_t0);
            if (threadIdx.x == 0) {
                atomicAdd(&subloop_timing[8], (unsigned long long)((chunk_len * (chunk_len + 1)) / 2));
            }
        }
        __syncthreads();

        int chunk_idx = chunk_start / effective_chunk_size;
        for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
            unsigned long long setup_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            int t = chunk_start + t_pos;
            float dA_target = dA_chunk_base + dA_prefix[t_pos];
            float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;
            float c_state_total = valid_d
                ? c_state_total_exact[(t * n_heads + head) * head_dim + d]
                : 0.0f;
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - setup_t0);
            }

            float local_old_state = 0.0f;
            float local_hf_chunk_scan = 0.0f;
            float probe_max_tri_delta = 0.0f;
            float probe_max_old_delta = 0.0f;
            float probe_inline_tri = 0.0f;
            float probe_tiled_tri = 0.0f;
            float probe_inline_old = 0.0f;
            float probe_tiled_old = 0.0f;
            int probe_first_mismatch_u = -1;
            int probe_first_mismatch_kind = 0;
            int probe_compared = 0;
            bool target_probe = term_probe != NULL && t == probe_row && head == probe_head && d == probe_d;

            if (valid_d) {
                if (subloop_timing != NULL) {
                    atomicAdd(&subloop_timing[5], 1ULL);
                    atomicAdd(&subloop_timing[6], (unsigned long long)(t_pos + 1));
                }
                for (int u_pos = 0; u_pos <= t_pos; u_pos++) {
                    int u = chunk_start + u_pos;
                    int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                    unsigned short tiled_tri_bits = tri_coeff_bits[pair_idx];
                    __nv_bfloat16 tiled_tri_bf16 = *reinterpret_cast<__nv_bfloat16*>(&tiled_tri_bits);
                    float tri_coeff = bf16_to_float(tiled_tri_bf16);
                    float old_coeff = old_coeff_tile[pair_idx];

                    if (target_probe) {
                        float dA_running = dA_chunk_base + dA_prefix[u_pos];
                        float dt_u = bf16_to_float(dt_in[u * n_heads + head]);
                        if (dt_bias != NULL) dt_u += dt_bias[head];
                        float dt_u_scale = dt_u;
                        if (use_softplus) {
                            dt_u = mamba2_chunk_cumsum_softplus(dt_u);
                            dt_u_scale = mamba2_chunk_cumsum_softplus(dt_u_scale);
                        }
                        float decay = (u_pos == t_pos) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));
                        int c_row_base_idx = (t * n_groups + group) * state_size;
                        int b_row_base_idx = (u * n_groups + group) * state_size;
                        float cb = mamba2_ssd_cb_dot_reverse(
                            C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                        float scale = decay * dt_u_scale;
                        __nv_bfloat16 inline_tri_bf16 = float_to_bf16(cb * scale);
                        unsigned short inline_tri_bits = *reinterpret_cast<unsigned short*>(&inline_tri_bf16);
                        float inline_tri = bf16_to_float(inline_tri_bf16);
                        float old_state_source = 0.0f;
                        if (chunk_start != 0) {
                            for (int s = 0; s < state_size; s++) {
                                float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                                float B_val = bf16_to_float(B_mat[b_row_base_idx + s]);
                                float B_bar = bf16_to_float(float_to_bf16(B_val * dt_u));
                                old_state_source += C_val * B_bar;
                            }
                        }
                        float inline_old = old_state_source * decay;
                        float tri_delta = fabsf(inline_tri - tri_coeff);
                        float old_delta = fabsf(inline_old - old_coeff);
                        if (tri_delta > probe_max_tri_delta) {
                            probe_max_tri_delta = tri_delta;
                            probe_inline_tri = inline_tri;
                            probe_tiled_tri = tri_coeff;
                        }
                        if (old_delta > probe_max_old_delta) {
                            probe_max_old_delta = old_delta;
                            probe_inline_old = inline_old;
                            probe_tiled_old = old_coeff;
                        }
                        if (probe_first_mismatch_u < 0 && inline_tri_bits != tiled_tri_bits) {
                            probe_first_mismatch_u = u;
                            probe_first_mismatch_kind = 1;
                            probe_inline_tri = inline_tri;
                            probe_tiled_tri = tri_coeff;
                        }
                        if (
                            probe_first_mismatch_u < 0
                            && __float_as_uint(inline_old) != __float_as_uint(old_coeff)
                        ) {
                            probe_first_mismatch_u = u;
                            probe_first_mismatch_kind = 2;
                            probe_inline_old = inline_old;
                            probe_tiled_old = old_coeff;
                        }
                        probe_compared++;
                    }

                    float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                    unsigned long long old_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                    local_old_state += old_coeff * x_u;
                    if (subloop_timing != NULL) {
                        mamba2_ssd_timing_add(subloop_timing, 1, clock64() - old_t0);
                    }
                    unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                    local_hf_chunk_scan += tri_coeff * x_u;
                    if (subloop_timing != NULL) {
                        mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
                    }
                }
            }

            if (valid_d) {
                unsigned long long d_prior_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                float prior_chunk_state = (chunk_start == 0) ? 0.0f : (c_state_total - local_old_state);
                float d_skip = D_val * x_val;
                float y = d_skip + prior_chunk_state + local_hf_chunk_scan;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 0, clock64() - d_prior_t0);
                }
                int flat_idx = (t * n_heads + head) * head_dim + d;
                unsigned long long cast_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                out[flat_idx] = float_to_bf16(y);
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 3, clock64() - cast_t0);
                }
                if (target_probe) {
                    float c_state_total_replayed = 0.0f;
                    int64_t entry_base = ((int64_t)chunk_idx * n_heads * head_dim + (int64_t)head * head_dim + d) * state_size;
                    int c_row_base_idx = (t * n_groups + group) * state_size;
                    for (int s = 0; s < state_size; s++) {
                        float h_s = entry_state[entry_base + s];
                        for (int v = chunk_start; v <= t; v++) {
                            float dt_v = mamba2_ssd_dt_value(dt_in, dt_bias, v, n_heads, head, use_softplus);
                            float A_bar_v = __expf(A_val * dt_v);
                            float B_val = bf16_to_float(B_mat[(v * n_groups + group) * state_size + s]);
                            float B_bar = bf16_to_float(float_to_bf16(B_val * dt_v));
                            float x_v = bf16_to_float(x[(v * n_heads + head) * head_dim + d]);
                            h_s = A_bar_v * h_s + B_bar * x_v;
                        }
                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                        c_state_total_replayed += C_val * h_s;
                    }
                    term_probe[0] = c_state_total;
                    term_probe[1] = c_state_total_replayed;
                    term_probe[2] = local_old_state;
                    term_probe[3] = prior_chunk_state;
                    term_probe[4] = local_hf_chunk_scan;
                    term_probe[5] = D_val * x_val;
                    term_probe[6] = y;
                    term_probe[7] = c_state_total - c_state_total_replayed;
                    term_probe[8] = (float)chunk_idx;
                    term_probe[9] = (float)chunk_start;
                    term_probe[10] = (float)t_pos;
                    term_probe[11] = (float)group;
                    term_probe[12] = dA_target;
                    term_probe[13] = x_val;
                    term_probe[14] = D_val;
                    term_probe[15] = 1.0f;
                    term_probe[16] = probe_max_tri_delta;
                    term_probe[17] = probe_max_old_delta;
                    term_probe[18] = (float)probe_first_mismatch_u;
                    term_probe[19] = (float)probe_first_mismatch_kind;
                    term_probe[20] = probe_inline_tri;
                    term_probe[21] = probe_tiled_tri;
                    term_probe[22] = probe_inline_old;
                    term_probe[23] = probe_tiled_old;
                    term_probe[24] = (float)probe_compared;
                    term_probe[25] = 1.0f;
                }
            }
        }

        dA_chunk_base += dA_prefix[chunk_len - 1];
        __syncthreads();
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 4, clock64() - thread_total_t0);
    }
}

/*
 * Opt-in parallel/chunked SSD architecture seed. These kernels mirror the
 * mamba-ssm prefill decomposition at the phase level:
 *   chunk cumsum -> chunk state -> state passing -> chunk scan output.
 * The path is selected only from Rust behind an explicit diagnostic env gate.
 */
extern "C" __global__ void mamba2_ssd_parallel_chunk_cumsum_kernel(
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    float* __restrict__ dt_out,             /* [n_chunks, n_heads, chunk_size] */
    float* __restrict__ dA_cumsum,          /* [n_chunks, n_heads, chunk_size] */
    int L,
    int n_heads,
    int chunk_size,
    int use_softplus)
{
    int head = blockIdx.x;
    int chunk_idx = blockIdx.y;
    if (head >= n_heads || chunk_size <= 0) return;

    int chunk_start = chunk_idx * chunk_size;
    float A_val = mamba2_ssd_a_value(A_log[head]);
    int64_t base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;
    int chunk_len = min(chunk_size, L - chunk_start);
    if (chunk_len <= 0) {
        for (int pos = threadIdx.x; pos < chunk_size; pos += blockDim.x) {
            dt_out[base + pos] = 0.0f;
            dA_cumsum[base + pos] = 0.0f;
        }
        return;
    }

    extern __shared__ float cumsum_smem[];
    float* src = cumsum_smem;
    float* dst = cumsum_smem + chunk_size;

    for (int pos = threadIdx.x; pos < chunk_len; pos += blockDim.x) {
        int t = chunk_start + pos;
        float dt = mamba2_ssd_dt_value(dt_in, dt_bias, t, n_heads, head, use_softplus);
        dt_out[base + pos] = dt;
        src[pos] = A_val * dt;
    }
    __syncthreads();

    for (int offset = 1; offset < chunk_len; offset <<= 1) {
        for (int pos = threadIdx.x; pos < chunk_len; pos += blockDim.x) {
            float value = src[pos];
            if (pos >= offset) value += src[pos - offset];
            dst[pos] = value;
        }
        __syncthreads();
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    float final_prefix = src[chunk_len - 1];
    for (int pos = threadIdx.x; pos < chunk_size; pos += blockDim.x) {
        if (pos >= chunk_len) {
            dt_out[base + pos] = 0.0f;
        }
        dA_cumsum[base + pos] = (pos < chunk_len) ? src[pos] : final_prefix;
    }
}

extern "C" __global__ void mamba2_ssd_parallel_chunk_state_kernel(
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ dt_out,       /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ A_log,        /* [n_heads] */
    const float* __restrict__ dA_cumsum,    /* [n_chunks, n_heads, chunk_size] */
    float* __restrict__ chunk_states,       /* [n_chunks, n_heads, head_dim, state_size] */
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size)
{
    int head = blockIdx.x;
    int d = blockIdx.y;
    int chunk_idx = blockIdx.z;
    int s = threadIdx.x;
    if (head >= n_heads || d >= head_dim || s >= state_size || chunk_size <= 0) return;

    int group = head / (n_heads / n_groups);
    int chunk_start = chunk_idx * chunk_size;
    int chunk_len = min(chunk_size, L - chunk_start);
    if (chunk_len <= 0) return;

    int64_t chunk_base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;
    (void)dA_cumsum;
    float A_val = mamba2_ssd_a_value(A_log[head]);
    float h = 0.0f;
    for (int pos = 0; pos < chunk_len; pos++) {
        int t = chunk_start + pos;
        float dt = dt_out[chunk_base + pos];
        float A_bar = __expf(A_val * dt);
        float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);
        float B_bar = bf16_to_float(float_to_bf16(B_val * dt));
        float x_val = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);
        h = A_bar * h + B_bar * x_val;
    }

    int64_t out_idx = (((int64_t)chunk_idx * n_heads + head) * head_dim + d) * state_size + s;
    chunk_states[out_idx] = h;
}

extern "C" __global__ void mamba2_ssd_parallel_state_passing_kernel(
    const float* __restrict__ chunk_states, /* [n_chunks, n_heads, head_dim, state_size] */
    const float* __restrict__ dA_cumsum,    /* [n_chunks, n_heads, chunk_size] */
    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
    float* __restrict__ entry_state,        /* [n_chunks, n_heads, head_dim, state_size] */
    int n_chunks,
    int n_heads,
    int head_dim,
    int state_size,
    int chunk_size)
{
    int head = blockIdx.x;
    int flat = blockIdx.y * blockDim.x + threadIdx.x;
    int elems_per_head = head_dim * state_size;
    if (head >= n_heads || flat >= elems_per_head || chunk_size <= 0) return;

    int d = flat / state_size;
    int s = flat - d * state_size;
    int64_t state_idx = ((int64_t)head * head_dim + d) * state_size + s;
    float state = ssm_state[state_idx];
    for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++) {
        int64_t entry_idx = (((int64_t)chunk_idx * n_heads + head) * head_dim + d) * state_size + s;
        entry_state[entry_idx] = state;
        int64_t chunk_base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;
        float dA_last = dA_cumsum[chunk_base + chunk_size - 1];
        float new_state = chunk_states[entry_idx];
        state = __expf(dA_last) * state + new_state;
    }
    ssm_state[state_idx] = state;
}

extern "C" __global__ void mamba2_ssd_parallel_chunk_scan_output_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    const float* __restrict__ A_log,        /* [n_heads] */
    const float* __restrict__ dt_out,       /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ dA_cumsum,    /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ entry_state,  /* [n_chunks, n_heads, head_dim, state_size] */
    const float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
    float* __restrict__ term_probe,         /* optional [32], NULL when disabled */
    unsigned long long* __restrict__ subloop_timing,/* optional [10], NULL when disabled */
    int probe_row,
    int probe_head,
    int probe_d,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    bool valid_d = d < head_dim;
    if (head >= n_heads || chunk_size <= 0) return;

    int group = head / (n_heads / n_groups);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    (void)A_log;
    (void)c_state_total_exact;
    int max_pair_count = (chunk_size * (chunk_size + 1)) / 2;
    extern __shared__ unsigned char parallel_chunk_scan_smem[];
    unsigned short* tri_coeff_bits = reinterpret_cast<unsigned short*>(parallel_chunk_scan_smem);
    unsigned long long thread_total_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;

    float dA_chunk_base = 0.0f;
    for (int chunk_start = 0, chunk_idx = 0; chunk_start < L; chunk_start += chunk_size, chunk_idx++) {
        int chunk_len = min(chunk_size, L - chunk_start);
        int64_t chunk_base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;

        unsigned long long coeff_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
        for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
            float dA_local_target = dA_cumsum[chunk_base + t_pos];
            float dA_target = dA_chunk_base + dA_local_target;
            int t_abs = chunk_start + t_pos;
            int c_row_base_idx = (t_abs * n_groups + group) * state_size;
            for (int u_pos = threadIdx.x; u_pos <= t_pos; u_pos += blockDim.x) {
                int u_abs = chunk_start + u_pos;
                int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                float dA_running = dA_chunk_base + dA_cumsum[chunk_base + u_pos];
                float dt_u = dt_out[chunk_base + u_pos];
                float decay = (u_pos == t_pos) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));
                int b_row_base_idx = (u_abs * n_groups + group) * state_size;
                float cb = mamba2_ssd_cb_dot_reverse(
                    C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
                float scale = decay * dt_u;
                __nv_bfloat16 tri_bf16 = float_to_bf16(cb * scale);
                tri_coeff_bits[pair_idx] = *reinterpret_cast<unsigned short*>(&tri_bf16);
            }
        }
        if (subloop_timing != NULL) {
            mamba2_ssd_timing_add(subloop_timing, 7, clock64() - coeff_t0);
            if (threadIdx.x == 0) {
                atomicAdd(&subloop_timing[8], (unsigned long long)((chunk_len * (chunk_len + 1)) / 2));
            }
        }
        __syncthreads();

        for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
            int t = chunk_start + t_pos;
            float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;
            float dA_local_target = dA_cumsum[chunk_base + t_pos];
            float dA_target = dA_chunk_base + dA_local_target;
            float prior_chunk_state = 0.0f;
            float local_hf_chunk_scan = 0.0f;
            float entry_dot = 0.0f;

            if (valid_d) {
                int flat_idx = (t * n_heads + head) * head_dim + d;
                unsigned long long prior_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                int c_row_base_idx = (t * n_groups + group) * state_size;
                int64_t entry_base =
                    (((int64_t)chunk_idx * n_heads + head) * head_dim + d) * state_size;
                if (chunk_start != 0) {
                    for (int s = 0; s < state_size; s++) {
                        float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                        entry_dot += C_val * entry_state[entry_base + s];
                    }
                    prior_chunk_state = __expf(dA_local_target) * entry_dot;
                }
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 0, clock64() - prior_t0);
                    atomicAdd(&subloop_timing[5], 1ULL);
                    atomicAdd(&subloop_timing[6], (unsigned long long)(t_pos + 1));
                }

                for (int u_pos = 0; u_pos <= t_pos; u_pos++) {
                    int u = chunk_start + u_pos;
                    int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                    unsigned short tri_bits = tri_coeff_bits[pair_idx];
                    __nv_bfloat16 tri_bf16 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits);
                    float tri_coeff = bf16_to_float(tri_bf16);
                    float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                    unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                    local_hf_chunk_scan += tri_coeff * x_u;
                    if (subloop_timing != NULL) {
                        mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
                    }
                }

                unsigned long long d_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                float d_skip = D_val * x_val;
                float y = d_skip + prior_chunk_state + local_hf_chunk_scan;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 0, clock64() - d_t0);
                }
                unsigned long long cast_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                out[flat_idx] = float_to_bf16(y);
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 3, clock64() - cast_t0);
                }
                if (term_probe != NULL && t == probe_row && head == probe_head && d == probe_d) {
                    term_probe[0] = entry_dot;
                    term_probe[1] = __expf(dA_local_target);
                    term_probe[2] = entry_dot * __expf(dA_local_target);
                    term_probe[3] = prior_chunk_state;
                    term_probe[4] = local_hf_chunk_scan;
                    term_probe[5] = d_skip;
                    term_probe[6] = y;
                    term_probe[7] = 0.0f;
                    term_probe[8] = (float)chunk_idx;
                    term_probe[9] = (float)chunk_start;
                    term_probe[10] = (float)t_pos;
                    term_probe[11] = (float)group;
                    term_probe[12] = dA_target;
                    term_probe[13] = x_val;
                    term_probe[14] = D_val;
                    term_probe[15] = 1.0f;
                    term_probe[16] = 0.0f;
                    term_probe[17] = 0.0f;
                    term_probe[18] = -1.0f;
                    term_probe[19] = 0.0f;
                    term_probe[20] = 0.0f;
                    term_probe[21] = 0.0f;
                    term_probe[22] = 0.0f;
                    term_probe[23] = 0.0f;
                    term_probe[24] = (float)(t_pos + 1);
                    term_probe[25] = 2.0f;
                }
            }
        }
        dA_chunk_base += dA_cumsum[chunk_base + chunk_len - 1];
        (void)max_pair_count;
        __syncthreads();
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 4, clock64() - thread_total_t0);
    }
}

extern "C" __global__ void mamba2_ssd_parallel_chunk_scan_output_by_chunk_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    const float* __restrict__ A_log,        /* [n_heads] */
    const float* __restrict__ dt_out,       /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ dA_cumsum,    /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ entry_state,  /* [n_chunks, n_heads, head_dim, state_size] */
    const float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
    float* __restrict__ term_probe,         /* optional [32], NULL when disabled */
    unsigned long long* __restrict__ subloop_timing,/* optional [10], NULL when disabled */
    int probe_row,
    int probe_head,
    int probe_d,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int chunk_idx = blockIdx.z;
    bool valid_d = d < head_dim;
    if (head >= n_heads || chunk_size <= 0) return;

    int chunk_start = chunk_idx * chunk_size;
    if (chunk_start >= L) return;
    int chunk_len = min(chunk_size, L - chunk_start);
    if (chunk_len <= 0) return;

    int group = head / (n_heads / n_groups);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    (void)A_log;
    (void)c_state_total_exact;
    int max_pair_count = (chunk_size * (chunk_size + 1)) / 2;
    extern __shared__ unsigned char parallel_chunk_scan_smem[];
    unsigned short* tri_coeff_bits = reinterpret_cast<unsigned short*>(parallel_chunk_scan_smem);
    __shared__ float dA_chunk_base_shared;
    unsigned long long thread_total_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;

    if (threadIdx.x == 0) {
        float dA_chunk_base = 0.0f;
        for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
            int prev_start = prev_chunk * chunk_size;
            int prev_len = min(chunk_size, L - prev_start);
            if (prev_len > 0) {
                int64_t prev_base = ((int64_t)prev_chunk * n_heads + head) * chunk_size;
                dA_chunk_base += dA_cumsum[prev_base + prev_len - 1];
            }
        }
        dA_chunk_base_shared = dA_chunk_base;
    }
    __syncthreads();
    float dA_chunk_base = dA_chunk_base_shared;
    int64_t chunk_base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;

    unsigned long long coeff_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        float dA_local_target = dA_cumsum[chunk_base + t_pos];
        float dA_target = dA_chunk_base + dA_local_target;
        int t_abs = chunk_start + t_pos;
        int c_row_base_idx = (t_abs * n_groups + group) * state_size;
        for (int u_pos = threadIdx.x; u_pos <= t_pos; u_pos += blockDim.x) {
            int u_abs = chunk_start + u_pos;
            int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
            float dA_running = dA_chunk_base + dA_cumsum[chunk_base + u_pos];
            float dt_u = dt_out[chunk_base + u_pos];
            float decay = (u_pos == t_pos) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));
            int b_row_base_idx = (u_abs * n_groups + group) * state_size;
            float cb = mamba2_ssd_cb_dot_reverse(
                C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
            float scale = decay * dt_u;
            __nv_bfloat16 tri_bf16 = float_to_bf16(cb * scale);
            tri_coeff_bits[pair_idx] = *reinterpret_cast<unsigned short*>(&tri_bf16);
        }
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 7, clock64() - coeff_t0);
        if (threadIdx.x == 0) {
            atomicAdd(&subloop_timing[8], (unsigned long long)((chunk_len * (chunk_len + 1)) / 2));
        }
    }
    __syncthreads();

    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        int t = chunk_start + t_pos;
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;
        float dA_local_target = dA_cumsum[chunk_base + t_pos];
        float dA_target = dA_chunk_base + dA_local_target;
        float prior_chunk_state = 0.0f;
        float local_hf_chunk_scan = 0.0f;
        float entry_dot = 0.0f;

        if (valid_d) {
            int flat_idx = (t * n_heads + head) * head_dim + d;
            unsigned long long prior_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            int c_row_base_idx = (t * n_groups + group) * state_size;
            int64_t entry_base =
                (((int64_t)chunk_idx * n_heads + head) * head_dim + d) * state_size;
            if (chunk_start != 0) {
                for (int s = 0; s < state_size; s++) {
                    float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                    entry_dot += C_val * entry_state[entry_base + s];
                }
                prior_chunk_state = __expf(dA_local_target) * entry_dot;
            }
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - prior_t0);
                atomicAdd(&subloop_timing[5], 1ULL);
                atomicAdd(&subloop_timing[6], (unsigned long long)(t_pos + 1));
            }

            for (int u_pos = 0; u_pos <= t_pos; u_pos++) {
                int u = chunk_start + u_pos;
                int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                unsigned short tri_bits = tri_coeff_bits[pair_idx];
                __nv_bfloat16 tri_bf16 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits);
                float tri_coeff = bf16_to_float(tri_bf16);
                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff * x_u;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
                }
            }

            unsigned long long d_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            float d_skip = D_val * x_val;
            float y = d_skip + prior_chunk_state + local_hf_chunk_scan;
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - d_t0);
            }
            unsigned long long cast_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            out[flat_idx] = float_to_bf16(y);
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 3, clock64() - cast_t0);
            }
            if (term_probe != NULL && t == probe_row && head == probe_head && d == probe_d) {
                term_probe[0] = entry_dot;
                term_probe[1] = __expf(dA_local_target);
                term_probe[2] = entry_dot * __expf(dA_local_target);
                term_probe[3] = prior_chunk_state;
                term_probe[4] = local_hf_chunk_scan;
                term_probe[5] = d_skip;
                term_probe[6] = y;
                term_probe[7] = 0.0f;
                term_probe[8] = (float)chunk_idx;
                term_probe[9] = (float)chunk_start;
                term_probe[10] = (float)t_pos;
                term_probe[11] = (float)group;
                term_probe[12] = dA_target;
                term_probe[13] = x_val;
                term_probe[14] = D_val;
                term_probe[15] = 1.0f;
                term_probe[16] = 0.0f;
                term_probe[17] = 0.0f;
                term_probe[18] = -1.0f;
                term_probe[19] = 0.0f;
                term_probe[20] = 0.0f;
                term_probe[21] = 0.0f;
                term_probe[22] = 0.0f;
                term_probe[23] = 0.0f;
                term_probe[24] = (float)(t_pos + 1);
                term_probe[25] = 3.0f;
            }
        }
    }
    (void)max_pair_count;
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 4, clock64() - thread_total_t0);
    }
}

extern "C" __global__ void mamba2_ssd_parallel_precompute_cb_kernel(
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    float* __restrict__ cb_tile,            /* [n_chunks, n_groups, max_pair_count] */
    int L,
    int state_size,
    int n_groups,
    int chunk_size)
{
    int group = blockIdx.x;
    int chunk_idx = blockIdx.y;
    if (group >= n_groups || chunk_size <= 0) return;

    int chunk_start = chunk_idx * chunk_size;
    if (chunk_start >= L) return;
    int chunk_len = min(chunk_size, L - chunk_start);
    if (chunk_len <= 0) return;

    int max_pair_count = (chunk_size * (chunk_size + 1)) / 2;
    int64_t tile_base = ((int64_t)chunk_idx * n_groups + group) * max_pair_count;
    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        int t_abs = chunk_start + t_pos;
        int c_row_base_idx = (t_abs * n_groups + group) * state_size;
        for (int u_pos = threadIdx.x; u_pos <= t_pos; u_pos += blockDim.x) {
            int u_abs = chunk_start + u_pos;
            int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
            int b_row_base_idx = (u_abs * n_groups + group) * state_size;
            cb_tile[tile_base + pair_idx] = mamba2_ssd_cb_dot_reverse(
                C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
        }
    }
}

extern "C" __global__ void mamba2_ssd_parallel_chunk_scan_output_precomputed_cb_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    const float* __restrict__ A_log,        /* [n_heads] */
    const float* __restrict__ dt_out,       /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ dA_cumsum,    /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ entry_state,  /* [n_chunks, n_heads, head_dim, state_size] */
    const float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
    const float* __restrict__ cb_tile,      /* [n_chunks, n_groups, max_pair_count] */
    float* __restrict__ term_probe,         /* optional [32], NULL when disabled */
    unsigned long long* __restrict__ subloop_timing,/* optional [10], NULL when disabled */
    int probe_row,
    int probe_head,
    int probe_d,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size)
{
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int chunk_idx = blockIdx.z;
    bool valid_d = d < head_dim;
    if (head >= n_heads || chunk_size <= 0) return;

    int chunk_start = chunk_idx * chunk_size;
    if (chunk_start >= L) return;
    int chunk_len = min(chunk_size, L - chunk_start);
    if (chunk_len <= 0) return;

    int group = head / (n_heads / n_groups);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    (void)A_log;
    (void)B_mat;
    (void)C_mat;
    (void)c_state_total_exact;
    int max_pair_count = (chunk_size * (chunk_size + 1)) / 2;
    extern __shared__ unsigned char parallel_chunk_scan_smem[];
    unsigned short* tri_coeff_bits = reinterpret_cast<unsigned short*>(parallel_chunk_scan_smem);
    __shared__ float dA_chunk_base_shared;
    unsigned long long thread_total_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;

    if (threadIdx.x == 0) {
        float dA_chunk_base = 0.0f;
        for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
            int prev_start = prev_chunk * chunk_size;
            int prev_len = min(chunk_size, L - prev_start);
            if (prev_len > 0) {
                int64_t prev_base = ((int64_t)prev_chunk * n_heads + head) * chunk_size;
                dA_chunk_base += dA_cumsum[prev_base + prev_len - 1];
            }
        }
        dA_chunk_base_shared = dA_chunk_base;
    }
    __syncthreads();
    float dA_chunk_base = dA_chunk_base_shared;
    int64_t chunk_base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;
    int64_t cb_tile_base = ((int64_t)chunk_idx * n_groups + group) * max_pair_count;

    unsigned long long coeff_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        float dA_local_target = dA_cumsum[chunk_base + t_pos];
        float dA_target = dA_chunk_base + dA_local_target;
        for (int u_pos = threadIdx.x; u_pos <= t_pos; u_pos += blockDim.x) {
            int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
            float dA_running = dA_chunk_base + dA_cumsum[chunk_base + u_pos];
            float dt_u = dt_out[chunk_base + u_pos];
            float decay = (u_pos == t_pos) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));
            float scale = decay * dt_u;
            __nv_bfloat16 tri_bf16 = float_to_bf16(cb_tile[cb_tile_base + pair_idx] * scale);
            tri_coeff_bits[pair_idx] = *reinterpret_cast<unsigned short*>(&tri_bf16);
        }
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 7, clock64() - coeff_t0);
        if (threadIdx.x == 0) {
            atomicAdd(&subloop_timing[8], (unsigned long long)((chunk_len * (chunk_len + 1)) / 2));
        }
    }
    __syncthreads();

    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        int t = chunk_start + t_pos;
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;
        float dA_local_target = dA_cumsum[chunk_base + t_pos];
        float dA_target = dA_chunk_base + dA_local_target;
        float prior_chunk_state = 0.0f;
        float local_hf_chunk_scan = 0.0f;
        float entry_dot = 0.0f;

        if (valid_d) {
            int flat_idx = (t * n_heads + head) * head_dim + d;
            unsigned long long prior_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            int c_row_base_idx = (t * n_groups + group) * state_size;
            int64_t entry_base =
                (((int64_t)chunk_idx * n_heads + head) * head_dim + d) * state_size;
            if (chunk_start != 0) {
                int s = 0;
                for (; s + 3 < state_size; s += 4) {
                    float C_val0 = bf16_to_float(C_mat[c_row_base_idx + s]);
                    entry_dot += C_val0 * entry_state[entry_base + s];
                    float C_val1 = bf16_to_float(C_mat[c_row_base_idx + s + 1]);
                    entry_dot += C_val1 * entry_state[entry_base + s + 1];
                    float C_val2 = bf16_to_float(C_mat[c_row_base_idx + s + 2]);
                    entry_dot += C_val2 * entry_state[entry_base + s + 2];
                    float C_val3 = bf16_to_float(C_mat[c_row_base_idx + s + 3]);
                    entry_dot += C_val3 * entry_state[entry_base + s + 3];
                }
                for (; s < state_size; s++) {
                    float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                    entry_dot += C_val * entry_state[entry_base + s];
                }
                prior_chunk_state = __expf(dA_local_target) * entry_dot;
            }
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - prior_t0);
                atomicAdd(&subloop_timing[5], 1ULL);
                atomicAdd(&subloop_timing[6], (unsigned long long)(t_pos + 1));
            }

            int u_pos = 0;
            for (; u_pos + 3 <= t_pos; u_pos += 4) {
                int u0 = chunk_start + u_pos;
                int pair_idx0 = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                unsigned short tri_bits0 = tri_coeff_bits[pair_idx0];
                __nv_bfloat16 tri_bf160 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits0);
                float tri_coeff0 = bf16_to_float(tri_bf160);
                float x_u0 = bf16_to_float(x[(u0 * n_heads + head) * head_dim + d]);
                unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff0 * x_u0;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
                }

                int u1 = chunk_start + u_pos + 1;
                int pair_idx1 = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos + 1);
                unsigned short tri_bits1 = tri_coeff_bits[pair_idx1];
                __nv_bfloat16 tri_bf161 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits1);
                float tri_coeff1 = bf16_to_float(tri_bf161);
                float x_u1 = bf16_to_float(x[(u1 * n_heads + head) * head_dim + d]);
                unsigned long long tri_t1 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff1 * x_u1;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t1);
                }

                int u2 = chunk_start + u_pos + 2;
                int pair_idx2 = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos + 2);
                unsigned short tri_bits2 = tri_coeff_bits[pair_idx2];
                __nv_bfloat16 tri_bf162 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits2);
                float tri_coeff2 = bf16_to_float(tri_bf162);
                float x_u2 = bf16_to_float(x[(u2 * n_heads + head) * head_dim + d]);
                unsigned long long tri_t2 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff2 * x_u2;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t2);
                }

                int u3 = chunk_start + u_pos + 3;
                int pair_idx3 = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos + 3);
                unsigned short tri_bits3 = tri_coeff_bits[pair_idx3];
                __nv_bfloat16 tri_bf163 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits3);
                float tri_coeff3 = bf16_to_float(tri_bf163);
                float x_u3 = bf16_to_float(x[(u3 * n_heads + head) * head_dim + d]);
                unsigned long long tri_t3 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff3 * x_u3;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t3);
                }
            }
            for (; u_pos <= t_pos; u_pos++) {
                int u = chunk_start + u_pos;
                int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                unsigned short tri_bits = tri_coeff_bits[pair_idx];
                __nv_bfloat16 tri_bf16 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits);
                float tri_coeff = bf16_to_float(tri_bf16);
                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff * x_u;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
                }
            }

            unsigned long long d_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            float d_skip = D_val * x_val;
            float y = d_skip + prior_chunk_state + local_hf_chunk_scan;
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - d_t0);
            }
            unsigned long long cast_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            out[flat_idx] = float_to_bf16(y);
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 3, clock64() - cast_t0);
            }
            if (term_probe != NULL && t == probe_row && head == probe_head && d == probe_d) {
                term_probe[0] = entry_dot;
                term_probe[1] = __expf(dA_local_target);
                term_probe[2] = entry_dot * __expf(dA_local_target);
                term_probe[3] = prior_chunk_state;
                term_probe[4] = local_hf_chunk_scan;
                term_probe[5] = d_skip;
                term_probe[6] = y;
                term_probe[7] = 0.0f;
                term_probe[8] = (float)chunk_idx;
                term_probe[9] = (float)chunk_start;
                term_probe[10] = (float)t_pos;
                term_probe[11] = (float)group;
                term_probe[12] = dA_target;
                term_probe[13] = x_val;
                term_probe[14] = D_val;
                term_probe[15] = 1.0f;
                term_probe[16] = 0.0f;
                term_probe[17] = 0.0f;
                term_probe[18] = -1.0f;
                term_probe[19] = 0.0f;
                term_probe[20] = 0.0f;
                term_probe[21] = 0.0f;
                term_probe[22] = 0.0f;
                term_probe[23] = 0.0f;
                term_probe[24] = (float)(t_pos + 1);
                term_probe[25] = 5.0f;
            }
        }
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 4, clock64() - thread_total_t0);
    }
}

extern "C" __global__ void mamba2_ssd_parallel_chunk_scan_output_state_split_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    const float* __restrict__ A_log,        /* [n_heads] */
    const float* __restrict__ dt_out,       /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ dA_cumsum,    /* [n_chunks, n_heads, chunk_size] */
    const float* __restrict__ entry_state,  /* [n_chunks, n_heads, head_dim, state_size] */
    const float* __restrict__ c_state_total_exact,/* [L, n_heads, head_dim] */
    float* __restrict__ term_probe,         /* optional [32], NULL when disabled */
    unsigned long long* __restrict__ subloop_timing,/* optional [10], NULL when disabled */
    int probe_row,
    int probe_head,
    int probe_d,
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int chunk_size,
    int d_tile,
    int state_lanes)
{
    int head = blockIdx.x;
    int chunk_idx = blockIdx.z;
    if (head >= n_heads || chunk_size <= 0 || d_tile <= 0 || state_lanes <= 0) return;

    int chunk_start = chunk_idx * chunk_size;
    if (chunk_start >= L) return;
    int chunk_len = min(chunk_size, L - chunk_start);
    if (chunk_len <= 0) return;

    int lane = threadIdx.x;
    int d_in_tile = lane / state_lanes;
    int state_lane = lane - d_in_tile * state_lanes;
    int d = blockIdx.y * d_tile + d_in_tile;
    bool valid_d = d_in_tile < d_tile && d < head_dim;

    int group = head / (n_heads / n_groups);
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;
    (void)A_log;
    (void)c_state_total_exact;
    int pair_count = (chunk_len * (chunk_len + 1)) / 2;
    int max_pair_count = (chunk_size * (chunk_size + 1)) / 2;
    extern __shared__ unsigned char parallel_chunk_scan_smem[];
    unsigned short* tri_coeff_bits = reinterpret_cast<unsigned short*>(parallel_chunk_scan_smem);
    int tri_bytes = max_pair_count * (int)sizeof(unsigned short);
    __shared__ float dA_chunk_base_shared;
    unsigned long long thread_total_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;

    if (threadIdx.x == 0) {
        float dA_chunk_base = 0.0f;
        for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
            int prev_start = prev_chunk * chunk_size;
            int prev_len = min(chunk_size, L - prev_start);
            if (prev_len > 0) {
                int64_t prev_base = ((int64_t)prev_chunk * n_heads + head) * chunk_size;
                dA_chunk_base += dA_cumsum[prev_base + prev_len - 1];
            }
        }
        dA_chunk_base_shared = dA_chunk_base;
    }
    __syncthreads();
    float dA_chunk_base = dA_chunk_base_shared;
    int64_t chunk_base = ((int64_t)chunk_idx * n_heads + head) * chunk_size;

    unsigned long long coeff_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        float dA_local_target = dA_cumsum[chunk_base + t_pos];
        float dA_target = dA_chunk_base + dA_local_target;
        int t_abs = chunk_start + t_pos;
        int c_row_base_idx = (t_abs * n_groups + group) * state_size;
        for (int u_pos = threadIdx.x; u_pos <= t_pos; u_pos += blockDim.x) {
            int u_abs = chunk_start + u_pos;
            int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
            float dA_running = dA_chunk_base + dA_cumsum[chunk_base + u_pos];
            float dt_u = dt_out[chunk_base + u_pos];
            float decay = (u_pos == t_pos) ? 1.0f : __expf(fminf(dA_target - dA_running, 0.0f));
            int b_row_base_idx = (u_abs * n_groups + group) * state_size;
            float cb = mamba2_ssd_cb_dot_reverse(
                C_mat, B_mat, c_row_base_idx, b_row_base_idx, state_size);
            float scale = decay * dt_u;
            __nv_bfloat16 tri_bf16 = float_to_bf16(cb * scale);
            tri_coeff_bits[pair_idx] = *reinterpret_cast<unsigned short*>(&tri_bf16);
        }
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 7, clock64() - coeff_t0);
        if (threadIdx.x == 0) {
            atomicAdd(&subloop_timing[8], (unsigned long long)pair_count);
        }
    }
    __syncthreads();

    for (int t_pos = 0; t_pos < chunk_len; t_pos++) {
        int t = chunk_start + t_pos;
        float dA_local_target = dA_cumsum[chunk_base + t_pos];
        float dA_target = dA_chunk_base + dA_local_target;
        float x_val = valid_d ? bf16_to_float(x[(t * n_heads + head) * head_dim + d]) : 0.0f;
        unsigned long long prior_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
        int c_row_base_idx = (t * n_groups + group) * state_size;
        int64_t entry_base =
            (((int64_t)chunk_idx * n_heads + head) * head_dim + d) * state_size;

        if (valid_d && state_lane == 0) {
            int flat_idx = (t * n_heads + head) * head_dim + d;
            float entry_dot = 0.0f;
            if (chunk_start != 0) {
                for (int s = 0; s < state_size; s++) {
                    float C_val = bf16_to_float(C_mat[c_row_base_idx + s]);
                    entry_dot += C_val * entry_state[entry_base + s];
                }
            }
            float prior_chunk_state = (chunk_start == 0) ? 0.0f : __expf(dA_local_target) * entry_dot;
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - prior_t0);
                atomicAdd(&subloop_timing[5], 1ULL);
                atomicAdd(&subloop_timing[6], (unsigned long long)(t_pos + 1));
            }

            float local_hf_chunk_scan = 0.0f;
            for (int u_pos = 0; u_pos <= t_pos; u_pos++) {
                int u = chunk_start + u_pos;
                int pair_idx = mamba2_ssd_lower_tri_pair_index(t_pos, u_pos);
                unsigned short tri_bits = tri_coeff_bits[pair_idx];
                __nv_bfloat16 tri_bf16 = *reinterpret_cast<__nv_bfloat16*>(&tri_bits);
                float tri_coeff = bf16_to_float(tri_bf16);
                float x_u = bf16_to_float(x[(u * n_heads + head) * head_dim + d]);
                unsigned long long tri_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
                local_hf_chunk_scan += tri_coeff * x_u;
                if (subloop_timing != NULL) {
                    mamba2_ssd_timing_add(subloop_timing, 2, clock64() - tri_t0);
                }
            }

            unsigned long long d_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            float d_skip = D_val * x_val;
            float y = d_skip + prior_chunk_state + local_hf_chunk_scan;
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 0, clock64() - d_t0);
            }
            unsigned long long cast_t0 = (subloop_timing != NULL) ? clock64() : 0ULL;
            out[flat_idx] = float_to_bf16(y);
            if (subloop_timing != NULL) {
                mamba2_ssd_timing_add(subloop_timing, 3, clock64() - cast_t0);
            }
            if (term_probe != NULL && t == probe_row && head == probe_head && d == probe_d) {
                term_probe[0] = entry_dot;
                term_probe[1] = __expf(dA_local_target);
                term_probe[2] = entry_dot * __expf(dA_local_target);
                term_probe[3] = prior_chunk_state;
                term_probe[4] = local_hf_chunk_scan;
                term_probe[5] = d_skip;
                term_probe[6] = y;
                term_probe[7] = 0.0f;
                term_probe[8] = (float)chunk_idx;
                term_probe[9] = (float)chunk_start;
                term_probe[10] = (float)t_pos;
                term_probe[11] = (float)group;
                term_probe[12] = dA_target;
                term_probe[13] = x_val;
                term_probe[14] = D_val;
                term_probe[15] = 1.0f;
                term_probe[16] = 0.0f;
                term_probe[17] = 0.0f;
                term_probe[18] = -1.0f;
                term_probe[19] = 0.0f;
                term_probe[20] = 0.0f;
                term_probe[21] = 0.0f;
                term_probe[22] = 0.0f;
                term_probe[23] = 0.0f;
                term_probe[24] = (float)(t_pos + 1);
                term_probe[25] = 4.0f;
            }
        }
        __syncthreads();
    }
    if (subloop_timing != NULL) {
        mamba2_ssd_timing_add(subloop_timing, 4, clock64() - thread_total_t0);
    }
}

extern "C" void krasis_mamba2_ssd_fwd(
    void* out, const void* x, const void* dt,
    const void* A_log, const void* B_mat, const void* C_mat,
    const void* D_vec, void* ssm_state,
    int B_batch, int L, int n_heads, int head_dim, int state_size,
    int n_groups, int chunk_size,
    const void* dt_bias, float dt_softplus_flag,
    void* stream)
{
    if (L == 0) return;
    (void)B_batch;  /* always 1 for inference */
    /* SSM state must be in float32 for numerical stability */
    /* Caller provides float32 ssm_state: [n_heads, head_dim, state_size] */

    int threads = min(256, head_dim);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    int blocks_d = (head_dim + threads - 1) / threads;
    dim3 grid(n_heads, blocks_d);
    int effective_chunk_size = (chunk_size > 0) ? chunk_size : L;
    size_t shared_mem_bytes = (size_t)effective_chunk_size * 2 * sizeof(float);

    mamba2_ssd_sequential_kernel<<<grid, threads, shared_mem_bytes, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)dt, (const float*)A_log,
        (const __nv_bfloat16*)B_mat, (const __nv_bfloat16*)C_mat,
        (const float*)D_vec, (float*)ssm_state,
        (const float*)dt_bias,
        L, n_heads, head_dim, state_size, n_groups,
        chunk_size,
        dt_softplus_flag > 0.5f ? 1 : 0);
}

/* ── MoE Gather: collect tokens by expert ID ─────────────────────────── */

/*
 * Given topk_ids [M, topk] and hidden [M, D] (bf16), gather tokens into
 * per-expert contiguous batches.
 *
 * expert_offsets [E+1]: exclusive prefix sum of tokens per expert (host-computed).
 * expert_token_map [M*topk]: maps each (token,k) slot to its position in the
 *   gathered buffer. Computed on host: for each (t,k) where topk_ids[t,k]==e,
 *   expert_token_map[t*topk+k] = expert_offsets[e] + count_e++.
 *
 * gathered [total_active, D] (bf16): output.
 * gather_src_map [total_active]: maps each row in gathered to source token index.
 *
 * Grid: (total_active, 1, 1), Block: (min(1024, D_padded32), 1, 1)
 */
extern "C" __global__ void moe_gather_kernel(
    __nv_bfloat16* __restrict__ gathered,
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ gather_src_map,
    int total_active,
    int D)
{
    int row = blockIdx.x;
    if (row >= total_active) return;
    int src_token = gather_src_map[row];
    const __nv_bfloat16* src = hidden + (int64_t)src_token * D;
    __nv_bfloat16* dst = gathered + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/* ── MoE Scatter + Accumulate (FP32 accumulator) ────────────────────── */

/*
 * Scatter expert outputs back and accumulate with routing weights.
 *
 * expert_out [total_active, D] (bf16): expert GEMM outputs.
 * accum [M, D] (fp32): accumulator (zero-initialized before MoE).
 *   Deterministic: each block owns one destination token and column tile, scans
 *   total_active rows in row order, and writes the sum once. Caller converts to
 *   BF16 after scatter completes.
 * gather_src_map [total_active]: source token index for each gathered row.
 * gather_weight_map [total_active]: routing weight (fp32) for each gathered row.
 *
 * Grid: (M, ceil(D / blockDim.x), 1), Block: (columns, 1, 1)
 */
extern "C" __global__ void moe_scatter_add_kernel(
    float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ expert_out,
    const int* __restrict__ gather_src_map,
    const float* __restrict__ gather_weight_map,
    int total_active,
    int D)
{
    int dst_token = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= D) return;

    float sum = 0.0f;
    for (int row = 0; row < total_active; row++) {
        if (gather_src_map[row] == dst_token) {
            float w = gather_weight_map[row];
            float val = bf16_to_float(expert_out[(int64_t)row * D + col]);
            sum += val * w;
        }
    }
    accum[(int64_t)dst_token * D + col] = sum;
}

/* ── MoE Zero Accumulator ────────────────────────────────────────────── */

/* Zero an FP32 buffer. Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_zero_accum_kernel(
    float* __restrict__ buf,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float* r = buf + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        r[i] = 0.0f;
    }
}

/* ── MoE Add Shared Expert ───────────────────────────────────────────── */

/* Add shared expert output (BF16) to FP32 MoE accumulator.
 * Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_add_shared_kernel(
    float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ shared_out,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float* a = accum + (int64_t)row * D;
    const __nv_bfloat16* s = shared_out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        a[i] += bf16_to_float(s[i]);
    }
}

/* ── MoE Add Shared (gated) ──────────────────────────────────────────── */

/* Same as moe_add_shared, but applies sigmoid gating: accum += sigmoid(gate[row]) * shared_out[row]
 * gate_values is [M] FP32 (output of hidden @ gate_weight GEMM). */
extern "C" __global__ void moe_add_shared_gated_kernel(
    float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_values,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float gate = 1.0f / (1.0f + __expf(-gate_values[row]));
    float* a = accum + (int64_t)row * D;
    const __nv_bfloat16* s = shared_out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        a[i] += gate * bf16_to_float(s[i]);
    }
}

/* ── MoE FP32 Accum -> BF16 output ──────────────────────────────────── */

/* Convert FP32 accumulator to BF16 output.
 * Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_accum_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ accum,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    __nv_bfloat16* o = out + (int64_t)row * D;
    const float* a = accum + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        o[i] = float_to_bf16(a[i]);
    }
}

/* ── MoE FP32 Accum -> BF16-rounded FP32 in-place ──────────────────────── */

/* Match BF16 module-boundary semantics while preserving the FP32 accumulator
 * storage used by the shared-expert add kernel.
 * Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_round_accum_bf16_inplace_kernel(
    float* __restrict__ accum,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float* a = accum + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        a[i] = bf16_to_float(float_to_bf16(a[i]));
    }
}

/* ── FP32 -> BF16 batch convert ──────────────────────────────────────── */

/* Convert FP32 buffer to BF16 (for cuBLAS output conversion).
 * Grid: (M, 1, 1), Block: (min(1024, N_padded32), 1, 1) */
extern "C" __global__ void fp32_to_bf16_batch_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const float* src = in + (int64_t)row * N;
    __nv_bfloat16* dst = out + (int64_t)row * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        dst[i] = float_to_bf16(src[i]);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  LINEAR ATTENTION (Gated DeltaNet) PREFILL KERNELS
 *
 *  These kernels implement the batched (multi-token) linear attention
 *  prefill path used by QCN (36/48 layers).
 *
 *  Algorithm: Gated Delta Rule with chunked parallel formulation.
 *  Reference: linear_attention.py _forward_chunked()
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── Uninterleave QKVZ (batched) ─────────────────────────────────────── */
/*
 * in_proj_qkvz output is interleaved per key-head group:
 *   [M, nk * group_dim]  where group_dim = 2*dk + 2*hr*dv
 * Within each key-head group:
 *   [dk (q), dk (k), hr*dv (v), hr*dv (z)]
 *
 * Output layout:
 *   q_out: [M, nk, dk]   (BF16)
 *   k_out: [M, nk, dk]   (BF16)
 *   v_out: [M, nv, dv]   (BF16)
 *   z_out: [M, nv, dv]   (BF16)
 *
 * Grid: (M, 1, 1), Block: (min(1024, nk*group_dim_padded), 1, 1)
 */
extern "C" __global__ void la_uninterleave_qkvz_kernel(
    __nv_bfloat16* __restrict__ q_out,    /* [M, nk*dk] */
    __nv_bfloat16* __restrict__ k_out,    /* [M, nk*dk] */
    __nv_bfloat16* __restrict__ v_out,    /* [M, nv*dv] */
    __nv_bfloat16* __restrict__ z_out,    /* [M, nv*dv] */
    const __nv_bfloat16* __restrict__ qkvz, /* [M, nk*group_dim] */
    int nk, int dk, int hr, int dv)
{
    int token = blockIdx.x;
    int group_dim = 2 * dk + 2 * hr * dv;
    int total = nk * group_dim;
    int key_dim = nk * dk;
    int nv = nk * hr;

    const __nv_bfloat16* src = qkvz + (int64_t)token * total;
    __nv_bfloat16* q_dst = q_out + (int64_t)token * key_dim;
    __nv_bfloat16* k_dst = k_out + (int64_t)token * key_dim;
    __nv_bfloat16* v_dst = v_out + (int64_t)token * (nv * dv);
    __nv_bfloat16* z_dst = z_out + (int64_t)token * (nv * dv);

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int head_group = i / group_dim;
        int offset = i % group_dim;
        __nv_bfloat16 val = src[i];

        if (offset < dk) {
            /* q: first dk elements */
            q_dst[head_group * dk + offset] = val;
        } else if (offset < 2 * dk) {
            /* k: next dk elements */
            k_dst[head_group * dk + (offset - dk)] = val;
        } else if (offset < 2 * dk + hr * dv) {
            /* v: next hr*dv elements -> reshape to [nv, dv] */
            int v_offset = offset - 2 * dk;
            int v_sub_head = v_offset / dv;
            int v_elem = v_offset % dv;
            int v_head = head_group * hr + v_sub_head;
            v_dst[v_head * dv + v_elem] = val;
        } else {
            /* z: last hr*dv elements -> reshape to [nv, dv] */
            int z_offset = offset - 2 * dk - hr * dv;
            int z_sub_head = z_offset / dv;
            int z_elem = z_offset % dv;
            int z_head = head_group * hr + z_sub_head;
            z_dst[z_head * dv + z_elem] = val;
        }
    }
}

/* ── Uninterleave BA (batched) ───────────────────────────────────────── */
/*
 * in_proj_ba output: [M, nk * 2*hr] interleaved per key-head group:
 *   [hr (b), hr (a)] per key head
 *
 * Output:
 *   b_out: [M, nv] (BF16)
 *   a_out: [M, nv] (BF16)
 *
 * Grid: (M, 1, 1), Block: (min(1024, nk*2*hr_padded), 1, 1)
 */
extern "C" __global__ void la_uninterleave_ba_kernel(
    __nv_bfloat16* __restrict__ b_out,    /* [M, nv] */
    __nv_bfloat16* __restrict__ a_out,    /* [M, nv] */
    const __nv_bfloat16* __restrict__ ba, /* [M, nk * 2*hr] */
    int nk, int hr)
{
    int token = blockIdx.x;
    int ba_group = 2 * hr;
    int total = nk * ba_group;
    int nv = nk * hr;

    const __nv_bfloat16* src = ba + (int64_t)token * total;
    __nv_bfloat16* b_dst = b_out + (int64_t)token * nv;
    __nv_bfloat16* a_dst = a_out + (int64_t)token * nv;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int head_group = i / ba_group;
        int offset = i % ba_group;
        __nv_bfloat16 val = src[i];

        if (offset < hr) {
            b_dst[head_group * hr + offset] = val;
        } else {
            a_dst[head_group * hr + (offset - hr)] = val;
        }
    }
}

/* ── Depthwise Conv1d + SiLU (batched over full sequence) ────────────── */
/*
 * Causal depthwise conv1d with SiLU activation over a full sequence.
 * Input: [conv_dim, M] (concatenated q_flat, k_flat, v_flat per token)
 * Conv state: [conv_dim, kernel_dim] (left-padded context)
 * Weight: [conv_dim, kernel_dim] (per-channel conv weights, no bias)
 * Output: [conv_dim, M] after conv + SiLU
 *
 * Also updates conv_state to last kernel_dim columns of input.
 *
 * Grid: (conv_dim, 1, 1), Block: (min(1024, M_padded), 1, 1)
 */
extern "C" __global__ void la_depthwise_conv1d_silu_kernel(
    float* __restrict__ output,          /* [conv_dim, M] FP32 */
    float* __restrict__ conv_state,      /* [conv_dim, kernel_dim] FP32, updated */
    const __nv_bfloat16* __restrict__ input, /* [M, conv_dim] BF16, row-major */
    const float* __restrict__ weight,    /* [conv_dim, kernel_dim] FP32 */
    int M, int conv_dim, int kernel_dim)
{
    int ch = blockIdx.x;
    if (ch >= conv_dim) return;

    float* st = conv_state + (int64_t)ch * kernel_dim;
    const float* wt = weight + (int64_t)ch * kernel_dim;
    float* out = output + (int64_t)ch * M;

    /* Process each token position */
    for (int t = threadIdx.x; t < M; t += blockDim.x) {
        float acc = 0.0f;
        for (int w = 0; w < kernel_dim; w++) {
            /* Position in the padded sequence: state has kernel_dim columns,
               then input has M columns. We want position (t + w) in this
               concatenated view, reading from right to left for the filter. */
            int src_pos = t + w - (kernel_dim - 1);
            float val;
            if (src_pos < 0) {
                /* Read from conv state (left padding) */
                val = st[kernel_dim + src_pos];  /* src_pos is negative */
            } else {
                /* Read from input: input is [M, conv_dim] row-major,
                   so element at position src_pos, channel ch is input[src_pos * conv_dim + ch] */
                val = bf16_to_float(input[src_pos * conv_dim + ch]);
            }
            acc += val * wt[w];
        }
        /* SiLU activation: x * sigmoid(x) */
        float sig = 1.0f / (1.0f + __expf(-acc));
        out[t] = acc * sig;
    }

    /* Update conv state with last kernel_dim tokens.
       Wait for all threads to finish reading before overwriting state. */
    __syncthreads();
    for (int w = threadIdx.x; w < kernel_dim; w += blockDim.x) {
        int src_pos = M - kernel_dim + w;
        if (src_pos < 0) {
            /* Still from old state */
            st[w] = st[kernel_dim + src_pos];
        } else {
            st[w] = bf16_to_float(input[src_pos * conv_dim + ch]);
        }
    }
}

/* ── L2 Norm per head (batched) ──────────────────────────────────────── */
/*
 * L2-normalize each head vector and optionally scale.
 * x: [M, num_heads, dim] FP32, in-place
 * Scale is applied after normalization: out = x / ||x|| * scale
 *
 * Grid: (M, num_heads, 1), Block: (min(256, dim_padded32), 1, 1)
 */
extern "C" __global__ void la_l2norm_per_head_kernel(
    float* __restrict__ x,     /* [M, num_heads, dim] in-place */
    float scale,
    int num_heads, int dim)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    float* vec = x + ((int64_t)token * num_heads + head) * dim;

    extern __shared__ float smem[];

    /* Compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = vec[i];
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(smem[0] + 1e-6f);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] = vec[i] * inv_norm * scale;
    }
}

/* ── Compute gate and beta (batched) ─────────────────────────────────── */
/*
 * beta = sigmoid(b)
 * gate = -exp(A_log) * softplus(a + dt_bias)
 *
 * b_in: [M, nv] BF16 (from uninterleave)
 * a_in: [M, nv] BF16 (from uninterleave)
 * A_log: [nv] FP32 (model parameter)
 * dt_bias: [nv] FP32 (model parameter)
 *
 * beta_out: [M, nv] FP32
 * gate_out: [M, nv] FP32
 *
 * Grid: (M, 1, 1), Block: (min(1024, nv_padded32), 1, 1)
 */
extern "C" __global__ void la_compute_gate_beta_kernel(
    float* __restrict__ beta_out,   /* [M, nv] FP32 */
    float* __restrict__ gate_out,   /* [M, nv] FP32 */
    const __nv_bfloat16* __restrict__ b_in,  /* [M, nv] BF16 */
    const __nv_bfloat16* __restrict__ a_in,  /* [M, nv] BF16 */
    const float* __restrict__ A_log,         /* [nv] FP32 */
    const float* __restrict__ dt_bias,       /* [nv] FP32 */
    int nv)
{
    int token = blockIdx.x;
    const __nv_bfloat16* b_row = b_in + (int64_t)token * nv;
    const __nv_bfloat16* a_row = a_in + (int64_t)token * nv;
    float* beta_row = beta_out + (int64_t)token * nv;
    float* gate_row = gate_out + (int64_t)token * nv;

    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        float b = bf16_to_float(b_row[i]);
        float a = bf16_to_float(a_row[i]);

        /* beta = sigmoid(b) */
        beta_row[i] = 1.0f / (1.0f + __expf(-b));

        /* gate = -exp(A_log) * softplus(a + dt_bias) */
        float a_val = __expf(A_log[i]);
        float x_sp = a + dt_bias[i];
        float sp = (x_sp > 20.0f) ? x_sp : logf(1.0f + __expf(x_sp));
        gate_row[i] = -a_val * sp;
    }
}

/* ── Repeat-interleave heads (batched) ───────────────────────────────── */
/*
 * Expand nk key heads to nv value heads (each key head repeated hr times).
 * input: [M, nk, dim] FP32
 * output: [M, nv, dim] FP32 where nv = nk * hr
 *
 * Grid: (M, nv, 1), Block: (min(256, dim_padded32), 1, 1)
 */
extern "C" __global__ void la_repeat_interleave_kernel(
    float* __restrict__ output,      /* [M, nv, dim] */
    const float* __restrict__ input, /* [M, nk, dim] */
    int nk, int dim, int hr)
{
    int token = blockIdx.x;
    int v_head = blockIdx.y;
    int k_head = v_head / hr;

    const float* src = input + ((int64_t)token * nk + k_head) * dim;
    float* dst = output + ((int64_t)token * (nk * hr) + v_head) * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/* ── BF16 to FP32 batched (for conv1d input transpose) ──────────────── */
/*
 * Convert [M, D] BF16 to [M, D] FP32
 * Grid: (M, 1, 1), Block: (min(1024, D_padded), 1, 1)
 */
extern "C" __global__ void la_bf16_to_fp32_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int D)
{
    int row = blockIdx.x;
    const __nv_bfloat16* src = in + (int64_t)row * D;
    float* dst = out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = bf16_to_float(src[i]);
    }
}

/* ── FP32 to BF16 batched ───────────────────────────────────────────── */
extern "C" __global__ void la_fp32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int D)
{
    int row = blockIdx.x;
    const float* src = in + (int64_t)row * D;
    __nv_bfloat16* dst = out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = float_to_bf16(src[i]);
    }
}

/* ── Gated RMSNorm (batched) ─────────────────────────────────────────── */
/*
 * Gated RMSNorm: out = rmsnorm(x) * weight * silu(gate)
 *
 * x: [M, nv, dv] FP32 (attention output)
 * gate: [M, nv, dv] BF16 (z from uninterleave)
 * weight: [dv] BF16 (per-head norm weight, shared across heads)
 * out: [M, nv*dv] BF16
 *
 * Grid: (M, nv, 1), Block: (min(256, dv_padded32), 1, 1)
 */
extern "C" __global__ void la_gated_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,      /* [M, nv*dv] BF16 */
    const float* __restrict__ x,          /* [M, nv, dv] FP32 */
    const __nv_bfloat16* __restrict__ gate, /* [M, nv, dv] BF16 */
    const float* __restrict__ weight,     /* [dv] FP32 */
    int nv, int dv, float eps)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    const float* x_head = x + ((int64_t)token * nv + head) * dv;
    const __nv_bfloat16* g_head = gate + ((int64_t)token * nv + head) * dv;
    __nv_bfloat16* o_head = out + ((int64_t)token * nv + head) * dv;

    extern __shared__ float smem[];

    /* Compute variance */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float v = x_head[i];
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)dv + eps);

    /* Normalize, scale by weight, multiply by silu(gate) */
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float normed = x_head[i] * rms_inv * weight[i];
        float g = bf16_to_float(g_head[i]);
        float silu_g = g / (1.0f + __expf(-g));
        o_head[i] = float_to_bf16(normed * silu_g);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Optimized BF16 LA pipeline — fused kernels for FLA path
 * ═══════════════════════════════════════════════════════════════════════
 *
 * These replace the old 4-kernel conv pipeline (concat + conv + transpose + split)
 * and the FP32 intermediate stages with an all-BF16 path.
 * Used only when FLA is available. Non-FLA fallback uses the old FP32 path.
 */

/* ── Fused Conv1d+SiLU (BF16 in, BF16 out) ──────────────────────────── */
/*
 * Replaces: concat_3_bf16 + la_depthwise_conv1d_silu + transpose + la_split_conv_output
 *
 * Reads q,k,v as separate BF16 inputs (no concat step).
 * Applies depthwise conv1d + SiLU per channel.
 * Writes q,k,v as separate BF16 outputs (no transpose/split step).
 *
 * Each thread processes channels in a coalesced pattern (adjacent threads
 * read adjacent channels within same token = adjacent memory addresses).
 *
 * Grid: (M, 1, 1), Block: (min(1024, conv_dim_pad32), 1, 1)
 */
extern "C" __global__ void la_fused_conv1d_silu_bf16_kernel(
    __nv_bfloat16* __restrict__ q_out,       /* [M, key_dim] BF16 */
    __nv_bfloat16* __restrict__ k_out,       /* [M, key_dim] BF16 */
    __nv_bfloat16* __restrict__ v_out,       /* [M, value_dim] BF16 */
    const float* __restrict__ conv_state,    /* [conv_dim, kernel_dim] FP32 */
    const __nv_bfloat16* __restrict__ q_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ k_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ v_in,  /* [M, value_dim] BF16 */
    const float* __restrict__ weight,        /* [conv_dim, kernel_dim] FP32 */
    int M, int key_dim, int value_dim, int kernel_dim)
{
    int token = blockIdx.x;
    int conv_dim = 2 * key_dim + value_dim;

    for (int ch = threadIdx.x; ch < conv_dim; ch += blockDim.x) {
        const float* wt = weight + (int64_t)ch * kernel_dim;

        float acc = 0.0f;
        for (int w = 0; w < kernel_dim; w++) {
            int src_pos = token + w - (kernel_dim - 1);
            float val;
            if (src_pos < 0) {
                /* Read from conv state (left padding) */
                val = conv_state[(int64_t)ch * kernel_dim + (kernel_dim + src_pos)];
            } else {
                /* Read from appropriate BF16 input buffer */
                if (ch < key_dim) {
                    val = bf16_to_float(q_in[(int64_t)src_pos * key_dim + ch]);
                } else if (ch < 2 * key_dim) {
                    val = bf16_to_float(k_in[(int64_t)src_pos * key_dim + (ch - key_dim)]);
                } else {
                    val = bf16_to_float(v_in[(int64_t)src_pos * value_dim + (ch - 2 * key_dim)]);
                }
            }
            acc += val * wt[w];
        }

        /* SiLU activation */
        float sig = 1.0f / (1.0f + __expf(-acc));
        float result = acc * sig;

        /* Write to appropriate BF16 output buffer */
        if (ch < key_dim) {
            q_out[(int64_t)token * key_dim + ch] = float_to_bf16(result);
        } else if (ch < 2 * key_dim) {
            k_out[(int64_t)token * key_dim + (ch - key_dim)] = float_to_bf16(result);
        } else {
            v_out[(int64_t)token * value_dim + (ch - 2 * key_dim)] = float_to_bf16(result);
        }
    }
}

extern "C" void krasis_la_fused_conv1d_silu_bf16(
    void* q_out, void* k_out, void* v_out,
    const void* conv_state,
    const void* q_in, const void* k_in, const void* v_in,
    const void* weight,
    int M, int key_dim, int value_dim, int kernel_dim, void* stream)
{
    if (M == 0) return;
    int conv_dim = 2 * key_dim + value_dim;
    int threads = min(1024, ((conv_dim + 31) / 32) * 32);
    la_fused_conv1d_silu_bf16_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        (const float*)conv_state,
        (const __nv_bfloat16*)q_in, (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)v_in,
        (const float*)weight, M, key_dim, value_dim, kernel_dim);
}

/* ── Update conv state after fused conv ──────────────────────────────── */
/*
 * Copies the last kernel_dim token positions into conv_state.
 * Tiny kernel (conv_dim * kernel_dim = ~32K values).
 *
 * Grid: (conv_dim, 1, 1), Block: (kernel_dim, 1, 1)
 */
extern "C" __global__ void la_update_conv_state_kernel(
    float* __restrict__ conv_state,          /* [conv_dim, kernel_dim] FP32 */
    const __nv_bfloat16* __restrict__ q_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ k_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ v_in,  /* [M, value_dim] BF16 */
    int M, int key_dim, int value_dim, int kernel_dim)
{
    int ch = blockIdx.x;
    int conv_dim = 2 * key_dim + value_dim;
    if (ch >= conv_dim) return;

    float* st = conv_state + (int64_t)ch * kernel_dim;

    for (int w = threadIdx.x; w < kernel_dim; w += blockDim.x) {
        int src_pos = M - kernel_dim + w;
        if (src_pos < 0) {
            /* Still from old state */
            st[w] = st[kernel_dim + src_pos];
        } else {
            if (ch < key_dim) {
                st[w] = bf16_to_float(q_in[(int64_t)src_pos * key_dim + ch]);
            } else if (ch < 2 * key_dim) {
                st[w] = bf16_to_float(k_in[(int64_t)src_pos * key_dim + (ch - key_dim)]);
            } else {
                st[w] = bf16_to_float(v_in[(int64_t)src_pos * value_dim + (ch - 2 * key_dim)]);
            }
        }
    }
}

extern "C" void krasis_la_update_conv_state(
    void* conv_state,
    const void* q_in, const void* k_in, const void* v_in,
    int M, int key_dim, int value_dim, int kernel_dim, void* stream)
{
    if (M == 0) return;
    int conv_dim = 2 * key_dim + value_dim;
    int threads = min(32, ((kernel_dim + 31) / 32) * 32);
    la_update_conv_state_kernel<<<conv_dim, threads, 0, (cudaStream_t)stream>>>(
        (float*)conv_state,
        (const __nv_bfloat16*)q_in, (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)v_in,
        M, key_dim, value_dim, kernel_dim);
}

/* ── Gate/Beta computation with BF16 output ──────────────────────────── */
/*
 * Same as la_compute_gate_beta but outputs BF16 for direct FLA consumption.
 * Eliminates 2 FP32->BF16 conversion kernels.
 */
extern "C" __global__ void la_compute_gate_beta_bf16_kernel(
    __nv_bfloat16* __restrict__ beta_out,   /* [M, nv] BF16 */
    __nv_bfloat16* __restrict__ gate_out,   /* [M, nv] BF16 */
    const __nv_bfloat16* __restrict__ b_in, /* [M, nv] BF16 */
    const __nv_bfloat16* __restrict__ a_in, /* [M, nv] BF16 */
    const float* __restrict__ A_log,        /* [nv] FP32 */
    const float* __restrict__ dt_bias,      /* [nv] FP32 */
    int nv)
{
    int token = blockIdx.x;
    const __nv_bfloat16* b_row = b_in + (int64_t)token * nv;
    const __nv_bfloat16* a_row = a_in + (int64_t)token * nv;
    __nv_bfloat16* beta_row = beta_out + (int64_t)token * nv;
    __nv_bfloat16* gate_row = gate_out + (int64_t)token * nv;

    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        float b = bf16_to_float(b_row[i]);
        float a = bf16_to_float(a_row[i]);
        /* beta = sigmoid(b) */
        beta_row[i] = float_to_bf16(1.0f / (1.0f + __expf(-b)));
        /* gate = -exp(A_log) * softplus(a + dt_bias)
         * Match the FP32 path's stable softplus to avoid BF16 fast-path overflow.
         */
        float x_sp = a + dt_bias[i];
        float sp = (x_sp > 20.0f) ? x_sp : logf(1.0f + __expf(x_sp));
        gate_row[i] = float_to_bf16(-__expf(A_log[i]) * sp);
    }
}

/* ── Fused Repeat-Interleave + L2 Norm (BF16) ───────────────────────── */
/*
 * Combines repeat_interleave (nk -> nv heads) with L2 normalization.
 * One pass: read from [M, nk, dk], normalize in FP32, write to [M, nv, dk] BF16.
 * Eliminates separate repeat_interleave + l2norm + their intermediate buffers.
 *
 * Grid: (M, nv, 1), Block: (min(256, dk_pad32), 1, 1)
 */
extern "C" __global__ void la_fused_repeat_l2norm_bf16_kernel(
    __nv_bfloat16* __restrict__ output,     /* [M, nv, dk] BF16 */
    const __nv_bfloat16* __restrict__ input, /* [M, nk, dk] BF16 */
    int nk, int dk, int hr, float scale)
{
    int token = blockIdx.x;
    int v_head = blockIdx.y;
    int k_head = v_head / hr;

    const __nv_bfloat16* src = input + ((int64_t)token * nk + k_head) * dk;
    __nv_bfloat16* dst = output + ((int64_t)token * (nk * hr) + v_head) * dk;

    extern __shared__ float smem[];

    /* Load values and compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float v = bf16_to_float(src[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    /* Tree reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(smem[0] + 1e-6f);

    /* Normalize, scale, and write BF16 */
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float v = bf16_to_float(src[i]);
        dst[i] = float_to_bf16(v * inv_norm * scale);
    }
}

/* ── Gated RMSNorm with BF16 input ──────────────────────────────────── */
/*
 * Same as la_gated_rmsnorm but reads x as BF16 (FLA output) instead of FP32.
 * Eliminates the BF16->FP32 conversion after FLA.
 *
 * x: [M, nv, dv] BF16 (FLA output)
 * gate: [M, nv, dv] BF16 (z from uninterleave)
 * weight: [dv] FP32
 * out: [M, nv*dv] BF16
 */
extern "C" __global__ void la_gated_rmsnorm_bf16in_kernel(
    __nv_bfloat16* __restrict__ out,          /* [M, nv*dv] BF16 */
    const __nv_bfloat16* __restrict__ x,      /* [M, nv, dv] BF16 */
    const __nv_bfloat16* __restrict__ gate,   /* [M, nv, dv] BF16 */
    const float* __restrict__ weight,         /* [dv] FP32 */
    int nv, int dv, float eps)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    const __nv_bfloat16* x_head = x + ((int64_t)token * nv + head) * dv;
    const __nv_bfloat16* g_head = gate + ((int64_t)token * nv + head) * dv;
    __nv_bfloat16* o_head = out + ((int64_t)token * nv + head) * dv;

    extern __shared__ float smem[];

    /* Compute variance in FP32 */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float v = bf16_to_float(x_head[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)dv + eps);

    /* Normalize, scale by weight, multiply by silu(gate) */
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float normed = bf16_to_float(x_head[i]) * rms_inv * weight[i];
        float g = bf16_to_float(g_head[i]);
        float silu_g = g / (1.0f + __expf(-g));
        o_head[i] = float_to_bf16(normed * silu_g);
    }
}

/* ── Chunked delta rule: cumsum of g ─────────────────────────────────── */
/*
 * Compute cumulative sum within each chunk.
 * g: [nv, num_chunks, chunk_size] FP32
 * g_cum: [nv, num_chunks, chunk_size] FP32 (output)
 *
 * Grid: (nv, num_chunks, 1), Block: (1, 1, 1)
 * (Sequential within chunk since chunk_size=64 is small)
 */
extern "C" __global__ void la_cumsum_kernel(
    float* __restrict__ g_cum,
    const float* __restrict__ g,
    int num_chunks, int chunk_size)
{
    int head = blockIdx.x;
    int chunk = blockIdx.y;
    int offset = (head * num_chunks + chunk) * chunk_size;

    float sum = 0.0f;
    for (int i = 0; i < chunk_size; i++) {
        sum += g[offset + i];
        g_cum[offset + i] = sum;
    }
}

/* ── Build decay mask and intra-chunk attention ──────────────────────── */
/*
 * For each chunk, compute:
 *   decay_mask[i,j] = exp(g_cum[i] - g_cum[j]) for i >= j, else 0
 *   attn[i,j] = -(k_beta @ k^T)[i,j] * decay_mask[i,j] for i > j, else 0
 *
 * k_beta, k: [nv, num_chunks, chunk_size, dk] FP32
 * g_cum: [nv, num_chunks, chunk_size] FP32
 *
 * attn: [nv, num_chunks, chunk_size, chunk_size] FP32 (strictly lower tri)
 *
 * Grid: (nv, num_chunks, 1), Block: (chunk_size, 1, 1)
 * Each thread handles one row of the chunk_size x chunk_size attn matrix.
 */
extern "C" __global__ void la_build_attn_matrix_kernel(
    float* __restrict__ attn,      /* [nv, num_chunks, CS, CS] */
    const float* __restrict__ k_beta, /* [nv, num_chunks, CS, dk] */
    const float* __restrict__ k,      /* [nv, num_chunks, CS, dk] */
    const float* __restrict__ g_cum,  /* [nv, num_chunks, CS] */
    int num_chunks, int chunk_size, int dk)
{
    int head = blockIdx.x;
    int chunk = blockIdx.y;
    int row = threadIdx.x;
    if (row >= chunk_size) return;

    int cs = chunk_size;
    int base_g = (head * num_chunks + chunk) * cs;
    int base_kbeta = ((head * num_chunks + chunk) * cs + row) * dk;
    int base_attn = (head * num_chunks + chunk) * cs * cs + row * cs;

    float g_i = g_cum[base_g + row];

    /* Compute attn[row, col] for col < row (strictly lower triangular) */
    for (int col = 0; col < cs; col++) {
        if (col >= row) {
            attn[base_attn + col] = 0.0f;
        } else {
            /* k_beta[row] @ k[col] -- dot product */
            float dot = 0.0f;
            int base_k_col = ((head * num_chunks + chunk) * cs + col) * dk;
            for (int d = 0; d < dk; d++) {
                dot += k_beta[base_kbeta + d] * k[base_k_col + d];
            }
            float g_j = g_cum[base_g + col];
            float decay = __expf(g_i - g_j);
            attn[base_attn + col] = -dot * decay;
        }
    }
}

/* ── Triangular solve: (I - A)x = b ─────────────────────────────────── */
/*
 * Forward substitution for unitriangular lower system.
 * A: [nv, num_chunks, CS, CS] (strictly lower triangular)
 * b: [nv, num_chunks, CS, dim] (RHS, overwritten with solution)
 *
 * Solves: (I - A)x = b => x = b + A*x (forward sub since A is strictly lower)
 *
 * Grid: (nv, num_chunks, 1), Block: (min(256, dim), 1, 1)
 * Sequential over rows within each chunk (CS=64 is small).
 */
extern "C" __global__ void la_triangular_solve_kernel(
    float* __restrict__ x,      /* [nv, num_chunks, CS, dim] in/out (starts as b) */
    const float* __restrict__ A, /* [nv, num_chunks, CS, CS] strictly lower tri */
    int num_chunks, int chunk_size, int dim)
{
    int head = blockIdx.x;
    int chunk = blockIdx.y;
    int cs = chunk_size;

    /* base offsets */
    int64_t x_base = ((int64_t)(head * num_chunks + chunk)) * cs * dim;
    int64_t a_base = ((int64_t)(head * num_chunks + chunk)) * cs * cs;

    /* Forward substitution: for row i, x[i] += sum_j(A[i,j] * x[j]) for j < i */
    for (int i = 1; i < cs; i++) {
        /* Each thread handles a subset of the dim dimension */
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++) {
                sum += A[a_base + i * cs + j] * x[x_base + j * dim + d];
            }
            x[x_base + i * dim + d] += sum;
        }
        __syncthreads();  /* Ensure row i is complete before row i+1 reads it */
    }
}

/* ── Chunk recurrence step ───────────────────────────────────────────── */
/*
 * Single chunk step of the recurrent delta rule.
 *
 * For chunk i:
 *   attn_intra = (q @ k^T) * decay_mask, masked upper 0
 *   v_prime = k_cumd @ state
 *   v_new = v_corrected - v_prime
 *   attn_inter = (q * exp(g)) @ state
 *   output = attn_inter + attn_intra @ v_new
 *   g_last_exp = exp(g_last)
 *   k_decay = exp(g_last - g) per row
 *   state = state * g_last_exp + (k * k_decay)^T @ v_new
 *
 * This is called sequentially for each chunk; state carries forward.
 *
 * q, k: [nv, CS, dk]
 * v_corrected: [nv, CS, dv] (from triangular solve)
 * k_cumd: [nv, CS, dk] (from triangular solve)
 * g_cum: [nv, CS] (cumulative gate values)
 * decay_mask: [nv, CS, CS] (precomputed from build_attn_matrix step)
 * state: [nv, dk, dv] (in/out)
 * output: [nv, CS, dv]
 *
 * Grid: (nv, 1, 1), Block: (min(256, dk), 1, 1)
 *
 * NOTE: This kernel is sequential over CS positions per head.
 * For correctness, it processes one head per block.
 */
extern "C" __global__ void la_chunk_recurrence_kernel(
    float* __restrict__ output,        /* [nv, CS, dv] */
    float* __restrict__ state,         /* [nv, dk, dv] in/out */
    const float* __restrict__ q,       /* [nv, CS, dk] */
    const float* __restrict__ k,       /* [nv, CS, dk] */
    const float* __restrict__ v_corr,  /* [nv, CS, dv] (value_corrected) */
    const float* __restrict__ k_cumd,  /* [nv, CS, dk] */
    const float* __restrict__ g_cum,   /* [nv, CS] */
    const float* __restrict__ attn,    /* [nv, CS, CS] (decay_mask for intra) */
    int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    float* st = state + (int64_t)head * dk * dv;
    const float* q_h = q + (int64_t)head * cs * dk;
    const float* k_h = k + (int64_t)head * cs * dk;
    const float* vc_h = v_corr + (int64_t)head * cs * dv;
    const float* kc_h = k_cumd + (int64_t)head * cs * dk;
    const float* g_h = g_cum + (int64_t)head * cs;
    const float* attn_h = attn + (int64_t)head * cs * cs;
    float* out_h = output + (int64_t)head * cs * dv;

    /* We need shared memory for v_new[CS * dv] - too large for 64*128=8192 floats.
       Instead, compute row by row. */

    /* Process each position in the chunk */
    for (int t = 0; t < cs; t++) {
        /* 1. v_prime[dv] = k_cumd[t] @ state  (dk x dk,dv -> dv) */
        /* 2. v_new[dv] = v_corrected[t] - v_prime */
        /* 3. attn_inter[dv] = (q[t] * exp(g[t])) @ state  (dk x dk,dv -> dv) */

        float g_t = g_h[t];

        /* For each output dimension d in dv: */
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* Compute v_prime[d] = sum_j k_cumd[t,j] * state[j,d] */
            float v_prime_d = 0.0f;
            for (int j = 0; j < dk; j++) {
                v_prime_d += kc_h[t * dk + j] * st[j * dv + d];
            }
            float v_new_d = vc_h[t * dv + d] - v_prime_d;

            /* Compute attn_inter[d] = sum_j (q[t,j] * exp(g_t)) * state[j,d] */
            float attn_inter_d = 0.0f;
            float exp_g = __expf(g_t);
            for (int j = 0; j < dk; j++) {
                attn_inter_d += q_h[t * dk + j] * exp_g * st[j * dv + d];
            }

            /* Compute attn_intra @ v_new for position t:
               sum over s<t of attn[t,s] * v_new[s,d]
               But we only have v_new for position t, not for earlier positions.
               We need to compute v_new for ALL positions first... */

            /* Actually, this needs to be restructured. The chunk step function
               in Python computes:
                 v_prime = k_cumd @ state  (all CS positions at once)
                 v_new = v_corrected - v_prime  (all CS positions)
                 attn_intra = (q @ k^T) * decay_mask  (CS x CS)
                 attn_intra masked upper = 0
                 output = (q * exp(g)) @ state + attn_intra @ v_new
                 state update uses g_last and k_decay

               The intra-chunk attention is a GEMM (CS x CS) @ (CS x dv).
               The inter-chunk attention is a GEMM (CS x dk) @ (dk x dv).
               We need to compute these as matrix operations, not row-by-row.

               For correctness in a single kernel, we need to pre-compute
               v_new for all positions, then do the matrix products. */

            /* Store partial result for now - v_new[t, d] */
            out_h[t * dv + d] = v_new_d;
        }
        __syncthreads();
    }

    /* Now out_h contains v_new[CS, dv].
       Compute the full output:
       attn_inter[CS, dv] = (q * exp(g)) @ state   -- (CS,dk) @ (dk,dv)
       attn_intra[CS, CS] = (q @ k^T) * decay_mask -- but we already have decay_mask as 'attn'
       Wait, the 'attn' parameter is the original build_attn_matrix output, which was the
       nilpotent correction matrix. The intra-chunk attention for the chunk step is different.
       The chunk step uses:
         attn_intra = (q_i @ k_i^T) * decay_mask_i  (not the nilpotent A matrix)

       This is getting complex. Let me restructure this as two separate kernels:
       1. Compute v_new = v_corrected - k_cumd @ state  (GEMM + subtract)
       2. Compute attn_intra = (q @ k^T) * decay, output = q*exp(g)@state + attn_intra@v_new
       3. State update

       For now, let's use cuBLAS for the GEMMs from Rust and only use CUDA kernels
       for element-wise ops. This matches the Python approach better.
    */

    /* Simpler approach: just compute v_new and output element-by-element.
       This is O(CS * dk * dv) per head which is 64 * 128 * 128 = ~1M FLOPs.
       With 32 heads, that's 32M FLOPs - tiny for a GPU. */

    /* Recompute properly: */
    /* First, build v_new[CS, dv] (already done above, stored in out_h) */
    /* Build intra attention: q @ k^T * decay (CS x CS) */

    /* We'll use shared memory for the CS x CS attention matrix */
    extern __shared__ float shared[];
    /* shared[0..CS*CS-1] = intra attention matrix */
    float* s_attn = shared;

    /* Build q@k^T * decay_mask (recompute decay from g_cum) */
    if (threadIdx.x == 0) {
        for (int i = 0; i < cs; i++) {
            for (int j = 0; j < cs; j++) {
                if (j >= i) {
                    s_attn[i * cs + j] = 0.0f;
                } else {
                    float dot = 0.0f;
                    for (int dd = 0; dd < dk; dd++) {
                        dot += q_h[i * dk + dd] * k_h[j * dk + dd];
                    }
                    float decay = __expf(g_h[i] - g_h[j]);
                    s_attn[i * cs + j] = dot * decay;
                }
            }
        }
    }
    __syncthreads();

    /* Compute inter + intra attention output */
    for (int t = 0; t < cs; t++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* attn_inter = (q[t] * exp(g[t])) @ state */
            float inter = 0.0f;
            float exp_g = __expf(g_h[t]);
            for (int j = 0; j < dk; j++) {
                inter += q_h[t * dk + j] * exp_g * st[j * dv + d];
            }

            /* attn_intra @ v_new = sum_s attn[t,s] * v_new[s,d] */
            float intra = 0.0f;
            for (int s = 0; s < t; s++) {
                intra += s_attn[t * cs + s] * out_h[s * dv + d];
            }

            out_h[t * dv + d] = inter + intra;
        }
        __syncthreads();
    }

    /* State update: state = state * exp(g_last) + (k * k_decay)^T @ v_new
       where v_new was stored in out_h before we overwrote it...
       We need to save v_new first. */

    /* This kernel is getting too complex. Let's split into multiple kernels
       and orchestrate from Rust. See la_chunk_* kernels below. */
}

/* ── Simpler chunked kernels (orchestrated from Rust) ─────────────────── */

/* Compute v_new = v_corrected - k_cumd @ state for all positions in one chunk.
 * k_cumd: [CS, dk], state: [dk, dv], v_corrected: [CS, dv]
 * v_new: [CS, dv] output
 * One block per head.
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_compute_v_new_kernel(
    float* __restrict__ v_new,         /* [nv, CS, dv] */
    const float* __restrict__ v_corr,  /* [nv, CS, dv] */
    const float* __restrict__ k_cumd,  /* [nv, CS, dk] */
    const float* __restrict__ state,   /* [nv, dk, dv] */
    int nv, int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= nv) return;
    int cs = chunk_size;

    const float* vc = v_corr + (int64_t)head * cs * dv;
    const float* kc = k_cumd + (int64_t)head * cs * dk;
    const float* st = state + (int64_t)head * dk * dv;
    float* vn = v_new + (int64_t)head * cs * dv;

    for (int t = 0; t < cs; t++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float v_prime = 0.0f;
            for (int j = 0; j < dk; j++) {
                v_prime += kc[t * dk + j] * st[j * dv + d];
            }
            vn[t * dv + d] = vc[t * dv + d] - v_prime;
        }
    }
}

/* Compute output for one chunk:
 *   attn_inter = (q * exp(g)) @ state
 *   attn_intra = tril((q @ k^T) * decay, -1) @ v_new
 *   output = attn_inter + attn_intra
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_chunk_output_kernel(
    float* __restrict__ output,       /* [nv, CS, dv] */
    const float* __restrict__ q,      /* [nv, CS, dk] */
    const float* __restrict__ k,      /* [nv, CS, dk] */
    const float* __restrict__ v_new,  /* [nv, CS, dv] */
    const float* __restrict__ g_cum,  /* [nv, CS] */
    const float* __restrict__ state,  /* [nv, dk, dv] */
    int nv, int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= nv) return;
    int cs = chunk_size;

    const float* q_h = q + (int64_t)head * cs * dk;
    const float* k_h = k + (int64_t)head * cs * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * cs;
    const float* st = state + (int64_t)head * dk * dv;
    float* out = output + (int64_t)head * cs * dv;

    for (int t = 0; t < cs; t++) {
        float exp_g = __expf(g_h[t]);

        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* Inter-chunk: (q[t] * exp(g[t])) @ state[:, d] */
            float inter = 0.0f;
            for (int j = 0; j < dk; j++) {
                inter += q_h[t * dk + j] * st[j * dv + d];
            }
            inter *= exp_g;

            /* Intra-chunk: sum_{s<=t} [(q[t] @ k[s]) * decay(t,s)] * v_new[s, d] */
            float intra = 0.0f;
            for (int s = 0; s <= t; s++) {
                float qk_dot = 0.0f;
                for (int j = 0; j < dk; j++) {
                    qk_dot += q_h[t * dk + j] * k_h[s * dk + j];
                }
                float decay = __expf(g_h[t] - g_h[s]);
                intra += qk_dot * decay * vn_h[s * dv + d];
            }

            out[t * dv + d] = inter + intra;
        }
    }
}

/* Update recurrent state after one chunk:
 *   state = state * exp(g_last) + (k * k_decay)^T @ v_new
 *   where k_decay[t] = exp(g_last - g_cum[t])
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_state_update_kernel(
    float* __restrict__ state,        /* [nv, dk, dv] in/out */
    const float* __restrict__ k,      /* [nv, CS, dk] */
    const float* __restrict__ v_new,  /* [nv, CS, dv] */
    const float* __restrict__ g_cum,  /* [nv, CS] */
    int nv, int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= nv) return;
    int cs = chunk_size;

    float* st = state + (int64_t)head * dk * dv;
    const float* k_h = k + (int64_t)head * cs * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * cs;

    float g_last = g_h[cs - 1];
    float g_last_exp = __expf(g_last);

    /* state[j, d] = state[j, d] * g_last_exp + sum_t k[t, j] * k_decay[t] * v_new[t, d] */
    for (int j = 0; j < dk; j++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float s = st[j * dv + d] * g_last_exp;
            for (int t = 0; t < cs; t++) {
                float k_decay = __expf(g_last - g_h[t]);
                s += k_h[t * dk + j] * k_decay * vn_h[t * dv + d];
            }
            st[j * dv + d] = s;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Concat 3 BF16 arrays row-wise: [M,a_dim]+[M,b_dim]+[M,c_dim] -> [M,total]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Replaces per-token memcpy loops for conv1d input preparation.
 * Grid: (M, 1, 1), Block: (min(256, a_dim+b_dim+c_dim), 1, 1)
 */
extern "C" __global__ void concat_3_bf16_kernel(
    __nv_bfloat16* __restrict__ out,       /* [M, a_dim+b_dim+c_dim] */
    const __nv_bfloat16* __restrict__ a,   /* [M, a_dim] */
    const __nv_bfloat16* __restrict__ b,   /* [M, b_dim] */
    const __nv_bfloat16* __restrict__ c,   /* [M, c_dim] */
    int a_dim, int b_dim, int c_dim)
{
    int token = blockIdx.x;
    int total = a_dim + b_dim + c_dim;
    __nv_bfloat16* dst = out + (int64_t)token * total;
    const __nv_bfloat16* a_src = a + (int64_t)token * a_dim;
    const __nv_bfloat16* b_src = b + (int64_t)token * b_dim;
    const __nv_bfloat16* c_src = c + (int64_t)token * c_dim;

    for (int d = threadIdx.x; d < total; d += blockDim.x) {
        if (d < a_dim) {
            dst[d] = a_src[d];
        } else if (d < a_dim + b_dim) {
            dst[d] = b_src[d - a_dim];
        } else {
            dst[d] = c_src[d - a_dim - b_dim];
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * Strided chunk kernels — read directly from [nv, total_len, dim] arrays
 * ═══════════════════════════════════════════════════════════════════════
 *
 * These replace the per-head memcpy storms in the chunk recurrence loop.
 * Instead of copying chunk data to contiguous buffers, these kernels
 * compute offsets from total_len stride and chunk_idx directly.
 */

/* Strided v_new computation:
 *   v_new[nv, CS, dv] = v_corr[strided] - k_cumd[strided] @ state[nv, dk, dv]
 *
 * v_corr: [nv, total_len, dv] FP32 (strided)
 * k_cumd: [nv, total_len, dk] FP32 (strided)
 * state:  [nv, dk, dv] FP32 (contiguous)
 * v_new:  [nv, CS, dv] FP32 (contiguous output)
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_compute_v_new_strided_kernel(
    float* __restrict__ v_new,         /* [nv, CS, dv] contiguous output */
    const float* __restrict__ v_corr,  /* [nv, total_len, dv] strided */
    const float* __restrict__ k_cumd,  /* [nv, total_len, dk] strided */
    const float* __restrict__ state,   /* [nv, dk, dv] contiguous */
    int chunk_size, int dk, int dv,
    int total_len, int chunk_idx)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    /* Strided offsets: head * total_len * dim + chunk_idx * CS * dim */
    const float* vc = v_corr + ((int64_t)head * total_len + chunk_idx * cs) * dv;
    const float* kc = k_cumd + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* st = state + (int64_t)head * dk * dv;
    float* vn = v_new + (int64_t)head * cs * dv;

    for (int t = 0; t < cs; t++) {
        /* Strided access: vc[t * dv] reads from total_len-strided array
         * but within a chunk, positions are contiguous (t * dv stride is fine
         * because the chunk is a contiguous slice of the total_len dimension). */
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float v_prime = 0.0f;
            for (int j = 0; j < dk; j++) {
                v_prime += kc[t * dk + j] * st[j * dv + d];
            }
            vn[t * dv + d] = vc[t * dv + d] - v_prime;
        }
    }
}

/* Strided chunk output:
 *   output[strided] = (q * exp(g)) @ state + tril(q @ k^T * decay) @ v_new
 *
 * Reads q, k, g_cum from strided [nv, total_len, dim] arrays.
 * Reads v_new from contiguous [nv, CS, dv] buffer.
 * Writes output directly to strided [nv, total_len, dv] buffer.
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_chunk_output_strided_kernel(
    float* __restrict__ output,        /* [nv, total_len, dv] strided */
    const float* __restrict__ q,       /* [nv, total_len, dk] strided */
    const float* __restrict__ k,       /* [nv, total_len, dk] strided */
    const float* __restrict__ v_new,   /* [nv, CS, dv] contiguous */
    const float* __restrict__ g_cum,   /* [nv, total_len] strided */
    const float* __restrict__ state,   /* [nv, dk, dv] contiguous */
    int chunk_size, int dk, int dv,
    int total_len, int chunk_idx)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    const float* q_h = q + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* k_h = k + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * total_len + chunk_idx * cs;
    const float* st = state + (int64_t)head * dk * dv;
    float* out = output + ((int64_t)head * total_len + chunk_idx * cs) * dv;

    for (int t = 0; t < cs; t++) {
        float exp_g = __expf(g_h[t]);

        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* Inter-chunk: (q[t] * exp(g[t])) @ state[:, d] */
            float inter = 0.0f;
            for (int j = 0; j < dk; j++) {
                inter += q_h[t * dk + j] * st[j * dv + d];
            }
            inter *= exp_g;

            /* Intra-chunk: sum_{s<=t} [(q[t] @ k[s]) * decay(t,s)] * v_new[s, d] */
            float intra = 0.0f;
            for (int s = 0; s <= t; s++) {
                float qk_dot = 0.0f;
                for (int j = 0; j < dk; j++) {
                    qk_dot += q_h[t * dk + j] * k_h[s * dk + j];
                }
                float decay = __expf(g_h[t] - g_h[s]);
                intra += qk_dot * decay * vn_h[s * dv + d];
            }

            out[t * dv + d] = inter + intra;
        }
    }
}

/* Strided state update:
 *   state = state * exp(g_last) + (k * k_decay)^T @ v_new
 *
 * Reads k, g_cum from strided [nv, total_len, dim] arrays.
 * Reads v_new from contiguous [nv, CS, dv] buffer.
 * Updates state [nv, dk, dv] in-place.
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_state_update_strided_kernel(
    float* __restrict__ state,        /* [nv, dk, dv] in/out */
    const float* __restrict__ k,      /* [nv, total_len, dk] strided */
    const float* __restrict__ v_new,  /* [nv, CS, dv] contiguous */
    const float* __restrict__ g_cum,  /* [nv, total_len] strided */
    int chunk_size, int dk, int dv,
    int total_len, int chunk_idx)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    float* st = state + (int64_t)head * dk * dv;
    const float* k_h = k + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * total_len + chunk_idx * cs;

    float g_last = g_h[cs - 1];
    float g_last_exp = __expf(g_last);

    for (int j = 0; j < dk; j++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float s = st[j * dv + d] * g_last_exp;
            for (int t = 0; t < cs; t++) {
                float k_decay = __expf(g_last - g_h[t]);
                s += k_h[t * dk + j] * k_decay * vn_h[t * dv + d];
            }
            st[j * dv + d] = s;
        }
    }
}


/* ── Multiply k_beta by exp(g_cum) ───────────────────────────────────── */
/*
 * Prepares k_beta_g = k_beta * exp(g_cum) for the second triangular solve.
 * k_beta: [nv, total_len, dk] FP32
 * g_cum: [nv, total_len] FP32 (per-chunk cumulative)
 * k_beta_g: [nv, total_len, dk] FP32 output
 *
 * Grid: (nv, total_len, 1), Block: (min(256, dk), 1, 1)
 */
extern "C" __global__ void la_scale_by_exp_g_kernel(
    float* __restrict__ k_beta_g,
    const float* __restrict__ k_beta,
    const float* __restrict__ g_cum,
    int nv, int total_len, int dk)
{
    int head = blockIdx.x;
    int pos = blockIdx.y;
    if (head >= nv || pos >= total_len) return;

    float eg = __expf(g_cum[head * total_len + pos]);
    const float* src = k_beta + ((int64_t)head * total_len + pos) * dk;
    float* dst = k_beta_g + ((int64_t)head * total_len + pos) * dk;

    for (int d = threadIdx.x; d < dk; d += blockDim.x) {
        dst[d] = src[d] * eg;
    }
}

/* ── FP32 2D Transpose ───────────────────────────────────────────────── */
/*
 * Transpose a [rows, cols] FP32 matrix to [cols, rows].
 * Grid: (cols, 1, 1), Block: (min(1024, rows_padded), 1, 1)
 */
extern "C" __global__ void la_transpose_f32_kernel(
    float* __restrict__ out,        /* [cols, rows] */
    const float* __restrict__ in,   /* [rows, cols] */
    int rows, int cols)
{
    int col = blockIdx.x;
    if (col >= cols) return;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        out[col * rows + row] = in[row * cols + col];
    }
}

/* ── Prepare beta-scaled k and v ─────────────────────────────────────── */
/*
 * v_beta = v * beta,  k_beta = k * beta
 * v: [M, nv, dv] FP32, k: [M, nv, dk] FP32, beta: [M, nv] FP32
 * Grid: (M, nv, 1), Block: (min(256, max(dk,dv)), 1, 1)
 */
extern "C" __global__ void la_apply_beta_kernel(
    float* __restrict__ v_beta,  /* [M, nv, dv] */
    float* __restrict__ k_beta,  /* [nv, M, dk] */
    const float* __restrict__ v, /* [M, nv, dv] */
    const float* __restrict__ k, /* [nv, M, dk] */
    const float* __restrict__ beta, /* [M, nv] */
    int nv, int dk, int dv, int total_len)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    float b = beta[token * nv + head];

    /* v_beta */
    const float* v_src = v + ((int64_t)token * nv + head) * dv;
    float* v_dst = v_beta + ((int64_t)token * nv + head) * dv;
    for (int d = threadIdx.x; d < dv; d += blockDim.x) {
        v_dst[d] = v_src[d] * b;
    }

    /* k_beta */
    const float* k_src = k + ((int64_t)head * total_len + token) * dk;
    float* k_dst = k_beta + ((int64_t)head * total_len + token) * dk;
    for (int d = threadIdx.x; d < dk; d += blockDim.x) {
        k_dst[d] = k_src[d] * b;
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * 3D Transpose: [A, B, C] -> [B, A, C]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Single-launch replacement for per-element memcpy storms.
 * Grid: (B, A, 1), Block: (min(256, C), 1, 1)
 * Each thread block copies one (b, a) row of C elements.
 */
extern "C" __global__ void transpose_3d_021_kernel(
    float* __restrict__ out,    /* [B, A, C] */
    const float* __restrict__ in, /* [A, B, C] */
    int A, int B, int C)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (a >= A || b >= B) return;

    const float* src = in  + ((int64_t)a * B + b) * C;
    float*       dst = out + ((int64_t)b * A + a) * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dst[c] = src[c];
    }
}

/* BF16 version */
extern "C" __global__ void transpose_3d_021_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int A, int B, int C)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (a >= A || b >= B) return;

    const __nv_bfloat16* src = in  + ((int64_t)a * B + b) * C;
    __nv_bfloat16*       dst = out + ((int64_t)b * A + a) * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dst[c] = src[c];
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * Gated Q Split: [M, H, 2*D] -> Q[M, H*D] + gate[M, H*D]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Used by QCN's gated GQA attention where q_proj outputs interleaved [query, gate].
 * Grid: (M, H, 1), Block: (min(256, D), 1, 1)
 */
extern "C" __global__ void gated_q_split_kernel(
    __nv_bfloat16* __restrict__ q_out,    /* [M, H*D] */
    __nv_bfloat16* __restrict__ gate_out, /* [M, H*D] */
    const __nv_bfloat16* __restrict__ qg, /* [M, H, 2*D] */
    int H, int D)
{
    int token = blockIdx.x;
    int head  = blockIdx.y;

    const __nv_bfloat16* src = qg + ((int64_t)token * H + head) * (2 * D);
    __nv_bfloat16* q_dst = q_out  + ((int64_t)token * H + head) * D;
    __nv_bfloat16* g_dst = gate_out + ((int64_t)token * H + head) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        q_dst[d] = src[d];
        g_dst[d] = src[D + d];
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * LA Conv Output Split: [M, conv_dim] -> q[M, key_dim] + k[M, key_dim] + v[M, value_dim]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * conv_dim = 2 * key_dim + value_dim
 * Grid: (M, 1, 1), Block: (min(256, conv_dim), 1, 1)
 */
extern "C" __global__ void la_split_conv_output_kernel(
    float* __restrict__ q_out,     /* [M, key_dim] */
    float* __restrict__ k_out,     /* [M, key_dim] */
    float* __restrict__ v_out,     /* [M, value_dim] */
    const float* __restrict__ inp, /* [M, conv_dim] */
    int key_dim, int value_dim)
{
    int token = blockIdx.x;
    int conv_dim = 2 * key_dim + value_dim;

    const float* src = inp + (int64_t)token * conv_dim;
    float* q = q_out + (int64_t)token * key_dim;
    float* k = k_out + (int64_t)token * key_dim;
    float* v = v_out + (int64_t)token * value_dim;

    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        float val = src[d];
        if (d < key_dim) {
            q[d] = val;
        } else if (d < 2 * key_dim) {
            k[d - key_dim] = val;
        } else {
            v[d - 2 * key_dim] = val;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * Flash Attention with Tensor Core MMA — BF16/FP8 KV cache, Causal, GQA
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Uses WMMA (Warp Matrix Multiply-Accumulate) for Q*K^T and P*V products.
 * Supports cross-chunk attention: reads from FP8 KV cache for
 * positions [0, start_pos) and from BF16 GEMM output for the current
 * chunk [start_pos, start_pos + M).
 *
 * Design:
 *   BR = 16 queries per block (one wmma M=16 tile)
 *   BC = 64 KV positions per tile
 *   1 warp (32 threads) per block
 *   16x better K/V tile amortization vs old BR=4 kernel
 *
 * Q*K^T: [16, d] * [d, 64] via wmma m16n16k16, d/16 * 4 = 32 MMA calls
 * P*V:   [16, 64] * [64, d] via wmma m16n16k16, 4 * d/16 = 32 MMA calls
 * Online softmax between phases, accumulator O in registers.
 *
 * Shared memory (head_dim=128):
 *   s_q:     [16, 128] bf16 = 4 KB  — query tile, loaded once
 *   s_k:     [64, 128] bf16 = 16 KB — K tile (reused for zero-pad)
 *   s_v:     [64, 128] bf16 = 16 KB — V tile
 *   s_scores [16, 64]  f32  = 4 KB  — score tile for softmax
 *   s_p:     [16, 64]  bf16 = 2 KB  — P (softmax output) for P*V
 *   s_o_tmp: [16, 16]  f32  = 1 KB  — partial O from each P*V fragment
 *   Total: ~43 KB (fits in default 48 KB limit)
 *
 * Grid: (ceil(M/16), num_q_heads, 1)
 * Block: (32, 1, 1) = 1 warp
 */
#include <mma.h>

#define FA_BC 16    /* KV tile size; 16 keeps 512-dim Gemma heads under smem limits */
#define FA_BR 16    /* queries per block = wmma tile M */
#define FA_TILE 16  /* wmma tile dimension */
#define FA_DPT 16   /* max dims per thread = ceil(512/32) */

__device__ __forceinline__ float fp8e4m3_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

extern "C" __global__ void flash_attn_tiled_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,       /* [M, num_q_heads, head_dim] */
    const __nv_fp8_e4m3* __restrict__ k_cache, /* [max_seq, kv_stride] FP8 or null */
    const __nv_fp8_e4m3* __restrict__ v_cache, /* [max_seq, kv_stride] FP8 or null */
    const __nv_bfloat16* __restrict__ k_cur,   /* [M, kv_stride] BF16 current chunk */
    const __nv_bfloat16* __restrict__ v_cur,   /* [M, kv_stride] BF16 current chunk */
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int start_pos, int kv_stride, int sliding_window,
    const int* __restrict__ vision_block_ids)
{
    int q_base = blockIdx.x * FA_BR;
    int qh = blockIdx.y;
    int kv_h = qh / (num_q_heads / num_kv_heads);
    int lane = threadIdx.x;  /* 0..31 */

    /* ── Shared memory layout ── */
    extern __shared__ char smem_fa[];
    __nv_bfloat16* s_q = (__nv_bfloat16*)smem_fa;                    /* [BR, hd] */
    __nv_bfloat16* s_k = s_q + FA_BR * head_dim;                     /* [BC, hd] */
    __nv_bfloat16* s_v = s_k + FA_BC * head_dim;                     /* [BC, hd] */
    float* s_scores = (float*)(s_v + FA_BC * head_dim);               /* [BR, BC] */
    __nv_bfloat16* s_p = (__nv_bfloat16*)(s_scores + FA_BR * FA_BC);  /* [BR, BC] */
    float* s_o_tmp = (float*)(s_p + FA_BR * FA_BC);                    /* [TILE, TILE] */

    /* ── Load Q tile to shared memory (contiguous row-major) ── */
    int q_stride = num_q_heads * head_dim;
    for (int idx = lane; idx < FA_BR * head_dim; idx += 32) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        int qi = q_base + r;
        s_q[r * head_dim + d] = (qi < M)
            ? q[(int64_t)qi * q_stride + qh * head_dim + d]
            : float_to_bf16(0.0f);
    }
    __syncthreads();

    /* ── Per-thread state ── */
    int dpt = head_dim / 32;  /* dims per thread (head_dim multiple of 32) */
    float row_max_arr[FA_BR], row_sum_arr[FA_BR];
    float O_reg[FA_BR * FA_DPT];  /* output accumulator: 16 rows * dpt dims */
    for (int r = 0; r < FA_BR; r++) {
        row_max_arr[r] = -1e30f;
        row_sum_arr[r] = 0.0f;
    }
    for (int i = 0; i < FA_BR * dpt; i++) O_reg[i] = 0.0f;

    /* Block-level causal/window bounds */
    int block_last_qi = min(q_base + FA_BR - 1, M - 1);
    int block_max_kv = (q_base < M) ? (start_pos + block_last_qi + 1) : 0;
    int block_min_kv = 0;
    if (sliding_window > 0 && q_base < M) {
        int first_abs_q = start_pos + q_base;
        block_min_kv = max(0, first_abs_q - sliding_window + 1);
        block_min_kv = (block_min_kv / FA_BC) * FA_BC;
    }

    /* ══ Main loop over KV tiles ══ */
    for (int kv_start = block_min_kv; kv_start < block_max_kv; kv_start += FA_BC) {
        int tile_end = min(kv_start + FA_BC, block_max_kv);
        int tile_size = tile_end - kv_start;

        /* ── Load K and V tile (cooperative, 32 threads) ── */
        for (int idx = lane; idx < FA_BC * head_dim; idx += 32) {
            int ki = idx / head_dim;
            int d = idx % head_dim;
            __nv_bfloat16 kval, vval;
            if (ki < tile_size) {
                int abs_pos = kv_start + ki;
                if (abs_pos < start_pos && k_cache != nullptr) {
                    int64_t off = (int64_t)abs_pos * kv_stride + kv_h * head_dim + d;
                    kval = float_to_bf16(fp8e4m3_to_float(k_cache[off]));
                    vval = float_to_bf16(fp8e4m3_to_float(v_cache[off]));
                } else {
                    int cp = abs_pos - start_pos;
                    if (cp >= 0 && cp < M) {
                        int64_t off = ((int64_t)cp * num_kv_heads + kv_h) * head_dim + d;
                        kval = k_cur[off];
                        vval = v_cur[off];
                    } else {
                        kval = float_to_bf16(0.0f);
                        vval = float_to_bf16(0.0f);
                    }
                }
            } else {
                kval = float_to_bf16(0.0f);
                vval = float_to_bf16(0.0f);
            }
            s_k[ki * head_dim + d] = kval;
            s_v[ki * head_dim + d] = vval;
        }
        __syncwarp();

        /* ── Phase 1: Q * K^T via WMMA → s_scores [BR, BC] ──
         * 4 output column tiles (BC/16), 8 k-dim iterations (hd/16 for hd=128)
         * = 32 wmma mma_sync calls total */
        for (int nj = 0; nj < FA_BC / FA_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> s_frag;
            nvcuda::wmma::fill_fragment(s_frag, 0.0f);

            for (int kk = 0; kk < head_dim / FA_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> q_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::col_major> k_frag;

                /* Q[0:16, kk*16:(kk+1)*16] from s_q, stride=head_dim */
                nvcuda::wmma::load_matrix_sync(q_frag,
                    &s_q[kk * FA_TILE], head_dim);
                /* K[nj*16:(nj+1)*16, kk*16:(kk+1)*16] from s_k, col_major transposes */
                nvcuda::wmma::load_matrix_sync(k_frag,
                    &s_k[nj * FA_TILE * head_dim + kk * FA_TILE], head_dim);

                nvcuda::wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            /* Store score tile [16, 16] to s_scores at column offset nj*16 */
            nvcuda::wmma::store_matrix_sync(
                &s_scores[nj * FA_TILE], s_frag, FA_BC,
                nvcuda::wmma::mem_row_major);
        }
        __syncwarp();

        /* ── Phase 2: Online softmax on s_scores [BR, BC] ──
         * Each of 32 threads processes a subset of each row's columns.
         * Warp-level reductions for max and sum. */
        for (int r = 0; r < FA_BR; r++) {
            int qi = q_base + r;
            if (qi >= M) continue;
            int abs_qi_r = start_pos + qi;
            float* row = &s_scores[r * FA_BC];

            /* Scale scores + causal mask + local max */
            float local_max = -1e30f;
            for (int c = lane; c < FA_BC; c += 32) {
                int abs_kv = kv_start + c;
                int row_min_kv = (sliding_window > 0) ? max(0, abs_qi_r - sliding_window + 1) : 0;
                bool in_tile = c < tile_size;
                bool causal_allowed = abs_kv <= abs_qi_r;
                bool same_vision_block = false;
                if (in_tile && vision_block_ids != nullptr && sliding_window > 0) {
                    int q_block = vision_block_ids[abs_qi_r];
                    int k_block = vision_block_ids[abs_kv];
                    same_vision_block = q_block >= 0 && q_block == k_block;
                }
                if (in_tile && abs_kv >= row_min_kv && (causal_allowed || same_vision_block)) {
                    row[c] *= softmax_scale;
                    local_max = fmaxf(local_max, row[c]);
                } else {
                    row[c] = -1e30f;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

            /* Update online softmax state */
            float old_max = row_max_arr[r];
            float new_max = fmaxf(old_max, local_max);
            float rescale = __expf(old_max - new_max);
            row_max_arr[r] = new_max;

            /* Compute P = exp(score - max), local sum */
            float local_sum = 0.0f;
            for (int c = lane; c < FA_BC; c += 32) {
                float p = __expf(row[c] - new_max);
                row[c] = p;
                local_sum += p;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

            row_sum_arr[r] = row_sum_arr[r] * rescale + local_sum;

            /* Rescale existing O accumulator for this row */
            for (int j = 0; j < dpt; j++)
                O_reg[r * dpt + j] *= rescale;

            /* Convert P to BF16 for WMMA P*V */
            for (int c = lane; c < FA_BC; c += 32)
                s_p[r * FA_BC + c] = float_to_bf16(row[c]);
        }
        __syncwarp();

        /* ── Phase 3: O += P * V via WMMA ──
         * Process head_dim/16 output column tiles.
         * For each: 4 k-dim iterations (BC/16), then store to s_o_tmp
         * and accumulate to O_reg. */
        for (int nj = 0; nj < head_dim / FA_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_partial;
            nvcuda::wmma::fill_fragment(o_partial, 0.0f);

            for (int kk = 0; kk < FA_BC / FA_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> p_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> v_frag;

                /* P[0:16, kk*16:(kk+1)*16], stride=BC */
                nvcuda::wmma::load_matrix_sync(p_frag,
                    &s_p[kk * FA_TILE], FA_BC);
                /* V[kk*16:(kk+1)*16, nj*16:(nj+1)*16], stride=head_dim */
                nvcuda::wmma::load_matrix_sync(v_frag,
                    &s_v[kk * FA_TILE * head_dim + nj * FA_TILE], head_dim);

                nvcuda::wmma::mma_sync(o_partial, p_frag, v_frag, o_partial);
            }

            /* Store [16, 16] partial output, accumulate to O_reg */
            nvcuda::wmma::store_matrix_sync(s_o_tmp, o_partial,
                FA_TILE, nvcuda::wmma::mem_row_major);
            __syncwarp();

            /* Map fragment columns to thread's O_reg:
             * Thread t owns dims: t, t+32, t+64, ...
             * Fragment nj covers columns [nj*16, nj*16+16).
             * Active threads: (nj*16)%32 .. (nj*16)%32+15 */
            int first_t = (nj * FA_TILE) % 32;
            int my_col = lane - first_t;
            if (my_col >= 0 && my_col < FA_TILE) {
                int d_global = nj * FA_TILE + my_col;
                int j = d_global / 32;
                for (int r = 0; r < FA_BR; r++)
                    O_reg[r * dpt + j] += s_o_tmp[r * FA_TILE + my_col];
            }
            __syncwarp();
        }
        __syncwarp();  /* before next tile load */
    }

    /* ── Write output ── */
    for (int r = 0; r < FA_BR; r++) {
        int qi = q_base + r;
        if (qi >= M) continue;
        float inv_sum = (row_sum_arr[r] > 0.0f) ? (1.0f / row_sum_arr[r]) : 0.0f;
        __nv_bfloat16* o_row = out + ((int64_t)qi * num_q_heads + qh) * head_dim;
        for (int j = 0; j < dpt; j++) {
            int d = lane + j * 32;
            if (d < head_dim)
                o_row[d] = float_to_bf16(O_reg[r * dpt + j] * inv_sum);
        }
    }
}

/* Specialized full-prefill fallback for Gemma GQA layers with head_dim=512.
 * The generic tiled kernel uses BC=16 to support all head dimensions <=512.
 * For the measured Gemma full-attention path, BC=32 still fits Ampere-class
 * opt-in shared memory and halves the number of causal KV tiles. */
#define FA512_BC 32
#define FA512_BR 16
#define FA512_TILE 16
#define FA512_DPT 16
#define FA512_HD 512

extern "C" __global__ void flash_attn_tiled_hd512_full_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,     /* [M, num_q_heads, 512] */
    const __nv_bfloat16* __restrict__ k_cur, /* [M, num_kv_heads, 512] */
    const __nv_bfloat16* __restrict__ v_cur, /* [M, num_kv_heads, 512] */
    int M, int num_q_heads, int num_kv_heads, float softmax_scale)
{
    int q_base = blockIdx.x * FA512_BR;
    int qh = blockIdx.y;
    int q_per_kv = num_q_heads / num_kv_heads;
    int kv_h = qh / q_per_kv;
    int lane = threadIdx.x;

    extern __shared__ char smem_fa512[];
    __nv_bfloat16* s_q = (__nv_bfloat16*)smem_fa512;                         /* [BR, 512] */
    __nv_bfloat16* s_k = s_q + FA512_BR * FA512_HD;                          /* [BC, 512] */
    __nv_bfloat16* s_v = s_k + FA512_BC * FA512_HD;                          /* [BC, 512] */
    float* s_scores = (float*)(s_v + FA512_BC * FA512_HD);                   /* [BR, BC] */
    __nv_bfloat16* s_p = (__nv_bfloat16*)(s_scores + FA512_BR * FA512_BC);   /* [BR, BC] */
    float* s_o_tmp = (float*)(s_p + FA512_BR * FA512_BC);                    /* [16, 16] */

    int q_stride = num_q_heads * FA512_HD;
    for (int idx = lane; idx < FA512_BR * FA512_HD; idx += 32) {
        int r = idx / FA512_HD;
        int d = idx % FA512_HD;
        int qi = q_base + r;
        s_q[r * FA512_HD + d] = (qi < M)
            ? q[(int64_t)qi * q_stride + qh * FA512_HD + d]
            : float_to_bf16(0.0f);
    }
    __syncthreads();

    float row_max_arr[FA512_BR], row_sum_arr[FA512_BR];
    float O_reg[FA512_BR * FA512_DPT];
    for (int r = 0; r < FA512_BR; r++) {
        row_max_arr[r] = -1e30f;
        row_sum_arr[r] = 0.0f;
    }
    for (int i = 0; i < FA512_BR * FA512_DPT; i++) O_reg[i] = 0.0f;

    int block_last_qi = min(q_base + FA512_BR - 1, M - 1);
    int block_max_kv = (q_base < M) ? (block_last_qi + 1) : 0;

    for (int kv_start = 0; kv_start < block_max_kv; kv_start += FA512_BC) {
        int tile_end = min(kv_start + FA512_BC, block_max_kv);
        int tile_size = tile_end - kv_start;

        for (int idx = lane; idx < FA512_BC * FA512_HD; idx += 32) {
            int ki = idx / FA512_HD;
            int d = idx % FA512_HD;
            __nv_bfloat16 kval, vval;
            if (ki < tile_size) {
                int abs_pos = kv_start + ki;
                int64_t off = ((int64_t)abs_pos * num_kv_heads + kv_h) * FA512_HD + d;
                kval = k_cur[off];
                vval = v_cur[off];
            } else {
                kval = float_to_bf16(0.0f);
                vval = float_to_bf16(0.0f);
            }
            s_k[ki * FA512_HD + d] = kval;
            s_v[ki * FA512_HD + d] = vval;
        }
        __syncwarp();

        for (int nj = 0; nj < FA512_BC / FA512_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> s_frag;
            nvcuda::wmma::fill_fragment(s_frag, 0.0f);

            for (int kk = 0; kk < FA512_HD / FA512_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> q_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::col_major> k_frag;

                nvcuda::wmma::load_matrix_sync(q_frag,
                    &s_q[kk * FA512_TILE], FA512_HD);
                nvcuda::wmma::load_matrix_sync(k_frag,
                    &s_k[nj * FA512_TILE * FA512_HD + kk * FA512_TILE], FA512_HD);

                nvcuda::wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            nvcuda::wmma::store_matrix_sync(
                &s_scores[nj * FA512_TILE], s_frag, FA512_BC,
                nvcuda::wmma::mem_row_major);
        }
        __syncwarp();

        for (int r = 0; r < FA512_BR; r++) {
            int qi = q_base + r;
            if (qi >= M) continue;
            float* row = &s_scores[r * FA512_BC];

            float local_max = -1e30f;
            for (int c = lane; c < FA512_BC; c += 32) {
                int abs_kv = kv_start + c;
                if (c < tile_size && abs_kv <= qi) {
                    row[c] *= softmax_scale;
                    local_max = fmaxf(local_max, row[c]);
                } else {
                    row[c] = -1e30f;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

            float old_max = row_max_arr[r];
            float new_max = fmaxf(old_max, local_max);
            float rescale = __expf(old_max - new_max);
            row_max_arr[r] = new_max;

            float local_sum = 0.0f;
            for (int c = lane; c < FA512_BC; c += 32) {
                float p = __expf(row[c] - new_max);
                row[c] = p;
                local_sum += p;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

            row_sum_arr[r] = row_sum_arr[r] * rescale + local_sum;

            for (int j = 0; j < FA512_DPT; j++)
                O_reg[r * FA512_DPT + j] *= rescale;

            for (int c = lane; c < FA512_BC; c += 32)
                s_p[r * FA512_BC + c] = float_to_bf16(row[c]);
        }
        __syncwarp();

        for (int nj = 0; nj < FA512_HD / FA512_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_partial;
            nvcuda::wmma::fill_fragment(o_partial, 0.0f);

            for (int kk = 0; kk < FA512_BC / FA512_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> p_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> v_frag;

                nvcuda::wmma::load_matrix_sync(p_frag,
                    &s_p[kk * FA512_TILE], FA512_BC);
                nvcuda::wmma::load_matrix_sync(v_frag,
                    &s_v[kk * FA512_TILE * FA512_HD + nj * FA512_TILE], FA512_HD);

                nvcuda::wmma::mma_sync(o_partial, p_frag, v_frag, o_partial);
            }

            nvcuda::wmma::store_matrix_sync(s_o_tmp, o_partial,
                FA512_TILE, nvcuda::wmma::mem_row_major);
            __syncwarp();

            int first_t = (nj * FA512_TILE) % 32;
            int my_col = lane - first_t;
            if (my_col >= 0 && my_col < FA512_TILE) {
                int d_global = nj * FA512_TILE + my_col;
                int j = d_global / 32;
                for (int r = 0; r < FA512_BR; r++)
                    O_reg[r * FA512_DPT + j] += s_o_tmp[r * FA512_TILE + my_col];
            }
            __syncwarp();
        }
        __syncwarp();
    }

    for (int r = 0; r < FA512_BR; r++) {
        int qi = q_base + r;
        if (qi >= M) continue;
        float inv_sum = (row_sum_arr[r] > 0.0f) ? (1.0f / row_sum_arr[r]) : 0.0f;
        __nv_bfloat16* o_row = out + ((int64_t)qi * num_q_heads + qh) * FA512_HD;
        for (int j = 0; j < FA512_DPT; j++) {
            int d = lane + j * 32;
            o_row[d] = float_to_bf16(O_reg[r * FA512_DPT + j] * inv_sum);
        }
    }
}

/* Two-query-head variant for devices where the wider BC=48/64 kernels do not
 * fit. It uses two warps per block, shares the K/V tile across two adjacent Q
 * heads, and keeps the exact same online-softmax math as the single-head path. */
#define FA512_Q2_BC16 16
#define FA512_Q2_BC32 32
#define FA512_Q2_QH 2
#define FA512_Q2_BR 16
#define FA512_Q2_TILE 16
#define FA512_Q2_DPT 16
#define FA512_Q2_HD 512
#define FA512_Q2_CLOCK_Q_LOAD 0
#define FA512_Q2_CLOCK_KV_LOAD 1
#define FA512_Q2_CLOCK_QK 2
#define FA512_Q2_CLOCK_SOFTMAX 3
#define FA512_Q2_CLOCK_PV 4
#define FA512_Q2_CLOCK_FINAL 5
#define FA512_Q2_CLOCK_TOTAL 6
#define FA512_Q2_CLOCK_BLOCKS 7
#define FA512_Q2_CLOCK_TILES 8

__device__ __forceinline__ unsigned long long krasis_globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void hd512_q2_clock_add(
    unsigned long long* __restrict__ debug_clocks,
    int slot,
    unsigned long long delta
) {
    if (debug_clocks && delta > 0) {
        atomicAdd(&debug_clocks[slot], delta);
    }
}

template <bool TIMED, int FA512_Q2_BC>
__device__ __forceinline__ void flash_attn_tiled_hd512_full_q2_impl(
    char* smem_fa512_q2,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,     /* [M, num_q_heads, 512] */
    const __nv_bfloat16* __restrict__ k_cur, /* [M, num_kv_heads, 512] */
    const __nv_bfloat16* __restrict__ v_cur, /* [M, num_kv_heads, 512] */
    int M, int num_q_heads, int num_kv_heads, float softmax_scale,
    unsigned long long* __restrict__ debug_clocks)
{
    int q_base = blockIdx.x * FA512_Q2_BR;
    int qh_group = blockIdx.y * FA512_Q2_QH;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int qh = qh_group + warp;
    if (qh >= num_q_heads) return;

    int q_per_kv = num_q_heads / num_kv_heads;
    int kv_h = qh_group / q_per_kv;

    __nv_bfloat16* s_q = (__nv_bfloat16*)smem_fa512_q2;                     /* [2, BR, 512] */
    __nv_bfloat16* s_k = s_q + FA512_Q2_QH * FA512_Q2_BR * FA512_Q2_HD;     /* [BC, 512] */
    __nv_bfloat16* s_v = s_k + FA512_Q2_BC * FA512_Q2_HD;                  /* [BC, 512] */
    float* s_scores = (float*)(s_v + FA512_Q2_BC * FA512_Q2_HD);           /* [2, BR, BC] */
    __nv_bfloat16* s_p = (__nv_bfloat16*)(s_scores + FA512_Q2_QH * FA512_Q2_BR * FA512_Q2_BC);
    float* s_o_tmp = (float*)(s_p + FA512_Q2_QH * FA512_Q2_BR * FA512_Q2_BC);

    __nv_bfloat16* s_q_w = s_q + warp * FA512_Q2_BR * FA512_Q2_HD;
    float* s_scores_w = s_scores + warp * FA512_Q2_BR * FA512_Q2_BC;
    __nv_bfloat16* s_p_w = s_p + warp * FA512_Q2_BR * FA512_Q2_BC;
    float* s_o_tmp_w = s_o_tmp + warp * FA512_Q2_TILE * FA512_Q2_TILE;

    unsigned long long t_block_start = 0;
    unsigned long long t_phase = 0;
    if (TIMED && warp == 0 && lane == 0) {
        t_block_start = krasis_globaltimer_ns();
        t_phase = t_block_start;
    }

    int q_stride = num_q_heads * FA512_Q2_HD;
    for (int idx = lane; idx < FA512_Q2_BR * FA512_Q2_HD; idx += 32) {
        int r = idx / FA512_Q2_HD;
        int d = idx % FA512_Q2_HD;
        int qi = q_base + r;
        s_q_w[r * FA512_Q2_HD + d] = (qi < M)
            ? q[(int64_t)qi * q_stride + qh * FA512_Q2_HD + d]
            : float_to_bf16(0.0f);
    }
    __syncthreads();
    if (TIMED && warp == 0 && lane == 0) {
        unsigned long long t = krasis_globaltimer_ns();
        hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_Q_LOAD, t - t_phase);
        t_phase = t;
    }

    float row_max_arr[FA512_Q2_BR], row_sum_arr[FA512_Q2_BR];
    float O_reg[FA512_Q2_BR * FA512_Q2_DPT];
    for (int r = 0; r < FA512_Q2_BR; r++) {
        row_max_arr[r] = -1e30f;
        row_sum_arr[r] = 0.0f;
    }
    for (int i = 0; i < FA512_Q2_BR * FA512_Q2_DPT; i++) O_reg[i] = 0.0f;

    int block_last_qi = min(q_base + FA512_Q2_BR - 1, M - 1);
    int block_max_kv = (q_base < M) ? (block_last_qi + 1) : 0;

    for (int kv_start = 0; kv_start < block_max_kv; kv_start += FA512_Q2_BC) {
        int tile_end = min(kv_start + FA512_Q2_BC, block_max_kv);
        int tile_size = tile_end - kv_start;

        for (int idx = threadIdx.x; idx < FA512_Q2_BC * FA512_Q2_HD; idx += 64) {
            int ki = idx / FA512_Q2_HD;
            int d = idx % FA512_Q2_HD;
            __nv_bfloat16 kval, vval;
            if (ki < tile_size) {
                int abs_pos = kv_start + ki;
                int64_t off = ((int64_t)abs_pos * num_kv_heads + kv_h) * FA512_Q2_HD + d;
                kval = k_cur[off];
                vval = v_cur[off];
            } else {
                kval = float_to_bf16(0.0f);
                vval = float_to_bf16(0.0f);
            }
            s_k[ki * FA512_Q2_HD + d] = kval;
            s_v[ki * FA512_Q2_HD + d] = vval;
        }
        __syncthreads();
        if (TIMED && warp == 0 && lane == 0) {
            unsigned long long t = krasis_globaltimer_ns();
            hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_KV_LOAD, t - t_phase);
            atomicAdd(&debug_clocks[FA512_Q2_CLOCK_TILES], 1ull);
            t_phase = t;
        }

        for (int nj = 0; nj < FA512_Q2_BC / FA512_Q2_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> s_frag;
            nvcuda::wmma::fill_fragment(s_frag, 0.0f);

            for (int kk = 0; kk < FA512_Q2_HD / FA512_Q2_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> q_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::col_major> k_frag;

                nvcuda::wmma::load_matrix_sync(q_frag,
                    &s_q_w[kk * FA512_Q2_TILE], FA512_Q2_HD);
                nvcuda::wmma::load_matrix_sync(k_frag,
                    &s_k[nj * FA512_Q2_TILE * FA512_Q2_HD + kk * FA512_Q2_TILE],
                    FA512_Q2_HD);

                nvcuda::wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            nvcuda::wmma::store_matrix_sync(
                &s_scores_w[nj * FA512_Q2_TILE], s_frag, FA512_Q2_BC,
                nvcuda::wmma::mem_row_major);
        }
        __syncwarp();
        if (TIMED && warp == 0 && lane == 0) {
            unsigned long long t = krasis_globaltimer_ns();
            hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_QK, t - t_phase);
            t_phase = t;
        }

        for (int r = 0; r < FA512_Q2_BR; r++) {
            int qi = q_base + r;
            if (qi >= M) continue;
            float* row = &s_scores_w[r * FA512_Q2_BC];

            float local_max = -1e30f;
            for (int c = lane; c < FA512_Q2_BC; c += 32) {
                int abs_kv = kv_start + c;
                if (c < tile_size && abs_kv <= qi) {
                    row[c] *= softmax_scale;
                    local_max = fmaxf(local_max, row[c]);
                } else {
                    row[c] = -1e30f;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

            float old_max = row_max_arr[r];
            float new_max = fmaxf(old_max, local_max);
            float rescale = __expf(old_max - new_max);
            row_max_arr[r] = new_max;

            float local_sum = 0.0f;
            for (int c = lane; c < FA512_Q2_BC; c += 32) {
                float p = __expf(row[c] - new_max);
                row[c] = p;
                local_sum += p;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

            row_sum_arr[r] = row_sum_arr[r] * rescale + local_sum;

            for (int j = 0; j < FA512_Q2_DPT; j++)
                O_reg[r * FA512_Q2_DPT + j] *= rescale;

            for (int c = lane; c < FA512_Q2_BC; c += 32)
                s_p_w[r * FA512_Q2_BC + c] = float_to_bf16(row[c]);
        }
        __syncwarp();
        if (TIMED && warp == 0 && lane == 0) {
            unsigned long long t = krasis_globaltimer_ns();
            hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_SOFTMAX, t - t_phase);
            t_phase = t;
        }

        for (int nj = 0; nj < FA512_Q2_HD / FA512_Q2_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_partial;
            nvcuda::wmma::fill_fragment(o_partial, 0.0f);

            for (int kk = 0; kk < FA512_Q2_BC / FA512_Q2_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> p_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> v_frag;

                nvcuda::wmma::load_matrix_sync(p_frag,
                    &s_p_w[kk * FA512_Q2_TILE], FA512_Q2_BC);
                nvcuda::wmma::load_matrix_sync(v_frag,
                    &s_v[kk * FA512_Q2_TILE * FA512_Q2_HD + nj * FA512_Q2_TILE],
                    FA512_Q2_HD);

                nvcuda::wmma::mma_sync(o_partial, p_frag, v_frag, o_partial);
            }

            nvcuda::wmma::store_matrix_sync(s_o_tmp_w, o_partial,
                FA512_Q2_TILE, nvcuda::wmma::mem_row_major);
            __syncwarp();

            int first_t = (nj * FA512_Q2_TILE) % 32;
            int my_col = lane - first_t;
            if (my_col >= 0 && my_col < FA512_Q2_TILE) {
                int d_global = nj * FA512_Q2_TILE + my_col;
                int j = d_global / 32;
                for (int r = 0; r < FA512_Q2_BR; r++)
                    O_reg[r * FA512_Q2_DPT + j] += s_o_tmp_w[r * FA512_Q2_TILE + my_col];
            }
            __syncwarp();
        }
        __syncthreads();
        if (TIMED && warp == 0 && lane == 0) {
            unsigned long long t = krasis_globaltimer_ns();
            hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_PV, t - t_phase);
            t_phase = t;
        }
    }

    for (int r = 0; r < FA512_Q2_BR; r++) {
        int qi = q_base + r;
        if (qi >= M) continue;
        float inv_sum = (row_sum_arr[r] > 0.0f) ? (1.0f / row_sum_arr[r]) : 0.0f;
        __nv_bfloat16* o_row = out + ((int64_t)qi * num_q_heads + qh) * FA512_Q2_HD;
        for (int j = 0; j < FA512_Q2_DPT; j++) {
            int d = lane + j * 32;
            o_row[d] = float_to_bf16(O_reg[r * FA512_Q2_DPT + j] * inv_sum);
        }
    }
    if (TIMED && warp == 0 && lane == 0) {
        unsigned long long t = krasis_globaltimer_ns();
        hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_FINAL, t - t_phase);
        hd512_q2_clock_add(debug_clocks, FA512_Q2_CLOCK_TOTAL, t - t_block_start);
        atomicAdd(&debug_clocks[FA512_Q2_CLOCK_BLOCKS], 1ull);
    }
}

extern "C" __global__ void flash_attn_tiled_hd512_full_q2_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale)
{
    extern __shared__ char smem_fa512_q2[];
    flash_attn_tiled_hd512_full_q2_impl<false, FA512_Q2_BC16>(
        smem_fa512_q2, out, q, k_cur, v_cur, M, num_q_heads, num_kv_heads,
        softmax_scale, nullptr);
}

extern "C" __global__ void flash_attn_tiled_hd512_full_q2_timed_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale,
    unsigned long long* __restrict__ debug_clocks)
{
    extern __shared__ char smem_fa512_q2[];
    flash_attn_tiled_hd512_full_q2_impl<true, FA512_Q2_BC16>(
        smem_fa512_q2, out, q, k_cur, v_cur, M, num_q_heads, num_kv_heads,
        softmax_scale, debug_clocks);
}

extern "C" __global__ void flash_attn_tiled_hd512_full_q2_bc32_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale)
{
    extern __shared__ char smem_fa512_q2[];
    flash_attn_tiled_hd512_full_q2_impl<false, FA512_Q2_BC32>(
        smem_fa512_q2, out, q, k_cur, v_cur, M, num_q_heads, num_kv_heads,
        softmax_scale, nullptr);
}

extern "C" __global__ void flash_attn_tiled_hd512_full_q2_bc32_timed_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale,
    unsigned long long* __restrict__ debug_clocks)
{
    extern __shared__ char smem_fa512_q2[];
    flash_attn_tiled_hd512_full_q2_impl<true, FA512_Q2_BC32>(
        smem_fa512_q2, out, q, k_cur, v_cur, M, num_q_heads, num_kv_heads,
        softmax_scale, debug_clocks);
}

/* Wider variants for devices with larger opt-in shared memory. These keep the
 * same full-attention math as flash_attn_tiled_hd512_full_kernel, but process
 * more KV columns per tile when runtime capability allows it. */
#define FA512_WIDE_BR 16
#define FA512_WIDE_TILE 16
#define FA512_WIDE_DPT 16
#define FA512_WIDE_HD 512

template <int FA512_WIDE_BC>
__device__ __forceinline__ void flash_attn_tiled_hd512_full_wide_impl(
    char* smem_fa512,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale)
{
    int q_base = blockIdx.x * FA512_WIDE_BR;
    int qh = blockIdx.y;
    int q_per_kv = num_q_heads / num_kv_heads;
    int kv_h = qh / q_per_kv;
    int lane = threadIdx.x;

    __nv_bfloat16* s_q = (__nv_bfloat16*)smem_fa512;
    __nv_bfloat16* s_k = s_q + FA512_WIDE_BR * FA512_WIDE_HD;
    __nv_bfloat16* s_v = s_k + FA512_WIDE_BC * FA512_WIDE_HD;
    float* s_scores = (float*)(s_v + FA512_WIDE_BC * FA512_WIDE_HD);
    __nv_bfloat16* s_p = (__nv_bfloat16*)(s_scores + FA512_WIDE_BR * FA512_WIDE_BC);
    float* s_o_tmp = (float*)(s_p + FA512_WIDE_BR * FA512_WIDE_BC);

    int q_stride = num_q_heads * FA512_WIDE_HD;
    for (int idx = lane; idx < FA512_WIDE_BR * FA512_WIDE_HD; idx += 32) {
        int r = idx / FA512_WIDE_HD;
        int d = idx % FA512_WIDE_HD;
        int qi = q_base + r;
        s_q[r * FA512_WIDE_HD + d] = (qi < M)
            ? q[(int64_t)qi * q_stride + qh * FA512_WIDE_HD + d]
            : float_to_bf16(0.0f);
    }
    __syncthreads();

    float row_max_arr[FA512_WIDE_BR], row_sum_arr[FA512_WIDE_BR];
    float O_reg[FA512_WIDE_BR * FA512_WIDE_DPT];
    for (int r = 0; r < FA512_WIDE_BR; r++) {
        row_max_arr[r] = -1e30f;
        row_sum_arr[r] = 0.0f;
    }
    for (int i = 0; i < FA512_WIDE_BR * FA512_WIDE_DPT; i++) O_reg[i] = 0.0f;

    int block_last_qi = min(q_base + FA512_WIDE_BR - 1, M - 1);
    int block_max_kv = (q_base < M) ? (block_last_qi + 1) : 0;

    for (int kv_start = 0; kv_start < block_max_kv; kv_start += FA512_WIDE_BC) {
        int tile_end = min(kv_start + FA512_WIDE_BC, block_max_kv);
        int tile_size = tile_end - kv_start;

        for (int idx = lane; idx < FA512_WIDE_BC * FA512_WIDE_HD; idx += 32) {
            int ki = idx / FA512_WIDE_HD;
            int d = idx % FA512_WIDE_HD;
            __nv_bfloat16 kval, vval;
            if (ki < tile_size) {
                int abs_pos = kv_start + ki;
                int64_t off = ((int64_t)abs_pos * num_kv_heads + kv_h) * FA512_WIDE_HD + d;
                kval = k_cur[off];
                vval = v_cur[off];
            } else {
                kval = float_to_bf16(0.0f);
                vval = float_to_bf16(0.0f);
            }
            s_k[ki * FA512_WIDE_HD + d] = kval;
            s_v[ki * FA512_WIDE_HD + d] = vval;
        }
        __syncwarp();

        for (int nj = 0; nj < FA512_WIDE_BC / FA512_WIDE_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> s_frag;
            nvcuda::wmma::fill_fragment(s_frag, 0.0f);

            for (int kk = 0; kk < FA512_WIDE_HD / FA512_WIDE_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> q_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::col_major> k_frag;

                nvcuda::wmma::load_matrix_sync(q_frag,
                    &s_q[kk * FA512_WIDE_TILE], FA512_WIDE_HD);
                nvcuda::wmma::load_matrix_sync(k_frag,
                    &s_k[nj * FA512_WIDE_TILE * FA512_WIDE_HD + kk * FA512_WIDE_TILE],
                    FA512_WIDE_HD);

                nvcuda::wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            nvcuda::wmma::store_matrix_sync(
                &s_scores[nj * FA512_WIDE_TILE], s_frag, FA512_WIDE_BC,
                nvcuda::wmma::mem_row_major);
        }
        __syncwarp();

        for (int r = 0; r < FA512_WIDE_BR; r++) {
            int qi = q_base + r;
            if (qi >= M) continue;
            float* row = &s_scores[r * FA512_WIDE_BC];

            float local_max = -1e30f;
            for (int c = lane; c < FA512_WIDE_BC; c += 32) {
                int abs_kv = kv_start + c;
                if (c < tile_size && abs_kv <= qi) {
                    row[c] *= softmax_scale;
                    local_max = fmaxf(local_max, row[c]);
                } else {
                    row[c] = -1e30f;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

            float old_max = row_max_arr[r];
            float new_max = fmaxf(old_max, local_max);
            float rescale = __expf(old_max - new_max);
            row_max_arr[r] = new_max;

            float local_sum = 0.0f;
            for (int c = lane; c < FA512_WIDE_BC; c += 32) {
                float p = __expf(row[c] - new_max);
                row[c] = p;
                local_sum += p;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

            row_sum_arr[r] = row_sum_arr[r] * rescale + local_sum;

            for (int j = 0; j < FA512_WIDE_DPT; j++)
                O_reg[r * FA512_WIDE_DPT + j] *= rescale;

            for (int c = lane; c < FA512_WIDE_BC; c += 32)
                s_p[r * FA512_WIDE_BC + c] = float_to_bf16(row[c]);
        }
        __syncwarp();

        for (int nj = 0; nj < FA512_WIDE_HD / FA512_WIDE_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_partial;
            nvcuda::wmma::fill_fragment(o_partial, 0.0f);

            for (int kk = 0; kk < FA512_WIDE_BC / FA512_WIDE_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> p_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> v_frag;

                nvcuda::wmma::load_matrix_sync(p_frag,
                    &s_p[kk * FA512_WIDE_TILE], FA512_WIDE_BC);
                nvcuda::wmma::load_matrix_sync(v_frag,
                    &s_v[kk * FA512_WIDE_TILE * FA512_WIDE_HD + nj * FA512_WIDE_TILE],
                    FA512_WIDE_HD);

                nvcuda::wmma::mma_sync(o_partial, p_frag, v_frag, o_partial);
            }

            nvcuda::wmma::store_matrix_sync(s_o_tmp, o_partial,
                FA512_WIDE_TILE, nvcuda::wmma::mem_row_major);
            __syncwarp();

            int first_t = (nj * FA512_WIDE_TILE) % 32;
            int my_col = lane - first_t;
            if (my_col >= 0 && my_col < FA512_WIDE_TILE) {
                int d_global = nj * FA512_WIDE_TILE + my_col;
                int j = d_global / 32;
                for (int r = 0; r < FA512_WIDE_BR; r++)
                    O_reg[r * FA512_WIDE_DPT + j] += s_o_tmp[r * FA512_WIDE_TILE + my_col];
            }
            __syncwarp();
        }
        __syncwarp();
    }

    for (int r = 0; r < FA512_WIDE_BR; r++) {
        int qi = q_base + r;
        if (qi >= M) continue;
        float inv_sum = (row_sum_arr[r] > 0.0f) ? (1.0f / row_sum_arr[r]) : 0.0f;
        __nv_bfloat16* o_row = out + ((int64_t)qi * num_q_heads + qh) * FA512_WIDE_HD;
        for (int j = 0; j < FA512_WIDE_DPT; j++) {
            int d = lane + j * 32;
            o_row[d] = float_to_bf16(O_reg[r * FA512_WIDE_DPT + j] * inv_sum);
        }
    }
}

extern "C" __global__ void flash_attn_tiled_hd512_full_bc48_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale)
{
    extern __shared__ char smem_fa512_wide[];
    flash_attn_tiled_hd512_full_wide_impl<48>(
        smem_fa512_wide, out, q, k_cur, v_cur, M, num_q_heads, num_kv_heads,
        softmax_scale);
}

extern "C" __global__ void flash_attn_tiled_hd512_full_bc64_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cur,
    const __nv_bfloat16* __restrict__ v_cur,
    int M, int num_q_heads, int num_kv_heads, float softmax_scale)
{
    extern __shared__ char smem_fa512_wide[];
    flash_attn_tiled_hd512_full_wide_impl<64>(
        smem_fa512_wide, out, q, k_cur, v_cur, M, num_q_heads, num_kv_heads,
        softmax_scale);
}


/* ══════════════════════════════════════════════════════════════════════════
 *  GPU-only MoE routing — replaces CPU round-trip for token binning
 *
 *  Two-phase approach:
 *  Phase 1: moe_count_experts_kernel
 *    Count how many tokens are assigned to each expert. Atomic increments.
 *    Grid: (M, 1, 1), each thread handles one token's topk assignments.
 *  Phase 2: moe_build_maps_kernel
 *    Build gather_src_map, gather_weight_map, and per-expert offsets.
 *    Grid: (M, 1, 1), each thread writes its topk entries.
 *
 *  This eliminates two stream syncs and CPU binning from forward_moe.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Phase 1: count tokens per expert using atomicAdd */
extern "C" __global__ void moe_count_experts_kernel(
    int* __restrict__ expert_counts,  /* [E] output: count per expert */
    const int* __restrict__ topk_ids, /* [M, topk] */
    int M, int topk, int E
) {
    int t = blockIdx.x;
    if (t >= M) return;
    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid >= 0 && eid < E) {
            atomicAdd(&expert_counts[eid], 1);
        }
    }
}

/* Phase 2: prefix sum on expert_counts to get offsets, then build maps.
 * This runs on a single block since E is typically small (512).
 * Performs prefix sum in shared memory, then each token writes its entries. */
extern "C" __global__ void moe_prefix_sum_kernel(
    int* __restrict__ expert_offsets,  /* [E+1] output: exclusive prefix sum */
    const int* __restrict__ expert_counts, /* [E] */
    int E
) {
    /* Single-block prefix sum over E experts.
     * Use char smem and reinterpret to avoid type conflict with global extern __shared__ float smem[]. */
    extern __shared__ char smem_raw_ps[];
    int* smem_i = reinterpret_cast<int*>(smem_raw_ps);
    int tid = threadIdx.x;

    /* Load counts into shared memory */
    for (int i = tid; i < E; i += blockDim.x) {
        smem_i[i] = expert_counts[i];
    }
    __syncthreads();

    /* Sequential prefix sum (E is small, typically 512) */
    if (tid == 0) {
        expert_offsets[0] = 0;
        for (int i = 0; i < E; i++) {
            expert_offsets[i + 1] = expert_offsets[i] + smem_i[i];
        }
    }
}

/* Phase 3: build gather maps using atomic offsets.
 * scale_factor is applied to weights (e.g. routed_scaling_factor for MoE). */
extern "C" __global__ void moe_build_maps_kernel(
    int* __restrict__ gather_src_map,     /* [total_active] output */
    float* __restrict__ gather_weight_map, /* [total_active] output */
    int* __restrict__ write_offsets,       /* [E] atomically incremented write positions */
    const int* __restrict__ topk_ids,      /* [M, topk] */
    const float* __restrict__ topk_weights,/* [M, topk] */
    const int* __restrict__ expert_offsets, /* [E+1] base offsets */
    int M, int topk, int E,
    float scale_factor
) {
    int t = blockIdx.x;
    if (t >= M) return;
    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid >= 0 && eid < E) {
            int base = expert_offsets[eid];
            int slot = atomicAdd(&write_offsets[eid], 1);
            int pos = base + slot;
            gather_src_map[pos] = t;
            gather_weight_map[pos] = topk_weights[t * topk + k] * scale_factor;
        }
    }
}

/* Stable map builder for validation paths that must be repeatable.
 * Rows are ordered by expert, then token position, then top-k slot. */
extern "C" __global__ void moe_build_maps_stable_kernel(
    int* __restrict__ gather_src_map,      /* [total_active] output */
    float* __restrict__ gather_weight_map, /* [total_active] output */
    int* __restrict__ write_offsets,       /* [E] output: count written per expert */
    const int* __restrict__ topk_ids,      /* [M, topk] */
    const float* __restrict__ topk_weights,/* [M, topk] */
    const int* __restrict__ expert_offsets,/* [E+1] base offsets */
    int M, int topk, int E,
    float scale_factor
) {
    int eid = blockIdx.x;
    if (eid >= E) return;

    extern __shared__ int stable_counts[];
    int tid = threadIdx.x;
    int start_t = ((long long)M * tid) / blockDim.x;
    int end_t = ((long long)M * (tid + 1)) / blockDim.x;

    int local_count = 0;
    for (int t = start_t; t < end_t; t++) {
        for (int k = 0; k < topk; k++) {
            int routed_eid = topk_ids[t * topk + k];
            if (routed_eid == eid) {
                local_count++;
            }
        }
    }

    stable_counts[tid] = local_count;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add = 0;
        if (tid >= offset) {
            add = stable_counts[tid - offset];
        }
        __syncthreads();
        stable_counts[tid] += add;
        __syncthreads();
    }

    int pos = expert_offsets[eid] + stable_counts[tid] - local_count;
    const int expert_end = expert_offsets[eid + 1];
    for (int t = start_t; t < end_t; t++) {
        for (int k = 0; k < topk; k++) {
            int routed_eid = topk_ids[t * topk + k];
            if (routed_eid == eid) {
                if (pos < expert_end) {
                    gather_src_map[pos] = t;
                    gather_weight_map[pos] = topk_weights[t * topk + k] * scale_factor;
                    pos++;
                }
            }
        }
    }

    if (tid == blockDim.x - 1) {
        write_offsets[eid] = stable_counts[tid];
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Fused MoE support kernels — moe_align_block_size and gather/scatter
 *
 *  These kernels produce the sorted_token_ids, expert_ids, and
 *  num_tokens_post_padded arrays required by the fused MarlinDefault
 *  kernel from sgl_kernel. The fused kernel processes ALL experts in
 *  one launch — our Rust code assembles contiguous expert weight buffers
 *  and calls MarlinDefault directly via dlopen.
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * moe_align_block_size: given topk_ids [M, topk], produce:
 *   sorted_token_ids [total_padded]  — which token maps to each sorted position
 *   expert_ids       [num_blocks]    — which expert each block processes
 *   num_tokens_post  [1]             — total_padded
 *
 * Three phases (launched separately from Rust):
 *   Phase 1: count tokens per expert (reuses moe_count_experts_kernel)
 *   Phase 2: prefix sum with block_size padding (this kernel)
 *   Phase 3: scatter tokens into sorted positions + fill expert_ids (this kernel)
 */

/* Phase 2: Prefix sum with block_size padding.
 * Input:  expert_counts [E]
 * Output: expert_offsets [E+1] (padded prefix sum), num_tokens_post [1]
 * Grid: (1,1,1), Block: (threads,1,1) with threads >= E
 */
extern "C" __global__ void moe_padded_prefix_sum_kernel(
    int* __restrict__ expert_offsets,
    int* __restrict__ num_tokens_post,
    const int* __restrict__ expert_counts,
    int E, int block_size
) {
    extern __shared__ int ps_smem[];
    int tid = threadIdx.x;
    int val = (tid < E) ? expert_counts[tid] : 0;
    // Pad to block_size
    int padded = ((val + block_size - 1) / block_size) * block_size;
    ps_smem[tid] = padded;
    __syncthreads();

    // Simple serial prefix sum (E is small, <= 1024)
    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < E; i++) {
            expert_offsets[i] = running;
            running += ps_smem[i];
        }
        expert_offsets[E] = running;
        num_tokens_post[0] = running;
    }
}

/* Phase 3: Scatter tokens into sorted positions.
 * Grid: (M, 1, 1), Block: (1, 1, 1)
 * Each thread handles one token, scatters its topk slots.
 */
extern "C" __global__ void moe_scatter_sorted_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ write_offsets,       // [E] atomically incremented
    const int* __restrict__ topk_ids,      // [M, topk]
    const int* __restrict__ expert_offsets, // [E+1]
    int M, int topk, int E
) {
    int t = blockIdx.x;
    if (t >= M) return;
    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid >= 0 && eid < E) {
            int base = expert_offsets[eid];
            int slot = atomicAdd(&write_offsets[eid], 1);
            sorted_token_ids[base + slot] = t * topk + k;  // vLLM format: token*topk+slot
        }
    }
}

/* Phase 4: finalize expert_ids and padding after scatter completes.
 * Grid: (E, 1, 1), Block: (1, 1, 1)
 * Each block handles one expert after the scatter writes are visible on-stream.
 */
extern "C" __global__ void moe_finalize_sorted_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids_out,
    const int* __restrict__ expert_offsets, // [E+1]
    const int* __restrict__ expert_counts,  // [E]
    int M, int topk, int E, int block_size
) {
    int e = blockIdx.x;
    if (e >= E) return;
    int base = expert_offsets[e];
    int count = expert_counts[e];
    int padded = expert_offsets[e + 1] - base;
    int num_blocks = padded / block_size;
    // Fill padding slots with M*topk (Marlin kernel checks >= prob_m * top_k)
    for (int p = count; p < padded; p++) {
        sorted_token_ids[base + p] = M * topk; // padding sentinel
    }
    // Fill expert_ids for each block
    int block_start = base / block_size;
    for (int b = 0; b < num_blocks; b++) {
        expert_ids_out[block_start + b] = e;
    }
}

/* Replicate hidden states for fused MoE: out[i] = src[i / topk] for i in [0, M*topk).
 * Creates [M*topk, dim] buffer where each token's hidden state appears topk times.
 * Required for the top_k=1 trick: avoids C_tmp collision in fp32_reduce.
 * Grid: (M * topk, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void moe_replicate_hidden_kernel(
    __nv_bfloat16* __restrict__ out,       // [M*topk, dim]
    const __nv_bfloat16* __restrict__ src, // [M, dim]
    int dim, int M, int topk
) {
    int idx = blockIdx.x;  // 0 to M*topk-1
    int token = idx / topk;
    if (token >= M) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[(int64_t)idx * dim + i] = src[(int64_t)token * dim + i];
    }
}

/* Gather tokens by sorted order for fused MoE input.
 * sorted_token_ids maps positions in the sorted sequence to original token indices.
 * Grid: (total_padded, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void moe_gather_sorted_kernel(
    __nv_bfloat16* __restrict__ out,       // [total_padded, dim]
    const __nv_bfloat16* __restrict__ src, // [M, dim]
    const int* __restrict__ sorted_ids,    // [total_padded]
    int dim, int M
) {
    int pos = blockIdx.x;
    int tid = sorted_ids[pos];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (tid < M) {
            out[pos * dim + i] = src[tid * dim + i];
        } else {
            out[pos * dim + i] = __float2bfloat16(0.0f); // padding
        }
    }
}

/* Scatter fused MoE output back to per-token FP32 accumulator.
 * The fused kernel outputs [total_sorted, hidden] with topk_weights already applied
 * (mul_topk_weights=True). sorted_token_ids[pos] gives the original token index.
 * Multiple sorted positions map to the same token (one per topk expert).
 *
 * Accumulate in sorted-position order for each destination token/column. This
 * avoids FP32 atomicAdd reduction-order nondeterminism.
 *
 * Grid: (M, ceil(hidden / blockDim.x), 1), Block: (columns, 1, 1)
 */
extern "C" __global__ void moe_scatter_fused_kernel(
    float* __restrict__ accum,              // [M, hidden] FP32 accumulator (pre-zeroed)
    const __nv_bfloat16* __restrict__ src,  // [total_sorted, hidden] fused output
    const int* __restrict__ sorted_ids,     // [total_sorted] maps sorted pos -> token*topk+slot
    int hidden, int M, float scale_factor,
    int topk,                               // topk for vLLM-format sorted_ids (divide to get token)
    int total_sorted
) {
    int token = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= M || col >= hidden || total_sorted <= 0) return;

    float sum = 0.0f;
    for (int pos = 0; pos < total_sorted; pos++) {
        int sid = sorted_ids[pos];
        if (sid < 0) continue;
        int tid = topk > 1 ? sid / topk : sid;
        if (tid == token) {
            sum += bf16_to_float(src[(int64_t)pos * hidden + col]) * scale_factor;
        }
    }
    accum[(int64_t)token * hidden + col] = sum;
}


/* Scatter fused MoE w2 output back to per-token FP32 accumulator with topk weights.
 * The fused Marlin w2 kernel writes compact rows directly at sorted_id = token*topk+slot,
 * so the routed contribution is accumulated in deterministic slot order by a block that
 * owns one destination token and column tile. The padded sorted_ids metadata is still
 * produced for the fused GEMM, but it is not needed for the final scatter once rows are
 * compact.
 *
 * Grid: (M, ceil(hidden / blockDim.x), 1), Block: (columns, 1, 1)
 */
extern "C" __global__ void moe_scatter_weighted_kernel(
    float* __restrict__ accum,              // [M, hidden] FP32 accumulator (pre-zeroed)
    const __nv_bfloat16* __restrict__ src,  // [M*topk, hidden] fused w2 output (indexed by sorted_id)
    const int* __restrict__ sorted_ids,     // [total_sorted] maps sorted position -> token*topk+slot
    const float* __restrict__ topk_weights, // [M * topk] routing weights
    int hidden, int M, int topk, int total_sorted, float scale_factor
) {
    int token = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int m_topk = M * topk;
    if (token >= M || col >= hidden || total_sorted < m_topk) return;

    float sum = 0.0f;
    int base = token * topk;
    for (int slot = 0; slot < topk; slot++) {
        int row = base + slot;
        float w = topk_weights[row] * scale_factor;
        if (w != 0.0f) {
            sum += bf16_to_float(src[(int64_t)row * hidden + col]) * w;
        }
    }
    accum[(int64_t)token * hidden + col] = sum;
}


/* ── Init / Cleanup (no-ops for the PTX kernel path) ──────────────────── */

extern "C" int krasis_prefill_init(int device) {
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" void krasis_prefill_cleanup(void) {
    /* Nothing to clean up for PTX kernels */
}

// ── 4-bit PolarQuant KV Cache (Prefill) ──────────────────────────────────

// 16-level codebook for quantized angles (normalized components).
__device__ __constant__ float polar4_codebook_p[16] = {
    -0.6892f, -0.5241f, -0.4115f, -0.3206f, -0.2412f, -0.1685f, -0.0997f, -0.0330f,
     0.0330f,  0.0997f,  0.1685f,  0.2412f,  0.3206f,  0.4115f,  0.5241f,  0.6892f
};

// Fixed sign flip for SRR (16 elements)
__device__ __constant__ float polar4_signs_p[16] = {
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// Fast Hadamard Transform for 16 elements (in-place)
__device__ inline void fht16_p(float* x) {
    float a, b;
    #pragma unroll
    for (int i = 0; i < 8; i++) { a = x[i]; b = x[i+8]; x[i] = a + b; x[i+8] = a - b; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { a = x[i]; b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
    #pragma unroll
    for (int i = 8; i < 12; i++) { a = x[i]; b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
    #pragma unroll
    for (int i = 0; i < 16; i += 4) {
        a = x[i]; b = x[i+2]; x[i] = a + b; x[i+2] = a - b;
        a = x[i+1]; b = x[i+3]; x[i+1] = a + b; x[i+3] = a - b;
    }
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        a = x[i]; b = x[i+1]; x[i] = a + b; x[i+1] = a - b;
    }
}

__device__ inline int quantize_polar4_p(float val) {
    int best_idx = 0;
    float min_diff = fabsf(val - polar4_codebook_p[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float diff = fabsf(val - polar4_codebook_p[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

__device__ inline float tq4_codebook_value_p(int idx, int head_dim) {
    // vLLM turboquant_4bit_nc Lloyd-Max centroids for N(0, 1), scaled to
    // N(0, 1/head_dim). These are algorithm constants, not model calibration.
    static const float lloyd4_unit[16] = {
        -2.7309222221f, -2.0684471130f, -1.6178817749f, -1.2562575340f,
        -0.9424482584f, -0.6568799019f, -0.3881377876f, -0.1284276545f,
         0.1284276545f,  0.3881377876f,  0.6568799019f,  0.9424482584f,
         1.2562575340f,  1.6178817749f,  2.0684471130f,  2.7309222221f,
    };
    return lloyd4_unit[idx] * rsqrtf((float)head_dim);
}

__device__ inline int quantize_tq4_p(float val, int head_dim) {
    int best_idx = 0;
    float best_diff = fabsf(val - tq4_codebook_value_p(0, head_dim));
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float diff = fabsf(val - tq4_codebook_value_p(i, head_dim));
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

__device__ inline void fht_shared_serial_p(float* x, int n) {
    for (int step = 1; step < n; step <<= 1) {
        int jump = step << 1;
        for (int base = 0; base < n; base += jump) {
            for (int j = 0; j < step; j++) {
                float a = x[base + j];
                float b = x[base + j + step];
                x[base + j] = a + b;
                x[base + j + step] = a - b;
            }
        }
    }
}

extern "C" __global__ void kv_cache_append_polar4_kernel(
    unsigned short* __restrict__ k_radius_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ k_angles_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x; // token index in prefill batch
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;

        float k_local[16], v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = __bfloat162float(k[src_offset + i]) * polar4_signs_p[i];
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        fht16_p(k_local);
        fht16_p(v_local);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] *= 0.25f;
            v_local[i] *= 0.25f;
        }

        float k_r = 0.0f, v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_r += k_local[i] * k_local[i];
            v_r += v_local[i] * v_local[i];
        }
        k_r = sqrtf(k_r + 1e-12f);
        v_r = sqrtf(v_r + 1e-12f);

        float inv_k_r = 1.0f / k_r;
        float inv_v_r = 1.0f / v_r;
        unsigned char* k_ang = k_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;

        float k_qnorm2 = 0.0f;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k0 = quantize_polar4_p(k_local[i*2] * inv_k_r);
            int k1 = quantize_polar4_p(k_local[i*2+1] * inv_k_r);
            k_ang[i] = (unsigned char)((k1 << 4) | k0);
            float k0v = polar4_codebook_p[k0];
            float k1v = polar4_codebook_p[k1];
            k_qnorm2 += k0v * k0v + k1v * k1v;
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }

        if (norm_correction & 1) {
            k_r = k_r / sqrtf(k_qnorm2 + 1e-12f);
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }

        __nv_bfloat16 k_rb = __float2bfloat16(k_r);
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        k_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_rb);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_k8v4_kernel(
    __nv_fp8_e4m3* __restrict__ k_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        int dst_offset = dst_pos * kv_stride + block_idx * 16;
        float v_local[16];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_cache[dst_offset + i] = bf16_to_fp8e4m3(k[src_offset + i]);
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);

        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }

        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

__device__ inline void pack_k4_16_p(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = (unsigned char)((codes[i * 2 + 1] << 4) | (codes[i * 2] & 0x0f));
    }
}

__device__ inline int unpack_k4_p(const unsigned char* src, int idx) {
    unsigned char packed = src[idx >> 1];
    return (idx & 1) ? (int)(packed >> 4) : (int)(packed & 0x0f);
}

__device__ inline float quantize_k4_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 7.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-7, min(7, q));
        codes[i] = (unsigned char)(q + 8);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

__device__ inline void pack_k6_16_p(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 12; i++) dst[i] = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int bit = i * 6;
        int byte = bit >> 3;
        int shift = bit & 7;
        unsigned int val = ((unsigned int)codes[i]) & 0x3fu;
        dst[byte] |= (unsigned char)(val << shift);
        if (shift > 2) {
            dst[byte + 1] |= (unsigned char)(val >> (8 - shift));
        }
    }
}

__device__ inline int unpack_k6_p(const unsigned char* src, int idx) {
    int bit = idx * 6;
    int byte = bit >> 3;
    int shift = bit & 7;
    unsigned int val = ((unsigned int)src[byte]) >> shift;
    if (shift > 2) {
        val |= ((unsigned int)src[byte + 1]) << (8 - shift);
    }
    return (int)(val & 0x3fu);
}

__device__ inline float quantize_k6_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 31.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-31, min(31, q));
        codes[i] = (unsigned char)(q + 32);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

__device__ inline void pack_k7_16_p(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 14; i++) dst[i] = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int bit = i * 7;
        int byte = bit >> 3;
        int shift = bit & 7;
        unsigned int val = ((unsigned int)codes[i]) & 0x7fu;
        dst[byte] |= (unsigned char)(val << shift);
        if (shift > 1) {
            dst[byte + 1] |= (unsigned char)(val >> (8 - shift));
        }
    }
}

__device__ inline int unpack_k7_p(const unsigned char* src, int idx) {
    int bit = idx * 7;
    int byte = bit >> 3;
    int shift = bit & 7;
    unsigned int val = ((unsigned int)src[byte]) >> shift;
    if (shift > 1) {
        val |= ((unsigned int)src[byte + 1]) << (8 - shift);
    }
    return (int)(val & 0x7fu);
}

__device__ inline float quantize_k7_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 63.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-63, min(63, q));
        codes[i] = (unsigned char)(q + 64);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

__device__ inline float quantize_k8_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 127.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-127, min(127, q));
        codes[i] = (unsigned char)(q + 128);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

extern "C" __global__ void kv_cache_append_k4v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float kval = __bfloat162float(k[src_offset + i]);
            k_local[i] = kval;
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k4_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 8;
        pack_k4_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_k6v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float kval = __bfloat162float(k[src_offset + i]);
            k_local[i] = kval;
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k6_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_k7v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float kval = __bfloat162float(k[src_offset + i]);
            k_local[i] = kval;
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k7_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 14;
        pack_k7_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_tq4_kernel(
    unsigned short* __restrict__ k_norm_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_meta_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ signs,
    int M,
    int num_kv_heads,
    int head_dim,
    int max_seq,
    int start_pos
) {
    int ti = blockIdx.x;
    int kv_head = blockIdx.y;
    if (ti >= M || kv_head >= num_kv_heads || threadIdx.x != 0) return;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    extern __shared__ float scratch[];
    int kv_stride = num_kv_heads * head_dim;
    int packed_hd = (head_dim + 1) >> 1;
    int token_head = dst_pos * num_kv_heads + kv_head;
    int src_off = ti * kv_stride + kv_head * head_dim;

    float k_norm2 = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = __bfloat162float(k[src_off + d]);
        scratch[d] = val * signs[d];
        k_norm2 += val * val;
    }
    float k_norm = sqrtf(k_norm2 + 1e-12f);
    fht_shared_serial_p(scratch, head_dim);
    float inv_sqrt_hd = rsqrtf((float)head_dim);
    float inv_norm = 1.0f / k_norm;
    float q_norm2 = 0.0f;
    unsigned char* k_pack = k_idx_cache + token_head * packed_hd;
    for (int d = 0; d < head_dim; d += 2) {
        int q0 = quantize_tq4_p(scratch[d] * inv_sqrt_hd * inv_norm, head_dim);
        int q1 = 0;
        q_norm2 += tq4_codebook_value_p(q0, head_dim) * tq4_codebook_value_p(q0, head_dim);
        if (d + 1 < head_dim) {
            q1 = quantize_tq4_p(scratch[d + 1] * inv_sqrt_hd * inv_norm, head_dim);
            q_norm2 += tq4_codebook_value_p(q1, head_dim) * tq4_codebook_value_p(q1, head_dim);
        }
        k_pack[d >> 1] = (unsigned char)((q1 << 4) | q0);
    }
    float corrected_norm = k_norm / sqrtf(q_norm2 + 1e-12f);
    __half k_nb = __float2half(corrected_norm);
    k_norm_cache[token_head] = *reinterpret_cast<unsigned short*>(&k_nb);

    float v_min = 1e30f;
    float v_max = -1e30f;
    for (int d = 0; d < head_dim; d++) {
        float val = __bfloat162float(v[src_off + d]);
        v_min = fminf(v_min, val);
        v_max = fmaxf(v_max, val);
    }
    float scale = (v_max - v_min) * (1.0f / 15.0f);
    if (scale < 1e-8f) scale = 1e-8f;
    float inv_scale = 1.0f / scale;
    unsigned char* v_pack = v_idx_cache + token_head * packed_hd;
    for (int d = 0; d < head_dim; d += 2) {
        int q0 = (int)floorf((__bfloat162float(v[src_off + d]) - v_min) * inv_scale + 0.5f);
        q0 = max(0, min(15, q0));
        int q1 = 0;
        if (d + 1 < head_dim) {
            q1 = (int)floorf((__bfloat162float(v[src_off + d + 1]) - v_min) * inv_scale + 0.5f);
            q1 = max(0, min(15, q1));
        }
        v_pack[d >> 1] = (unsigned char)((q1 << 4) | q0);
    }
    __half scale_b = __float2half(scale);
    __half zero_b = __float2half(v_min);
    v_meta_cache[token_head * 2] = *reinterpret_cast<unsigned short*>(&scale_b);
    v_meta_cache[token_head * 2 + 1] = *reinterpret_cast<unsigned short*>(&zero_b);
}

/* ── Polar4 KV Cache Dequant + Concat for Cross-Chunk FA2 ───────────────
 * Reconstructs Polar4 cache [0..cache_len] back to BF16 K/V in the original
 * domain, then copies current-chunk BF16 K/V [0..m] into [cache_len..cache_len+m].
 *
 * Grid: (cache_len + m, 1, 1), Block: (threads, 1, 1)
 * Threads grid-stride over 16-value blocks within the KV stride.
 */
extern "C" __global__ void kv_cache_dequant_concat_polar4_kernel(
    __nv_bfloat16* __restrict__ out,                /* [cache_len+m, kv_stride] BF16 output */
    const unsigned short* __restrict__ radius_cache,/* [max_seq, num_blocks] BF16 radii */
    const unsigned char* __restrict__ angles_cache, /* [max_seq, num_blocks * 8] packed 4-bit */
    const __nv_bfloat16* __restrict__ kv_new,       /* [m, kv_stride] BF16 current chunk */
    int cache_len,                                  /* number of cached tokens */
    int m,                                          /* current chunk size */
    int kv_stride)                                  /* num_kv_heads * head_dim */
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;

        if (ti < cache_len) {
            float local[16];
            float r = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&radius_cache[ti * num_blocks + block_idx])
            );
            const unsigned char* ang = angles_cache + (ti * num_blocks + block_idx) * 8;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char p = ang[i];
                local[i * 2] = r * polar4_codebook_p[p & 0xF];
                local[i * 2 + 1] = r * polar4_codebook_p[p >> 4];
            }

            // Inverse SRR: x = S * H(y) / 4
            fht16_p(local);
            int64_t dst_off = (int64_t)ti * kv_stride + base;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                out[dst_off + i] = __float2bfloat16(local[i] * 0.25f * polar4_signs_p[i]);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                int64_t dst_off = (int64_t)ti * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = kv_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_append_k6v6_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_scale_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    (void)norm_correction;
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = __bfloat162float(k[src_offset + i]);
            v_local[i] = __bfloat162float(v[src_offset + i]);
        }

        unsigned char codes[16];
        float k_scale = quantize_k6_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        float v_scale = quantize_k6_one_pass_ls_p(v_local, codes);
        unsigned char* v_pack = v_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(v_pack, codes);
        __nv_bfloat16 v_sb = __float2bfloat16(v_scale);
        v_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_sb);
    }
}


extern "C" __global__ void kv_cache_convert_fp8_to_k4v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_fp8_e4m3* __restrict__ k_fp8,
    const __nv_fp8_e4m3* __restrict__ v_fp8,
    int M,
    int kv_stride,
    int max_seq,
    int norm_correction,
    int source_start_pos
) {
    int ti = blockIdx.x;
    if (ti >= M || max_seq <= 0) return;
    int dst_pos = (source_start_pos + ti) % max_seq;

    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = fp8e4m3_to_float(k_fp8[src_offset + i]);
            v_local[i] = fp8e4m3_to_float(v_fp8[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k4_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 8;
        pack_k4_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_convert_fp8_to_k6v6_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_scale_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_fp8_e4m3* __restrict__ k_fp8,
    const __nv_fp8_e4m3* __restrict__ v_fp8,
    int M,
    int kv_stride,
    int max_seq,
    int norm_correction,
    int source_start_pos
) {
    (void)norm_correction;
    int ti = blockIdx.x;
    if (ti >= M || max_seq <= 0) return;
    int dst_pos = (source_start_pos + ti) % max_seq;

    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = fp8e4m3_to_float(k_fp8[src_offset + i]);
            v_local[i] = fp8e4m3_to_float(v_fp8[src_offset + i]);
        }

        unsigned char codes[16];
        float k_scale = quantize_k6_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        float v_scale = quantize_k6_one_pass_ls_p(v_local, codes);
        unsigned char* v_pack = v_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(v_pack, codes);
        __nv_bfloat16 v_sb = __float2bfloat16(v_scale);
        v_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_sb);
    }
}

extern "C" __global__ void kv_cache_append_k8v6_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_scale_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    (void)norm_correction;
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = __bfloat162float(k[src_offset + i]);
            v_local[i] = __bfloat162float(v[src_offset + i]);
        }

        unsigned char codes[16];
        float k_scale = quantize_k8_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) k_pack[i] = codes[i];
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        float v_scale = quantize_k6_one_pass_ls_p(v_local, codes);
        unsigned char* v_pack = v_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(v_pack, codes);
        __nv_bfloat16 v_sb = __float2bfloat16(v_scale);
        v_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_sb);
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k4_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)(unpack_k4_p(k_pack, i) - 8);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k6_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 12;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)(unpack_k6_p(k_pack, i) - 32);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k7_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 14;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)(unpack_k7_p(k_pack, i) - 64);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k8_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)((int)k_pack[i] - 128);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_tq4_k_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_norm_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const float* __restrict__ signs,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int num_kv_heads,
    int head_dim)
{
    int ti = blockIdx.x;
    int kv_head = blockIdx.y;
    if (threadIdx.x != 0) return;
    int packed_hd = (head_dim + 1) >> 1;
    int kv_stride = num_kv_heads * head_dim;
    int64_t dst_base = (int64_t)ti * kv_stride + kv_head * head_dim;
    if (ti < cache_len) {
        extern __shared__ float scratch[];
        int token_head = ti * num_kv_heads + kv_head;
        float kn = __half2float(*reinterpret_cast<const __half*>(
            &k_norm_cache[token_head]));
        for (int d = 0; d < head_dim; d++) {
            unsigned char p = k_idx_cache[token_head * packed_hd + (d >> 1)];
            int idx = ((d & 1) == 0) ? (p & 0xF) : (p >> 4);
            scratch[d] = kn * tq4_codebook_value_p(idx, head_dim);
        }
        fht_shared_serial_p(scratch, head_dim);
        float inv_sqrt_hd = rsqrtf((float)head_dim);
        for (int d = 0; d < head_dim; d++) {
            out[dst_base + d] = __float2bfloat16(scratch[d] * inv_sqrt_hd * signs[d]);
        }
    } else {
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_base = (int64_t)ci * kv_stride + kv_head * head_dim;
            for (int d = 0; d < head_dim; d++) out[dst_base + d] = k_new[src_base + d];
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_tq4_v_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ v_meta_cache,
    const unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ v_new,
    int cache_len,
    int m,
    int num_kv_heads,
    int head_dim)
{
    int ti = blockIdx.x;
    int kv_head = blockIdx.y;
    int packed_hd = (head_dim + 1) >> 1;
    int kv_stride = num_kv_heads * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int64_t dst_off = (int64_t)ti * kv_stride + kv_head * head_dim + d;
        if (ti < cache_len) {
            int token_head = ti * num_kv_heads + kv_head;
            float scale = __half2float(*reinterpret_cast<const __half*>(
                &v_meta_cache[token_head * 2]));
            float zero = __half2float(*reinterpret_cast<const __half*>(
                &v_meta_cache[token_head * 2 + 1]));
            unsigned char p = v_idx_cache[token_head * packed_hd + (d >> 1)];
            int idx = ((d & 1) == 0) ? (p & 0xF) : (p >> 4);
            out[dst_off] = __float2bfloat16(zero + scale * (float)idx);
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + kv_head * head_dim + d;
                out[dst_off] = v_new[src_off];
            }
        }
    }
}
