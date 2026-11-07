// Gemma full-attention continuation. Extends the existing WMMA prefill
// equation to a cache prefix; current K/V remain BF16, cached FP8 is decoded
// exactly as in the generic fallback. Sliding/image masks use the old path.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x){return __float2bfloat16(x);}
#define FA512_BC 32
#define FA512_BR 16
#define FA512_TILE 16
#define FA512_DPT 16
#define FA512_HD 512

extern "C" __global__ void gemma4_full_history_wmma(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,     /* [M, num_q_heads, 512] */
    const __nv_bfloat16* __restrict__ k_cur, /* [M, num_kv_heads, 512] */
    const __nv_bfloat16* __restrict__ v_cur, /* [M, num_kv_heads, 512] */
    int M, int num_q_heads, int num_kv_heads, float softmax_scale,
    const void* k_cache, const void* v_cache, int prefix, int cache_dtype)
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
    int block_max_kv = (q_base < M) ? (prefix + block_last_qi + 1) : 0;

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
                if (abs_pos < prefix) {
                    if (cache_dtype == 1) {
                        __nv_fp8_e4m3 fk, fv;
                        fk.__x = ((const unsigned char*)k_cache)[off];
                        fv.__x = ((const unsigned char*)v_cache)[off];
                        kval = float_to_bf16(float(fk));
                        vval = float_to_bf16(float(fv));
                    } else {
                        kval = ((const __nv_bfloat16*)k_cache)[off];
                        vval = ((const __nv_bfloat16*)v_cache)[off];
                    }
                } else {
                    off -= (int64_t)prefix * num_kv_heads * FA512_HD;
                    kval = k_cur[off]; vval = v_cur[off];
                }
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
            if (qi >= M) {
                for (int c=lane;c<FA512_BC;c+=32) s_p[r*FA512_BC+c]=float_to_bf16(0.f);
                continue;
            }
            float* row = &s_scores[r * FA512_BC];

            float local_max = -1e30f;
            for (int c = lane; c < FA512_BC; c += 32) {
                int abs_kv = kv_start + c;
                if (c < tile_size && abs_kv <= prefix + qi) {
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


// Diagnostic readback completion is enforced by the native replay entry point.

// Bounded sliding-layer prefill, including image attention masks.
#define FA_BC 16    /* KV tile size; 16 keeps 512-dim Gemma heads under smem limits */
#define FA_BR 16    /* queries per block = wmma tile M */
#define FA_TILE 16  /* wmma tile dimension */
#define FA_DPT 8   /* max dims per thread = ceil(512/32) */

__device__ __forceinline__ float fp8e4m3_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

extern "C" __global__ void gemma4_window_ring(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,       /* [M, num_q_heads, head_dim] */
    const void* __restrict__ k_cache, /* [max_seq, kv_stride] BF16/FP8 or null */
    const void* __restrict__ v_cache, /* [max_seq, kv_stride] BF16/FP8 or null */
    const __nv_bfloat16* __restrict__ k_cur,   /* [M, kv_stride] BF16 current chunk */
    const __nv_bfloat16* __restrict__ v_cur,   /* [M, kv_stride] BF16 current chunk */
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int start_pos, int kv_stride, int sliding_window,
    int cache_dtype, /* 0=BF16, 1=FP8 E4M3 */
    const int* __restrict__ vision_block_ids, int cache_capacity)
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

    // Image blocks are kept within a chunk by the native planner. Extend
    // every query tile to the end of its image, preserving full bidirectionality.
    if (vision_block_ids != nullptr) {
        for (int r=0;r<FA_BR;r++) {
            int qi=q_base+r;
            if(qi>=M) break;
            int absolute=start_pos+qi, id=vision_block_ids[absolute];
            if(id<0) continue;
            int end=absolute+1;
            while(end<start_pos+M && vision_block_ids[end]==id) end++;
            block_max_kv=max(block_max_kv,end);
        }
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
                    int64_t off = (int64_t)(abs_pos % cache_capacity) * kv_stride + kv_h * head_dim + d;
                    if (cache_dtype == 0) {
                        kval = static_cast<const __nv_bfloat16*>(k_cache)[off];
                        vval = static_cast<const __nv_bfloat16*>(v_cache)[off];
                    } else {
                        kval = float_to_bf16(fp8e4m3_to_float(
                            static_cast<const __nv_fp8_e4m3*>(k_cache)[off]));
                        vval = float_to_bf16(fp8e4m3_to_float(
                            static_cast<const __nv_fp8_e4m3*>(v_cache)[off]));
                    }
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
            if (qi >= M) {
                for(int c=lane;c<FA_BC;c+=32) s_p[r*FA_BC+c]=float_to_bf16(0.f);
                continue;
            }
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


extern "C" __global__ void gemma4_ring_append(void*kc,void*vc,const __nv_bfloat16*k,const __nv_bfloat16*v,int rows,int stride,int prefix,int cap,int dtype){
 int keep=min(rows,cap),t=rows-keep+blockIdx.x;
 if(blockIdx.x>=keep)return;
 int dst=(prefix+t)%cap;
 for(int d=threadIdx.x;d<stride;d+=blockDim.x){
  int64_t a=(int64_t)t*stride+d,b=(int64_t)dst*stride+d;
  if(dtype==0){((__nv_bfloat16*)kc)[b]=k[a];((__nv_bfloat16*)vc)[b]=v[a];}
  else{((__nv_fp8_e4m3*)kc)[b]=__nv_fp8_e4m3(float(k[a]));((__nv_fp8_e4m3*)vc)[b]=__nv_fp8_e4m3(float(v[a]));}
 }
}
extern "C" __global__ void gemma4_ring_gather(const unsigned char*source,unsigned char*output,int rows,int row_bytes,int first,int cap){
 int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x,total=(int64_t)rows*row_bytes;
 if(i>=total)return;
 int row=i/row_bytes,col=i%row_bytes;
 output[i]=source[(int64_t)((first+row)%cap)*row_bytes+col];
}

// Full attention in bounded query bands. Scores use FP32 accumulation;
// normalized probabilities round once to BF16 before the value contraction.
extern "C" __global__ void gemma4_full_kv_pack(
    __nv_bfloat16* ko, __nv_bfloat16* vo, const void* ck, const void* cv,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int total, int prefix, int head, int dtype) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total*512)return;
    int row=i/512,d=i%512;
    int64_t off=((int64_t)(row<prefix?row:row-prefix)*2+head)*512+d;
    if(row>=prefix){ko[i]=k[off];vo[i]=v[off];}
    else if(dtype==0){ko[i]=((const __nv_bfloat16*)ck)[off];vo[i]=((const __nv_bfloat16*)cv)[off];}
    else{__nv_fp8_e4m3 a,b;a.__x=((const unsigned char*)ck)[off];b.__x=((const unsigned char*)cv)[off];ko[i]=__float2bfloat16(float(a));vo[i]=__float2bfloat16(float(b));}
}
extern "C" __global__ void gemma4_full_softmax(
    const float* score, __nv_bfloat16* probability, int keys, int first_query) {
    __shared__ float reduction[256];
    int row=blockIdx.x,t=threadIdx.x,last=min(keys-1,first_query+row);
    float m=-INFINITY;
    for(int i=t;i<=last;i+=256)m=fmaxf(m,score[(int64_t)row*keys+i]);
    reduction[t]=m;__syncthreads();
    for(int n=128;n;n/=2){if(t<n)reduction[t]=fmaxf(reduction[t],reduction[t+n]);__syncthreads();}
    m=reduction[0];float sum=0;
    for(int i=t;i<=last;i+=256)sum+=expf(score[(int64_t)row*keys+i]-m);
    reduction[t]=sum;__syncthreads();
    for(int n=128;n;n/=2){if(t<n)reduction[t]+=reduction[t+n];__syncthreads();}
    float inv=1.f/reduction[0];
    for(int i=t;i<keys;i+=256)probability[(int64_t)row*keys+i]=__float2bfloat16(i<=last?expf(score[(int64_t)row*keys+i]-m)*inv:0.f);
}

// Sliding attention uses only the key band needed by these query rows. Future
// keys are included for image blocks and explicitly masked for ordinary text.
extern "C" __global__ void gemma4_window_kv_pack(
    __nv_bfloat16* ko,__nv_bfloat16* vo,const void* ck,const void* cv,
    const __nv_bfloat16* k,const __nv_bfloat16* v,
    int keys,int first,int prefix,int head,int dtype,int capacity) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=keys*256)return;
    int absolute=first+i/256,d=i%256;
    int row=absolute<prefix?absolute%capacity:absolute-prefix;
    int64_t offset=((int64_t)row*8+head)*256+d;
    if(absolute>=prefix){ko[i]=k[offset];vo[i]=v[offset];}
    else if(dtype==0){ko[i]=((const __nv_bfloat16*)ck)[offset];vo[i]=((const __nv_bfloat16*)cv)[offset];}
    else{__nv_fp8_e4m3 a,b;a.__x=((const unsigned char*)ck)[offset];b.__x=((const unsigned char*)cv)[offset];ko[i]=__float2bfloat16(float(a));vo[i]=__float2bfloat16(float(b));}
}
extern "C" __global__ void gemma4_window_softmax(
    const float* score,__nv_bfloat16* probability,int keys,int key_first,
    int query_first,const int* blocks) {
    __shared__ float reduction[256];int row=blockIdx.x,t=threadIdx.x,q=query_first+row;
    int image=blocks?blocks[q]:-1;float maximum=-INFINITY;
    for(int j=t;j<keys;j+=256){int p=key_first+j;bool allowed=p>=q-1023&&(p<=q||(image>=0&&blocks[p]==image));if(allowed)maximum=fmaxf(maximum,score[(int64_t)row*keys+j]);}
    reduction[t]=maximum;__syncthreads();
    for(int n=128;n;n/=2){if(t<n)reduction[t]=fmaxf(reduction[t],reduction[t+n]);__syncthreads();}
    maximum=reduction[0];float sum=0;
    for(int j=t;j<keys;j+=256){int p=key_first+j;bool allowed=p>=q-1023&&(p<=q||(image>=0&&blocks[p]==image));if(allowed)sum+=expf(score[(int64_t)row*keys+j]-maximum);}
    reduction[t]=sum;__syncthreads();
    for(int n=128;n;n/=2){if(t<n)reduction[t]+=reduction[t+n];__syncthreads();}
    float inv=1.f/reduction[0];
    for(int j=t;j<keys;j+=256){int p=key_first+j;bool allowed=p>=q-1023&&(p<=q||(image>=0&&blocks[p]==image));probability[(int64_t)row*keys+j]=__float2bfloat16(allowed?expf(score[(int64_t)row*keys+j]-maximum)*inv:0.f);}
}
