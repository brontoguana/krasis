// Copyright (c) 2023, Tri Dao. See ../flash_attn/LICENSE.
// Bounded native SDPA contraction for Qwen image attention.
#define KRASIS_FA_VENDOR
#include "flash_fwd_launch_template.h"
#include <cmath>
#include <cstring>
#include <vector>
static void set_params_from_raw(
    flash::Flash_fwd_params &params,
    // Pointers (all device pointers)
    void *q_ptr, void *k_ptr, void *v_ptr, void *out_ptr,
    void *softmax_lse_ptr,   // [num_heads, total_q] or [batch, num_heads, seqlen_q]
    void *cu_seqlens_q_ptr,  // [batch+1], int32  (NULL for fixed-length)
    void *cu_seqlens_k_ptr,  // [batch+1], int32  (NULL for fixed-length)
    // Dimensions
    int batch_size,
    int seqlen_q,       // max seqlen_q (or actual if fixed-length)
    int seqlen_k,       // max seqlen_k (or actual if fixed-length)
    int num_heads,
    int num_heads_k,
    int head_dim,
    int total_q,        // sum of all q sequence lengths (for varlen)
    int total_k,        // sum of all k sequence lengths (for varlen)
    // Config
    float softmax_scale,
    bool is_causal,
    bool is_bf16,
    bool unpadded_lse   // true for varlen: LSE in [num_heads, total_q] format
) {
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_dim;
    params.total_q = total_q;

    // Rounded dimensions (for kernel indexing)
    params.seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;
    params.d_rounded = ((head_dim + 31) / 32) * 32;

    // Softmax scaling
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = 0.0f;

    // No dropout for inference
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // Causal mask
    params.is_causal = is_causal;
    params.window_size_left = -1;  // no window
    params.window_size_right = is_causal ? 0 : -1;

    // Cumulative sequence lengths for variable-length batching
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_ptr);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_ptr);
    params.is_seqlens_k_cumulative = true;

    params.unpadded_lse = unpadded_lse;

    // Strides for varlen (packed format): [total, heads, head_dim]
    if (cu_seqlens_q_ptr != nullptr) {
        // Varlen: tensors are [total_q, num_heads, head_dim]
        params.q_row_stride = num_heads * head_dim;
        params.k_row_stride = num_heads_k * head_dim;
        params.v_row_stride = num_heads_k * head_dim;
        params.o_row_stride = num_heads * head_dim;
        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;
        // Batch strides not used in varlen mode
        params.q_batch_stride = 0;
        params.k_batch_stride = 0;
        params.v_batch_stride = 0;
        params.o_batch_stride = 0;
    } else {
        // Fixed-length: tensors are [batch, seqlen, num_heads, head_dim]
        params.q_row_stride = num_heads * head_dim;
        params.k_row_stride = num_heads_k * head_dim;
        params.v_row_stride = num_heads_k * head_dim;
        params.o_row_stride = num_heads * head_dim;
        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;
        params.q_batch_stride = seqlen_q * num_heads * head_dim;
        params.k_batch_stride = seqlen_k * num_heads_k * head_dim;
        params.v_batch_stride = seqlen_k * num_heads_k * head_dim;
        params.o_batch_stride = seqlen_q * num_heads * head_dim;
    }

    // Not used: rotary, KV append, paged KV, ALiBi
    params.rotary_dim = 0;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.is_rotary_interleaved = false;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.seqlenq_ngroups_swapped = false;
    params.p_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.blockmask = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;
    params.num_splits = 0;
    params.seqlen_knew = 0;
    params.is_kv_fp8 = false;  // Caller sets this explicitly for FP8 path
}


extern "C" int krasis_qwen_vision_sdpa(void*q,void*k,void*v,void*y,void*lse,void*accum,void*lse_accum,int n,int heads,int dim,int splits,void*stream) {
 if(n<1||n>1024||dim!=72||splits<1||splits>8)return -1;
 flash::Flash_fwd_params p;
 set_params_from_raw(p,q,k,v,y,lse,nullptr,nullptr,1,n,n,heads,heads,dim,n,n,float(1.0/std::sqrt(double(dim))),false,true,false);
 p.num_splits=splits;p.oaccum_ptr=accum;p.softmax_lseaccum_ptr=lse_accum;
 using Traits=Flash_fwd_kernel_traits<96,64,128,4,false,false,cutlass::bfloat16_t>;
 auto st=(cudaStream_t)stream;dim3 grid((n+63)/64,splits>1?splits:1,splits>1?heads:heads);
 if(splits>1){
  auto kernel=flash::flash_fwd_splitkv_kernel<Traits,false,false,false,false,false,false,true,false>;
  auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Traits::kSmemSize);if(status!=cudaSuccess)return int(status);
  kernel<<<grid,128,Traits::kSmemSize,st>>>(p);
  dim3 combine((heads*n+15)/16);
  if(splits<=2)flash::flash_fwd_splitkv_combine_kernel<Traits,16,1,false><<<combine,128,0,st>>>(p);
  else if(splits<=4)flash::flash_fwd_splitkv_combine_kernel<Traits,16,2,false><<<combine,128,0,st>>>(p);
  else flash::flash_fwd_splitkv_combine_kernel<Traits,16,3,false><<<combine,128,0,st>>>(p);
 }else{
  flash::run_mha_fwd_hdim96<cutlass::bfloat16_t,false>(p,st);
 }
 return int(cudaGetLastError());
}

// Gemma uses unscaled dot products; padded channels are zero. Its valid image
// rows are supplied without processor padding, preserving the key mask.
extern "C" int krasis_gemma_vision_sdpa(void*q,void*k,void*v,void*y,void*lse,int n,int heads,void*stream){
 if(n<1||n>2520||heads!=16)return -1;
 flash::Flash_fwd_params p;
 set_params_from_raw(p,q,k,v,y,lse,nullptr,nullptr,1,n,n,heads,heads,96,n,n,1.0f,false,true,false);
 flash::run_mha_fwd_hdim96<cutlass::bfloat16_t,false>(p,(cudaStream_t)stream);
 return int(cudaGetLastError());
}
