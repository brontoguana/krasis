#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
// Selected F32 MLA primitives; native codec helpers retained byte-for-byte.
__device__ inline void mla_pack_k4_16(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = (unsigned char)((codes[i * 2 + 1] << 4) | (codes[i * 2] & 0x0f));
    }
}

__device__ inline float mla_quantize_k4_one_pass_ls(
    const float* src,
    unsigned char* codes
) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

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
    if (ls_den > 1e-12f) {
        scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return scale;
}

__device__ inline void mla_store_k4_block(
    unsigned char* dst,
    const float* values
) {
    unsigned char codes[16];
    float scale = mla_quantize_k4_one_pass_ls(values, codes);
    __nv_bfloat16 scale_bf16 = __float2bfloat16(scale);
    *reinterpret_cast<unsigned short*>(dst) =
        *reinterpret_cast<unsigned short*>(&scale_bf16);
    mla_pack_k4_16(dst + 2, codes);
}

__device__ inline void mla_pack_k6_16(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 12; i++) dst[i] = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int bit = i * 6;
        int byte = bit >> 3;
        int shift = bit & 7;
        unsigned int value = ((unsigned int)codes[i]) & 0x3fu;
        dst[byte] |= (unsigned char)(value << shift);
        if (shift > 2) {
            dst[byte + 1] |= (unsigned char)(value >> (8 - shift));
        }
    }
}

__device__ inline float mla_quantize_k6_one_pass_ls(
    const float* src,
    unsigned char* codes
) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float scale = fmaxf(max_abs * (1.0f / 31.0f), 1e-8f);
    float inv_scale = 1.0f / scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-31, min(31, q));
        codes[i] = (unsigned char)(q + 32);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return scale;
}

__device__ inline void mla_store_k6_block(
    unsigned char* dst,
    const float* values
) {
    unsigned char codes[16];
    float scale = mla_quantize_k6_one_pass_ls(values, codes);
    __nv_bfloat16 scale_bf16 = __float2bfloat16(scale);
    *reinterpret_cast<unsigned short*>(dst) =
        *reinterpret_cast<unsigned short*>(&scale_bf16);
    mla_pack_k6_16(dst + 2, codes);
}

// One admitted native-size block per actual row. No padding/RoPE suffix mutation.
extern "C" __global__ void mla_f32_rmsnorm(
 float* __restrict__ data,const float* __restrict__ weight,float eps,
 unsigned long long rows,int width,int row_stride,
 unsigned short* __restrict__ bf16_output,int bf16_stride) {
 const unsigned long long row=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 if(row>=rows)return;
 const int tid=threadIdx.x,num_threads=blockDim.x;
 float* h=data+row*(unsigned long long)row_stride;
 float sum_sq=0.0f;
 for(int i=tid;i<width;i+=num_threads)sum_sq+=h[i]*h[i];
 for(int offset=warpSize/2;offset>0;offset>>=1)sum_sq+=__shfl_down_sync(0xffffffff,sum_sq,offset);
 __shared__ float warp_sums[32];
 const int warp_id=tid/warpSize,lane_id=tid%warpSize;
 if(lane_id==0)warp_sums[warp_id]=sum_sq;
 __syncthreads();
 if(tid==0){float total=0.0f;int num_warps=(num_threads+warpSize-1)/warpSize;for(int w=0;w<num_warps;w++)total+=warp_sums[w];warp_sums[0]=rsqrtf(total/(float)width+eps);}
 __syncthreads();
 const float rms_scale=warp_sums[0];
 for(int i=tid;i<width;i+=num_threads){const float scale=weight==nullptr?1.0f:weight[i];const float value=h[i]*rms_scale*scale;h[i]=value;if(bf16_output){const __nv_bfloat16 bf=__float2bfloat16_rn(value);bf16_output[row*(unsigned long long)bf16_stride+i]=*reinterpret_cast<const unsigned short*>(&bf);}}
}

// Input query is token/head-major F32. Both native RoPE layouts become half-split.
// Stage all source components before an in-place permutation; this works beyond one warp.
extern "C" __global__ void mla_f32_prepare_query(
 float* __restrict__ query,float* __restrict__ kv,
 unsigned short* __restrict__ query_bf16,unsigned long long bf16_head_stride_rows,
 const float* __restrict__ cos_table,const float* __restrict__ sin_table,
 const int* __restrict__ positions,unsigned long long rows,
 int heads,int nope,int rope,int kv_rank,int kv_stride,int interleaved) {
 const int head=blockIdx.x;
 const unsigned long long row=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 if(row>=rows||head>=heads)return;
 const int tid=threadIdx.x,threads=blockDim.x,qwidth=nope+rope,half=rope/2;
 float* q=query+(row*(unsigned long long)heads+head)*qwidth;
 float* k=kv+row*(unsigned long long)kv_stride+kv_rank;
 extern __shared__ float staged[];
 for(int i=tid;i<rope;i+=threads){staged[i]=q[nope+i];if(head==0)staged[rope+i]=k[i];}
 __syncthreads();
 if(rope>0){const int position=positions[row];for(int i=tid;i<half;i+=threads){const int re=interleaved?2*i:i,im=interleaved?2*i+1:half+i;const unsigned long long at=(unsigned long long)position*half+i;const float cv=cos_table[at],sv=sin_table[at],qr=staged[re],qi=staged[im];q[nope+i]=qr*cv-qi*sv;q[nope+half+i]=qi*cv+qr*sv;if(head==0){const float kr=staged[rope+re],ki=staged[rope+im];k[i]=kr*cv-ki*sv;k[half+i]=ki*cv+kr*sv;}}}
 __syncthreads();
 if(query_bf16){for(int i=tid;i<qwidth;i+=threads){const __nv_bfloat16 bf=__float2bfloat16_rn(q[i]);query_bf16[((unsigned long long)head*bf16_head_stride_rows+row)*qwidth+i]=*reinterpret_cast<const unsigned short*>(&bf);}}
}

template<int Bits> __device__ __forceinline__ void mla_f32_cache_body(
 unsigned char* ckv_cache,unsigned char* kpe_cache,const float* kv,
 unsigned long long rows,int base_position,int kv_rank,int kv_stride,int ccd,int rope) {
 const unsigned long long row=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 if(row>=rows)return;
 const int block=blockIdx.x*blockDim.x+threadIdx.x;
 const int cblocks=ccd/16,rblocks=rope/16;
 const float* data=kv+row*(unsigned long long)kv_stride;
 const unsigned long long position=(unsigned long long)base_position+row;
 const int block_bytes=Bits==4?10:14;
 if(block<cblocks){float values[16];const int base=block*16;
 #pragma unroll
 for(int i=0;i<16;i++)values[i]=base+i<kv_rank?data[base+i]:0.0f;
 unsigned char* dst=ckv_cache+(position*cblocks+block)*block_bytes;
 if constexpr(Bits==4)mla_store_k4_block(dst,values);else mla_store_k6_block(dst,values);}
 if(block<rblocks){float values[16];const int base=block*16;
 #pragma unroll
 for(int i=0;i<16;i++)values[i]=data[kv_rank+base+i];
 unsigned char* dst=kpe_cache+(position*rblocks+block)*block_bytes;
 if constexpr(Bits==4)mla_store_k4_block(dst,values);else mla_store_k6_block(dst,values);}
}
extern "C" __global__ void mla_f32_cache_k4(unsigned char* ckv,unsigned char* kpe,const float* kv,unsigned long long rows,int base,int rank,int stride,int ccd,int rope){mla_f32_cache_body<4>(ckv,kpe,kv,rows,base,rank,stride,ccd,rope);}
extern "C" __global__ void mla_f32_cache_k6(unsigned char* ckv,unsigned char* kpe,const float* kv,unsigned long long rows,int base,int rank,int stride,int ccd,int rope){mla_f32_cache_body<6>(ckv,kpe,kv,rows,base,rank,stride,ccd,rope);}

// Native scalar F32 query/BF16 WKC summation, batched over independent rows/heads.
extern "C" __global__ void mla_f32_absorb_wkc(
 float* __restrict__ output,const float* __restrict__ query,
 const __nv_bfloat16* __restrict__ weight,unsigned long long rows,
 int heads,int nope,int rope,int ccd) {
 const int head=blockIdx.x;
 const unsigned long long row=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 if(row>=rows||head>=heads)return;
 const float* q=query+(row*(unsigned long long)heads+head)*(nope+rope);
 const __nv_bfloat16* w=weight+(unsigned long long)head*nope*ccd;
 float* out=output+(row*(unsigned long long)heads+head)*ccd;
 for(int d=threadIdx.x;d<ccd;d+=blockDim.x){float acc=0.0f;for(int k=0;k<nope;k++)acc+=q[k]*__bfloat162float(w[(unsigned long long)k*ccd+d]);out[d]=acc;}
}
