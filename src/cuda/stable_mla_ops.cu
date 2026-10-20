#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ uint64_t stable_block() {
    return ((uint64_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}
__device__ __forceinline__ uint64_t stable_element() {
    return stable_block() * blockDim.x + threadIdx.x;
}
template<bool K6> __device__ __forceinline__ float stable_cache_value(
    const unsigned char* cache, int position, int width, int element) {
    const int bytes = K6 ? 14 : 10;
    const unsigned char* p = cache + ((uint64_t)position * (width / 16) + element / 16) * bytes;
    const float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
    const int j = element & 15;
    int code;
    if (K6) {
        const int bit = j * 6, byte = bit >> 3, shift = bit & 7;
        unsigned int word = p[2 + byte];
        if (byte + 1 < 12) word |= (unsigned int)p[3 + byte] << 8;
        code = (word >> shift) & 63;
    } else {
        code = (p[2 + j / 2] >> ((j & 1) * 4)) & 15;
    }
    return scale * (float)(code - (K6 ? 32 : 8));
}
// All query components are actual F32. No packed-BF16 rotary-query substitution.
template<bool K6> __device__ void stable_gather(
    float* gathered, float* query, float* initial,
    const float* q_abs, const float* q_rope,
    const unsigned char* ckv, const unsigned char* kpe,
    const int* selected, const int* positions,
    int row_start, int rows, int heads, int ccd, int rope, int slots, int context,
    uint64_t q_abs_row_stride, uint64_t q_abs_head_stride,
    uint64_t q_rope_row_stride, uint64_t q_rope_head_stride) {
    const int warps = blockDim.x / warpSize;
    const int warp = threadIdx.x / warpSize, lane = threadIdx.x & (warpSize - 1);
    const int selected_groups = (slots - 1) / warps + 1;
    const int query_groups = (heads - 1) / warps + 1;
    uint64_t block = stable_block();
    const uint64_t gather_blocks = (uint64_t)rows * selected_groups;
    const uint64_t query_blocks = (uint64_t)rows * query_groups;
    const int width = ccd + rope;
    if (block < gather_blocks) {
        const int row = block / selected_groups;
        const uint64_t wide_slot = (block % selected_groups) * warps + warp;
        if (wide_slot >= (uint64_t)slots) return;
        const int slot = (int)wide_slot;
        const int global = row_start + row;
        const int length = (int)min(max((int64_t)positions[global] + 1, (int64_t)0), (int64_t)context);
        int position = lane == 0 ? selected[(uint64_t)global * slots + slot] : 0;
        position = __shfl_sync(0xffffffffu, position, 0);
        for (int dim = lane; dim < width; dim += warpSize) {
            float value = 0.0f;
            if (position >= 0 && position < length) {
                value = dim < ccd ? stable_cache_value<K6>(ckv, position, ccd, dim)
                                  : stable_cache_value<K6>(kpe, position, rope, dim - ccd);
            }
            gathered[((uint64_t)row * slots + slot) * width + dim] = value;
        }
        return;
    }
    block -= gather_blocks;
    if (block < query_blocks) {
        const int row = block / query_groups;
        const uint64_t wide_head = (block % query_groups) * warps + warp;
        if (wide_head >= (uint64_t)heads) return;
        const int head = (int)wide_head;
        const uint64_t global = (uint64_t)row_start + row;
        for (int dim = lane; dim < width; dim += warpSize) {
            query[((uint64_t)row * heads + head) * width + dim] = dim < ccd
                ? q_abs[global * q_abs_row_stride + (uint64_t)head * q_abs_head_stride + dim]
                : q_rope[global * q_rope_row_stride + (uint64_t)head * q_rope_head_stride + dim - ccd];
        }
        return;
    }
    block -= query_blocks;
    if (block >= (uint64_t)rows) return;
    const int row = block, global = row_start + row;
    const int length = (int)min(max((int64_t)positions[global] + 1, (int64_t)0), (int64_t)context);
    for (int slot = threadIdx.x; slot < slots; slot += blockDim.x) {
        const int position = selected[(uint64_t)global * slots + slot];
        const float value = position >= 0 && position < length ? 0.0f : -INFINITY;
        for (int head = 0; head < heads; head++)
            initial[((uint64_t)row * heads + head) * slots + slot] = value;
    }
}
#define STABLE_GATHER(NAME,K6) \
extern "C" __global__ void NAME(float* g,float* q,float* s,const float* a,const float* r, \
 const unsigned char* c,const unsigned char* k,const int* i,const int* p, \
 int start,int rows,int heads,int ccd,int rope,int slots,int context, \
 uint64_t ar,uint64_t ah,uint64_t rr,uint64_t rh) { \
 stable_gather<K6>(g,q,s,a,r,c,k,i,p,start,rows,heads,ccd,rope,slots,context,ar,ah,rr,rh); }
STABLE_GATHER(stable_mla_gather_k4,false)
STABLE_GATHER(stable_mla_gather_k6,true)

extern "C" __global__ void stable_mla_widen_bf16(float* out,const __nv_bfloat16* input,uint64_t elements) {
    const uint64_t i=stable_element();
    if(i<elements) out[i]=__bfloat162float(input[i]);
}
// F32 head-major projection -> token-major BF16 output. Round only the output.
extern "C" __global__ void stable_mla_scatter_bf16(__nv_bfloat16* out,const float* input,
    int rows,int heads,int width,uint64_t head_stride_rows) {
    const uint64_t i=stable_element(), count=(uint64_t)rows*heads*width;
    if(i>=count)return;
    const int dim=i%width;
    const uint64_t row_head=i/width;
    const int head=row_head%heads, row=row_head/heads;
    out[i]=__float2bfloat16_rn(input[((uint64_t)head*head_stride_rows+row)*width+dim]);
}

// Same warp softmax equation as the retained gathered-prefill path.
extern "C" __global__ void stable_mla_softmax(float* weights,const float* scores,
    int rows,int heads,int slots) {
    const int lane=threadIdx.x&(warpSize-1), warp=threadIdx.x/warpSize;
    const int warps=blockDim.x/warpSize;
    const uint64_t row=stable_block()*warps+warp;
    if(row>=(uint64_t)rows*heads)return;
    const float* score=scores+row*slots;
    float maximum=-INFINITY;
    for(int k=lane;k<slots;k+=warpSize)maximum=fmaxf(maximum,score[k]);
    for(int offset=warpSize/2;offset>0;offset>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
    maximum=__shfl_sync(0xffffffffu,maximum,0);
    float sum=0.0f;
    for(int k=lane;k<slots;k+=warpSize){const float v=score[k];if(isfinite(v))sum+=__expf(v-maximum);}
    for(int offset=warpSize/2;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
    sum=__shfl_sync(0xffffffffu,sum,0);
    for(int k=lane;k<slots;k+=warpSize){const float v=score[k];weights[row*slots+k]=isfinite(v)&&sum>0.0f?__expf(v-maximum)/sum:0.0f;}
}
