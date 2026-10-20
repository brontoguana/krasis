#include <cuda_bf16.h>
#include <mma.h>
// Private unadopted shared projection. Existing dispatches are unchanged.
// F32 normalized input and original F32 weights; all dimensions are runtime.
__device__ __forceinline__ float hc_add(float a,float b) {
    float out;asm volatile("add.rn.f32 %0, %1, %2;":"=f"(out):"f"(a),"f"(b));return out;
}
__device__ __forceinline__ float hc_sub(float a,float b) {
    float out;asm volatile("sub.rn.f32 %0, %1, %2;":"=f"(out):"f"(a),"f"(b));return out;
}
__device__ __forceinline__ void hc_twosum(float a,float b,float& sum,float& error) {
    sum=hc_add(a,b);
    const bool larger=(__float_as_uint(a)&0x7fffffffU)>=(__float_as_uint(b)&0x7fffffffU);
    error=hc_sub(larger?b:a,hc_sub(sum,larger?a:b));
    if((__float_as_uint(error)&0x7fffffffU)==0)error=0.0f;
}
__device__ __forceinline__ void hc_carry(float x,float& hi,float& lo) {
    float sum,error;hc_twosum(hi,x,sum,error);
    hc_twosum(sum,hc_add(lo,error),hi,lo);
}
__device__ __forceinline__ void hc_split(float x,float* base,int stride,int index) {
    namespace w=nvcuda::wmma;
    const float high=__uint_as_float(__float_as_uint(x)&0xffffe000U);
    const float r=hc_sub(x,high);
    const float middle=__uint_as_float(__float_as_uint(r)&0xffffe000U);
    const float tail=hc_sub(r,middle);
    const float low=__uint_as_float(__float_as_uint(tail)&0xffffe000U);
    base[index]=w::__float_to_tf32(high);
    base[stride+index]=w::__float_to_tf32(middle);
    base[2*stride+index]=w::__float_to_tf32(low);
}
extern "C" __global__ void ordered_hc_f32_tf32x9(
    const unsigned short* state,const float* inv_rms,const float* weight,
    float* output,int rows,int cols,unsigned long long tokens) {
    namespace w=nvcuda::wmma;
    const unsigned long long token_base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
    const int row_base=(int)blockIdx.y*32;
    if(token_base>=tokens||row_base>=rows)return;
    const int tid=threadIdx.x,warp=tid>>5;
    // 3 weight components and 3 input components, each [32,32] with aligned stride40.
    __shared__ __align__(32) float staging[6*32*40];
    float* aa=staging;float* bb=aa+3*32*40;
    w::fragment<w::accumulator,16,16,8,float> part,hi,lo;
    w::fill_fragment(hi,0.0f);w::fill_fragment(lo,0.0f);
    for(long long kbase=0;kbase<cols;kbase+=32) {
        for(int i=tid;i<32*32;i+=blockDim.x) {
            const int r=i/32,k=i%32,row=row_base+r;
            const long long col=kbase+k;
            const float v=(row<rows&&col<cols)?weight[(long long)row*cols+col]:0.0f;
            hc_split(v,aa,32*40,r*40+k);
            float x=0.0f;
            if(token_base+r<tokens&&col<cols) {
                const unsigned short bits=state[(token_base+r)*(unsigned long long)cols+col];
                x=__fmul_rn(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits)),inv_rms[token_base+r]);
            }
            hc_split(x,bb,32*40,r*40+k);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<32;k+=8) {
            #pragma unroll
            for(int ac=0;ac<3;++ac) {
                w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> a;
                w::load_matrix_sync(a,aa+ac*32*40+(warp/2)*16*40+k,40);
                #pragma unroll
                for(int bc=0;bc<3;++bc) {
                    w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::col_major> b;
                    w::load_matrix_sync(b,bb+bc*32*40+(warp%2)*16*40+k,40);
                    w::fill_fragment(part,0.0f);
                    w::mma_sync(part,a,b,part);
                    for(int i=0;i<part.num_elements;++i)hc_carry(part.x[i],hi.x[i],lo.x[i]);
                }
            }
        }
        __syncthreads();
    }
    for(int i=0;i<part.num_elements;++i)part.x[i]=hc_add(hi.x[i],lo.x[i]);
    w::store_matrix_sync(staging+warp*256,part,16,w::mem_row_major);
    __syncthreads();
    for(int i=tid;i<4*256;i+=blockDim.x) {
        const int wi=i/256,local=i%256,row=row_base+(wi/2)*16+local/16;
        const unsigned long long token=token_base+(wi%2)*16+local%16;
        if(row<rows&&token<tokens)output[token*(unsigned long long)rows+row]=staging[i];
    }
}
