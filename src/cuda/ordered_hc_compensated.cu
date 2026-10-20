#include <cuda_bf16.h>
// Unadopted block-parallel alternative. Exact F32 normalization, compensated F32 product/sum, F64 final reduction.
// Runtime block width is fixed for a loaded operator, independent of M.
extern "C" __global__ void ordered_hc_f64_block(
    const unsigned short* state,const float* inverse,const float* weight,
    float* output,int rows,int cols,unsigned long long tokens) {
    const int row=(int)blockIdx.y,tid=threadIdx.x;
    const unsigned long long base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*4;
    if(row>=rows||base>=tokens)return; // uniform block
    float sums[4]={0.0f,0.0f,0.0f,0.0f};
    float errors[4]={0.0f,0.0f,0.0f,0.0f};
    for(long long k=tid;k<cols;k+=blockDim.x) {
        const float w=weight[(long long)row*cols+k];
        #pragma unroll
        for(int t=0;t<4;++t)if(base+t<tokens) {
            const unsigned short bits=state[(base+t)*(unsigned long long)cols+k];
            const float x=__fmul_rn(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits)),inverse[base+t]);
            const float product=__fmul_rn(x,w);
            const float residual=__fmaf_rn(x,w,-product);
            const float sum=__fadd_rn(sums[t],product);
            const float bv=__fsub_rn(sum,sums[t]);
            const float error=__fadd_rn(__fsub_rn(sums[t],__fsub_rn(sum,bv)),__fsub_rn(product,bv));
            errors[t]=__fadd_rn(errors[t],__fadd_rn(error,residual));
            sums[t]=sum;
        }
    }
    extern __shared__ double scratch[];
    #pragma unroll
    for(int t=0;t<4;++t)if(base+t<tokens) {
        scratch[tid]=__dadd_rn((double)sums[t],(double)errors[t]);__syncthreads();
        for(int step=blockDim.x/2;step>0;step/=2) {
            if(tid<step)scratch[tid]=__dadd_rn(scratch[tid],scratch[tid+step]);
            __syncthreads();
        }
        if(tid==0)output[(base+t)*(unsigned long long)rows+row]=__double2float_rn(scratch[0]);
        __syncthreads();
    }
}
