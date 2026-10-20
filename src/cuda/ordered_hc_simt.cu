#include <cuda_bf16.h>
// Unadopted alternative: F32 normalized operands, hardware F64 accumulation.
// Four tokens share each weight load. A warp owns one output row; its K-order
// and reduction tree are identical for every batch/singleton/partition.
extern "C" __global__ void ordered_hc_f64_simt(
    const unsigned short* state,const float* inverse,const float* weight,
    float* output,int rows,int cols,unsigned long long tokens) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int row=(int)blockIdx.y*4+warp;
    const unsigned long long base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*4;
    if(row>=rows||base>=tokens)return; // uniform within each warp
    double sums[4]={0.0,0.0,0.0,0.0};
    for(long long k=lane;k<cols;k+=32) {
        const double w=(double)weight[(long long)row*cols+k];
        #pragma unroll
        for(int t=0;t<4;++t) {
            if(base+t<tokens) {
                const unsigned short bits=state[(base+t)*(unsigned long long)cols+k];
                const float x=__fmul_rn(__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits)),inverse[base+t]);
                sums[t]=__dadd_rn(sums[t],__dmul_rn((double)x,w));
            }
        }
    }
    #pragma unroll
    for(int t=0;t<4;++t) {
        if(base+t<tokens) {
            for(int offset=16;offset>0;offset>>=1)sums[t]=__dadd_rn(sums[t],__shfl_down_sync(0xffffffff,sums[t],offset));
            if(lane==0)output[(base+t)*(unsigned long long)rows+row]=__double2float_rn(sums[t]);
        }
    }
}
