#include <cuda_bf16.h>
// One identical row operation in full GPU prefill and eager/captured continuation.
// The power-of-two thread count depends on width and real device/function limits.
extern "C" __global__ void common_rmsnorm_bf16(
 unsigned short* output,const unsigned short* input,const unsigned short* weight,
 int width,unsigned long long rows,float eps) {
 const unsigned long long row=(unsigned long long)blockIdx.y*gridDim.x+blockIdx.x;
 if(row>=rows)return;
 const unsigned long long base=row*(unsigned long long)width;
 extern __shared__ float sums[];
 float sum=0.f;
 for(unsigned long long col=threadIdx.x;col<(unsigned long long)width;col+=blockDim.x){
  const float x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+base+col));
  sum=__fadd_rn(sum,__fmul_rn(x,x));
 }
 sums[threadIdx.x]=sum;__syncthreads();
 for(unsigned stride=blockDim.x/2;stride;stride>>=1){
  if(threadIdx.x<stride)sums[threadIdx.x]=__fadd_rn(sums[threadIdx.x],sums[threadIdx.x+stride]);
  __syncthreads();
 }
 const float inv=rsqrtf(__fadd_rn(__fdiv_rn(sums[0],(float)width),eps));
 for(unsigned long long col=threadIdx.x;col<(unsigned long long)width;col+=blockDim.x){
  const float x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+base+col));
  const float w=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(weight+col));
  const __nv_bfloat16 y=__float2bfloat16_rn(__fmul_rn(__fmul_rn(x,inv),w));
  output[base+col]=*reinterpret_cast<const unsigned short*>(&y);
 }
}
