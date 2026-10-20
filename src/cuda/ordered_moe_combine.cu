#include <cuda_bf16.h>
// Router-slot order and explicit F32 FMA for every token count. Preserve routed
// F32 through the optional shared branch, then round exactly once to BF16.
extern "C" __global__ void common_moe_reduce(
 const unsigned short* expert_outputs,const float* weights,float* routed,
 unsigned short* mirror,int width,int topk,unsigned long long tokens,float scale) {
 const unsigned long long row=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 const unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=tokens||col>=(unsigned long long)width)return;
 float sum=0.0f;
 for(int slot=0;slot<topk;slot++) {
  const float weight=__fmul_rn(weights[row*(unsigned long long)topk+slot],scale);
  if(weight==0.0f)continue;
  const unsigned short bits=expert_outputs[(row*topk+slot)*(unsigned long long)width+col];
  const float value=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
  sum=__fmaf_rn(value,weight,sum);
 }
 const unsigned long long at=row*(unsigned long long)width+col;routed[at]=sum;
 if(mirror){const __nv_bfloat16 value=__float2bfloat16_rn(sum);mirror[at]=*reinterpret_cast<const unsigned short*>(&value);}
}
extern "C" __global__ void common_moe_combine(
 const float* routed,const unsigned short* shared,const float* gate,unsigned short* output,float* combined,
 int width,unsigned long long tokens,float shared_weight) {
 const unsigned long long row=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 const unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=tokens||col>=(unsigned long long)width)return;
 const unsigned long long at=row*(unsigned long long)width+col;
 float value=routed[at];
 if(shared) {
  const float weight=gate?__fdiv_rn(1.0f,__fadd_rn(1.0f,__expf(-gate[row]))):shared_weight;
  const float s=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(shared+at));
  value=__fmaf_rn(s,weight,value);
 }
 if(combined)combined[at]=value;
 if(output){const __nv_bfloat16 result=__float2bfloat16_rn(value);output[at]=*reinterpret_cast<const unsigned short*>(&result);}
}
