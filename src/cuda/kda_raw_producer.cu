#include <cuda_bf16.h>
#include <stdint.h>
extern "C" __global__ void kda_raw_domain(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const __nv_bfloat16* g,const __nv_bfloat16* beta,const float* A,const float* bias,int first,int rows,int H,int D,int* error){
 if(atomicAdd(error,0))return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;
 bool bad=false;if(i<uint64_t(rows)*hd){const uint64_t at=uint64_t(first)*hd+i;bad=((__bfloat16_as_ushort(q[at])&0x7f80u)==0x7f80u)||((__bfloat16_as_ushort(k[at])&0x7f80u)==0x7f80u)||((__bfloat16_as_ushort(v[at])&0x7f80u)==0x7f80u)||((__bfloat16_as_ushort(g[at])&0x7f80u)==0x7f80u);}
 if(i<uint64_t(rows)*H)bad|=(__bfloat16_as_ushort(beta[uint64_t(first)*H+i])&0x7f80u)==0x7f80u;
 if(i<uint64_t(H))bad|=(__float_as_uint(A[i])&0x7f800000u)==0x7f800000u;
 if(i<hd)bad|=(__float_as_uint(bias[i])&0x7f800000u)==0x7f800000u;
 if(bad)atomicCAS(error,0,4);
}
extern "C" __global__ void kda_raw_prepare(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const __nv_bfloat16* g,const __nv_bfloat16* beta,const float* A,const float* bias,float* Q,float* K,float* V,float* G,float* B,int first,int T,int H,int D,float lower,float scale,float eps,const int* error){
 if(*error)return;int row=blockIdx.x,h=blockIdx.y,tid=threadIdx.x;if(row>=T||h>=H)return;
 extern __shared__ float reduce[];long long at=((long long)(first+row)*H+h)*D,ot=((long long)row*H+h)*D;
 float qv=tid<D?__bfloat162float(q[at+tid]):0.f,kv=tid<D?__bfloat162float(k[at+tid]):0.f;
 reduce[tid]=qv*qv;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
 float qi=rsqrtf(reduce[0]+eps)*scale;__syncthreads();reduce[tid]=kv*kv;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
 float ki=rsqrtf(reduce[0]+eps);if(tid<D){Q[ot+tid]=qv*qi;K[ot+tid]=kv*ki;V[ot+tid]=__bfloat162float(v[at+tid]);float raw=__bfloat162float(g[at+tid])+bias[h*D+tid];G[ot+tid]=lower/(1.f+__expf(-__expf(A[h])*raw));}if(tid==0)B[row*H+h]=1.f/(1.f+__expf(-__bfloat162float(beta[(first+row)*H+h])));
}
