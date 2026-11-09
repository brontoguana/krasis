// Bounded K6 unpacking and high/residual BF16 intermediates.
// Cache bytes, selection order and causal masking are unchanged.
#pragma once
namespace mla_bf16 {
__device__ __forceinline__ void store(__nv_bfloat16* high,__nv_bfloat16* low,long long i,float x){
 const auto h=__float2bfloat16_rn(x);high[i]=h;low[i]=__float2bfloat16_rn(x-__bfloat162float(h));
}
}
extern "C" __global__ void mla_bf16_unpack(
 __nv_bfloat16* values,__nv_bfloat16* query,const float* q,const unsigned char* cache,
 const int* indices,const int* positions,int capacity,int padded_capacity,int context,int rows
){
 const int row=blockIdx.y,slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
 const long long nv=(long long)rows*padded_capacity*512,nq=(long long)rows*64*512;
 if(slot<padded_capacity){
  const int pos=slot<capacity?indices[(long long)row*capacity+slot]:-1;
  const bool valid=pos>=0 && pos<context && pos<=positions[row];
  for(int d=lane;d<512;d+=32){
   float x=valid?mla_prefill_load_k6_value(cache,pos,512,d):0.0f;
   mla_bf16::store(values,values+nv,((long long)row*padded_capacity+slot)*512+d,x);
  }
 }
 if(slot<64){
  for(int d=lane;d<512;d+=32){long long i=((long long)row*64+slot)*512+d;mla_bf16::store(query,query+nq,i,q[i]);}
 }
}
extern "C" __global__ void mla_bf16_softmax(
 __nv_bfloat16* probabilities,const float* scores,const int* indices,const int* positions,
 float scale,int capacity,int padded_capacity,int context,int rows
){
 const int row=blockIdx.y,h=blockIdx.x,tid=threadIdx.x;
 const long long plane=(long long)rows*64*padded_capacity,base=((long long)row*64+h)*padded_capacity;
 const int seq=min(max(positions[row]+1,0),context);
 extern __shared__ float bf16_softmax_shared[];float* reduce=bf16_softmax_shared;float* weight=bf16_softmax_shared+8;
 float m=-INFINITY;
 for(int i=tid;i<padded_capacity;i+=256){
  const int pos=i<capacity?indices[(long long)row*capacity+i]:-1;
  const float s=pos>=0 && pos<seq?scores[base+i]*scale:-INFINITY;
  weight[i]=s;m=fmaxf(m,s);
 }
 for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,o));
 if((tid&31)==0)reduce[tid/32]=m;
 __syncthreads();
 if(tid==0){float v=reduce[0];for(int w=1;w<8;w++)v=fmaxf(v,reduce[w]);reduce[0]=v;}
 __syncthreads();m=reduce[0];__syncthreads();float sum=0;
 for(int i=tid;i<padded_capacity;i+=256){float e=isfinite(weight[i])?__expf(weight[i]-m):0;weight[i]=e;sum+=e;}
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 if((tid&31)==0)reduce[tid/32]=sum;
 __syncthreads();
 if(tid==0){float s=0;for(int w=0;w<8;w++)s+=reduce[w];reduce[0]=s>0?1/s:0;}
 __syncthreads();
 for(int i=tid;i<padded_capacity;i+=256)mla_bf16::store(probabilities,probabilities+plane,base+i,weight[i]*reduce[0]);
}
