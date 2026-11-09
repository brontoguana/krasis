#pragma once
// Keep packed equations and row order; issue four adjacent FP32 stores together.
extern "C" __global__ void mla_prefill_gather_k6_vec4(
 float* values,float* query,const float* q,const unsigned char* cache,
 const int* indices,const int* positions,int capacity,int context
){
 const int row=blockIdx.y,slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
 if(slot<capacity){
  const int seq=min(max(positions[row]+1,0),context);
  int pos=lane==0?indices[(long long)row*capacity+slot]:0;
  pos=__shfl_sync(0xffffffff,pos,0);
  const bool valid=pos>=0 && pos<seq;
  #pragma unroll
  for(int d=lane*4;d<512;d+=128){
   float4 v=make_float4(0,0,0,0);
   if(valid){
    v.x=mla_prefill_load_k6_value(cache,pos,512,d);
    v.y=mla_prefill_load_k6_value(cache,pos,512,d+1);
    v.z=mla_prefill_load_k6_value(cache,pos,512,d+2);
    v.w=mla_prefill_load_k6_value(cache,pos,512,d+3);
   }
   reinterpret_cast<float4*>(values+((long long)row*capacity+slot)*512)[d/4]=v;
  }
 }
 if(slot<64){
  #pragma unroll
  for(int d=lane*4;d<512;d+=128){
   const long long base=((long long)row*64+slot)*512;
   reinterpret_cast<float4*>(query+base)[d/4]=reinterpret_cast<const float4*>(q+base)[d/4];
  }
 }
}
