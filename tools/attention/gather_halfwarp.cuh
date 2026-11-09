#pragma once
// Read each aligned 14-byte K6 group once per half warp. Keep scalar FP32
// scale multiplication and coalesced output ordering; no precision change.
extern "C" __global__ void mla_prefill_gather_k6_halfwarp(
 float* values,float* query,const float* q,const unsigned char* cache,
 const int* indices,const int* positions,int capacity,int context
){
 const int row=blockIdx.y,slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
 if(slot<capacity){
  const int seq=min(max(positions[row]+1,0),context);
  int pos=lane==0?indices[(long long)row*capacity+slot]:0;
  pos=__shfl_sync(0xffffffff,pos,0);
  const bool valid=pos>=0 && pos<seq;
  const int element=lane&15,shift=(element*6)&15,word=(element*6)/16+1;
  #pragma unroll
  for(int base=0;base<512;base+=32){
   const int dim=base+lane,group=dim/16;
   unsigned int packet=0;
   if(valid && element<7)
    packet=reinterpret_cast<const unsigned short*>(cache+(long long)pos*448+group*14)[element];
   unsigned int scale_bits=__shfl_sync(0xffffffff,packet,0,16);
   unsigned int lo=__shfl_sync(0xffffffff,packet,word,16);
   unsigned int hi=__shfl_sync(0xffffffff,packet,word+1,16);
   int code=((lo>>shift)|(hi<<(16-shift)))&63;
   float scale=__uint_as_float(scale_bits<<16);
   values[((long long)row*capacity+slot)*512+dim]=valid?scale*(float)(code-32):0.0f;
  }
 }
 if(slot<64){
  for(int d=lane;d<512;d+=32){long long i=((long long)row*64+slot)*512+d;query[i]=q[i];}
 }
}
