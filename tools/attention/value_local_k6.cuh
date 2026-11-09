#pragma once
// Offline structural screen: each CTA contracts 32 heads x 64 features.
// A 32-selection K6 band is unpacked directly into CTA-owned shared storage.
// Every output retains one ascending FP32 FMA chain; no partial-sum workspace.
__global__ void value_local_k6(float* output,const float* weights,
 const unsigned char* cache,const int* indices,const int* positions,int cap,int context){
 __shared__ float values[32][64];
 __shared__ float probs[32][32];
 int row=blockIdx.z,h0=blockIdx.y*32,d0=blockIdx.x*64;
 int lane=threadIdx.x%32,head=threadIdx.x/32;
 float acc[4][2]={};
 int seq=min(max(positions[row]+1,0),context);
 for(int begin=0;begin<cap;begin+=32){
  for(int i=threadIdx.x;i<32*64;i+=256){int k=i/64,d=i%64,slot=begin+k;
   int pos=slot<cap?indices[(long long)row*cap+slot]:-1;
   values[k][d]=pos>=0&&pos<seq?mla_prefill_load_k6_value(cache,pos,512,d0+d):0.0f;
  }
  for(int i=threadIdx.x;i<32*32;i+=256){int h=i/32,k=i%32;
   probs[h][k]=begin+k<cap?weights[((long long)row*64+h0+h)*cap+begin+k]:0.0f;
  }
  __syncthreads();
  #pragma unroll 4
  for(int k=0;k<32;k++){
   float v0=values[k][lane],v1=values[k][lane+32];
   #pragma unroll
   for(int h=0;h<4;h++){float p=probs[head+h*8][k];acc[h][0]=__fmaf_rn(p,v0,acc[h][0]);acc[h][1]=__fmaf_rn(p,v1,acc[h][1]);}
  }
  __syncthreads();
 }
 #pragma unroll
 for(int h=0;h<4;h++)for(int d=0;d<2;d++)output[((long long)row*64+h0+head+h*8)*512+d0+lane+d*32]=acc[h][d];
}
