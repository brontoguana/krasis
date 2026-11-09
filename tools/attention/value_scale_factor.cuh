#pragma once
#include "../../src/cuda/mla_sparse_tensor.cuh"
// Offline only. One 16-feature scale group and all heads per CTA.
__global__ void value_scale_factor(float* output,const float* weights,
 const unsigned char* cache,const int* indices,const int* positions,int cap,int context){
#if __CUDA_ARCH__ >= 800
 using namespace mla_tensor;
 __shared__ __align__(256) float codes[128][16];
 __shared__ __align__(256) float probs[64][128];
 __shared__ float scales[128];
 const int row=blockIdx.z,d0=blockIdx.x*16,head=(threadIdx.x/32)*16;
 const int seq=min(max(positions[row]+1,0),context);
 Acc acc;wmma::fill_fragment(acc,0.0f);
 for(int begin=0;begin<cap;begin+=128){
  int slot=begin+threadIdx.x;
  int pos=slot<cap?indices[(long long)row*cap+slot]:-1;
  bool valid=pos>=0&&pos<seq;
  const unsigned char* packed=valid?cache+(long long)pos*448+(d0/16)*14:cache;
  scales[threadIdx.x]=valid?bf16_to_float(*reinterpret_cast<const __nv_bfloat16*>(packed)):0.0f;
  for(int d=0;d<16;d++)codes[threadIdx.x][d]=valid?float(mla_prefill_unpack_k6(packed+2,d)-32):0.0f;
  __syncthreads();
  for(int i=threadIdx.x;i<64*128;i+=128){int h=i/128,k=i%128;
   float p=begin+k<cap?weights[((long long)row*64+h)*cap+begin+k]:0.0f;
   probs[h][k]=__fmul_rn(p,scales[k]);
  }
  __syncthreads();
  for(int k=0;k<128;k+=8){
   A ah,al;Br b;
   wmma::load_matrix_sync(ah,&probs[head][k],128);
   wmma::load_matrix_sync(b,&codes[k][0],16);
   parts(ah,al);
   // Integer codes are exactly representable; no code residual product.
   wmma::mma_sync(acc,ah,b,acc);wmma::mma_sync(acc,al,b,acc);
  }
  __syncthreads();
 }
 wmma::store_matrix_sync(output+((long long)row*64+head)*512+d0,acc,512,wmma::mem_row_major);
#endif
}
