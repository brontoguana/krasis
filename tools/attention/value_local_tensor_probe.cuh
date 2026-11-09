#pragma once

// Offline only: complete-K ownership, no global partial-output storage.
__global__ void value_local_tensor_probe(float* output,const float* weights,
 const unsigned char* cache,const int* indices,const int* positions,int cap,int context,unsigned long long* ticks){
#if __CUDA_ARCH__ >= 800
 using namespace mla_tensor;
 __shared__ __align__(256) float values[128][64];
 __shared__ __align__(256) float probs[32][128];
 const int row=blockIdx.z,h0=blockIdx.y*32,d0=blockIdx.x*64;
 const int warp=threadIdx.x/32,wh=(warp/4)*16,wd=(warp%4)*16;
 const int seq=min(max(positions[row]+1,0),context);
 unsigned long long prep=0,compute=0,wait1=0,wait2=0,t0=0,t1=0,t2=0,t3=0;
 Acc acc;wmma::fill_fragment(acc,0.0f);
 for(int begin=0;begin<cap;begin+=128){
  if(threadIdx.x==0)t0=clock64();
  for(int i=threadIdx.x;i<128*64;i+=256){int k=i/64,d=i%64,slot=begin+k;
   int pos=slot<cap?indices[(long long)row*cap+slot]:-1;
   values[k][d]=pos>=0&&pos<seq?mla_prefill_load_k6_value(cache,pos,512,d0+d):0.0f;
  }
  for(int i=threadIdx.x;i<32*128;i+=256){int h=i/128,k=i%128;
   probs[h][k]=begin+k<cap?weights[((long long)row*64+h0+h)*cap+begin+k]:0.0f;
  }
  if(threadIdx.x==0){t1=clock64();prep+=t1-t0;}
  __syncthreads();
  if(threadIdx.x==0){t2=clock64();wait1+=t2-t1;}
  for(int k=0;k<128;k+=8){
   A ah,al;Br bh,bl;
   wmma::load_matrix_sync(ah,&probs[wh][k],128);
   wmma::load_matrix_sync(bh,&values[k][wd],64);
   parts(ah,al);parts(bh,bl);product(acc,ah,al,bh,bl);
  }
  if(threadIdx.x==0){t3=clock64();compute+=t3-t2;}
  __syncthreads();
  if(threadIdx.x==0)wait2+=clock64()-t3;
 }
 if(threadIdx.x==0){int i=(row*2+blockIdx.y)*8+blockIdx.x;ticks[4*i]=prep;ticks[4*i+1]=wait1;ticks[4*i+2]=compute;ticks[4*i+3]=wait2;}
 wmma::store_matrix_sync(output+((long long)row*64+h0+wh)*512+d0+wd,acc,512,wmma::mem_row_major);
#endif
}
