// TF32 high/residual contractions over exact FP32 unpacking of K6 values.
// No persistent cache representation or cache quantisation is changed.
#include <mma.h>
namespace mla_tensor {
using namespace nvcuda;
using A=wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major>;
using Bc=wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major>;
using Br=wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major>;
using Acc=wmma::fragment<wmma::accumulator,16,16,8,float>;
template<class F> __device__ __forceinline__ void parts(F&hi,F&lo){
 #pragma unroll
 for(int i=0;i<F::num_elements;i++){const float x=hi.x[i];hi.x[i]=wmma::__float_to_tf32(x);lo.x[i]=wmma::__float_to_tf32(x-hi.x[i]);}
}
template<class F> __device__ __forceinline__ void product(Acc&c,const A&ah,const A&al,const F&bh,const F&bl){
 wmma::mma_sync(c,ah,bh,c);wmma::mma_sync(c,ah,bl,c);wmma::mma_sync(c,al,bh,c);wmma::mma_sync(c,al,bl,c);
}
}
extern "C" __global__ void mla_sparse_k6_tensor_scores(
 float* scores,const float* q,const float* values,int padded_capacity
){
#if __CUDA_ARCH__ >= 800
 using namespace mla_tensor;
 const int head=blockIdx.x*16,slot=blockIdx.y*64+(threadIdx.x/32)*16;
 if(slot>=padded_capacity)return;
 A ah,al;Bc bh,bl;Acc acc;wmma::fill_fragment(acc,0.0f);
 for(int band=0;band<512;band+=128){
  Acc local;wmma::fill_fragment(local,0.0f);
  for(int k=band;k<band+128;k+=8){
   wmma::load_matrix_sync(ah,q+(long long)head*512+k,512);
   wmma::load_matrix_sync(bh,values+(long long)slot*512+k,512);
   parts(ah,al);parts(bh,bl);product(local,ah,al,bh,bl);
  }
  #pragma unroll
  for(int i=0;i<Acc::num_elements;i++)acc.x[i]+=local.x[i];
 }
 wmma::store_matrix_sync(scores+(long long)head*padded_capacity+slot,acc,padded_capacity,wmma::mem_row_major);
#endif
}
extern "C" __global__ void mla_sparse_k6_tensor_softmax(
 float* probabilities,const float* scores,const int* indices,float scale,
 int capacity,int padded_capacity,int sequence_length,const int* sequence_pointer
){
 const int seq=sequence_pointer?*sequence_pointer:sequence_length;
 const int count=min(max(seq,0),capacity),tid=threadIdx.x,h=blockIdx.x;
 const int nw=blockDim.x/32;
 extern __shared__ float tensor_softmax_shared[];
 float* reduce=tensor_softmax_shared;float* weight=reduce+nw;
 float m=-1e30f;
 for(int i=tid;i<padded_capacity;i+=blockDim.x){
  int pos=i<count?indices[i]:-1;
  const float s=pos>=0 && pos<seq?scores[(long long)h*padded_capacity+i]*scale:-1e30f;
  weight[i]=s;m=fmaxf(m,s);
 }
 for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,o));
 if((tid&31)==0)reduce[tid/32]=m;
 __syncthreads();
 if(tid==0){float mx=reduce[0];for(int w=1;w<nw;w++)mx=fmaxf(mx,reduce[w]);reduce[0]=mx;}
 __syncthreads();m=reduce[0];__syncthreads();
 float sum=0;
 for(int i=tid;i<padded_capacity;i+=blockDim.x){
  int pos=i<count?indices[i]:-1;
  const float e=pos>=0 && pos<seq?__expf(weight[i]-m):0.0f;weight[i]=e;sum+=e;
 }
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 if((tid&31)==0)reduce[tid/32]=sum;
 __syncthreads();
 if(tid==0){float s=0;for(int w=0;w<nw;w++)s+=reduce[w];reduce[0]=s>0?1.0f/s:0.0f;}
 __syncthreads();const float inv=reduce[0];
 for(int i=tid;i<padded_capacity;i+=blockDim.x)probabilities[(long long)h*padded_capacity+i]=weight[i]*inv;
}
extern "C" __global__ void mla_sparse_k6_tensor_values(
 float* partial,const float* probabilities,const float* values,
 int capacity,int padded_capacity
){
#if __CUDA_ARCH__ >= 800
 using namespace mla_tensor;
 const int head=blockIdx.x*16,split=blockIdx.y,column=blockIdx.z*64+(threadIdx.x/32)*16;
 const int splits=(capacity+127)/128,begin=split*128,end=min(begin+128,padded_capacity);
 A ah,al;Br bh,bl;Acc acc;wmma::fill_fragment(acc,0.0f);
 for(int k=begin;k<end;k+=8){
  wmma::load_matrix_sync(ah,probabilities+(long long)head*padded_capacity+k,padded_capacity);
  wmma::load_matrix_sync(bh,values+(long long)k*512+column,512);
  parts(ah,al);parts(bh,bl);product(acc,ah,al,bh,bl);
 }
 wmma::store_matrix_sync(partial+(long long)head*splits*512+split*512+column,acc,splits*512,wmma::mem_row_major);
#endif
}
extern "C" __global__ void mla_sparse_k6_tensor_combine(
 float* output,const float* partial,int splits
){
 const int h=blockIdx.x;
 for(int d=threadIdx.x;d<512;d+=blockDim.x){
  float acc=0;
  for(int s=0;s<splits;s++)acc+=partial[((long long)h*splits+s)*512+d];
  output[h*512+d]=acc;
 }
}
