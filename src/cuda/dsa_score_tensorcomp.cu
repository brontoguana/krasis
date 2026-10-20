#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
// Router F32 output: same fixed K16 compensated dense contraction for every batch.
extern "C" __global__ void dsa_score_tensorcomp_f32(const unsigned short* input,
 const unsigned short* weight,float* output,int rows,int cols,unsigned long long tokens, int heads,int head_start,int head_count,float scale) {
 const unsigned long long mb=(unsigned long long)blockIdx.x*64;
 const unsigned long long nb=((unsigned long long)blockIdx.z*gridDim.y+blockIdx.y)*64;
 const int tid=threadIdx.x,lane=tid%32,wm=(tid/32)/4,wn=(tid/32)%4;
 __shared__ __align__(32) __nv_bfloat16 a[64*16],b[64*16];
 __shared__ __align__(32) float c[64*64];
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> cf;
 float hi[decltype(cf)::num_elements]={},lo[decltype(cf)::num_elements]={};
 const bool live=mb+wm*16<tokens&&nb+wn*16<(unsigned long long)rows;
 for(long long kb=0;kb<cols;kb+=16) {
  for(int i=tid;i<64*16;i+=blockDim.x) {
   const int r=i/16,k=i%16;
   unsigned short av=0,bv=0;
   if(mb+r<tokens&&kb+k<cols)av=input[(((mb+r)/(unsigned long long)head_count)*(unsigned long long)heads+(mb+r)%(unsigned long long)head_count+(unsigned long long)head_start)*(unsigned long long)cols+kb+k];
   if(nb+r<(unsigned long long)rows&&kb+k<cols)bv=weight[(nb+r)*(unsigned long long)cols+kb+k];
   a[i]=*reinterpret_cast<__nv_bfloat16*>(&av);b[i]=*reinterpret_cast<__nv_bfloat16*>(&bv);
  }
  __syncthreads();
  if(live) {
   wmma::load_matrix_sync(af,a+wm*16*16,16);wmma::load_matrix_sync(bf,b+wn*16*16,16);
   wmma::fill_fragment(cf,0.0f);wmma::mma_sync(cf,af,bf,cf);
   for(int j=0;j<cf.num_elements;++j) {
    const float p=cf.x[j],s=__fadd_rn(hi[j],p),v=__fsub_rn(s,hi[j]);
    const float e=__fadd_rn(__fsub_rn(hi[j],__fsub_rn(s,v)),__fsub_rn(p,v));
    hi[j]=s;lo[j]=__fadd_rn(lo[j],e);
   }
  }
  __syncthreads();
 }
 if(live) {
  for(int j=0;j<cf.num_elements;++j)cf.x[j]=__double2float_rn(__dadd_rn((double)hi[j],(double)lo[j]));
  wmma::store_matrix_sync(c+wm*16*64+wn*16,cf,64,wmma::mem_row_major);__syncwarp();
  for(int i=lane;i<256;i+=32) {
   const int m=wm*16+i/16,n=wn*16+i%16;
   if(mb+m<tokens&&nb+n<(unsigned long long)rows) {
    const unsigned long long token=(mb+m)/(unsigned long long)head_count, head=(mb+m)%(unsigned long long)head_count;
    output[(head*(tokens/(unsigned long long)head_count)+token)*(unsigned long long)rows+nb+n]=__fmul_rn(c[m*64+n],scale);
   }
  }
 }
}
