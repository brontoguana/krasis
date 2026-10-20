#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
// Fixed K traversal and fragment shape for every M, including singleton/tails.
extern "C" __global__ void bf16_projection_wmma(const unsigned short* input,
 const unsigned short* weight,unsigned short* output,int rows,int cols,unsigned long long tokens) {
 const unsigned long long mb=(unsigned long long)blockIdx.x*16;
 const unsigned long long nb=((unsigned long long)blockIdx.z*gridDim.y+blockIdx.y)*16;
 const int lane=threadIdx.x;
 __shared__ __align__(32) __nv_bfloat16 a[256],b[256];
 __shared__ __align__(32) float c[256];
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> cf;
 wmma::fill_fragment(cf,0.0f);
 for(long long kb=0;kb<cols;kb+=16) {
  for(int i=lane;i<256;i+=32) {
   const int r=i/16,k=i%16;
   unsigned short av=0,bv=0;
   if(mb+r<tokens&&kb+k<cols)av=input[(mb+r)*(unsigned long long)cols+kb+k];
   if(nb+r<(unsigned long long)rows&&kb+k<cols)bv=weight[(nb+r)*(unsigned long long)cols+kb+k];
   a[i]=*reinterpret_cast<__nv_bfloat16*>(&av);b[i]=*reinterpret_cast<__nv_bfloat16*>(&bv);
  }
  __syncwarp();
  wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf,b,16);
  wmma::mma_sync(cf,af,bf,cf);__syncwarp();
 }
 wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);__syncwarp();
 for(int i=lane;i<256;i+=32) {
  const int m=i/16,n=i%16;
  if(mb+m<tokens&&nb+n<(unsigned long long)rows) {
   __nv_bfloat16 value=__float2bfloat16_rn(c[i]);
   output[(mb+m)*(unsigned long long)rows+nb+n]=*reinterpret_cast<unsigned short*>(&value);
  }
 }
}
// F64 control: same loaded operands, fixed per-thread K traversal and reduction.
extern "C" __global__ void bf16_projection_f64(const unsigned short* input,
 const unsigned short* weight,unsigned short* output,int rows,int cols,unsigned long long tokens) {
 const unsigned long long mb=(unsigned long long)blockIdx.x*4;
 const unsigned long long n=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 const int tid=threadIdx.x;
 if(mb>=tokens||n>=(unsigned long long)rows)return;
 double sum[4]={0.0,0.0,0.0,0.0};
 for(long long k=tid;k<cols;k+=blockDim.x) {
  const unsigned short wb=weight[n*(unsigned long long)cols+k];
  const double w=(double)__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&wb));
  for(int t=0;t<4;++t)if(mb+t<tokens) {
   const unsigned short xb=input[(mb+t)*(unsigned long long)cols+k];
   const double x=(double)__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&xb));
   sum[t]=__dadd_rn(sum[t],__dmul_rn(x,w));
  }
 }
 extern __shared__ double scratch[];
 for(int t=0;t<4;++t)if(mb+t<tokens) {
  scratch[tid]=sum[t];__syncthreads();
  for(int step=blockDim.x/2;step>0;step/=2) {
   if(tid<step)scratch[tid]=__dadd_rn(scratch[tid],scratch[tid+step]);
   __syncthreads();
  }
  if(tid==0) {
   __nv_bfloat16 value=__float2bfloat16_rn(__double2float_rn(scratch[0]));
   output[(mb+t)*(unsigned long long)rows+n]=*reinterpret_cast<unsigned short*>(&value);
  }
  __syncthreads();
 }
}
// Explicit compensated variant; no automatic precision substitution.
extern "C" __global__ void bf16_projection_compensated(const unsigned short* input,
 const unsigned short* weight,unsigned short* output,int rows,int cols,unsigned long long tokens) {
 const unsigned long long mb=(unsigned long long)blockIdx.x*4;
 const unsigned long long n=(unsigned long long)blockIdx.z*gridDim.y+blockIdx.y;
 const int tid=threadIdx.x;
 if(mb>=tokens||n>=(unsigned long long)rows)return;
 float hi[4]={0,0,0,0},lo[4]={0,0,0,0};
 for(long long k=tid;k<cols;k+=blockDim.x) {
  const unsigned short wb=weight[n*(unsigned long long)cols+k];
  const float w=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&wb));
  for(int t=0;t<4;++t)if(mb+t<tokens) {
   const unsigned short xb=input[(mb+t)*(unsigned long long)cols+k];
   const float x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&xb));
   const float p=__fmul_rn(x,w),pe=__fmaf_rn(x,w,-p);
   const float s=__fadd_rn(hi[t],p),v=__fsub_rn(s,hi[t]);
   const float e=__fadd_rn(__fsub_rn(hi[t],__fsub_rn(s,v)),__fsub_rn(p,v));
   hi[t]=s;lo[t]=__fadd_rn(lo[t],__fadd_rn(e,pe));
  }
 }
 extern __shared__ double scratch[];
 for(int t=0;t<4;++t)if(mb+t<tokens) {
  scratch[tid]=__dadd_rn((double)hi[t],(double)lo[t]);__syncthreads();
  for(int step=blockDim.x/2;step>0;step/=2) {
   if(tid<step)scratch[tid]=__dadd_rn(scratch[tid],scratch[tid+step]);
   __syncthreads();
  }
  if(tid==0) {
   __nv_bfloat16 value=__float2bfloat16_rn(__double2float_rn(scratch[0]));
   output[(mb+t)*(unsigned long long)rows+n]=*reinterpret_cast<unsigned short*>(&value);
  }
  __syncthreads();
 }
}

#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
// A fixed 16-product tensor-core partial, then F64 accumulation across K.
// Fixed arithmetic is independent of M, including incomplete tiles.
extern "C" __global__ void bf16_projection_chunked(const unsigned short* input,
 const unsigned short* weight,unsigned short* output,int rows,int cols,unsigned long long tokens) {
 const unsigned long long mb=(unsigned long long)blockIdx.x*16;
 const unsigned long long nb=((unsigned long long)blockIdx.z*gridDim.y+blockIdx.y)*16;
 const int lane=threadIdx.x;
 __shared__ __align__(32) __nv_bfloat16 a[256],b[256];
 __shared__ __align__(32) float c[256];
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> cf;
 double sums[8]={0,0,0,0,0,0,0,0};
 for(long long kb=0;kb<cols;kb+=16) {
  for(int i=lane;i<256;i+=32) {
   const int r=i/16,k=i%16;
   unsigned short av=0,bv=0;
   if(mb+r<tokens&&kb+k<cols)av=input[(mb+r)*(unsigned long long)cols+kb+k];
   if(nb+r<(unsigned long long)rows&&kb+k<cols)bv=weight[(nb+r)*(unsigned long long)cols+kb+k];
   a[i]=*reinterpret_cast<__nv_bfloat16*>(&av);b[i]=*reinterpret_cast<__nv_bfloat16*>(&bv);
  }
  __syncwarp();
  wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf,b,16);
  wmma::fill_fragment(cf,0.0f);wmma::mma_sync(cf,af,bf,cf);
  wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);__syncwarp();
  for(int j=0;j<8;++j)sums[j]=__dadd_rn(sums[j],(double)c[lane+j*32]);
  __syncwarp();
 }
 for(int j=0;j<8;++j) {
  const int i=lane+j*32,m=i/16,n=i%16;
  if(mb+m<tokens&&nb+n<(unsigned long long)rows) {
   __nv_bfloat16 value=__float2bfloat16_rn(__double2float_rn(sums[j]));
   output[(mb+m)*(unsigned long long)rows+nb+n]=*reinterpret_cast<unsigned short*>(&value);
  }
 }
}

#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
// Sixteen warps share 64x16 operand tiles. Each warp owns a 16x16 result.
// Each tensor partial contains sixteen products, accumulated across K in F64.
extern "C" __global__ void bf16_projection_tiled(const unsigned short* input,
 const unsigned short* weight,unsigned short* output,int rows,int cols,unsigned long long tokens) {
 const unsigned long long mb=(unsigned long long)blockIdx.x*64;
 const unsigned long long nb=((unsigned long long)blockIdx.z*gridDim.y+blockIdx.y)*64;
 const int tid=threadIdx.x,lane=tid%32,wm=(tid/32)/4,wn=(tid/32)%4;
 __shared__ __align__(32) __nv_bfloat16 a[64*16],b[64*16];
 __shared__ __align__(32) float c[64*64];
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> cf;
 double sums[decltype(cf)::num_elements]={};
 const bool live=mb+wm*16<tokens&&nb+wn*16<(unsigned long long)rows;
 for(long long kb=0;kb<cols;kb+=16) {
  for(int i=tid;i<64*16;i+=blockDim.x) {
   const int r=i/16,k=i%16;
   unsigned short av=0,bv=0;
   if(mb+r<tokens&&kb+k<cols)av=input[(mb+r)*(unsigned long long)cols+kb+k];
   if(nb+r<(unsigned long long)rows&&kb+k<cols)bv=weight[(nb+r)*(unsigned long long)cols+kb+k];
   a[i]=*reinterpret_cast<__nv_bfloat16*>(&av);b[i]=*reinterpret_cast<__nv_bfloat16*>(&bv);
  }
  __syncthreads();
  if(live) {
   wmma::load_matrix_sync(af,a+wm*16*16,16);wmma::load_matrix_sync(bf,b+wn*16*16,16);
   wmma::fill_fragment(cf,0.0f);wmma::mma_sync(cf,af,bf,cf);
   for(int j=0;j<cf.num_elements;++j)sums[j]=__dadd_rn(sums[j],(double)cf.x[j]);
  }
  __syncthreads();
 }
 if(live) {
  for(int j=0;j<cf.num_elements;++j)cf.x[j]=__double2float_rn(sums[j]);
  wmma::store_matrix_sync(c+wm*16*64+wn*16,cf,64,wmma::mem_row_major);__syncwarp();
  for(int i=lane;i<256;i+=32) {
   const int m=wm*16+i/16,n=wn*16+i%16;
   if(mb+m<tokens&&nb+n<(unsigned long long)rows) {
    __nv_bfloat16 value=__float2bfloat16_rn(c[m*64+n]);
    output[(mb+m)*(unsigned long long)rows+nb+n]=*reinterpret_cast<unsigned short*>(&value);
   }
  }
 }
}

#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
// Sixteen warps share 64x16 operand tiles. Each warp owns a 16x16 result.
// Sixteen-product partials use compensated F32 accumulation and one final F64 merge.
extern "C" __global__ void bf16_projection_tensorcomp(const unsigned short* input,
 const unsigned short* weight,unsigned short* output,int rows,int cols,unsigned long long tokens) {
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
   if(mb+r<tokens&&kb+k<cols)av=input[(mb+r)*(unsigned long long)cols+kb+k];
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
    __nv_bfloat16 value=__float2bfloat16_rn(c[m*64+n]);
    output[(mb+m)*(unsigned long long)rows+nb+n]=*reinterpret_cast<unsigned short*>(&value);
   }
  }
 }
}
