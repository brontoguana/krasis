#include <cuda_bf16.h>
#include <mma.h>
// Explicit prototype contract: logical F32 Marlin dequantization, two TF32
// components, fixed eight-product partials and compensated F32 accumulation.
// No expanded weight allocation, BF16 weight rounding or M-dependent reduction.
__device__ __forceinline__ void add_part(float x,float& hi,float& lo) {
 const float s=__fadd_rn(hi,x),v=__fsub_rn(s,hi);
 const float e=__fadd_rn(__fsub_rn(hi,__fsub_rn(s,v)),__fsub_rn(x,v));
 const float r=__fadd_rn(lo,e),t=__fadd_rn(s,r),u=__fsub_rn(t,s);
 lo=__fadd_rn(__fsub_rn(s,__fsub_rn(t,u)),__fsub_rn(r,u));hi=t;
}
extern "C" __global__ void marlin_projection_tensorcomp(
 const unsigned int* packed,const unsigned short* scales,const int* inverse_weight,
 const int* inverse_scale,const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int scale_perm_size,int bits,unsigned long long tokens) {
 namespace w=nvcuda::wmma;
 const unsigned long long mb=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const int nb=blockIdx.y*64,tid=threadIdx.x,warp=tid/32;
 if(mb>=tokens||nb>=rows)return;
 __shared__ __align__(32) float staging[(2*64+32)*40];
 float* ah=staging;float* al=ah+64*40;float* bx=al+64*40;
 w::fragment<w::accumulator,16,16,8,float> ch,cl,hi,lo;
 w::fill_fragment(hi,0.0f);w::fill_fragment(lo,0.0f);
 const int pack=32/bits;const unsigned mask=(1u<<bits)-1;
 for(int kb=0;kb<cols;kb+=32) {
  for(int i=tid;i<64*32;i+=blockDim.x) {
   const int n=nb+i/32,k=kb+i%32;float value=0.0f;
   if(n<rows&&k<cols) {
    const unsigned long long tiled=(unsigned long long)(n/16)*256+(k%16)*16+n%16;
    const unsigned long long permuted=(tiled/1024)*1024+inverse_weight[tiled%1024];
    const unsigned long long word=(unsigned long long)(k/16)*(rows*16/pack)+permuted/pack;
    const int q=(int)((packed[word]>>((permuted%pack)*bits))&mask)-(1<<(bits-1));
    const unsigned long long sf=(unsigned long long)(k/group_size)*rows+n;
    const unsigned long long sp=(sf/scale_perm_size)*scale_perm_size+inverse_scale[sf%scale_perm_size];
    const float scale=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(scales+sp));
    value=__fmul_rn((float)q,scale);
   }
   const float high=__uint_as_float(__float_as_uint(value)&0xffffe000U);
   const float low=__fsub_rn(value,high);
   ah[(i/32)*40+i%32]=w::__float_to_tf32(high);
   al[(i/32)*40+i%32]=w::__float_to_tf32(low);
  }
  for(int i=tid;i<32*32;i+=blockDim.x) {
   const unsigned long long m=mb+i/32;const int k=kb+i%32;
   float x=0.0f;if(m<tokens&&k<cols)x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+m*cols+k));
   bx[(i/32)*40+i%32]=w::__float_to_tf32(x);
  }
  __syncthreads();
  #pragma unroll
  for(int kk=0;kk<32;kk+=8) {
   w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> a;
   w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::col_major> b;
   w::load_matrix_sync(b,bx+(warp%2)*16*40+kk,40);
   w::fill_fragment(ch,0.0f);w::fill_fragment(cl,0.0f);
   w::load_matrix_sync(a,ah+(warp/2)*16*40+kk,40);w::mma_sync(ch,a,b,ch);
   w::load_matrix_sync(a,al+(warp/2)*16*40+kk,40);w::mma_sync(cl,a,b,cl);
   for(int j=0;j<ch.num_elements;j++){add_part(ch.x[j],hi.x[j],lo.x[j]);add_part(cl.x[j],hi.x[j],lo.x[j]);}
  }
  __syncthreads();
 }
 for(int j=0;j<ch.num_elements;j++)ch.x[j]=__fadd_rn(hi.x[j],lo.x[j]);
 w::store_matrix_sync(staging+warp*256,ch,16,w::mem_row_major);__syncthreads();
 for(int i=tid;i<8*256;i+=blockDim.x) {
  const int wi=i/256,j=i%256,n=nb+(wi/2)*16+j/16;
  const unsigned long long m=mb+(wi%2)*16+j%16;
  if(n<rows&&m<tokens){const float sum=staging[i];output_f32[m*rows+n]=sum;
   const __nv_bfloat16 y=__float2bfloat16_rn(sum);output[m*rows+n]=*reinterpret_cast<const unsigned short*>(&y);}
 }
}
