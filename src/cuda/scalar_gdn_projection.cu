#include <cuda_bf16.h>
#include <mma.h>
// Isolated HQQ4 K1-product tensor-core control. Every K8 MMA has exactly one live input K slot; all eight slots are summed with ordered compensation. No serving dispatch.
__device__ __forceinline__ float component_trunc(float x) {return __uint_as_float(__float_as_uint(x)&0xffffe000u);}
__device__ __forceinline__ float protected_add(float a,float b) {
 float out;asm volatile("add.rn.f32 %0, %1, %2;":"=f"(out):"f"(a),"f"(b));return out;
}
__device__ __forceinline__ float protected_sub(float a,float b) {
 float out;asm volatile("sub.rn.f32 %0, %1, %2;":"=f"(out):"f"(a),"f"(b));return out;
}
__device__ __forceinline__ void magnitude_ordered_twosum(float a,float b,float& sum,float& error) {
 sum=protected_add(a,b);
 const bool a_larger=(__float_as_uint(a)&0x7fffffffU)>=(__float_as_uint(b)&0x7fffffffU);
 const float large=a_larger?a:b,small=a_larger?b:a;
 error=protected_sub(small,protected_sub(sum,large));
 // general_twosum's exact-zero error is positive zero, including (-0,-0).
 if((__float_as_uint(error)&0x7fffffffU)==0)error=0.0f;
}
__device__ __forceinline__ void persistent_ordered_twosum_add(float x,float& hi,float& lo) {
 float sum,error;magnitude_ordered_twosum(hi,x,sum,error);
 const float residue=protected_add(lo,error);magnitude_ordered_twosum(sum,residue,hi,lo);
}
// RN BF16 directly from the renormalized FP32 pair. At a BF16 midpoint,
// the retained residual decides the side; rounding through FP32 first loses it.
__device__ __forceinline__ __nv_bfloat16 round_compensated_bf16(float sum,float residual) {
 unsigned int b=__float_as_uint(sum);
 if((b&0xffffU)==0x8000U&&(b&0x7f800000U)!=0x7f800000U&&residual!=0.0f) {
  const bool toward_larger_magnitude=(residual>0.0f)==(sum>0.0f);
  b=toward_larger_magnitude?b+1U:b-1U;
 }
 return __float2bfloat16_rn(__uint_as_float(b));
}
extern "C" __global__ void hqq4_canonical_ordered_tf32x3(
 const unsigned char* packed_w,const float* scales,const float* zeros,
 const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int packed_row_stride_bytes,
 int scales_row_stride_bytes,unsigned long long tokens) {
 namespace w=nvcuda::wmma;
 const unsigned long long token_base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const int row_base=blockIdx.y*64;
 if(token_base>=tokens||row_base>=rows)return;
 const int tid=threadIdx.x,warp=tid>>5;
 // Fixed aligned float strides, valid for every supported model geometry.
 __shared__ __align__(32) float staging[(3*64+32)*40];
 __shared__ __align__(32) float masked_bx[32*40];
 float* ah=staging;float* am=ah+64*40;float* al=am+64*40;
 float* bx=al+64*40;
 w::fragment<w::accumulator,16,16,8,float> ch,cm,cl;
 w::fragment<w::accumulator,16,16,8,float> carry_hi,carry_lo;
 w::fill_fragment(carry_hi,0.0f);w::fill_fragment(carry_lo,0.0f);
 for(int kbase=0;kbase<cols;kbase+=32) {
   // Reconstruct exactly the old F32 weight before the explicitly different decomposition.
   for(int quartet=tid;quartet<64*8;quartet+=blockDim.x) {
     const int rr=quartet/8,kk=(quartet%8)*4,row=row_base+rr,col=kbase+kk;
     float scale=0.0f,zero=0.0f;unsigned int bits=0;
     if(row<rows&&col<cols) {
       const unsigned char* q=packed_w+(long long)row*packed_row_stride_bytes+(col/4)*2;
       bits=(unsigned int)q[0]|((unsigned int)q[1]<<8);
       const int group=col/group_size;
       scale=((const float*)((const char*)scales+(long long)row*scales_row_stride_bytes))[group];
       zero=((const float*)((const char*)zeros+(long long)row*scales_row_stride_bytes))[group];
     }
     #pragma unroll
     for(int off=0;off<4;++off) {
       float weight=(row<rows&&col+off<cols)?__fmul_rn(__fsub_rn((float)((bits>>(off*4))&15),zero),scale):0.0f;
       const float high=component_trunc(weight);
       const float residual=__fsub_rn(weight,high);
       const float middle=component_trunc(residual);
       const float low=component_trunc(__fsub_rn(residual,middle));
       // Converted TF32 values are used only by WMMA; never ordinary float operations.
       ah[rr*40+kk+off]=w::__float_to_tf32(high);
       am[rr*40+kk+off]=w::__float_to_tf32(middle);
       al[rr*40+kk+off]=w::__float_to_tf32(low);
     }
   }
   for(int i=tid;i<32*32;i+=blockDim.x) {
     const int token=i/32,kk=i%32,col=kbase+kk;
     float x=0.0f;
     if(token_base+token<tokens&&col<cols) {
       x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+(token_base+token)*(unsigned long long)cols+col));
     }
     bx[token*40+kk]=w::__float_to_tf32(x);
   }
   __syncthreads();
   #pragma unroll
   for(int kk=0;kk<32;kk+=8) {
    for(int product_slot=0;product_slot<8;++product_slot) {
     for(int item=tid;item<32*8;item+=blockDim.x) {
      const int token=item/8,k=item%8;
      masked_bx[token*40+k]=k==product_slot?bx[token*40+kk+k]:0.0f;
     }
     __syncthreads();
     w::fill_fragment(ch,0.0f);w::fill_fragment(cm,0.0f);w::fill_fragment(cl,0.0f);
     w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> a;
     w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::col_major> b;
     w::load_matrix_sync(b,masked_bx+(warp%2)*16*40,40);
     w::load_matrix_sync(a,ah+(warp/2)*16*40+kk,40);
     w::mma_sync(ch,a,b,ch);
     w::load_matrix_sync(a,am+(warp/2)*16*40+kk,40);
     w::mma_sync(cm,a,b,cm);
     w::load_matrix_sync(a,al+(warp/2)*16*40+kk,40);
     w::mma_sync(cl,a,b,cl);
     for(int i=0;i<ch.num_elements;++i) {
       persistent_ordered_twosum_add(ch.x[i],carry_hi.x[i],carry_lo.x[i]);
       persistent_ordered_twosum_add(cm.x[i],carry_hi.x[i],carry_lo.x[i]);
       persistent_ordered_twosum_add(cl.x[i],carry_hi.x[i],carry_lo.x[i]);
     }
     __syncthreads();
    }
   }
   __syncthreads();
 }
 // Uniform fragment element combination is documented; no mapping assumptions.
 for(int i=0;i<ch.num_elements;++i)magnitude_ordered_twosum(carry_hi.x[i],carry_lo.x[i],ch.x[i],cl.x[i]);
 // All staged operands are dead after the last barrier, so the output reuses storage.
 float* scratch=staging+warp*256;
 w::store_matrix_sync(scratch,ch,16,w::mem_row_major);
 w::store_matrix_sync(staging+8*256+warp*256,cl,16,w::mem_row_major);
 __syncthreads();
 for(int i=tid;i<8*256;i+=blockDim.x) {
   const int wi=i/256,local=i%256;
   const int row=row_base+(wi/2)*16+local/16;
   const unsigned long long token=token_base+(wi%2)*16+local%16;
   if(row<rows&&token<tokens) {
     const float sum=staging[i];
     if(output_f32) output_f32[token*(unsigned long long)rows+row]=sum;
     const __nv_bfloat16 value=round_compensated_bf16(sum,staging[8*256+i]);
     output[token*(unsigned long long)rows+row]=*reinterpret_cast<const unsigned short*>(&value);
   }
 }
}
