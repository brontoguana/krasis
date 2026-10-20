#include <cuda_bf16.h>
// Explicit NVIDIA mma.m16n8k16 BF16 fragment mapping:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
// Packed Marlin stays packed. A warp computes 32 input rows x8 output channels;
// real logical F32 weights are split into exact BF16 high+residue components.
__device__ __forceinline__ void accumulate(float x,float& hi,float& lo) {
 const float s=__fadd_rn(hi,x),v=__fsub_rn(s,hi);
 const float e=__fadd_rn(__fsub_rn(hi,__fsub_rn(s,v)),__fsub_rn(x,v));
 const float r=__fadd_rn(lo,e),t=__fadd_rn(s,r),u=__fsub_rn(t,s);
 lo=__fadd_rn(__fsub_rn(s,__fsub_rn(t,u)),__fsub_rn(r,u));hi=t;
}
__device__ __forceinline__ void mma_partial(const unsigned* a,const unsigned* b,float* c) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
 : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
 "f"(0.0f),"f"(0.0f),"f"(0.0f),"f"(0.0f));
}
__device__ __forceinline__ unsigned input_pair(const unsigned short* input,unsigned long long row,long long col,int cols,unsigned long long tokens) {
 return row<tokens ? *reinterpret_cast<const unsigned*>(input+row*(unsigned long long)cols+col):0u;
}
template<int Bits> __device__ __forceinline__ void fragment_body(
 const unsigned* packed,const unsigned short* scales,const int* invw,const int* invs,
 const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int scale_perm_size,unsigned long long tokens) {
 const int tid=threadIdx.x,lane=tid%32,warp=tid/32,gid=lane/4,sub=lane%4;
 const unsigned long long mb=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const int nb=blockIdx.y*64,n=nb+warp*8+gid;
 if(mb>=tokens||nb>=rows)return;
 const int pack=32/Bits,mask=(1u<<Bits)-1;
 const unsigned long long stride=(unsigned long long)rows*(16/pack);
 unsigned long long column[4];int shift[4];
 #pragma unroll
 for(int j=0;j<4;j++) {
  const int tk=sub*2+(j&1)+(j>=2?8:0);
  const int p=invw[(n%64/16)*256+tk*16+n%16];
  column[j]=(unsigned long long)(n/64)*(1024/pack)+p/pack;shift[j]=(p%pack)*Bits;
 }
 const unsigned long long scale_column=(unsigned long long)(n/scale_perm_size)*scale_perm_size+invs[n%scale_perm_size];
 const bool aligned=(group_size%16)==0;
 int prior_group=-1;float cached_scale=0.0f;
 float hi[2][4]={{}},lo[2][4]={{}};
 for(long long kb=0;kb<cols;kb+=16) {
  const unsigned long long packed_row=(kb/16)*stride;
  if(aligned) {const int group=kb/group_size;if(group!=prior_group){prior_group=group;cached_scale=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(scales+(unsigned long long)group*rows+scale_column));}}
  unsigned short hb[4],lb[4];
  #pragma unroll
  for(int j=0;j<4;j++) {
   const int q=(int)((packed[packed_row+column[j]]>>shift[j])&mask)-(1<<(Bits-1));
   const long long k=kb+sub*2+(j&1)+(j>=2?8:0);
   const float scale=aligned?cached_scale:__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(scales+(unsigned long long)(k/group_size)*rows+scale_column));
   const float value=__fmul_rn((float)q,scale);hb[j]=__float_as_uint(value)>>16;
   const float high=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(hb+j));
   const __nv_bfloat16 low=__float2bfloat16_rn(__fsub_rn(value,high));lb[j]=*reinterpret_cast<const unsigned short*>(&low);
  }
  const unsigned bh[2]={(unsigned)hb[0]|((unsigned)hb[1]<<16),(unsigned)hb[2]|((unsigned)hb[3]<<16)};
  const unsigned bl[2]={(unsigned)lb[0]|((unsigned)lb[1]<<16),(unsigned)lb[2]|((unsigned)lb[3]<<16)};
  #pragma unroll
  for(int t=0;t<2;t++) {
   const unsigned long long row=mb+t*16+gid;
   const unsigned a[4]={input_pair(input,row,kb+sub*2,cols,tokens),input_pair(input,row+8,kb+sub*2,cols,tokens),input_pair(input,row,kb+sub*2+8,cols,tokens),input_pair(input,row+8,kb+sub*2+8,cols,tokens)};
   float ch[4],cl[4];mma_partial(a,bh,ch);mma_partial(a,bl,cl);
   #pragma unroll
   for(int j=0;j<4;j++){accumulate(ch[j],hi[t][j],lo[t][j]);accumulate(cl[j],hi[t][j],lo[t][j]);}
  }
 }
 #pragma unroll
 for(int t=0;t<2;t++) {
  #pragma unroll
  for(int j=0;j<4;j++) {
   const unsigned long long m=mb+t*16+gid+(j>=2?8:0);
   const int out_n=nb+warp*8+sub*2+(j&1);
   if(m<tokens&&out_n<rows) {
    const float value=__fadd_rn(hi[t][j],lo[t][j]);output_f32[m*(unsigned long long)rows+out_n]=value;
    const __nv_bfloat16 y=__float2bfloat16_rn(value);output[m*(unsigned long long)rows+out_n]=*reinterpret_cast<const unsigned short*>(&y);
   }
  }
 }
}
extern "C" __global__ void marlin_projection_tensorcomp(
 const unsigned* packed,const unsigned short* scales,const int* inverse_weight,
 const int* inverse_scale,const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int scale_perm_size,int bits,unsigned long long tokens) {
 __shared__ int iw[1024],is[64];
 for(int i=threadIdx.x;i<1024;i+=blockDim.x)iw[i]=inverse_weight[i];
 if(threadIdx.x<scale_perm_size)is[threadIdx.x]=inverse_scale[threadIdx.x];
 __syncthreads();
 if(bits==4)fragment_body<4>(packed,scales,iw,is,input,output,output_f32,rows,cols,group_size,scale_perm_size,tokens);
 else if(bits==8)fragment_body<8>(packed,scales,iw,is,input,output,output_f32,rows,cols,group_size,scale_perm_size,tokens);
}
