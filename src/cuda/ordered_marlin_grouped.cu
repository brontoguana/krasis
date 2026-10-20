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
__device__ __forceinline__ unsigned input_pair(const unsigned short* input,unsigned long long row,long long col,int cols,unsigned long long tokens,const int* sorted=nullptr,unsigned long long row_end=0) {
 if(sorted && row>=row_end)return 0u;
 const unsigned long long slot=sorted?(unsigned int)sorted[row]:row;
 return slot<tokens ? *reinterpret_cast<const unsigned*>(input+slot*(unsigned long long)cols+col):0u;
}
template<int Bits,bool Aligned> __device__ __forceinline__ void fragment_body(
 const unsigned* packed,const unsigned short* scales,const int* invw,const int* invs,
 const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int scale_perm_size,unsigned long long tokens,
 const int* sorted=nullptr,unsigned long long start=0,unsigned long long row_end=0) {
 const int tid=threadIdx.x,lane=tid%32,warp=tid/32,gid=lane/4,sub=lane%4;
 const unsigned long long mb=sorted?start:((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const unsigned long long end=sorted?row_end:tokens;
 const int nb=blockIdx.y*64,n=nb+warp*8+gid;
 if(mb>=end||nb>=rows)return;
 const int pack=32/Bits,mask=(1u<<Bits)-1;
 const unsigned long long stride=(unsigned long long)rows*(16/pack);
 // Exact closed form of the native Marlin permutation. Each lane's four
 // K coefficients are in one packed word; constants describe the codec only.
 const unsigned long long column=Bits==4 ? (unsigned long long)(n/64)*128+(n%8)*16+sub*4+(n%64)/16 : (unsigned long long)(n/64)*256+(n%8)*32+sub*8+(n%64)/8;
 const int base_shift=Bits==4 ? (n%16/8)*8:0;
 const unsigned long long scale_column=scale_perm_size==64 ? (unsigned long long)(n/64)*64+(n%8)*8+(n%64)/8 : (unsigned long long)(n/32)*32+(n%8/2)*8+(n%32/8)*2+n%2;
 unsigned long long group_index=0,next_group=(unsigned long long)group_size;
 float cached_scale=Aligned?__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(scales+scale_column)):0.0f;
 float hi[2][4]={{}},lo[2][4]={{}};
 for(long long kb=0;kb<cols;kb+=16) {
  const unsigned long long packed_row=(kb/16)*stride;
  if(Aligned && (unsigned long long)kb>=next_group) {++group_index;next_group+=(unsigned long long)group_size;cached_scale=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(scales+group_index*rows+scale_column));}
  const unsigned packed_word=packed[packed_row+column]>>base_shift;
  unsigned short hb[4],lb[4];
  #pragma unroll
  for(int j=0;j<4;j++) {
   const int sh=Bits==4 ? (j%2)*16+(j/2)*4 : (j%2)*16+(j/2)*8;
   const int q=(int)((packed_word>>sh)&mask)-(1<<(Bits-1));
   const long long k=kb+sub*2+(j&1)+(j>=2?8:0);
   const float scale=Aligned?cached_scale:__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(scales+(unsigned long long)(k/group_size)*rows+scale_column));
   const float value=__fmul_rn((float)q,scale);hb[j]=__float_as_uint(value)>>16;
   const float high=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(hb+j));
   const __nv_bfloat16 low=__float2bfloat16_rn(__fsub_rn(value,high));lb[j]=*reinterpret_cast<const unsigned short*>(&low);
  }
  const unsigned bh[2]={(unsigned)hb[0]|((unsigned)hb[1]<<16),(unsigned)hb[2]|((unsigned)hb[3]<<16)};
  const unsigned bl[2]={(unsigned)lb[0]|((unsigned)lb[1]<<16),(unsigned)lb[2]|((unsigned)lb[3]<<16)};
  #pragma unroll
  for(int t=0;t<2;t++) {
   if(mb+t*16>=end)continue; // uniform dead tile; live partials unchanged
   const unsigned long long row=mb+t*16+gid;
   const unsigned a[4]={input_pair(input,row,kb+sub*2,cols,tokens,sorted,end),input_pair(input,row+8,kb+sub*2,cols,tokens,sorted,end),input_pair(input,row,kb+sub*2+8,cols,tokens,sorted,end),input_pair(input,row+8,kb+sub*2+8,cols,tokens,sorted,end)};
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
   if(m<end&&out_n<rows) {
    const unsigned long long slot=sorted?(unsigned int)sorted[m]:m;
    if(slot>=tokens)continue;
    const float value=__fadd_rn(hi[t][j],lo[t][j]);if(output_f32)output_f32[slot*(unsigned long long)rows+out_n]=value;
    const __nv_bfloat16 y=__float2bfloat16_rn(value);output[slot*(unsigned long long)rows+out_n]=*reinterpret_cast<const unsigned short*>(&y);
   }
  }
 }
}
extern "C" __global__ void marlin_projection_tensorcomp(
 const unsigned* packed,const unsigned short* scales,const int* inverse_weight,
 const int* inverse_scale,const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int scale_perm_size,int bits,unsigned long long tokens) {
 if(bits==4){if(group_size%16==0)fragment_body<4,true>(packed,scales,inverse_weight,inverse_scale,input,output,output_f32,rows,cols,group_size,scale_perm_size,tokens);else fragment_body<4,false>(packed,scales,inverse_weight,inverse_scale,input,output,output_f32,rows,cols,group_size,scale_perm_size,tokens);}
 else if(bits==8){if(group_size%16==0)fragment_body<8,true>(packed,scales,inverse_weight,inverse_scale,input,output,output_f32,rows,cols,group_size,scale_perm_size,tokens);else fragment_body<8,false>(packed,scales,inverse_weight,inverse_scale,input,output,output_f32,rows,cols,group_size,scale_perm_size,tokens);}
}

// Expanded input/output slots use the existing prefill sorter contract (top_k=1).
// Each CTA stays within exactly one expert block. -1 blocks are precision filters.
extern "C" __global__ void marlin_projection_grouped(
 const unsigned* packed,const unsigned short* scales,const unsigned long long* packed_ptrs,
 const unsigned long long* scale_ptrs,const unsigned short* input,unsigned short* output,
 const int* sorted_ids,const int* expert_ids,const int* post_count,
 int block_size,int experts,int rows,int cols,int group_size,int bits,
 unsigned long long slots,unsigned long long sorted_capacity) {
 const unsigned long long tiles=((unsigned long long)block_size+31)/32;
 const unsigned long long bid=blockIdx.x/tiles;
 const unsigned long long block_start=bid*(unsigned long long)block_size;
 const unsigned long long count=(unsigned int)(*post_count);
 if(block_start>=count||block_start>=sorted_capacity)return;
 const int expert=expert_ids[bid];
 if(expert<0||expert>=experts)return;
 const unsigned long long start=block_start+(blockIdx.x%tiles)*32;
 unsigned long long end=block_start+(unsigned long long)block_size;
 if(end>count)end=count;if(end>sorted_capacity)end=sorted_capacity;
 if(start>=end)return;
 const unsigned long long weight_words=(unsigned long long)rows*cols/(32/bits);
 const unsigned long long scale_values=(unsigned long long)rows*((cols+(long long)group_size-1)/group_size);
 const unsigned* pw=packed_ptrs?reinterpret_cast<const unsigned*>(packed_ptrs[expert]):packed+(unsigned long long)expert*weight_words;
 const unsigned short* ps=scale_ptrs?reinterpret_cast<const unsigned short*>(scale_ptrs[expert]):scales+(unsigned long long)expert*scale_values;
 const int scale_perm_size=group_size<cols?64:32;
 if(bits==4){if(group_size%16==0)fragment_body<4,true>(pw,ps,nullptr,nullptr,input,output,nullptr,rows,cols,group_size,scale_perm_size,slots,sorted_ids,start,end);else fragment_body<4,false>(pw,ps,nullptr,nullptr,input,output,nullptr,rows,cols,group_size,scale_perm_size,slots,sorted_ids,start,end);}
 else if(bits==8){if(group_size%16==0)fragment_body<8,true>(pw,ps,nullptr,nullptr,input,output,nullptr,rows,cols,group_size,scale_perm_size,slots,sorted_ids,start,end);else fragment_body<8,false>(pw,ps,nullptr,nullptr,input,output,nullptr,rows,cols,group_size,scale_perm_size,slots,sorted_ids,start,end);}
}
