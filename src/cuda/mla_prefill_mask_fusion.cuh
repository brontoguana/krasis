// Preserve FP32 operands and warp reductions while moving causal masking to
// softmax. The score GEMM can then omit the initialized-score read/write.
#pragma once
extern "C" __global__ void mla_prefill_gather_k6_no_mask(
 float* values,float* query,const float* q,const unsigned char* cache,
 const int* indices,const int* positions,int capacity,int context
){
 const int row=blockIdx.y,slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
 if(slot<capacity){
  const int seq=min(max(positions[row]+1,0),context);
  int pos=lane==0?indices[(long long)row*capacity+slot]:0;
  pos=__shfl_sync(0xffffffff,pos,0);
  for(int d=lane;d<512;d+=32)
   values[((long long)row*capacity+slot)*512+d]=pos>=0 && pos<seq?mla_prefill_load_k6_value(cache,pos,512,d):0.0f;
 }
 if(slot<64){
  for(int d=lane;d<512;d+=32){long long i=((long long)row*64+slot)*512+d;query[i]=q[i];}
 }
}
extern "C" __global__ void mla_prefill_masked_softmax_f32(
 float* weights,const float* scores,const int* indices,const int* positions,
 int rows,int capacity,int context
){
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 const long long score_row=(long long)blockIdx.x*8+warp;
 if(score_row>=(long long)rows*64)return;
 const int row=score_row/64,seq=min(max(positions[row]+1,0),context);
 const float* score=scores+score_row*capacity;
 const int* selected=indices+(long long)row*capacity;
 float local_max=-INFINITY;
 for(int i=lane;i<capacity;i+=32){
  int pos=selected[i];float v=pos>=0&&pos<seq?score[i]:-INFINITY;
  local_max=fmaxf(local_max,v);
 }
 for(int offset=16;offset;offset>>=1)local_max=fmaxf(local_max,__shfl_down_sync(0xffffffff,local_max,offset));
 const float row_max=__shfl_sync(0xffffffff,local_max,0);
 float local_sum=0;
 for(int i=lane;i<capacity;i+=32){
  int pos=selected[i];float v=pos>=0&&pos<seq?score[i]:-INFINITY;
  if(isfinite(v))local_sum+=__expf(v-row_max);
 }
 for(int offset=16;offset;offset>>=1)local_sum+=__shfl_down_sync(0xffffffff,local_sum,offset);
 const float denominator=__shfl_sync(0xffffffff,local_sum,0);
 for(int i=lane;i<capacity;i+=32){
  int pos=selected[i];float v=pos>=0&&pos<seq?score[i]:-INFINITY;
  weights[score_row*capacity+i]=isfinite(v)&&denominator>0?__expf(v-row_max)/denominator:0.0f;
 }
}
