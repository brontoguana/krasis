#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <math.h>
constexpr double ln2 = 0.693147180559945309417232121458176568;
constexpr double inv_ln2 = 1.442695040888963407359924681001892137;
// Full exponent interval below the theoretical half-minsubnormal threshold.
constexpr double zero_bound = -double(FLT_MAX_EXP - FLT_MIN_EXP + FLT_MANT_DIG + 2) * ln2;
__device__ __forceinline__ float ieee_mul(float a,float b){float z;asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(a),"f"(b));return z;}
__device__ __forceinline__ float ieee_fma(float a,float b,float c){float z;asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(a),"f"(b),"f"(c));return z;}
__device__ __forceinline__ float ieee_add(float a,float b){float z;asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(a),"f"(b));return z;}
__device__ __forceinline__ float ieee_sub(float a,float b){float z;asm("sub.rn.f32 %0,%1,%2;":"=f"(z):"f"(a),"f"(b));return z;}
__device__ __forceinline__ unsigned dual_product(unsigned bits,float fh,float fl,int n){
    const unsigned sign=bits&0x80000000u,absolute=bits&0x7fffffffu;
    if(absolute==0||(n==0&&__float_as_uint(fh)==0x3f800000u&&__float_as_uint(fl)==0))return bits;
    if(n==INT_MAX)return sign;
    unsigned sig=absolute&0x7fffffu;int exponent=int(absolute>>23)-127;
    if((absolute>>23)==0){const int shift=__clz(sig)-8;sig<<=shift;exponent=-126-shift;}else sig|=0x800000u;
    const float m=__uint_as_float(0x3f800000u|(sig&0x7fffffu));
    const float p=ieee_mul(m,fh),error=ieee_fma(m,fl,ieee_fma(m,fh,-p));
    const float high=ieee_add(p,error),residual=ieee_sub(error,ieee_sub(high,p));
    const unsigned hb=__float_as_uint(high),significand=(hb&0x7fffffu)|0x800000u;
    const int e=exponent-n+int((hb>>23)&255u)-127;
    if(e>=-126)return sign|(unsigned(e+127)<<23)|(hb&0x7fffffu);
    const int shift=-126-e;if(shift>FLT_MANT_DIG)return sign;
    const unsigned retained=significand>>shift,removed=significand&((1u<<shift)-1u),half=1u<<(shift-1);
    const bool up=removed>half||(removed==half&&(residual>0.f||(residual==0.f&&(retained&1u))));
    return sign|(retained+unsigned(up));
}
__device__ __forceinline__ double widen_ieee(float x){double z;asm("cvt.f64.f32 %0,%1;":"=d"(z):"f"(x));return z;}
__device__ __forceinline__ void factor(double g,float& hi,float& lo,int& power){
 if(g<=zero_bound){hi=0.f;lo=0.f;power=INT_MAX;return;}
 power=int(floor(-g*inv_ln2));const double remainder=fma(double(power),ln2,g),value=exp(remainder);
 double hd;asm("cvt.rn.f32.f64 %0,%1;":"=f"(hi):"d"(value));asm("cvt.f64.f32 %0,%1;":"=d"(hd):"f"(hi));
 const double residual=value-hd;asm("cvt.rn.f32.f64 %0,%1;":"=f"(lo):"d"(residual));
}
__device__ __forceinline__ float decay(float value,const float* hi,const float* lo,const int* power,size_t i){return __uint_as_float(dual_product(__float_as_uint(value),hi[i],lo[i],power[i]));}
// Exact prepared F32 import; Q already includes its producer scale.

extern "C" __global__ void kda_dual_decay_tables(const float* g,float* ih,float* il,int* ip,float* ph,float* pl,int* pp,int H,int D,const int* valid_rows,const int* error){
 if(*error)return;const int T=*valid_rows;if(blockIdx.y*16>=T)return;
 const int B=16,group=blockIdx.y,h=blockIdx.x,flat=int(blockIdx.z)*int(blockDim.x)+int(threadIdx.x),x=flat%D,idx=flat/D,t=idx/B,j=idx%B,valid=min(B,T-group*B);
 const size_t mat=((size_t)group*H+h)*B*B,offset=(size_t)group*B*H*D;
 if(idx<B){const size_t dst=(((size_t)group*H+h)*B+idx)*D+x;double sum=0.;for(int row=0;row<=idx&&row<valid;row++)sum+=widen_ieee(g[offset+(row*H+h)*D+x]);
  if(idx<valid)factor(sum,ih[dst],il[dst],ip[dst]);else{ih[dst]=0.f;il[dst]=0.f;ip[dst]=INT_MAX;}}
 if(idx<B*B){const size_t dst=(mat+idx)*D+x;if(t<valid&&j<=t){double sum=0.;for(int row=j+1;row<=t;row++)sum+=widen_ieee(g[offset+(row*H+h)*D+x]);factor(sum,ph[dst],pl[dst],pp[dst]);}
 else{ph[dst]=0.f;pl[dst]=0.f;pp[dst]=INT_MAX;}}
}
extern "C" __global__ void kda_dual_coeff_ieee(const float* q,const float* k,const float* beta,const float* ph,const float* pl,const int* pp,float* A,float* Q,int H,int D,const int* valid_rows,const int* error){
 if(*error)return;const int T=*valid_rows;if(blockIdx.y*16>=T)return;
 const int B=16,group=blockIdx.y,h=blockIdx.x,idx=threadIdx.x,t=idx/B,j=idx%B,valid=min(B,T-group*B);
 const size_t mat=((size_t)group*H+h)*B*B,offset=(size_t)group*B*H*D;
 if(idx<B*B){float av=0.f,qv=0.f;if(t<valid&&j<=t){for(int x=0;x<D;x++){float z=decay(k[offset+(j*H+h)*D+x],ph,pl,pp,(mat+idx)*D+x);if(j<t)av=ieee_fma(k[offset+(t*H+h)*D+x],z,av);qv=ieee_fma(q[offset+(t*H+h)*D+x],z,qv);}}
 A[mat+idx]=(t<valid&&j<t)?ieee_mul(av,beta[(group*B+t)*H+h]):0.f;Q[mat+idx]=qv;}
}
extern "C" __global__ void kda_dual_inverse_ieee(const float* A,float* inverse,const int* valid_rows,const int* error){
 if(*error||blockIdx.y*16>=*valid_rows)return;
 const int B=16,idx=threadIdx.x,t=idx/B,j=idx%B;const size_t mat=((size_t)blockIdx.y*gridDim.x+blockIdx.x)*B*B;
 __shared__ float inv[B*B];inv[idx]=t==j?1.f:0.f;__syncthreads();
 for(int row=0;row<B;row++){if(t==row&&j<row){float z=0.f;for(int k=j;k<row;k++)z=ieee_fma(-A[mat+row*B+k],inv[k*B+j],z);inv[idx]=z;}__syncthreads();}
 inverse[mat+idx]=inv[idx];
}
extern "C" __global__ void kda_dual_carry_ieee(const float* v,const float* beta,const float* k,const float* q,const float* ih,const float* il,const int* ip,const float* ph,const float* pl,const int* pp,const float* Q,const float* inverse,const float* incoming,float* out,float* states,float* final_state,float* committed,int H,int D,const int* valid_rows,int all_states,const int* error){
 if(*error)return;const int T=*valid_rows;
 const int B=16,C=16,MAX_D=128,h=blockIdx.x,lane=threadIdx.x%C,t=threadIdx.x/C,y=blockIdx.y*C+lane;const bool active=y<D;
 __shared__ float S[MAX_D*C],rhs[B*C],delta[B*C];const size_t HD=(size_t)H*D,ST=HD*D;
 for(int x=t;x<D;x+=B){const float z=active?incoming[(h*D+x)*D+y]:0.f;S[x*C+lane]=z;if(active)committed[(h*D+x)*D+y]=z;}
 __syncthreads();
 for(int group=0;group*B<T;group++){
  const int valid=min(B,T-group*B);const size_t offset=(size_t)group*B*HD,mat=((size_t)group*H+h)*B*B,inc=((size_t)group*H+h)*B;
  float z=0.f;if(active&&t<valid){for(int x=0;x<D;x++)z=ieee_fma(k[offset+(t*H+h)*D+x],decay(S[x*C+lane],ih,il,ip,(inc+t)*D+x),z);rhs[t*C+lane]=ieee_mul(beta[(group*B+t)*H+h],ieee_sub(v[(group*B+t)*HD+h*D+y],z));}else rhs[t*C+lane]=0.f;
  __syncthreads();z=0.f;for(int j=0;j<=t;j++)z=ieee_fma(inverse[mat+t*B+j],rhs[j*C+lane],z);delta[t*C+lane]=z;
  __syncthreads();
  if(active&&t<valid){float z=0.f;for(int x=0;x<D;x++)z=ieee_fma(q[offset+(t*H+h)*D+x],decay(S[x*C+lane],ih,il,ip,(inc+t)*D+x),z);for(int j=0;j<=t;j++)z=ieee_fma(Q[mat+t*B+j],delta[j*C+lane],z);out[(group*B+t)*HD+h*D+y]=z;}
  if(active&&all_states)for(int x=t;x<D;x+=B)for(int u=0;u<valid;u++){float z=decay(S[x*C+lane],ih,il,ip,(inc+u)*D+x);for(int j=0;j<=u;j++)z=ieee_fma(decay(k[offset+(j*H+h)*D+x],ph,pl,pp,(mat+u*B+j)*D+x),delta[j*C+lane],z);states[(group*B+u)*ST+(h*D+x)*D+y]=z;}
  __syncthreads();
  for(int x=t;x<D;x+=B){const int u=valid-1;float z=decay(S[x*C+lane],ih,il,ip,(inc+u)*D+x);for(int j=0;j<=u;j++)z=ieee_fma(decay(k[offset+(j*H+h)*D+x],ph,pl,pp,(mat+u*B+j)*D+x),delta[j*C+lane],z);S[x*C+lane]=z;if(active&&valid==B)committed[(h*D+x)*D+y]=z;}
  __syncthreads();
 }
 if(active)for(int x=t;x<D;x+=B)final_state[(h*D+x)*D+y]=S[x*C+lane];
}

extern "C" __global__ void kda_batch_emit_copy(const float* out,const float* final_state,const float* completed,float* emitted,float* incoming,float* complete,float* latest,const int* error,int P,int M,int H,int D){
 if(*error)return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D,st=hd*D;
 if(i<uint64_t(M)*hd)emitted[i]=__uint_as_float(__float_as_uint(out[uint64_t(P)*hd+i]));
 if(i<st){incoming[i]=__uint_as_float(__float_as_uint(completed[i]));complete[i]=__uint_as_float(__float_as_uint(completed[i]));latest[i]=__uint_as_float(__float_as_uint(final_state[i]));}
}

extern "C" __global__ void kda_cursor_emit_copy(const float* out,const float* final_state,const float* completed,float* emitted,float* incoming,float* complete,float* latest,const int* pending,const int* error,int H,int D){
 if(*error)return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D,st=hd*D;const int valid=*pending;
 if(i<hd)emitted[i]=__uint_as_float(__float_as_uint(out[uint64_t(valid-1)*hd+i]));
 if(i<st){latest[i]=__uint_as_float(__float_as_uint(final_state[i]));if(valid==16){incoming[i]=__uint_as_float(__float_as_uint(completed[i]));complete[i]=__uint_as_float(__float_as_uint(completed[i]));}}
}

// Transaction scratch preparation never advances persistent cursor/raw history.
extern "C" __global__ void kda_cursor_stage_count(const int* pending,const uint64_t* absolute,int* count,int* error){
 if(blockIdx.x||threadIdx.x)return;if(atomicAdd(error,0))return;const int p=*pending;
 if(p<0||p>=16||uint64_t(p)!=(*absolute%16)||*absolute==UINT64_MAX){atomicCAS(error,0,1);return;}
 *count=p+1;
}

__device__ __forceinline__ uint16_t kda_rne_bf16(float x){const uint32_t u=__float_as_uint(x);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
extern "C" __global__ void kda_commit_validate(const float* out,const float* final_state,const float* completed,const int* count,int P,int M,int H,int D,int* error){
 // Existing errors suppress stale scratch reads. Atomic firstfault updates
 // avoid races between peer validators; later boundaries gate commits.
 if(atomicAdd(error,0))return;
 const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D,st=hd*D;const int n=*count;
 const int rows=M==0?1:M;const int first=M==0?n-1:P;
 if(i<uint64_t(rows)*hd){const float x=out[uint64_t(first)*hd+i];if((__float_as_uint(x)&0x7f800000)==0x7f800000||(kda_rne_bf16(x)&0x7f80)==0x7f80)atomicCAS(error,0,4);}
 if(i<st){if((__float_as_uint(final_state[i])&0x7f800000)==0x7f800000||(__float_as_uint(completed[i])&0x7f800000)==0x7f800000)atomicCAS(error,0,4);}
}
extern "C" __global__ void kda_emit_bf16(const float* out,uint16_t* emitted,const int* count,int P,int M,int H,int D,int* error){
 if(atomicAdd(error,0))return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int rows=M==0?1:M,first=M==0?*count-1:P;
 if(i<uint64_t(rows)*hd)emitted[i]=kda_rne_bf16(out[uint64_t(first)*hd+i]);
}

extern "C" __global__ void kda_cursor_commit_meta(int* pending,uint64_t* absolute,const int* count,int* error){
 if(blockIdx.x||threadIdx.x)return;if(atomicAdd(error,0))return;*pending=*count==16?0:*count;*absolute+=1;
}
