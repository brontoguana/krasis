#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ float vector_bits(float x){return __uint_as_float(__float_as_uint(x));}
__device__ __forceinline__ float vector_invalid(){return __uint_as_float(0x7fc1a123u);}
extern "C" __global__ void kda_prefix_validate(const int* pending,int* error,const uint64_t* absolute,uint64_t expected,int M){
 if(threadIdx.x||blockIdx.x||*error)return;
 if(M<=0||*pending<0||*pending>=16||*absolute!=expected||*pending!=int(expected%16)||expected>UINT64_MAX-uint64_t(M))*error=2;
}
extern "C" __global__ void kda_cursor_validate(const int* pending,int* error,const uint64_t* absolute){
 if(threadIdx.x||blockIdx.x||*error)return;
 if(*pending<0||*pending>=16||*absolute==UINT64_MAX||*pending!=int(*absolute%16))*error=1;
}
extern "C" __global__ void kda_prepared_validate(const float* q,const float* k,const float* v,const float* g,const float* beta,const int* count,int capacity,int H,int D,int* error){
 if(atomicAdd(error,0))return;
 const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int rows=*count;
 if(rows<0||rows>capacity){if(i==0)atomicCAS(error,0,3);return;}
 bool bad=false;
 if(i<uint64_t(rows)*hd){const uint32_t gb=__float_as_uint(g[i]);bad=((__float_as_uint(q[i])&0x7f800000u)==0x7f800000u)||((__float_as_uint(k[i])&0x7f800000u)==0x7f800000u)||((__float_as_uint(v[i])&0x7f800000u)==0x7f800000u)||((gb&0x7f800000u)==0x7f800000u)||(!(gb&0x80000000u)&&(gb&0x7fffffffu)!=0);}
 if(i<uint64_t(rows)*H){const uint32_t b=__float_as_uint(beta[i]),m=b&0x7fffffffu;bad|=((b&0x7f800000u)==0x7f800000u)||((b&0x80000000u)&&m!=0)||m>0x3f800000u;}
 if(bad)atomicCAS(error,0,3);
}
extern "C" __global__ void kda_prefix_prepend(const float* sq,const float* sk,const float* sv,const float* sg,const float* sb,const float* rq,const float* rk,const float* rv,const float* rg,const float* rb,float* q,float* k,float* v,float* g,float* beta,const int* pending,const int* error,const uint64_t* absolute,uint64_t expected,int M,int H,int D){
 if(*error)return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int P=int(expected%16);
 if(i<uint64_t(M+P)*hd){const uint64_t row=i/hd,at=row<uint64_t(P)?i:i-uint64_t(P)*hd;q[i]=vector_bits(row<uint64_t(P)?rq[at]:sq[at]);k[i]=vector_bits(row<uint64_t(P)?rk[at]:sk[at]);v[i]=vector_bits(row<uint64_t(P)?rv[at]:sv[at]);g[i]=vector_bits(row<uint64_t(P)?rg[at]:sg[at]);}
 if(i<uint64_t(M+P)*H){const uint64_t row=i/H,at=row<uint64_t(P)?i:i-uint64_t(P)*H;beta[i]=vector_bits(row<uint64_t(P)?rb[at]:sb[at]);}
}
extern "C" __global__ void kda_prepared_transport_import(const float* q,const float* k,const float* v,const float* g,const float* beta,float* Q,float* K,float* V,float* G,float* B,const int* count,int padded,int H,int D,const int* error){
 if(*error)return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int T=*count;
 if(i<uint64_t(padded)*hd){const bool valid=i<uint64_t(T)*hd;Q[i]=valid?vector_bits(q[i]):0.f;K[i]=valid?vector_bits(k[i]):0.f;V[i]=valid?vector_bits(v[i]):0.f;G[i]=valid?vector_bits(g[i]):0.f;}
 if(i<uint64_t(padded)*H)B[i]=i<uint64_t(T)*H?vector_bits(beta[i]):0.f;
}
extern "C" __global__ void kda_prefix_raw_finalize(const float* q,const float* k,const float* v,const float* g,const float* beta,float* rq,float* rk,float* rv,float* rg,float* rb,int* pending,const int* error,uint64_t* absolute,int P,int M,int H,int D){
 if(*error)return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int T=M+P,R=T%16,tail=T-R;
 if(i<16*hd){const bool valid=i/hd<uint64_t(R);const uint64_t at=uint64_t(tail)*hd+i;rq[i]=valid?vector_bits(q[at]):vector_invalid();rk[i]=valid?vector_bits(k[at]):vector_invalid();rv[i]=valid?vector_bits(v[at]):vector_invalid();rg[i]=valid?vector_bits(g[at]):vector_invalid();}
 if(i<16ull*H){const bool valid=i/H<uint64_t(R);const uint64_t at=uint64_t(tail)*H+i;rb[i]=valid?vector_bits(beta[at]):vector_invalid();}
 if(i==0){*pending=R;*absolute+=uint64_t(M);}
}
extern "C" __global__ void kda_cursor_stage_raw(const float* q,const float* k,const float* v,const float* g,const float* b,const float* pq,const float* pk,const float* pv,const float* pg,const float* pb,float* jq,float* jk,float* jv,float* jg,float* jb,const int* count,int* error,int H,int D){
 if(atomicAdd(error,0))return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int p=*count-1;
 if(i<16*hd){const int t=i/hd;const uint64_t x=i%hd;const float* srcq=t<p?pq:q;const float* srck=t<p?pk:k;const float* srcv=t<p?pv:v;const float* srcg=t<p?pg:g;const uint64_t at=t<p?i:x;const bool valid=t<=p;jq[i]=valid?vector_bits(srcq[at]):vector_invalid();jk[i]=valid?vector_bits(srck[at]):vector_invalid();jv[i]=valid?vector_bits(srcv[at]):vector_invalid();jg[i]=valid?vector_bits(srcg[at]):vector_invalid();}
 if(i<16ull*H){const int t=i/H;const uint64_t at=t<p?i:i%H;jb[i]=t<=p?vector_bits(t<p?pb[at]:b[at]):vector_invalid();}
}
extern "C" __global__ void kda_cursor_commit_raw(const float* jq,const float* jk,const float* jv,const float* jg,const float* jb,float* pq,float* pk,float* pv,float* pg,float* pb,const int* count,int* error,int H,int D){
 if(atomicAdd(error,0))return;const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,hd=uint64_t(H)*D;const int keep=*count==16?0:*count;
 if(i<16*hd){const bool valid=i/hd<uint64_t(keep);pq[i]=valid?vector_bits(jq[i]):vector_invalid();pk[i]=valid?vector_bits(jk[i]):vector_invalid();pv[i]=valid?vector_bits(jv[i]):vector_invalid();pg[i]=valid?vector_bits(jg[i]):vector_invalid();}
 if(i<16ull*H)pb[i]=i/H<uint64_t(keep)?vector_bits(jb[i]):vector_invalid();
}

// Diagnostic-only retained PreparedOwner failure table. Same marker ABI; no recurrence arithmetic.
extern "C" __global__ void kda_prepared_logits_guard(const uint64_t* errors, float* logits, int owners) {
 if(blockIdx.x || threadIdx.x) return;
 for(int i=0;i<owners;++i) if(*reinterpret_cast<const int*>(errors[i])!=0) {
  reinterpret_cast<uint32_t*>(logits)[0]=0x7fc04b52u; return;
 }
}
