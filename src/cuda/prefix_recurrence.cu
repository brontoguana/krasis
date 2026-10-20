#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>
extern "C" __global__ void batch_g(const float* g,float* G,int H,int D,int B,int T) {
 int group=blockIdx.y;T=min(B,T-group*B);g+=group*B*H*D;G+=group*B*H*D;
 int hd=blockIdx.x*blockDim.x+threadIdx.x;if(hd>=H*D)return;
 float s=0.f;for(int t=0;t<B;t++){if(t<T)s=s+g[t*H*D+hd];G[t*H*D+hd]=s;}
}

extern "C" __global__ void batch_precompute_compact(const float* G,const float* k,const float* q,float* KG,float* QG,float* pair,int H,int D,int B,int T){
 int group=blockIdx.y,hd=blockIdx.x*blockDim.x+threadIdx.x;T=min(B,T-group*B);if(hd>=H*D)return;
 size_t offset=(size_t)group*B*H*D;for(int t=0;t<B;t++){size_t i=offset+t*H*D+hd;KG[i]=t<T?k[i]*expf(G[i]):0.f;QG[i]=t<T?q[i]*expf(G[i]):0.f;}
}

extern "C" __global__ void batch_coeff_compact(const float* q,const float* k,float* pair,const float* beta,const float* G,float* A,float* Q,int H,int D,int B,int T){
 int group=blockIdx.y,h=blockIdx.x,valid=min(B,T-group*B),idx=threadIdx.x;
 size_t offset=(size_t)group*B*H*D,mat=(size_t)group*H*B*B,po=(size_t)group*D*H*B;
 __shared__ float gs[16*128],ks[16*128];
 for(int i=idx;i<B*D;i+=blockDim.x){int t=i/D,x=i%D;gs[i]=G[offset+(t*H+h)*D+x];ks[i]=t<valid?k[offset+(t*H+h)*D+x]:0.f;}
 __syncthreads();
 if(idx<B*B){int j=idx%B,t=idx/B;float av=0.f,r=0.f;if(t<valid&&j<=t){for(int x=0;x<D;x++){float z=ks[j*D+x]*expf(gs[t*D+x]-gs[j*D+x]);if(j<t)av=fmaf(ks[t*D+x],z,av);r=fmaf(q[offset+(t*H+h)*D+x],z,r);}}A[mat+(h*B+t)*B+j]=(t<valid&&j<t)?av*beta[(group*B+t)*H+h]:0.f;Q[mat+(h*B+t)*B+j]=r;}
 for(int i=idx;i<D*B;i+=blockDim.x){int x=i/B,j=i%B,u=valid-1;float z=j<=u?ks[j*D+x]*expf(gs[u*D+x]-gs[j*D+x]):0.f;pair[po+(x*H+h)*B+j]=z;}
}

extern "C" __global__ void batch_inverse(const float* A,float* inverse,int B) {
 int group=blockIdx.y;A+=group*gridDim.x*B*B;inverse+=group*gridDim.x*B*B;
 extern __shared__ float buf[];float* R=buf;float* X=R+B*B;float* Y=X+B*B;
 int h=blockIdx.x,idx=threadIdx.x;int cells=B*B;
 for(int i=idx;i<cells;i+=blockDim.x){R[i]=(i/B==i%B?1.f:0.f)-A[h*cells+i];X[i]=-A[h*cells+i];}__syncthreads();
 // R=(I+X), X=-A; R <- R*(I+X^2), X <- X^2 at each
 // doubling. Strictly lower X is nilpotent, so no iterative convergence rule.
 for(int span=2;span<B;span*=2){
  for(int i=idx;i<cells;i+=blockDim.x){int t=i/B,j=i%B;float v=0.f;for(int n=0;n<B;n++)v=fmaf(X[t*B+n],X[n*B+j],v);Y[i]=v;}__syncthreads();
  for(int i=idx;i<cells;i+=blockDim.x)X[i]=Y[i];__syncthreads();
  for(int i=idx;i<cells;i+=blockDim.x){int t=i/B,j=i%B;float v=R[i];for(int n=0;n<B;n++)v=fmaf(R[t*B+n],X[n*B+j],v);Y[i]=v;}__syncthreads();
  for(int i=idx;i<cells;i+=blockDim.x)R[i]=Y[i];__syncthreads();
 }
 for(int i=idx;i<cells;i+=blockDim.x)inverse[h*cells+i]=R[i];
}

extern "C" __global__ void batch_carry_compact(const float* v,const float* beta,const float* G,const float* KG,const float* QG,const float* pair,const float* raw_k,const float* Q,const float* inverse,const float* incoming,float* out,float* states,float* final_state,float* committed,int H,int D,int B,int T,int start,int all_states,float scale,const int* error){
 if(*error)return;
 const int C=16,MAX_D=128;const int h=blockIdx.x,lane=threadIdx.x%C,t=threadIdx.x/C,y=blockIdx.y*C+lane;const bool active=y<D;
 __shared__ float S[MAX_D*C],rhs[16*C],delta[16*C];const size_t HD=(size_t)H*D,ST=HD*D;
 for(int x=t;x<D;x+=16){float z=active?incoming[(h*D+x)*D+y]:0.f;S[x*C+lane]=z;if(active)committed[(h*D+x)*D+y]=z;}
 __syncthreads();
 for(int group=start/B;group*B<T;group++){
  int valid=min(B,T-group*B);const size_t offset=(size_t)group*B*HD,mat=(size_t)group*H*B*B,po=(size_t)group*D*H*B;
  float z=0.f;if(active&&t<valid){for(int x=0;x<D;x++)z=fmaf(KG[offset+(t*H+h)*D+x],S[x*C+lane],z);rhs[t*C+lane]=beta[(group*B+t)*H+h]*(v[(group*B+t)*HD+h*D+y]-z);}else rhs[t*C+lane]=0.f;
  __syncthreads();z=0.f;for(int j=0;j<B;j++)z=fmaf(inverse[mat+(h*B+t)*B+j],rhs[j*C+lane],z);delta[t*C+lane]=z;
  __syncthreads();
  if(active&&t<valid){float z=0.f;for(int x=0;x<D;x++)z=fmaf(QG[offset+(t*H+h)*D+x],S[x*C+lane],z);for(int j=0;j<B;j++)z=fmaf(Q[mat+(h*B+t)*B+j],delta[j*C+lane],z);out[(group*B+t)*HD+h*D+y]=z*scale;}
  if(active&&all_states)for(int x=t;x<D;x+=16)for(int u=0;u<valid;u++){float z=S[x*C+lane]*expf(G[offset+(u*H+h)*D+x]);for(int j=0;j<B;j++){float p=j<=u?raw_k[offset+(j*H+h)*D+x]*expf(G[offset+(u*H+h)*D+x]-G[offset+(j*H+h)*D+x]):0.f;z=fmaf(p,delta[j*C+lane],z);}states[(group*B+u)*ST+(h*D+x)*D+y]=z;}
  __syncthreads();
  for(int x=t;x<D;x+=16){const int u=valid-1;float z=S[x*C+lane]*expf(G[offset+(u*H+h)*D+x]);for(int j=0;j<B;j++){float p=j<=u?pair[po+(x*H+h)*B+j]:0.f;z=fmaf(p,delta[j*C+lane],z);}S[x*C+lane]=z;if(active&&valid==B)committed[(h*D+x)*D+y]=z;}
  __syncthreads();
 }
 if(active)for(int x=t;x<D;x+=16)final_state[(h*D+x)*D+y]=S[x*C+lane];
}

#include <cuda_bf16.h>
extern "C" __global__ void raw_prepare(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const __nv_bfloat16* g,const __nv_bfloat16* beta,const float* A,const float* bias,float* Q,float* K,float* V,float* G,float* B,int T,int H,int D,float lower,float scale,float eps){
 int row=blockIdx.x,h=blockIdx.y,tid=threadIdx.x;if(row>=T||h>=H)return;
 extern __shared__ float reduce[];long long at=((long long)row*H+h)*D;
 float qv=tid<D?__bfloat162float(q[at+tid]):0.f,kv=tid<D?__bfloat162float(k[at+tid]):0.f;
 reduce[tid]=qv*qv;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
 float qi=rsqrtf(reduce[0]+eps)*scale;__syncthreads();reduce[tid]=kv*kv;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
 float ki=rsqrtf(reduce[0]+eps);if(tid<D){Q[at+tid]=qv*qi;K[at+tid]=kv*ki;V[at+tid]=__bfloat162float(v[at+tid]);float raw=__bfloat162float(g[at+tid])+bias[h*D+tid];G[at+tid]=lower/(1.f+__expf(-__expf(A[h])*raw));}if(tid==0)B[row*H+h]=1.f/(1.f+__expf(-__bfloat162float(beta[row*H+h])));
}

#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>
extern "C" __global__ void cursor_batch_g(const float* g,float* G,int H,int D,int B,const int* valid_rows) {
 int T=*valid_rows;

 int group=blockIdx.y;T=min(B,T-group*B);g+=group*B*H*D;G+=group*B*H*D;
 int hd=blockIdx.x*blockDim.x+threadIdx.x;if(hd>=H*D)return;
 float s=0.f;for(int t=0;t<B;t++){if(t<T)s=s+g[t*H*D+hd];G[t*H*D+hd]=s;}
}

extern "C" __global__ void cursor_batch_precompute_compact(const float* G,const float* k,const float* q,float* KG,float* QG,float* pair,int H,int D,int B,const int* valid_rows){
 int T=*valid_rows;

 int group=blockIdx.y,hd=blockIdx.x*blockDim.x+threadIdx.x;T=min(B,T-group*B);if(hd>=H*D)return;
 size_t offset=(size_t)group*B*H*D;for(int t=0;t<B;t++){size_t i=offset+t*H*D+hd;KG[i]=t<T?k[i]*expf(G[i]):0.f;QG[i]=t<T?q[i]*expf(G[i]):0.f;}
}

extern "C" __global__ void cursor_batch_coeff_compact(const float* q,const float* k,float* pair,const float* beta,const float* G,float* A,float* Q,int H,int D,int B,const int* valid_rows){
 int T=*valid_rows;

 int group=blockIdx.y,h=blockIdx.x,valid=min(B,T-group*B),idx=threadIdx.x;
 size_t offset=(size_t)group*B*H*D,mat=(size_t)group*H*B*B,po=(size_t)group*D*H*B;
 __shared__ float gs[16*128],ks[16*128];
 for(int i=idx;i<B*D;i+=blockDim.x){int t=i/D,x=i%D;gs[i]=G[offset+(t*H+h)*D+x];ks[i]=t<valid?k[offset+(t*H+h)*D+x]:0.f;}
 __syncthreads();
 if(idx<B*B){int j=idx%B,t=idx/B;float av=0.f,r=0.f;if(t<valid&&j<=t){for(int x=0;x<D;x++){float z=ks[j*D+x]*expf(gs[t*D+x]-gs[j*D+x]);if(j<t)av=fmaf(ks[t*D+x],z,av);r=fmaf(q[offset+(t*H+h)*D+x],z,r);}}A[mat+(h*B+t)*B+j]=(t<valid&&j<t)?av*beta[(group*B+t)*H+h]:0.f;Q[mat+(h*B+t)*B+j]=r;}
 for(int i=idx;i<D*B;i+=blockDim.x){int x=i/B,j=i%B,u=valid-1;float z=j<=u?ks[j*D+x]*expf(gs[u*D+x]-gs[j*D+x]):0.f;pair[po+(x*H+h)*B+j]=z;}
}

extern "C" __global__ void cursor_batch_carry_compact(const float* v,const float* beta,const float* G,const float* KG,const float* QG,const float* pair,const float* raw_k,const float* Q,const float* inverse,const float* incoming,float* out,float* states,float* final_state,float* committed,int H,int D,int B,const int* valid_rows,int start,int all_states,float scale,const int* error){
 if(*error)return;
 int T=*valid_rows;

 const int C=16,MAX_D=128;const int h=blockIdx.x,lane=threadIdx.x%C,t=threadIdx.x/C,y=blockIdx.y*C+lane;const bool active=y<D;
 __shared__ float S[MAX_D*C],rhs[16*C],delta[16*C];const size_t HD=(size_t)H*D,ST=HD*D;
 for(int x=t;x<D;x+=16){float z=active?incoming[(h*D+x)*D+y]:0.f;S[x*C+lane]=z;if(active)committed[(h*D+x)*D+y]=z;}
 __syncthreads();
 for(int group=start/B;group*B<T;group++){
  int valid=min(B,T-group*B);const size_t offset=(size_t)group*B*HD,mat=(size_t)group*H*B*B,po=(size_t)group*D*H*B;
  float z=0.f;if(active&&t<valid){for(int x=0;x<D;x++)z=fmaf(KG[offset+(t*H+h)*D+x],S[x*C+lane],z);rhs[t*C+lane]=beta[(group*B+t)*H+h]*(v[(group*B+t)*HD+h*D+y]-z);}else rhs[t*C+lane]=0.f;
  __syncthreads();z=0.f;for(int j=0;j<B;j++)z=fmaf(inverse[mat+(h*B+t)*B+j],rhs[j*C+lane],z);delta[t*C+lane]=z;
  __syncthreads();
  if(active&&t<valid){float z=0.f;for(int x=0;x<D;x++)z=fmaf(QG[offset+(t*H+h)*D+x],S[x*C+lane],z);for(int j=0;j<B;j++)z=fmaf(Q[mat+(h*B+t)*B+j],delta[j*C+lane],z);out[(group*B+t)*HD+h*D+y]=z*scale;}
  if(active&&all_states)for(int x=t;x<D;x+=16)for(int u=0;u<valid;u++){float z=S[x*C+lane]*expf(G[offset+(u*H+h)*D+x]);for(int j=0;j<B;j++){float p=j<=u?raw_k[offset+(j*H+h)*D+x]*expf(G[offset+(u*H+h)*D+x]-G[offset+(j*H+h)*D+x]):0.f;z=fmaf(p,delta[j*C+lane],z);}states[(group*B+u)*ST+(h*D+x)*D+y]=z;}
  __syncthreads();
  for(int x=t;x<D;x+=16){const int u=valid-1;float z=S[x*C+lane]*expf(G[offset+(u*H+h)*D+x]);for(int j=0;j<B;j++){float p=j<=u?pair[po+(x*H+h)*B+j]:0.f;z=fmaf(p,delta[j*C+lane],z);}S[x*C+lane]=z;if(active&&valid==B)committed[(h*D+x)*D+y]=z;}
  __syncthreads();
 }
 if(active)for(int x=t;x<D;x+=16)final_state[(h*D+x)*D+y]=S[x*C+lane];
}
#include <cuda_bf16.h>
extern "C" __global__ void cursor_raw_prepare(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const __nv_bfloat16* g,const __nv_bfloat16* beta,const float* A,const float* bias,float* Q,float* K,float* V,float* G,float* B,const int* valid_rows,int H,int D,float lower,float scale,float eps){
 int T=*valid_rows;
 int row=blockIdx.x,h=blockIdx.y,tid=threadIdx.x;if(row>=T||h>=H)return;
 extern __shared__ float reduce[];long long at=((long long)row*H+h)*D;
 float qv=tid<D?__bfloat162float(q[at+tid]):0.f,kv=tid<D?__bfloat162float(k[at+tid]):0.f;
 reduce[tid]=qv*qv;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
 float qi=rsqrtf(reduce[0]+eps)*scale;__syncthreads();reduce[tid]=kv*kv;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
 float ki=rsqrtf(reduce[0]+eps);if(tid<D){Q[at+tid]=qv*qi;K[at+tid]=kv*ki;V[at+tid]=__bfloat162float(v[at+tid]);float raw=__bfloat162float(g[at+tid])+bias[h*D+tid];G[at+tid]=lower/(1.f+__expf(-__expf(A[h])*raw));}if(tid==0)B[row*H+h]=1.f/(1.f+__expf(-__bfloat162float(beta[row*H+h])));
}


#include <cuda_bf16.h>
// Device pending rows and absolute consumed position. No fixture cursor.
extern "C" __global__ void cursor_advance(int* pending,int* error,const unsigned long long* absolute) {
 if(threadIdx.x||blockIdx.x)return;
 if(*error)return;
 if(*pending<0||*pending>=16||*absolute==~0ull||*pending!=(int)(*absolute%16ull)){*error=1;return;}
 (*pending)++;
}
extern "C" __global__ void cursor_append(const unsigned short* sq,const unsigned short* sk,const unsigned short* sv,const unsigned short* sg,const unsigned short* sb,unsigned short* q,unsigned short* k,unsigned short* v,unsigned short* g,unsigned short* beta,const int* meta,const int* error,int H,int D) {
 if(*error)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x,hd=H*D,valid=meta[0];
 if(i<16*hd){int row=i/hd,col=i%hd;if(row==valid-1){int at=col;q[i]=sq[at];k[i]=sk[at];v[i]=sv[at];g[i]=sg[at];}else if(valid==1){q[i]=k[i]=v[i]=g[i]=0x7fc1;}}
 if(i<16*H){int row=i/H,col=i%H;if(row==valid-1)beta[i]=sb[col];else if(valid==1)beta[i]=0x7fc1;}
}
extern "C" __global__ void cursor_output_reset(float* out,int guard,int HD) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<16*HD+2*guard)out[i]=__uint_as_float(0x3f123456);
}
extern "C" __global__ void cursor_emit_commit(const float* out,const float* committed,float* incoming,unsigned short* emitted,int* meta,const int* error,int H,int D,int guard) {
 if(*error)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x,HD=H*D,ST=HD*D;
 if(i<HD)emitted[i]=__bfloat16_as_ushort(__float2bfloat16_rn(out[guard+(meta[0]-1)*HD+i]));
 if(meta[0]==16&&i<ST)incoming[i]=committed[i];
}
extern "C" __global__ void cursor_finish(int* meta,const int* error,unsigned long long* absolute) {
 if(!threadIdx.x&&!blockIdx.x&&!*error){(*absolute)++;if(meta[0]==16)meta[0]=0;}
}
extern "C" __global__ void cursor_clear_completed_raw(unsigned short* q,unsigned short* k,unsigned short* v,unsigned short* g,unsigned short* beta,const int* meta,const int* error,int H,int D){
 if(*error||meta[0]!=16)return;int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<16*H*D)q[i]=k[i]=v[i]=g[i]=0x7fc1;
 if(i<16*H)beta[i]=0x7fc1;
}


// Single-thread admission precedes parallel payload readers on the same stream.
// Only canonical reset clears an existing owner error.
extern "C" __global__ void prefix_validate(
 const int* pending,int* error,const unsigned long long* absolute,unsigned long long expected,int M) {
 if(threadIdx.x||blockIdx.x||*error)return;
 if(M<=0||*pending<0||*pending>=16||*absolute!=expected||*pending!=(int)(expected%16ull)||expected>~0ull-(unsigned long long)M){*error=2;return;}
}
// Parallel chunk prepend reads the immutable admission result; it never
// writes error metadata while other payload threads are reading it.
extern "C" __global__ void prefix_prepend(
 const unsigned short* sq,const unsigned short* sk,const unsigned short* sv,const unsigned short* sg,const unsigned short* sb,
 const unsigned short* rq,const unsigned short* rk,const unsigned short* rv,const unsigned short* rg,const unsigned short* rb,
 unsigned short* q,unsigned short* k,unsigned short* v,unsigned short* g,unsigned short* beta,
 const int* pending,const int* error,const unsigned long long* absolute,unsigned long long expected,int M,int H,int D) {
 int P=expected%16ull,i=blockIdx.x*blockDim.x+threadIdx.x,HD=H*D;
 if(*error)return;
 if(i<(M+P)*HD){int row=i/HD,col=i%HD;if(row<P){q[i]=rq[i];k[i]=rk[i];v[i]=rv[i];g[i]=rg[i];}else{int at=(row-P)*HD+col;q[i]=sq[at];k[i]=sk[at];v[i]=sv[at];g[i]=sg[at];}}
 if(i<(M+P)*H){int row=i/H,col=i%H;beta[i]=row<P?rb[i]:sb[(row-P)*H+col];}
}
extern "C" __global__ void prefix_finalize(
 const unsigned short* q,const unsigned short* k,const unsigned short* v,const unsigned short* g,const unsigned short* beta,
 unsigned short* rq,unsigned short* rk,unsigned short* rv,unsigned short* rg,unsigned short* rb,
 const float* output,unsigned short* emitted,const float* final_state,float* displayed,const float* committed,float* incoming,
 int* pending,const int* error,unsigned long long* absolute,int P,int M,int H,int D) {
 if(*error)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x,HD=H*D,ST=HD*D,T=M+P,R=T%16,tail=T-R;
 if(i<M*HD)emitted[i]=__bfloat16_as_ushort(__float2bfloat16_rn(output[P*HD+i]));
 if(i<ST){displayed[i]=final_state[i];incoming[i]=committed[i];}
 if(i<16*HD){int row=i/HD;int at=tail*HD+i;if(row<R){rq[i]=q[at];rk[i]=k[at];rv[i]=v[at];rg[i]=g[at];}else{rq[i]=rk[i]=rv[i]=rg[i]=0x7fc1;}}
 if(i<16*H)rb[i]=i/H<R?beta[tail*H+i]:0x7fc1;
 // Only metadata writes depend on metadata; all other threads use immutable
 // launch dimensions, so this update does not race their arithmetic.
 if(i==0){*pending=R;*absolute+=(unsigned long long)M;}
}
extern "C" __global__ void prefix_copy_display(const float* current,float* displayed,const int* error,int elements){if(*error)return;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<elements)displayed[i]=current[i];}


extern "C" __global__ void prefix_logits_guard(
 const unsigned long long* errors, unsigned int* logits, int count) {
 for(int i=0;i<count;i++) {
  const int* error=reinterpret_cast<const int*>(errors[i]);
  if(*error) {logits[0]=0x7fc04b52u;return;}
 }
}
