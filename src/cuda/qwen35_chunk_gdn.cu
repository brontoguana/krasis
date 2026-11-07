// Bounded FP32 chunked GDN contract; Q/K already BF16-normalized.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
namespace {
constexpr int B=64,D=128;
__global__ void prep(const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,const __nv_bfloat16*beta,const __nv_bfloat16*a,const float*al,const float*dt,float*Q,float*K,float*V,float*KB,float*VB,float*G,int start,int rows,int H){
 int h=blockIdx.x,t=threadIdx.x; __shared__ float g[B];
 if(t<B){float x=t+start<rows?float(a[(t+start)*H+h])+dt[h]:0.f;g[t]=t+start<rows?-expf(al[h])*(x>20.f?x:log1pf(expf(x))):0.f;}
 __syncthreads();
 // Torch sizes its scan from all head/chunk rows. Its minimum 16 x
 // threads scan two 32-value bands with the previous total folded into
 // the second band's first value (different FP32 order from a 64 scan).
 int logrows=0;while((1<<logrows)<H*((rows+B-1)/B))++logrows;
 int logx=(9+6-logrows)/2;if(logx<4)logx=4;
 int width=logx==4?32:64;
 for(int base=0;base<B;base+=width){
  if(t==0 && base)g[base]+=g[base-1];__syncthreads();
  for(int ss=1;ss<=width/2;ss*=2){if(t<width/2){int at=(t/ss)*(2*ss)+ss+t%ss;g[base+at]+=g[base+(t/ss)*(2*ss)+ss-1];}__syncthreads();}
 }
 if(t<B)G[h*B+t]=g[t];
 for(int i=t;i<B*D;i+=blockDim.x){int r=i/D,c=i%D,src=((start+r)*H+h)*D+c,ix=h*B*D+i;float be=start+r<rows?float(beta[(start+r)*H+h]):0.f;float kv=start+r<rows?float(k[src]):0.f;float vv=start+r<rows?float(v[src]):0.f;Q[ix]=start+r<rows?float(q[src])*float(1.0/std::sqrt(128.0)):0.f;K[ix]=kv;V[ix]=vv;KB[ix]=kv*be;VB[ix]=vv*be;}
}
__global__ void transform(float*A,const float*G){
 int h=blockIdx.x,j=threadIdx.x;__shared__ float a[B*B];
 for(int ix=j;ix<B*B;ix+=64){int r=ix/B,c=ix%B;a[ix]=r>c?-A[h*B*B+ix]*expf(G[h*B+r]-G[h*B+c]):0.f;}
 __syncthreads();
 for(int r=1;r<B;r++){
  float z=0.f;
  if(j<r){float acc[4]={0.f,0.f,0.f,0.f};for(int i=0;i<r;i++)acc[i%4]+=a[r*B+i]*a[i*B+j];z=((acc[0]+acc[1])+acc[2])+acc[3];z+=a[r*B+j];}
  __syncthreads();if(j<r)a[r*B+j]=z;__syncthreads();
 }
 for(int ix=j;ix<B*B;ix+=64)A[h*B*B+ix]=a[ix]+(ix/B==ix%B?1.f:0.f);
}
__global__ void decay(const float*K,const float*KB,const float*Q,const float*G,float*KD,float*QE,float*KT){
 int ix=blockIdx.x*blockDim.x+threadIdx.x,H=gridDim.y,h=blockIdx.y;if(ix>=B*D)return;int r=ix/D;int at=h*B*D+ix;float g=G[h*B+r];KD[at]=KB[at]*expf(g);QE[at]=Q[at]*expf(g);KT[at]=K[at]*expf(G[h*B+B-1]-g);
}
__global__ void mask(float*A,const float*G){int ix=blockIdx.x*blockDim.x+threadIdx.x,h=blockIdx.y;if(ix>=B*B)return;int r=ix/B,c=ix%B;A[h*B*B+ix]=r>=c?A[h*B*B+ix]*expf(G[h*B+r]-G[h*B+c]):0.f;}
__global__ void subtract(float*A,const float*V){int ix=blockIdx.x*blockDim.x+threadIdx.x,h=blockIdx.y;if(ix<B*D)A[h*B*D+ix]=V[h*B*D+ix]-A[h*B*D+ix];}
__global__ void finish(const float*X,const float*Y,const float*S,const float*G,float*state,__nv_bfloat16*out,int start,int rows,int H){int ix=blockIdx.x*blockDim.x+threadIdx.x,h=blockIdx.y;
 if(ix<B*D){int t=ix/D,c=ix%D;if(start+t<rows)out[((start+t)*H+h)*D+c]=__float2bfloat16(X[h*B*D+ix]+Y[h*B*D+ix]);}
 if(ix<D*D)state[h*D*D+ix]=state[h*D*D+ix]*expf(G[h*B+B-1])+S[h*D*D+ix];
}
int gemm(cublasHandle_t handle,const float*A,const float*BB,float*C,int m,int n,int k,int H,bool ta=false,bool tb=false){float one=1.f,zero=0.f;return int(cublasSgemmStridedBatched(handle,tb?CUBLAS_OP_T:CUBLAS_OP_N,ta?CUBLAS_OP_T:CUBLAS_OP_N,n,m,k,&one,BB,tb?k:n,(long long)k*n,A,ta?m:k,(long long)m*k,&zero,C,n,(long long)m*n,H));}
}
// Workspace: heads * (12*64*128 + 2*64*64 + 64 + 128*128) floats.
extern "C" int krasis_qwen_chunk_gdn(const void*q,const void*k,const void*v,const void*beta,const void*a,const float*al,const float*dt,float*state,void*out,int rows,int H,float*ws,void*blas,void*stream){
 if(rows<1||rows>4096||H<1||H>64)return -1;
 auto st=(cudaStream_t)stream;auto handle=(cublasHandle_t)blas;
 struct Restore {cublasHandle_t h;cudaStream_t s;cublasMath_t m;~Restore(){cublasSetStream(h,s);cublasSetMathMode(h,m);}};
 cudaStream_t old_stream;cublasMath_t old_math;
 if(cublasGetStream(handle,&old_stream)!=CUBLAS_STATUS_SUCCESS||cublasGetMathMode(handle,&old_math)!=CUBLAS_STATUS_SUCCESS)return -2;
 Restore restore{handle,old_stream,old_math};
 auto status=cublasSetStream(handle,st);if(status)return int(status);
 status=cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);if(status)return int(status);
 float *Q=ws,*K=Q+H*B*D,*V=K+H*B*D,*KB=V+H*B*D,*VB=KB+H*B*D,*KD=VB+H*B*D,*QE=KD+H*B*D,*KT=QE+H*B*D,*VV=KT+H*B*D,*KC=VV+H*B*D,*VN=KC+H*B*D,*Y=VN+H*B*D,*A=Y+H*B*D,*AT=A+H*B*B,*G=AT+H*B*B,*S=G+H*B;
 #define MM(a,b,c,m,n,k,ta,tb) do{int err=gemm(handle,a,b,c,m,n,k,H,ta,tb);if(err)return err;}while(0)
 for(int start=0;start<rows;start+=B){
 prep<<<H,128,0,st>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(const __nv_bfloat16*)beta,(const __nv_bfloat16*)a,al,dt,Q,K,V,KB,VB,G,start,rows,H);
 MM(KB,K,A,B,B,D,false,true);transform<<<H,64,0,st>>>(A,G);
 decay<<<dim3((B*D+255)/256,H),256,0,st>>>(K,KB,Q,G,KD,QE,KT);
 MM(A,VB,VV,B,D,B,false,false);MM(A,KD,KC,B,D,B,false,false);
 MM(Q,K,AT,B,B,D,false,true);mask<<<dim3(16,H),256,0,st>>>(AT,G);
 MM(KC,state,VN,B,D,D,false,false);subtract<<<dim3(32,H),256,0,st>>>(VN,VV);
 MM(QE,state,Y,B,D,D,false,false);MM(AT,VN,VV,B,D,B,false,false);
 MM(KT,VN,S,D,D,B,true,false);
 finish<<<dim3(64,H),256,0,st>>>(Y,VV,S,G,state,(__nv_bfloat16*)out,start,rows,H);
 }
 return int(cudaGetLastError());
}
namespace {
__global__ void norm128(__nv_bfloat16*out,const __nv_bfloat16*x,const __nv_bfloat16*z,const float*w,int H,float eps){
 int h=blockIdx.y,t=blockIdx.x,lane=threadIdx.x;int base=(t*H+h)*128;
 float sums[4];
 #pragma unroll
 for(int j=0;j<4;j++){float a=float(x[base+lane*4+j]);sums[j]=a*a;}
 float total=((sums[0]+sums[1])+sums[2])+sums[3];
 for(int off=16;off;off/=2)total+=__shfl_down_sync(0xffffffff,total,off);
 total=__shfl_sync(0xffffffff,total,0);float inv=rsqrtf(total*(1.f/128.f)+eps);
 #pragma unroll
 for(int j=0;j<4;j++){int c=lane*4+j;float a=float(__float2bfloat16(float(x[base+c])*inv));a=float(__float2bfloat16(a*float(__float2bfloat16(w[c]))));float g=float(z[base+c]);out[base+c]=__float2bfloat16(a*(g/(1.f+expf(-g))));}
}
}
extern "C" int krasis_qwen_norm128(void*out,const void*x,const void*z,const float*w,int rows,int heads,float eps,void*stream){
 if(rows<1||rows>4096||heads<1||heads>64)return -1;
 norm128<<<dim3(rows,heads),32,0,(cudaStream_t)stream>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)x,(const __nv_bfloat16*)z,w,heads,eps);
 return int(cudaGetLastError());
}
namespace {
__global__ void outer_norm(__nv_bfloat16*res,__nv_bfloat16*out,const __nv_bfloat16*x,const float*w,int dim,float eps){
 int row=blockIdx.x,lane=threadIdx.x;int base=row*dim;float acc[4]={0.f,0.f,0.f,0.f};
 for(int group=lane*4;group<dim;group+=blockDim.x*4){
  #pragma unroll
  for(int j=0;j<4;j++){int c=group+j;if(c<dim){float v=float(x[base+c]);if(res){v=float(__float2bfloat16(v+float(res[base+c])));res[base+c]=__float2bfloat16(v);}acc[j]+=v*v;}}
 }
 float total=((acc[0]+acc[1])+acc[2])+acc[3];extern __shared__ float shared[];shared[lane]=total;__syncthreads();
 for(int off=blockDim.x/2;off>=32;off/=2){if(lane<off)shared[lane]+=shared[lane+off];__syncthreads();}
 total=shared[lane];for(int off=16;off;off/=2)total+=__shfl_down_sync(0xffffffff,total,off);
 if(lane==0)shared[0]=total;__syncthreads();float inv=rsqrtf(shared[0]*(1.f/dim)+eps);
 for(int c=lane;c<dim;c+=blockDim.x){float v=float(res?res[base+c]:x[base+c]);out[base+c]=__float2bfloat16((v*inv)*w[c]);}
}
}
extern "C" int krasis_qwen_outer_norm(void*res,void*out,const void*x,const float*w,int rows,int dim,float eps,void*stream){
 if(rows<1||rows>1048576||dim<128||dim>8192||dim%4)return -1;
 // Frozen torch Mean uses vectorized input and at most 512 threads;
 // map its per-row block-x width without retaining 16 rows in one block.
 int output_pow=1;while(output_pow*2<=rows)output_pow*=2;
 int height=output_pow<16?output_pow:16;int width=512/height;
 int power=1;while(power*2<=dim/4)power*=2;if(width>power)width=power;
 outer_norm<<<rows,width,width*sizeof(float),(cudaStream_t)stream>>>((__nv_bfloat16*)res,(__nv_bfloat16*)out,(const __nv_bfloat16*)x,w,dim,eps);
 return int(cudaGetLastError());
}
namespace {
__global__ void text_softmax_warp(__nv_bfloat16*score,int queries,int keys,int first,float scale){
 int row=blockIdx.x%queries,lane=threadIdx.x;score+=(size_t)blockIdx.x*keys;
 float values[32],mx=-INFINITY;
 int count=(keys+31)/32;
 for(int i=0;i<count;i++){int j=i*32+lane;float v=j<keys && j<=first+row?float(__float2bfloat16(float(score[j])*scale)):-INFINITY;values[i]=v;mx=fmaxf(mx,v);}
 for(int off=16;off;off/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off));
 float sum=0.f;for(int i=0;i<count;i++){values[i]=expf(values[i]-mx);sum+=values[i];}
 for(int off=16;off;off/=2)sum+=__shfl_xor_sync(0xffffffff,sum,off);
 for(int i=0;i<count;i++){int j=i*32+lane;if(j<keys)score[j]=__float2bfloat16(values[i]/sum);}
}
__global__ void text_softmax_block(__nv_bfloat16*score,int queries,int keys,int first,float scale){
 int row=blockIdx.x%queries;score+=(size_t)blockIdx.x*keys;__shared__ float shared[128];float mx=-INFINITY;
 for(int j=threadIdx.x;j<keys;j+=128){float v=j<=first+row?float(__float2bfloat16(float(score[j])*scale)):-INFINITY;mx=fmaxf(mx,v);}shared[threadIdx.x]=mx;__syncthreads();
 for(int off=64;off;off/=2){if(threadIdx.x<off)shared[threadIdx.x]=fmaxf(shared[threadIdx.x],shared[threadIdx.x+off]);__syncthreads();}mx=shared[0];float sum=0.f;
 for(int j=threadIdx.x;j<keys;j+=128){float v=j<=first+row?float(__float2bfloat16(float(score[j])*scale)):-INFINITY;sum+=expf(v-mx);}shared[threadIdx.x]=sum;__syncthreads();
 for(int off=64;off;off/=2){if(threadIdx.x<off)shared[threadIdx.x]+=shared[threadIdx.x+off];__syncthreads();}sum=shared[0];
 for(int j=threadIdx.x;j<keys;j+=128){float v=j<=first+row?float(__float2bfloat16(float(score[j])*scale)):-INFINITY;score[j]=__float2bfloat16(expf(v-mx)/sum);}
}
}
extern "C" int krasis_qwen_text_softmax(void*score,int queries,int keys,int first,int heads,float scale,void*stream){
 if(queries<1||queries>256||keys<1||keys>4096||heads<1||heads>256||first<0||first+queries>keys)return -1;
 if(keys<=1024)text_softmax_warp<<<queries*heads,32,0,(cudaStream_t)stream>>>((__nv_bfloat16*)score,queries,keys,first,scale);
 else text_softmax_block<<<queries*heads,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)score,queries,keys,first,scale);
 return int(cudaGetLastError());
}
