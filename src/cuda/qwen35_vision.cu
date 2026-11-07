// Qwen3.5 vision equations: BF16 projection/residual boundaries, FP32 norms,
// two-dimensional rotate-half RoPE and noncausal per-image attention.
#include <cuda_bf16.h>
#include <math.h>
extern "C" __global__ void qv_finish(const float* src, const __nv_bfloat16* bias, __nv_bfloat16* out, int total, int width, int act) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
 // Conv3d adds bias after its BF16 convolution output; Linear uses
 // addmm's FP32 bias accumulation before the BF16 output boundary.
 float input=act==3?__bfloat162float(__float2bfloat16(src[i])):src[i];
 float v=__bfloat162float(__float2bfloat16(input+(bias?__bfloat162float(bias[i%width]):0.f)));
 if(act==1)v=.5f*v*(1.f+tanhf(.7978845608028654f*(v+.044715f*v*v*v)));
 if(act==2)v=.5f*v*(1.f+erff(v*.7071067811865475f));
 out[i]=__float2bfloat16(v);
}
// Welford reduction scheduling adapted from PyTorch's CUDA LayerNorm.
// See PYTORCH_VISION_NOTICE for copyright and BSD redistribution terms.
struct QvStats {float mean,variance,count;};
__device__ QvStats qv_combine(QvStats left,QvStats right){
 float count=right.count+left.count;if(count==0)return {0,0,0};
 float delta=left.mean-right.mean,coef=1.f/count,a=right.count*coef,b=left.count*coef;
 return {a*right.mean+b*left.mean,right.variance+left.variance+delta*delta*right.count*b,count};
}
extern "C" __global__ void qv_norm(const __nv_bfloat16* x,const __nv_bfloat16* w,const __nv_bfloat16* b,__nv_bfloat16* y,int width){
 __shared__ float means[4],vars[4],counts[4];int row=blockIdx.x,t=threadIdx.x,lane=t%32,warp=t/32;
 QvStats value={0,0,0};
 for(int base=t*4;base<width;base+=128*4){
  #pragma unroll
  for(int j=0;j<4;j++){float v=__bfloat162float(x[row*width+base+j]),delta=v-value.mean,count=value.count+1.f;float mean=value.mean+delta*(1.f/count);value={mean,value.variance+delta*(v-mean),count};}
 }
 for(int shift=16;shift;shift/=2){QvStats other={__shfl_down_sync(0xffffffff,value.mean,shift),__shfl_down_sync(0xffffffff,value.variance,shift),__shfl_down_sync(0xffffffff,value.count,shift)};value=qv_combine(value,other);}
 for(int shift=2;shift;shift/=2){
  if(lane==0&&warp>=shift&&warp<2*shift){means[warp-shift]=value.mean;vars[warp-shift]=value.variance;counts[warp-shift]=value.count;}__syncthreads();
  if(lane==0&&warp<shift)value=qv_combine(value,{means[warp],vars[warp],counts[warp]});__syncthreads();
 }
 if(t==0){means[0]=value.mean;vars[0]=value.variance/width;}__syncthreads();
 float mean=means[0],inv=rsqrtf(vars[0]+1e-6f);
 for(int c=t;c<width;c+=128)y[row*width+c]=__float2bfloat16(__bfloat162float(w[c])*(inv*(__bfloat162float(x[row*width+c])-mean))+__bfloat162float(b[c]));
}
extern "C" __global__ void qv_add(__nv_bfloat16* x,const __nv_bfloat16* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__float2bfloat16(__bfloat162float(x[i])+__bfloat162float(y[i]));}
extern "C" __global__ void qv_position(__nv_bfloat16* x,const __nv_bfloat16* pos,int n,int hidden,int gh,int gw,int side){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*hidden)return;int p=i/hidden,c=i%hidden;
 int group=p/4,sub=p%4;int row=(group/(gw/2))*2+sub/2,col=(group%(gw/2))*2+sub%2;
 float h=gh>1?float(row)*(side-1)/(gh-1):0,w=gw>1?float(col)*(side-1)/(gw-1):0;
 int h0=int(h),w0=int(w),h1=min(h0+1,side-1),w1=min(w0+1,side-1);float a=h-h0,b=w-w0;
 // Match the published BF16 interpolation boundaries: cast weights, each
 // gathered product and each left-associated sum before adding positions.
 float f0=__bfloat162float(__float2bfloat16((1-a)*(1-b)));
 float f1=__bfloat162float(__float2bfloat16((1-a)*b));
 float f2=__bfloat162float(__float2bfloat16(a*(1-b)));
 float f3=__bfloat162float(__float2bfloat16(a*b));
 float t0=__bfloat162float(__float2bfloat16(__bfloat162float(pos[(h0*side+w0)*hidden+c])*f0));
 float t1=__bfloat162float(__float2bfloat16(__bfloat162float(pos[(h0*side+w1)*hidden+c])*f1));
 float t2=__bfloat162float(__float2bfloat16(__bfloat162float(pos[(h1*side+w0)*hidden+c])*f2));
 float t3=__bfloat162float(__float2bfloat16(__bfloat162float(pos[(h1*side+w1)*hidden+c])*f3));
 float v=__bfloat162float(__float2bfloat16(t0+t1));
 v=__bfloat162float(__float2bfloat16(v+t2));v=__bfloat162float(__float2bfloat16(v+t3));
 x[i]=__float2bfloat16(__bfloat162float(x[i])+v);
}
extern "C" __global__ void qv_rope(__nv_bfloat16* qkv,int n,int heads,int dim,int gw){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*2*heads*(dim/2))return;
 int d=i%(dim/2),h=i/(dim/2)%heads,which=i/(dim/2)/heads%2,p=i/(dim/2)/heads/2;
 int group=p/4,sub=p%4,row=(group/(gw/2))*2+sub/2,col=(group%(gw/2))*2+sub%2;
 int coord=d<dim/4?row:col, f=d%(dim/4);float angle=coord*(1.f/powf(10000.f,float(f)/(dim/4)));
 int j=((p*3+which)*heads+h)*dim+d;float a=__bfloat162float(qkv[j]),b=__bfloat162float(qkv[j+dim/2]);float s,c;sincosf(angle,&s,&c);
 qkv[j]=__float2bfloat16(__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s)));qkv[j+dim/2]=__float2bfloat16(__fadd_rn(__fmul_rn(b,c),__fmul_rn(a,s)));
}
extern "C" __global__ void qv_attention(const __nv_bfloat16* qkv,__nv_bfloat16* out,int n,int heads,int dim){
 // One warp owns a query/head; no cross-query scratch or N^2 buffer.
 int lane=threadIdx.x,head=blockIdx.x%heads,q=blockIdx.x/heads;float acc[4]={0,0,0,0},query[4];
 for(int j=0;j<4;j++){int d=lane+32*j;query[j]=d<dim?__bfloat162float(qkv[((q*3)*heads+head)*dim+d]):0;}
 float m=-INFINITY,z=0;
 for(int k=0;k<n;k++){
  float dot=0;for(int j=0;j<4;j++){int d=lane+32*j;if(d<dim)dot+=query[j]*__bfloat162float(qkv[((k*3+1)*heads+head)*dim+d]);}
  for(int s=16;s;s/=2)dot+=__shfl_down_sync(0xffffffff,dot,s);dot=__shfl_sync(0xffffffff,dot,0)*rsqrtf(float(dim));
  float nm=fmaxf(m,dot),old=expf(m-nm),now=expf(dot-nm);z=z*old+now;m=nm;
  for(int j=0;j<4;j++){int d=lane+32*j;if(d<dim)acc[j]=acc[j]*old+now*__bfloat162float(qkv[((k*3+2)*heads+head)*dim+d]);}
 }
 for(int j=0;j<4;j++){int d=lane+32*j;if(d<dim)out[(q*heads+head)*dim+d]=__float2bfloat16(acc[j]/z);}
}

extern "C" __global__ void qv_split_pad(const __nv_bfloat16* src,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,int n,int heads,int dim,int padded){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*heads*padded)return;int d=i%padded,h=i/padded%heads,t=i/padded/heads;
 q[i]=d<dim?src[((t*3)*heads+h)*dim+d]:__float2bfloat16(0.f);
 k[i]=d<dim?src[((t*3+1)*heads+h)*dim+d]:__float2bfloat16(0.f);
 v[i]=d<dim?src[((t*3+2)*heads+h)*dim+d]:__float2bfloat16(0.f);
}
extern "C" __global__ void qv_unpad(const __nv_bfloat16* src,__nv_bfloat16* out,int total,int dim,int padded){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)out[i]=src[i/dim*padded+i%dim];}
