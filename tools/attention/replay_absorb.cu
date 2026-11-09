// Native latent-absorption baseline versus BF16-input/FP32-output GEMM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define B(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(e));}while(0)
std::vector<char> read(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||(size_t)f.tellg()!=n)throw std::runtime_error("size "+p);std::vector<char>x(n);f.seekg(0);f.read(x.data(),n);return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,x.size()));C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
#include "absorb-baseline.cuh"
__global__ void absorb_pack(float*out,const float*head_major,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*64*512)return;
 int d=i%512,h=(i/512)%64,row=i/(512*64);
 out[i]=head_major[((long long)h*rows+row)*512+d];
}
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("replay_absorb CAPTURE START ROWS WKC OUTPUT");
 std::string prefix=std::string(argv[1])+"/fused-"+argv[2]+"-",out=argv[5];int rows=std::stoi(argv[3]);size_t n=(size_t)rows*64*512;
 auto q=upload<__nv_bfloat16>(read(prefix+"packed-q.bf16",(size_t)rows*64*256*2));auto w=upload<__nv_bfloat16>(read(argv[4],(size_t)64*256*512*2));auto ref=read(prefix+"q-absorbed.f32",n*4);
 auto output=alloc<float>(n);auto scratch=alloc<float>((size_t)1024*64*512);auto workspace=alloc<char>(16*1024*1024);
 cublasHandle_t handle;B(cublasCreate(&handle));B(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));B(cublasSetWorkspace(handle,workspace,16*1024*1024));
 cudaEvent_t a,b;C(cudaEventCreate(&a));C(cudaEventCreate(&b));
 for(int band:{0,128,256,512,1024}){
  for(int rep=0;rep<3;rep++){
   C(cudaEventRecord(a));
   if(band==0)mla_prefill_absorb_wkc_kernel<<<dim3(64,rows),256>>>(output,q,w,rows,64,256,256,512);
   else for(int start=0;start<rows;start+=band){int count=std::min(band,rows-start);const float one=1,zero=0;
    B(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,512,count,256,&one,w,CUDA_R_16BF,512,(long long)256*512,q+(size_t)start*64*256,CUDA_R_16BF,64*256,256,&zero,scratch,CUDA_R_32F,512,(long long)count*512,64,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    absorb_pack<<<(count*64*512+255)/256,256>>>(output+(size_t)start*64*512,scratch,count);
   }
   C(cudaEventRecord(b));C(cudaEventSynchronize(b));float ms;C(cudaEventElapsedTime(&ms,a,b));printf("{\"band\":%d,\"rep\":%d,\"ms\":%.6f}\n",band,rep,ms);fflush(stdout);
  }
  std::vector<float>x(n);C(cudaMemcpy(x.data(),output,n*4,cudaMemcpyDeviceToHost));double maxerr=0,ss=0;size_t bad=0,nonfinite=0,first=n;
  for(size_t i=0;i<n;i++){float v;memcpy(&v,ref.data()+i*4,4);double e=std::abs((double)x[i]-v);if(!std::isfinite(x[i])||!std::isfinite(v))nonfinite++;maxerr=std::max(maxerr,e);ss+=e*e;if(e>2e-4){bad++;if(first==n)first=i;}}
  printf("{\"band\":%d,\"values\":%zu,\"max_abs\":%.9g,\"rms\":%.9g,\"over_2e_4\":%zu,\"nonfinite\":%zu,\"first_bad_index\":%zu}\n",band,n,maxerr,std::sqrt(ss/n),bad,nonfinite,first);fflush(stdout);
  if(bad||nonfinite){std::ofstream f(out+"/failed-band"+std::to_string(band)+".f32",std::ios::binary);f.write((char*)x.data(),n*4);return 2;}
 }
 return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
