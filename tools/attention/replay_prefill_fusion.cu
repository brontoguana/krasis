// Paired replay of the production gathered pipeline and bounded tensor path.
#include "gather_baseline.cuh"
__device__ inline float mla_load_k6_value(const unsigned char* p,int pos,int dim,int d){return mla_prefill_load_k6_value(p,pos,dim,d);}
#include "prefill_mask_fusion.cuh"
#include <cublas_v2.h>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define B(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS "+std::to_string(e));}while(0)
std::vector<char> read(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||(size_t)f.tellg()!=n)throw std::runtime_error("size "+p);std::vector<char>x(n);f.seekg(0);f.read(x.data(),n);return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,x.size()));C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("replay CAPTURE START ROWS CONTEXT CAPACITY OUT");
 const std::string prefix=std::string(argv[1])+"/fused-"+argv[2]+"-",out=argv[6];
 const int rows=std::stoi(argv[3]),context=std::stoi(argv[4]),cap=std::stoi(argv[5]),padded=cap,max_tile=32;
 const size_t elements=(size_t)rows*64*512;
 auto q=upload<float>(read(prefix+"q-absorbed.f32",elements*4));auto ids=upload<int>(read(prefix+"selected.i32",(size_t)rows*cap*4));auto positions=upload<int>(read(prefix+"positions.i32",rows*4));auto cache=upload<unsigned char>(read(prefix+"ckv-cache.bin",(size_t)context*448));
 auto reference=read(prefix+"baseline-output.f32",elements*4);auto output=alloc<float>(elements);
 auto dense=alloc<float>((size_t)max_tile*padded*512),query=alloc<float>((size_t)max_tile*64*512),scores=alloc<float>((size_t)max_tile*64*padded),probs=alloc<float>((size_t)max_tile*64*padded);
 auto workspace=alloc<char>(16<<20);cublasHandle_t handle;B(cublasCreate(&handle));B(cublasSetStream(handle,0));B(cublasSetWorkspace(handle,workspace,16<<20));
 cudaEvent_t events[5];for(auto&event:events)C(cudaEventCreate(&event));
 for(int mode:{0,1,1,0}){
  const int tile=32;double stage_ms[4]={};
  for(int start=0;start<rows;start+=tile){int n=std::min(tile,rows-start);C(cudaEventRecord(events[0]));
   if(!mode){
    int blocks=n*((cap+7)/8+(64+7)/8+1);
    mla_prefill_gather_query_k6_f32_structured_kernel<<<blocks,256>>>(dense,query,scores,q,(const __nv_bfloat16*)q,cache,nullptr,ids,positions,start,n,64,256,256,0,512,cap,context);
   }else mla_prefill_gather_k6_no_mask<<<dim3((std::max(cap,64)+7)/8,n),256>>>(dense,query,q+(size_t)start*64*512,cache,ids+(size_t)start*cap,positions+start,cap,context);
   C(cudaEventRecord(events[1]));
   const float scale=.0625f,one=1,zero=0;
   {
    B(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,cap,64,512,&scale,dense,CUDA_R_32F,512,(long long)cap*512,query,CUDA_R_32F,512,64*512,mode?&zero:&one,scores,CUDA_R_32F,cap,(long long)64*cap,n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
   }
   C(cudaEventRecord(events[2]));
   if(!mode)mla_prefill_softmax_weights_warp_f32_kernel<<<(n*64+7)/8,256>>>(probs,scores,n,64,cap);
   else mla_prefill_masked_softmax_f32<<<(n*64+7)/8,256>>>(probs,scores,ids+(size_t)start*cap,positions+start,n,cap,context);
   C(cudaEventRecord(events[3]));
   {
    B(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,512,64,cap,&one,dense,CUDA_R_32F,512,(long long)cap*512,probs,CUDA_R_32F,cap,(long long)64*cap,&zero,output+(size_t)start*64*512,CUDA_R_32F,512,64*512,n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
   }
   C(cudaEventRecord(events[4]));C(cudaEventSynchronize(events[4]));
   for(int j=0;j<4;j++){float ms;C(cudaEventElapsedTime(&ms,events[j],events[j+1]));stage_ms[j]+=ms;}
  }
  std::vector<float> actual(elements);C(cudaMemcpy(actual.data(),output,elements*4,cudaMemcpyDeviceToHost));
  size_t bad=0,nonfinite=0,bits=0;double maxerr=0,ss=0;const float* ref=(const float*)reference.data();
  for(size_t i=0;i<elements;i++){if(memcmp(&actual[i],&ref[i],4))bits++;double e=std::abs((double)actual[i]-ref[i]);if(!std::isfinite(actual[i])||!std::isfinite(ref[i]))nonfinite++;if(e>2e-4)bad++;maxerr=std::max(maxerr,e);ss+=e*e;}
  printf("{\"mode\":%d,\"tile\":%d,\"stage_ms\":[%.6f,%.6f,%.6f,%.6f],\"values\":%zu,\"over_2e_4\":%zu,\"nonfinite\":%zu,\"max_abs\":%.9g,\"rms\":%.9g}\n",mode,tile,stage_ms[0],stage_ms[1],stage_ms[2],stage_ms[3],elements,bad,nonfinite,maxerr,std::sqrt(ss/elements));fflush(stdout);
  printf("{\"mode\":%d,\"bit_differences\":%zu}\n",mode,bits);fflush(stdout);
  // Retain a complete real row for independent float64 reconstruction.
  std::ofstream f(out+"/pipeline-mode"+std::to_string(mode)+"-last-row.f32",std::ios::binary);f.write((char*)(actual.data()+(rows-1)*64*512),64*512*4);
  if(bad||nonfinite)return 2;
 }
 B(cublasDestroy(handle));return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
