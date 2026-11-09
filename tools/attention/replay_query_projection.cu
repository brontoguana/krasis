// Bounded BF16 query projection replay on captured native rows.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define B(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS "+std::to_string(e));}while(0)
std::vector<__nv_bfloat16> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);size_t n=f.tellg();std::vector<__nv_bfloat16>x(n/2);f.seekg(0);f.read((char*)x.data(),n);if(!f)throw std::runtime_error("read "+p);return x;}
int main(int argc,char**argv){try{
 if(argc<4)throw std::runtime_error("replay_query_projection CAPTURE ROWS OUT_PREFIX [--diagnostics] [--force-tiny] [--repeat-row]");
 bool diagnostic=false,force_tiny=false,repeat_row=false;
 for(int i=4;i<argc;i++){std::string flag=argv[i];if(flag=="--diagnostics")diagnostic=true;else if(flag=="--force-tiny")force_tiny=true;else if(flag=="--repeat-row")repeat_row=true;else throw std::runtime_error("unknown flag");}
 std::string d=argv[1],o=argv[3];int m=std::stoi(argv[2]),n=4096,k=1536;if(m<1||m>32768)throw std::runtime_error("rows");
 auto w=read(d+"/indexer-weight-wq_b.bf16"),row=read(d+"/indexer-lora.bf16");if(w.size()!=size_t(n)*k||row.size()!=k)throw std::runtime_error("shape");
 auto bank=read(d+"/q_norm-chunk-prefix.bf16"); if(bank.size()<size_t(m)*k && !repeat_row)throw std::runtime_error("real bank too short"); std::vector<__nv_bfloat16>x(size_t((m+63)/64)*64*k);if(repeat_row){for(int r=0;r<m;r++)std::copy(row.begin(),row.end(),x.begin()+size_t(r)*k);}else std::copy(bank.begin(),bank.begin()+size_t(m)*k,x.begin());
 __nv_bfloat16 *dw,*dx,*dy;C(cudaMalloc(&dw,w.size()*2));C(cudaMalloc(&dx,x.size()*2));C(cudaMalloc(&dy,size_t((m+63)/64)*64*n*2));C(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));C(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
 cublasHandle_t h;B(cublasCreate(&h));cudaStream_t s;C(cudaStreamCreate(&s));B(cublasSetStream(h,s));B(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));float a=1,b=0;
 std::vector<__nv_bfloat16> baseline(size_t(m)*n),out(baseline.size());
 for(int mode:(diagnostic?std::vector<int>{0,1,2,3}:std::vector<int>{0,1})){
 auto launch=[&](){if(mode==0 || (m<64 && !force_tiny))B(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,1,k,&a,dw,CUDA_R_16BF,k,0,dx,CUDA_R_16BF,k,k,&b,dy,CUDA_R_16BF,n,n,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
 else if(mode==3 && m%64==0) {B(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,64,k,&a,dw,CUDA_R_16BF,k,0,dx,CUDA_R_16BF,k,64*k,&b,dy,CUDA_R_16BF,n,64*n,m/64,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));}
 else {int tile=(mode==1||mode==3)?64:m;for(int r=0;r<m;r+=tile){int count=mode==1?64:std::min(tile,m-r);B(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,count,k,&a,dw,CUDA_R_16BF,k,dx+size_t(r)*k,CUDA_R_16BF,k,&b,dy+size_t(r)*n,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));}}};
 launch();C(cudaStreamSynchronize(s));
 for(int rep=0;rep<5;rep++){cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));C(cudaEventRecord(begin,s));launch();C(cudaEventRecord(end,s));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));printf("{\"mode\":%d,\"rows\":%d,\"rep\":%d,\"ms\":%.8f}\n",mode,m,rep,ms);fflush(stdout);C(cudaEventDestroy(begin));C(cudaEventDestroy(end));}
 cudaGraph_t graph;cudaGraphExec_t exec;C(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));launch();C(cudaStreamEndCapture(s,&graph));C(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));C(cudaGraphLaunch(exec,s));C(cudaStreamSynchronize(s));std::vector<__nv_bfloat16> graphout(out.size());C(cudaMemcpy(graphout.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(graph));
 launch();C(cudaStreamSynchronize(s));
 C(cudaMemcpy(out.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));if(memcmp(out.data(),graphout.data(),out.size()*2)!=0)throw std::runtime_error("graph mismatch");
if(mode==0)baseline=out;size_t neq=0;float maxdiff=0;for(size_t i=0;i<out.size();i++){if(!std::isfinite(__bfloat162float(out[i])))throw std::runtime_error("nonfinite projection output");neq+=memcmp(&out[i],&baseline[i],2)!=0;maxdiff=std::max(maxdiff,fabsf(__bfloat162float(out[i])-__bfloat162float(baseline[i])));}
 printf("{\"mode\":%d,\"values\":%zu,\"different\":%zu,\"maxdiff\":%.9g}\n",mode,out.size(),neq,maxdiff);fflush(stdout);
 std::ofstream f(o+"-mode"+std::to_string(mode)+".bf16",std::ios::binary);f.write((char*)out.data(),size_t(m)*n*2);
 if(mode==1 && neq && !force_tiny)throw std::runtime_error("complete projection mismatch");
 }
 B(cublasDestroy(h));C(cudaStreamDestroy(s));C(cudaFree(dw));C(cudaFree(dx));C(cudaFree(dy));return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
