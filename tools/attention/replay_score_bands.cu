// Offline physical row bands; all serving kernels are extracted unchanged.
#include "score_kernels.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define B(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas "+std::to_string(e));}while(0)
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>std::vector<T> read(std::string p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);size_t n=f.tellg();if(n%sizeof(T))throw std::runtime_error("file size");std::vector<T> v(n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);return v;}
template<class T>T* upload(const std::vector<T>&v){T*p=alloc<T>(v.size());C(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T>void save(std::string p,T*d,size_t n){std::vector<T>v(n);C(cudaMemcpy(v.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));std::ofstream f(p,std::ios::binary);f.write((char*)v.data(),n*sizeof(T));}
__global__ void tie_rows(__nv_bfloat16*q){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<4096){q[5*4096+i]=__float2bfloat16(0);q[6*4096+i]=__float2bfloat16(__bfloat162float(q[6*4096+i])*0x1p-16f);}}
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("replay CAPTURE OUTPUT CHUNK_ROWS ROWS CONTEXT POSITION_START FIXTURE");
 std::string d=argv[1],o=argv[2];int chunk=atoi(argv[3]),m=atoi(argv[4]),n=atoi(argv[5]),pos0=atoi(argv[6]),fixture=atoi(argv[7]);
 if(m<1||m>chunk||n<1||n>16387)throw std::runtime_error("geometry");
 if((fixture!=0&&fixture!=1)||(fixture==1&&m<7))throw std::runtime_error("tie fixture requires at least seven rows");
 cudaStream_t s;C(cudaStreamCreate(&s));cublasHandle_t h;B(cublasCreate(&h));B(cublasSetStream(h,s));B(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));auto workspace=alloc<char>(16<<20);B(cublasSetWorkspace(h,workspace,16<<20));float one=1,zero=0,scale=0.01562500186264515f;
 auto bank=read<__nv_bfloat16>(d+"/q_norm-chunk-prefix.bf16");auto hidden=read<__nv_bfloat16>(d+"/input-chunk-prefix.bf16");auto qw=read<__nv_bfloat16>(d+"/indexer-weight-wq_b.bf16");auto hw=read<__nv_bfloat16>(d+"/indexer-weight-weights_proj.bf16");auto keys=read<__nv_bfloat16>(d+"/indexer-pool-key-prefix.bf16");
 if(bank.size()!=size_t(chunk)*1536||hidden.size()!=size_t(chunk)*4096||keys.size()<size_t(n)*128||qw.size()!=4096*1536||hw.size()!=32*4096)throw std::runtime_error("bank extent");
 int padded=(chunk+63)/64*64;bank.resize(size_t(padded)*1536);auto db=upload(bank),dh=upload(hidden),dqw=upload(qw),dhw=upload(hw);auto qfull=alloc<__nv_bfloat16>(size_t(padded)*4096),wfull=alloc<__nv_bfloat16>(size_t(chunk)*32),k=upload(keys);
 if(chunk<64) B(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,4096,1,1536,&one,dqw,CUDA_R_16BF,1536,0,db,CUDA_R_16BF,1536,1536,&zero,qfull,CUDA_R_16BF,4096,4096,chunk,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
 else for(int r=0;r<chunk;r+=64)B(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,4096,64,1536,&one,dqw,CUDA_R_16BF,1536,db+size_t(r)*1536,CUDA_R_16BF,1536,&zero,qfull+size_t(r)*4096,CUDA_R_16BF,4096,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
 B(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,32,1,4096,&one,dhw,CUDA_R_16BF,4096,0,dh,CUDA_R_16BF,4096,4096,&zero,wfull,CUDA_R_16BF,32,32,chunk,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));C(cudaStreamSynchronize(s));
 auto q=qfull+size_t(chunk-m)*4096,w=wfull+size_t(chunk-m)*32;
 // Verify the reconstructed last producer row before using any derived bank.
 auto actualq=read<__nv_bfloat16>(d+"/indexer-rope-query.bf16"),actualw=read<__nv_bfloat16>(d+"/indexer-head-weights.bf16");if(actualq.size()!=4096||actualw.size()!=32)throw std::runtime_error("producer row extent");std::vector<__nv_bfloat16> checkq(4096),checkw(32);C(cudaMemcpy(checkq.data(),q+size_t(m-1)*4096,8192,cudaMemcpyDeviceToHost));C(cudaMemcpy(checkw.data(),w+size_t(m-1)*32,64,cudaMemcpyDeviceToHost));if(memcmp(checkq.data(),actualq.data(),8192)||memcmp(checkw.data(),actualw.data(),64))throw std::runtime_error("producer reconstruction mismatch");
 std::vector<int> positions(m);for(int r=0;r<m;r++)positions[r]=pos0+r;
 if(fixture){positions[0]=-1;if(m>1)positions[1]=0;if(m>2)positions[2]=3;if(m>3)positions[3]=4;if(m>4)positions[4]=4*n+9;tie_rows<<<16,256,0,s>>>(q);}
 auto p=upload(positions);auto tmp=alloc<float>(size_t(m)*n),out=alloc<float>(size_t(m)*n);int top=std::min(512,n),runs=(n+1023)/1024,stride=runs*top;
 auto sa=alloc<float>(size_t(m)*stride),sb=alloc<float>(size_t(m)*stride);auto ia=alloc<int>(size_t(m)*stride),ib=alloc<int>(size_t(m)*stride),selected=alloc<int>(size_t(m)*top),raw=alloc<int>(size_t(m)*2051);
 auto score=[&](int mode){int band=mode&&m>=64?std::max(1,m/8):m;for(int r=0;r<m;r+=band){int rows=std::min(band,m-r),heads=std::min(32,m/rows);for(int hs=0;hs<32;hs+=heads){int hc=std::min(heads,32-hs);B(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,128,&scale,k,CUDA_R_16BF,128,0,q+size_t(r)*4096+hs*128,CUDA_R_16BF,4096,128,&zero,tmp,CUDA_R_32F,n,size_t(rows)*n,hc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));dsa_prefill_accumulate_gemm_scores_kernel<<<(size_t(rows)*n+255)/256,256,0,s>>>(out+size_t(r)*n,tmp,w+size_t(r)*32,p+r,rows,n,n,32,hs,hc,rows*n,hs==0,0,4);}}};
 auto select=[&](){dsa_prefill_topk_radix_sort_rows_kernel<<<dim3(runs,m),256,0,s>>>(out,sa,ia,selected,p,m,n,top,stride,runs,1024,4);int nr=runs;bool first=true;while(nr>1){int next=(nr+1)/2;dsa_prefill_topk_linear_merge_rows_kernel<<<dim3(next,m),256,0,s>>>(first?sa:sb,first?ia:ib,first?sb:sa,first?ib:ia,selected,m,top,stride,nr,1024);nr=next;first=!first;}};
 auto expand=[&](){dsa_prefill_kpool_expand_kernel<<<(size_t(m)*2051+255)/256,256,0,s>>>(selected,raw,p,m,top,2051,4);};
 auto pipeline=[&](int mode){score(mode);select();expand();};
 for(int mode: {0,1,1,0}){
 pipeline(mode);C(cudaStreamSynchronize(s));save(o+"-score"+std::to_string(mode)+".f32",out,size_t(m)*n);save(o+"-selected"+std::to_string(mode)+".i32",selected,size_t(m)*top);save(o+"-raw"+std::to_string(mode)+".i32",raw,size_t(m)*2051);
 cudaGraph_t graph;cudaGraphExec_t exec;C(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));pipeline(mode);C(cudaStreamEndCapture(s,&graph));C(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));C(cudaGraphLaunch(exec,s));C(cudaStreamSynchronize(s));save(o+"-graph-score"+std::to_string(mode)+".f32",out,size_t(m)*n);save(o+"-graph-selected"+std::to_string(mode)+".i32",selected,size_t(m)*top);save(o+"-graph-raw"+std::to_string(mode)+".i32",raw,size_t(m)*2051);
 for(int rep=0;rep<5;rep++)for(int part=0;part<4;part++){cudaEvent_t a,b;C(cudaEventCreate(&a));C(cudaEventCreate(&b));C(cudaEventRecord(a,s));for(int i=0;i<5;i++){if(part==0)pipeline(mode);if(part==1)score(mode);if(part==2){select();expand();}if(part==3)C(cudaGraphLaunch(exec,s));}C(cudaEventRecord(b,s));C(cudaEventSynchronize(b));float ms;C(cudaEventElapsedTime(&ms,a,b));printf("{\"mode\":%d,\"part\":%d,\"rep\":%d,\"ms\":%.8f}\n",mode,part,rep,ms/5);fflush(stdout);C(cudaEventDestroy(a));C(cudaEventDestroy(b));}C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(graph));}
 save(o+"-q.bf16",q,size_t(m)*4096);save(o+"-weights.bf16",w,size_t(m)*32);save(o+"-positions.i32",p,m);B(cublasDestroy(h));C(cudaStreamDestroy(s));for(void*x:{(void*)workspace,(void*)db,(void*)dh,(void*)dqw,(void*)dhw,(void*)qfull,(void*)wfull,(void*)k,(void*)p,(void*)tmp,(void*)out,(void*)sa,(void*)sb,(void*)ia,(void*)ib,(void*)selected,(void*)raw})C(cudaFree(x));return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
