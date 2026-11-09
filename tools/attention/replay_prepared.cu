// Offline shared-preparation screen; all four TF32 component products use FP32 accumulation.
#include "gather_baseline.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include "prefill_mask_fusion.cuh"
#include <cstring>
#include <functional>
#include <mma.h>
__device__ float tfpart(float x){unsigned y;asm("cvt.rna.tf32.f32 %0, %1;":"=r"(y):"f"(x));return __uint_as_float(y);}
__global__ void prepare_once(float*vh,float*vl,float*qh,float*ql,const float*q,const unsigned char*cache,const int*ids,const int*pos,int cap,int context){
 int row=blockIdx.y,slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
 if(slot<cap){int seq=min(max(pos[row]+1,0),context);int idx=lane==0?ids[(long long)row*cap+slot]:0;idx=__shfl_sync(0xffffffff,idx,0);
 for(int d=lane;d<512;d+=32){long long i=((long long)row*cap+slot)*512+d;float x=idx>=0&&idx<seq?mla_prefill_load_k6_value(cache,idx,512,d):0.f;float h=tfpart(x);vh[i]=h;vl[i]=tfpart(__fsub_rn(x,h));}}
 if(slot<64)for(int d=lane;d<512;d+=32){long long i=((long long)row*64+slot)*512+d;float x=q[i],h=tfpart(x);qh[i]=h;ql[i]=tfpart(__fsub_rn(x,h));}
}
__global__ void prepare_prob(float*hi,float*lo,size_t n){for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float x=hi[i],h=tfpart(x);hi[i]=h;lo[i]=tfpart(__fsub_rn(x,h));}}

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#define C(x) do{auto status=(x);if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}while(0)
#define B(x) do{auto status=(x);if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS "+std::to_string(status));}while(0)
std::vector<char> read(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||(size_t)f.tellg()!=n)throw std::runtime_error("size "+p);std::vector<char>x(n);f.seekg(0);f.read(x.data(),n);return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,x.size()));C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
struct Plan{
 cublasLtMatmulDesc_t operation; cublasLtMatrixLayout_t a,b,c;
 Plan(bool score,int n,int cap){
  B(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F_PEDANTIC,CUDA_R_32F));
  cublasOperation_t transpose=score?CUBLAS_OP_T:CUBLAS_OP_N;
  B(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
  B(cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,512,cap,512));
  B(cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,score?512:cap,64,score?512:cap));
  B(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,score?cap:512,64,score?cap:512));
  long long strides[]={(long long)512*cap,(long long)(score?512:cap)*64,(long long)(score?cap:512)*64};
  int i=0;for(auto layout:{a,b,c}){
   B(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&n,sizeof(n)));
   B(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(strides[i])));i++;
  }
 }
 ~Plan(){cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);cublasLtMatmulDescDestroy(operation);}
};

int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("replay CAPTURE START ROWS CONTEXT CAP OUT");
 std::string prefix=std::string(argv[1])+"/fused-"+argv[2]+"-",out=argv[6];
 int rows=std::stoi(argv[3]),context=std::stoi(argv[4]),cap=std::stoi(argv[5]),tile=32;
 if(rows<1||context<1||cap<1||cap>4096)throw std::runtime_error("geometry");
 auto q=upload<float>(read(prefix+"q-absorbed.f32",(size_t)rows*64*512*4));
 auto ids=upload<int>(read(prefix+"selected.i32",(size_t)rows*cap*4));
 auto positions=upload<int>(read(prefix+"positions.i32",rows*4));
 auto cache=upload<unsigned char>(read(prefix+"ckv-cache.bin",(size_t)context*448));
 auto dense=alloc<float>((size_t)tile*cap*512),query=alloc<float>((size_t)tile*64*512);
 auto scores=alloc<float>((size_t)tile*64*cap),probs=alloc<float>((size_t)tile*64*cap),output=alloc<float>((size_t)tile*64*512);
 auto workspace=alloc<char>(16<<20);cublasHandle_t h;B(cublasCreate(&h));B(cublasSetStream(h,cudaStreamPerThread));B(cublasSetWorkspace(h,workspace,16<<20));
 B(cublasSetMathMode(h,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
 cublasLtHandle_t lt;B(cublasLtCreate(&lt));const float one=1,zero=0,scale=.0625f;
 std::map<std::pair<int,int>,std::unique_ptr<Plan>>plans;
 std::map<std::pair<int,int>,cublasLtMatmulAlgo_t>algos;
 auto gather=[&](int start,int n){mla_prefill_gather_k6_no_mask<<<dim3((std::max(cap,64)+7)/8,n),256,0,cudaStreamPerThread>>>(dense,query,q+(size_t)start*64*512,cache,ids+(size_t)start*cap,positions+start,cap,context);};
 auto classic=[&](bool score,int n){B(cublasGemmStridedBatchedEx(h,score?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,score?cap:512,64,score?512:cap,score?&scale:&one,dense,CUDA_R_32F,512,(long long)cap*512,score?query:probs,CUDA_R_32F,score?512:cap,(long long)64*(score?512:cap),&zero,score?scores:output,CUDA_R_32F,score?cap:512,(long long)64*(score?cap:512),n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));};
 auto contract=[&](bool score,int n){auto key=std::make_pair(score?0:1,n);if(!algos.count(key)){classic(score,n);return;}auto&p=*plans.at(key);auto&algo=algos.at(key);float*dst=score?scores:output;B(cublasLtMatmul(lt,p.operation,score?&scale:&one,dense,p.a,score?query:probs,p.b,&zero,dst,p.c,dst,p.c,&algo,workspace,16<<20,cudaStreamPerThread));};
 auto softmax=[&](int start,int n){mla_prefill_masked_softmax_f32<<<(n*64+7)/8,256,0,cudaStreamPerThread>>>(probs,scores,ids+(size_t)start*cap,positions+start,n,cap,context);};
 for(int n:{std::min(tile,rows),rows%tile})if(n&&!plans.count({0,n})){
  gather(0,n);classic(true,n);softmax(0,n);
  for(int stage=0;stage<2;stage++){
   auto key=std::make_pair(stage,n);plans[key]=std::make_unique<Plan>(stage==0,n,cap);auto&p=*plans[key];
   if(n!=32||cap!=2051)continue; // Match accepted production fallback for unprepared tails.
   cublasLtMatmulPreference_t pref;B(cublasLtMatmulPreferenceCreate(&pref));size_t bytes=16<<20;B(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
   // Accepted full-batch plan is the third bounded heuristic result for both stages.
   cublasLtMatmulHeuristicResult_t choices[4];int count=0;B(cublasLtMatmulAlgoGetHeuristic(lt,p.operation,p.a,p.b,p.c,p.c,pref,4,choices,&count));
   if(count<3||choices[2].state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("accepted plan unavailable");
   classic(stage==0,n);size_t values=(size_t)n*64*(stage==0?cap:512);std::vector<float>ref(values),actual(values);C(cudaMemcpy(ref.data(),stage==0?scores:output,values*4,cudaMemcpyDeviceToHost));
   algos[key]=choices[2].algo;contract(stage==0,n);C(cudaMemcpy(actual.data(),stage==0?scores:output,values*4,cudaMemcpyDeviceToHost));
   if(memcmp(ref.data(),actual.data(),values*4))throw std::runtime_error("accepted exact plan mismatch");
   printf("{\"plan_stage\":%d,\"batch\":%d,\"workspace\":%zu,\"algorithm_data\":[",stage,n,choices[2].workspaceSize);for(int i=0;i<8;i++)printf("%s%llu",i?",":"",(unsigned long long)choices[2].algo.data[i]);printf("]}\n");
   B(cublasLtMatmulPreferenceDestroy(pref));
  }
 }

 const size_t per_row=(2ull*cap*512+2ull*64*512+2ull*64*cap+64ull*512)*4;
 const size_t output_bytes=32ull*64*512*4;
 const int ctile=std::min(16,(int)((172212989-output_bytes)/(per_row-64*512*4)));if(ctile<1)throw std::runtime_error("workspace");
 auto vh=alloc<float>((size_t)ctile*cap*512),vl=alloc<float>((size_t)ctile*cap*512);
 auto qh=alloc<float>((size_t)ctile*64*512),ql=alloc<float>((size_t)ctile*64*512);
 auto cs=alloc<float>((size_t)ctile*64*cap),pl=alloc<float>((size_t)ctile*64*cap);float*co=output;
 printf("{\"physical_tile\":%d,\"arena_bytes\":%zu,\"library_workspace_bytes\":16777216,\"padding\":0}\n",ctile,ctile*(per_row-64*512*4)+output_bytes);
 auto prep=[&](int start,int n){prepare_once<<<dim3((std::max(cap,64)+7)/8,n),256,0,cudaStreamPerThread>>>(vh,vl,qh,ql,q+(size_t)start*64*512,cache,ids+(size_t)start*cap,positions+start,cap,context);};
 auto four=[&](bool score,int n){
  // HH, LH, HL, LL; each call accumulates into the same FP32 destination.
  for(int part=0;part<4;part++){
   const float* left=(part&1)?vl:vh;const float*right=score?((part&2)?ql:qh):((part&2)?pl:cs);
   B(cublasGemmStridedBatchedEx(h,score?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,score?cap:512,64,score?512:cap,score?&scale:&one,left,CUDA_R_32F,512,(long long)cap*512,right,CUDA_R_32F,score?512:cap,(long long)64*(score?512:cap),part?&one:&zero,score?cs:co,CUDA_R_32F,score?cap:512,(long long)64*(score?cap:512),n,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
 };
 auto csoft=[&](int start,int n){mla_prefill_masked_softmax_f32<<<(n*64+7)/8,256,0,cudaStreamPerThread>>>(cs,cs,ids+(size_t)start*cap,positions+start,n,cap,context);};
 auto pprep=[&](int n){size_t num=(size_t)n*64*cap;prepare_prob<<<(num+255)/256,256,0,cudaStreamPerThread>>>(cs,pl,num);};
 auto candidate=[&](int start,int n){for(int x=0;x<n;x+=ctile){int m=std::min(ctile,n-x);co=output+(size_t)x*64*512;prep(start+x,m);four(true,m);csoft(start+x,m);pprep(m);four(false,m);}};
 auto accepted=[&](int start,int n){gather(start,n);contract(true,n);softmax(start,n);contract(false,n);};
 cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
 auto measure=[&](const char*name,std::function<void()>launch){std::vector<float>times;for(int rep=0;rep<14;rep++){
 if(std::string(name)=="candidate_softmax_one_tile"||std::string(name)=="candidate_probability_split_one_tile"||std::string(name)=="candidate_values_one_tile"){
 int m=std::min(ctile,rows);prep(0,m);four(true,m);
 if(std::string(name)!="candidate_softmax_one_tile")csoft(0,m);
 if(std::string(name)=="candidate_values_one_tile")pprep(m);
 }
 C(cudaEventRecord(begin,cudaStreamPerThread));launch();C(cudaEventRecord(end,cudaStreamPerThread));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));if(rep>=2)times.push_back(ms);}std::sort(times.begin(),times.end());printf("{\"scope\":\"%s\",\"median_ms\":%.9g,\"samples\":[",name,(times[5]+times[6])/2);for(size_t i=0;i<times.size();i++)printf("%s%.9g",i?",":"",times[i]);printf("]}\n");fflush(stdout);};
 int n=std::min(tile,rows),start=rows-n;
 for(int mode:{0,1,1,0})measure(mode?"candidate_pipeline":"accepted_pipeline",[&]{if(mode)candidate(start,n);else accepted(start,n);});
 int m=std::min(ctile,rows);prep(0,m);four(true,m);csoft(0,m);pprep(m);four(false,m);
 measure("candidate_preparation_one_tile",[&]{prep(0,m);});
 measure("candidate_scores_one_tile",[&]{four(true,m);});
 prep(0,m);four(true,m);csoft(0,m);
 measure("candidate_softmax_one_tile",[&]{csoft(0,m);});
 // Reinitialize probabilities for each observation; include only split in timed events.
 prep(0,m);four(true,m);csoft(0,m);
 measure("candidate_probability_split_one_tile",[&]{pprep(m);});
 measure("candidate_values_one_tile",[&]{four(false,m);});
 auto save=[&](const char*name,const float*p,size_t size){std::vector<float>x(size);C(cudaMemcpy(x.data(),p,size*4,cudaMemcpyDeviceToHost));std::ofstream f(out+"/"+name,std::ios::binary|std::ios::app);f.write((char*)x.data(),size*4);};
 // Complete readbacks outside timing. Preserve original accepted geometry.
 for(int x=0;x<rows;x+=tile){int count=std::min(tile,rows-x);accepted(x,count);save("scores.f32",scores,(size_t)count*64*cap);save("probs.f32",probs,(size_t)count*64*cap);save("accepted.f32",output,(size_t)count*64*512);}
 for(int x=0;x<rows;x+=ctile){int count=std::min(ctile,rows-x);co=output;prep(x,count);four(true,count);save("candidate-scores.f32",cs,(size_t)count*64*cap);csoft(x,count);save("candidate-probs.f32",cs,(size_t)count*64*cap);pprep(count);four(false,count);save("candidate.f32",co,(size_t)count*64*512);}
 // Candidate graph includes both physical tiles for natural32; retain final tile proof.
 const int gs=((rows-1)/tile)*tile,gn=rows-gs;
 for(int mode:{0,1}){cudaGraph_t graph;cudaGraphExec_t executable;C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));if(mode)candidate(gs,gn);else accepted(gs,gn);C(cudaStreamEndCapture(cudaStreamPerThread,&graph));C(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
 measure(mode?"graph_candidate_pipeline":"graph_accepted_pipeline",[&]{C(cudaGraphLaunch(executable,cudaStreamPerThread));});
 std::vector<float>x((size_t)gn*64*512);C(cudaMemcpy(x.data(),output,x.size()*4,cudaMemcpyDeviceToHost));std::ofstream f(out+(mode?"/graph-candidate.f32":"/graph-accepted.f32"),std::ios::binary);f.write((char*)x.data(),x.size()*4);
 C(cudaGraphExecDestroy(executable));C(cudaGraphDestroy(graph));}
 size_t free,total;C(cudaMemGetInfo(&free,&total));printf("{\"minimum_free_checked_bytes\":%zu,\"rows\":%d,\"context\":%d,\"capacity\":%d}\n",free,rows,context,cap);if(free<600ull*1024*1024)throw std::runtime_error("reserve");
 C(cudaStreamSynchronize(cudaStreamPerThread));B(cublasLtDestroy(lt));B(cublasDestroy(h));return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
