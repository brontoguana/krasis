// Bounded pedantic-FP32 algorithm comparison on the same gathered operands.
#include "gather_baseline.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include "prefill_mask_fusion.cuh"
#include <cstring>
#include <functional>
#include "value_local_tensor.cuh"
#include "value_local_tensor_probe.cuh"
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
 cublasLtHandle_t lt;B(cublasLtCreate(&lt));const float one=1,zero=0,scale=.0625f;
 std::map<std::pair<int,int>,std::unique_ptr<Plan>>plans;
 std::map<std::pair<int,int>,cublasLtMatmulAlgo_t>algos;
 auto gather=[&](int start,int n){mla_prefill_gather_k6_no_mask<<<dim3((std::max(cap,64)+7)/8,n),256,0,cudaStreamPerThread>>>(dense,query,q+(size_t)start*64*512,cache,ids+(size_t)start*cap,positions+start,cap,context);};
 auto classic=[&](bool score,int n){B(cublasGemmStridedBatchedEx(h,score?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,score?cap:512,64,score?512:cap,score?&scale:&one,dense,CUDA_R_32F,512,(long long)cap*512,score?query:probs,CUDA_R_32F,score?512:cap,(long long)64*(score?512:cap),&zero,score?scores:output,CUDA_R_32F,score?cap:512,(long long)64*(score?cap:512),n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));};
 auto contract=[&](bool score,int n){auto key=std::make_pair(score?0:1,n);if(!algos.count(key)){classic(score,n);return;}auto&p=*plans.at(key);auto&algo=algos.at(key);float*dst=score?scores:output;B(cublasLtMatmul(lt,p.operation,score?&scale:&one,dense,p.a,score?query:probs,p.b,&zero,dst,p.c,dst,p.c,&algo,workspace,16<<20,cudaStreamPerThread));};
 auto local_value=[&](int start,int n){value_local_tensor<<<dim3(8,2,n),256,0,cudaStreamPerThread>>>(output,probs,cache,ids+(size_t)start*cap,positions+start,cap,context);C(cudaGetLastError());};
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
 cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
 auto measure=[&](const char*name,std::function<void()>launch){std::vector<float>times;for(int rep=0;rep<14;rep++){C(cudaEventRecord(begin,cudaStreamPerThread));launch();C(cudaEventRecord(end,cudaStreamPerThread));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));if(rep>=2)times.push_back(ms);}std::sort(times.begin(),times.end());printf("{\"scope\":\"%s\",\"median_ms\":%.9g,\"samples\":[",name,(times[5]+times[6])/2);for(size_t i=0;i<times.size();i++)printf("%s%.9g",i?",":"",times[i]);printf("]}\n");fflush(stdout);};
 int n=std::min(tile,rows),start=rows-n;gather(start,n);contract(true,n);softmax(start,n);contract(false,n);C(cudaStreamSynchronize(cudaStreamPerThread));
 measure("resident_score",[&]{contract(true,n);});measure("resident_value",[&]{contract(false,n);});
 for(int mode:{0,1,1,0}){ // 0 full pipeline; 1 resident contractions with unchanged softmax.
  std::string label=mode?"resident_pipeline":"full_pipeline";
  measure(label.c_str(),[&]{if(!mode)gather(start,n);contract(true,n);softmax(start,n);contract(false,n);});
 }
 measure("direct_value",[&]{local_value(start,n);});
 for(int mode:{0,1,1,0}){measure(mode?"candidate_pipeline":"accepted_pipeline",[&]{gather(start,n);contract(true,n);softmax(start,n);if(mode)local_value(start,n);else contract(false,n);});}
 // Complete intermediate/output readback is outside all ordinary timing.
 size_t compared=0,bits=0,bad=0;double maximum=0;
 for(int offset=0;offset<rows;offset+=tile){int count=std::min(tile,rows-offset);gather(offset,count);contract(true,count);softmax(offset,count);contract(false,count);
  std::vector<float>ref((size_t)count*64*512),actual(ref.size());C(cudaMemcpy(ref.data(),output,ref.size()*4,cudaMemcpyDeviceToHost));
  auto save=[&](const char*name,const float*p,size_t size){std::vector<float>x(size);C(cudaMemcpy(x.data(),p,size*4,cudaMemcpyDeviceToHost));std::ofstream f(out+"/"+name,std::ios::binary|std::ios::app);f.write((char*)x.data(),size*4);};
  if(rows<=64){save("scores.f32",scores,(size_t)count*64*cap);save("probs.f32",probs,(size_t)count*64*cap);save("accepted.f32",output,ref.size());}
  local_value(offset,count);C(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
  for(size_t i=0;i<actual.size();i++){bits+=memcmp(&actual[i],&ref[i],4)!=0;double e=std::abs((double)actual[i]-ref[i]);if(!std::isfinite(e)||e>2e-4)bad++;if(std::isfinite(e))maximum=std::max(maximum,e);}compared+=actual.size();
  if(rows<=64)save("candidate.f32",output,actual.size());
 }
 printf("{\"comparison\":true,\"values\":%zu,\"different_bits\":%zu,\"over_2e_4\":%zu,\"max_abs\":%.9g}\n",compared,bits,bad,maximum);fflush(stdout);
 // Graph both full pipelines, checking all outputs of the last tile.
 const int graph_start=((rows-1)/tile)*tile,graph_n=rows-graph_start;
 for(int mode:{0,1}){cudaGraph_t graph;cudaGraphExec_t executable;C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));gather(graph_start,graph_n);contract(true,graph_n);softmax(graph_start,graph_n);if(mode)local_value(graph_start,graph_n);else contract(false,graph_n);C(cudaStreamEndCapture(cudaStreamPerThread,&graph));C(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
  measure(mode?"graph_candidate_pipeline":"graph_accepted_pipeline",[&]{C(cudaGraphLaunch(executable,cudaStreamPerThread));});
  if(rows<=64){std::vector<float>x((size_t)graph_n*64*512);C(cudaMemcpy(x.data(),output,x.size()*4,cudaMemcpyDeviceToHost));std::ofstream f(out+(mode?"/graph-candidate.f32":"/graph-accepted.f32"),std::ios::binary);f.write((char*)x.data(),x.size()*4);}
  C(cudaGraphExecDestroy(executable));C(cudaGraphDestroy(graph));
 }
 measure("gather",[&]{gather(start,n);});measure("softmax",[&]{softmax(start,n);});
 // Intrusive diagnostic only, excluded from ordinary timings above.
 auto ticks=alloc<unsigned long long>((size_t)n*16*4);
 for(int trial=0;trial<3;trial++){
 value_local_tensor_probe<<<dim3(8,2,n),256,0,cudaStreamPerThread>>>(output,probs,cache,ids+(size_t)start*cap,positions+start,cap,context,ticks);
 std::vector<unsigned long long>host((size_t)n*16*4);C(cudaMemcpy(host.data(),ticks,host.size()*8,cudaMemcpyDeviceToHost));
 double stage[4]={};for(size_t i=0;i<host.size();i++)stage[i%4]+=host[i];
 printf("{\"stage_probe\":%d,\"preparation_cycles_sum\":%.0f,\"first_barrier_cycles_sum\":%.0f,\"contraction_cycles_sum\":%.0f,\"second_barrier_cycles_sum\":%.0f,\"ctas\":%d}\n",trial,stage[0],stage[1],stage[2],stage[3],n*16);
 }
 cudaFuncAttributes attr;C(cudaFuncGetAttributes(&attr,value_local_tensor));int active=0;C(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,value_local_tensor,256,0));printf("{\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"active_cta_per_sm\":%d}\n",attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,active);
 size_t free,total;C(cudaMemGetInfo(&free,&total));printf("{\"minimum_free_checked_bytes\":%zu,\"rows\":%d,\"context\":%d,\"capacity\":%d}\n",free,rows,context,cap);if(free<600ull*1024*1024)throw std::runtime_error("reserve");
 C(cudaStreamSynchronize(cudaStreamPerThread));B(cublasLtDestroy(lt));B(cublasDestroy(h));return bad?2:0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
