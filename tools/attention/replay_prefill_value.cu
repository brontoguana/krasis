// Bounded value-only reassociation on actual FP32 gathered operands.
#include "gather_baseline.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include "prefill_mask_fusion.cuh"
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
 if(argc!=7&&argc!=8)throw std::runtime_error("replay CAPTURE START ROWS CONTEXT CAPACITY OUT");
 const bool prefer_exact=argc==8&&std::string(argv[7])=="exact";
 printf("{\"prefer_exact\":%s}\n",prefer_exact?"true":"false");
 std::string prefix=std::string(argv[1])+"/fused-"+argv[2]+"-",out=argv[6];
 const int rows=std::stoi(argv[3]),context=std::stoi(argv[4]),cap=std::stoi(argv[5]),tile=32;
 if(rows<32||context<1||cap<1||cap>4096)throw std::runtime_error("unqualified geometry");
 const size_t elements=(size_t)rows*64*512;
 auto q=upload<float>(read(prefix+"q-absorbed.f32",elements*4));auto ids=upload<int>(read(prefix+"selected.i32",(size_t)rows*cap*4));auto positions=upload<int>(read(prefix+"positions.i32",rows*4));auto cache=upload<unsigned char>(read(prefix+"ckv-cache.bin",(size_t)context*448));
 auto reference=read(prefix+"baseline-output.f32",elements*4);auto output=alloc<float>(elements);
 auto dense=alloc<float>((size_t)tile*cap*512),query=alloc<float>((size_t)tile*64*512),scores=alloc<float>((size_t)tile*64*cap),probs=alloc<float>((size_t)tile*64*cap);
 const size_t workspace_bytes=16<<20;auto workspace=alloc<char>(workspace_bytes);
 cublasHandle_t handle;B(cublasCreate(&handle));B(cublasSetStream(handle,cudaStreamPerThread));B(cublasSetWorkspace(handle,workspace,workspace_bytes));
 cublasLtHandle_t lt;B(cublasLtCreate(&lt));
 bool fusion=false;
 auto gather=[&](int start,int n){if(fusion){mla_prefill_gather_k6_no_mask<<<dim3((std::max(cap,64)+7)/8,n),256,0,cudaStreamPerThread>>>(dense,query,q+(size_t)start*64*512,cache,ids+(size_t)start*cap,positions+start,cap,context);return;}int blocks=n*((cap+7)/8+8+1);mla_prefill_gather_query_k6_f32_structured_kernel<<<blocks,256,0,cudaStreamPerThread>>>(dense,query,scores,q,(const __nv_bfloat16*)q,cache,nullptr,ids,positions,start,n,64,256,256,0,512,cap,context);};
 const float one=1,zero=0,scale=.0625f;
 auto classic=[&](bool score,int n,int start){
  B(cublasGemmStridedBatchedEx(handle,score?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,score?cap:512,64,score?512:cap,score?&scale:&one,dense,CUDA_R_32F,512,(long long)cap*512,score?query:probs,CUDA_R_32F,score?512:cap,(long long)64*(score?512:cap),score?(fusion?&zero:&one):&zero,score?scores:output+(size_t)start*64*512,CUDA_R_32F,score?cap:512,(long long)64*(score?cap:512),n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
 };
 auto softmax=[&](int n,int start=0){if(fusion)mla_prefill_masked_softmax_f32<<<(n*64+7)/8,256,0,cudaStreamPerThread>>>(probs,scores,ids+(size_t)start*cap,positions+start,n,cap,context);else mla_prefill_softmax_weights_warp_f32_kernel<<<(n*64+7)/8,256,0,cudaStreamPerThread>>>(probs,scores,n,64,cap);};
 auto launch_lt=[&](Plan&p,const cublasLtMatmulAlgo_t&algorithm,bool score,int start){
  float* destination=score?scores:output+(size_t)start*64*512;
  B(cublasLtMatmul(lt,p.operation,score?&scale:&one,dense,p.a,score?query:probs,p.b,score?(fusion?&zero:&one):&zero,destination,p.c,destination,p.c,&algorithm,workspace,workspace_bytes,cudaStreamPerThread));
 };
 cublasLtMatmulAlgo_t selected[2],exact_value;bool admitted[2]={false,false},exact_admitted=false;double exact_best=1e30;
 std::vector<float> expected_score(tile*64*cap),expected_value(tile*64*512);
 gather(0,tile);classic(true,tile,0);softmax(tile);classic(false,tile,0);
 C(cudaMemcpy(expected_score.data(),scores,expected_score.size()*4,cudaMemcpyDeviceToHost));
 C(cudaMemcpy(expected_value.data(),output,expected_value.size()*4,cudaMemcpyDeviceToHost));
 cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
 for(int stage=0;stage<2;stage++){
  const bool score=stage==0;Plan p(score,tile,cap);cublasLtMatmulPreference_t preference;B(cublasLtMatmulPreferenceCreate(&preference));
  B(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
  cublasLtMatmulHeuristicResult_t choices[16];int found=0;
  B(cublasLtMatmulAlgoGetHeuristic(lt,p.operation,p.a,p.b,p.c,p.c,preference,16,choices,&found));
  double best=1e30;
  // First four successful heuristic configurations; no full algorithm sweep.
  for(int choice=-1;choice<std::min(found,4);choice++){
   if(choice>=0 && choices[choice].state!=CUBLAS_STATUS_SUCCESS)continue;
   std::vector<float> times;
   for(int rep=0;rep<12;rep++){
    if(score)gather(0,tile);
    C(cudaEventRecord(begin,cudaStreamPerThread));
    if(choice<0)classic(score,tile,0);else launch_lt(p,choices[choice].algo,score,0);
    C(cudaEventRecord(end,cudaStreamPerThread));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));if(rep>=2)times.push_back(ms);
   }
   const auto&expected=score?expected_score:expected_value;std::vector<float>actual(expected.size());
   C(cudaMemcpy(actual.data(),score?scores:output,actual.size()*4,cudaMemcpyDeviceToHost));
   size_t bad=0,bits=0;double maximum=0;
   for(size_t i=0;i<actual.size();i++){
    if(memcmp(&actual[i],&expected[i],4))bits++;
    if(actual[i]==expected[i])continue;
    double error=std::abs((double)actual[i]-expected[i]);
    if(!std::isfinite(error)||error>2e-4)bad++;if(std::isfinite(error))maximum=std::max(maximum,error);
   }
   std::sort(times.begin(),times.end());double median=(times[4]+times[5])/2;int tile_id=-1;size_t written=0;
   if(choice>=0)B(cublasLtMatmulAlgoConfigGetAttribute(&choices[choice].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile_id,sizeof(tile_id),&written));
   printf("{\"stage\":%d,\"choice\":%d,\"tile_id\":%d,\"median_ms\":%.9g,\"bit_differences\":%zu,\"over_2e_4\":%zu,\"max_abs\":%.9g}\n",stage,choice,tile_id,median,bits,bad,maximum);fflush(stdout);
   if(!score && choice>=0 && !bad && bits==0 && median<exact_best){exact_best=median;exact_value=choices[choice].algo;exact_admitted=true;}
   if(choice>=0 && !bad && (score?bits==0:maximum<=2e-4) && median<best){best=median;selected[stage]=choices[choice].algo;admitted[stage]=true;}
  }
  B(cublasLtMatmulPreferenceDestroy(preference));
  // Restore canonical score and probabilities for independent value-stage tuning.
  gather(0,tile);classic(true,tile,0);softmax(tile);C(cudaDeviceSynchronize());
 }
 if(!admitted[0]||!admitted[1]||!exact_admitted)throw std::runtime_error("no passing bounded Lt choice");
 std::map<std::pair<int,int>,std::unique_ptr<Plan>> plans;
 for(int n:{tile,rows%tile})if(n)for(int stage=0;stage<2;stage++)plans[{stage,n}]=std::make_unique<Plan>(stage==0,n,cap);
 for(bool graph:{false,true})for(int mode:{5,7,7,5}){
  fusion=true;
  auto pipeline=[&](){
   for(int start=0;start<rows;start+=tile){int n=std::min(tile,rows-start);gather(start,n);
    if(n==tile)launch_lt(*plans[{0,n}],selected[0],true,start);else classic(true,n,start);
    softmax(n,start);
    if(n==tile)launch_lt(*plans[{1,n}],(mode&2)?selected[1]:exact_value,false,start);else classic(false,n,start);
   }
  };
  cudaGraph_t captured=nullptr;cudaGraphExec_t executable=nullptr;
  if(graph){C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));pipeline();C(cudaStreamEndCapture(cudaStreamPerThread,&captured));C(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0));}
  auto run=[&](){if(graph)C(cudaGraphLaunch(executable,cudaStreamPerThread));else pipeline();};
  run();C(cudaDeviceSynchronize());
  for(int rep=0;rep<3;rep++){
   C(cudaEventRecord(begin,cudaStreamPerThread));run();C(cudaEventRecord(end,cudaStreamPerThread));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));
   printf("{\"mode\":%d,\"graph\":%s,\"rep\":%d,\"ms\":%.6f}\n",mode,graph?"true":"false",rep,ms);fflush(stdout);
  }
  std::vector<float>actual(elements);C(cudaMemcpy(actual.data(),output,elements*4,cudaMemcpyDeviceToHost));const float*expected=(const float*)reference.data();size_t bad=0,bits=0,nonfinite=0;double maximum=0;
  for(size_t i=0;i<elements;i++){if(memcmp(&actual[i],&expected[i],4))bits++;double error=std::abs((double)actual[i]-expected[i]);if(!std::isfinite(actual[i])||!std::isfinite(expected[i]))nonfinite++;if(error>2e-4)bad++;maximum=std::max(maximum,error);}
  printf("{\"mode\":%d,\"graph\":%s,\"values\":%zu,\"bit_differences\":%zu,\"over_2e_4\":%zu,\"nonfinite\":%zu,\"max_abs\":%.9g}\n",mode,graph?"true":"false",elements,bits,bad,nonfinite,maximum);fflush(stdout);
  for(int row:{0,1,17,31,32,rows-1})if(row<rows){std::ofstream f(out+"/value-mode"+std::to_string(mode)+(graph?"-graph":"-eager")+"-row"+std::to_string(row)+".f32",std::ios::binary);f.write((char*)(actual.data()+(size_t)row*64*512),64*512*4);}
  if(graph){C(cudaGraphExecDestroy(executable));C(cudaGraphDestroy(captured));}
  if(bad||nonfinite)return 2;
 }
 B(cublasLtDestroy(lt));B(cublasDestroy(handle));return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
