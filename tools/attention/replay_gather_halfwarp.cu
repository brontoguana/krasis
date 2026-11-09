// Actual packed K6 operands, complete gather outputs, eager/graph, repeated timing.
#include "gather_baseline.cuh"
#include "prefill_mask_fusion.cuh"
#include "gather_halfwarp.cuh"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
std::vector<char> read(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||(size_t)f.tellg()!=n)throw std::runtime_error("size "+p);std::vector<char>x(n);f.seekg(0);f.read(x.data(),n);return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,x.size()));C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("replay CAPTURE START ROWS CONTEXT CAPACITY OUT");
 const std::string prefix=std::string(argv[1])+"/fused-"+argv[2]+"-",out=argv[6];
 const int rows=std::stoi(argv[3]),context=std::stoi(argv[4]),cap=std::stoi(argv[5]),tile=32;
 if(rows<1||context<1||cap<1)throw std::runtime_error("dimensions");
 auto q=upload<float>(read(prefix+"q-absorbed.f32",(size_t)rows*64*512*4));auto ids=upload<int>(read(prefix+"selected.i32",(size_t)rows*cap*4));auto positions=upload<int>(read(prefix+"positions.i32",rows*4));auto cache=upload<unsigned char>(read(prefix+"ckv-cache.bin",(size_t)context*448));
 const size_t nv=(size_t)tile*cap*512,nq=(size_t)tile*64*512;
 auto v=alloc<float>(nv),query=alloc<float>(nq),refv=alloc<float>(nv),refq=alloc<float>(nq);
 size_t compared=0,diff=0;
 auto launch=[&](int mode,int start){int n=std::min(tile,rows-start);dim3 grid((std::max(cap,64)+7)/8,n);auto input=q+(size_t)start*64*512;auto index=ids+(size_t)start*cap;auto position=positions+start;
  if(mode)mla_prefill_gather_k6_halfwarp<<<grid,256,0,cudaStreamPerThread>>>(v,query,input,cache,index,position,cap,context);
  else mla_prefill_gather_k6_no_mask<<<grid,256,0,cudaStreamPerThread>>>(refv,refq,input,cache,index,position,cap,context);
  C(cudaGetLastError());
 };
 // Read back complete actual outputs band by band, including every masked row.
 for(int start=0;start<rows;start+=tile){int n=std::min(tile,rows-start);launch(0,start);launch(1,start);C(cudaStreamSynchronize(cudaStreamPerThread));
  for(auto pair:{std::make_pair(v,refv),std::make_pair(query,refq)}){
   size_t count=(size_t)n*(pair.first==v?cap:64)*512;std::vector<unsigned>a(count),b(count);C(cudaMemcpy(a.data(),pair.first,count*4,cudaMemcpyDeviceToHost));C(cudaMemcpy(b.data(),pair.second,count*4,cudaMemcpyDeviceToHost));
   for(size_t i=0;i<count;i++)diff+=a[i]!=b[i];compared+=count;
   if(rows<=64){std::ofstream f(out+(pair.first==v?"/values.f32":"/query.f32"),std::ios::binary|std::ios::app);f.write((char*)a.data(),count*4);}
  }
 }
 printf("{\"compared\":%zu,\"different\":%zu}\n",compared,diff);fflush(stdout);if(diff)return 2;
 for(bool graph:{false,true})for(int mode:{0,1,1,0}){
  cudaGraph_t captured=nullptr;cudaGraphExec_t executable=nullptr;
  if(graph){C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));for(int start=0;start<rows;start+=tile)launch(mode,start);C(cudaStreamEndCapture(cudaStreamPerThread,&captured));C(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0));}
  auto run=[&](){if(graph)C(cudaGraphLaunch(executable,cudaStreamPerThread));else for(int start=0;start<rows;start+=tile)launch(mode,start);};
  run();C(cudaStreamSynchronize(cudaStreamPerThread));
  if(graph){
   size_t count=(size_t)((rows-1)%tile+1)*cap*512;std::vector<unsigned>a(count),b(count);
   C(cudaMemcpy(a.data(),v,count*4,cudaMemcpyDeviceToHost));C(cudaMemcpy(b.data(),refv,count*4,cudaMemcpyDeviceToHost));
   if(a!=b)throw std::runtime_error("graph gather differs from eager control");
   count=(size_t)((rows-1)%tile+1)*64*512;a.resize(count);b.resize(count);
   C(cudaMemcpy(a.data(),query,count*4,cudaMemcpyDeviceToHost));C(cudaMemcpy(b.data(),refq,count*4,cudaMemcpyDeviceToHost));
   if(a!=b)throw std::runtime_error("graph query differs from eager control");
  }
  cudaEvent_t a,b;C(cudaEventCreate(&a));C(cudaEventCreate(&b));std::vector<float>times;
  for(int rep=0;rep<5;rep++){C(cudaEventRecord(a,cudaStreamPerThread));run();C(cudaEventRecord(b,cudaStreamPerThread));C(cudaEventSynchronize(b));float ms;C(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);}
  std::sort(times.begin(),times.end());printf("{\"mode\":%d,\"graph\":%s,\"median_ms\":%.6f}\n",mode,graph?"true":"false",times[2]);fflush(stdout);C(cudaEventDestroy(a));C(cudaEventDestroy(b));if(graph){C(cudaGraphExecDestroy(executable));C(cudaGraphDestroy(captured));}
 }
 size_t free,total;C(cudaMemGetInfo(&free,&total));if(free<600ull*1024*1024)throw std::runtime_error("reserve");C(cudaDeviceReset());return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
