// Offline replay of complete captured sparse attention operands.
#include "attention_baseline.cuh"
#include "mla_sparse_split.cuh"
#include "mla_sparse_tensor.cuh"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
std::vector<char> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);std::vector<char>x(f.tellg());f.seekg(0);f.read(x.data(),x.size());return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,std::max<size_t>(x.size(),4)));if(!x.empty())C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){try{
 if(argc!=7 && argc!=8)throw std::runtime_error("replay INPUT OUTPUT CONTEXT CAPACITY ROPE THREADS MODE");
 std::string in=argv[1],out=argv[2];int seq=std::stoi(argv[3]),cap=std::stoi(argv[4]),rope=std::stoi(argv[5]);int threads=std::stoi(argv[6]);int mode=argc>7?std::stoi(argv[7]):1;int heads=64,dim=512,splits=(cap+127)/128;
 auto q=upload<float>(read(in+"/q.f32"));auto qp=upload<float>(rope?read(in+"/qp.f32"):std::vector<char>());
 auto cache=upload<unsigned char>(read(in+"/cache.bin"));auto pe=upload<unsigned char>(rope?read(in+"/pe.bin"):std::vector<char>());auto ids=upload<int>(read(in+"/indices.i32"));
 auto n=upload<int>(std::vector<char>((char*)&seq,(char*)&seq+4));auto original=alloc<float>(heads*dim);auto candidate=alloc<float>(heads*dim);auto partial=alloc<float>(std::max((size_t)heads*splits*(dim+2),(size_t)(heads+dim)*cap));
 const int padded=(cap+15)/16*16;
 auto dense=alloc<float>((size_t)padded*512);auto scores=alloc<float>((size_t)heads*padded);auto probabilities=alloc<float>((size_t)heads*padded);auto tensor_partial=alloc<float>((size_t)heads*splits*512);
 auto staged=alloc<unsigned char>((size_t)cap*(dim+rope)/16*14);auto staged_ids=alloc<int>(cap);auto staged_count=alloc<int>(1);
 auto run=[&](bool optimized,cudaStream_t stream){
  mla_stage_selected_packed_g<<<cap,256,0,stream>>>(staged,staged+(size_t)cap*dim/16*14,staged_ids,staged_count,cache,pe,ids,n,cap,dim/16*14,rope/16*14);

  if(optimized && mode==5){
   mla_sparse_k6_unpack_values<<<padded,256,0,stream>>>(dense,staged,staged_ids,dim,cap,seq,staged_count);
   mla_sparse_k6_tensor_scores<<<dim3(heads/16,(padded+63)/64),128,0,stream>>>(scores,q,dense,padded);
   mla_sparse_k6_tensor_softmax<<<heads,threads,(padded+threads/32)*4,stream>>>(probabilities,scores,staged_ids,.0625f,cap,padded,seq,staged_count);
   mla_sparse_k6_tensor_values<<<dim3(heads/16,splits,8),128,0,stream>>>(tensor_partial,probabilities,dense,cap,padded);
   mla_sparse_k6_tensor_combine<<<heads,256,0,stream>>>(candidate,tensor_partial,splits);
  } else if(optimized && (mode==2 || mode==3)){
   float* values=mode==3?partial+(size_t)heads*cap:nullptr;
   if(values)mla_sparse_k6_unpack_values<<<cap,256,0,stream>>>(values,staged,staged_ids,dim,cap,seq,staged_count);
   mla_sparse_k6_scores<<<dim3(heads,(cap+255)/256),256,(dim+rope)*4,stream>>>(partial,q,qp,staged,staged+(size_t)cap*dim/16*14,staged_ids,.0625f,dim,rope,cap,seq,staged_count);
   mla_sparse_k6_scored_output<<<heads,threads,(MLA_TILE_SIZE+threads/32)*4,stream>>>(candidate,partial,staged,staged_ids,values,dim,cap,seq,staged_count);
  } else if(optimized && mode==4){mla_sparse_k6_split_warp<<<dim3(heads,splits),256,(dim+rope+128)*4,stream>>>(partial,q,qp,staged,staged+(size_t)cap*dim/16*14,staged_ids,.0625f,dim,rope,cap,seq,staged_count);mla_sparse_k6_combine<<<heads,256,0,stream>>>(candidate,partial,dim,splits);}
  else if(optimized){mla_sparse_k6_split<<<dim3(heads,splits),256,(dim+rope+128)*4,stream>>>(partial,q,qp,staged,staged+(size_t)cap*dim/16*14,staged_ids,.0625f,dim,rope,cap,seq,staged_count);mla_sparse_k6_combine<<<heads,256,0,stream>>>(candidate,partial,dim,splits);}
  else mla_sparse_attention_k6_g<<<heads,threads,(dim+rope+(threads+31)/32+MLA_TILE_SIZE)*4,stream>>>(original,q,qp,staged,staged+(size_t)cap*dim/16*14,staged_ids,.0625f,heads,dim,rope,cap,staged_count);
 };
 cudaEvent_t a,b;C(cudaEventCreate(&a));C(cudaEventCreate(&b));
 for(bool graph:{false,true})for(bool optimized:{false,true}){
  cudaStream_t stream;C(cudaStreamCreate(&stream));cudaGraph_t g;cudaGraphExec_t exec;
  if(graph){C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));run(optimized,stream);C(cudaStreamEndCapture(stream,&g));C(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));}
  auto launch=[&](){if(graph)C(cudaGraphLaunch(exec,stream));else run(optimized,stream);};
  for(int i=0;i<10;i++)launch();C(cudaStreamSynchronize(stream));
  for(int rep=0;rep<5;rep++){
   C(cudaEventRecord(a,stream));for(int i=0;i<100;i++)launch();C(cudaEventRecord(b,stream));C(cudaEventSynchronize(b));float ms;C(cudaEventElapsedTime(&ms,a,b));
   printf("{\"graph\":%s,\"candidate\":%s,\"rep\":%d,\"ms\":%.8f}\n",graph?"true":"false",optimized?"true":"false",rep,ms/100);fflush(stdout);
  }
  std::vector<float>x(heads*dim);C(cudaMemcpy(x.data(),optimized?candidate:original,x.size()*4,cudaMemcpyDeviceToHost));
  std::string file=out+"/"+(optimized?"candidate":"baseline")+(graph?"-graph":"-eager")+".f32";std::ofstream f(file,std::ios::binary);f.write((char*)x.data(),x.size()*4);
  if(graph){C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(g));}C(cudaStreamDestroy(stream));
 }
 C(cudaDeviceSynchronize());return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
