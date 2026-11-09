#include "hqq-load.cuh"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
std::vector<char> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);std::vector<char>x(f.tellg());f.seekg(0);f.read(x.data(),x.size());return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,x.size()));C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
int main(int argc,char**argv){try{
 if(argc!=12)throw std::runtime_error("replay_hqq_load DIR PREFIX INPUT ROWS COLS GROUP PACKED_STRIDE SCALE_STRIDE ZERO_STRIDE WARPS OUTPUT");
 std::string in=argv[1],prefix=in+"/"+argv[2],out=argv[11];int rows=std::stoi(argv[4]),cols=std::stoi(argv[5]),group=std::stoi(argv[6]),ps=std::stoi(argv[7]),ss=std::stoi(argv[8]),zs=std::stoi(argv[9]),warps=std::stoi(argv[10]);
 if(cols%128 || ps%4)throw std::runtime_error("cooperative-load geometry unsupported");
 auto packed=upload<unsigned char>(read(prefix+".packed"));auto scales=upload<float>(read(prefix+".scales.f32"));auto zeros=upload<float>(read(prefix+".zeros.f32"));auto input=upload<unsigned short>(read(in+"/"+argv[3]));
 unsigned short *a,*b;C(cudaMalloc(&a,rows*2));C(cudaMalloc(&b,rows*2));cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
 auto run=[&](bool candidate,cudaStream_t stream){
  if(candidate)hqq6_decode_gemv_bf16_vec4_warp_load<<<(rows+warps-1)/warps,warps*32,cols*2,stream>>>(packed,scales,zeros,input,b,rows,cols,group,ps,ss,zs);
  else hqq6_decode_gemv_bf16_vec4<<<(rows+warps-1)/warps,warps*32,cols*2,stream>>>(packed,scales,zeros,input,a,rows,cols,group,ps,ss,zs);
 };
 for(bool graph:{false,true})for(bool candidate:{false,true}){
  cudaStream_t stream;C(cudaStreamCreate(&stream));cudaGraph_t g;cudaGraphExec_t exec;
  if(graph){C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));run(candidate,stream);C(cudaStreamEndCapture(stream,&g));C(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));}
  auto launch=[&](){if(graph)C(cudaGraphLaunch(exec,stream));else run(candidate,stream);};
  for(int i=0;i<20;i++)launch();C(cudaStreamSynchronize(stream));
  for(int rep=0;rep<5;rep++){
   C(cudaEventRecord(begin,stream));for(int i=0;i<100;i++)launch();C(cudaEventRecord(end,stream));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));
   printf("{\"graph\":%s,\"candidate\":%s,\"rep\":%d,\"ms\":%.8f}\n",graph?"true":"false",candidate?"true":"false",rep,ms/100);fflush(stdout);
  }
  std::vector<unsigned short>x(rows);C(cudaMemcpy(x.data(),candidate?b:a,rows*2,cudaMemcpyDeviceToHost));std::ofstream f(out+"/"+(candidate?"candidate":"baseline")+(graph?"-graph":"-eager")+".bf16",std::ios::binary);f.write((char*)x.data(),rows*2);
  if(graph){C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(g));}C(cudaStreamDestroy(stream));
 }
 std::vector<unsigned short>x(rows),y(rows);C(cudaMemcpy(x.data(),a,rows*2,cudaMemcpyDeviceToHost));C(cudaMemcpy(y.data(),b,rows*2,cudaMemcpyDeviceToHost));size_t bad=0;for(int i=0;i<rows;i++)bad+=x[i]!=y[i];printf("{\"complete_output_rows\":%d,\"bit_differences\":%zu}\n",rows,bad);return bad?2:0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
