// Complete-output qualification on captured prefill queries and selections.
#include "attention_baseline.cuh"
#include "tensor-batch.cuh"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#define C(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
std::vector<char> read(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||(size_t)f.tellg()!=n)throw std::runtime_error("size "+p);std::vector<char>x(n);f.seekg(0);f.read(x.data(),n);return x;}
template<class T>T* upload(const std::vector<char>&x){T*p;C(cudaMalloc(&p,x.size()));C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(size_t n){T*p;C(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("replay_batch CAPTURE START ROWS CONTEXT CAPACITY OUT MAX_TILE");
 std::string prefix=std::string(argv[1])+"/fused-"+argv[2]+"-",out=argv[6];
 int rows=std::stoi(argv[3]),context=std::stoi(argv[4]),cap=std::stoi(argv[5]),max_tile=std::stoi(argv[7]),padded=(cap+15)/16*16,splits=(cap+127)/128;
 size_t elements=(size_t)rows*64*512;
 auto q=upload<float>(read(prefix+"q-absorbed.f32",elements*4));auto ids=upload<int>(read(prefix+"selected.i32",(size_t)rows*cap*4));auto positions=upload<int>(read(prefix+"positions.i32",rows*4));auto cache=upload<unsigned char>(read(prefix+"ckv-cache.bin",(size_t)context*448));
 auto baseline=read(prefix+"baseline-output.f32",elements*4);auto output=alloc<float>(elements);
 auto dense=alloc<float>((size_t)max_tile*padded*512);auto masks=alloc<int>((size_t)max_tile*cap);auto scores=alloc<float>((size_t)max_tile*64*padded);auto probs=alloc<float>((size_t)max_tile*64*padded);auto partial=alloc<float>((size_t)max_tile*64*splits*512);
 cudaEvent_t a,b;C(cudaEventCreate(&a));C(cudaEventCreate(&b));
 for(int tile:{8,16,32}){if(tile>max_tile)continue;
  for(int rep=0;rep<3;rep++){
   C(cudaEventRecord(a));
   for(int start=0;start<rows;start+=tile){int n=std::min(tile,rows-start);
    mla_batch_unpack<<<dim3((padded+7)/8,n),256>>>(dense,masks,cache,ids+(size_t)start*cap,positions+start,cap,padded,context);
    mla_batch_scores<<<dim3(4,(padded+63)/64,n),128>>>(scores,q+(size_t)start*64*512,dense,padded);
    mla_batch_softmax<<<dim3(64,n),256,(padded+8)*4>>>(probs,scores,masks,.0625f,cap,padded,cap,nullptr);
    mla_batch_values<<<dim3(4,splits,8*n),128>>>(partial,probs,dense,cap,padded);
    mla_batch_combine<<<dim3(64,n),256>>>(output+(size_t)start*64*512,partial,splits);
   }
   C(cudaEventRecord(b));C(cudaEventSynchronize(b));float ms;C(cudaEventElapsedTime(&ms,a,b));printf("{\"tile\":%d,\"rep\":%d,\"ms\":%.6f}\n",tile,rep,ms);fflush(stdout);
  }
  std::vector<float>actual(elements);C(cudaMemcpy(actual.data(),output,elements*4,cudaMemcpyDeviceToHost));size_t bad=0,nonfinite=0,first=elements;double maxerr=0,ss=0;
  const float* reference=(const float*)baseline.data();
  for(size_t i=0;i<elements;i++){double e=std::abs((double)actual[i]-reference[i]);if(!std::isfinite(actual[i])||!std::isfinite(reference[i]))nonfinite++;if(e>2e-4){bad++;if(first==elements)first=i;}maxerr=std::max(maxerr,e);ss+=e*e;}
  printf("{\"tile\":%d,\"values\":%zu,\"over_2e_4\":%zu,\"nonfinite\":%zu,\"max_abs\":%.9g,\"rms\":%.9g,\"first_bad_index\":%zu}\n",tile,elements,bad,nonfinite,maxerr,std::sqrt(ss/elements),first);fflush(stdout);
  if(bad||nonfinite){std::ofstream f(out+"/failed-tile"+std::to_string(tile)+".f32",std::ios::binary);f.write((char*)actual.data(),elements*4);return 2;}
 }
 return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
