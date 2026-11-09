// Preserve each original FP32 FMA chain while reusing weights across rows.
#include "absorb-baseline.cuh"
#include "absorb-coherent.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#define C(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
template<int Tile> __global__ void ordered_absorb(float* output,const __nv_bfloat16* q,const __nv_bfloat16* w,int rows) {
    const int head=blockIdx.z,start=blockIdx.y*Tile,dim=blockIdx.x*128+threadIdx.x;
    __shared__ __nv_bfloat16 query[Tile][256];
    for(int i=threadIdx.x;i<Tile*256;i+=128) {
        const int r=i/256,k=i%256;
        query[r][k]=start+r<rows?q[((long long)(start+r)*64+head)*256+k]:__float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    float acc[Tile]={};
    #pragma unroll 1
    for(int k=0;k<256;k++) {
        const float weight=bf16_to_float(w[((long long)head*256+k)*512+dim]);
        #pragma unroll
        for(int r=0;r<Tile;r++) acc[r]+=bf16_to_float(query[r][k])*weight;
    }
    #pragma unroll
    for(int r=0;r<Tile;r++) if(start+r<rows) output[((long long)(start+r)*64+head)*512+dim]=acc[r];
}
std::vector<char> read_prefix(const std::string& p,size_t n) {
    std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error(p);
    std::vector<char> x(n); f.read(x.data(),n); if(!f) throw std::runtime_error("short input: "+p); return x;
}
template<class T>T* upload(const std::vector<char>& x) { T* p; C(cudaMalloc(&p,x.size())); C(cudaMemcpy(p,x.data(),x.size(),cudaMemcpyHostToDevice)); return p; }
int main(int argc,char**argv) { try {
    if(argc!=4) throw std::runtime_error("replay_absorb_ordered CAPTURE WKC ROWS");
    int rows=std::stoi(argv[3]); if(rows<1||rows>32768) throw std::runtime_error("row extent");
    std::string prefix=std::string(argv[1])+"/fused-0-";size_t n=(size_t)rows*64*512;
    auto q=upload<__nv_bfloat16>(read_prefix(prefix+"packed-q.bf16",(size_t)rows*64*256*2));
    auto w=upload<__nv_bfloat16>(read_prefix(argv[2],64*256*512*2));
    auto expected=read_prefix(prefix+"q-absorbed.f32",n*4); float* out; C(cudaMalloc(&out,n*4));
    cudaStream_t stream; C(cudaStreamCreate(&stream));
    for(int mode:{0,1,8,16,32}) for(bool graph:{false,true}) {
        auto enqueue=[&]() {
            if(mode==0) mla_prefill_absorb_wkc_kernel<<<dim3(64,rows),256,0,stream>>>(out,q,w,rows,64,256,256,512);
            if(mode==1) mla_prefill_absorb_wkc_coherent<<<dim3(rows,64),256,0,stream>>>(out,q,w,rows,64,256,256,512);
            if(mode==8) ordered_absorb<8><<<dim3(4,(rows+7)/8,64),128,0,stream>>>(out,q,w,rows);
            if(mode==16) ordered_absorb<16><<<dim3(4,(rows+15)/16,64),128,0,stream>>>(out,q,w,rows);
            if(mode==32) ordered_absorb<32><<<dim3(4,(rows+31)/32,64),128,0,stream>>>(out,q,w,rows);
        };
        cudaGraph_t g;cudaGraphExec_t exec;
        if(graph){C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));enqueue();C(cudaStreamEndCapture(stream,&g));C(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));}
        auto launch=[&](){if(graph)C(cudaGraphLaunch(exec,stream));else enqueue();};
        launch();C(cudaStreamSynchronize(stream));
        for(int rep=0;rep<5;rep++) {
            cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
            C(cudaEventRecord(begin,stream));launch();C(cudaEventRecord(end,stream));C(cudaEventSynchronize(end));
            float ms;C(cudaEventElapsedTime(&ms,begin,end));
            printf("{\"mode\":%d,\"graph\":%s,\"rows\":%d,\"rep\":%d,\"ms\":%.8f}\n",mode,graph?"true":"false",rows,rep,ms);fflush(stdout);
            C(cudaEventDestroy(begin));C(cudaEventDestroy(end));
        }
        std::vector<char> actual(n*4);C(cudaMemcpy(actual.data(),out,n*4,cudaMemcpyDeviceToHost));
        bool equal=std::memcmp(actual.data(),expected.data(),n*4)==0;
        printf("{\"mode\":%d,\"graph\":%s,\"rows\":%d,\"values\":%zu,\"bit_exact\":%s}\n",mode,graph?"true":"false",rows,n,equal?"true":"false");fflush(stdout);
        if(!equal) throw std::runtime_error("complete ordered absorption mismatch");
        if(graph){C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(g));}
    }
    C(cudaStreamDestroy(stream));C(cudaFree(q));C(cudaFree(w));C(cudaFree(out));return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;} }
