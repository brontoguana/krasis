// Exact native selection replay, including dynamic graph history and ties.
#include "index_kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#define C(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
template<class T> T* allocation(size_t n) { T* p; C(cudaMalloc(&p,n*sizeof(T))); return p; }
int main(int argc,char**argv) { try {
    if(argc!=2) throw std::runtime_error("replay_index INPUT");
    std::ifstream f(std::string(argv[1])+"/scores.f32",std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("scores missing");
    std::vector<float> actual((size_t)f.tellg()/4); f.seekg(0); f.read((char*)actual.data(),actual.size()*4);
    if(actual.empty()) throw std::runtime_error("empty scores");
    constexpr int capacity=32768,topk=512,width=1024,runs=capacity/width;
    auto scores=allocation<float>(capacity),sa=allocation<float>(runs*topk),sb=allocation<float>(runs*topk);
    auto ia=allocation<int>(runs*topk),ib=allocation<int>(runs*topk),out=allocation<int>(topk),count=allocation<int>(1);
    cudaStream_t stream; C(cudaStreamCreate(&stream));
    auto enqueue=[&](bool radix) {
        if(radix) dsa_topk_radix_sort_chunks_g<<<runs,256,0,stream>>>(scores,sa,ia,out,count,capacity,topk,width);
        else dsa_topk_sort_chunks_g<<<runs,256,width*8,stream>>>(scores,sa,ia,out,count,capacity,topk,width);
        int nr=runs,pass=0; bool first=true;
        while(nr>1) {
            int next=(nr+1)/2;
            auto os=first?sb:sa; auto oi=next==1?out:(first?ib:ia);
            if(radix) dsa_topk_linear_merge_runs_g<<<next,256,0,stream>>>(first?sa:sb,first?ia:ib,os,oi,out,count,capacity,topk,width,pass);
            else dsa_topk_merge_runs_g<<<next,256,width*8,stream>>>(first?sa:sb,first?ia:ib,os,oi,out,count,capacity,topk,width,pass);
            nr=next;first=!first;pass++;
        }
    };
    cudaGraph_t graph[2]; cudaGraphExec_t exec[2];
    for(int mode=0;mode<2;mode++) {
        C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); enqueue(mode);
        C(cudaStreamEndCapture(stream,&graph[mode])); C(cudaGraphInstantiate(&exec[mode],graph[mode],nullptr,nullptr,0));
    }
    for(int fixture=0;fixture<3;fixture++) {
        std::vector<float> input(capacity);
        for(int i=0;i<capacity;i++) input[i]=fixture==0?actual[i%actual.size()]:(fixture==1?float(i%17):((i&1)?-0.0f:0.0f));
        C(cudaMemcpy(scores,input.data(),capacity*4,cudaMemcpyHostToDevice));
        for(int context: {0,1,3,511,512,513,1023,1024,1025,16387,32767,32768}) {
            std::vector<int> expected(capacity); std::iota(expected.begin(),expected.end(),0); expected.resize(context);
            std::sort(expected.begin(),expected.end(),[&](int a,int b){if(input[a]!=input[b])return input[a]>input[b];return a<b;});
            expected.resize(std::min(context,topk)); expected.resize(topk,INT_MAX);
            C(cudaMemcpy(count,&context,4,cudaMemcpyHostToDevice));
            for(int mode=0;mode<2;mode++) for(int captured=0;captured<2;captured++) {
                if(captured) C(cudaGraphLaunch(exec[mode],stream)); else enqueue(mode);
                C(cudaStreamSynchronize(stream)); std::vector<int> got(topk); C(cudaMemcpy(got.data(),out,topk*4,cudaMemcpyDeviceToHost));
                if(got!=expected) throw std::runtime_error("complete top-k mismatch fixture="+std::to_string(fixture)+" context="+std::to_string(context)+" mode="+std::to_string(mode));
            }
        }
    }
    int context=actual.size(); if(context>capacity) throw std::runtime_error("capture exceeds replay capacity");
    C(cudaMemcpy(count,&context,4,cudaMemcpyHostToDevice)); C(cudaMemcpy(scores,actual.data(),context*4,cudaMemcpyHostToDevice));
    for(int mode=0;mode<2;mode++) {
        for(int i=0;i<10;i++) C(cudaGraphLaunch(exec[mode],stream));
        cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
        C(cudaEventRecord(begin,stream)); for(int i=0;i<100;i++) C(cudaGraphLaunch(exec[mode],stream));
        C(cudaEventRecord(end,stream));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));
        printf("{\"mode\":%d,\"context\":%d,\"topk\":512,\"mean_ms\":%.8f,\"complete_checks_pass\":true}\n",mode,context,ms/100);
        C(cudaEventDestroy(begin));C(cudaEventDestroy(end));
    }
    for(int mode=0;mode<2;mode++){C(cudaGraphExecDestroy(exec[mode]));C(cudaGraphDestroy(graph[mode]));}
    C(cudaStreamDestroy(stream));
    for(void* p:{(void*)scores,(void*)sa,(void*)sb,(void*)ia,(void*)ib,(void*)out,(void*)count})C(cudaFree(p));
    return 0;
} catch(const std::exception&e) { fprintf(stderr,"%s\n",e.what()); return 1; } }
