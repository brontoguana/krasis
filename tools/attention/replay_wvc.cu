// Replay the unchanged native BF16 boundary on both paired attention outputs.
#include "wvc-native.cuh"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#define C(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
template<class T>T* load(const std::string& path,size_t n) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||(size_t)f.tellg()!=n*sizeof(T)) throw std::runtime_error("input extent: "+path);
    std::vector<T> data(n);f.seekg(0);f.read((char*)data.data(),n*sizeof(T));
    T* out;C(cudaMalloc(&out,n*sizeof(T)));C(cudaMemcpy(out,data.data(),n*sizeof(T),cudaMemcpyHostToDevice));return out;
}
template<class T>void save(const std::string&path,const T*data,size_t n) {
    std::vector<T> host(n);C(cudaMemcpy(host.data(),data,n*sizeof(T),cudaMemcpyDeviceToHost));
    std::ofstream f(path,std::ios::binary);f.write((char*)host.data(),n*sizeof(T));if(!f)throw std::runtime_error("output write");
}
int main(int argc,char**argv) {try {
    if(argc!=3)throw std::runtime_error("replay_wvc PAIRED_DIRECTORY VALUE_DIM");
    std::string dir=argv[1];int vd=std::stoi(argv[2]);if(vd<=0||vd>512)throw std::runtime_error("value dimension");
    auto weight=load<__nv_bfloat16>(dir+"/w_vc.bf16",64*vd*512);
    __nv_bfloat16* out;float* before;C(cudaMalloc(&out,64*vd*2));C(cudaMalloc(&before,64*vd*4));
    for(std::string name:{"baseline","candidate"}) {
        auto attention=load<float>(dir+"/"+name+"-graph.f32",64*512);
        mla_apply_wvc<<<64,256>>>(out,attention,weight,64,vd,512);
        mla_apply_wvc_before_rounding<<<64,256>>>(before,attention,weight,64,vd,512);
        C(cudaDeviceSynchronize());save(dir+"/wvc-"+name+".bf16",out,64*vd);save(dir+"/wvc-"+name+".f32",before,64*vd);
        C(cudaFree(attention));
    }
    C(cudaFree(weight));C(cudaFree(out));C(cudaFree(before));return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;} }
