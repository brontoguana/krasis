// Offline M64/N256/K64 thread-group screen using existing licensed headers.
// Original 256 threads and warp mapping; only the candidate register cap changes.
#include "marlin_vendor_common.h"
#include "moe/kernel.h"
#include "moe/marlin_template.h"
#include "capped_marlin_template.h"

template<int N> auto selected() {
  if constexpr (N==256) return device::marlin_moe::Marlin<nv_bfloat16,host::kU4B8.id(),host::kBFloat16.id(),256,4,16,4,false,4,8,false>;
  else return device::marlin_moe_capped::Marlin<nv_bfloat16,host::kU4B8.id(),host::kBFloat16.id(),256,4,16,4,false,4,8,false>;
}
extern "C" int routed_resources(int n, int* out) {
  if(n!=192 && n!=256) return -1;
  auto fn=n==256?selected<256>():selected<192>();
  cudaFuncAttributes attr; auto err=cudaFuncGetAttributes(&attr,fn);
  if(err!=cudaSuccess)return err;
  // Existing homogeneous group128 cache-size equation, plus1024-byte guard.
  int shared=71680;
  err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
  if(err!=cudaSuccess)return err;
  int active=0;err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,shared);
  if(err!=cudaSuccess)return err;
  out[0]=attr.numRegs;out[1]=attr.sharedSizeBytes;out[2]=attr.localSizeBytes;out[3]=shared;out[4]=active;
  return 0;
}
extern "C" int routed_n(int n, const void* a, void* c, void* tmp,
 const void* ids,const void* experts,const void* count,const void* weights,
 void* locks,const void* wp,const void* sp,int m,int width,int k,int sms,void* stream) {
  if((n!=192 && n!=256)||m<=0||width%256||k%128||sms<=0)return -1;
  auto fn=n==256?selected<256>():selected<192>();
  int shared=71680;
  auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
  if(e!=cudaSuccess)return e;
  fn<<<sms,256,shared,reinterpret_cast<cudaStream_t>(stream)>>>(
   (const int4*)a,nullptr,(int4*)c,(int4*)tmp,nullptr,nullptr,nullptr,nullptr,nullptr,
   (const int*)ids,(const int*)experts,(const int*)count,(const float*)weights,
   1,false,false,k/128,m,width,k,(int*)locks,false,false,true,shared,
   (const int64_t*)wp,(const int64_t*)sp);
  return cudaGetLastError();
}
