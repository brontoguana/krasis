"""Generate row-batched diagnostic wrappers from the measured native kernels."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir()
s=(a.source/'src/cuda/mla_sparse_tensor.cuh').read_text().replace('mla_sparse_k6_tensor_','mla_batch_')
s=s.replace(''' using namespace mla_tensor;
 const int head=blockIdx.x*16,slot=''', ''' const int batch_row=blockIdx.z;
 scores+=(long long)batch_row*64*padded_capacity;q+=(long long)batch_row*64*512;values+=(long long)batch_row*padded_capacity*512;
 using namespace mla_tensor;
 const int head=blockIdx.x*16,slot=''',1)
s=s.replace(''' const int seq=sequence_pointer?*sequence_pointer:sequence_length;''',''' const int batch_row=blockIdx.y;
 probabilities+=(long long)batch_row*64*padded_capacity;scores+=(long long)batch_row*64*padded_capacity;indices+=(long long)batch_row*capacity;
 const int seq=sequence_pointer?*sequence_pointer:sequence_length;''',1)
s=s.replace(''' using namespace mla_tensor;
 const int head=blockIdx.x*16,split=''',''' const int batch_row=blockIdx.z/8;
 partial+=(long long)batch_row*64*((capacity+127)/128)*512;
 probabilities+=(long long)batch_row*64*padded_capacity;values+=(long long)batch_row*padded_capacity*512;
 using namespace mla_tensor;
 const int head=blockIdx.x*16,split=''',1).replace('column=blockIdx.z*64+','column=(blockIdx.z%8)*64+',1)
s=s.replace(''' const int h=blockIdx.x;
 for(int d=threadIdx.x;d<512;d+=blockDim.x){''',''' const int batch_row=blockIdx.y;
 output+=(long long)batch_row*64*512;partial+=(long long)batch_row*64*splits*512;
 const int h=blockIdx.x;
 for(int d=threadIdx.x;d<512;d+=blockDim.x){''',1)
s+='''
extern "C" __global__ void mla_batch_unpack(
 float* values,int* masks,const unsigned char* cache,const int* indices,
 const int* positions,int capacity,int padded_capacity,int context
){
 const int row=blockIdx.y,slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
 if(slot>=padded_capacity)return;
 const int pos=slot<capacity?indices[(long long)row*capacity+slot]:-1;
 const bool valid=pos>=0 && pos<context && pos<=positions[row];
 if(lane==0 && slot<capacity)masks[(long long)row*capacity+slot]=valid?slot:-1;
 for(int d=lane;d<512;d+=32)values[((long long)row*padded_capacity+slot)*512+d]=valid?mla_load_k6_value(cache,pos,512,d):0.0f;
}
'''
(a.out/'tensor-batch.cuh').write_text(s)
