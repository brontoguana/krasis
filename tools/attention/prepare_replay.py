"""Extract baseline kernels and reconstruct independent packed-cache equations."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def unpack(path):
 raw=np.fromfile(path,dtype='u1').reshape(-1,32,14)
 scale=(raw[:,:,:2].copy().view('<u2').reshape(-1,32).astype('<u4')<<16).view('<f4')
 codes=np.empty((len(raw),32,16),dtype='i4')
 for j in range(16):
  bit=j*6;byte=bit//8;shift=bit%8
  value=raw[:,:,2+byte].astype('u4')>>shift
  if shift>2:value|=raw[:,:,3+byte].astype('u4')<<(8-shift)
  codes[:,:,j]=(value&63).astype('i4')-32
 return (scale[:,:,None].astype('f8')*codes).reshape(-1,512)

def oracle(directory,context,scale=0.0625):
 q=np.fromfile(directory/'q.f32',dtype='<f4').reshape(64,512)
 ids=np.fromfile(directory/'indices.i32',dtype='<i4')[:context]
 cache=unpack(directory/'cache.bin');assert 0<=context<=len(cache)
 valid=(ids>=0)&(ids<context);kv=cache[ids[valid]]
 if len(kv):
  scores=q.astype('f8')@kv.T*scale
  weights=np.exp(scores-scores.max(axis=1,keepdims=True))
  expected=(weights@kv)/weights.sum(axis=1,keepdims=True)
 else:expected=np.zeros((64,512))
 expected.astype('<f8').tofile(directory/'oracle.f64')

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();O=a.out;O.mkdir()
 s=(a.source/'src/cuda/decode_kernels.cu').read_text()
 x='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cstdint>\n#define MLA_TILE_SIZE 4096\n'
 x+=s[s.index('__device__ inline void mla_pack_k6_16'):s.index('// Write compressed KV + rope PE to compact signed-INT6')]
 x+=s[s.index('__device__ __forceinline__ void mla_sparse_attention_k6_body'):s.index('// MLA de-interleave kernel:')].replace('#include "mla_sparse_split.cuh"','').replace('#include "mla_sparse_tensor.cuh"','')
 (O/'attention_baseline.cuh').write_text(x)
 D=a.capture;desc=json.loads((D/'descriptor.json').read_text());sel=json.loads((D/'selection.json').read_text())
 assert desc['kv_format']==7 and desc['cache_dim']==512 and desc['heads']==64 and desc['rope']==0 and desc['nope']==256
 for src,dst in [('q_absorbed.f32','q.f32'),('ckv-prefix.raw','cache.bin'),('selected.i32','indices.i32')]:
  (O/dst).write_bytes((D/src).read_bytes())
 context=(O/'cache.bin').stat().st_size//448;oracle(O,context)
 (O/'input.json').write_text(json.dumps(dict(descriptor=desc,selection=sel,context=context,capacity=(O/'indices.i32').stat().st_size//4,hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in O.iterdir()if f.is_file()}),indent=2))
if __name__=='__main__':main()
