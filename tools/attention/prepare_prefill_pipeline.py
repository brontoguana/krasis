"""Extract native replay kernels and independent real-operand K6 fixtures."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_replay import unpack

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('capture',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args();a.out.mkdir(parents=True)
s=(a.source/'src/cuda/prefill_kernels.cu').read_text()
header='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cstdint>\n__device__ inline float bf16_to_float(__nv_bfloat16 x){return __bfloat162float(x);}\n'
lo=s.index('__device__ inline int mla_prefill_unpack_k6(')
hi=s.index('#include "mla_prefill_mask_fusion.cuh"',lo)
header+=s[lo:hi]
lo=s.rfind('template',0,s.index('__device__ inline void mla_prefill_gather_query_structured_f32_body'))
hi=s.index('/* Reorder the FP32 sparse-attention result',lo)
# K4 is outside this replay's scope; native K6 template instantiation eliminates it.
header+='__device__ inline float mla_prefill_load_k4_value(const unsigned char*,int,int,int){return 0;}\n'+s[lo:hi]
(a.out/'gather_baseline.cuh').write_text(header)
base=a.capture/'fused-0-'
cache=Path(str(base)+'ckv-cache.bin');context=cache.stat().st_size//448
assert context==18000
q=np.memmap(str(base)+'q-absorbed.f32',dtype='<f4',mode='r',shape=(context,64,512))
indices=np.memmap(str(base)+'selected.i32',dtype='<i4',mode='r',shape=(context,2051))
positions=np.fromfile(str(base)+'positions.i32',dtype='<i4')
kv=unpack(cache)
selected_rows=[0,1,31,32,511,4095,8191,17999]
for capacity in [1,17,129,2051,4096]:
    root=a.out/f'c{capacity}';root.mkdir(exist_ok=True)
    query=np.array(q[selected_rows]);ids=np.full((len(query),capacity),-1,dtype='<i4')
    count=min(capacity,2051);ids[:,:count]=indices[selected_rows,:count]
    pos=np.array(positions[selected_rows],dtype='<i4')
    # Include all-masked, empty, repeated, future and out-of-context selections.
    ids[0,:]=-1;pos[1]=-1
    ids[2,:]=min(int(pos[2]),context-1)
    ids[3,::2]=context+1
    ids[4,::3]=-2
    ids[5,::5]=context-1
    expected=[]
    for qi,ii,pp in zip(query,ids,pos):
        valid=(ii>=0)&(ii<min(max(int(pp)+1,0),context))
        keys=kv[ii[valid]]
        if len(keys):
            scores=qi.astype('f8')@keys.T*0.0625
            weights=np.exp(scores-scores.max(axis=1,keepdims=True))
            value=(weights@keys)/weights.sum(axis=1,keepdims=True)
        else:value=np.zeros((64,512),dtype='f8')
        expected.append(value)
    np.stack(expected).astype('<f8').tofile(root/'oracle.f64')
    np.stack(expected).astype('<f4').tofile(root/'fused-0-baseline-output.f32')
    query.astype('<f4').tofile(root/'fused-0-q-absorbed.f32')
    ids.tofile(root/'fused-0-selected.i32');pos.tofile(root/'fused-0-positions.i32')
    # A regular link preserves cache bytes without duplicating the artifact.
    import os
    os.link(cache,root/'fused-0-ckv-cache.bin')
    (root/'descriptor.json').write_text(json.dumps(dict(capacity=capacity,rows=len(query),context=context,source_rows=selected_rows,scale=0.0625,threshold_abs=2e-4),indent=2)+'\n')
