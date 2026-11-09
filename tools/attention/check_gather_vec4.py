"""Independent NumPy reconstruction of complete packed K6 gather/query outputs."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('result',type=Path);p.add_argument('rows',type=int);p.add_argument('capacity',type=int);p.add_argument('output',type=Path);a=p.parse_args()
f=lambda name:a.fixture/('fused-0-'+name)
b=np.fromfile(f('ckv-cache.bin'),np.uint8).reshape(-1,32,14);context=b.shape[0]
scale=(b[:,:,0].astype(np.uint32)|(b[:,:,1].astype(np.uint32)<<8))<<16;scale=scale.view(np.float32)
codes=[]
for i in range(16):
 bit=i*6;byte=bit//8;shift=bit%8
 v=b[:,:,2+byte].astype(np.uint16)>>shift
 if shift>2:v|=b[:,:,3+byte].astype(np.uint16)<<(8-shift)
 codes.append((v&63).astype(np.int32)-32)
cache=(np.stack(codes,axis=-1).astype(np.float32)*scale[:,:,None]).reshape(context,512)
ids=np.fromfile(f('selected.i32'),np.int32).reshape(a.rows,a.capacity);positions=np.fromfile(f('positions.i32'),np.int32)
valid=(ids>=0)&(ids<context)&(ids<=positions[:,None]);expected=np.zeros((a.rows,a.capacity,512),np.float32);expected[valid]=cache[ids[valid]]
actual=np.fromfile(a.result/'values.f32',np.float32).reshape(expected.shape);query=np.fromfile(f('q-absorbed.f32'),np.float32);query_actual=np.fromfile(a.result/'query.f32',np.float32)
result=dict(rows=a.rows,capacity=a.capacity,values=actual.size,masked=int((~valid).sum()),gather_exact=bool(np.array_equal(expected.view(np.uint32),actual.view(np.uint32))),query_exact=bool(np.array_equal(query.view(np.uint32),query_actual.view(np.uint32))),max_abs=float(np.max(np.abs(expected-actual))))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(result);assert result['gather_exact']and result['query_exact']
