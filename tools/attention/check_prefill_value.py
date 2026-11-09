"""Independent float64 selected-K6 attention equations for complete saved rows."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('result',type=Path);p.add_argument('rows',type=int);p.add_argument('capacity',type=int);p.add_argument('output',type=Path);a=p.parse_args()
f=lambda name:a.capture/('fused-0-'+name)
b=np.fromfile(f('ckv-cache.bin'),np.uint8).reshape(-1,32,14);context=b.shape[0]
scale=((b[:,:,0].astype(np.uint32)|(b[:,:,1].astype(np.uint32)<<8))<<16).view(np.float32)
codes=[]
for i in range(16):
 bit=i*6;byte=bit//8;shift=bit%8;v=b[:,:,2+byte].astype(np.uint16)>>shift
 if shift>2:v|=b[:,:,3+byte].astype(np.uint16)<<(8-shift)
 codes.append((v&63).astype(np.int32)-32)
cache=(np.stack(codes,axis=-1).astype(np.float32)*scale[:,:,None]).reshape(context,512)
q=np.memmap(f('q-absorbed.f32'),dtype=np.float32,mode='r',shape=(a.rows,64,512));ids=np.memmap(f('selected.i32'),dtype=np.int32,mode='r',shape=(a.rows,a.capacity));positions=np.fromfile(f('positions.i32'),np.int32)
results=[]
for row in sorted(set([0,1,17,31,32,a.rows-1])):
 if row>=a.rows:continue
 index=np.asarray(ids[row]);valid=(index>=0)&(index<context)&(index<=positions[row]);v=np.zeros((a.capacity,512),np.float64);v[valid]=cache[index[valid]]
 scores=np.asarray(q[row],dtype=np.float64)@v.T*.0625;scores[:,~valid]=-np.inf
 maximum=scores.max(axis=1,keepdims=True);prob=np.exp(scores-maximum);prob=np.divide(prob,prob.sum(axis=1,keepdims=True),where=np.isfinite(maximum),out=np.zeros_like(prob));expected=prob@v
 for mode in [5,7]:
  for execution in ['eager','graph']:
   path=a.result/f'value-mode{mode}-{execution}-row{row}.f32';actual=np.fromfile(path,np.float32).reshape(64,512);error=np.abs(actual-expected);finite=bool(np.isfinite(actual).all());bad=int(np.count_nonzero(error>2e-4))
   result=dict(row=row,mode=mode,execution=execution,values=actual.size,finite=finite,over_2e_4=bad,max_abs=float(error.max()))
   results.append(result);print(result,flush=True)
   assert finite and bad==0
 for mode in [5,7]:
  x=np.fromfile(a.result/f'value-mode{mode}-eager-row{row}.f32',np.uint32);y=np.fromfile(a.result/f'value-mode{mode}-graph-row{row}.f32',np.uint32);assert np.array_equal(x,y),'graph arithmetic drift'
a.output.write_text(json.dumps(dict(scope='Independent float64 equations from exact K6 bytes and actual selected indices; complete outputs for six rows, threshold unchanged.',results=results),indent=2)+'\n')
