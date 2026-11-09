"""Independent complete FP64 packed equations and eager/graph comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from prepare_replay import unpack
p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('run',type=Path);p.add_argument('result',type=Path);a=p.parse_args()
d=json.loads((a.fixture/'descriptor.json').read_text());rows,cap,context=[d[k]for k in ['rows','capacity','context']]
q=np.fromfile(a.fixture/'fused-0-q-absorbed.f32',dtype='<f4').reshape(rows,64,512)
ids=np.fromfile(a.fixture/'fused-0-selected.i32',dtype='<i4').reshape(rows,cap)
pos=np.fromfile(a.fixture/'fused-0-positions.i32',dtype='<i4')
cache=unpack(a.fixture/'fused-0-ckv-cache.bin');assert cache.shape==(context,512)
scores=np.fromfile(a.run/'scores.f32',dtype='<f4').reshape(rows,64,cap)
probs=np.fromfile(a.run/'probs.f32',dtype='<f4').reshape(rows,64,cap)
control=np.fromfile(a.run/'accepted.f32',dtype='<f4').reshape(rows,64,512)
candidate=np.fromfile(a.run/'candidate.f32',dtype='<f4').reshape(rows,64,512)
cs=np.fromfile(a.run/'candidate-scores.f32',dtype='<f4').reshape(rows,64,cap)
cp=np.fromfile(a.run/'candidate-probs.f32',dtype='<f4').reshape(rows,64,cap)
checks={};maximum={};counts={}
def compare(name,x,y,threshold=2e-4):
 e=np.abs(x.astype('f8')-y);finite=np.isfinite(e).all();bad=int(np.count_nonzero(e>threshold));checks[name]=bool(finite and bad==0);maximum[name]=float(e.max(initial=0));counts[name]=int(e.size)
for row in range(rows):
 valid=(ids[row]>=0)&(ids[row]<min(max(int(pos[row])+1,0),context))
 keys=np.zeros((cap,512),dtype='f8');keys[valid]=cache[ids[row,valid]]
 exact=q[row].astype('f8')@keys.T*0.0625
 compare(f'scores_{row}',scores[row],exact)
 compare(f'candidate_scores_{row}',cs[row],exact)
 masked=np.where(valid[None,:],exact,-np.inf)
 if valid.any():
  weights=np.exp(masked-masked.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True)
 else:weights=np.zeros_like(masked)
 compare(f'probs_{row}',probs[row],weights)
 compare(f'candidate_probs_{row}',cp[row],weights)
 compare(f'accepted_{row}',control[row],weights@keys)
 compare(f'candidate_{row}',candidate[row],weights@keys)
 compare(f'packed_value_actual_probs_{row}',candidate[row],cp[row].astype('f8')@keys)
bitwise_control=bool(np.array_equal(control.view('u4'),candidate.view('u4')))
compare('complete_control_threshold',candidate,control)
for arm,expected in [('accepted',control),('candidate',candidate)]:
 graph=np.fromfile(a.run/f'graph-{arm}.f32',dtype='<f4').reshape(-1,64,512)
 checks[f'graph_{arm}_bitwise']=bool(np.array_equal(graph.view('u4'),expected[-len(graph):].view('u4')))
result=dict(bitwise_control=bitwise_control,checks=checks,all_passed=all(checks.values()),maximum_abs=maximum,values=counts,rows=rows,capacity=cap,context=context,threshold_abs=2e-4,scope='Independent FP64 packed-scale/codes, all scores/probabilities/outputs; value also reconstructed using actual native probabilities. Operator evidence only.')
a.result.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(all_passed=result['all_passed'],failed=[k for k,v in checks.items()if not v],maximum_abs=max(maximum.values()),rows=rows,capacity=cap)))
assert result['all_passed']
