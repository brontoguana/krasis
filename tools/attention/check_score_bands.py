"""Complete native score/selection checks against independent FP64 equations."""
import argparse,json,os
os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();D=a.directory
bf=lambda p:(np.fromfile(p,'<u2').astype('u4')<<16).view('f4').astype('f8')
receipts=[]
for f in json.loads((D/'fixtures.json').read_text()):
 base=D/'run'/f['label'];path=lambda suffix:Path(str(base)+suffix)
 m,n=f['rows'],f['context'];q=bf(path('-q.bf16')).reshape(m,32,128);w=bf(path('-weights.bf16')).reshape(m,32)
 k=bf(Path(f['capture'])/'indexer-pool-key-prefix.bf16').reshape(-1,128)[:n]
 pos=np.fromfile(path('-positions.i32'),'<i4');limits=np.minimum(np.maximum(pos+1,0)//4,n)
 mask=np.arange(n)[None,:]<limits[:,None]
 expected=np.zeros((m,n),'f8')
 for h in range(32):expected+=w[:,h,None]*np.maximum((q[:,h]@k.T)*np.float32(0.01562500186264515),0)
 scores=[np.fromfile(path(f'-score{i}.f32'),'<f4').reshape(m,n)for i in [0,1]]
 indices=[np.fromfile(path(f'-selected{i}.i32'),'<i4').reshape(m,min(512,n))for i in [0,1]]
 errors=[];changed=int(np.count_nonzero(scores[0][mask]!=scores[1][mask]));cutoffs=[];minimum=None;ties=0
 for i in [0,1]:
  delta=np.abs(scores[i][mask]-expected[mask]);err=float(delta.max(initial=0));assert np.isfinite(scores[i][mask]).all()and err<=2e-4,(f['label'],i,err)
  errors.append(err)
  assert np.array_equal(scores[i][mask],np.fromfile(path(f'-graph-score{i}.f32'),'<f4').reshape(m,n)[mask])
  assert path(f'-selected{i}.i32').read_bytes()==path(f'-graph-selected{i}.i32').read_bytes()
 raws=[np.fromfile(path(f'-raw{i}.i32'),'<i4').reshape(m,2051)for i in [0,1]]
 for i in [0,1]:assert path(f'-raw{i}.i32').read_bytes()==path(f'-graph-raw{i}.i32').read_bytes()
 for r in range(m):
  count=int(limits[r]);order=np.lexsort((np.arange(count),-scores[0][r,:count]));top=min(512,n);want=np.full(top,np.iinfo('i4').max,'i4');want[:min(count,top)]=order[:top]
  for i in [0,1]:assert np.array_equal(want,indices[i][r]),(f['label'],r,i)
  visible=max(int(pos[r])+1,0);want_raw=[]
  for pool in order[:top]:want_raw.extend([int(pool)*4+j for j in range(4)])
  want_raw.extend(range(visible//4*4,visible));want_raw=want_raw[:2051]+[-1]*max(0,2051-len(want_raw))
  for i in [0,1]:assert np.array_equal(raws[i][r],np.array(want_raw,'i4')),(f['label'],r,i,'raw expansion')
  if count>top:cutoffs.append(float(scores[0][r,order[top-1]]-scores[0][r,order[top]]))
  if count>1:
   gaps=np.diff(-scores[0][r,order]);ties+=int(np.count_nonzero(gaps==0));v=float(gaps.min());minimum=v if minimum is None else min(minimum,v)
 receipts.append(dict(label=f['label'],rows=m,context=n,valid_scores=int(mask.sum()),different_scores=changed,different_selections=int(np.count_nonzero(indices[0]!=indices[1])),raw_selection_values=m*2051,complete_raw_expansion=True,independent_max_abs=errors,threshold=2e-4,minimum_cutoff_margin=min(cutoffs,default=None),minimum_adjacent_margin=minimum,exact_adjacent_ties=ties,complete_graph_eager_match=True,invalid_rows=int(np.count_nonzero(limits==0))))
 print(json.dumps(receipts[-1]),flush=True)
(D/'equations.json').write_text(json.dumps(receipts,indent=2)+'\n')
