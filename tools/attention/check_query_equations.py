"""Independent float64 BF16 query equations and typed-boundary diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('fixtures',type=Path);p.add_argument('results',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
def f32(u):return (np.asarray(u,dtype='u4')<<np.uint32(16)).view('f4')
def bf16(x):
 # Direct float64 -> BF16 RNE: avoid double rounding through float32.
 magnitude=np.abs(x)
 lower_bits=magnitude.astype('f4').view('u4')>>16
 lower=(lower_bits<<16).view('f4').astype('f8')
 lower_bits=lower_bits-(lower>magnitude).astype('u4')
 lower=(lower_bits<<16).view('f4').astype('f8')
 upper=((lower_bits+1)<<16).view('f4').astype('f8')
 dl,du=magnitude-lower,upper-magnitude
 bits=lower_bits+((du<dl)|((du==dl)&((lower_bits&1)!=0))).astype('u4')
 return (bits | (np.signbit(x).astype('u4')<<15)).astype('u2')

rows=[]
for path,count,label in json.loads(a.fixtures.read_text()):
 d=Path(path);w=f32(np.fromfile(d/'indexer-weight-wq_b.bf16',dtype='<u2')).reshape(4096,1536).astype('f8')
 x=f32(np.fromfile(d/'q_norm-chunk-prefix.bf16',dtype='<u2')).reshape(-1,1536)
 assert np.isfinite(x).all() and np.isfinite(w).all()
 base=np.memmap(a.results/(label+'-mode0.bf16'),mode='r',dtype='<u2',shape=(count,4096))
 cand=np.memmap(a.results/(label+'-mode1.bf16'),mode='r',dtype='<u2',shape=(count,4096))
 changed=np.argwhere(base!=cand)
 # Full independent equations for bounded tails; disjoint actual rows for banks.
 indices=np.arange(count) if count<=129 else np.unique([0,1,31,63,64,count//2,count-2,count-1])
 exact=x[indices].astype('f8')@w.T
 assert np.isfinite(exact).all()
 ideal=bf16(exact)
 result=dict(label=label,rows=count,values=int(base.size),changed=int(len(changed)),independent_rows=indices.tolist(),independent_values=int(exact.size),baseline_rounding_disagreements=int(np.count_nonzero(base[indices]!=ideal)),candidate_rounding_disagreements=int(np.count_nonzero(cand[indices]!=ideal)),baseline_max_abs=float(np.max(abs(f32(base[indices]).astype('f8')-exact))),candidate_max_abs=float(np.max(abs(f32(cand[indices]).astype('f8')-exact))))
 if count<=129:
  detailed=[]
  for r,c in changed:
   lo,hi=sorted([float(f32(base[r,c])),float(f32(cand[r,c]))]);mid=(lo+hi)/2
   detailed.append(dict(row=int(r),column=int(c),baseline=float(f32(base[r,c])),candidate=float(f32(cand[r,c])),independent=float(exact[r,c]),midpoint=mid,distance_from_midpoint=float(exact[r,c]-mid)))
  result['changed_boundaries']=detailed
 rows.append(result)
 (a.output or a.results/'independent-query-equations.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(label,result['changed'],result['baseline_rounding_disagreements'],result['candidate_rounding_disagreements'],flush=True)
