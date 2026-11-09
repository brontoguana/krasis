"""Complete-output oracle checks at the retained2e-4 absolute threshold."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('results',type=Path);a=p.parse_args()
ref=np.fromfile(a.input/'oracle.f64',dtype='<f8');base=np.fromfile(a.results/'baseline-eager.f32',dtype='<f4');rows=[]
for f in sorted(a.results.glob('*.f32')):
 x=np.fromfile(f,dtype='<f4');assert x.shape==ref.shape
 d=np.abs(x-ref);delta=np.abs(x.astype('f8')-base)
 rows.append(dict(file=f.name,values=x.size,finite=bool(np.isfinite(x).all()),max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),over_2e_4=int(np.count_nonzero(d>2e-4)),baseline_max_abs=float(delta.max()),baseline_bit_exact=bool(np.array_equal(x.view('u4'),base.view('u4')))))
summary=dict(rows=rows,pass_oracle=all(r['finite']and r['over_2e_4']==0 for r in rows))
(a.results/'comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert summary['pass_oracle']
