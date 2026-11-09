import json,sys
from pathlib import Path
import numpy as np
O=Path(sys.argv[1]);result=[]
def logsoftmax(x):
 x=x.astype('f8');x-=x.max();return x-np.log(np.exp(x).sum())
for n in [3072,16384,65536]:
 plan=json.loads((O/'teacher-plan.json').read_text());trial=next(t for t in plan['trials']if t['label']==f'pre{n}proof0');tokens=trial['forced_inputs'];rows=[]
 for step,name in enumerate(['first-logits.f32']+[f'step-{i}-logits.f32'for i in range(63)]):
  a=np.fromfile(O/'teacher'/f'pre{n}proof0'/name,dtype='<f4');b=np.fromfile(O/'teacher'/f'pre{n}proof1'/name,dtype='<f4');assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all()
  la,lb=map(logsoftmax,[a,b]);p=np.exp(la)
  r=dict(step=step,max_logit_abs=float(abs(a.astype('f8')-b).max()),kl=float((p*(la-lb)).sum()),argmax_equal=bool(a.argmax()==b.argmax()),bitwise=bool(np.array_equal(a.view('u4'),b.view('u4'))))
  if step<len(tokens):r.update(token=tokens[step],nll_baseline=float(-la[tokens[step]]),nll_candidate=float(-lb[tokens[step]]),delta_nll=float(la[tokens[step]]-lb[tokens[step]]))
  rows.append(r)
 result.append(dict(history=n,vectors=len(rows),mean_kl=float(np.mean([r['kl'] for r in rows])),max_kl=max(r['kl'] for r in rows),mean_delta_nll=float(np.mean([r['delta_nll'] for r in rows if 'delta_nll' in r])),max_delta_nll=max(r.get('delta_nll',0) for r in rows),argmax_agreement=sum(r['argmax_equal'] for r in rows),all_bitwise=all(r['bitwise']for r in rows),rows=rows))
(O/'likelihood.json').write_text(json.dumps(dict(scope='Relative full-distribution and fixed-token likelihood screening; not whole-model admission',thresholds_changed=False,results=result),indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items()if k!='rows'}for r in result],indent=2))
