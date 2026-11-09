"""Check complete short logits and the unaffected state before timing a candidate."""
import argparse,json
from pathlib import Path
import numpy as np
from compare_runs import metrics
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--candidate-label',default='numerical3072');p.add_argument('--require-exact',action='store_true');a=p.parse_args()
L=a.baseline/'teacher/numerical3072';R=a.candidate/'teacher'/a.candidate_label
rows=[]
for name in ['first-logits.f32']+[f'step-{i}-logits.f32'for i in range(63)]:
 x=np.fromfile(L/name,dtype='<f4');y=np.fromfile(R/name,dtype='<f4');r=metrics(x,y)
 r.update(file=name,bit_exact=bool(np.array_equal(x.view('u4'),y.view('u4'))),argmax_equal=bool(x.argmax()==y.argmax()));rows.append(r)
left=json.loads((L/'final-state/allocations.json').read_text());right=json.loads((R/'final-state/allocations.json').read_text());assert left['logical_tokens']==right['logical_tokens']
x={r['allocation']['name']:r for r in left['allocations']};y={r['allocation']['name']:r for r in right['allocations']};assert x.keys()==y.keys()
stem=[];state=[]
for name,r in x.items():
 v=y[name];assert r['used_bytes']==v['used_bytes']
 eq=(L/'final-state'/r['file']).read_bytes()==(R/'final-state'/v['file']).read_bytes()
 item=dict(name=name,bytes=r['used_bytes'],bit_exact=eq);state.append(item)
 if r['allocation']['layer_idx']<3:stem.append(item)
result=dict(logits=rows,state=state,all_state_exact=all(r['bit_exact']for r in state),all_logits_exact=all(r['bit_exact']for r in rows),unaffected_state=stem,finite=all(r['finite']for r in rows),unaffected_state_exact=all(r['bit_exact']for r in stem))
(a.candidate/('early-comparison.json'if a.candidate_label=='numerical3072'else'early-'+a.candidate_label+'-comparison.json')).write_text(json.dumps(result,indent=2));print(json.dumps(dict(finite=result['finite'],unaffected_state_exact=result['unaffected_state_exact'],argmax_agreements=sum(r['argmax_equal']for r in rows),max_logit_abs=max(r['max_abs']for r in rows)),indent=2))
assert result['finite']and result['unaffected_state_exact'],'candidate changes the unaffected input-state stem or produces nonfinite logits'

if a.require_exact:assert result['all_state_exact']and result['all_logits_exact'],'strict relative continuation gate failed; evidence retained'
