"""Check actual projection/index boundaries and independent causal score equations."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--history',type=int,nargs='+',default=[3072,16384,65536]);a=p.parse_args()
def bf(path):return (np.fromfile(path,'<u2').astype('u4')<<16).view('f4').astype('f8')
rows=[]
for n in a.history:
 captures=[]
 for mode in [0,1]:
  response=json.loads((a.run/f'pre{n}proof{mode}-response.json').read_text());rid=response['id']
  captures.append({d.parent.name:d for d in (a.run/'capture').glob(f'mla_prefill_operands-*/request-{rid}-*') if (d/'indexer-operands.json').exists()})
 assert captures[0] and captures[0].keys()==captures[1].keys(),(n,captures)
 for key,d in sorted(captures[0].items()):
  c=captures[1][key];meta=json.loads((d/'indexer-operands.json').read_text());layout=json.loads((d/'indexer-score-layout.json').read_text())
  names=['indexer-lora.bf16','indexer-raw-query.bf16','indexer-rope-query.bf16','indexer-head-weights.bf16','indexer-scores.f32','selected.i32']
  for name in names:assert (d/name).read_bytes()==(c/name).read_bytes(),(n,key,name)
  assert meta['rope_dim']==0
  assert (d/'indexer-raw-query.bf16').read_bytes()==(d/'indexer-rope-query.bf16').read_bytes()
  h,w=meta['heads'],meta['head_dim'];q=bf(d/'indexer-rope-query.bf16').reshape(h,w);keys=bf(d/meta['scoring_key_file']).reshape(-1,w);weights=bf(d/'indexer-head-weights.bf16')
  scores=np.fromfile(d/'indexer-scores.f32','<f4');assert np.isfinite(scores).all()
  assert len(scores)==meta['prefix']//meta['pool_ratio']
  expected=(np.maximum((q@keys[:len(scores)].T)*layout['score_scale'],0)*weights[:,None]).sum(axis=0)
  delta=abs(expected-scores);assert np.isfinite(expected).all() and delta.max()<=2e-4,(key,delta.max())
  order=np.lexsort((np.arange(len(scores)),-scores));top=meta['score_topk'];ratio=meta['pool_ratio']
  expanded=(order[:top,None]*ratio+np.arange(ratio)).ravel().tolist()
  expanded+=list(range((meta['prefix']//ratio)*ratio,meta['prefix']))
  expanded+= [-1]*(meta['raw_output_topk']-len(expanded))
  actual=np.fromfile(d/'selected.i32','<i4');assert np.array_equal(actual,np.array(expanded,'i4')),(key,'independent stable selection')
  gap=float(scores[order[top-1]]-scores[order[top]]) if len(order)>top else None
  sorted_scores=scores[order];adjacent=sorted_scores[:-1]-sorted_scores[1:]
  rows.append(dict(history=n,capture=key,projection_width=h*w,score_values=len(scores),selected_values=len(actual),all_boundaries_bitwise=True,independent_score_max_abs=float(delta.max()),unchanged_score_abs_threshold=2e-4,cutoff_margin=gap,minimum_adjacent_margin=float(adjacent.min()),exact_score_ties=int(np.count_nonzero(adjacent==0)),changed_selections=0,masked_sentinels=int(np.count_nonzero(actual<0)),chunk_start=meta['chunk_start'],chunk_rows=meta['chunk_rows']))
result=dict(scope='Actual native BF16 projection and causal index consequences; no whole-model admission',rows=rows,all_bitwise=True,changed_selections=0)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
