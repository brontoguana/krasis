"""Compare complete value-intervention states/logits and capture-free paired requests.
Metrics preserve changed bits and inherited failures; no model-admission threshold is invented.
"""
import argparse,json,statistics
from pathlib import Path
import numpy as np
from compare_runs import metrics,states
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('accepted',type=Path);p.add_argument('output',type=Path);p.add_argument('--quality-baseline',type=Path);a=p.parse_args();O=a.run;result=[]
for n in [3072,16384,65536]:
 L=O/'teacher'/f'pre{n}proof0';R=O/'teacher'/f'pre{n}proof1';old=a.accepted/'teacher'/f'pre{n}proof1'
 logit_rows=[]
 for name in ['first-logits.f32']+[f'step-{i}-logits.f32'for i in range(63)]:
  x=np.fromfile(L/name,'<f4');y=np.fromfile(R/name,'<f4');z=np.fromfile(old/name,'<f4');assert x.size and x.shape==y.shape==z.shape
  assert np.array_equal(x.view('<u4'),z.view('<u4')),(n,name,'baseline changed from accepted capture')
  r=metrics(x,y);r.update(file=name,bit_exact=bool(np.array_equal(x.view('<u4'),y.view('<u4'))),argmax_equal=bool(x.argmax()==y.argmax()));assert r['finite'];logit_rows.append(r)
 state=states(L/'final-state',R/'final-state');oldstate=states(old/'final-state',L/'final-state');assert oldstate['bit_exact'],(n,'accepted baseline state changed')
 paired=[]
 for mode,suffixes in [(0,['0a','0b']),(1,['1a','1b'])]:
  trials=[json.loads((O/f'pre{n}{s}-validation.json').read_text())for s in suffixes]
  chunks=[3084]if n==3072 else([16397]if n==16384 else[32768,32768,13]);expected=[];start=0
  for chunk in chunks:
   expected.extend([[layer,start,chunk,4096,1536,1 if chunk>=64 else 0]for layer in range(3,44,4)]);start+=chunk
  for t in trials:
   r=t['receipt'];assert r['indexer_projection_execution']==1 and r['prefill_attention_execution']==1+mode
   assert r['indexer_projection_launches']==expected and r['completed_decode_steps']==63
  paired.append(dict(mode=mode,wall_s=[t['wall_seconds']for t in trials],median_s=statistics.median(t['wall_seconds']for t in trials),decode_ms_per_forward=[1000*t['receipt']['decode_elapsed_seconds']/63 for t in trials]))
 first=min([r['layer']for r in state['rows']if not r['bit_exact']],default=None)
 result.append(dict(history=n,baseline_matches_accepted=True,logits=logit_rows,state=state,first_changed_state_layer=first,paired=paired,wall_improvement_pct=100*(1-paired[1]['median_s']/paired[0]['median_s'])))
quality_before=json.loads(((a.quality_baseline or a.accepted)/'quality/RESULT.json').read_text());quality_after=json.loads((O/'quality/RESULT.json').read_text());left={r['label']:r for r in quality_before['rows']};right={r['label']:r for r in quality_after['rows']};assert left.keys()==right.keys()
quality=[dict(label=k,baseline_pass=left[k]['pass'],candidate_pass=right[k]['pass'],text_equal=left[k]['text']==right[k]['text'])for k in left]
summary=dict(scope=__doc__,rows=result,quality=quality,new_heldout_failure=any(r['baseline_pass']and not r['candidate_pass']for r in quality),likelihood=json.loads((O/'likelihood.json').read_text()),threshold_policy='Independent operator2e-4 unchanged. Added R1 cold2e-6 screen failed and remains rejected; R2 screening policy uses established2e-4 operator gate. Model metrics/quality reported separately; no absolute admission inferred.')
a.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(rows=[dict(history=r['history'],paired=r['paired'],wall_improvement_pct=r['wall_improvement_pct'],state_exact=r['state']['bit_exact'],first_changed_state_layer=r['first_changed_state_layer'],max_logit_abs=max(v['max_abs']for v in r['logits']),argmax_agreement=sum(v['argmax_equal']for v in r['logits']))for r in result],quality=quality),indent=2))
