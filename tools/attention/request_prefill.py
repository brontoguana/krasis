"""Replay one pre-admitted frozen prefill trial; run under an exclusive GPU controller."""
import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('run',type=Path);p.add_argument('label');p.add_argument('phase',choices=['proof','timing'])
p.add_argument('--base',default='http://127.0.0.1:18375')
a=p.parse_args();O=a.run;label=a.label;proof=a.phase=='proof'
body=json.loads((O/(label+'-request.json')).read_text());before=dict(monotonic_ns=time.monotonic_ns(),realtime_ns=time.time_ns())
with urllib.request.urlopen(urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=1200)as f:raw=f.read()
after=dict(monotonic_ns=time.monotonic_ns(),realtime_ns=time.time_ns());(O/(label+'-response.json')).write_bytes(raw)
v=json.loads(raw);trial=next(t for t in json.loads((O/'teacher-plan.json').read_text())['trials']if t['label']==label);D=O/'teacher'/label
assert v['krasis_debug']['input_token_ids']==body['debug_matched_raw_input_token_ids']
assert v['krasis_debug']['completion_token_ids'][:63]==trial['forced_inputs']
receipt=json.loads((D/'deferred-steps.json').read_text());assert len(receipt['step_receipts'])==63
assert receipt['prefill_attention_execution']==trial['prefill_attention_execution']
stats=receipt['prefill_tensor_stats'];assert (stats[0]>0)==(trial['prefill_attention_execution']==1),stats
assert stats[2]<=172212989
if proof:
 assert len(list(D.glob('*logits.f32')))==64
 assert (D/'final-state/allocations.json').exists()
else:assert not list(D.glob('*logits.f32'))
result=dict(before=before,after=after,wall_seconds=(after['monotonic_ns']-before['monotonic_ns'])/1e9,timing=v.get('krasis_timing'),receipt=receipt,usage=v.get('usage'))
(O/(label+'-validation.json')).write_text(json.dumps(result,indent=2)+'\n');print(label,result['wall_seconds'],receipt['prefill_tensor_stats'],flush=True)
