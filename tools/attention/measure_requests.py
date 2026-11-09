"""Summarise paired ordinary and frozen-token requests from immutable run directories."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
rows=[]
for n in [3072,16384,65536]:
 for kind in ['frozen','chat','code']:
  arms=[]
  for root in [a.baseline,a.candidate]:
   trials=[]
   for rep in [0,1]:
    if kind=='frozen':
     r=json.loads((root/f'fixed{n}r{rep}-validation.json').read_text())
     trials.append(dict(wall_s=r['wall_seconds'],decode_ms=r['decode_ms_per_forward']))
    else:
     r=json.loads((root/f'ordinary-{kind}-{n}-r{rep}/result.json').read_text());assert r['performance_valid']
     t=r['engine_timing'];o=t['overhead']
     trials.append(dict(wall_s=r['wall_seconds'],first_content_s=r['ttft_seconds'],prefill_s=o['prefill_ms']/1000,decode_ms=t['decode_time_ms']/t['decode_tokens']))
   arms.append(dict(trials=trials,medians={k:statistics.median(t[k]for t in trials)for k in trials[0]}))
  rows.append(dict(history=n,kind=kind,baseline=arms[0],candidate=arms[1],speedups={k:arms[0]['medians'][k]/arms[1]['medians'][k]for k in arms[0]['medians']}))
a.output.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
