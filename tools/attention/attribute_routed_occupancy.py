"""Join retained exact-prefix transfer counts to source-mapped expert shapes.
The65,536-ID diagnostic covers the first two full chunks, not the13-token tail.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('attribution',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
rows=[]
for line in a.log.read_text(errors='replace').splitlines():
 if line.startswith('[PREFILL-TRANSFER] '):
  v=json.loads(line.split(' ',1)[1])
  if v['prompt_tokens']==65536:rows.append(v)
assert len(rows)==336
mapping={}
for r in rows:
 key=(r['layer'],r['chunk'],r['band_row_start']//8192);assert key not in mapping
 counts=r['selection_counts'];assert len(counts)==288 and sum(counts)==r['rows']*8==65536 and min(counts)>=0
 mapping[key]=r
trace=json.loads(a.attribution.read_text());joined=[]
for g in trace['groups']:
 if g['rows']!=8192:continue
 key=(g['layer'],g['chunk'],g['band']);r=mapping[key]
 joined.append({**g,'expert_token_counts':r['selection_counts'],'padded64':sum((n+63)//64*64 for n in r['selection_counts']),'padded32':sum((n+31)//32*32 for n in r['selection_counts']),'diagnostic_copied_bytes':r['copied_bytes'],'resident_ids':r['resident_ids']})
assert len(joined)==len(mapping)==336
counts=np.asarray([r['expert_token_counts']for r in joined],dtype=np.int64).ravel();live=counts[counts>0];tokens=int(counts.sum())
bins=[]
for lo,hi in [(1,32),(33,64),(65,128),(129,256),(257,512),(513,8192)]:
 x=counts[(counts>=lo)&(counts<=hi)];bins.append(dict(min=lo,max=hi,expert_bands=len(x),token_assignments=int(x.sum()),token_fraction=float(x.sum()/tokens)))
result=dict(scope=__doc__,mapping_note='Actual accepted-prefix diagnostic counts joined by layer/chunk/band. Frozen65K request shares the first65,536 causal IDs and full-chunk geometry; diagnostic transfer/residency bytes are not assigned to ordinary timing or trace requests.',groups=joined,counts=dict(total_expert_token_assignments=tokens,nonzero_expert_bands=len(live),minimum=int(live.min()),maximum=int(live.max()),percentiles={str(q):float(np.percentile(live,q))for q in [25,50,75,90,99]},bins=bins,padded64=sum(r['padded64']for r in joined),padded32=sum(r['padded32']for r in joined)),tail_projection_ms=sum(g['w1_ms']+g['w2_ms']for g in trace['groups']if g['rows']==13))
a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k!='groups'})
