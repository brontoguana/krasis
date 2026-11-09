"""Compare complete logits and registered states without changing tolerances."""
import argparse,json
from pathlib import Path
import numpy as np
from prepare_replay import unpack

def metrics(a,b):
 assert a.shape==b.shape
 finite=np.isfinite(a)&np.isfinite(b);delta=a.astype('f8')-b.astype('f8')
 good=delta[finite];left=a.astype('f8')[finite]
 return dict(values=a.size,finite=bool(finite.all()),max_abs=float(np.max(np.abs(good)))if good.size else None,rms=float(np.sqrt(np.mean(good*good)))if good.size else None,relative_l2=float(np.sqrt(np.sum(good*good)/max(np.sum(left*left),1e-300))))

def states(left,right):
 x=json.loads((left/'allocations.json').read_text());y=json.loads((right/'allocations.json').read_text())
 assert x['logical_tokens']==y['logical_tokens']
 xs={r['allocation']['name']:r for r in x['allocations']};ys={r['allocation']['name']:r for r in y['allocations']};assert xs.keys()==ys.keys()
 rows=[]
 for name,u in xs.items():
  v=ys[name];meta=u['allocation'];assert u['used_bytes']==v['used_bytes']
  for key in ['kind','dtype','element_size','shape','strides_bytes','growth']:assert meta[key]==v['allocation'][key],(name,key)
  a=left/u['file'];b=right/v['file'];ab=np.fromfile(a,dtype='u1');bb=np.fromfile(b,dtype='u1');assert ab.size==bb.size==u['used_bytes']
  row=dict(name=name,layer=meta['layer_idx'],kind=meta['kind'],bytes=ab.size,bit_exact=bool(np.array_equal(ab,bb)),differing_bytes=int(np.count_nonzero(ab!=bb)))
  if meta['dtype']=='float32':row['numeric']=metrics(ab.view('<f4'),bb.view('<f4'))
  elif meta['dtype']=='bfloat16':row['numeric']=metrics((ab.view('<u2').astype('<u4')<<16).view('<f4'),(bb.view('<u2').astype('<u4')<<16).view('<f4'))
  elif meta['kind']=='mla_compressed_kv':row['unpacked_k6']=metrics(unpack(a),unpack(b))
  rows.append(row)
 return dict(logical_tokens=x['logical_tokens'],allocations=len(rows),bytes=sum(r['bytes']for r in rows),bit_exact=all(r['bit_exact']for r in rows),rows=rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir()
 summary=[]
 for n in [3072,16384,65536]:
  label=f'numerical{n}';L=a.baseline/'teacher'/label;R=a.candidate/'teacher'/label
  files=['first-logits.f32']+[f'step-{i}-logits.f32'for i in range(63)];rows=[]
  for name in files:
   x=np.fromfile(L/name,dtype='<f4');y=np.fromfile(R/name,dtype='<f4');v=metrics(x,y)
   v.update(file=name,bit_exact=bool(np.array_equal(x.view('u4'),y.view('u4'))),argmax_baseline=int(x.argmax()),argmax_candidate=int(y.argmax()))
   rows.append(v)
  state=states(L/'final-state',R/'final-state')
  (a.out/(label+'-logits.json')).write_text(json.dumps(rows,indent=2));(a.out/(label+'-state.json')).write_text(json.dumps(state,indent=2))
  summary.append(dict(label=label,logit_vectors=len(rows),logit_values=sum(r['values']for r in rows),all_logits_finite=all(r['finite']for r in rows),all_logits_bit_exact=all(r['bit_exact']for r in rows),argmax_agreements=sum(r['argmax_baseline']==r['argmax_candidate']for r in rows),max_logit_abs=max(r['max_abs']for r in rows),state_bytes=state['bytes'],all_state_bit_exact=state['bit_exact'],first_differing_state_layer=min([r['layer']for r in state['rows']if not r['bit_exact']],default=None)))
 prefixes=[]
 for height in [3000,65550]:
  paths=[]
  for root in [a.baseline,a.candidate]:
   choices=list((root/'capture-before-lifecycle').glob(f'sequence_state-layer0-height{height}-row0/**/device-0/allocations.json'))
   assert len(choices)==1,(root,height,choices)
   paths.append(choices[0].parent)
  state=states(*paths)
  logits=[np.fromfile(p.parent/'first-logits.f32.bin',dtype='<f4')for p in paths]
  (a.out/f'prefix-{height}-state.json').write_text(json.dumps(state,indent=2))
  prefixes.append(dict(height=height,allocations=state['allocations'],bytes=state['bytes'],all_state_bit_exact=state['bit_exact'],logits=metrics(*logits),all_logits_bit_exact=bool(np.array_equal(logits[0].view('u4'),logits[1].view('u4')))))
 (a.out/'SUMMARY.json').write_text(json.dumps(dict(scope=__doc__,rows=summary,prefixes=prefixes),indent=2));print(json.dumps(dict(rows=summary,prefixes=prefixes),indent=2))
if __name__=='__main__':main()
