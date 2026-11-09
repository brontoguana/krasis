"""Localise the first changed MLA value boundary on actual native captures."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from prepare_replay import unpack
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def selected(root,layer):
 paths=list(root.glob(f'mla_prefill_operands-layer{layer}-height65536-row65535/*'))
 assert len(paths)==1,(root,layer,paths);return paths[0]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bf16(x):
 u=x.view('<u4');return ((u+0x7fff+((u>>16)&1))>>16).astype('<u2')
rows=[]
for layer in [3,23,43]:
 L,R=selected(a.baseline,layer),selected(a.candidate,layer)
 same={name:digest(L/name)==digest(R/name)for name in ['q_absorbed.f32','selected.i32','position.i32','ckv-prefix.raw','w_vc.bf16','w_kc.bf16']}
 actual=[];equations=[]
 for root in [L,R]:
  q=np.fromfile(root/'q_absorbed.f32','<f4').reshape(64,512).astype('f8');ids=np.fromfile(root/'selected.i32','<i4');pos=int(np.fromfile(root/'position.i32','<i4')[0]);cache=unpack(root/'ckv-prefix.raw');valid=(ids>=0)&(ids<len(cache))&(ids<=pos);keys=cache[ids[valid]].astype('f8')
  score=q@keys.T*.0625;w=np.exp(score-score.max(axis=1,keepdims=True));expected=w@keys/w.sum(axis=1,keepdims=True)
  y=np.fromfile(root/'attention.f32','<f4').reshape(64,512);assert np.isfinite(y).all();delta=np.abs(y-expected);equations.append(dict(values=y.size,max_abs=float(delta.max()),over_2e_4=int(np.count_nonzero(delta>2e-4))));actual.append(y)
  assert equations[-1]['over_2e_4']==0
 x,y=actual;delta=np.abs(x.astype('f8')-y);bx,by=bf16(x),bf16(y)
 wvc=[np.fromfile(root/'wvc_output.bf16','<u2')for root in [L,R]];assert wvc[0].shape==wvc[1].shape
 if layer==3:assert all(same.values()),same
 rows.append(dict(layer=layer,inputs_equal=same,independent=equations,attention=dict(values=x.size,different_bits=int(np.count_nonzero(x.view('<u4')!=y.view('<u4'))),max_abs=float(delta.max())),packed_attention_bf16=dict(values=bx.size,different=int(np.count_nonzero(bx!=by))),wvc_bf16=dict(values=wvc[0].size,different=int(np.count_nonzero(wvc[0]!=wvc[1])))))
result=dict(scope='Actual65K native last-row boundary. Layer3 is the first modified attention layer; exact packed cache/query/selection and weights isolate the value-contraction intervention. Later-layer operands may differ after propagation. BF16 rounding is reconstructed explicitly. This is not a full-continuation substitution proof.',rows=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
