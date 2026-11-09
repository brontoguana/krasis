"""Offline actual-route BF16 ceiling; reconstruction excluded from timed contractions."""
import argparse, hashlib, json, statistics, struct, subprocess, time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);p.add_argument('--column-weight',action='store_true');a=p.parse_args()
assert not a.output.exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
C=a.capture/'prefill_moe_w1_operands-layer3-height8192-row8191/prefill-4-device-0';W=a.capture/'prefill_moe_row-layer3-height8192-row8191/prefill-4-device-0'
d=json.loads((C/'dispatch.json').read_text());selected=json.loads((W/'routing.json').read_text())['selected_experts']
ids=np.fromfile(C/'sorted-ids.i32',np.int32)[:d['actual_padded_sorted_count']].reshape(-1,64);blocks=np.fromfile(C/'expert-block-ids.i32',np.int32)[:len(ids)]
occ={int(e):int(np.count_nonzero(ids[blocks==e]<d['size_m'])) for e in set(blocks)};hot=max(occ,key=occ.get);assert hot==252 and occ[hot]==691
perm=[]
for i in range(32):
 ps=[16*r+i//4+8*b for b in range(2) for r in [2*(i%4),2*(i%4)+1,2*(i%4+4),2*(i%4+4)+1]]
 for j in range(4):perm.extend(v+256*j for v in ps)
perm=np.array(perm).reshape(-1,8)[:,[0,2,4,6,1,3,5,7]].ravel();assert len(set(perm.tolist()))==1024
sp=np.array([i+8*j for i in range(8) for j in range(8)])
hashes={};reconstruct=[]
def bf_file(path):
 b=path.read_bytes();hashes[str(path)]=hashlib.sha256(b).hexdigest();return torch.from_numpy(np.frombuffer(b,np.uint16).copy()).view(torch.bfloat16)
def unpack(wb,sb,k,n):
 start=time.monotonic();words=np.frombuffer(wb,'<u4');q=((words[:,None]>>np.arange(0,32,4,dtype=np.uint32))&15).astype(np.int8).reshape(-1,1024)
 un=np.empty_like(q);un[:,perm]=q
 q=un.reshape(k//16,n//16,16,16).transpose(0,2,1,3).reshape(k,n).astype(np.float32)-8
 sraw=np.frombuffer(sb,np.uint16).reshape(-1,64);su=np.empty_like(sraw);su[:,sp]=sraw
 sf=torch.from_numpy(su.copy()).view(torch.bfloat16).float().numpy().reshape(k//128,n)
 y=torch.from_numpy(q*sf[np.arange(k)//128]).to(torch.bfloat16).cuda();torch.cuda.synchronize();reconstruct.append(time.monotonic()-start);return y
x=bf_file(C/'actual-A.bf16').reshape(d['size_m'],4096).cuda()
assert np.unique(x[::8].view(torch.int16).cpu().numpy(),axis=0).shape[0]==8192
arms={};stride=8388608+262144+4194304+131072
with a.cache.open('rb') as f:
 h=f.read(64);assert h[:4]==b'KRAS' and struct.unpack_from('<I5Q',h,4)==(7,4096,2048,288,42,128)
 for e in selected+[hot]:
  f.seek(64+e*stride);wb=f.read(8388608);sb=f.read(262144);dw=f.read(4194304);ds=f.read(131072)
  if e in selected:
   for name,b in [('w1',wb),('w2',dw)]:assert b==(W/f'{name}-expert{e}-packed.bin').read_bytes()
   for name,b in [('w1',sb),('w2',ds)]:assert b==(W/f'{name}-expert{e}-scales.bf16').read_bytes()
  hashes[f'expert{e}-weights']=hashlib.sha256(wb+sb+dw+ds).hexdigest()
  ix=ids[blocks==e].ravel();ix=ix[ix<d['size_m']];m=len(ix);pad=(m+63)//64*64
  gu=unpack(wb,sb,4096,4096);down=unpack(dw,ds,2048,4096)
  if a.column_weight:gu=gu.T.contiguous().T;down=down.T.contiguous().T
  ixg=torch.from_numpy(ix.astype(np.int64)).cuda();gx=x[ixg];gy=(gx@gu).float();g=gy[:,:2048].clamp(max=10);u=gy[:,2048:].clamp(-10,10);activation=(torch.nn.functional.silu(g)*u).to(torch.bfloat16)
  for name,source,weight,k,n in [('gate_up',gx,gu,4096,4096),('down',activation,down,2048,4096)]:
   xx=torch.zeros((pad,k),dtype=torch.bfloat16,device='cuda');xx[:m]=source;yy=torch.empty((pad,n),dtype=torch.bfloat16,device='cuda')
   arms[e,name]=(xx,weight,yy,m,pad,k,n)
def measure(items,useful):
 def call():
  for xx,w,y,m,pad,k,n in items:torch.mm(xx[:m] if useful else xx,w,out=y[:m] if useful else y)
 for _ in range(5):call()
 t=[]
 for _ in range(18):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();call();en.record();en.synchronize();t.append(st.elapsed_time(en))
 flops=sum(2*(z[3] if useful else z[4])*z[5]*z[6] for z in items)
 return dict(flops=flops,median_ms=statistics.median(t),timings_ms=t,tflops=flops/(statistics.median(t)*1e9))
results={}
for label,experts in [('original',selected),('hot',selected+[hot])]:
 results[label]={name:{mode:measure([arms[e,name] for e in experts],mode=='useful') for mode in ['useful','padded']} for name in ['gate_up','down']}
free=torch.cuda.mem_get_info()[0];assert free>=600*1024**2
v=dict(weight_layout='column_major' if a.column_weight else 'row_major',scope='Resident independently reconstructed BF16 contraction diagnostic only; separate launches per expert, actual gate/up slots; down operands independently regenerated using BF16 projections and raw-gate clamp10 (not captured native activation bytes); no routed scheduling, unpack, transfers, activation or scatter in timings; no deployable speed claim',occupancies={e:occ[e] for e in selected+[hot]},results=results,host_unpack_and_upload_seconds=reconstruct,minimum_free_bytes=free,sha256=hashes,device=torch.cuda.get_device_name())
a.output.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v for k,v in v.items() if k not in ['sha256','results']}),flush=True)
