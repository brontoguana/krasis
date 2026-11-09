"""Unchanged routed controls for structural feasibility. No candidate dispatch."""
import argparse,ctypes,hashlib,json,statistics,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('control',type=Path);p.add_argument('vendor',type=Path);p.add_argument('out',type=Path);p.add_argument('--hot-cache',type=Path);p.add_argument('--invocation',required=True);a=p.parse_args()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
O=a.out;O.mkdir(exist_ok=False);torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
C=a.run/'capture/prefill_moe_w1_operands-layer3-height8192-row8191'/a.invocation
W=a.run/'capture/prefill_moe_row-layer3-height8192-row8191'/a.invocation
d=json.loads((C/'dispatch.json').read_text());r=json.loads((W/'routing.json').read_text());selected=r['selected_experts'];assert len(set(selected))==8
assert d['height']==8192 and d['moe_block_size']==64 and d['size_k']==4096 and d['bits']==4
hashes={}
def read(path,dtype):
 hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return np.fromfile(path,dtype).copy()
def bf(path):return torch.from_numpy(read(path,np.uint16)).view(torch.bfloat16)
x=bf(C/'actual-A.bf16').reshape(d['size_m'],4096).cuda();actual=x[::8].view(torch.int16).cpu().numpy()
assert np.unique(actual,axis=0).shape[0]>8000,'natural rows are not distinct'
sorted_ids=read(C/'sorted-ids.i32',np.int32);expert_blocks=read(C/'expert-block-ids.i32',np.int32);count=int(read(C/'post-count.i32',np.int32)[0]);assert count==d['actual_padded_sorted_count']
ids=sorted_ids[:count].reshape(-1,64);expert_blocks=expert_blocks[:count//64]
cache_receipt=None
if a.hot_cache:
 import struct
 with a.hot_cache.open('rb') as f:
  header=f.read(64);assert header[:4]==b'KRAS' and struct.unpack_from('<I',header,4)[0]==7
  assert struct.unpack_from('<5Q',header,8)==(4096,2048,288,42,128)
  stride=8388608+262144+4194304+131072
  # Layer3 is the first routed layer. Verify every retained expert byte first.
  for e in selected:
   f.seek(64+e*stride)
   assert f.read(8388608)==(W/f'w1-expert{e}-packed.bin').read_bytes()
   assert f.read(262144)==(W/f'w1-expert{e}-scales.bf16').read_bytes()
  full_ids=ids
  occ={int(e):int(np.count_nonzero(full_ids[expert_blocks==e]<d['size_m']))for e in set(expert_blocks)}
  hot=max(occ,key=occ.get);assert hot not in selected
  f.seek(64+hot*stride);hotw=f.read(8388608);hots=f.read(262144)
  cache_receipt=dict(header_hex=header.hex(),size=a.hot_cache.stat().st_size,hot_expert=hot,hot_occupancy=occ[hot],verified_retained_experts=selected.copy(),packed_sha256=hashlib.sha256(hotw).hexdigest(),scales_sha256=hashlib.sha256(hots).hexdigest())
 selected=selected+[hot]
mask=np.isin(expert_blocks,selected);ids=ids[mask].reshape(-1);blocks=expert_blocks[mask]
assert np.all(ids>=0) and np.all(ids<=d['size_m'])
valid=ids[ids<d['size_m']];assert len(set(valid.tolist()))==len(valid)
occupancies={int(e):int(np.count_nonzero(ids.reshape(-1,64)[blocks==e]<d['size_m']))for e in selected}
ids=torch.from_numpy(ids).cuda();blocks=torch.from_numpy(blocks.copy()).cuda();count=torch.tensor([ids.numel()],dtype=torch.int32,device='cuda');tw=torch.from_numpy(read(C/'topk-weights.f32',np.float32)).cuda()
P=ctypes.c_void_p;I=ctypes.c_int;B=ctypes.c_bool;ptr=lambda t:P(t.data_ptr());stream=P(torch.cuda.current_stream().cuda_stream);sms=torch.cuda.get_device_properties(0).multi_processor_count
lib=ctypes.CDLL(str(a.control.resolve()));fn=lib.routed_n;fn.argtypes=[I]+[P]*10+[I]*4+[P];fn.restype=I
res=lib.routed_resources;res.argtypes=[I,ctypes.POINTER(I)];res.restype=I
resources={}
for n in [256]:
 v=(I*5)();assert res(n,v)==0;resources[n]=dict(zip(['registers','static_shared','local_bytes','dynamic_shared','active_blocks_per_sm'],v))
vendor=ctypes.CDLL(str(a.vendor.resolve())).krasis_marlin_moe_mm_bf16;vendor.argtypes=[P]*15+[I,I,B,B,I,I,I,P,P,B,B,B,B,I,I,I,P,I,I,I,B,B,B,P,P];vendor.restype=None
class Scalar(ctypes.Structure):
 _fields_=[('exp',ctypes.c_uint8),('mantissa',ctypes.c_uint8),('signed',ctypes.c_uint8),('pad',ctypes.c_uint8),('bias',I),('finite',ctypes.c_uint8),('nan',ctypes.c_uint8),('pad1',ctypes.c_uint8*2)]
q=Scalar(0,4,0,0,8,0,1,(ctypes.c_uint8*2)(0,0));assert ctypes.sizeof(q)==12
wp=torch.zeros(d['experts'],dtype=torch.int64);sp=wp.clone();alloc=[]
for e in selected:
 w=torch.from_numpy(np.frombuffer(hotw,np.uint8).copy()).cuda() if a.hot_cache and e==hot else torch.from_numpy(read(W/f'w1-expert{e}-packed.bin',np.uint8)).cuda();s=torch.from_numpy(np.frombuffer(hots,np.uint16).copy()).view(torch.bfloat16).cuda() if a.hot_cache and e==hot else bf(W/f'w1-expert{e}-scales.bf16').cuda();alloc.extend([w,s]);wp[e]=w.data_ptr();sp[e]=s.data_ptr()
wp=wp.cuda();sp=sp.cuda();m=d['size_m'];out=torch.empty((m,4096),dtype=torch.bfloat16,device='cuda');tmp=torch.zeros(d['C_tmp_f32'],device='cuda');locks=torch.zeros(d['workspace_u32'],dtype=torch.int32,device='cuda')
valid_gpu=torch.from_numpy(valid.astype(np.int64)).cuda()
def call(n):
 stream=P(torch.cuda.current_stream().cuda_stream)
 if n==0:
  vendor(ptr(x),P(),ptr(out),ptr(tmp),P(),P(),P(),P(),P(),P(),P(),ptr(ids),ptr(blocks),ptr(count),ptr(tw),64,1,False,False,m,4096,runtime_k,ptr(locks),P(ctypes.addressof(q)),False,False,True,False,runtime_k//128,128,0,stream,-1,-1,sms,False,True,False,ptr(wp),ptr(sp))
 else:assert fn(n,ptr(x),ptr(out),ptr(tmp),ptr(ids),ptr(blocks),ptr(count),ptr(tw),ptr(locks),ptr(wp),ptr(sp),m,4096,runtime_k,sms,stream)==0

runtime_k=4096
result={}
for stage in ['gate_up','down']:
 outputs={};timings={};graphs={};graph_timings={}
 for n in [0,256]:
  out.fill_(float('nan'));tmp.zero_();locks.zero_();call(n);torch.cuda.synchronize()
  outputs[n]=out[valid_gpu].clone();assert torch.isfinite(outputs[n]).all()
  timings[n]=[];graph_timings[n]=[]
  for _ in range(3):call(n)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):call(n)
  graphs[n]=graph
  out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
  assert torch.equal(out[valid_gpu],outputs[n]),'complete graph output mismatch'
 for mode in ['eager','graph']:
  for _ in range(9):
   for n in [0,256,256,0]:
    st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
    if mode=='eager':call(n)
    else:graphs[n].replay()
    en.record();en.synchronize();(timings if mode=='eager' else graph_timings)[n].append(st.elapsed_time(en))
 assert torch.equal(outputs[0],outputs[256]),'same-build control differs'
 result[stage]=dict(values=outputs[0].numel(),complete_bitwise=True,graph_complete_bitwise=True,eager_ms={n:statistics.median(t)for n,t in timings.items()},graph_ms={n:statistics.median(t)for n,t in graph_timings.items()},timings_ms=timings,graph_timings_ms=graph_timings)
 for n,y in outputs.items():y.view(torch.int16).cpu().numpy().tofile(O/f'{stage}-{n}.bf16')
 if stage=='gate_up':
  # Identical regenerated down operands for both controls; not captured activation qualification.
  gu=outputs[0].float();gate=gu[:,:2048].clamp(max=10);up=gu[:,2048:].clamp(-10,10)
  activation=(torch.nn.functional.silu(gate)*up).to(torch.bfloat16)
  x=torch.zeros((m,2048),dtype=torch.bfloat16,device='cuda');x[valid_gpu]=activation;runtime_k=2048
  wp=torch.zeros(d['experts'],dtype=torch.int64);sp=wp.clone();alloc=[]
  for e in selected:
   if a.hot_cache and e==hot:
    with a.hot_cache.open('rb') as f:
     f.seek(64+e*stride+8388608+262144);wb=f.read(4194304);sb=f.read(131072)
    w=torch.from_numpy(np.frombuffer(wb,np.uint8).copy()).cuda();scale=torch.from_numpy(np.frombuffer(sb,np.uint16).copy()).view(torch.bfloat16).cuda()
    hashes['hot_down']=hashlib.sha256(wb+sb).hexdigest()
   else:
    w=torch.from_numpy(read(W/f'w2-expert{e}-packed.bin',np.uint8)).cuda();scale=bf(W/f'w2-expert{e}-scales.bf16').cuda()
   alloc.extend([w,scale]);wp[e]=w.data_ptr();sp[e]=scale.data_ptr()
  wp=wp.cuda();sp=sp.cuda()
free=torch.cuda.mem_get_info()[0];assert free>=600*1024**2
summary=dict(stages=result,occupancies=occupancies,padded_rows=ids.numel(),useful_rows=len(valid),resources=resources,source_library_sha256=hashlib.sha256(a.vendor.read_bytes()).hexdigest(),control_library_sha256=hashlib.sha256(a.control.read_bytes()).hexdigest(),sha256=hashes,minimum_free_bytes=free,scope='Natural complete gate/up selected slots; down uses identical regenerated activation operands for both unchanged controls. No candidate or serving qualification.')
(O/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in ['sha256','stages']}),flush=True)
