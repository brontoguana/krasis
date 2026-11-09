"""Bounded natural routed N-tiling replay. No serving dispatch changes."""
import argparse,ctypes,hashlib,json,statistics,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('candidate',type=Path);p.add_argument('vendor',type=Path);p.add_argument('out',type=Path);p.add_argument('--counter-arm',type=int,choices=[0,128]);p.add_argument('--invocation',required=True);a=p.parse_args()
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
mask=np.isin(expert_blocks,selected);ids=ids[mask].reshape(-1);blocks=expert_blocks[mask]
assert np.all(ids>=0) and np.all(ids<=d['size_m'])
valid=ids[ids<d['size_m']];assert len(set(valid.tolist()))==len(valid)
occupancies={int(e):int(np.count_nonzero(ids.reshape(-1,64)[blocks==e]<d['size_m']))for e in selected}
ids=torch.from_numpy(ids).cuda();blocks=torch.from_numpy(blocks.copy()).cuda();count=torch.tensor([ids.numel()],dtype=torch.int32,device='cuda');tw=torch.from_numpy(read(C/'topk-weights.f32',np.float32)).cuda()
P=ctypes.c_void_p;I=ctypes.c_int;B=ctypes.c_bool;ptr=lambda t:P(t.data_ptr());stream=P(torch.cuda.current_stream().cuda_stream);sms=torch.cuda.get_device_properties(0).multi_processor_count
lib=ctypes.CDLL(str(a.candidate.resolve()));fn=lib.routed_n;fn.argtypes=[I]+[P]*10+[I]*4+[P];fn.restype=I
res=lib.routed_resources;res.argtypes=[I,ctypes.POINTER(I)];res.restype=I
resources={}
for n in [256,128]:
 v=(I*5)();assert res(n,v)==0;resources[n]=dict(zip(['registers','static_shared','local_bytes','dynamic_shared','active_blocks_per_sm'],v))
vendor=ctypes.CDLL(str(a.vendor.resolve())).krasis_marlin_moe_mm_bf16;vendor.argtypes=[P]*15+[I,I,B,B,I,I,I,P,P,B,B,B,B,I,I,I,P,I,I,I,B,B,B,P,P];vendor.restype=None
class Scalar(ctypes.Structure):
 _fields_=[('exp',ctypes.c_uint8),('mantissa',ctypes.c_uint8),('signed',ctypes.c_uint8),('pad',ctypes.c_uint8),('bias',I),('finite',ctypes.c_uint8),('nan',ctypes.c_uint8),('pad1',ctypes.c_uint8*2)]
q=Scalar(0,4,0,0,8,0,1,(ctypes.c_uint8*2)(0,0));assert ctypes.sizeof(q)==12
wp=torch.zeros(d['experts'],dtype=torch.int64);sp=wp.clone();alloc=[]
for e in selected:
 w=torch.from_numpy(read(W/f'w1-expert{e}-packed.bin',np.uint8)).cuda();s=bf(W/f'w1-expert{e}-scales.bf16').cuda();alloc.extend([w,s]);wp[e]=w.data_ptr();sp[e]=s.data_ptr()
wp=wp.cuda();sp=sp.cuda();m=d['size_m'];out=torch.empty((m,4096),dtype=torch.bfloat16,device='cuda');tmp=torch.zeros(d['C_tmp_f32'],device='cuda');locks=torch.zeros(d['workspace_u32'],dtype=torch.int32,device='cuda')
valid_gpu=torch.from_numpy(valid.astype(np.int64)).cuda()
def call(n):
 if n==0:
  vendor(ptr(x),P(),ptr(out),ptr(tmp),P(),P(),P(),P(),P(),P(),P(),ptr(ids),ptr(blocks),ptr(count),ptr(tw),64,1,False,False,m,4096,4096,ptr(locks),P(ctypes.addressof(q)),False,False,True,False,32,128,0,stream,-1,-1,sms,False,True,False,ptr(wp),ptr(sp))
 else:assert fn(n,ptr(x),ptr(out),ptr(tmp),ptr(ids),ptr(blocks),ptr(count),ptr(tw),ptr(locks),ptr(wp),ptr(sp),m,4096,4096,sms,stream)==0
if a.counter_arm is not None:
 tmp.zero_();locks.zero_();call(a.counter_arm);torch.cuda.synchronize()
 (O/'counter-resources.json').write_text(json.dumps(dict(arm=a.counter_arm,resources=resources,occupancies=occupancies),indent=2)+'\n')
 raise SystemExit(0)
outputs={};timings={}
for n in [0,256,128]:
 out.fill_(float('nan'));tmp.zero_();locks.zero_();call(n);torch.cuda.synchronize();y=out[valid_gpu].clone();assert torch.isfinite(y).all();outputs[n]=y
 timings[n]=[]
 for _ in range(3):call(n)
for _ in range(9):
 for n in [0,256,128,128,256,0]:
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();call(n);en.record();en.synchronize();timings[n].append(st.elapsed_time(en))
assert torch.equal(outputs[0],outputs[256]),'offline baseline must reproduce active vendor'
comparison=dict(values=outputs[0].numel(),different=int(torch.count_nonzero(outputs[0]!=outputs[128])),max_abs=float((outputs[0].float()-outputs[128].float()).abs().max()))
# Preserve complete selected outputs. Independent equations run outside timings.
for n,y in outputs.items():y.view(torch.int16).cpu().numpy().tofile(O/f'output-{n}.bf16')
result=dict(source_library_sha256=hashlib.sha256(a.vendor.read_bytes()).hexdigest(),candidate_library_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),distinct_token_rows=np.unique(actual,axis=0).shape[0],scope='Eight actual experts, every naturally routed distinct slot from one8192-token band; original expert ordering/slot ids/M64/full-K, no repeated fabricated inputs. Other experts omitted; not a full-band or request speed claim.',occupancies=occupancies,resources=resources,comparison=comparison,median_ms={str(n):statistics.median(t)for n,t in timings.items()},timings_ms=timings,gain=statistics.median(timings[0])/statistics.median(timings[128])-1,minimum_free_bytes=torch.cuda.mem_get_info()[0],sha256=hashes)
assert result['minimum_free_bytes']>=600*1024**2
(O/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['sha256','timings_ms']}),flush=True)
