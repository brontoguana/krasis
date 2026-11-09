"""Offline actual-weight MoE block replay; repeated rows are labelled explicitly.
No model runtime, routing-policy change or request speed claim is made here.
"""
import argparse,ctypes,hashlib,json,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('library',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU already owned'
import numpy as np
import torch
C=a.capture.resolve();L=a.library.resolve();O=a.output.resolve();O.mkdir(exist_ok=False)
meta=json.loads((C/'routing.json').read_text());selected=meta['selected_experts'];E=meta['experts'];T=len(selected);sms=torch.cuda.get_device_properties(0).multi_processor_count
lib=ctypes.CDLL(str(L));fn=lib.krasis_marlin_moe_mm_bf16;P=ctypes.c_void_p;I=ctypes.c_int;B=ctypes.c_bool
fn.argtypes=[P]*15+[I,I,B,B,I,I,I,P,P,B,B,B,B,I,I,I,P,I,I,I,B,B,B,P,P];fn.restype=None
class Scalar(ctypes.Structure):
 _fields_=[('exponent',ctypes.c_uint8),('mantissa',ctypes.c_uint8),('signed',ctypes.c_uint8),('pad0',ctypes.c_uint8),('bias',ctypes.c_int32),('finite',ctypes.c_uint8),('nan',ctypes.c_uint8),('pad1',ctypes.c_uint8*2)]
q=Scalar(0,4,0,0,8,0,1,(ctypes.c_uint8*2)(0,0));assert ctypes.sizeof(q)==12
stream=P(torch.cuda.current_stream().cuda_stream);ptr=lambda t:P(t.data_ptr());identities={};results=[]
def read(path,dtype):
 identities[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return np.fromfile(path,dtype=dtype).copy()
def bf(path):return torch.from_numpy(read(path,np.uint16)).view(torch.bfloat16)
for stage,input_name in [('w1','w1-input-slots.bf16'),('w2','activation-slots.bf16')]:
 d=json.loads((C/f'{stage}-expert{selected[0]}-descriptor.json').read_text());n,k,gs=d['n'],d['k'],d['group_size'];assert d['bits']==4
 base=bf(C/input_name).reshape(T,k).cuda();allocs=[];wp=torch.zeros(E,dtype=torch.int64);sp=wp.clone()
 for eid in selected:
  w=torch.from_numpy(read(C/f'{stage}-expert{eid}-packed.bin',np.uint8)).cuda();s=bf(C/f'{stage}-expert{eid}-scales.bf16').cuda();allocs.extend([w,s]);wp[eid]=w.data_ptr();sp[eid]=s.data_ptr()
 wp=wp.cuda();sp=sp.cuda()
 for m in [1,13,31,32,33,63,64,65,127,228,512]:
  # All eight saved expert operands remain distinct; no fabricated claim of
  # natural history or routing distribution. Repetition isolates tile occupancy.
  x=base.repeat(m,1);tw=torch.ones(m*T,device='cuda');out=torch.empty((m*T,n),device='cuda',dtype=torch.bfloat16);tmp=torch.zeros(((m*T+E*64)*n,),device='cuda');ws=torch.zeros(sms*4,device='cuda',dtype=torch.int32)
  outputs=[];times=[]
  for block in [64,32]:
   ids=[];blocks=[]
   for slot,eid in enumerate(selected):
    padded=(m+block-1)//block*block;ids.extend([row*T+slot for row in range(m)]+[m*T]*(padded-m));blocks.extend([eid]*(padded//block))
   ids=torch.tensor(ids,dtype=torch.int32,device='cuda');blocks=torch.tensor(blocks,dtype=torch.int32,device='cuda');count=torch.tensor([ids.numel()],dtype=torch.int32,device='cuda')
   args=[ptr(x),P(),ptr(out),ptr(tmp),P(),P(),P(),P(),P(),P(),P(),ptr(ids),ptr(blocks),ptr(count),ptr(tw),block,1,False,False,m*T,n,k,ptr(ws),P(ctypes.addressof(q)),False,False,True,False,k//gs,gs,0,stream,-1,-1,sms,False,True,False,ptr(wp),ptr(sp)]
   def call():
    ws.zero_();tmp.zero_();fn(*args)
   for _ in range(3):call()
   torch.cuda.synchronize();measure=[]
   for _ in range(7):
    ws.zero_();tmp.zero_();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();fn(*args);end.record();end.synchronize();measure.append(start.elapsed_time(end))
   assert torch.isfinite(out).all();outputs.append(out.clone());times.append(measure)
  torch.cuda.synchronize();delta=(outputs[0].float()-outputs[1].float()).abs();free,_=torch.cuda.mem_get_info();assert free>=600*1024**2
  r=dict(stage=stage,expert_tokens=m,selected_experts=selected,shape=[m*T,n,k],block64_ms=times[0],block32_ms=times[1],speedup=statistics.median(times[0])/statistics.median(times[1]),different=int(torch.count_nonzero(delta)),values=out.numel(),max_abs=float(delta.max()),minimum_free_bytes=free,scope='Eight distinct actual expert-slot operands, repeated per occupancy; identical packed weights/scales and output slot order. Kernel-only timing excludes scratch reset.')
  results.append(r);print(json.dumps(r),flush=True)
  del x,tw,out,tmp,ws,outputs,ids,blocks,count,args
 del base,allocs,wp,sp,w,s
 torch.cuda.empty_cache()
identities[str(L)]=hashlib.sha256(L.read_bytes()).hexdigest()
(O/'RESULT.json').write_text(json.dumps(dict(results=results,sha256=identities,gpu=torch.cuda.get_device_name(0)),indent=2)+'\n')
