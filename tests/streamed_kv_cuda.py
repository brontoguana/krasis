"""Standalone full-extent/graph/lifecycle probe; never a throughput benchmark.

Run with KRASIS_STREAMED_KV_DIAGNOSTIC=1 against an isolated native build.
"""
import gc,json,os,sys
from pathlib import Path
import torch
import importlib
StreamedKvBacking=importlib.import_module(os.environ.get('KRASIS_STREAMED_KV_PROBE_MODULE','krasis.krasis')).StreamedKvBacking
assert os.environ.get('KRASIS_STREAMED_KV_DIAGNOSTIC')=='1'
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
torch.cuda.init()
# Warm native context and Torch before attributing an allocation delta.
w=StreamedKvBacking([1,8192,16,16],0,1024);z=torch.utils.dlpack.from_dlpack(w);assert z.data_ptr()==w.pointer
del z,w;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
width=int(os.environ.get('KRASIS_STREAMED_KV_TEST_ROW_BYTES','448'));assert width in (320,448)
shape=[3,8192,16,width];rows=131072
resident=torch.zeros(shape,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
resident_free=torch.cuda.mem_get_info()[0];del resident;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
before=torch.cuda.mem_get_info()[0]
owner=StreamedKvBacking(shape,0,32768);t=torch.utils.dlpack.from_dlpack(owner);assert t.data_ptr()==owner.pointer
initial=torch.cuda.mem_get_info()[0]
view=t.reshape(3,rows,width)
values=(torch.arange(rows,device='cuda',dtype=torch.int64)%251).to(torch.uint8)
for layer in range(3):view[layer].copy_(values[:,None].expand(rows,width))
idx=torch.tensor([0,1,32767,32768,65535,65536,131071],device='cuda',dtype=torch.int64)
result=torch.empty(7,width,dtype=torch.uint8,device='cuda')
for _ in range(3):result.copy_(view[0].index_select(0,idx))
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):result.copy_(view[0].index_select(0,idx))
checks=[]
for end in [0,16,32767,32768,32769,65535,65536,131072,32768]:
 owner.diagnostic_position(end);graph.replay();torch.cuda.synchronize()
 assert torch.equal(result.cpu(),(idx.cpu()%251).to(torch.uint8)[:,None].expand(7,width))
 checks.append({'end':end,'free_bytes':torch.cuda.mem_get_info()[0],'stats':owner.stats()})
# Replay writes through the same virtual pointer while physical pages move.
write_idx=torch.tensor([13],device='cuda',dtype=torch.int64)
write_value=torch.full((1,width),14,device='cuda',dtype=torch.uint8)
for _ in range(3):view[0].index_copy_(0,write_idx,write_value)
write_graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(write_graph):view[0].index_copy_(0,write_idx,write_value)
for pos in [13,32781,65549,131070,13]:
 owner.diagnostic_position(max(pos+1,65536))
 write_idx.fill_(pos);write_value.fill_((pos%251+1)%251);write_graph.replay();torch.cuda.synchronize()
 assert torch.all(view[0,pos].cpu()==(pos%251+1)%251)
 write_value.fill_(pos%251);write_graph.replay();torch.cuda.synchronize()
 assert torch.all(view[0,pos].cpu()==pos%251)
# Compare complete backing, not only selected rows.
expected=(torch.arange(rows,dtype=torch.int64)%251).to(torch.uint8)[:,None].expand(rows,width)
for layer in range(3):assert torch.equal(view[layer].cpu(),expected)
checkpoint=t.cpu();t.zero_();owner.diagnostic_position(65536);graph.replay();torch.cuda.synchronize();assert torch.count_nonzero(result).item()==0
t.copy_(checkpoint);owner.diagnostic_position(131072);graph.replay();torch.cuda.synchronize()
assert torch.equal(result.cpu(),(idx.cpu()%251).to(torch.uint8)[:,None].expand(7,width))
for bad in [-1,131073]:
 try:owner.diagnostic_position(bad)
 except (RuntimeError,OverflowError):pass
 else:raise AssertionError('invalid position admitted')
# Tensor aliases retain the owner beyond the explicit startup keepalive.
ptr=t.data_ptr();del owner;gc.collect();graph.replay();torch.cuda.synchronize();assert t.data_ptr()==ptr
assert torch.equal(result.cpu(),(idx.cpu()%251).to(torch.uint8)[:,None].expand(7,width))
assert min(x['free_bytes'] for x in checks)>=600*1024*1024
(out/'RESULT.json').write_text(json.dumps({'pass':True,'scope':'allocation/complete bytes/old entries/window transitions/reset/restore/graph reads and writes/pointer lifetime','shape':shape,'before_free':before,'resident_free':resident_free,'initial_streamed_free':initial,'backend':'vmm','resident_allocation_delta':before-resident_free,'streamed_initial_delta':before-initial,'observations':checks,'not_qualified':['model arithmetic','request cancellation','multi-request isolation','throughput','canonical PQ/ADQ']},indent=2)+'\n')
print((out/'RESULT.json').read_text())
