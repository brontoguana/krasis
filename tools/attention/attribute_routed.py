"""Attribute fused expert pairs from existing trace and source-order boundaries."""
import argparse,json,sqlite3
from pathlib import Path
from analyse_trace import duration,union
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('timeline',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
phase=next(v for v in json.loads(a.timeline.read_text())['phases']if v['phase']=='through_decode_entry');lo,hi=phase['bounds_ns']
c=sqlite3.connect('file:'+str(a.trace.resolve())+'?mode=ro',uri=True)
rows=c.execute('select k.start,k.end,k.streamId,k.gridX,k.blockX,k.dynamicSharedMemory,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.start<? order by k.start',(lo,hi)).fetchall()
moe=[r for r in rows if 'device::marlin_moe::Marlin<'in r[-1]]
assert len(moe)==756 and len({r[2] for r in moe})==1
stream=moe[0][2];compute=[r for r in rows if r[2]==stream];groups=[]
for band in range(9):
 for layer in range(3,45):
  i=band//4*168+(layer-3)*4+band%4 if band<8 else 336+layer-3
  # Native large chunks execute all four 8K bands within each layer.
  w1,w2=moe[2*i:2*i+2];between=[r for r in compute if w1[1]<=r[0]<w2[0]]
  assert len(between)==1 and between[0][-1]=='deepseek_v4_swiglu_batched_kernel'
  after=[r for r in compute if w2[1]<=r[0]][:3]
  assert after[0][-1].startswith('moe_scatter_weighted_runtime_kernel')
  assert [r[-1]for r in after[1:]]==['moe_add_shared_kernel','moe_accum_to_bf16_kernel']
  at=compute.index(w1);replicate=compute[at-1];assert replicate[-1]=='moe_replicate_hidden_kernel'
  previous=next(r for r in reversed(compute[:at-1])if r[-1]=='moe_finalize_sorted_kernel')
  group=dict(route_ready_ns=previous[1],replicate_start_ns=replicate[0],replicate_ms=(replicate[1]-replicate[0])/1e6,layer=layer,chunk=band//4 if band<8 else 2,band=band%4 if band<8 else 0,rows=8192 if band<8 else 13,topk=8,experts=288,gate_up=[4096,4096],down=[4096,2048],bits=4,group_size=128,block_rows=64,thread_tile=[64,256],grid=w1[3],threads=w1[4],shared_bytes=w1[5],w1_ms=(w1[1]-w1[0])/1e6,w2_ms=(w2[1]-w2[0])/1e6,activation_ms=(between[0][1]-between[0][0])/1e6,weighted_scatter_ms=(after[0][1]-after[0][0])/1e6,start_ns=w1[0],end_ns=after[-1][1])
  groups.append(group)
copy=[(s,e)for s,e in c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(lo,hi))]
intervals=[(r[0],r[1])for r in moe];total=duration(intervals);overlap=total+duration(copy)-duration(intervals+copy)
other=[(r[0],r[1])for r in rows if 'device::marlin_moe::Marlin<'not in r[-1]]
blockers=other+copy
exclusive=duration(intervals+blockers)-duration(blockers)
windows=[]
for g in groups:
 start,end=g['route_ready_ns'],g['replicate_start_ns']
 k=[(max(start,r[0]),min(end,r[1]))for r in rows if r[0]<end and r[1]>start]
 m=[(max(start,x),min(end,y))for x,y in copy if x<end and y>start]
 kg,mem,window_total=duration(k),duration(m),duration(k+m);overlap_window=kg+mem-window_total
 windows.append(dict(layer=g['layer'],chunk=g['chunk'],band=g['band'],wall_ms=(end-start)/1e6,kernel_only_ms=(kg-overlap_window)/1e6,copy_only_ms=(mem-overlap_window)/1e6,overlap_ms=overlap_window/1e6,uncovered_ms=(end-start-window_total)/1e6))
assert abs(total/1e6-sum(g['w1_ms']+g['w2_ms']for g in groups))<1e-6
assert 0<=exclusive<=total-overlap<=total
for w in windows:
 assert abs(sum(w[k]for k in ['kernel_only_ms','copy_only_ms','overlap_ms','uncovered_ms'])-w['wall_ms'])<1e-6
result=dict(projection_excluding_other_kernels_and_copies_ms=exclusive/1e6,route_to_replication_windows=windows,window_note='Source-order staging/dependency windows include host bookkeeping and concurrent shared work. Their disjoint intervals are descriptive, not an additive or causal recoverable-wait claim.',scope='Existing candidate trace, native source-order mapping. Shapes are N,K. Kernel durations are elapsed execution scopes, not additive savings.',groups=groups,totals={k:sum(g[k]for g in groups)for k in ['w1_ms','w2_ms','activation_ms','weighted_scatter_ms','replicate_ms']},projection_union_ms=total/1e6,projection_copy_overlap_ms=overlap/1e6,projection_without_copy_ms=(total-overlap)/1e6,occupancy_note='Per-expert token counts are not in this trace. Do not infer occupancy from the fixed grid. Source uses host-visible expert counts, exact-demand pointer tables and padded stable ordering.',staging_note='Copy-only, overlap and uncovered request intervals remain in the original disjoint timeline; staging is not part of the 17.83-second kernel family.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k not in ['groups','route_to_replication_windows']})
