"""Recover one-head score dispatch and logical traffic from retained trace."""
import argparse,json,sqlite3,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('timeline',type=Path);p.add_argument('out',type=Path);p.add_argument('--candidate',action='store_true');a=p.parse_args()
lo,hi=json.loads(a.timeline.read_text())['phases'][0]['bounds_ns'];c=sqlite3.connect('file:'+str(a.trace.resolve())+'?mode=ro',uri=True)
r=c.execute('select k.start,k.end,k.streamId,k.gridX,k.gridY,k.gridZ,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.start<? order by k.start',(lo,hi)).fetchall()
acc='dsa_prefill_accumulate_gemm_scores_kernel';streams={x[2]for x in r if x[-1]==acc};assert len(streams)==1;r=[x for x in r if x[2]in streams]
groups=[];current=[];score=[];selection=[]
for i,x in enumerate(r):
 if x[-1]==acc:
  assert 'gemm' in r[i-1][-1];current.append(x);score.append(r[i-1])
 if x[-1]=='dsa_prefill_topk_radix_sort_rows_kernel':
  m=x[4]
  def grids(n):
   if not a.candidate or m<64:return [((m*n+255)//256)]*32
   band=max(1,m//8);v=[]
   for b in range(0,m,band):
    count=min(band,m-b);heads=min(32,m//count);v.extend([((count*n+255)//256)]*((32+heads-1)//heads))
   return v
  possible=[n for n in [8192,16384,16387]if grids(n)==[v[3]for v in current]];assert len(possible)==1,(m,len(current),possible);n=possible[0]
  j=i;sel=[]
  while j<len(r)and r[j][-1]in ['dsa_prefill_topk_radix_sort_rows_kernel','dsa_prefill_topk_linear_merge_rows_kernel']:sel.append(r[j]);j+=1
  selection+=sel
  groups.append(dict(rows=m,context=n,accumulation_ms=sum(v[1]-v[0]for v in current)/1e6,score_ms=sum(v[1]-v[0]for v in score)/1e6,selection_ms=sum(v[1]-v[0]for v in sel)/1e6,gap_ms=(sel[-1][1]-score[0][0]-sum(v[1]-v[0]for v in current+score+sel))/1e6))
  current=[];score=[]
assert not current and len(groups)==836
# Layer streams repeat per-chunk tiles; identify each complete source chunk by row sum.
cursors=collections.defaultdict(int);valid=full=out_traffic=candidate_out=candidate_positions=candidate_calls=0
for g in groups:
 n,m=g['context'],g['rows'];start=cursors[n];position_start={8192:0,16384:32768,16387:65536}[n]+start
 v=sum(min(max(position_start+i+1,0)//4,n)for i in range(m));full+=32*m*n*4;valid+=32*v*4;out_traffic+=(2*32-1)*v*4
 band=max(1,m//8) if m>=64 else m
 for b in range(0,m,band):
  rows=min(band,m-b);heads=min(32,m//rows);passes=(32+heads-1)//heads;candidate_calls+=passes;candidate_positions+=passes*rows*n*4;vband=sum(min(max(position_start+i+1,0)//4,n)for i in range(b,b+rows));candidate_out+=(2*passes-1)*vband*4
 cursors[n]+=m
 if cursors[n]=={8192:32768,16384:32768,16387:13}[n]:cursors[n]=0
assert not any(cursors.values())
result=dict(scope='Exact-profile65K predecode trace; logical requested bytes, not measured DRAM. Same-stream dispatch gaps are not exposed request waits.',candidate_trace=a.candidate,tile_count=len(groups),accumulation_launches=(candidate_calls if a.candidate else 32*len(groups)),head_batches=[1],heads=32,head_dim=128,temporary_dtype='FP32',causal_gemm_bands=False,causal_mask='floor((position+1)/4), applied in accumulation and selection',score_gemm_ms=sum(g['score_ms']for g in groups),accumulation_ms=sum(g['accumulation_ms']for g in groups),selection_ms=sum(g['selection_ms']for g in groups),same_stream_gap_ms=sum(g['gap_ms']for g in groups),temporary_write_bytes=full,temporary_valid_read_bytes=valid,final_score_read_write_bytes=out_traffic,candidate_final_score_read_write_bytes=candidate_out,baseline_score_gemm_and_accumulation_launches_each=32*len(groups),candidate_score_gemm_and_accumulation_launches_each=candidate_calls,tiny_row_fallback_below=64,head_weight_logical_read_bytes=valid//2,position_logical_read_bytes=full,candidate_position_logical_read_bytes=candidate_positions,tiles=[dict(rows=m,context=n,count=count)for(m,n),count in collections.Counter((g['rows'],g['context'])for g in groups).items()],limits='No profiling API captures GEMM arguments. Reconstructed geometry is cross-checked against native source, saved score layouts, trace grids and32 accumulation passes. Physical bands preserve full-context GEMM extent; no causal GEMM work removal is claimed.')
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
