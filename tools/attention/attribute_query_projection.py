"""Map native query projection groups between pooled-key construction and RoPE."""
import argparse,json,sqlite3
from pathlib import Path
from analyse_trace import duration
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('timeline',type=Path);p.add_argument('receipt',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
phase=next(v for v in json.loads(a.timeline.read_text())['phases']if v['phase']=='through_decode_entry');lo,hi=phase['bounds_ns']
receipt=json.loads(a.receipt.read_text())['receipt'];calls=receipt['indexer_projection_launches']
c=sqlite3.connect('file:'+str(a.trace.resolve())+'?mode=ro',uri=True)
rows=c.execute('select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.start<? order by k.start',(lo,hi)).fetchall()
streams={s:[]for _,_,s,_ in rows}
for row in rows:streams[row[2]].append(row)
groups=[]
for stream,items in streams.items():
 for i,op in enumerate(items):
  if op[3]!='dsa_prefill_kpool_build_kernel':continue
  j=i+1
  while j<len(items) and items[j][3]!='dsa_prefill_rope_query_bf16_kernel':j+=1
  assert j<len(items),'query RoPE missing'
  chain=items[i+1:j];assert len(chain)>=2
  head_weight=chain[-1];assert 'gemv'in head_weight[3].lower(),head_weight
  query=chain[:-1];assert all('gemm'in r[3].lower() or 'gemv'in r[3].lower() or 'reduce'in r[3].lower() for r in query),query
  groups.append(dict(start_ns=query[0][0],end_ns=query[-1][1],stream=stream,kernel_count=len(query),kernel_ms=duration([(r[0],r[1])for r in query])/1e6,span_ms=(query[-1][1]-query[0][0])/1e6,names=sorted(set(r[3]for r in query))))
groups.sort(key=lambda r:r['start_ns']);assert len(groups)==len(calls),(len(groups),len(calls))
for g,call in zip(groups,calls):
 layer,start,rows,n,k,path=call;g.update(layer=layer,chunk_start=start,rows=rows,n=n,k=k,dispatch=path)
 assert (n,k)==(4096,1536)
 if path==1:
  g['matrix_calls']=(rows+63)//64
  assert g['kernel_count']>=g['matrix_calls'],(g,call)
 else:assert g['kernel_count']==1,(g,call)
result=dict(scope='Native source boundaries plus checksum-admitted actual dispatch receipts',groups=groups,total_kernel_ms=sum(g['kernel_ms']for g in groups),total_span_ms=sum(g['span_ms']for g in groups),full_chunk_calls=sum(g['rows']==32768 for g in groups),small_chunk_calls=sum(g['rows']<64 for g in groups),overlap_note='Per-operation elapsed scopes; use the timeline disjoint union for complete-request attribution')
a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k!='groups'})
