"""Disjoint GPU intervals and kernel scopes for one complete streaming request."""
import argparse,collections,json,sqlite3
from pathlib import Path

def union(intervals):
 out=[]
 for x,y in sorted(intervals):
  if y<=x:continue
  if out and x<=out[-1][1]:out[-1]=(out[-1][0],max(y,out[-1][1]))
  else:out.append((x,y))
 return out

def duration(intervals):return sum(y-x for x,y in union(intervals))

def main():
 p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('result',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
 c=sqlite3.connect('file:'+str(a.sqlite.resolve())+'?mode=ro',uri=True)
 tables={r[0]for r in c.execute("select name from sqlite_master where type='table'")}
 epoch=c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
 v=json.loads(a.result.read_text());clocks=[v['clock_before'],v['clock_after']]
 offsets=[r['realtime_ns']-r['monotonic_ns']for r in clocks];offset=sum(offsets)//2
 cv=lambda seconds:round(seconds*1e9)+offset-epoch
 lo,first,hi=map(cv,[v['started_monotonic'],v['first_monotonic'],v['ended_monotonic']])
 markers=[r[0]for r in c.execute("select k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where s.value in ('embedding_lookup','embedding_lookup_g') and k.start>=? and k.start<? order by k.start",(lo,hi))]
 bounds=[('through_first_content',lo,first),('after_first_content',first,hi),('complete_request',lo,hi)]
 if len(markers)>=63:
  markers=markers[-63:];bounds.append(('decode_steady61',markers[1],markers[-1]))
  envelope=c.execute('select min(start),max(end)from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(lo,markers[0])).fetchone()
  if envelope[0]is not None:bounds.append(('pre_decode_kernel_envelope',*envelope))
 rows=[]
 for label,start,end in bounds:
  def intervals(table):return [(max(start,x),min(end,y))for x,y in c.execute(f'select start,end from {table} where end>? and start<?',(start,end))]if table in tables else[]
  k=intervals('CUPTI_ACTIVITY_KIND_KERNEL');m=intervals('CUPTI_ACTIVITY_KIND_MEMCPY');g=intervals('CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
  kg=duration(k+g);mem=duration(m);total=duration(k+g+m);overlap=kg+mem-total
  disjoint=dict(kernels_or_graph_only_ms=(kg-overlap)/1e6,copy_only_ms=(mem-overlap)/1e6,overlap_ms=overlap/1e6,uncovered_ms=(end-start-total)/1e6)
  assert abs(sum(disjoint.values())-(end-start)/1e6)<1e-4
  kernels=[]
  for name,count,ns in c.execute('select s.value,count(*),sum(min(k.end,?)-max(k.start,?))from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.end>? and k.start<? group by s.value order by 3 desc',(end,start,start,end)):
   kernels.append(dict(name=name,count=count,summed_ms=ns/1e6))
  copies=[dict(kind=kind,count=count,bytes=nbytes)for kind,count,nbytes in c.execute('select copyKind,count(*),sum(bytes)from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<? group by copyKind',(start,end))]if'CUPTI_ACTIVITY_KIND_MEMCPY'in tables else[]
  rows.append(dict(phase=label,bounds_ns=[start,end],wall_ms=(end-start)/1e6,disjoint=disjoint,kernels=kernels,copies=copies))
 a.out.write_text(json.dumps(dict(clock_offset_range_ns=max(offsets)-min(offsets),phases=rows,caveats=['Kernel-only intervals include memory stalls and any embedded mapped reads.','Named kernel elapsed sums are investigation scopes, not additive critical-path savings.','Uncovered intervals include host, transport and untraced work; they are not necessarily idle.','Copy bytes count whole overlapping events; inspect boundary spanners for exact subinterval attribution.','The pre-decode kernel envelope can include setup/restoration kernels.']),indent=2))
if __name__=='__main__':main()
