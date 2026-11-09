"""Attribute selected-row attention from verified same-stream launch sequences."""
import argparse
import collections
import json
import sqlite3
from pathlib import Path
from analyse_trace import duration

p=argparse.ArgumentParser()
p.add_argument('trace',type=Path)
p.add_argument('timeline',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
phase=next(v for v in json.loads(a.timeline.read_text())['phases']if v['phase']=='through_decode_entry')
lo,hi=phase['bounds_ns']
c=sqlite3.connect('file:'+str(a.trace.resolve())+'?mode=ro',uri=True)
rows=c.execute('select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.end>?and k.start<?order by k.start',(lo,hi)).fetchall()
copies=[(max(lo,x),min(hi,y))for x,y in c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where end>?and start<?',(lo,hi))]
stages={name:[]for name in ['gather','score','softmax','value']}
selected=set();tiles=collections.Counter()
streams=collections.defaultdict(list)
for i,row in enumerate(rows):streams[row[2]].append(i)
legacy={'mla_prefill_gather_query_k6_f32_structured_kernel','mla_prefill_gather_query_k6_f32_kernel'}
for stream,indices in streams.items():
    for offset,i in enumerate(indices):
        name=rows[i][3]
        if name not in legacy and name not in {'mla_bf16_unpack','mla_prefill_gather_k6_no_mask'}:continue
        fused=name=='mla_prefill_gather_k6_no_mask'
        residual=name=='mla_bf16_unpack'
        if fused:
            sequence=[i];groups={'gather':[0],'score':[],'softmax':[],'value':[]}
            cursor=offset+1
            while cursor<len(indices) and rows[indices[cursor]][3]!='mla_prefill_masked_softmax_f32':
                kernel=rows[indices[cursor]][3]
                assert 'gemm'in kernel.lower() or 'splitKreduce_kernel'in kernel,kernel
                groups['score'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
            assert sum('gemm'in rows[sequence[j]][3].lower()for j in groups['score'])==1
            assert cursor<len(indices),'masked softmax missing'
            groups['softmax'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
            assert cursor<len(indices)and 'gemm'in rows[indices[cursor]][3].lower(),'value GEMM missing'
            groups['value'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
            while cursor<len(indices)and 'splitKreduce_kernel'in rows[indices[cursor]][3]:
                groups['value'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
        elif residual:
            sequence=[i];groups={'gather':[0],'score':[],'softmax':[],'value':[]}
            cursor=offset+1
            # A library GEMM may also launch a separate split-K reduction.
            while cursor<len(indices) and rows[indices[cursor]][3]!='mla_bf16_softmax':
                kernel=rows[indices[cursor]][3]
                assert 'gemm'in kernel.lower() or 'splitKreduce_kernel'in kernel,kernel
                groups['score'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
            assert sum('gemm'in rows[sequence[j]][3].lower()for j in groups['score'])==3
            assert cursor<len(indices),'softmax missing'
            groups['softmax'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
            for product in range(3):
                assert cursor<len(indices)and 'gemm'in rows[indices[cursor]][3].lower(),'value GEMM missing'
                groups['value'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
                while cursor<len(indices)and 'splitKreduce_kernel'in rows[indices[cursor]][3]:
                    groups['value'].append(len(sequence));sequence.append(indices[cursor]);cursor+=1
        else:
            sequence=indices[offset:offset+4];assert len(sequence)==4,'incomplete attention tile'
            names=[rows[j][3]for j in sequence]
            assert names[1]=='ampere_sgemm_128x128_tn'and names[2]=='mla_prefill_softmax_weights_warp_f32_kernel'and names[3]=='ampere_sgemm_128x128_nn',names
            groups={'gather':[0],'score':[1],'softmax':[2],'value':[3]}
        assert not selected.intersection(sequence),'attention tiles overlap'
        selected.update(sequence);tiles['fused_f32'if fused else'residual'if residual else'legacy']+=1
        for stage,positions in groups.items():stages[stage].extend(sequence[j]for j in positions)
other=[(max(lo,x),min(hi,y))for i,(x,y,_,_)in enumerate(rows)if i not in selected]+copies
report=[];all_intervals=[]
for name,indices in stages.items():
    intervals=[(max(lo,rows[i][0]),min(hi,rows[i][1]))for i in indices]
    total=duration(intervals);exposed=duration(intervals+other)-duration(other)
    overlap_copy=total+duration(copies)-duration(intervals+copies)
    report.append(dict(operation=name,count=len(indices),elapsed_sum_ms=sum(y-x for x,y in intervals)/1e6,
                       interval_union_ms=total/1e6,exposed_without_other_gpu_work_ms=exposed/1e6,
                       traced_copy_overlap_ms=overlap_copy/1e6,
                       kernels=dict(collections.Counter(rows[i][3]for i in indices))))
    all_intervals+=intervals
result=dict(stages=report,verified_tiles=dict(tiles),combined_interval_ms=duration(all_intervals)/1e6,
            pipeline_only_without_other_gpu_work_ms=(duration(all_intervals+other)-duration(other))/1e6,
            prefill_disjoint=phase['disjoint'],operations=dict(score='scaled Q_absorbed @ selected_K6_values.T',value='softmax_scores @ selected_K6_values'),
            interpretation='Same-stream sequences identify actual attention calls, including residual products and legacy tails. Each tile feeds value projection on the required main-stream chain. Exposure excludes concurrent traced copies and other kernels; it is not a prediction of recoverable wall time. Host launch gaps lie outside kernel intervals.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='stages'},indent=2))
