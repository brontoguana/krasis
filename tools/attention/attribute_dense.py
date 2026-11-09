"""Map a retained large BF16 GEMM family using complete source-order anchors."""
import argparse,collections,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('timeline',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
phase=next(p for p in json.loads(a.timeline.read_text())['phases']if p['phase']=='through_decode_entry');lo,hi=phase['bounds_ns']
c=sqlite3.connect('file:'+str(a.trace.resolve())+'?mode=ro',uri=True)
rows=c.execute('select k.start,k.end,k.streamId,k.gridX,k.gridY,k.registersPerThread,k.dynamicSharedMemory,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.start<? order by k.start',(lo,hi)).fetchall()
family=lambda v:'bf16_256x128_32x3_tn_align8' in v[-1]
streams={v[2]for v in rows if family(v)};assert len(streams)==1
r=[v for v in rows if v[2]in streams];tag={}
for i,v in enumerate(r):
 if v[-1]=='kimi_delta_conv_silu_parallel_bf16' and(i<2 or r[i-2][-1]!='kimi_delta_conv_silu_parallel_bf16'):
  q=[j for j in range(i-1,max(-1,i-10),-1)if family(r[j])]
  if len(q)>=3:
   for name,j in zip(['v','k','q'],q[:3]):assert j not in tag;tag[j]='kda_'+name
 if v[-1]=='kimi_delta_rmsnorm_gate_bf16':
  for j in range(i+1,min(len(r),i+5)):
   if family(r[j]):assert j not in tag;tag[j]='kda_out';break
 if v[-1]=='gated_activation_split_batched_kernel':
  before=[j for j in range(i-1,max(-1,i-4),-1)if family(r[j])]
  if len(before)==2:
   for name,j in zip(['up','gate'],before):assert j not in tag;tag[j]='dense_mlp_'+name
   after=[j for j in range(i+1,min(len(r),i+3))if family(r[j])];assert len(after)==1;tag[after[0]]='dense_mlp_down'
remaining=[i for i,v in enumerate(r)if family(v)and i not in tag];assert len(remaining)==44
for j,i in enumerate(remaining):
 name='mla_q_b'if j%2==0 else'mla_out'
 assert r[i-1][-1]=='hqq_dequant_bf16_kernel'
 assert r[i][3:5]==(2048,8 if j%2==0 else 2)
 if j%2:assert r[i+1][-1]=='deepseek_v4_hc_post_kernel'
 tag[i]=name
expected={**{f'kda_{n}':68 for n in ['q','k','v','out']},**{f'dense_mlp_{n}':6 for n in ['gate','up','down']},'mla_q_b':22,'mla_out':22}
assert dict(collections.Counter(tag.values()))==expected
assert len(tag)==sum(family(v)for v in r)==334
shapes={**{f'kda_{n}':[32768,8192,4096]for n in ['q','k','v']},'kda_out':[32768,4096,8192],'dense_mlp_gate':[32768,12288,4096],'dense_mlp_up':[32768,12288,4096],'dense_mlp_down':[32768,4096,12288],'mla_q_b':[32768,16384,1536],'mla_out':[32768,4096,16384]}
groups={}
for name in expected:
 subset=[r[i]for i in tag if tag[i]==name]
 groups[name]=dict(count=len(subset),milliseconds=sum(v[1]-v[0]for v in subset)/1e6,shape_mnk=shapes[name],registers=sorted({v[5]for v in subset}),dynamic_shared=sorted({v[6]for v in subset}),preceded_by_hqq_dequant=sum(r[i-1][-1]=='hqq_dequant_bf16_kernel'for i in tag if tag[i]==name))
result=dict(scope='Retained exact-profile65K trace; no new profiling or request timing. Kernel-name dispatch combined with source ordering, convolution/RMS-gate/MLP anchors and launch geometry. Shapes are source/config-derived, not profiler API-argument capture.',kernel=next(v[-1]for v in r if family(v)),groups=groups,total_ms=sum(v['milliseconds']for v in groups.values()),count=334,dispatch=dict(operand_dtype='BF16',output_dtype='BF16',compute='CUBLAS_COMPUTE_32F',algorithm='CUBLAS_GEMM_DEFAULT',operations='CUBLAS_OP_T / CUBLAS_OP_N',hqq='Materialize HQQ6 group128 weights into existing BF16 scratch before GEMM; dense MLP uses retained BF16 weights.',epilogue='No activation requested by gemm_ex. The relu substring is in the generic compiled kernel name; actual nonlinearities are separate native kernels.'),next_intervention='Prioritise KDA Q/K/V projection dispatch(3.556s), preserving the HQQ6-to-BF16 reconstruction and BF16 output boundary. Capture actual full Q/K/V operands and test one alternative tensor-core plan with exact-output screening; no arithmetic substitution is qualified here.',limits='Elapsed kernel scopes are not additive savings. No changed dense dispatch or new quality claim. Tiny13-row tail dispatches to other kernel names and is outside this334-launch family.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(total_ms=result['total_ms'],groups=groups),indent=2))
