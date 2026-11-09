"""Host TF32 split and finite-range characterization; no model admission."""
from pathlib import Path
import sys,json
import numpy as np
from prepare_replay import unpack
D=Path(sys.argv[1]);F=Path(sys.argv[2])
def parts(x):
 x=np.asarray(x,dtype='<f4');bits=x.view('<u4')
 with np.errstate(over='ignore',invalid='ignore'):
  hi=((bits+np.uint32(0x1000))&np.uint32(0xffffe000)).view('<f4')
  residual=np.subtract(x,hi,dtype=np.float32)
  lo=((residual.view('<u4')+np.uint32(0x1000))&np.uint32(0xffffe000)).view('<f4')
 return hi,lo
records=[]
for fixture in sorted(F.iterdir()):
 name=fixture.name
 arrays={'cache':unpack(fixture/'fused-0-ckv-cache.bin').astype('f4'),'query':np.fromfile(fixture/'fused-0-q-absorbed.f32',dtype='<f4'),'probability':np.fromfile(D/f'run-r3-{name}'/'candidate-probs.f32',dtype='<f4')}
 record={'fixture':name,'arrays':{}}
 for key,x in arrays.items():
  hi,lo=parts(x)
  err=np.abs(hi.astype('f8')+lo.astype('f8')-x.astype('f8'))
  record['arrays'][key]={'count':int(x.size),'max_abs_input':float(np.abs(x).max(initial=0)),'min_nonzero_abs':float(np.min(np.abs(x[x!=0]),initial=np.inf)),'max_split_error':float(err.max(initial=0)),'all_finite':bool(np.isfinite(x).all()and np.isfinite(hi).all()and np.isfinite(lo).all())}
 # A deliberately conservative bound on every four-product dot, avoiding cancellation assumptions.
 maximum={k:r['max_abs_input']for k,r in record['arrays'].items()}
 cap=json.loads((fixture/'descriptor.json').read_text())['capacity']
 record['score_absolute_sum_bound']=512*maximum['query']*maximum['cache']
 record['value_absolute_sum_bound']=cap*maximum['probability']*maximum['cache']
 record['accumulation_range_safe']=bool(max(record['score_absolute_sum_bound'],record['value_absolute_sum_bound'])<np.finfo('f4').max/16)
 records.append(record)
witnesses=[]
for label,x in [('max_finite',np.finfo('f4').max),('smallest_subnormal',np.nextafter(np.float32(0),np.float32(1)))]:
 hi,lo=parts(np.array([x],dtype='f4'))
 witnesses.append({'case':label,'input':float(x),'high':str(hi[0]),'low':str(lo[0]),'exact_reconstruction':bool(np.isfinite(hi[0])and np.isfinite(lo[0])and float(hi[0])+float(lo[0])==float(x))})
result={'all_retained_ranges_pass':all(r['accumulation_range_safe']and all(a['all_finite']for a in r['arrays'].values())for r in records),'fixtures':records,'extreme_witnesses':witnesses,'scope':'Host round-to-nearest ties-away TF32 component model; extreme finite FP32 can overflow high conversion or lose subnormals. No universal finite-input equivalence. Retained complete GPU equations/finite outputs are checked separately; hardware may flush subnormals.'}
(D/'RANGE.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_retained_ranges_pass'];print(json.dumps({'retained_ranges_pass':True,'fixtures':len(records),'extreme_witnesses':witnesses}))
