"""Tail/masking stress cases derived from real queries and packed cache bytes."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from prepare_replay import oracle
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir()
ids=np.fromfile(a.input/'indices.i32',dtype='<i4');cases=[]
for capacity,context in [(1,32000),(3,32000),(127,32000),(128,32000),(129,32000),(2047,32000),(2048,32000),(2051,32000),(4096,32000),(2051,3),(2051,127),(2051,0)]:
 O=a.out/f'c{capacity}-n{context}';O.mkdir()
 for name in ['q.f32','cache.bin']:shutil.copyfile(a.input/name,O/name)
 selected=np.resize(ids,capacity).copy()
 if context<32000 and context:selected=np.arange(capacity,dtype='<i4')%context
 if capacity>=3:selected[-1]=-1;selected[-2]=context;selected[1]=selected[0]
 selected.astype('<i4').tofile(O/'indices.i32');oracle(O,context)
 cases.append(dict(directory=str(O.resolve()),context=context,capacity=capacity,scope='derived selection/masking stress; actual captured query and packed cache bytes'))
(a.out/'cases.json').write_text(json.dumps(cases,indent=2))
