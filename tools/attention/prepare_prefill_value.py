"""Tile an actual captured row, retaining its own selected indices and packed cache.
The repeated row is an explicit arithmetic probe, not a natural batch/latency claim.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('output',type=Path);a=p.parse_args();C=a.capture;O=a.output;O.mkdir(parents=True)
d=json.loads((C/'descriptor.json').read_text());cache=json.loads((C/'cache.json').read_text());assert d['kv_format']==7 and d['rope']==0 and d['heads']==64 and d['cache_dim']==512
files=[('q_absorbed.f32','q-absorbed.f32','<f4',32768),('selected.i32','selected.i32','<i4',2051),('position.i32','positions.i32','<i4',1),('attention.f32','baseline-output.f32','<f4',32768)];identities={}
for source,destination,dtype,width in files:
 path=C/source;x=np.fromfile(path,dtype=dtype);assert x.size==width;(np.tile(x,32)).tofile(O/('fused-0-'+destination));identities[source]=hashlib.sha256(path.read_bytes()).hexdigest()
assert (C/'ckv-prefix.raw').stat().st_size==cache['prefix']*448
os.link(C/'ckv-prefix.raw',O/'fused-0-ckv-cache.bin')
identities['ckv-prefix.raw']=hashlib.sha256((C/'ckv-prefix.raw').read_bytes()).hexdigest()
(O/'fixture.json').write_text(json.dumps(dict(scope=__doc__,layer=d['layer'],height=d['height'],context=cache['prefix'],rows=32,capacity=2051,source_sha256=identities),indent=2)+'\n')
