"""Attribute static local-memory instructions to compiler source locations.
Counts are static instructions, not execution counts or measured DRAM traffic.
"""
import argparse,collections,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('disassembly',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
counts=collections.defaultdict(collections.Counter);fn='';location=('unknown',0)
for line in a.disassembly.read_text().splitlines():
 if '.section\t.text.' in line:
  fn=line.split('.text.')[1].split(',')[0]
 m=re.search(r'//## File "([^"]+)", line (\d+)',line)
 if m:location=(Path(m[1]).name,int(m[2]))
 op=re.search(r'\b(LDL|STL)(?:\.| )',line)
 if op:counts[fn][(*location,op[1])]+=1
result=dict(scope='Static local-memory SASS instruction/source attribution only. Compiler locations may be inlined; no dynamic spill-hit, bandwidth or DRAM claim.',functions={fn:[dict(file=f,line=l,operation=op,instructions=count)for(f,l,op),count in c.most_common()]for fn,c in counts.items()})
a.output.write_text(json.dumps(result,indent=2)+'\n')
