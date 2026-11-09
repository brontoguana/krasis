"""Frozen free-generation text, coding and long-history supplemental oracles."""
import argparse,json,os,re,subprocess,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--base',default='http://127.0.0.1:18375')
p.add_argument('--model',default='GLM-5.3-Flash')
p.add_argument('--tasks',type=Path,default=Path(__file__).with_name('prefill_quality_tasks.json'))
a=p.parse_args()
O=a.run;Q=O/'quality';Q.mkdir();tasks=json.loads(a.tasks.read_text());rows=[]
for task in tasks:
 body=dict(model=a.model,messages=task['messages'],temperature=0,max_tokens=task['max_tokens'],stream=False,prefix_cache=False,chat_template_kwargs=dict(enable_thinking=False),debug_first_token_boundary=True)
 (Q/(task['label']+'-request.json')).write_text(json.dumps(body)+'\n')
 start=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=1200)as f:raw=f.read()
 result=json.loads(raw);(Q/(task['label']+'-response.json')).write_bytes(raw)
 text=result['choices'][0]['message'].get('content')or''
 row=dict(label=task['label'],seconds=time.monotonic()-start,finish_reason=result['choices'][0]['finish_reason'],prompt_tokens=result.get('usage',{}).get('prompt_tokens'),text=text)
 try:
  if task['kind']=='exact':
   row['pass']=text.strip()==task['expected']
  elif task['kind']=='json':
   row['pass']=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))==task['expected']
  else:
   source=re.sub(r'^```(?:javascript|js)?\s*|\s*```$','',text.strip())
   (Q/(task['label']+'.cjs')).write_text(source+'\n')
   (Q/(task['label']+'-test.cjs')).write_text(task['test'].replace('SOLUTION',task['label']+'.cjs'))
   name='prefill-quality-'+str(os.getpid())
   try:
    process=subprocess.run(['podman','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges','--pids-limit=32','--memory=256m','--cpus=1','--user=65534:65534','--volume',str(Q.resolve())+':/task:ro','docker.io/library/node:22-bookworm-slim','node','/task/'+task['label']+'-test.cjs'],capture_output=True,text=True,timeout=60)
   except subprocess.TimeoutExpired:
    subprocess.run(['podman','rm','--force',name],capture_output=True,timeout=30);raise
   (Q/(task['label']+'-test.log')).write_text(process.stdout+process.stderr)
   row['pass']=process.returncode==0
 except Exception as e:row.update(pass_=False,error=repr(e));row['pass']=False
 rows.append(row);(Q/'RESULT.json').write_text(json.dumps(dict(scope='Supplemental frozen oracles; not ADQ or whole-model admission',rows=rows,all_pass=all(r['pass']for r in rows)),indent=2)+'\n');print(task['label'],row['pass'],repr(text[:100]),flush=True)
