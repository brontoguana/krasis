//! Complete distinct retained rows; actual native RMSNorm reproduction and shared path.
use super::*;
#[path = "../../../src/common_rmsnorm.rs"] mod common_rmsnorm;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NormPlan { source:Identity, ptx:Identity, build:Identity, native_prefill:Identity, native_decode:Identity, weights:Identity, inputs:Vec<Identity>, expected_prefill:Identity, expected_decode:Identity, native_expected_gpu_uuid:String, native_prefill_threads:usize, native_decode_threads:usize, eps:f32, widths:Vec<usize>, rows:usize, partition_rows:usize, gpu_ordinal:usize, gpu_uuid:String, safety_margin_bytes:usize, raw_limit:usize }
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {fn drop(&mut self){if let Err(e)=self.0.synchronize(){eprintln!("norm completion unknown: {e}");std::process::abort();}}}
fn write_raw(out:&Path,name:&str,v:&[u16],raw:&mut usize,limit:usize)->Result<Value,String>{let b:Vec<u8>=v.iter().flat_map(|v|v.to_le_bytes()).collect();*raw=raw.checked_add(b.len()).filter(|n|*n<=limit).ok_or("norm raw limit exceeded")?;let f=out.join(name);use std::io::Write;let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&f).map_err(|e|e.to_string())?;file.write_all(&b).map_err(|e|e.to_string())?;file.sync_all().map_err(|e|e.to_string())?;Ok(json!({"file":name,"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b))}))}
fn guarded(values:&[u16])->Vec<u16>{let mut v=vec![0x7fc1;16];v.extend_from_slice(values);v.extend([0x7fc1;16]);v}
fn check_guards(v:&[u16])->Result<(),String>{if v[..16].iter().chain(v[v.len()-16..].iter()).any(|x|*x!=0x7fc1){Err("norm guard modified".into())}else{Ok(())}}
pub fn execute(path:&Path,out:&Path)->Result<Value,String>{
 let p:NormPlan=read_json(path)?;p.source.read()?;let ptx=String::from_utf8(p.ptx.read()?).map_err(|e|e.to_string())?;let build:Value=serde_json::from_slice(&p.build.read()?).map_err(|e|e.to_string())?;
 if build["passed"]!=true||build["source"]!=serde_json::to_value(&p.source).unwrap()||build["ptx"]!=serde_json::to_value(&p.ptx).unwrap(){return Err("norm compiler binding mismatch".into());}
 let weights=u16s(&p.weights.read()?)?;let input_rows=p.inputs.iter().map(|v|u16s(&v.read()?)).collect::<Result<Vec<_>,_>>()?;
 if weights.is_empty()||input_rows.is_empty()||input_rows.iter().any(|v|v.len()!=weights.len())||p.rows==0||p.partition_rows==0||p.widths.is_empty()||p.widths.iter().any(|v|*v==0||*v>weights.len()){return Err("invalid norm retained row geometry".into());}
 let dev=CudaDevice::new_with_stream(p.gpu_ordinal).map_err(|e|e.to_string())?;let mut uuid=sys::CUuuid{bytes:[0;16]};unsafe{sys::lib().cuDeviceGetUuid_v2(&mut uuid,*dev.cu_device())}.result().map_err(|e|e.to_string())?;let h=uuid.bytes.iter().map(|v|format!("{:02x}",*v as u8)).collect::<String>();let actual=format!("GPU-{}-{}-{}-{}-{}",&h[..8],&h[8..12],&h[12..16],&h[16..20],&h[20..]);if actual!=p.gpu_uuid{return Err("norm GPU UUID mismatch".into());}
 let program=common_rmsnorm::Program::load(dev.clone(),&ptx)?;
 dev.load_ptx(String::from_utf8(p.native_prefill.read()?).map_err(|e|e.to_string())?.into(),"norm_native_prefill",&["rmsnorm_batched_kernel"]).map_err(|e|e.to_string())?;
 dev.load_ptx(String::from_utf8(p.native_decode.read()?).map_err(|e|e.to_string())?.into(),"norm_native_decode",&["rmsnorm"]).map_err(|e|e.to_string())?;
 let native_pre=dev.get_func("norm_native_prefill","rmsnorm_batched_kernel").ok_or("native prefill RMSNorm missing")?;let native_dec=dev.get_func("norm_native_decode","rmsnorm").ok_or("native decode RMSNorm missing")?;
 let(mut free,mut total)=(0usize,0usize);unsafe{sys::lib().cuMemGetInfo_v2(&mut free,&mut total)}.result().map_err(|e|e.to_string())?;
 let need=p.rows.checked_mul(weights.len()).and_then(|n|n.checked_mul(6)).and_then(|n|n.checked_add(weights.len()*6+256)).ok_or("norm owner extent overflow")?;if free<p.safety_margin_bytes.checked_add(need).ok_or("norm margin overflow")?{return Err("norm actual free cannot admit explicit owners and margin".into());}
 let mut raw=0;let mut native_records=Vec::new();let mut records=Vec::new();let mut min_free=free;
 // One exact actual original row, not a replacement operand.
 {
  let x=dev.htod_copy(guarded(&input_rows[0])).map_err(|e|e.to_string())?;let w=dev.htod_copy(guarded(&weights)).map_err(|e|e.to_string())?;let y=dev.htod_copy(vec![0x7fc1u16;weights.len()+32]).map_err(|e|e.to_string())?;let _completion=Completion(dev.clone());
  for (name,function,threads,expected) in [("prefill",native_pre,p.native_prefill_threads,&p.expected_prefill),("decode",native_dec,p.native_decode_threads,&p.expected_decode)]{
   let mut max_threads=0;unsafe{sys::lib().cuFuncGetAttribute(&mut max_threads,sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function.cu_function())}.result().map_err(|e|e.to_string())?;if threads==0||threads>max_threads as usize{return Err("native norm original launch exceeds function limits".into());}
   let cfg=cudarc::driver::LaunchConfig{grid_dim:(1,1,1),block_dim:(threads as u32,1,1),shared_mem_bytes:((if name=="prefill"{threads}else{weights.len()})*4)as u32};
   unsafe{use cudarc::driver::LaunchAsync;if name=="prefill"{function.clone().launch(cfg,(*y.device_ptr()+32,*x.device_ptr()+32,*w.device_ptr()+32,weights.len()as i32,p.eps))}else{function.clone().launch(cfg,(*y.device_ptr()+32,*x.device_ptr()+32,*w.device_ptr()+32,p.eps,weights.len()as i32))}}.map_err(|e|e.to_string())?;
   let result=dev.dtoh_sync_copy(&y).map_err(|e|e.to_string())?;check_guards(&result)?;let observed=&result[16..result.len()-16];let expected_values=u16s(&expected.read()?)?;let exact=observed==expected_values;if actual==p.native_expected_gpu_uuid&&!exact{return Err(format!("actual original {name} native RMSNorm not reproduced"));}
   let file=write_raw(out,&format!("native-{name}.bf16"),observed,&mut raw,p.raw_limit)?;native_records.push(json!({"mode":name,"exactOriginalCapture":exact,"file":file}));
  }
 }
 for &width in &p.widths {
  let count=p.rows.checked_mul(width).ok_or("norm count overflow")?;let mut values=Vec::with_capacity(count);for row in 0..p.rows{values.extend_from_slice(&input_rows[row%input_rows.len()][..width]);}let host_x=guarded(&values);let host_w=guarded(&weights[..width]);
  let x=dev.htod_copy(host_x.clone()).map_err(|e|e.to_string())?;let w=dev.htod_copy(host_w.clone()).map_err(|e|e.to_string())?;let y=dev.htod_copy(vec![0x7fc1u16;count+32]).map_err(|e|e.to_string())?;let inplace=dev.htod_copy(host_x.clone()).map_err(|e|e.to_string())?;let _completion=Completion(dev.clone());
  if width==p.widths[0] {
   let (op,ip,wp)=(*y.device_ptr()+32,*x.device_ptr()+32,*w.device_ptr()+32);
   let bad=[(op,ip,wp,0,1,p.eps,*dev.cu_stream()),(op,ip,wp,1,0,p.eps,*dev.cu_stream()),(op,ip,wp,1,1,f32::NAN,*dev.cu_stream()),(op,ip,wp,1,1,-1.,*dev.cu_stream()),(op,ip,0,1,1,p.eps,*dev.cu_stream()),(op,ip,wp,1,1,p.eps,std::ptr::null_mut()),(ip+2,ip,wp,2,1,p.eps,*dev.cu_stream()),(wp,ip,wp,1,1,p.eps,*dev.cu_stream()),(op,ip,wp,2,usize::MAX,p.eps,*dev.cu_stream())];
   for(o,x,w,d,m,e,stream)in bad {if unsafe{program.enqueue(o,x,w,d,m,e,stream)}.is_ok(){return Err("invalid norm contract accepted".into());}}
  }
  let mut baseline=None;let mut modes=Vec::new();
  for mode in ["full","partitions","singletons","graph","error_recovery","inplace"]{
   unsafe{sys::lib().cuMemsetD16Async(*y.device_ptr(),0x7fc1,count+32,*dev.cu_stream())}.result().map_err(|e|e.to_string())?;
   let launch=||unsafe{program.enqueue(*y.device_ptr()+32,*x.device_ptr()+32,*w.device_ptr()+32,width,p.rows,p.eps,*dev.cu_stream())};
   match mode {
    "partitions"|"singletons"=>{let step=if mode=="singletons"{1}else{p.partition_rows};for first in (0..p.rows).step_by(step){let rows=step.min(p.rows-first);let offset=32+(first*width*2)as u64;unsafe{program.enqueue(*y.device_ptr()+offset,*x.device_ptr()+offset,*w.device_ptr()+32,width,rows,p.eps,*dev.cu_stream())}?;}},
    "graph"=>graph_owner::completed_graph_launch(&dev,launch)?,
    "error_recovery"=>{if graph_owner::completed_graph_launch(&dev,||{launch()?;Err("intentional norm captured enqueue error".into())}).is_ok(){return Err("norm capture error accepted".into());}launch()?;},
    "inplace"=>unsafe{program.enqueue(*inplace.device_ptr()+32,*inplace.device_ptr()+32,*w.device_ptr()+32,width,p.rows,p.eps,*dev.cu_stream())}?,
    _=>launch()?,
   }
   let all=if mode=="inplace"{dev.dtoh_sync_copy(&inplace)}else{dev.dtoh_sync_copy(&y)}.map_err(|e|e.to_string())?;check_guards(&all)?;let data=&all[16..all.len()-16];
   if let Some(ref expected)=baseline{if data!=expected{return Err(format!("norm {mode} differs width{width}"));}}else{baseline=Some(data.to_vec());}
   if dev.dtoh_sync_copy(&x).map_err(|e|e.to_string())?!=host_x||dev.dtoh_sync_copy(&w).map_err(|e|e.to_string())?!=host_w{return Err("norm input/weight modified".into());}
   let file=write_raw(out,&format!("width{width}-{mode}.bf16"),data,&mut raw,p.raw_limit)?;modes.push(json!({"mode":mode,"exactFull":true,"guardsAndReadonlyExact":true,"file":file}));
   unsafe{sys::lib().cuMemGetInfo_v2(&mut free,&mut total)}.result().map_err(|e|e.to_string())?;min_free=min_free.min(free);if free<p.safety_margin_bytes{return Err("norm measured margin violated".into());}
  }
  records.push(json!({"width":width,"rows":p.rows,"distinctRetainedInputs":input_rows.len(),"modes":modes}));
 }
 Ok(json!({"passed":true,"servingQualified":false,"gpuUuid":actual,"rawBytes":raw,"minCompletedFreeBytes":min_free,"native":native_records,"negativeContractsRejected":9,"records":records,"scope":"Actual native prototype row and distinct retained coefficient-preserving inputs; batch/partition/singleton/graph/capture-error/inplace/readOnly/guards. Complete nearest references independently audited; no model or speed qualification."}))
}
