use cudarc::driver::{sys,CudaDevice};
use std::sync::Arc;
struct Events<'a>{dev:&'a Arc<CudaDevice>,start:sys::CUevent,end:sys::CUevent}
impl Drop for Events<'_>{fn drop(&mut self){if let Err(e)=self.dev.synchronize(){eprintln!("FATAL: event completion unknown: {e}");std::process::abort();}for event in [self.start,self.end]{if !event.is_null(){if let Err(e)=unsafe{sys::lib().cuEventDestroy_v2(event)}.result(){eprintln!("FATAL: timing event teardown: {e}");std::process::abort();}}}}}
fn measure(dev:&Arc<CudaDevice>,enqueue:impl Fn()->Result<(),String>)->Result<f32,String>{
 let mut owner=Events{dev,start:std::ptr::null_mut(),end:std::ptr::null_mut()};
 unsafe{sys::lib().cuEventCreate(&mut owner.start,0)}.result().map_err(|e|e.to_string())?;
 unsafe{sys::lib().cuEventCreate(&mut owner.end,0)}.result().map_err(|e|e.to_string())?;
 super::graph_owner::completed_graph_launch(dev,||{
  unsafe{sys::lib().cuEventRecordWithFlags(owner.start,*dev.cu_stream(),sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL as u32)}.result().map_err(|e|e.to_string())?;
  enqueue()?;
  unsafe{sys::lib().cuEventRecordWithFlags(owner.end,*dev.cu_stream(),sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL as u32)}.result().map_err(|e|e.to_string())?;
  Ok(())
 })?;
 let mut ms=0.0;unsafe{sys::lib().cuEventElapsedTime(&mut ms,owner.start,owner.end)}.result().map_err(|e|e.to_string())?;
 if !ms.is_finite()||ms<=0.0{return Err("invalid actual GPU event interval".into());}Ok(ms)
}
pub(super) fn samples(dev:&Arc<CudaDevice>,enqueue:impl Fn()->Result<(),String>)->Result<serde_json::Value,String>{
 enqueue()?;dev.synchronize().map_err(|e|e.to_string())?;
 let mut repeats=1usize;let mut discovery=Vec::new();
 loop{let ms=measure(dev,||{for _ in 0..repeats{enqueue()?;}Ok(())})?;discovery.push(serde_json::json!({"repeats":repeats,"elapsedMs":ms}));if ms>=20.0{break;}repeats=repeats.checked_mul(2).ok_or("timing repeat extent overflow")?;}
 let mut values=Vec::new();for _ in 0..5{let ms=measure(dev,||{for _ in 0..repeats{enqueue()?;}Ok(())})?;values.push(ms as f64/repeats as f64);}
 let mut sorted=values.clone();sorted.sort_by(f64::total_cmp);
 Ok(serde_json::json!({"repeats":repeats,"discovery":discovery,"gpuMsPerCall":values,"medianGpuMs":sorted[sorted.len()/2],"method":"actual external CUDA events inside completed graph replay, no sanitizer; five raw readings, workload repeats measured adaptively"}))
}
