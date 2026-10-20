#[path="../../bf16-projection-probe/src/graph_owner.rs"]mod graph_owner;
// Fixture adapter exercising the real shared production candidate, not a copy of Lt enqueue.
#[path="../../../src/stable_mla_gemm.rs"]mod stable;
use cudarc::driver::{CudaDevice,CudaSlice,DeviceSlice};
use serde_json::{json,Value};use std::{cell::RefCell,fs,path::Path,sync::Arc};
pub(super) struct Probe{device:Arc<CudaDevice>,ar:usize,ac:usize,br:usize,columns:usize,discovery_columns:usize,cr:usize,score:bool,interleaved_b:bool,alpha:f32,beta:f32,candidates:RefCell<Option<stable::Candidates>>}
impl Probe{
 pub(super) fn new(device:Arc<CudaDevice>,ccd:usize,heads:usize,selected:usize,scale:f32,score:bool)->Result<Self,String>{
  Ok(Self{device,ar:ccd,ac:selected,br:if score{ccd}else{selected},columns:heads,discovery_columns:heads,cr:if score{selected}else{ccd},score,interleaved_b:false,alpha:if score{scale}else{1.0},beta:if score{1.0}else{0.0},candidates:RefCell::new(None)})
 }
 pub(super) fn set_interleaved_b(&mut self){self.interleaved_b=true;}
 pub(super) fn set_columns(&mut self,columns:usize)->Result<(),String>{if columns==0{return Err("zero columns".into())}self.columns=columns;Ok(())}
 fn shape(&self,batch:usize,columns:usize)->Result<stable::Shape,String>{
  let matrix=|rows:usize,cols:usize|->Result<stable::Matrix,String>{Ok(stable::Matrix{rows,cols,leading:rows,batch,stride:rows.checked_mul(cols).ok_or("fixture stride overflow")?})};
  let mut b=matrix(self.br,columns)?;if self.interleaved_b{b.leading=self.br.checked_mul(batch).ok_or("interleaved B leading overflow")?;b.stride=self.br;}
  let shape=stable::Shape{a:matrix(self.ar,self.ac)?,b,c:matrix(self.cr,columns)?,transpose_a:self.score,transpose_b:false};shape.validate()?;Ok(shape)
 }
 pub(super) fn run(&self,a:&CudaSlice<f32>,b:&CudaSlice<f32>,initial:&[f32],batch:usize,root:&Path,label:&str,retain_all:bool,profile:bool,controls:bool)->Result<Value,String>{
  if profile{return Err("new shared adapter requires its own cost qualification; old prototype timings are not reused".into())}
  let shape=self.shape(batch,self.columns)?;if initial.len()!=shape.c.elements(true)?{return Err("fixture C physical extent".into())}
  let input=self.device.htod_sync_copy(initial).map_err(|e|e.to_string())?;let output=self.device.alloc_zeros::<f32>(initial.len()).map_err(|e|e.to_string())?;
  let av=stable::Span::f32(a,0,a.len())?;let bv=stable::Span::f32(b,0,b.len())?;let cv=stable::Span::f32(&input,0,input.len())?;let ov=stable::Span::f32(&output,0,output.len())?;
  let mut candidates=self.candidates.borrow_mut();if candidates.is_none(){
   let discover=self.shape(2,self.discovery_columns)?;let mut aligns=[av.alignment()?,bv.alignment()?,cv.alignment()?,ov.alignment()?];
   for(i,m)in [discover.a,discover.b,discover.c,discover.c].into_iter().enumerate(){let stride=m.stride.checked_mul(std::mem::size_of::<f32>()).ok_or("fixture alignment extent")?;if stride>0{let low=stride&stride.wrapping_neg();aligns[i]=aligns[i].min(low);}while aligns[i]>u32::MAX as usize{aligns[i]/=2;}}
   *candidates=Some(stable::Candidates::discover(self.device.clone(),discover,0,aligns)?);
  }
  let mut cases=Vec::new();for(index,candidate)in candidates.as_ref().ok_or("candidate set absent")?.values.iter().enumerate(){
   let algorithm=candidate.configuration();let prepared=match candidate.prepare(shape,0){Ok(p)=>p,Err(e)=>{cases.push(json!({"index":index,"algo":algorithm,"supported":false,"status":e}));continue;}};
   unsafe{prepared.enqueue(&av,&bv,&cv,&ov,None,self.alpha,self.beta,*self.device.cu_stream())?;}
   self.device.synchronize().map_err(|e|e.to_string())?;let values=self.device.dtoh_sync_copy(&output).map_err(|e|e.to_string())?;if values.iter().any(|v|v.is_nan()||v.is_infinite()&&*v!=f32::NEG_INFINITY){return Err("invalid shared Lt output".into())}
   let checks=if controls&&(batch==1||self.columns==1){
    let bad=stable::Span::f32(&output,0,output.len().checked_sub(1).ok_or("negative extent")?)?;
    let capacity=unsafe{prepared.enqueue(&av,&bv,&cv,&bad,None,self.alpha,self.beta,*self.device.cu_stream())}.err().ok_or("undersized output was admitted")?;if !capacity.contains("capacity"){return Err(format!("wrong capacity rejection: {capacity}"))}
    let alias=unsafe{prepared.enqueue(&av,&bv,&cv,&av,None,self.alpha,self.beta,*self.device.cu_stream())}.err().ok_or("output alias was admitted")?;if !alias.contains("aliases"){return Err(format!("wrong alias rejection: {alias}"))}
    let byte_count=output.len().checked_mul(std::mem::size_of::<f32>()).ok_or("byte view extent")?;let unaligned_owner=self.device.alloc_zeros::<u8>(byte_count.checked_add(1).ok_or("byte guard overflow")?).map_err(|e|e.to_string())?;let alignment=stable::Span::f32_bytes(&unaligned_owner,1,output.len()).err().ok_or("unaligned F32 workspace was admitted")?;
    let raw=stable::Span::bytes(&unaligned_owner,0,byte_count)?;
    let dtype=unsafe{prepared.enqueue(&av,&bv,&cv,&raw,None,self.alpha,self.beta,*self.device.cu_stream())}.err().ok_or("raw bytes admitted as F32 output")?;if !dtype.contains("dtype"){return Err(format!("wrong raw dtype rejection: {dtype}"))}if !alignment.contains("alignment"){return Err(format!("wrong unaligned rejection: {alignment}"))}
    let nonfinite=unsafe{prepared.enqueue(&av,&bv,&cv,&ov,None,f32::NAN,self.beta,*self.device.cu_stream())}.err().ok_or("nonfinite scale admitted")?;if !nonfinite.contains("nonfinite"){return Err("wrong nonfinite rejection".into())}
    let a_before=self.device.dtoh_sync_copy(a).map_err(|e|e.to_string())?;let b_before=self.device.dtoh_sync_copy(b).map_err(|e|e.to_string())?;
    graph_owner::completed_graph_launch(&self.device,||unsafe{prepared.enqueue(&av,&bv,&cv,&ov,None,self.alpha,self.beta,*self.device.cu_stream())})?;
    let after_graph=self.device.dtoh_sync_copy(&output).map_err(|e|e.to_string())?;if super::serial(&after_graph)!=super::serial(&values){return Err("shared Lt graph differs from eager".into())}
    let error=graph_owner::completed_graph_launch(&self.device,||{unsafe{prepared.enqueue(&av,&bv,&cv,&ov,None,self.alpha,self.beta,*self.device.cu_stream())?;}Err("injected callback error after actual captured Lt enqueue".into())}).err().ok_or("failed graph callback succeeded")?;if !error.contains("injected callback error"){return Err(format!("callback error was lost: {error}"))}
    if super::serial(&self.device.dtoh_sync_copy(&output).map_err(|e|e.to_string())?)!=super::serial(&values){return Err("rejected capture changed output".into())}
    let unwind=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{let _=graph_owner::completed_graph_launch(&self.device,||->Result<(),String>{unsafe{prepared.enqueue(&av,&bv,&cv,&ov,None,self.alpha,self.beta,*self.device.cu_stream())?;}panic!("injected captured Lt callback unwind")});}));if unwind.is_ok(){return Err("callback unwind did not occur".into())}
    graph_owner::completed_graph_launch(&self.device,||unsafe{prepared.enqueue(&av,&bv,&cv,&ov,None,self.alpha,self.beta,*self.device.cu_stream())})?;
    self.device.synchronize().map_err(|e|e.to_string())?;
    if super::serial(&self.device.dtoh_sync_copy(&output).map_err(|e|e.to_string())?)!=super::serial(&values)||super::serial(&self.device.dtoh_sync_copy(a).map_err(|e|e.to_string())?)!=super::serial(&a_before)||super::serial(&self.device.dtoh_sync_copy(b).map_err(|e|e.to_string())?)!=super::serial(&b_before)||super::serial(&self.device.dtoh_sync_copy(&input).map_err(|e|e.to_string())?)!=super::serial(initial){return Err("graph/error recovery mutated result or readonly operands".into())}
    Some(json!({"nondefault_graph_exact":true,"capacity_rejected":capacity,"alias_rejected":alias,"alignment_rejected":alignment,"raw_dtype_rejected":dtype,"nonfinite_rejected":true,"captured_callback_error_preserved":true,"failed_capture_output_unchanged":true,"captured_callback_unwind_cleanup":true,"fresh_graph_recovery_exact":true,"all_readonly_operands_exact":true,"scope":"Real prepared plan/actual allocations borrowed through completed capture/replay/error/unwind. No model snapshot/restore/sequence poison qualification and no injected CUDA device fault."}))
   }else{None};
   let row=shape.c.rows.checked_mul(shape.c.cols).ok_or("fixture row extent")?;let first=&values[..row];let all_same=values.chunks_exact(row).all(|x|x.iter().zip(first).all(|(a,b)|a.to_bits()==b.to_bits()));let raw=super::serial(first);let name=format!("batch{batch}-{label}-algo{index}.f32");fs::write(root.join(&name),&raw).map_err(|e|e.to_string())?;
   let all_file=if retain_all{let name=format!("batch{batch}-{label}-algo{index}-all.f32");fs::write(root.join(&name),super::serial(&values)).map_err(|e|e.to_string())?;Some(name)}else{None};
   cases.push(json!({"index":index,"algo":algorithm,"supported":true,"all_rows_equal":all_same,"first_row_file":name,"first_row_sha256":super::hash(&raw),"all_output_file":all_file,"all_output_sha256":super::hash(&super::serial(&values)),"controls":checks,"prepared_identity":prepared.identity,"workspace_bytes":prepared.workspace_bytes}));
  }
  Ok(json!({"stage":label,"batch":batch,"candidate_discovery_batch":2,"workspace_bytes":0,"shared_adapter":true,"cases":cases}))
 }
}
