//! Actual device-limit crossing with retained numerical operands; no prompt generation.
use super::*;
struct Completion(Arc<CudaDevice>);
impl Drop for Completion { fn drop(&mut self) { if let Err(e)=self.0.synchronize(){eprintln!("row-grid completion failed: {e}");std::process::abort();} } }
#[path = "../../../src/common_moe_combine.rs"]
mod common_moe_combine;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GridPlan { marlin: Identity, combine: Identity, input: Identity, weights: Identity, gpu_ordinal:usize, gpu_uuid:String, safety_margin_bytes:usize }
pub fn execute(path:&Path,_out:&Path)->Result<Value,String>{
 let p:GridPlan=read_json(path)?;
 let marlin=String::from_utf8(p.marlin.read()?).map_err(|e|e.to_string())?;
 let combine=String::from_utf8(p.combine.read()?).map_err(|e|e.to_string())?;
 let source=u16s(&p.input.read()?)?;let wb=p.weights.read()?;
 if source.len()<32||wb.len()<32{return Err("retained operands too short".into());}
 let mut weights=Vec::new();for b in wb[..32].chunks_exact(4){let v=f32::from_le_bytes(b.try_into().unwrap());if !v.is_finite(){return Err("nonfinite retained route weight".into());}weights.push(v);}
 let dev=CudaDevice::new_with_stream(p.gpu_ordinal).map_err(|e|e.to_string())?;
 let mut uuid=sys::CUuuid{bytes:[0;16]};unsafe{sys::lib().cuDeviceGetUuid_v2(&mut uuid,*dev.cu_device())}.result().map_err(|e|e.to_string())?;let h=uuid.bytes.iter().map(|v|format!("{:02x}",*v as u8)).collect::<String>();
 let actual=format!("GPU-{}-{}-{}-{}-{}",&h[..8],&h[8..12],&h[12..16],&h[16..20],&h[20..]);if actual!=p.gpu_uuid{return Err("GPU identity mismatch".into());}
 let program=common_marlin::Program::load(dev.clone(),&marlin)?;let comb=common_moe_combine::Program::load(dev.clone(),&combine)?;
 let max_y=usize::try_from(dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let mut records=Vec::new();let mut min_free=usize::MAX;
 for rows in [16,max_y-1,max_y,max_y+1,max_y.checked_mul(3).and_then(|v|v.checked_add(7)).ok_or("row overflow")?]{
  let mut x=Vec::with_capacity(rows*2+32);let mut e=Vec::with_capacity(rows*8+32);let mut w=Vec::with_capacity(rows*8+32);
  for row in 0..rows{let k=row%16;x.extend_from_slice(&source[2*k..2*k+2]);for slot in 0..8{e.push(source[(k+slot)%32]);w.push(weights[slot]);}}
  x.extend([0x7fc1;32]);e.extend([0x7fc1;32]);w.extend([f32::from_bits(0x7fc10001);32]);
  let input=dev.htod_copy(x.clone()).map_err(|e|e.to_string())?;let experts=dev.htod_copy(e.clone()).map_err(|e|e.to_string())?;let weights_dev=dev.htod_copy(w.clone()).map_err(|e|e.to_string())?;
  let output=dev.htod_copy(vec![0x7fc1u16;rows+32]).map_err(|e|e.to_string())?;let routed=dev.htod_copy(vec![f32::from_bits(0x7fc10001);rows+32]).map_err(|e|e.to_string())?;let mirror=dev.htod_copy(vec![0x7fc1u16;rows+32]).map_err(|e|e.to_string())?;
  let _completion=Completion(dev.clone());let(mut free,mut total)=(0,0);unsafe{sys::lib().cuMemGetInfo_v2(&mut free,&mut total)}.result().map_err(|e|e.to_string())?;min_free=min_free.min(free);if free<p.safety_margin_bytes{return Err("row control safety margin".into());}
  unsafe{program.enqueue_activation(*input.device_ptr(),*output.device_ptr(),0,1,1,0.,false,2,1,rows,*dev.cu_stream())?;comb.enqueue_reduce(*experts.device_ptr(),*weights_dev.device_ptr(),*routed.device_ptr(),*mirror.device_ptr(),1,8,rows,1.,*dev.cu_stream())?;comb.enqueue_combine(*routed.device_ptr(),0,0,*mirror.device_ptr(),0,1,rows,1.,*dev.cu_stream())?;}
  dev.synchronize().map_err(|e|e.to_string())?;
  let y=dev.dtoh_sync_copy(&output).map_err(|e|e.to_string())?;let r=dev.dtoh_sync_copy(&routed).map_err(|e|e.to_string())?;let m=dev.dtoh_sync_copy(&mirror).map_err(|e|e.to_string())?;
  for row in 0..rows{let gate=half::bf16::from_bits(source[2*(row%16)]).to_f32().max(0.);let target=half::bf16::from_f32(gate*gate).to_bits();if y[row]!=target{return Err(format!("activation row {row}"));}let mut sum=0f32;for slot in 0..8{if weights[slot]!=0.{sum=half::bf16::from_bits(source[(row%16+slot)%32]).to_f32().mul_add(weights[slot],sum);}}if r[row].to_bits()!=sum.to_bits()||m[row]!=half::bf16::from_f32(sum).to_bits(){return Err(format!("combine row {row}"));}}
  if y[rows..].iter().any(|&v|v!=0x7fc1)||m[rows..].iter().any(|&v|v!=0x7fc1)||r[rows..].iter().any(|v|v.to_bits()!=0x7fc10001){return Err("suffix guard modified".into());}
  if dev.dtoh_sync_copy(&input).map_err(|e|e.to_string())?!=x||dev.dtoh_sync_copy(&experts).map_err(|e|e.to_string())?!=e{return Err("operand modified".into());}
  records.push(json!({"rows":rows,"completeActivationAndF32ReductionExact":true,"guardsExact":true}));
 }
 Ok(json!({"passed":true,"servingQualified":false,"gpuUuid":actual,"deviceMaxGridY":max_y,"minCompletedFreeBytes":min_free,"records":records,"scope":"Actual Y-limit crossings; distinct retained BF16 operands and route weights, complete independent CPU F32 arithmetic comparison. No model acceptance or speed measurement."}))
}
