//! Context-owned stateless BF16 RMSNorm with one batch/singleton arithmetic path.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest,Sha256};
use std::sync::Arc;
pub(crate) struct Program { _device:Arc<CudaDevice>, function:CudaFunction, thread_limit:usize, warp:usize, grid:[usize;2], shared_limit:usize, pub identity:String }
fn region(ptr:u64,bytes:usize)->Result<u64,String>{if ptr==0||ptr%2!=0||bytes==0{return Err("invalid RMSNorm region".into());}ptr.checked_add(u64::try_from(bytes).map_err(|_|"RMSNorm extent exceeds u64")?).ok_or_else(||"RMSNorm pointer overflow".into())}
impl Program {
 pub(crate) fn load(device:Arc<CudaDevice>,ptx:&str)->Result<Self,String>{
  device.bind_to_thread().map_err(|e|e.to_string())?;
  device.load_ptx(ptx.into(),"common_rmsnorm",&["common_rmsnorm_bf16"]).map_err(|e|e.to_string())?;
  let function=device.get_func("common_rmsnorm","common_rmsnorm_bf16").ok_or("common RMSNorm kernel missing")?;
  use sys::CUdevice_attribute as A;
  let attr=|a|->Result<usize,String>{let v=device.attribute(a).map_err(|e|e.to_string())?;usize::try_from(v).ok().filter(|v|*v>0).ok_or_else(||"invalid RMSNorm device limit".into())};
  let mut max_threads=0;unsafe{sys::lib().cuFuncGetAttribute(&mut max_threads,sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function.cu_function())}.result().map_err(|e|e.to_string())?;
  let limit=usize::try_from(max_threads).map_err(|_|"negative RMSNorm function limit")?.min(attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?);
  if limit==0{return Err("zero RMSNorm thread limit".into());}let thread_limit=1usize<<(usize::BITS-1-limit.leading_zeros());let warp=attr(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
  if !warp.is_power_of_two()||thread_limit<warp{return Err("RMSNorm warp/thread geometry unsupported".into());}
  let grid=[attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?];let shared_limit=attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
  Ok(Self{_device:device,function,thread_limit,warp,grid,shared_limit,identity:format!("bf16_rms_tree_f32_explicit_rne_v1:{:x}",Sha256::digest(ptx.as_bytes()))})
 }
 /// SAFETY: actual same-context input/weight/output owners and stream lease remain
 /// held through completion; output is unique. Exact in-place operation is allowed.
 pub(crate) unsafe fn enqueue(&self,output:u64,input:u64,weight:u64,width:usize,rows:usize,eps:f32,stream:sys::CUstream)->Result<(),String>{
  if width==0||width>i32::MAX as usize||rows==0||!eps.is_finite()||eps<0.||stream.is_null(){return Err("invalid common RMSNorm geometry/epsilon/stream".into());}
  let wb=width.checked_mul(2).ok_or("RMSNorm width overflow")?;let bytes=rows.checked_mul(wb).ok_or("RMSNorm row extent overflow")?;
  let oe=region(output,bytes)?;let ie=region(input,bytes)?;let we=region(weight,wb)?;
  if (output!=input&&output<ie&&input<oe)||(output<we&&weight<oe){return Err("RMSNorm output partially aliases input or weight".into());}
  let threads=width.min(self.thread_limit).next_power_of_two().max(self.warp);let shared=threads.checked_mul(4).ok_or("RMSNorm shared extent overflow")?;
  if shared>self.shared_limit{return Err("RMSNorm shared memory exceeds real device limit".into());}
  let gx=rows.min(self.grid[0]);let gy=rows.div_ceil(gx);if gy>self.grid[1]{return Err("RMSNorm rows exceed real device grid".into());}
  let(mut o,mut x,mut w,mut d,mut m,mut e)=(output,input,weight,width as i32,rows as u64,eps);
  let mut args=[(&mut o as *mut u64).cast(),(&mut x as *mut u64).cast(),(&mut w as *mut u64).cast(),(&mut d as *mut i32).cast(),(&mut m as *mut u64).cast(),(&mut e as *mut f32).cast()];
  sys::lib().cuLaunchKernel(self.function.cu_function(),gx as u32,gy as u32,1,threads as u32,1,1,shared as u32,stream,args.as_mut_ptr(),std::ptr::null_mut()).result().map_err(|e|e.to_string())
 }
}
