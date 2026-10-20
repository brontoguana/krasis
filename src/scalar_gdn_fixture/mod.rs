//! Installed diagnostic only. Real Torch owners admitted cold; entire native
//! operation/control suite runs inside Rust with the Python GIL released.
mod owner;
mod graph;
use crate::{cuda_region::CudaRegion,model_tensor_owner::ModelTensorOwner,scalar_gdn_owner::{LatestState,Owner},scalar_gdn_layout::Layout,shared_gdn_workspace::Program,stable_mla_gemm::Element};
use cudarc::driver::{CudaDevice,sys};
use pyo3::{prelude::*,exceptions::PyValueError,types::{PyBytes,PyList}};
use std::{path::Path,sync::{Arc,Mutex,OnceLock}};
static INJECT:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
pub(super) fn after_first_enqueue()->Result<(),String>{
    if INJECT.swap(false,std::sync::atomic::Ordering::Relaxed){Err("injected installed scalar fixture error after actual GPU enqueue".into())}else{Ok(())}
}
struct PoolRelease;
impl Drop for PoolRelease {
    fn drop(&mut self){
        if let Some(pool)=LATEST.get(){
            match pool.lock(){Ok(mut owners)=>owners.clear(),Err(_)=>crate::vram_monitor::fatal_cuda_context_error("scalar fixture cold-owner cleanup","owner pool mutex poisoned")}
        }
    }
}
static LATEST:OnceLock<Mutex<Vec<LatestState>>>=OnceLock::new();

pub(super) fn read_latest(d:&Arc<CudaDevice>,state:&Owner)->Result<Vec<u8>,String>{
    d.bind_to_thread().map_err(|e|e.to_string())?;
    d.synchronize().map_err(|e|e.to_string())?;
    let(start,end)=state.latest_region().extent()?;
    let bytes=usize::try_from(end-start).map_err(|e|e.to_string())?;
    let mut data=vec![0u8;bytes];
    unsafe{sys::lib().cuMemcpyDtoH_v2(data.as_mut_ptr().cast(),start,bytes)}.result().map_err(|e|e.to_string())?;
    Ok(data)
}
pub(super) fn take_latest(program:&Arc<Program>,layout:&Layout,initial:&[u8])->Result<Owner,String>{
    let latest=LATEST.get().ok_or("installed actual tensor owners not initialized")?.lock().map_err(|_|"actual tensor owner fixture pool poisoned")?.pop().ok_or("actual tensor owner fixture pool exhausted")?;
    let state=Owner::new(program.clone(),layout.clone(),latest)?;
    if read_latest(&program.device,&state)?!=initial{return Err("actual FP32 model tensor initial bytes differ from retained operand".into());}
    Ok(state)
}
fn control(mode:&str,plan:&Path,root:&Path,check:bool)->Result<(),String>{
    match mode{"owner"|"owner-error"=>owner::run(plan,root,check),"graph"=>graph::run(plan,root,check),_=>Err("installed scalar control must be owner or graph".into())}
}
#[pyfunction]
pub(crate) fn scalar_gdn_fixture_preflight(mode:&str,plan_path:&str)->PyResult<()>{
    control(mode,Path::new(plan_path),Path::new("."),true).map_err(PyValueError::new_err)
}
#[pyfunction]
pub(crate) fn scalar_gdn_fixture_installed(py:Python<'_>,mode:String,plan_path:String,output:String,factory:&Bound<'_,PyAny>)->PyResult<String>{
    scalar_gdn_fixture_preflight(&mode,&plan_path)?;
    if LATEST.get().is_some(){return Err(PyValueError::new_err("installed fixture requires a fresh process"));}
    let plan:serde_json::Value=serde_json::from_slice(&std::fs::read(&plan_path).map_err(|e|PyValueError::new_err(e.to_string()))?).map_err(|e|PyValueError::new_err(e.to_string()))?;
    let uuid=plan["gpu_uuid"].as_str().ok_or_else(||PyValueError::new_err("actual fixture GPU UUID missing"))?;
    if std::env::var("CUDA_VISIBLE_DEVICES").as_deref()!=Ok(uuid){return Err(PyValueError::new_err("select exact admitted GPU UUID before any Torch initialization"));}
    let native:serde_json::Value=serde_json::from_slice(&std::fs::read(plan["native_plan"]["path"].as_str().ok_or_else(||PyValueError::new_err("native plan absent"))?).map_err(|e|PyValueError::new_err(e.to_string()))?).map_err(|e|PyValueError::new_err(e.to_string()))?;
    let full=native["cases"].as_array().ok_or_else(||PyValueError::new_err("native cases absent"))?.iter().find(|x|x["name"]=="full").ok_or_else(||PyValueError::new_err("native full absent"))?;
    let initial=std::fs::read(full["files"]["initial-state.f32"]["path"].as_str().ok_or_else(||PyValueError::new_err("actual initial state absent"))?).map_err(|e|PyValueError::new_err(e.to_string()))?;
    // Counts are the exact simultaneously constructed fixture owners, not model
    // geometry, production allocation counts or a memory reserve.
    let count=match mode.as_str(){"owner"|"owner-error"=>8usize,"graph"=>2usize,_=>return Err(PyValueError::new_err("invalid fixture mode"))};
    let created=factory.call1((PyBytes::new(py,&initial),count))?;
    let tensors=created.downcast::<PyList>().map_err(|e|PyValueError::new_err(e.to_string()))?;
    if tensors.len()!=count{return Err(PyValueError::new_err("factory returned wrong actual tensor count"));}
    let device=CudaDevice::new_with_stream(0).map_err(|e|PyValueError::new_err(e.to_string()))?;
    let mut owners=Vec::with_capacity(count);
    for tensor in tensors.iter(){let owner=ModelTensorOwner::admit(py,&tensor,device.clone(),Element::F32)?;owners.push(LatestState::model(owner).map_err(PyValueError::new_err)?);}
    drop(created);
    LATEST.set(Mutex::new(owners)).map_err(|_|PyValueError::new_err("actual owner pool already initialized"))?;
    let release=PoolRelease;
    INJECT.store(mode=="owner-error",std::sync::atomic::Ordering::Relaxed);
    // All prepared controls, actual state copies, graphs and readbacks are Rust.
    // Native resources are constructed/dropped inside this closure, not moved
    // across threads or wrapped in per-token Python orchestration/timing.
    let outcome=py.allow_threads(||control(&mode,Path::new(&plan_path),Path::new(&output),false));
    let remaining=LATEST.get().unwrap().lock().map_err(|_|PyValueError::new_err("actual owner pool poisoned"))?.len();
    drop(release); // All unused cold tensors released also on error/unwind.
    outcome.map_err(PyValueError::new_err)?;
    if remaining!=0{return Err(PyValueError::new_err("fixture did not consume exact real-owner inventory"));}
    Ok(serde_json::json!({"passed":true,"actual_model_tensor_owners":count,"gil_released_for_entire_native_control_suite":true,"native_resources_moved_across_threads":false,"operational_serving_dispatch_changed":false}).to_string())
}

mod phase;
#[pyfunction]
pub(crate) fn scalar_gdn_phase_preflight(plan_path:&str)->PyResult<String>{
    phase::Prepared::load(Path::new(plan_path)).map(|p|p.admission().to_string()).map_err(PyValueError::new_err)
}
#[pyfunction]
pub(crate) fn scalar_gdn_phase_installed(py:Python<'_>,plan_path:String,output:String,factory:&Bound<'_,PyAny>)->PyResult<String>{
    use pyo3::types::PyTuple;
    static STARTED:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
    let prepared=phase::Prepared::load(Path::new(&plan_path)).map_err(PyValueError::new_err)?;
    if Path::new(&output).exists(){return Err(PyValueError::new_err("fresh phase output required before any CUDA initialization"));}
    if std::env::var("CUDA_VISIBLE_DEVICES").as_deref()!=Ok(prepared.uuid()){return Err(PyValueError::new_err("select admitted actual GPU UUID before any Torch initialization"));}
    if STARTED.swap(true,std::sync::atomic::Ordering::Relaxed){return Err(PyValueError::new_err("phase gate requires a fresh process"));}
    // Exact control cardinality: normal thread/phase family and a separate
    // unwind family. These are test instances, never production model budgets.
    let created=factory.call1((PyBytes::new(py,prepared.initial()),prepared.conv_shape.clone(),2usize))?;
    let tensors=created.downcast::<PyList>().map_err(|e|PyValueError::new_err(e.to_string()))?;
    if tensors.len()!=2{return Err(PyValueError::new_err("phase factory returned wrong family count"));}
    let device=CudaDevice::new_with_stream(0).map_err(|e|PyValueError::new_err(e.to_string()))?;
    let mut pairs=Vec::new();
    for pair in tensors.iter(){
        let pair=pair.downcast::<PyTuple>().map_err(|e|PyValueError::new_err(e.to_string()))?;
        if pair.len()!=2{return Err(PyValueError::new_err("phase factory must return actual latest/convolution tensor pairs"));}
        pairs.push((ModelTensorOwner::admit(py,&pair.get_item(0)?,device.clone(),Element::F32)?,ModelTensorOwner::admit(py,&pair.get_item(1)?,device.clone(),Element::F32)?));
    }
    drop(created);
    py.allow_threads(move||phase::run(prepared,Path::new(&output),device,pairs)).map(|r|r.to_string()).map_err(PyValueError::new_err)
}
