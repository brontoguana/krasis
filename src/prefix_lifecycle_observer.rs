//! Cold diagnostic observations; no token, graph replay or timing wrapper.
use cudarc::driver::sys;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
static ACTIVE: AtomicBool = AtomicBool::new(false);
static RECORDS: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());
pub fn begin() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("use the built dev lifecycle command".into());
    }
    if ACTIVE.swap(true, Ordering::SeqCst) {
        return Err("lifecycle observation already active".into());
    }
    RECORDS
        .lock()
        .map_err(|_| "lifecycle observation poisoned")?
        .clear();
    Ok(())
}
pub fn record(event: &str, address: usize, result: Option<String>) {
    if ACTIVE.load(Ordering::Relaxed) {
        let mut context = std::ptr::null_mut();
        let rc = unsafe { sys::lib().cuCtxGetCurrent(&mut context) };
        RECORDS.lock().expect("lifecycle observation poisoned").push(serde_json::json!({"event":event,"address":address,"result":result,"actual_context":context as usize,"context_status":format!("{rc:?}")}));
    }
}
pub fn finish() -> Result<Vec<serde_json::Value>, String> {
    if !ACTIVE.swap(false, Ordering::SeqCst) {
        return Err("lifecycle observation was not active".into());
    }
    Ok(std::mem::take(
        &mut *RECORDS
            .lock()
            .map_err(|_| "lifecycle observation poisoned")?,
    ))
}
pub unsafe fn destroy_graph(exec: sys::CUgraphExec) -> sys::CUresult {
    let result = sys::lib().cuGraphExecDestroy(exec);
    record(
        "actual_graph_exec_destroy",
        exec as usize,
        Some(format!("{result:?}")),
    );
    result
}
