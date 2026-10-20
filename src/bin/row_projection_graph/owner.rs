// Diagnostic ownership only. Arithmetic and kernel parameters are unchanged.
use cudarc::driver::{sys, CudaDevice};
use std::sync::Arc;
struct CapturedLaunch<'a> {
    device: &'a Arc<CudaDevice>,
    active: bool,
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
}
impl CapturedLaunch<'_> {
    fn close(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.active {
            let result =
                unsafe { sys::lib().cuStreamEndCapture(*self.device.cu_stream(), &mut self.graph) };
            self.active = false;
            if let Err(e) = result.result() {
                errors.push(format!("end capture: {e}"));
            }
        }
        if let Err(e) = self.device.synchronize() {
            errors.push(format!("graph completion: {e}"));
        }
        if !self.executable.is_null() {
            let result = unsafe { sys::lib().cuGraphExecDestroy(self.executable) };
            if let Err(e) = result.result() {
                errors.push(format!("destroy executable: {e}"));
            } else {
                self.executable = std::ptr::null_mut();
            }
        }
        if !self.graph.is_null() {
            let result = unsafe { sys::lib().cuGraphDestroy(self.graph) };
            if let Err(e) = result.result() {
                errors.push(format!("destroy graph: {e}"));
            } else {
                self.graph = std::ptr::null_mut();
            }
        }
        errors
    }
}
impl Drop for CapturedLaunch<'_> {
    fn drop(&mut self) {
        if self.active || !self.graph.is_null() || !self.executable.is_null() {
            for e in self.close() {
                eprintln!("ERROR: graph diagnostic teardown: {e}");
            }
        }
    }
}
pub(super) fn completed_graph_launch(
    device: &Arc<CudaDevice>,
    enqueue: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    if device.cu_stream().is_null() {
        return Err("graph diagnostic requires explicit nondefault stream".into());
    }
    device
        .synchronize()
        .map_err(|e| format!("setup completion: {e}"))?;
    let mut owner = CapturedLaunch {
        device,
        active: false,
        graph: std::ptr::null_mut(),
        executable: std::ptr::null_mut(),
    };
    unsafe {
        sys::lib().cuStreamBeginCapture_v2(
            *device.cu_stream(),
            sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    }
    .result()
    .map_err(|e| format!("begin capture: {e}"))?;
    owner.active = true;
    let mut errors = Vec::new();
    if let Err(e) = enqueue() {
        errors.push(format!("captured enqueue: {e}"));
    }
    let end = unsafe { sys::lib().cuStreamEndCapture(*device.cu_stream(), &mut owner.graph) };
    owner.active = false;
    if let Err(e) = end.result() {
        errors.push(format!("end capture: {e}"));
    }
    if errors.is_empty() && owner.graph.is_null() {
        errors.push("successful capture returned null graph".into());
    }
    if errors.is_empty() {
        let result = unsafe {
            sys::lib().cuGraphInstantiateWithFlags(&mut owner.executable, owner.graph, 0)
        };
        if let Err(e) = result.result() {
            errors.push(format!("instantiate: {e}"));
        } else if owner.executable.is_null() {
            errors.push("successful instantiation returned null executable".into());
        }
    }
    if errors.is_empty() {
        if let Err(e) =
            unsafe { sys::lib().cuGraphLaunch(owner.executable, *device.cu_stream()) }.result()
        {
            errors.push(format!("replay: {e}"));
        }
    }
    // All allocation owners in the caller remain borrowed through completion.
    errors.extend(owner.close());
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}
