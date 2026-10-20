//! Optional diagnostic completion evidence; no synchronization or timing events.
use cudarc::driver::{sys as cu, CudaDevice};
use std::sync::Arc;

pub(super) struct Progress {
    enabled: bool,
    device: Option<Arc<CudaDevice>>,
    pending: Vec<(cu::CUevent, String)>,
}
impl Progress {
    pub(super) fn new(device: &Arc<CudaDevice>) -> Self {
        let enabled = std::env::var("KRASIS_PREFILL_COMPLETION_PROGRESS").as_deref() == Ok("1");
        Self {
            enabled,
            device: enabled.then(|| Arc::clone(device)),
            pending: Vec::new(),
        }
    }
    pub(super) fn submitted(&mut self, label: impl FnOnce() -> String) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        self.poll()?;
        eprintln!("[PREFILL-PROGRESS] host_submission {}", label());
        Ok(())
    }
    pub(super) fn recorded(
        &mut self,
        stream: cu::CUstream,
        label: impl FnOnce() -> String,
    ) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        self.device
            .as_ref()
            .unwrap()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let label = label();
        unsafe {
            let mut event = std::ptr::null_mut();
            // CU_EVENT_DISABLE_TIMING: completion only, no performance measurement.
            let result = cu::lib().cuEventCreate(&mut event, 2);
            if result != cu::CUresult::CUDA_SUCCESS {
                return Err(format!("progress event create: {result:?}"));
            }
            let result = cu::lib().cuEventRecord(event, stream);
            if result != cu::CUresult::CUDA_SUCCESS {
                cu::lib().cuEventDestroy_v2(event);
                return Err(format!("progress event record: {result:?}"));
            }
            eprintln!("[PREFILL-PROGRESS] gpu_work_submitted {label}");
            self.pending.push((event, label));
        }
        self.poll()
    }
    pub(super) fn poll(&mut self) -> Result<(), String> {
        if !self.pending.is_empty() {
            self.device
                .as_ref()
                .unwrap()
                .bind_to_thread()
                .map_err(|e| e.to_string())?;
        }
        while let Some((event, label)) = self.pending.first() {
            let result = unsafe { cu::lib().cuEventQuery(*event) };
            match result {
                cu::CUresult::CUDA_ERROR_NOT_READY => break,
                cu::CUresult::CUDA_SUCCESS => {
                    eprintln!("[PREFILL-PROGRESS] gpu_completed {label}");
                    let result = unsafe { cu::lib().cuEventDestroy_v2(*event) };
                    if result != cu::CUresult::CUDA_SUCCESS {
                        return Err(format!("progress event destroy: {result:?}"));
                    }
                    self.pending.remove(0);
                }
                _ => return Err(format!("progress event query: {result:?}")),
            }
        }
        Ok(())
    }
}
impl Drop for Progress {
    fn drop(&mut self) {
        // CUDA permits destruction of incomplete events without blocking.
        if !self.pending.is_empty() {
            if let Err(error) = self.device.as_ref().unwrap().bind_to_thread() {
                eprintln!("[PREFILL-PROGRESS] event_cleanup_context_error {error}");
                return;
            }
        }
        for (event, label) in self.pending.drain(..) {
            eprintln!("[PREFILL-PROGRESS] completion_unobserved {label}");
            let result = unsafe { cu::lib().cuEventDestroy_v2(event) };
            if result != cu::CUresult::CUDA_SUCCESS {
                eprintln!("[PREFILL-PROGRESS] event_cleanup_error {result:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_progress_neither_formats_labels_nor_uses_cuda() {
        let mut progress = Progress {
            enabled: false,
            device: None,
            pending: Vec::new(),
        };
        progress
            .submitted(|| panic!("disabled submission label evaluated"))
            .unwrap();
        progress
            .recorded(std::ptr::null_mut(), || {
                panic!("disabled completion label evaluated")
            })
            .unwrap();
        progress.poll().unwrap();
        assert!(progress.pending.is_empty());
    }
}
