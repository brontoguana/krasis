// Borrowed copy-stream completion, unadopted component. No model dispatch.
use cudarc::driver::{sys, CudaDevice};
use std::sync::Arc;
fn check(result: sys::CUresult, label: &str) -> Result<(), String> {
    result.result().map_err(|e| format!("{label}: {e}"))
}
struct CopyPhase<'a, Owners: ?Sized> {
    device: &'a Arc<CudaDevice>,
    stream: sys::CUstream,
    _retained: &'a Owners,
    completed: bool,
}
impl<Owners: ?Sized> CopyPhase<'_, Owners> {
    fn complete(&mut self) -> Result<(), String> {
        self.device
            .bind_to_thread()
            .map_err(|e| format!("copy completion context bind: {e}"))?;
        check(
            unsafe { sys::lib().cuStreamSynchronize(self.stream) },
            "copy stream completion",
        )?;
        self.completed = true;
        Ok(())
    }
}
impl<Owners: ?Sized> Drop for CopyPhase<'_, Owners> {
    fn drop(&mut self) {
        if !self.completed {
            if let Err(e) = self.complete() {
                fatal_incomplete_copy(&e);
            }
        }
    }
}
/// The retained owners must include the stream and every allocation touched by
/// the callback. Their borrows remain live through callback failure/unwinding.
///
/// # Safety
/// Caller owns this live CUDA stream in `device`'s context, and retains every
/// queued copy's host/device allocation in `retained`. Callback enqueue is unsafe
/// at its actual FFI sites. This does not replace outer request admission/poison.
pub unsafe fn with_copy_completion<Owners: ?Sized, T>(
    device: &Arc<CudaDevice>,
    stream: sys::CUstream,
    retained: &Owners,
    callback: impl FnOnce(&Owners) -> Result<T, String>,
) -> Result<T, String> {
    if stream.is_null() {
        return Err("copy phase requires an owned nondefault stream".into());
    }
    device
        .bind_to_thread()
        .map_err(|e| format!("copy phase context bind: {e}"))?;
    let (mut current, mut stream_ctx) = (std::ptr::null_mut(), std::ptr::null_mut());
    check(
        unsafe { sys::lib().cuCtxGetCurrent(&mut current) },
        "copy current context",
    )?;
    check(
        unsafe { sys::lib().cuStreamGetCtx(stream, &mut stream_ctx) },
        "copy stream context",
    )?;
    if current != *device.cu_primary_ctx() || stream_ctx != current {
        return Err("copy phase stream/context mismatch".into());
    }
    let mut phase = CopyPhase {
        device,
        stream,
        _retained: retained,
        completed: false,
    };
    let result = callback(retained);
    let completion = phase.complete();
    if let Err(error) = &completion {
        fatal_incomplete_copy(error);
    }
    join_results(result, completion)
}
fn fatal_incomplete_copy(error: &str) -> ! {
    eprintln!("KRASIS FATAL: copy completion is unknown: {error}. Owners cannot be safely released; restart Krasis to recover.");
    crate::vram_monitor::fatal_cuda_context_error("sequence copy completion", error)
}
fn join_results<T>(result: Result<T, String>, completion: Result<(), String>) -> Result<T, String> {
    match (result, completion) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(original), Err(completion)) => Err(format!("{original}; {completion}")),
    }
}

// Pure error policy controls; they do not claim an injected CUDA device failure.
pub fn error_contract_controls() -> Result<(), String> {
    let bad = join_results::<()>(Err("callback".into()), Err("completion".into()));
    let complete = join_results(Ok(()), Err("completion".into()));
    let callback = join_results::<()>(Err("callback".into()), Ok(()));
    if bad.as_ref().map(|_| ()).map_err(String::as_str) != Err("callback; completion")
        || complete.as_ref().map(|_| ()).map_err(String::as_str) != Err("completion")
        || callback.as_ref().map(|_| ()).map_err(String::as_str) != Err("callback")
    {
        return Err("completion error contract changed".into());
    }
    Ok(())
}

#[cfg(test)]
mod completion_policy_tests {
    #[test]
    fn unknown_completion_exits_before_borrowed_owner_release() {
        let key = "KRASIS_TEST_COPY_FATAL_DROP";
        if let Some(path) = std::env::var_os(key) {
            struct Witness(std::path::PathBuf);
            impl Drop for Witness {
                fn drop(&mut self) {
                    std::fs::write(&self.0, b"owner released").unwrap();
                }
            }
            let _owner = Witness(path.into());
            super::fatal_incomplete_copy(
                "synthetic completion-policy control; no CUDA device failure injected",
            );
        }
        let path = std::env::temp_dir().join(format!(
            "krasis-copy-fatal-drop-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!(
                    "{}::unknown_completion_exits_before_borrowed_owner_release",
                    module_path!().split_once("::").unwrap().1
                ),
                "--nocapture",
            ])
            .env(key, &path)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(crate::vram_monitor::CUDA_CONTEXT_POISONED_EXIT_CODE)
        );
        assert!(
            !path.exists(),
            "owner destructor ran after unknown completion"
        );
        assert!(String::from_utf8_lossy(&result.stderr).contains("copy completion is unknown"));
    }
}
