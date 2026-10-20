//! Actual compute-stream owners and whole-invocation ordering, never raw swaps.
use crate::{gpu_prefill::PrefillComputeStream, vram_monitor::fatal_cuda_context_error};
use cudarc::driver::{sys, CudaDevice};
use std::sync::{Arc, Mutex};
fn check(code: sys::CUresult, label: &str) -> Result<(), String> {
    code.result().map_err(|e| format!("{label}: {e}"))
}
#[derive(Clone)]
pub(crate) enum Stream {
    Decode(Arc<CudaDevice>),
    Prefill(Arc<PrefillComputeStream>),
}
impl Stream {
    pub(crate) fn device(&self) -> Arc<CudaDevice> {
        match self {
            Self::Decode(d) => d.clone(),
            Self::Prefill(p) => p.device(),
        }
    }
    pub(crate) fn handle(&self) -> sys::CUstream {
        match self {
            Self::Decode(d) => *d.cu_stream(),
            Self::Prefill(p) => p.handle(),
        }
    }
    pub(crate) fn context(&self) -> sys::CUcontext {
        *self.device().cu_primary_ctx()
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Decode(a), Self::Decode(b)) => Arc::ptr_eq(a, b),
            (Self::Prefill(a), Self::Prefill(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    pub(crate) fn admit(&self, device: &Arc<CudaDevice>) -> Result<(), String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        if self.handle().is_null() || self.context() != *device.cu_primary_ctx() {
            return Err(
                "scalar phase requires a real owned nondefault stream on actual program context"
                    .into(),
            );
        }
        let mut context = std::ptr::null_mut();
        check(
            unsafe { sys::lib().cuStreamGetCtx(self.handle(), &mut context) },
            "scalar actual phase stream context",
        )?;
        if context != *device.cu_primary_ctx() {
            return Err("scalar phase stream owner/driver context mismatch".into());
        }
        Ok(())
    }
    pub(crate) fn drain(&self) -> Result<(), String> {
        self.device().bind_to_thread().map_err(|e| e.to_string())?;
        check(
            unsafe { sys::lib().cuStreamSynchronize(self.handle()) },
            "scalar phase stream completion",
        )
    }
    pub(crate) fn drain_or_fatal(&self, label: &str) {
        if let Err(e) = self.drain() {
            fatal_cuda_context_error(label, &e);
        }
    }
}

pub(crate) struct Event {
    handle: sys::CUevent,
    // Last: real context remains alive through event destruction.
    device: Arc<CudaDevice>,
}
// SAFETY: event is an opaque CUDA driver resource, with immutable context and
// retained device. Driver calls bind that context. Record/wait sequencing is
// controlled by cold construction or Execution's mutex, not shared host fields.
// Last-owner destruction cannot overlap a Rust borrow of this event.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}
impl Event {
    pub(crate) fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut handle = std::ptr::null_mut();
        check(
            unsafe {
                sys::lib().cuEventCreate(
                    &mut handle,
                    sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                )
            },
            "scalar phase completion event create",
        )?;
        Ok(Self { handle, device })
    }
    fn same_context(&self, stream: &Stream) -> Result<(), String> {
        if stream.context() != *self.device.cu_primary_ctx() {
            return Err("scalar event/stream actual context differs".into());
        }
        self.device.bind_to_thread().map_err(|e| e.to_string())
    }
    pub(crate) fn record(&self, stream: &Stream) -> Result<(), String> {
        self.same_context(stream)?;
        check(
            unsafe { sys::lib().cuEventRecord(self.handle, stream.handle()) },
            "scalar phase completion record",
        )
    }
    pub(crate) fn wait(&self, stream: &Stream) -> Result<(), String> {
        self.same_context(stream)?;
        check(
            unsafe { sys::lib().cuStreamWaitEvent(stream.handle(), self.handle, 0) },
            "scalar preceding phase wait",
        )
    }
    fn complete(&self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        check(
            unsafe { sys::lib().cuEventSynchronize(self.handle) },
            "scalar recorded phase completion",
        )
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        if let Err(e) = self
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())
            .and_then(|_| {
                check(
                    unsafe { sys::lib().cuEventDestroy_v2(self.handle) },
                    "scalar phase event release",
                )
            })
        {
            fatal_cuda_context_error("scalar event release unknown", &e);
        }
    }
}

struct Timeline {
    streams: Vec<Stream>,
    recorded: bool,
    poisoned: bool,
    // Event owns its actual device until all timeline state is released.
    event: Event,
}
impl Timeline {
    fn drain_all(&self) -> Result<(), String> {
        let mut error = None;
        if self.recorded {
            if let Err(e) = self.event.complete() {
                error = Some(e);
            }
        }
        for stream in &self.streams {
            if let Err(e) = stream.drain() {
                error.get_or_insert(e);
            }
        }
        error.map_or(Ok(()), Err)
    }
    fn drain_or_fatal(&self, label: &str) {
        if let Err(e) = self.drain_all() {
            fatal_cuda_context_error(label, &e);
        }
    }
}
/// Serializes one device's submitted phases; no cross-device/global lock and no
/// per-layer host wait. Wrap capture+replay OUTSIDE CUDA stream capture.
pub(crate) struct Execution {
    inner: Mutex<Timeline>,
}
impl Execution {
    pub(crate) fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        let stream = Stream::Decode(device.clone());
        stream.admit(&device)?;
        Ok(Self {
            inner: Mutex::new(Timeline {
                streams: vec![stream],
                recorded: false,
                poisoned: false,
                event: Event::new(device)?,
            }),
        })
    }
    pub(crate) fn admit_prefill(&mut self, owner: Arc<PrefillComputeStream>) -> Result<(), String> {
        let phase = Stream::Prefill(owner);
        let state = self
            .inner
            .get_mut()
            .map_err(|_| "scalar execution mutex poisoned")?;
        if state.recorded || state.poisoned {
            return Err("scalar phase stream must be admitted before execution/calibration".into());
        }
        phase.admit(&state.event.device)?;
        if state.streams.iter().any(|s| s.same(&phase)) {
            return Err("scalar prefill phase already admitted".into());
        }
        state.streams.push(phase);
        Ok(())
    }
    /// # Safety
    /// All resources referenced by submitted work must be retained OUTSIDE work
    /// through this boundary. Any closure-local resource must itself drain on
    /// error/unwind before destruction. Capture cleanup also belongs to work.
    pub(crate) unsafe fn execute<T>(
        &self,
        stream: &Stream,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar execution phase held/poisoned")?;
        if !state.streams.iter().any(|s| s.same(stream)) {
            return Err("scalar execution actual stream owner not admitted".into());
        }
        stream
            .device()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        if state.poisoned {
            return Err("scalar execution requires explicit verified reset/restore".into());
        }
        let mut capture = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        check(
            unsafe { sys::lib().cuStreamIsCapturing(stream.handle(), &mut capture) },
            "scalar outer execution capture check",
        )?;
        if capture != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err("scalar execution domain must wrap, not run inside, graph capture".into());
        }
        if state.recorded {
            state.event.wait(stream)?;
        }
        state.poisoned = true;
        // Captured/borrowed owners must outlive this entire boundary, including
        // unwind cleanup; closure-local owners need their own completion guard.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        // Capture callback must have ended/cleaned capture before this boundary.
        if let Err(e) = check(
            unsafe { sys::lib().cuStreamIsCapturing(stream.handle(), &mut capture) },
            "scalar outer execution final capture check",
        ) {
            state.drain_or_fatal("scalar failed final capture query");
            match result {
                Err(panic) => {
                    drop(state);
                    std::panic::resume_unwind(panic)
                }
                _ => return Err(e),
            }
        }
        if capture != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            fatal_cuda_context_error(
                "scalar callback left stream capture active",
                "cannot release phase owners during unfinished capture",
            );
        }
        if let Err(e) = state.event.record(stream) {
            state.drain_or_fatal("scalar failed phase completion");
            match result {
                Err(panic) => {
                    drop(state);
                    std::panic::resume_unwind(panic)
                }
                _ => return Err(e),
            }
        }
        state.recorded = true;
        match result {
            Ok(Ok(value)) => {
                state.poisoned = false;
                Ok(value)
            }
            Ok(Err(e)) => {
                state.drain_or_fatal("scalar failed phase queued work");
                Err(e)
            }
            Err(panic) => {
                state.drain_or_fatal("scalar unwound phase queued work");
                drop(state);
                std::panic::resume_unwind(panic)
            }
        }
    }
    /// Final CPU-visible publication barrier. It joins the last actual phase
    /// on the admitted copy stream without draining other streams on success.
    /// Work retains its real host/device copy owners and completes that copy.
    /// # Safety
    /// No sampling/logit/HTTP publication may precede a successful result. Work
    /// uses completion-before-release guards on errors and unwinds.
    pub(crate) unsafe fn publication<T>(
        &self,
        stream: &Stream,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar publication phase held/poisoned")?;
        if state.poisoned {
            return Err(
                "scalar publication rejects poisoned sequence; verified reset/restore required"
                    .into(),
            );
        }
        if !state.streams.iter().any(|s| s.same(stream)) {
            return Err("scalar publication stream not cold-admitted".into());
        }
        stream
            .device()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        state.poisoned = true;
        if state.recorded {
            state.event.wait(stream)?;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        match result {
            Ok(Ok(value)) => {
                state.poisoned = false;
                Ok(value)
            }
            Ok(Err(e)) => {
                state.drain_or_fatal("scalar rejected publication completion");
                Err(e)
            }
            Err(panic) => {
                state.drain_or_fatal("scalar unwound publication completion");
                drop(state);
                std::panic::resume_unwind(panic)
            }
        }
    }
    /// Caller holds real state/transfer owners in work. It must complete its
    /// actual copy stream with the existing copy-completion guard. Success does
    /// not implicitly recover a poisoned sequence; recovered explicitly below.
    /// # Safety
    /// Same owner/unwind obligations as execute; work additionally completes
    /// every copy stream before releasing transfer-local resources.
    pub(crate) unsafe fn quiescent<T>(
        &self,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar quiescent phase held/poisoned")?;
        if state.poisoned {
            return Err(
                "scalar quiescent snapshot rejects poisoned sequence; explicit recovery required"
                    .into(),
            );
        }
        let was_poisoned = false;
        state.poisoned = true;
        state.drain_or_fatal("scalar quiescent entry completion");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        state.drain_or_fatal("scalar quiescent transfer completion");
        state.recorded = false;
        match result {
            Ok(Ok(v)) => {
                state.poisoned = was_poisoned;
                Ok(v)
            }
            Ok(Err(e)) => Err(e),
            Err(panic) => {
                drop(state);
                std::panic::resume_unwind(panic)
            }
        }
    }
    /// # Safety
    /// On success work must have reset/restored EVERY registered state region,
    /// host cursor and owner poison flag, with validated inventory/identity.
    /// Work retains all copy owners and completes its actual transfer streams
    /// even on error/unwind. A partial restore may never return success here.
    /// The recovery and poison clear are atomic under the same phase lock.
    pub(crate) unsafe fn recover<T>(
        &self,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar recovery phase held/poisoned")?;
        state.poisoned = true;
        state.drain_or_fatal("scalar recovery entry completion");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        state.drain_or_fatal("scalar verified recovery completion");
        state.recorded = false;
        match result {
            Ok(Ok(v)) => {
                state.poisoned = false;
                Ok(v)
            }
            Ok(Err(e)) => Err(e),
            Err(panic) => {
                drop(state);
                std::panic::resume_unwind(panic)
            }
        }
    }
}
impl Drop for Execution {
    fn drop(&mut self) {
        match self.inner.get_mut() {
            Ok(state) => state.drain_or_fatal("scalar execution domain teardown"),
            Err(_) => fatal_cuda_context_error("scalar execution teardown", "phase mutex poisoned"),
        }
    }
}
impl Execution {
    pub(crate) fn complete_before_release(&self) {
        match self.inner.try_lock() {
            Ok(mut state) => {
                state.drain_or_fatal("scalar model owner teardown");
                state.recorded = false;
            }
            Err(_) => fatal_cuda_context_error(
                "scalar model owner teardown",
                "execution is held or poisoned",
            ),
        }
    }
}
