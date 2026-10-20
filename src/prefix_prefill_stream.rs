// Actual prefill compute stream ownership. Raw dispatch uses the same handle.
pub(crate) struct PrefillComputeStream {
    handle: cuda_sys::CUstream,
    device: Arc<CudaDevice>,
}
// SAFETY: CUDA stream is an opaque immutable driver handle, with retained device.
// Every operation binds that device on the receiving thread. Last-Arc teardown
// cannot race a live owner lease; model/family enqueue ordering remains external.
unsafe impl Send for PrefillComputeStream {}
unsafe impl Sync for PrefillComputeStream {}
impl PrefillComputeStream {
    pub(crate) fn create(device: Arc<CudaDevice>) -> Result<Arc<Self>, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut handle = std::ptr::null_mut();
        let result = unsafe {
            cuda_sys::lib().cuStreamCreate(
                &mut handle,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        };
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!("Create prefill stream: {result:?}"));
        }
        Ok(Arc::new(Self { handle, device }))
    }
    pub(crate) fn handle(&self) -> cuda_sys::CUstream {
        self.handle
    }
    pub(crate) fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }
    pub(crate) fn drain(&self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let result = unsafe { cuda_sys::lib().cuStreamSynchronize(self.handle) };
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!("prefill owner final stream drain: {result:?}"));
        }
        Ok(())
    }
}
impl Drop for PrefillComputeStream {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            crate::vram_monitor::fatal_cuda_context_error(
                "retained prefill compute stream completion",
                &error,
            );
        }
        let result = unsafe { cuda_sys::lib().cuStreamDestroy_v2(self.handle) };
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            crate::vram_monitor::fatal_cuda_context_error(
                "retained prefill compute stream release",
                &format!("{result:?}"),
            );
        }
    }
}
