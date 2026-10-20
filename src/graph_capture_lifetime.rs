// Real graph-capture resources. These guards are cold setup/error boundaries;
// no guard or host synchronization is added to ordinary graph replay.
struct CaptureOrigin {
    device: Arc<CudaDevice>,
    stream: cuda_sys::CUstream,
}
impl CaptureOrigin {
    fn validate(device: Arc<CudaDevice>) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let stream = *device.cu_stream();
        let mut status = cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe { cuda_sys::lib().cuStreamIsCapturing(stream, &mut status) }
            .result()
            .map_err(|e| format!("graph capture entry status:{e}"))?;
        if status != cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err("graph capture cannot take ownership of an already captured stream".into());
        }
        Ok(Self { device, stream })
    }
    /// # Safety
    /// Caller retains its actual cuBLAS owner and exclusive Store invocation
    /// through this guard. Declare the guard AFTER the taken graph, so error
    /// cleanup completes before graph buffers can be released during unwind.
    unsafe fn guard(self, blas: cublas_sys::cublasHandle_t) -> Result<CaptureFrame, String> {
        let mut original_blas = std::ptr::null_mut();
        let status = cublas_sys::lib().cublasGetStream_v2(blas, &mut original_blas);
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("graph capture original cuBLAS stream:{status:?}"));
        }
        Ok(CaptureFrame {
            device: self.device,
            original: self.stream,
            stream: self.stream,
            temporary: false,
            blas,
            original_blas,
            armed: true,
        })
    }
}
struct CaptureFrame {
    device: Arc<CudaDevice>,
    original: cuda_sys::CUstream,
    stream: cuda_sys::CUstream,
    temporary: bool,
    // Borrowed from the actual Store, which is retained by the caller.
    blas: cublas_sys::cublasHandle_t,
    original_blas: cublas_sys::cudaStream_t,
    armed: bool,
}
impl CaptureFrame {
    fn prepare_stream(&mut self, temporary: bool) -> Result<cuda_sys::CUstream, String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        if temporary {
            let mut stream = std::ptr::null_mut();
            unsafe {
                cuda_sys::lib().cuStreamCreate(
                    &mut stream,
                    cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            }
            .result()
            .map_err(|e| format!("graph capture stream creation:{e}"))?;
            if stream.is_null() {
                return Err("graph capture stream creation returned null".into());
            }
            self.stream = stream;
            self.temporary = true;
        }
        // Preserve the existing pre-capture completion boundary. Its real
        // temporary stream is already owned if this operation fails.
        self.device
            .synchronize()
            .map_err(|e| format!("sync before capture:{e}"))?;
        unsafe {
            if self.temporary {
                self.set_device_stream(self.stream);
            }
            cublas_result::set_stream(self.blas, self.stream as cublas_sys::cudaStream_t)
                .map_err(|e| format!("cuBLAS capture stream:{e}"))?;
        }
        Ok(self.stream)
    }
    /// Same serialized, actual-device stream replacement used by capture.
    /// Restore it on EVERY exit, including errors before normal cleanup.
    unsafe fn set_device_stream(&self, stream: cuda_sys::CUstream) {
        self.device.set_stream_for_capture(stream);
    }
    fn restore_streams(&mut self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        unsafe {
            // Existing owned-stream captures never mutate CudaDevice's cell.
            // Only the temporary-stream caller must satisfy the unsafe setter.
            if self.temporary {
                self.set_device_stream(self.original);
            }
            cublas_result::set_stream(self.blas, self.original_blas)
                .map_err(|e| format!("cuBLAS original stream restore:{e}"))?;
            if self.temporary {
                cuda_sys::lib()
                    .cuStreamSynchronize(self.stream)
                    .result()
                    .map_err(|e| format!("graph temporary stream completion:{e}"))?;
                let result = cuda_sys::lib().cuStreamDestroy_v2(self.stream);
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    crate::vram_monitor::fatal_cuda_context_error(
                        "graph temporary stream release",
                        &format!("{result:?}"),
                    );
                }
                self.temporary = false;
            }
        }
        self.stream = self.original;
        Ok(())
    }
    fn disarm(&mut self) {
        self.armed = false;
    }
    fn abort_recording(&mut self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut status = cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe {
            cuda_sys::lib()
                .cuStreamIsCapturing(self.stream, &mut status)
                .result()
                .map_err(|e| format!("failed graph capture status:{e}"))?;
            if status != cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                let mut graph = std::ptr::null_mut();
                let end = cuda_sys::lib().cuStreamEndCapture(self.stream, &mut graph);
                cuda_sys::lib()
                    .cuStreamIsCapturing(self.stream, &mut status)
                    .result()
                    .map_err(|e| format!("failed graph capture cleanup status:{e}"))?;
                if status != cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    return Err(
                        "failed graph capture remains active; owners cannot be released".into(),
                    );
                }
                if !graph.is_null() {
                    let release = cuda_sys::lib().cuGraphDestroy(graph);
                    if release != cuda_sys::CUresult::CUDA_SUCCESS {
                        crate::vram_monitor::fatal_cuda_context_error(
                            "failed capture graph release",
                            &format!("{release:?}"),
                        );
                    }
                }
                if end != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!(
                        "failed capture ended with {end:?}; actual stream is no longer capturing"
                    );
                }
            }
        }
        Ok(())
    }
    fn error_completion(&mut self) -> Result<(), String> {
        self.abort_recording()?;
        unsafe {
            // Error/unwind only: complete ALL real context streams before the
            // local graph and its allocation/program owners can be dropped.
            cuda_sys::lib()
                .cuCtxSynchronize()
                .result()
                .map_err(|e| format!("failed capture context completion:{e}"))?;
        }
        self.restore_streams()
    }
}
impl Drop for CaptureFrame {
    fn drop(&mut self) {
        if self.armed {
            if let Err(error) = self.error_completion() {
                crate::vram_monitor::fatal_cuda_context_error(
                    "graph capture owner completion",
                    &error,
                );
            }
        }
    }
}
struct OwnedCaptureGraph {
    pointer: cuda_sys::CUgraph,
    device: Arc<CudaDevice>,
}
impl Drop for OwnedCaptureGraph {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            if let Err(error) = self.device.bind_to_thread() {
                crate::vram_monitor::fatal_cuda_context_error(
                    "captured graph release context",
                    &error.to_string(),
                );
            }
            let result = unsafe { cuda_sys::lib().cuGraphDestroy(self.pointer) };
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                crate::vram_monitor::fatal_cuda_context_error(
                    "captured graph release",
                    &format!("{result:?}"),
                );
            }
        }
    }
}
struct OwnedCaptureExec {
    pointer: cuda_sys::CUgraphExec,
    device: Arc<CudaDevice>,
}
impl OwnedCaptureExec {
    fn into_store(mut self) -> CudaGraphExecPtr {
        let pointer = self.pointer;
        self.pointer = std::ptr::null_mut();
        CudaGraphExecPtr(pointer, self.device.clone())
    }
}
impl Drop for OwnedCaptureExec {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            if let Err(error) = self.device.bind_to_thread() {
                crate::vram_monitor::fatal_cuda_context_error(
                    "captured executable release context",
                    &error.to_string(),
                );
            }
            let result = unsafe { cuda_sys::lib().cuGraphExecDestroy(self.pointer) };
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                crate::vram_monitor::fatal_cuda_context_error(
                    "captured executable release",
                    &format!("{result:?}"),
                );
            }
        }
    }
}
