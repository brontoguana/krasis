//! Cold setup transfers. Allocation, copy and completion use the same device
//! stream; external producers hand off through an explicit event. No hot-loop
//! orchestration, default-stream assumption, new stream, or allocator fallback.
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

fn check(result: sys::CUresult, operation: &str) -> Result<(), String> {
    result.result().map_err(|e| format!("{operation}: {e}"))
}

struct Event {
    handle: sys::CUevent,
    device: Arc<CudaDevice>,
}
impl Drop for Event {
    fn drop(&mut self) {
        if self.device.bind_to_thread().is_err()
            || unsafe { sys::lib().cuEventDestroy_v2(self.handle) } != sys::CUresult::CUDA_SUCCESS
        {
            eprintln!("FATAL: setup transfer event release failed");
            std::process::abort();
        }
    }
}

/// Declare AFTER all borrowed source/destination owners. Completes queued work
/// before those owners can disappear on errors or Rust unwind.
pub(crate) struct Completion {
    event: Option<Event>,
    device: Arc<CudaDevice>,
    pub(crate) stream: sys::CUstream,
    armed: bool,
}
impl Completion {
    pub(crate) fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let stream = *device.cu_stream();
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe {
            check(
                sys::lib().cuStreamIsCapturing(stream, &mut status),
                "setup capture status",
            )?;
        }
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err("setup transfer cannot synchronize a captured stream".into());
        }
        Ok(Self {
            event: None,
            device,
            stream,
            armed: true,
        })
    }
    fn wait_for(&mut self, producer: sys::CUstream) -> Result<(), String> {
        let mut current = std::ptr::null_mut();
        let mut source = std::ptr::null_mut();
        unsafe {
            check(
                sys::lib().cuCtxGetCurrent(&mut current),
                "setup current context",
            )?;
            check(
                sys::lib().cuStreamGetCtx(producer, &mut source),
                "setup producer context",
            )?;
        }
        if current != source {
            return Err("setup producer stream belongs to another CUDA context".into());
        }
        if producer == self.stream {
            return Ok(());
        }
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                sys::lib().cuEventCreate(
                    &mut handle,
                    sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                ),
                "setup handoff event",
            )?;
        }
        self.event = Some(Event {
            handle,
            device: self.device.clone(),
        });
        unsafe {
            check(
                sys::lib().cuEventRecord(handle, producer),
                "setup producer completion record",
            )?;
            check(
                sys::lib().cuStreamWaitEvent(self.stream, handle, 0),
                "setup consumer dependency",
            )?;
        }
        Ok(())
    }
    pub(crate) fn finish(mut self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        unsafe {
            check(
                sys::lib().cuStreamSynchronize(self.stream),
                "setup transfer completion",
            )?;
        }
        self.armed = false;
        Ok(())
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if self.armed
            && (self.device.bind_to_thread().is_err()
                || unsafe { sys::lib().cuStreamSynchronize(self.stream) }
                    != sys::CUresult::CUDA_SUCCESS)
        {
            eprintln!("FATAL: pending setup transfer could not complete before owner release");
            std::process::abort();
        }
    }
}

/// Borrowed host inputs remain alive until all their copies complete. Padding
/// keeps its initialized zeros; every nonempty segment has checked real bounds.
pub(crate) fn upload_segments(
    device: Arc<CudaDevice>,
    bytes: usize,
    segments: &[(usize, &[u8])],
) -> Result<CudaSlice<u8>, String> {
    if bytes == 0 {
        return Err("setup upload allocation is empty".into());
    }
    for (offset, data) in segments {
        if offset.checked_add(data.len()).is_none_or(|end| end > bytes) {
            return Err("setup upload segment exceeds allocation".into());
        }
    }
    let buffer = device.alloc_zeros::<u8>(bytes).map_err(|e| e.to_string())?;
    let completion = Completion::new(device)?;
    for (offset, data) in segments {
        if data.is_empty() {
            continue;
        } // Empty scale planes carry no data.
        let pointer = (*buffer.device_ptr())
            .checked_add(*offset as u64)
            .ok_or("setup upload address overflow")?;
        unsafe {
            check(
                sys::lib().cuMemcpyHtoDAsync_v2(
                    pointer,
                    data.as_ptr().cast(),
                    data.len(),
                    completion.stream,
                ),
                "setup segment upload",
            )?;
        }
    }
    completion.finish()?;
    Ok(buffer)
}

pub(crate) fn sequence_lengths(
    device: Arc<CudaDevice>,
    layers: usize,
) -> Result<(CudaSlice<i32>, Vec<CudaSlice<i32>>), String> {
    let global = device.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
    let mut per_layer = Vec::new();
    per_layer
        .try_reserve_exact(layers)
        .map_err(|e| e.to_string())?;
    for _ in 0..layers {
        per_layer.push(device.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?);
    }
    let initial = 1i32;
    let completion = Completion::new(device)?;
    for buffer in std::iter::once(&global).chain(per_layer.iter()) {
        unsafe {
            check(
                sys::lib().cuMemcpyHtoDAsync_v2(
                    *buffer.device_ptr(),
                    (&initial as *const i32).cast(),
                    std::mem::size_of::<i32>(),
                    completion.stream,
                ),
                "setup sequence length upload",
            )?;
        }
    }
    completion.finish()?;
    Ok((global, per_layer))
}

/// # Safety
/// Source device allocations belong to this context and remain owned by the
/// caller through this function, including its error/unwind completion.
pub(crate) unsafe fn rope_tables(
    device: Arc<CudaDevice>,
    cos: u64,
    sin: u64,
    count: usize,
    producer: sys::CUstream,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>), String> {
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .filter(|&n| n > 0)
        .ok_or("setup RoPE extent empty/overflow")?;
    for pointer in [cos, sin] {
        if pointer == 0 || pointer.checked_add(bytes as u64).is_none() {
            return Err("setup RoPE source null/overflow".into());
        }
        let mut ordinal = -1i32;
        check(
            sys::lib().cuPointerGetAttribute(
                (&mut ordinal as *mut i32).cast(),
                sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                pointer,
            ),
            "setup RoPE source device",
        )?;
        if ordinal != device.ordinal() as i32 {
            return Err("setup RoPE source belongs to another device".into());
        }
    }
    let d_cos = device
        .alloc_zeros::<f32>(count)
        .map_err(|e| e.to_string())?;
    let d_sin = device
        .alloc_zeros::<f32>(count)
        .map_err(|e| e.to_string())?;
    let mut completion = Completion::new(device)?;
    completion.wait_for(producer)?;
    check(
        sys::lib().cuMemcpyDtoDAsync_v2(*d_cos.device_ptr(), cos, bytes, completion.stream),
        "setup RoPE cos copy",
    )?;
    check(
        sys::lib().cuMemcpyDtoDAsync_v2(*d_sin.device_ptr(), sin, bytes, completion.stream),
        "setup RoPE sin copy",
    )?;
    completion.finish()?;
    Ok((d_cos, d_sin))
}
