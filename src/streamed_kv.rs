//! Experimental full-context MLA backing with stable virtual addresses.
//!
//! A fixed-budget VMM backend maps cold pages to host NUMA memory and hot
//! pages to an explicitly allocated GPU pool. Managed placement remains a
//! separate diagnostic backend; its physical accounting is driver controlled.
//! CPU-preferred, GPU-accessed managed pages remain accessible to existing
//! kernels at their absolute token offsets. Only a bounded recent range is
//! explicitly prefetched to the GPU. This changes placement, never selection.
//! CUDA may service old-page reads over PCIe; prefetch bytes are NOT a traffic
//! counter. This candidate requires concurrent managed access and remains off
//! until lifecycle, numerical, reserve and request-level qualification passes.
use cudarc::driver::{sys, CudaDevice};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

mod dlpack;
mod vmm;
const PAGE: usize = 65536;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCS: OnceLock<Mutex<Vec<Weak<Backing>>>> = OnceLock::new();
fn checked(rc: sys::CUresult, op: &str) -> Result<(), String> {
    if rc == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("streamed KV {op}: {rc:?}"))
    }
}
#[derive(Default)]
struct Placement {
    ranges: Vec<(usize, usize)>,
    gpu_prefetch_bytes: u64,
    host_prefetch_bytes: u64,
}
struct Backing {
    ptr: u64,
    bytes: usize,
    layers: usize,
    rows: usize,
    stride: usize,
    window: usize,
    device: Arc<CudaDevice>,
    ordinal: i32,
    placement: Mutex<Placement>,
    vmm: Option<vmm::Vmm>,
}
unsafe impl Send for Backing {}
unsafe impl Sync for Backing {}
impl Drop for Backing {
    fn drop(&mut self) {
        // The tensor owner must outlive all graphs borrowing it, just like the
        // ordinary KV tensor. Synchronize queued users before releasing pages.
        if self.vmm.is_some() {
            return;
        } // VMM owns mappings/physical handles.
        if self.device.bind_to_thread().is_ok() && self.device.synchronize().is_ok() {
            let rc = unsafe { sys::lib().cuMemFree_v2(self.ptr) };
            if rc != sys::CUresult::CUDA_SUCCESS {
                log::error!("streamed KV free: {rc:?}");
            }
        } else {
            log::error!("streamed KV context unavailable: retaining allocation rather than freeing live memory");
        }
    }
}
fn range(rows: usize, stride: usize, window: usize, end: usize) -> Result<(usize, usize), String> {
    range_with_page(rows, stride, window, end, PAGE)
}
fn range_with_page(
    rows: usize,
    stride: usize,
    window: usize,
    end: usize,
    page: usize,
) -> Result<(usize, usize), String> {
    if page == 0 {
        return Err("streamed KV zero migration granularity".into());
    }
    if end > rows || stride == 0 || window == 0 {
        return Err("streamed KV invalid row/window bounds".into());
    }
    let bytes = rows
        .checked_mul(stride)
        .ok_or("streamed KV row extent overflow")?;
    // Keep the full working-set commitment even for short requests. Otherwise
    // HCS could consume the temporarily unused window and strand a later long
    // request. One extra migration page covers non-aligned token boundaries.
    let resident = window
        .checked_mul(stride)
        .and_then(|x| x.checked_add(page - 1))
        .map(|x| x / page)
        .and_then(|x| x.checked_add(1))
        .and_then(|x| x.checked_mul(page))
        .ok_or("streamed KV window overflow")?
        .min(bytes);
    let wanted = end
        .saturating_sub(window)
        .checked_mul(stride)
        .ok_or("streamed KV start overflow")?
        / page
        * page;
    let start = wanted.min(bytes - resident);
    Ok((start, start + resident))
}
impl Backing {
    fn move_range(&self, layer: usize, end: usize, stream: sys::CUstream) -> Result<(), String> {
        if layer >= self.layers {
            return Err("streamed KV layer out of bounds".into());
        }
        let next = range_with_page(
            self.rows,
            self.stride,
            self.window,
            end,
            self.vmm.as_ref().map_or(PAGE, |v| v.page),
        )?;
        let mut state = self
            .placement
            .lock()
            .map_err(|_| "streamed KV placement lock poisoned")?;
        if let Some(vmm) = &self.vmm {
            let (gpu, host) = vmm.move_layer(layer, end, stream)?;
            state.gpu_prefetch_bytes += gpu;
            state.host_prefetch_bytes += host;
            state.ranges[layer] = next;
            return Ok(());
        }
        let old = state.ranges[layer];
        if old == next {
            return Ok(());
        }
        let base = self.ptr + (layer * self.rows * self.stride) as u64;
        let mut free = 0usize;
        let mut total = 0usize;
        checked(
            unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) },
            "free-memory check",
        )?;
        let old_bytes = old.1 - old.0;
        let new_bytes = next.1 - next.0;
        if free.saturating_add(old_bytes) < new_bytes.saturating_add(600 * 1024 * 1024) {
            return Err("streamed KV placement cannot preserve 600MiB reserve".into());
        }
        // Do not evict a page still belonging to the overlapping recent range.
        let old_parts = [(old.0, old.1.min(next.0)), (old.0.max(next.1), old.1)];
        for (lo, hi) in old_parts {
            if lo < hi {
                checked(
                    unsafe { sys::lib().cuMemPrefetchAsync(base + lo as u64, hi - lo, -1, stream) },
                    "evict host",
                )?;
                state.host_prefetch_bytes += (hi - lo) as u64;
            }
        }
        let new_parts = [(next.0, next.1.min(old.0)), (next.0.max(old.1), next.1)];
        for (lo, hi) in new_parts {
            if lo < hi {
                checked(
                    unsafe {
                        sys::lib().cuMemPrefetchAsync(
                            base + lo as u64,
                            hi - lo,
                            self.ordinal,
                            stream,
                        )
                    },
                    "prefetch device",
                )?;
                state.gpu_prefetch_bytes += (hi - lo) as u64;
            }
        }
        state.ranges[layer] = next;
        Ok(())
    }
}
#[pyclass]
pub struct StreamedKvBacking {
    inner: Arc<Backing>,
    shape: Vec<usize>,
}
#[pymethods]
impl StreamedKvBacking {
    #[new]
    #[pyo3(signature=(shape,device,window_tokens,backend="vmm"))]
    fn new(
        shape: Vec<usize>,
        device: usize,
        window_tokens: usize,
        backend: &str,
    ) -> PyResult<Self> {
        let make = || -> Result<Self, String> {
            if shape.len() != 4 || shape.iter().any(|&x| x == 0) || window_tokens == 0 {
                return Err("streamed KV requires nonempty [layers,pages,page_size,row_bytes] and positive window".into());
            }
            let bytes = shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .ok_or("streamed KV allocation overflow")?;
            let rows = shape[1]
                .checked_mul(shape[2])
                .ok_or("streamed KV capacity overflow")?;
            if window_tokens >= rows {
                return Err("streamed KV window must be smaller than context capacity".into());
            }
            let dev = CudaDevice::new(device).map_err(|e| format!("{e:?}"))?;
            dev.bind_to_thread().map_err(|e| format!("{e:?}"))?;
            let ordinal = i32::try_from(device).map_err(|_| "device ordinal overflow")?;
            let mut architecture = 0;
            checked(
                unsafe {
                    sys::lib().cuDeviceGetAttribute(
                        &mut architecture,
                        sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        ordinal,
                    )
                },
                "query architecture",
            )?;
            if architecture < 8 {
                return Err("host-backed KV candidate requires Ampere or newer".into());
            }
            if backend == "vmm" {
                let storage = vmm::Vmm::new(dev.clone(), shape[0], rows, shape[3], window_tokens)?;
                let inner = Arc::new(Backing {
                    ptr: storage.ptr,
                    bytes,
                    layers: shape[0],
                    rows,
                    stride: shape[3],
                    window: window_tokens,
                    device: dev,
                    ordinal,
                    placement: Mutex::new(Placement::default()),
                    vmm: Some(storage),
                });
                // Fixed device commitment is allocated before HCS admission.
                inner
                    .placement
                    .lock()
                    .map_err(|_| "streamed KV lock poisoned")?
                    .ranges = vec![(0, 0); shape[0]];
                ALLOCS
                    .get_or_init(Default::default)
                    .lock()
                    .map_err(|_| "streamed KV registry poisoned")?
                    .push(Arc::downgrade(&inner));
                ACTIVE.store(true, Ordering::Release);
                return Ok(Self { inner, shape });
            }
            if backend != "managed" {
                return Err("streamed KV backend must be vmm or managed".into());
            }
            let mut supported = 0;
            checked(
                unsafe {
                    sys::lib().cuDeviceGetAttribute(
                        &mut supported,
                        sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                        ordinal,
                    )
                },
                "query concurrent managed access",
            )?;
            if supported != 1 {
                return Err("streamed KV requires concurrent managed access; use resident KV on this platform".into());
            }
            // Per-layer ranges must not share migration pages.
            if rows
                .checked_mul(shape[3])
                .ok_or("streamed KV layer overflow")?
                % PAGE
                != 0
            {
                return Err("streamed KV layer extent must be 64KiB aligned".into());
            }
            let initial = range(rows, shape[3], window_tokens, window_tokens)?;
            let commitment = (initial.1 - initial.0)
                .checked_mul(shape[0])
                .ok_or("streamed KV GPU commitment overflow")?;
            let mut free = 0usize;
            let mut total = 0usize;
            checked(
                unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) },
                "initial budget",
            )?;
            if free < commitment.saturating_add(600 * 1024 * 1024) {
                return Err("streamed KV initial GPU window cannot preserve 600MiB reserve".into());
            }
            let mut major = 0;
            checked(
                unsafe {
                    sys::lib().cuDeviceGetAttribute(
                        &mut major,
                        sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        ordinal,
                    )
                },
                "query architecture",
            )?;
            if major < 8 {
                return Err("streamed KV candidate requires Ampere or newer".into());
            }
            let mut ptr = 0;
            checked(
                unsafe { sys::lib().cuMemAllocManaged(&mut ptr, bytes, 1) },
                "allocate",
            )?;
            let inner = Arc::new(Backing {
                ptr,
                bytes,
                layers: shape[0],
                rows,
                stride: shape[3],
                window: window_tokens,
                device: dev,
                ordinal,
                placement: Mutex::new(Placement {
                    ranges: vec![(0, 0); shape[0]],
                    ..Default::default()
                }),
                vmm: None,
            });
            checked(
                unsafe {
                    sys::lib().cuMemAdvise(
                        ptr,
                        bytes,
                        sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                        -1,
                    )
                },
                "prefer host",
            )?;
            checked(
                unsafe {
                    sys::lib().cuMemAdvise(
                        ptr,
                        bytes,
                        sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_ACCESSED_BY,
                        ordinal,
                    )
                },
                "map device",
            )?;
            // Initialization is startup only, before registration or graph use.
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
            for layer in 0..shape[0] {
                inner.move_range(layer, window_tokens, *inner.device.cu_stream())?;
            }
            inner.device.synchronize().map_err(|e| format!("{e:?}"))?;
            ALLOCS
                .get_or_init(Default::default)
                .lock()
                .map_err(|_| "streamed KV registry poisoned")?
                .push(Arc::downgrade(&inner));
            ACTIVE.store(true, Ordering::Release);
            log::warn!("EXPERIMENT full-context managed MLA backing: bytes={bytes} layers={} capacity={rows} window={window_tokens}; remote old-key reads remain enabled",shape[0]);
            Ok(Self { inner, shape })
        };
        make().map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.inner.ordinal)
    }
    #[pyo3(signature=(stream=None))]
    fn __dlpack__(&self, py: Python<'_>, stream: Option<usize>) -> PyResult<PyObject> {
        let _ = stream;
        dlpack::export(py, self.inner.clone(), &self.shape)
    }
    #[getter]
    fn gpu_commitment_bytes(&self) -> usize {
        self.inner.vmm.as_ref().map_or(0, |v| v.gpu_bytes)
    }
    #[getter]
    fn pointer(&self) -> u64 {
        self.inner.ptr
    }
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("shape", PyTuple::new_bound(py, self.shape.iter().copied()))?;
        d.set_item("strides", py.None())?;
        d.set_item("typestr", "|u1")?;
        d.set_item("data", (self.inner.ptr, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }
    fn diagnostic_position(&self, end: usize) -> PyResult<()> {
        if std::env::var("KRASIS_STREAMED_KV_DIAGNOSTIC").as_deref() != Ok("1") {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("diagnostic_position requires KRASIS_STREAMED_KV_DIAGNOSTIC=1; never use for inference"));
        }
        let run = || -> Result<(), String> {
            self.inner
                .device
                .bind_to_thread()
                .map_err(|e| e.to_string())?;
            self.inner.device.synchronize().map_err(|e| e.to_string())?;
            for layer in 0..self.inner.layers {
                self.inner
                    .move_range(layer, end, *self.inner.device.cu_stream())?;
            }
            self.inner.device.synchronize().map_err(|e| e.to_string())
        };
        run().map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    fn diagnostic_prepare_owned(&self,end:usize)->PyResult<()> {
        if std::env::var("KRASIS_STREAMED_KV_DIAGNOSTIC").as_deref()!=Ok("1") {return Err(pyo3::exceptions::PyRuntimeError::new_err("diagnostic_prepare_owned requires explicit diagnostic mode"));}
        self.inner.device.bind_to_thread().map_err(|e|pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        prepare_matching(end,*self.inner.device.cu_stream(),*self.inner.device.cu_primary_ctx(),|base,_|base==self.inner.ptr).map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    fn stats(&self) -> PyResult<(usize, u64, u64)> {
        let s =
            self.inner.placement.lock().map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err("placement lock poisoned")
            })?;
        Ok((
            self.inner.bytes,
            s.gpu_prefetch_bytes,
            s.host_prefetch_bytes,
        ))
    }
}
fn live() -> Result<Vec<Arc<Backing>>, String> {
    let mut all = ALLOCS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "streamed KV registry poisoned")?;
    all.retain(|p| p.strong_count() > 0);
    ACTIVE.store(!all.is_empty(),Ordering::Release);
    Ok(all.iter().filter_map(Weak::upgrade).collect())
}
/// Read counters after a diagnostic request; no synchronization or GPU calls.
pub(crate) fn diagnostic_snapshot() -> Result<Vec<serde_json::Value>,String> {
    let mut result=Vec::new();
    for a in live()? {
        let state=a.placement.lock().map_err(|_|"streamed KV placement lock poisoned")?;
        result.push(serde_json::json!({"device":a.ordinal,"base_pointer":a.ptr,"logical_bytes":a.bytes,"layers":a.layers,
            "capacity_rows":a.rows,"window_rows":a.window,
            "backend":if a.vmm.is_some(){"vmm"}else{"managed"},
            "fixed_gpu_bytes":a.vmm.as_ref().map(|v|v.gpu_bytes),
            "migration_h2d_bytes":state.gpu_prefetch_bytes,"migration_d2h_bytes":state.host_prefetch_bytes,
            "traffic_scope":"explicit migrations only; sparse reads of host-backed old keys are additional PCIe traffic"}));
    }
    Ok(result)
}
/// Call only outside capture, before graph replay, on the graph's actual stream.
pub(crate) fn prepare_matching<F:Fn(u64,usize)->bool>(
    end: usize,
    stream: sys::CUstream,
    context: sys::CUcontext,
    owns:F,
) -> Result<(), String> {
    if !ACTIVE.load(Ordering::Acquire) {
        return Ok(());
    }
    for a in live()? {
        if *a.device.cu_primary_ctx() == context && owns(a.ptr,a.bytes) {
            for layer in 0..a.layers {
                a.move_range(layer, end, stream)?;
            }
        }
    }
    Ok(())
}
/// Prefill prepares only the MLA layer about to run, before cache append.
pub(crate) fn prepare_layer(ptr: u64, end: usize, stream: sys::CUstream) -> Result<(), String> {
    if !ACTIVE.load(Ordering::Acquire) {
        return Ok(());
    }
    for a in live()? {
        if ptr >= a.ptr && ptr < a.ptr + a.bytes as u64 {
            let layer_bytes = a.rows * a.stride;
            let offset = (ptr - a.ptr) as usize;
            if offset % layer_bytes != 0 {
                return Err("streamed KV MLA pointer not layer aligned".into());
            }
            a.move_range(offset / layer_bytes, end, stream)?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placement_covers_recent_rows_and_retains_absolute_extent() {
        for end in [0, 1, 15, 16, 32767, 32768, 32769, 65536, 131072] {
            let (lo, hi) = range(131072, 448, 32768, end).unwrap();
            assert!(lo <= end.saturating_sub(32768) * 448 && hi >= end * 448);
            assert!(hi - lo <= 32768 * 448 + 2 * PAGE);
        }
        assert!(range(131072, 448, 32768, 131073).is_err());
        assert!(range(131072, 448, 0, 1).is_err());
    }
}

/// Batched/speculative writes need their own placement/transaction qualification.
pub(crate) fn require_scalar_for<F:Fn(u64,usize)->bool>(context:sys::CUcontext,owns:F) -> Result<(), String> {
    if ACTIVE.load(Ordering::Acquire) && live()?.iter().any(|a|*a.device.cu_primary_ctx()==context && owns(a.ptr,a.bytes)) {
        return Err("host-backed MLA candidate supports scalar decode only; batched/speculative execution is not admitted".into());
    }
    Ok(())
}

/// Range admission is based on the retained native VMM owner, not a caller's
/// pointer descriptor. cuPointerGetAttribute reports only one mapped page.
pub(crate) fn owns_vmm_range(ptr: u64, bytes: usize, ordinal: usize) -> Result<bool, String> {
    if !ACTIVE.load(Ordering::Acquire) {
        return Ok(false);
    }
    let end = ptr
        .checked_add(u64::try_from(bytes).map_err(|_| "VMM extent exceeds u64")?)
        .ok_or("VMM extent overflow")?;
    Ok(live()?.iter().any(|a| {
        a.vmm.is_some()
            && a.device.ordinal() == ordinal
            && ptr >= a.ptr
            && end <= a.ptr + a.bytes as u64
    }))
}
