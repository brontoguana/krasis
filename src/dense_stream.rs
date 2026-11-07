//! Request-serialized, phase-ordered dense BF16 weight streaming.
//!
//! All large weights have immutable pinned host owners. Every layer's pointers
//! address offsets within ONE reusable device slab. Copies and their consumers
//! use the same stream, including graph capture: overwrite follows consumption.
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{Arc, Mutex},
};
pub const EMBEDDING: usize = usize::MAX - 1;
pub const HEAD: usize = usize::MAX;
struct Source {
    ptr: *mut c_void,
    bytes: usize,
    offset: usize,
}
unsafe impl Send for Source {}
unsafe impl Sync for Source {}
impl Drop for Source {
    fn drop(&mut self) {
        unsafe {
            sys::lib().cuMemFreeHost(self.ptr);
        }
    }
}
struct Groups {
    sources: BTreeMap<usize, Vec<Source>>,
    extents: BTreeMap<usize, usize>,
    sealed: bool,
}
pub struct DenseStream {
    // Fields drop in declaration order. Free pinned host sources while the
    // CUDA context is still owned, then the device slab, then the context.
    groups: Mutex<Groups>,
    slab: CudaSlice<u8>,
    device: Arc<CudaDevice>,
}
impl DenseStream {
    pub fn new(device: Arc<CudaDevice>, bytes: usize) -> Result<Arc<Self>, String> {
        if bytes == 0 {
            return Err("dense stream slab must be nonempty".into());
        }
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
        if free < bytes.saturating_add(600 * 1024 * 1024) {
            return Err("dense weight slab would breach the 600 MiB reserve".into());
        }
        let slab = device.alloc_zeros::<u8>(bytes).map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            device,
            slab,
            groups: Mutex::new(Groups {
                sources: BTreeMap::new(),
                extents: BTreeMap::new(),
                sealed: false,
            }),
        }))
    }
    // Source is copied immediately. No caller-owned host pointer escapes.
    pub unsafe fn register(
        &self,
        group: usize,
        source: usize,
        bytes: usize,
    ) -> Result<u64, String> {
        if source == 0 || bytes == 0 {
            return Err("empty streamed weight".into());
        }
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut g = self
            .groups
            .lock()
            .map_err(|_| "dense stream lock poisoned")?;
        if g.sealed {
            return Err("streamed weights cannot change after sealing".into());
        }
        let offset = align(*g.extents.get(&group).unwrap_or(&0))?;
        let end = offset
            .checked_add(bytes)
            .ok_or("streamed extent overflow")?;
        if end > self.slab.len() {
            return Err(format!(
                "streamed group {group} needs {end} bytes, slab {}",
                self.slab.len()
            ));
        }
        let mut ptr = std::ptr::null_mut();
        check(
            sys::lib().cuMemHostAlloc(&mut ptr, bytes, 0),
            "pin dense source",
        )?;
        let owner = Source { ptr, bytes, offset };
        std::ptr::copy_nonoverlapping(source as *const u8, ptr as *mut u8, bytes);
        g.sources.entry(group).or_default().push(owner);
        g.extents.insert(group, end);
        Ok(*self.slab.device_ptr() + offset as u64)
    }
    pub fn seal(&self) -> Result<(), String> {
        self.groups
            .lock()
            .map_err(|_| "dense stream lock poisoned")?
            .sealed = true;
        Ok(())
    }
    pub fn stage(&self, group: usize, stream: sys::CUstream) -> Result<(), String> {
        let g = self
            .groups
            .lock()
            .map_err(|_| "dense stream lock poisoned")?;
        if !g.sealed {
            return Err("dense stream used before registration sealed".into());
        }
        let sources = g
            .sources
            .get(&group)
            .ok_or_else(|| format!("missing dense weight group {group}"))?;
        for s in sources {
            unsafe {
                check(
                    sys::lib().cuMemcpyHtoDAsync_v2(
                        *self.slab.device_ptr() + s.offset as u64,
                        s.ptr,
                        s.bytes,
                        stream,
                    ),
                    "stage dense weight",
                )?;
            }
        }
        Ok(())
    }
    pub fn stats(&self) -> Result<String, String> {
        let g = self
            .groups
            .lock()
            .map_err(|_| "dense stream lock poisoned")?;
        Ok(serde_json::json!({"device_slab_bytes":self.slab.len(),"pinned_host_bytes":g.sources.values().flatten().map(|s|s.bytes).sum::<usize>(),"group_bytes":g.extents,"sealed":g.sealed}).to_string())
    }
}
impl Drop for DenseStream {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        let _ = self.device.synchronize();
    }
}
fn check(status: sys::CUresult, label: &str) -> Result<(), String> {
    if status == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{label}: {status:?}"))
    }
}
pub fn align(n: usize) -> Result<usize, String> {
    n.checked_add(255)
        .map(|v| v & !255)
        .ok_or_else(|| "dense alignment overflow".into())
}
pub fn stage(
    bank: &Option<Arc<DenseStream>>,
    group: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    if let Some(bank) = bank {
        bank.stage(group, stream)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn offsets_align_and_reject_overflow() {
        assert_eq!(super::align(1).unwrap(), 256);
        assert_eq!(super::align(256).unwrap(), 256);
        assert!(super::align(usize::MAX).is_err());
    }
}
