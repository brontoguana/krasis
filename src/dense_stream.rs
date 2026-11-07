//! Request-serialized packed or reference BF16 weight streaming.
//!
//! All large weights have immutable pinned host owners. Every layer's pointers
//! address offsets within ONE reusable device slab. Copies and their consumers
//! use the same stream, including graph capture: overwrite follows consumption.
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{Arc, Mutex, OnceLock},
};
pub const EMBEDDING: usize = usize::MAX - 1;
pub const HEAD: usize = usize::MAX;
struct Source {
    ptr: *mut c_void,
    bytes: usize,
    offset: usize,
    mapping: Option<memmap2::MmapMut>,
    mapped_access: bool,
}
unsafe impl Send for Source {}
unsafe impl Sync for Source {}
impl Drop for Source {
    fn drop(&mut self) {
        unsafe {
            if self.mapping.is_some() {
                sys::lib().cuMemHostUnregister(self.ptr);
            } else {
                sys::lib().cuMemFreeHost(self.ptr);
            }
        }
    }
}
struct Groups {
    sources: BTreeMap<usize, Vec<Source>>,
    extents: BTreeMap<usize, usize>,
    sealed: bool,
    resident_used: usize,
}
struct DenseMath {
    function: crate::gpu_prefill::MarlinMmFn,
    partial: CudaSlice<f32>,
    locks: CudaSlice<u32>,
    sms: usize,
}
pub struct DenseStream {
    // Fields drop in declaration order. Free pinned host sources while the
    // CUDA context is still owned, then the device slab, then the context.
    groups: Mutex<Groups>,
    slab: CudaSlice<u8>,
    resident: CudaSlice<u8>,
    projection: CudaSlice<u16>,
    math: OnceLock<DenseMath>,
    device: Arc<CudaDevice>,
}
impl DenseStream {
    pub fn new(
        device: Arc<CudaDevice>,
        bytes: usize,
        resident_bytes: usize,
    ) -> Result<Arc<Self>, String> {
        if bytes == 0 {
            return Err("dense stream slab must be nonempty".into());
        }
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
        if free
            < bytes
                .saturating_add(resident_bytes.max(1))
                .saturating_add(8 * 1024 * 1024)
                .saturating_add(600 * 1024 * 1024)
        {
            return Err("dense weight slab would breach the 600 MiB reserve".into());
        }
        let slab = device.alloc_zeros::<u8>(bytes).map_err(|e| e.to_string())?;
        let resident = device
            .alloc_zeros::<u8>(resident_bytes.max(1))
            .map_err(|e| e.to_string())?;
        let projection = device
            .alloc_zeros::<u16>(256 * 16384)
            .map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            projection,
            math: OnceLock::new(),
            resident,
            device,
            slab,
            groups: Mutex::new(Groups {
                sources: BTreeMap::new(),
                extents: BTreeMap::new(),
                sealed: false,
                resident_used: 0,
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
        let owner = Source {
            ptr,
            bytes,
            offset,
            mapping: None,
            mapped_access: false,
        };
        std::ptr::copy_nonoverlapping(source as *const u8, ptr as *mut u8, bytes);
        g.sources.entry(group).or_default().push(owner);
        g.extents.insert(group, end);
        Ok(*self.slab.device_ptr() + offset as u64)
    }
    /// Own the only host mapping of a cached weight until all graph readers end.
    /// The private mapping is authoritative; CUDA pins its pages. No tensor or
    /// independent pinned copy is retained. Clean OS file cache is reclaimable.
    pub fn register_file(
        &self,
        group: usize,
        path: &str,
        bytes: usize,
        mapped_access: bool,
    ) -> Result<u64, String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        if bytes == 0 || file.metadata().map_err(|e| e.to_string())?.len() != bytes as u64 {
            return Err("cached weight extent mismatch".into());
        }
        let mapping =
            unsafe { memmap2::MmapOptions::new().map_copy(&file) }.map_err(|e| e.to_string())?;
        let ptr = mapping.as_ptr() as *mut c_void;
        let mut g = self
            .groups
            .lock()
            .map_err(|_| "dense stream lock poisoned")?;
        if g.sealed {
            return Err("streamed weights cannot change after sealing".into());
        }
        let resident_offset = align(g.resident_used)?;
        let keep_resident = !mapped_access
            && resident_offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.resident.len());
        let offset = if keep_resident {
            resident_offset
        } else {
            align(*g.extents.get(&group).unwrap_or(&0))?
        };
        let end = offset
            .checked_add(bytes)
            .ok_or("streamed extent overflow")?;
        if !mapped_access && !keep_resident && end > self.slab.len() {
            return Err("packed group exceeds device slab".into());
        }
        unsafe {
            check(
                sys::lib().cuMemHostRegister_v2(ptr, bytes, 2),
                "register cached weight pages",
            )?;
        }
        let owner = Source {
            ptr,
            bytes,
            offset,
            mapping: Some(mapping),
            mapped_access: mapped_access || keep_resident,
        };
        let gpu = if mapped_access {
            let mut gpu = 0;
            unsafe {
                check(
                    sys::lib().cuMemHostGetDevicePointer_v2(&mut gpu, ptr, 0),
                    "mapped cached weight",
                )?;
            }
            gpu
        } else if keep_resident {
            let gpu = *self.resident.device_ptr() + offset as u64;
            unsafe {
                check(
                    sys::lib().cuMemcpyHtoD_v2(gpu, ptr, bytes),
                    "initialize resident packed weight",
                )?;
            }
            g.resident_used = end;
            gpu
        } else {
            *self.slab.device_ptr() + offset as u64
        };
        g.sources.entry(group).or_default().push(owner);
        if !mapped_access && !keep_resident {
            g.extents.insert(group, end);
        } else {
            g.extents.entry(group).or_insert(0);
        }
        Ok(gpu)
    }
    pub fn enable_tensorcore_decode(&self) -> Result<(), String> {
        use sys::CUdevice_attribute as A;
        if self.math.get().is_some() {
            return Ok(());
        }
        let sms = usize::try_from(
            self.device
                .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|_| "invalid SM count")?;
        let count = sms * crate::gpu_prefill::MARLIN_MAX_LOCK_SLOTS_PER_SM;
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
        if free < 262144 * 4 + count * 4 + 600 * 1024 * 1024 {
            return Err("dense math workspace would breach reserve".into());
        }
        let function =
            crate::gpu_prefill::load_marlin_mm().ok_or("validated Marlin sidecar unavailable")?;
        let partial = self
            .device
            .alloc_zeros::<f32>(262144)
            .map_err(|e| e.to_string())?;
        let locks = self
            .device
            .alloc_zeros::<u32>(count)
            .map_err(|e| e.to_string())?;
        self.math
            .set(DenseMath {
                function,
                partial,
                locks,
                sms,
            })
            .map_err(|_| "dense math initialized twice".to_string())
    }
    /// M=1 tensor-core contraction on the caller's stream, with graph-stable scratch.
    /// FP32-producing projections retain their established independent dispatch.
    pub fn tensorcore_gemv(
        &self,
        packed: u64,
        scales: u64,
        input: u64,
        output: u64,
        rows: usize,
        cols: usize,
        group: usize,
        stream: sys::CUstream,
    ) -> Result<bool, String> {
        let Some(math) = self.math.get() else {
            return Ok(false);
        };
        if rows == 0
            || rows > math.partial.len()
            || rows % 64 != 0
            || cols % 64 != 0
            || ![64,128].contains(&group)
            || cols % group != 0
        {
            return Ok(false);
        }
        unsafe {
            check(
                sys::lib().cuMemsetD32Async(*math.locks.device_ptr(), 0, math.locks.len(), stream),
                "dense math locks",
            )?;
            check(
                sys::lib().cuMemsetD32Async(*math.partial.device_ptr(), 0, rows, stream),
                "dense math reduction",
            )?;
            (math.function)(
                input as _,
                packed as _,
                output as _,
                *math.partial.device_ptr() as _,
                scales as _,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                1,
                rows as i32,
                cols as i32,
                cols as i32,
                *math.locks.device_ptr() as _,
                &crate::gpu_prefill::ScalarType::U4B8,
                false,
                true,
                false,
                (cols / group) as i32,
                group as i32,
                self.device.ordinal() as i32,
                stream as u64,
                -1,
                -1,
                math.sms as i32,
                false,
                true,
                false,
            );
        }
        Ok(true)
    }
    pub fn projection_scratch(&self, elements: usize) -> Result<u64, String> {
        if elements > self.projection.len() {
            return Err("Qwen projection scratch bound exceeded".into());
        }
        Ok(*self.projection.device_ptr())
    }
    pub fn host_pointer(&self, group:usize) -> Result<usize,String> {
        let g=self.groups.lock().map_err(|_|"dense stream lock poisoned")?;
        let sources=g.sources.get(&group).ok_or("missing host weight group")?;
        if sources.len()!=1 {return Err("host pointer requires exactly one owner".into());}
        Ok(sources[0].ptr as usize)
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
            if s.mapped_access {
                continue;
            }
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
        Ok(serde_json::json!({"device_slab_bytes":self.slab.len(),"resident_capacity_bytes":self.resident.len(),"resident_weight_bytes":g.resident_used,"forward_stream_bytes":g.sources.values().flatten().filter(|s|!s.mapped_access).map(|s|s.bytes).sum::<usize>(),"pinned_host_bytes":g.sources.values().flatten().map(|s|s.bytes).sum::<usize>(),"group_bytes":g.extents,"tensorcore_decode":self.math.get().is_some(),"sealed":g.sealed}).to_string())
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
