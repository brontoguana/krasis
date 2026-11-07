//! Fixed physical GPU window over a full-context host-NUMA VMM allocation.
//! Map each migration unit separately: CUDA forbids partial cuMemUnmap.
use super::{checked, range_with_page};
use cudarc::driver::{sys, CudaDevice};
use std::sync::{Arc, Mutex};

struct State {
    owners: Vec<Vec<Option<usize>>>,
    mapped: Vec<bool>,
    poisoned: bool,
}
pub(super) struct Vmm {
    device: Arc<CudaDevice>,
    pub ptr: u64,
    bytes: usize,
    host_alias: u64,
    gpu_alias: u64,
    host_handles: Vec<sys::CUmemGenericAllocationHandle>,
    host_alias_mapped: Vec<bool>,
    gpu_handles: Vec<sys::CUmemGenericAllocationHandle>,
    gpu_alias_mapped: Vec<bool>,
    pub page: usize,
    pub gpu_bytes: usize,
    layers: usize,
    rows: usize,
    stride: usize,
    window: usize,
    pages_per_layer: usize,
    slots: usize,
    state: Mutex<State>,
}
unsafe impl Send for Vmm {}
unsafe impl Sync for Vmm {}
fn prop(kind: sys::CUmemLocationType_enum, id: i32) -> sys::CUmemAllocationProp {
    let mut p = sys::CUmemAllocationProp::default();
    p.type_ = sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type_ = kind;
    p.location.id = id;
    p
}
fn access(ptr: u64, bytes: usize, ordinal: i32, host: bool) -> Result<(), String> {
    let mut d = sys::CUmemAccessDesc::default();
    d.location.type_ = sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = ordinal;
    d.flags = sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    let mut descs = vec![d];
    if host {
        let mut h = d;
        h.location.type_ = sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_HOST_NUMA;
        h.location.id = 0;
        descs.push(h);
    }
    checked(
        unsafe { sys::lib().cuMemSetAccess(ptr, bytes, descs.as_ptr(), descs.len()) },
        "VMM set access",
    )
}
impl Vmm {
    pub fn new(
        device: Arc<CudaDevice>,
        layers: usize,
        rows: usize,
        stride: usize,
        window: usize,
    ) -> Result<Self, String> {
        let ordinal = device.ordinal() as i32;
        let mut supported = 0;
        checked(
            unsafe {
                sys::lib().cuDeviceGetAttribute(&mut supported,sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,ordinal)
            },
            "query VMM support",
        )?;
        if supported != 1 {
            return Err("host-backed KV VMM unavailable; use resident path".into());
        }
        let mut numa = 0;
        checked(
            unsafe {
                sys::lib().cuDeviceGetAttribute(
                    &mut numa,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                    ordinal,
                )
            },
            "query host NUMA",
        )?;
        let hp = prop(
            sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_HOST_NUMA,
            numa.max(0),
        );
        let gp = prop(
            sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
            ordinal,
        );
        let mut hg = 0;
        let mut gg = 0;
        checked(
            unsafe {
                sys::lib().cuMemGetAllocationGranularity(
                    &mut hg,
                    &hp,
                    sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "host VMM granularity",
        )?;
        checked(
            unsafe {
                sys::lib().cuMemGetAllocationGranularity(
                    &mut gg,
                    &gp,
                    sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "GPU VMM granularity",
        )?;
        let page = hg.max(gg);
        if hg == 0 || gg == 0 || page % hg != 0 || page % gg != 0 {
            return Err("host/GPU VMM granularities incompatible".into());
        }
        let layer_bytes = rows.checked_mul(stride).ok_or("VMM layer overflow")?;
        if layer_bytes % page != 0 {
            return Err(format!(
                "VMM KV layer extent {layer_bytes} not aligned to migration granularity {page}"
            ));
        }
        let bytes = layers
            .checked_mul(layer_bytes)
            .ok_or("VMM allocation overflow")?;
        let (lo, hi) = range_with_page(rows, stride, window, window, page)?;
        let slots = (hi - lo) / page;
        let gpu_bytes = layers
            .checked_mul(slots)
            .and_then(|x| x.checked_mul(page))
            .ok_or("VMM GPU budget overflow")?;
        let mut free = 0;
        let mut total = 0;
        checked(
            unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) },
            "VMM budget",
        )?;
        if free < gpu_bytes.saturating_add(600 * 1024 * 1024) {
            return Err("VMM working set cannot preserve600MiB reserve".into());
        }
        let pages = bytes / page;
        let mut v = Self {
            device,
            ptr: 0,
            bytes,
            host_alias: 0,
            gpu_alias: 0,
            host_handles: Vec::new(),
            host_alias_mapped: vec![false; pages],
            gpu_handles: Vec::new(),
            gpu_alias_mapped: vec![false; layers * slots],
            page,
            gpu_bytes,
            layers,
            rows,
            stride,
            window,
            pages_per_layer: layer_bytes / page,
            slots,
            state: Mutex::new(State {
                owners: vec![vec![None; slots]; layers],
                mapped: vec![false; pages],
                poisoned: false,
            }),
        };
        // Separate physical handles avoid unsupported nonzero map offsets.
        checked(
            unsafe { sys::lib().cuMemAddressReserve(&mut v.host_alias, bytes, page, 0, 0) },
            "reserve host alias",
        )?;
        for p in 0..pages {
            let mut h = 0;
            checked(
                unsafe { sys::lib().cuMemCreate(&mut h, page, &hp, 0) },
                "create host-NUMA page",
            )?;
            v.host_handles.push(h);
            checked(
                unsafe { sys::lib().cuMemMap(v.host_alias + (p * page) as u64, page, 0, h, 0) },
                "map host alias page",
            )?;
            v.host_alias_mapped[p] = true;
        }
        access(v.host_alias, bytes, ordinal, true)?;
        unsafe { std::ptr::write_bytes(v.host_alias as *mut u8, 0, bytes) };
        checked(
            unsafe { sys::lib().cuMemAddressReserve(&mut v.ptr, bytes, page, 0, 0) },
            "reserve stable context",
        )?;
        for p in 0..pages {
            checked(
                unsafe {
                    sys::lib().cuMemMap(v.ptr + (p * page) as u64, page, 0, v.host_handles[p], 0)
                },
                "map host context page",
            )?;
            v.state.get_mut().map_err(|_| "VMM state poisoned")?.mapped[p] = true;
        }
        access(v.ptr, bytes, ordinal, false)?;
        checked(
            unsafe { sys::lib().cuMemAddressReserve(&mut v.gpu_alias, gpu_bytes, page, 0, 0) },
            "reserve GPU alias",
        )?;
        for slot in 0..layers * slots {
            let mut g = 0;
            checked(
                unsafe { sys::lib().cuMemCreate(&mut g, page, &gp, 0) },
                "create GPU window page",
            )?;
            v.gpu_handles.push(g);
            checked(
                unsafe { sys::lib().cuMemMap(v.gpu_alias + (slot * page) as u64, page, 0, g, 0) },
                "map GPU alias page",
            )?;
            v.gpu_alias_mapped[slot] = true;
        }
        access(v.gpu_alias, gpu_bytes, ordinal, false)?;
        for l in 0..layers {
            v.move_layer(l, window, *v.device.cu_stream())?;
        }
        v.device.synchronize().map_err(|e| e.to_string())?;
        log::warn!("EXPERIMENT VMM full-context KV: host_bytes={bytes} fixed_gpu_bytes={gpu_bytes} granularity={page} layers={layers} window={window} complete_capacity={rows}");
        Ok(v)
    }
    pub fn move_layer(
        &self,
        layer: usize,
        end: usize,
        stream: sys::CUstream,
    ) -> Result<(u64, u64), String> {
        if layer >= self.layers {
            return Err("VMM KV layer out of bounds".into());
        }
        let (lo, hi) = range_with_page(self.rows, self.stride, self.window, end, self.page)?;
        let first = lo / self.page;
        let last = hi / self.page;
        let mut s = self.state.lock().map_err(|_| "VMM state lock poisoned")?;
        if s.poisoned {
            return Err("VMM mapping transition previously failed; restart candidate".into());
        }
        let incoming: Vec<usize> = (first..last)
            .filter(|&p| s.owners[layer][p % self.slots] != Some(p))
            .collect();
        if incoming.is_empty() {
            return Ok((0, 0));
        }
        let mut capture = sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_NONE;
        checked(
            unsafe { sys::lib().cuStreamIsCapturing(stream, &mut capture) },
            "VMM capture check",
        )?;
        if capture != sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err("KV VMM mapping transition cannot run inside CUDA capture".into());
        }
        s.poisoned = true;
        // Mapping changes occur only between operations, never during capture
        // or a live replay. One coarse boundary wait protects all old users.
        checked(
            unsafe { sys::lib().cuStreamSynchronize(stream) },
            "VMM mapping boundary wait",
        )?;
        let layer_base = layer * self.pages_per_layer;
        let mut dtoh = 0u64;
        for &p in &incoming {
            let slot = p % self.slots;
            if let Some(old) = s.owners[layer][slot] {
                let offset = (layer_base + old) * self.page;
                let gpu_offset = (layer * self.slots + slot) * self.page;
                checked(
                    unsafe {
                        sys::lib().cuMemcpyAsync(
                            self.host_alias + offset as u64,
                            self.gpu_alias + gpu_offset as u64,
                            self.page,
                            stream,
                        )
                    },
                    "VMM evict authoritative page",
                )?;
                dtoh += self.page as u64;
            }
        }
        checked(
            unsafe { sys::lib().cuStreamSynchronize(stream) },
            "VMM writeback wait",
        )?;
        for &p in &incoming {
            let slot = p % self.slots;
            if let Some(old) = s.owners[layer][slot] {
                let page_idx = layer_base + old;
                let offset = page_idx * self.page;
                checked(
                    unsafe { sys::lib().cuMemUnmap(self.ptr + offset as u64, self.page) },
                    "unmap evicted GPU page",
                )?;
                s.mapped[page_idx] = false;
                checked(
                    unsafe {
                        sys::lib().cuMemMap(
                            self.ptr + offset as u64,
                            self.page,
                            0,
                            self.host_handles[page_idx],
                            0,
                        )
                    },
                    "restore host mapping",
                )?;
                s.mapped[page_idx] = true;
                access(
                    self.ptr + offset as u64,
                    self.page,
                    self.device.ordinal() as i32,
                    false,
                )?;
            }
        }
        for &p in &incoming {
            let offset = (layer_base + p) * self.page;
            let gpu_offset = (layer * self.slots + p % self.slots) * self.page;
            checked(
                unsafe {
                    sys::lib().cuMemcpyAsync(
                        self.gpu_alias + gpu_offset as u64,
                        self.host_alias + offset as u64,
                        self.page,
                        stream,
                    )
                },
                "VMM fill GPU page",
            )?;
        }
        checked(
            unsafe { sys::lib().cuStreamSynchronize(stream) },
            "VMM fill wait",
        )?;
        for &p in &incoming {
            let page_idx = layer_base + p;
            let offset = page_idx * self.page;
            let slot = p % self.slots;
            let gpu_offset = (layer * self.slots + slot) * self.page;
            checked(
                unsafe { sys::lib().cuMemUnmap(self.ptr + offset as u64, self.page) },
                "unmap incoming host page",
            )?;
            s.mapped[page_idx] = false;
            checked(
                unsafe {
                    sys::lib().cuMemMap(
                        self.ptr + offset as u64,
                        self.page,
                        0,
                        self.gpu_handles[layer * self.slots + slot],
                        0,
                    )
                },
                "map incoming GPU page",
            )?;
            s.mapped[page_idx] = true;
            access(
                self.ptr + offset as u64,
                self.page,
                self.device.ordinal() as i32,
                false,
            )?;
            s.owners[layer][slot] = Some(p);
        }
        s.poisoned = false;
        Ok(((incoming.len() * self.page) as u64, dtoh))
    }
}
impl Drop for Vmm {
    fn drop(&mut self) {
        if self.device.bind_to_thread().is_err() || self.device.synchronize().is_err() {
            log::error!("VMM context unavailable; retaining backing");
            return;
        }
        unsafe {
            if self.ptr != 0 {
                if let Ok(s) = self.state.get_mut() {
                    for (i, &mapped) in s.mapped.iter().enumerate() {
                        if mapped {
                            let _ =
                                sys::lib().cuMemUnmap(self.ptr + (i * self.page) as u64, self.page);
                        }
                    }
                }
                let _ = sys::lib().cuMemAddressFree(self.ptr, self.bytes);
            }
            if self.host_alias != 0 {
                for (i, &mapped) in self.host_alias_mapped.iter().enumerate() {
                    if mapped {
                        let _ = sys::lib()
                            .cuMemUnmap(self.host_alias + (i * self.page) as u64, self.page);
                    }
                }
                let _ = sys::lib().cuMemAddressFree(self.host_alias, self.bytes);
            }
            if self.gpu_alias != 0 {
                for (i, &mapped) in self.gpu_alias_mapped.iter().enumerate() {
                    if mapped {
                        let _ = sys::lib()
                            .cuMemUnmap(self.gpu_alias + (i * self.page) as u64, self.page);
                    }
                }
                let _ = sys::lib().cuMemAddressFree(self.gpu_alias, self.gpu_bytes);
            }
            for h in self.host_handles.drain(..) {
                let _ = sys::lib().cuMemRelease(h);
            }
            for g in self.gpu_handles.drain(..) {
                let _ = sys::lib().cuMemRelease(g);
            }
        }
    }
}
