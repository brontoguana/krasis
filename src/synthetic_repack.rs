//! Default-off synthetic CPU-layout -> Marlin GPU repack cost probe.
//!
//! The authoritative host cache remains Marlin. Runtime copy paths feed those
//! bytes to the genuine canonical-layout repacker and discard the result. This
//! preserves model numerics while measuring the stream dependency, GPU memory
//! traffic, launch overhead, and low-VRAM staging-ring footprint of a future
//! CPU-friendly host cache.

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

const MODULE_NAME: &str = "decode_kernels";
const FUSED_KERNEL: &str = "cpu_expert_to_marlin_repack_batched";

pub(crate) fn synthetic_repack_enabled() -> bool {
    std::env::var("KRASIS_SYNTH_REPACK")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn timing_enabled() -> bool {
    std::env::var("KRASIS_DECODE_TIMING")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RepackMatrixShape {
    pub n: usize,
    pub k: usize,
    pub packed_bytes: usize,
    pub scales_bytes: usize,
}

impl RepackMatrixShape {
    fn validate(self, group_size: usize, label: &str) -> Result<(), String> {
        if self.n == 0 || self.k == 0 || group_size == 0 {
            return Err(format!(
                "synthetic repack {label} requires positive N/K/group_size, got N={} K={} group_size={}",
                self.n, self.k, group_size
            ));
        }
        if self.n % 64 != 0 || self.k % 32 != 0 || self.k % group_size != 0 {
            return Err(format!(
                "synthetic repack {label} unsupported shape N={} K={} group_size={}: require N%64=0, K%32=0, K%group_size=0",
                self.n, self.k, group_size
            ));
        }
        let expected_packed = self
            .n
            .checked_mul(self.k)
            .and_then(|elements| elements.checked_div(2))
            .ok_or_else(|| format!("synthetic repack {label} packed-size overflow"))?;
        let expected_scales = self
            .n
            .checked_mul(self.k / group_size)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| format!("synthetic repack {label} scale-size overflow"))?;
        if self.packed_bytes != expected_packed || self.scales_bytes != expected_scales {
            return Err(format!(
                "synthetic repack {label} byte contract mismatch: packed={}/{} scales={}/{} for N={} K={} group_size={}",
                self.packed_bytes,
                expected_packed,
                self.scales_bytes,
                expected_scales,
                self.n,
                self.k,
                group_size
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SyntheticRepackStats {
    pub launches: u64,
    pub experts: u64,
    pub source_bytes: u64,
    pub copy_seconds: f64,
    pub kernel_seconds: f64,
}

pub(crate) struct SyntheticRepackRing {
    d_input_ptrs: CudaSlice<u64>,
    h_input_ptrs: Vec<u64>,
    d_scratch: CudaSlice<u8>,
    slot_count: usize,
    slot_stride: usize,
    component_offsets: [usize; 4],
    w13: RepackMatrixShape,
    w2: RepackMatrixShape,
    group_size: usize,
    repack_stream: cuda_sys::CUstream,
    copy_start_events: Vec<cuda_sys::CUevent>,
    input_ready_events: Vec<cuda_sys::CUevent>,
    completion_events: Vec<cuda_sys::CUevent>,
    fused_kernel: cuda_sys::CUfunction,
    timing_start: cuda_sys::CUevent,
    timing_stop: cuda_sys::CUevent,
    timing: bool,
    next_batch: usize,
    stats: SyntheticRepackStats,
}

unsafe impl Send for SyntheticRepackRing {}
unsafe impl Sync for SyntheticRepackRing {}

struct PendingCudaResources {
    stream: cuda_sys::CUstream,
    events: Vec<cuda_sys::CUevent>,
}

impl PendingCudaResources {
    fn new() -> Self {
        Self {
            stream: std::ptr::null_mut(),
            events: Vec::new(),
        }
    }

    fn disarm(&mut self) {
        self.stream = std::ptr::null_mut();
        self.events.clear();
    }
}

impl Drop for PendingCudaResources {
    fn drop(&mut self) {
        unsafe {
            for event in self.events.drain(..) {
                if !event.is_null() {
                    let _ = cuda_sys::lib().cuEventDestroy_v2(event);
                }
            }
            if !self.stream.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.stream);
                self.stream = std::ptr::null_mut();
            }
        }
    }
}

impl SyntheticRepackRing {
    pub(crate) fn matches_layout(
        &self,
        slot_count: usize,
        max_experts_per_layer: usize,
        w13: RepackMatrixShape,
        w2: RepackMatrixShape,
        group_size: usize,
    ) -> bool {
        self.slot_count == slot_count
            && self.completion_events.len() == max_experts_per_layer.div_ceil(slot_count)
            && self.w13 == w13
            && self.w2 == w2
            && self.group_size == group_size
    }

    pub(crate) fn new(
        device: &Arc<CudaDevice>,
        slot_count: usize,
        max_experts_per_layer: usize,
        w13: RepackMatrixShape,
        w2: RepackMatrixShape,
        group_size: usize,
    ) -> Result<Self, String> {
        if slot_count == 0 || max_experts_per_layer == 0 {
            return Err(format!(
                "synthetic repack ring requires positive slot/layer capacity, got slots={} max_experts={}",
                slot_count, max_experts_per_layer
            ));
        }
        w13.validate(group_size, "w13")?;
        w2.validate(group_size, "w2")?;

        let component_offsets = [
            0,
            w13.packed_bytes,
            w13.packed_bytes + w13.scales_bytes,
            w13.packed_bytes + w13.scales_bytes + w2.packed_bytes,
        ];
        let slot_stride = component_offsets[3]
            .checked_add(w2.scales_bytes)
            .ok_or_else(|| "synthetic repack expert-slot size overflow".to_string())?;
        let scratch_bytes = slot_stride
            .checked_mul(slot_count)
            .ok_or_else(|| "synthetic repack scratch-ring size overflow".to_string())?;
        let fused_func = device
            .get_func(MODULE_NAME, FUSED_KERNEL)
            .ok_or_else(|| format!("synthetic repack kernel {FUSED_KERNEL} is unavailable"))?;
        let fused_kernel = extract_cu_function(&fused_func);

        let batch_capacity = max_experts_per_layer.div_ceil(slot_count).max(1);
        // Every queued batch needs its own device pointer table. The copy stream
        // is allowed to upload batch N+1 while the repack stream still consumes
        // batch N, so a single shared pointer table would race.
        let d_input_ptrs = device
            .alloc_zeros::<u64>(slot_count * 4 * batch_capacity)
            .map_err(|error| format!("allocate synthetic repack pointer ring: {error:?}"))?;
        let d_scratch = device
            .alloc_zeros::<u8>(scratch_bytes)
            .map_err(|error| format!("allocate synthetic repack scratch ring: {error:?}"))?;

        // Until Self exists, constructor failures must tear down raw CUDA
        // resources explicitly. Successful construction disarms this guard and
        // transfers ownership to SyntheticRepackRing::drop.
        let mut pending = PendingCudaResources::new();
        unsafe {
            let result = cuda_sys::lib().cuStreamCreate(
                &mut pending.stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("create synthetic repack stream: {result:?}"));
            }
        }

        let timing = timing_enabled();
        let mut copy_start_events = Vec::with_capacity(batch_capacity);
        let mut input_ready_events = Vec::with_capacity(batch_capacity);
        let mut completion_events = Vec::with_capacity(batch_capacity);
        for _ in 0..batch_capacity {
            let copy_start = if timing {
                create_event(true)?
            } else {
                std::ptr::null_mut()
            };
            if !copy_start.is_null() {
                pending.events.push(copy_start);
            }
            copy_start_events.push(copy_start);
            let input_ready = create_event(timing)?;
            pending.events.push(input_ready);
            input_ready_events.push(input_ready);
            let completion = create_event(false)?;
            pending.events.push(completion);
            completion_events.push(completion);
        }
        let timing_start = if timing {
            create_event(true)?
        } else {
            std::ptr::null_mut()
        };
        if !timing_start.is_null() {
            pending.events.push(timing_start);
        }
        let timing_stop = if timing {
            create_event(true)?
        } else {
            std::ptr::null_mut()
        };
        if !timing_stop.is_null() {
            pending.events.push(timing_stop);
        }
        let repack_stream = pending.stream;
        pending.disarm();

        eprintln!(
            "SYNTHETIC REPACK RING enabled slots={} max_experts_per_layer={} batches={} slot_bytes={} scratch_bytes={} scratch_mib={:.3} w13={}x{} w2={}x{} group_size={} timing={}",
            slot_count,
            max_experts_per_layer,
            batch_capacity,
            slot_stride,
            scratch_bytes,
            scratch_bytes as f64 / (1024.0 * 1024.0),
            w13.n,
            w13.k,
            w2.n,
            w2.k,
            group_size,
            timing,
        );

        Ok(Self {
            d_input_ptrs,
            h_input_ptrs: vec![0; slot_count * 4],
            d_scratch,
            slot_count,
            slot_stride,
            component_offsets,
            w13,
            w2,
            group_size,
            repack_stream,
            copy_start_events,
            input_ready_events,
            completion_events,
            fused_kernel,
            timing_start,
            timing_stop,
            timing,
            next_batch: 0,
            stats: SyntheticRepackStats::default(),
        })
    }

    pub(crate) fn begin_layer(&mut self) {
        self.next_batch = 0;
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        self.slot_count * self.slot_stride
    }

    pub(crate) fn stats(&self) -> SyntheticRepackStats {
        self.stats
    }

    pub(crate) fn take_stats(&mut self) -> SyntheticRepackStats {
        std::mem::take(&mut self.stats)
    }

    pub(crate) fn begin_copy_batch(
        &mut self,
        copy_stream: cuda_sys::CUstream,
    ) -> Result<(), String> {
        if self.next_batch >= self.input_ready_events.len() {
            return Err(format!(
                "synthetic repack copy batch exceeds configured capacity {}",
                self.input_ready_events.len()
            ));
        }
        if self.timing {
            unsafe {
                let result = cuda_sys::lib()
                    .cuEventRecord(self.copy_start_events[self.next_batch], copy_stream);
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!(
                        "record synthetic repack copy timing start: {result:?}"
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn launch_batch(
        &mut self,
        copy_stream: cuda_sys::CUstream,
        pointers: &[[u64; 4]],
    ) -> Result<(), String> {
        let batch = pointers.len();
        if batch == 0 {
            return Ok(());
        }
        if batch > self.slot_count {
            return Err(format!(
                "synthetic repack batch {} exceeds ring slots {}",
                batch, self.slot_count
            ));
        }
        if self.next_batch >= self.input_ready_events.len() {
            return Err(format!(
                "synthetic repack layer requires more than {} batches; max-expert/ring contract is inconsistent",
                self.input_ready_events.len()
            ));
        }

        self.h_input_ptrs.fill(0);
        for (slot, ptrs) in pointers.iter().enumerate() {
            for component in 0..4 {
                self.h_input_ptrs[component * self.slot_count + slot] = ptrs[component];
            }
        }
        unsafe {
            let pointer_batch_offset =
                self.next_batch * self.slot_count * 4 * std::mem::size_of::<u64>();
            let result = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                *self.d_input_ptrs.device_ptr() + pointer_batch_offset as u64,
                self.h_input_ptrs.as_ptr() as *const std::ffi::c_void,
                self.h_input_ptrs.len() * std::mem::size_of::<u64>(),
                copy_stream,
            );
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("upload synthetic repack pointer ring: {result:?}"));
            }
            let input_ready = self.input_ready_events[self.next_batch];
            let result = cuda_sys::lib().cuEventRecord(input_ready, copy_stream);
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "record synthetic repack input-ready event: {result:?}"
                ));
            }
            let result = cuda_sys::lib().cuStreamWaitEvent(self.repack_stream, input_ready, 0);
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "wait synthetic repack input-ready event: {result:?}"
                ));
            }
            if self.next_batch > 0 {
                let previous = self.completion_events[self.next_batch - 1];
                let result = cuda_sys::lib().cuStreamWaitEvent(self.repack_stream, previous, 0);
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!(
                        "wait synthetic repack ring-reuse completion event: {result:?}"
                    ));
                }
            }
            if self.timing {
                let result = cuda_sys::lib().cuEventRecord(self.timing_start, self.repack_stream);
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("record synthetic repack timing start: {result:?}"));
                }
            }
        }

        self.launch_experts(batch)?;

        unsafe {
            if self.timing {
                let result = cuda_sys::lib().cuEventRecord(self.timing_stop, self.repack_stream);
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("record synthetic repack timing stop: {result:?}"));
                }
                let result = cuda_sys::lib().cuEventSynchronize(self.timing_stop);
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!(
                        "synchronize synthetic repack timing stop: {result:?}"
                    ));
                }
                let mut elapsed_ms = 0.0f32;
                let result = cuda_sys::lib().cuEventElapsedTime(
                    &mut elapsed_ms,
                    self.timing_start,
                    self.timing_stop,
                );
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("measure synthetic repack events: {result:?}"));
                }
                self.stats.kernel_seconds += elapsed_ms as f64 / 1000.0;
                let mut copy_elapsed_ms = 0.0f32;
                let result = cuda_sys::lib().cuEventElapsedTime(
                    &mut copy_elapsed_ms,
                    self.copy_start_events[self.next_batch],
                    self.input_ready_events[self.next_batch],
                );
                if result != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!(
                        "measure synthetic repack input copy events: {result:?}"
                    ));
                }
                self.stats.copy_seconds += copy_elapsed_ms as f64 / 1000.0;
            }
            let completion = self.completion_events[self.next_batch];
            let result = cuda_sys::lib().cuEventRecord(completion, self.repack_stream);
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "record synthetic repack completion event: {result:?}"
                ));
            }
        }

        self.stats.launches += 1;
        self.stats.experts += batch as u64;
        self.stats.source_bytes += (batch * self.slot_stride) as u64;
        self.next_batch += 1;
        Ok(())
    }

    pub(crate) fn completion_event(&self) -> Option<cuda_sys::CUevent> {
        self.next_batch
            .checked_sub(1)
            .map(|index| self.completion_events[index])
    }

    pub(crate) fn repack_stream(&self) -> cuda_sys::CUstream {
        self.repack_stream
    }

    fn launch_experts(&self, batch: usize) -> Result<(), String> {
        let batch_pointer_words = self.slot_count * 4;
        let batch_pointer_base = self.next_batch * batch_pointer_words;
        let pointer_stride = self.slot_count * std::mem::size_of::<u64>();
        let mut w13_ptrs = *self.d_input_ptrs.device_ptr()
            + (batch_pointer_base * std::mem::size_of::<u64>()) as u64;
        let mut w13s_ptrs = w13_ptrs + pointer_stride as u64;
        let mut w2_ptrs = w13s_ptrs + pointer_stride as u64;
        let mut w2s_ptrs = w2_ptrs + pointer_stride as u64;
        let mut output_base = *self.d_scratch.device_ptr();
        let mut batch_i32 = i32::try_from(batch)
            .map_err(|_| format!("synthetic repack batch exceeds i32: {batch}"))?;
        let mut w13_n = i32::try_from(self.w13.n)
            .map_err(|_| format!("synthetic repack W13 N exceeds i32: {}", self.w13.n))?;
        let mut w13_k = i32::try_from(self.w13.k)
            .map_err(|_| format!("synthetic repack W13 K exceeds i32: {}", self.w13.k))?;
        let mut w2_n = i32::try_from(self.w2.n)
            .map_err(|_| format!("synthetic repack W2 N exceeds i32: {}", self.w2.n))?;
        let mut w2_k = i32::try_from(self.w2.k)
            .map_err(|_| format!("synthetic repack W2 K exceeds i32: {}", self.w2.k))?;
        let mut group_size = i32::try_from(self.group_size).map_err(|_| {
            format!(
                "synthetic repack group size exceeds i32: {}",
                self.group_size
            )
        })?;
        let mut stride = self.slot_stride as u64;
        let mut w13s_offset = self.component_offsets[1] as u64;
        let mut w2_offset = self.component_offsets[2] as u64;
        let mut w2s_offset = self.component_offsets[3] as u64;
        let blocks = (self.w13.k / 32)
            .checked_mul(self.w13.n / 64)
            .and_then(|w13_blocks| {
                (self.w2.k / 32)
                    .checked_mul(self.w2.n / 64)
                    .and_then(|w2_blocks| w13_blocks.checked_add(w2_blocks))
            })
            .ok_or_else(|| "synthetic repack fused grid overflow".to_string())?;
        unsafe {
            launch_raw(
                self.fused_kernel,
                (blocks as u32, batch as u32, 1),
                (128, 1, 1),
                self.repack_stream,
                &mut [
                    &mut w13_ptrs as *mut _ as *mut std::ffi::c_void,
                    &mut w13s_ptrs as *mut _ as *mut std::ffi::c_void,
                    &mut w2_ptrs as *mut _ as *mut std::ffi::c_void,
                    &mut w2s_ptrs as *mut _ as *mut std::ffi::c_void,
                    &mut output_base as *mut _ as *mut std::ffi::c_void,
                    &mut batch_i32 as *mut _ as *mut std::ffi::c_void,
                    &mut w13_n as *mut _ as *mut std::ffi::c_void,
                    &mut w13_k as *mut _ as *mut std::ffi::c_void,
                    &mut w2_n as *mut _ as *mut std::ffi::c_void,
                    &mut w2_k as *mut _ as *mut std::ffi::c_void,
                    &mut group_size as *mut _ as *mut std::ffi::c_void,
                    &mut stride as *mut _ as *mut std::ffi::c_void,
                    &mut w13s_offset as *mut _ as *mut std::ffi::c_void,
                    &mut w2_offset as *mut _ as *mut std::ffi::c_void,
                    &mut w2s_offset as *mut _ as *mut std::ffi::c_void,
                ],
            )
        }
    }
}

impl Drop for SyntheticRepackRing {
    fn drop(&mut self) {
        unsafe {
            if !self.repack_stream.is_null() {
                let _ = cuda_sys::lib().cuStreamSynchronize(self.repack_stream);
            }
            for event in self.copy_start_events.drain(..) {
                if !event.is_null() {
                    let _ = cuda_sys::lib().cuEventDestroy_v2(event);
                }
            }
            for event in self.input_ready_events.drain(..) {
                if !event.is_null() {
                    let _ = cuda_sys::lib().cuEventDestroy_v2(event);
                }
            }
            for event in self.completion_events.drain(..) {
                if !event.is_null() {
                    let _ = cuda_sys::lib().cuEventDestroy_v2(event);
                }
            }
            if !self.timing_start.is_null() {
                let _ = cuda_sys::lib().cuEventDestroy_v2(self.timing_start);
            }
            if !self.timing_stop.is_null() {
                let _ = cuda_sys::lib().cuEventDestroy_v2(self.timing_stop);
            }
            if !self.repack_stream.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.repack_stream);
                self.repack_stream = std::ptr::null_mut();
            }
        }
    }
}

fn create_event(timing: bool) -> Result<cuda_sys::CUevent, String> {
    let mut event = std::ptr::null_mut();
    let flags = if timing {
        0
    } else {
        cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32
    };
    unsafe {
        let result = cuda_sys::lib().cuEventCreate(&mut event, flags);
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!("create synthetic repack CUDA event: {result:?}"));
        }
    }
    Ok(event)
}

unsafe fn launch_raw(
    function: cuda_sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: cuda_sys::CUstream,
    params: &mut [*mut std::ffi::c_void],
) -> Result<(), String> {
    let result = cuda_sys::lib().cuLaunchKernel(
        function,
        grid.0,
        grid.1,
        grid.2,
        block.0,
        block.1,
        block.2,
        0,
        stream,
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if result == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "launch synthetic repack kernel: {result:?} grid={grid:?} block={block:?}"
        ))
    }
}

fn extract_cu_function(function: &cudarc::driver::CudaFunction) -> cuda_sys::CUfunction {
    unsafe {
        let struct_ptr = function as *const _ as *const u8;
        let word0: cuda_sys::CUfunction = std::ptr::read(struct_ptr as *const _);
        let mut attribute = 0i32;
        if cuda_sys::lib().cuFuncGetAttribute(
            &mut attribute,
            cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
            word0,
        ) == cuda_sys::CUresult::CUDA_SUCCESS
        {
            word0
        } else {
            std::ptr::read(struct_ptr.add(std::mem::size_of::<usize>()) as *const _)
        }
    }
}
