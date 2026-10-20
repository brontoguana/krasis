//! Explicit diagnostic recurrence contract. Model dispatch is not yet admitted.
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{c_void, CString},
    path::PathBuf,
    sync::Arc,
};

pub const BLOCK_ROWS: usize = 16;
include!("prefix_recurrence_guard.rs");
const MODULE_NAME: &str = "fixed16-original-raw-f32-v1";
const SYMBOLS: &[&str] = &[
    "batch_g",
    "batch_precompute_compact",
    "batch_coeff_compact",
    "batch_inverse",
    "batch_carry_compact",
    "raw_prepare",
    "cursor_advance",
    "cursor_append",
    "cursor_output_reset",
    "cursor_raw_prepare",
    "cursor_batch_g",
    "cursor_batch_precompute_compact",
    "cursor_batch_coeff_compact",
    "cursor_batch_carry_compact",
    "cursor_emit_commit",
    "cursor_clear_completed_raw",
    "cursor_finish",
    "prefix_validate",
    "prefix_prepend",
    "prefix_finalize",
    "prefix_copy_display",
    "prefix_logits_guard",
];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub contract: String,
    pub selected_layers: Vec<usize>,
    pub max_new_rows: usize,
    pub reserve_bytes: usize,
    pub ptx: PathBuf,
    pub ptx_sha256: String,
    pub source_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub heads: usize,
    pub dim: usize,
    pub gate_lower_bound: f32,
    pub a_log: u64,
    pub dt_bias: u64,
}
include!("prefix_workspace_layout.rs");
impl Geometry {
    fn checked(self, max_new_rows: usize) -> Result<(usize, usize, usize), String> {
        if self.heads == 0
            || self.dim == 0
            || self.dim > 128
            || max_new_rows == 0
            || !self.gate_lower_bound.is_finite()
            || self.gate_lower_bound >= 0.0
            || self.a_log == 0
            || self.dt_bias == 0
        {
            return Err("fixed16 diagnostic geometry/parameters unsupported".into());
        }
        let layout = PrefixWorkspaceLayout::checked(self.heads, self.dim, max_new_rows)?;
        Ok((layout.hd, layout.state, layout.padded))
    }
}
fn ptr<T: cudarc::driver::DeviceRepr>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}
fn status(rc: sys::CUresult, operation: &str) -> Result<(), String> {
    if rc == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("fixed16 {operation}: {rc:?}"))
    }
}
fn arguments(pointers: &mut [u64], integers: &mut [i32], floats: &mut [f32]) -> Vec<*mut c_void> {
    pointers
        .iter_mut()
        .map(|x| (x as *mut u64).cast())
        .chain(integers.iter_mut().map(|x| (x as *mut i32).cast()))
        .chain(floats.iter_mut().map(|x| (x as *mut f32).cast()))
        .collect()
}
/// Transient arrays for one explicitly ordered context/geometry execution domain.
/// Separate owners retain their own persistent carries and raw tails.
pub struct PrefixWorkspace {
    device: Arc<CudaDevice>,
    heads: usize,
    dim: usize,
    max_new_rows: usize,
    padded_rows: usize,
    pub allocation_bytes: usize,
    raw: Vec<CudaSlice<u16>>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    g: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    gs: CudaSlice<f32>,
    kg: CudaSlice<f32>,
    qg: CudaSlice<f32>,
    pair: CudaSlice<f32>,
    a: CudaSlice<f32>,
    qk: CudaSlice<f32>,
    inverse: CudaSlice<f32>,
    out: CudaSlice<f32>,
}
include!("prefix_workspace.rs");
pub struct PrefixRecurrence {
    device: Arc<CudaDevice>,
    module: sys::CUmodule,
    functions: BTreeMap<&'static str, sys::CUfunction>,
    geometry: Geometry,
    max_new_rows: usize,
    workspace: Arc<PrefixWorkspace>,
    committed_incoming: CudaSlice<f32>,
    complete: CudaSlice<f32>,
    latest: CudaSlice<f32>,
    raw_pending: Vec<CudaSlice<u16>>,
    pending: CudaSlice<i32>,
    error: CudaSlice<i32>,
    absolute: CudaSlice<u64>,
    /// Payload owned by this layer, plus its workspace only for the dedicated constructor.
    pub allocation_bytes: usize,
    pub persistent_bytes: usize,
    pub arithmetic_sha256: String,
}
impl std::fmt::Debug for PrefixRecurrence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixRecurrence")
            .field("geometry", &self.geometry)
            .field("allocation_bytes", &self.allocation_bytes)
            .field("arithmetic_sha256", &self.arithmetic_sha256)
            .finish()
    }
}
// Owners retain their context. Dispatchers must bind it and serialize their
// existing execution stream; no host lock or cursor read is used per token.
unsafe impl Send for PrefixRecurrence {}
unsafe impl Sync for PrefixRecurrence {}
impl Drop for PrefixRecurrence {
    fn drop(&mut self) {
        crate::prefix_lifecycle_observer::record(
            "actual_owner_drop_enter",
            self.module as usize,
            None,
        );
        if self.device.bind_to_thread().is_ok() {
            unsafe {
                // Prefill can use a caller-owned stream. Complete this context
                // before unloading functions or releasing any owner allocation.
                let rc = sys::lib().cuCtxSynchronize();
                if rc != sys::CUresult::CUDA_SUCCESS {
                    log::error!("fixed16 context cleanup synchronization: {rc:?}");
                }
                let result = sys::lib().cuModuleUnload(self.module);
                crate::prefix_lifecycle_observer::record(
                    "actual_owner_module_unload",
                    self.module as usize,
                    Some(format!("{result:?}")),
                );
            }
        }
    }
}
impl PrefixRecurrence {
    pub fn new(
        device: Arc<CudaDevice>,
        geometry: Geometry,
        plan: &Plan,
        reserve_bytes: usize,
    ) -> Result<Self, String> {
        Self::new_impl(device, geometry, plan, reserve_bytes, None)
    }
    /// A stable block workspace may be shared only by owners dispatched on an
    /// existing ordered stream. Every accepted graph retains these owners.
    pub fn new_shared_decode(
        device: Arc<CudaDevice>,
        geometry: Geometry,
        plan: &Plan,
        reserve_bytes: usize,
        workspace: Arc<PrefixWorkspace>,
    ) -> Result<Self, String> {
        workspace.validate(&device, geometry, 1)?;
        if workspace.padded_rows != BLOCK_ROWS {
            return Err(
                "fixed16 shared decode arena must contain exactly one contract block".into(),
            );
        }
        Self::new_impl(device, geometry, plan, reserve_bytes, Some(workspace))
    }
    fn new_impl(
        device: Arc<CudaDevice>,
        geometry: Geometry,
        plan: &Plan,
        reserve_bytes: usize,
        shared: Option<Arc<PrefixWorkspace>>,
    ) -> Result<Self, String> {
        if plan.contract != MODULE_NAME
            || plan.selected_layers.is_empty()
            || plan
                .selected_layers
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != plan.selected_layers.len()
            || !plan.ptx.is_absolute()
            || plan.source_sha256.len() != 64
        {
            return Err("fixed16 diagnostic plan admission failed".into());
        }
        if plan.source_sha256
            != format!(
                "{:x}",
                Sha256::digest(include_bytes!("cuda/prefix_recurrence.cu"))
            )
        {
            return Err("fixed16 source identity differs from compiled runtime contract".into());
        }
        let (hd, st, padded) = geometry.checked(plan.max_new_rows)?;
        let groups = padded / BLOCK_ROWS;
        let persistent_bytes = BLOCK_ROWS
            .checked_mul(
                4usize
                    .checked_mul(hd)
                    .and_then(|x| x.checked_add(geometry.heads))
                    .ok_or("fixed16 persistent raw extent overflow")?,
            )
            .and_then(|x| x.checked_mul(std::mem::size_of::<u16>()))
            .and_then(|x| {
                3usize
                    .checked_mul(st)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                    .and_then(|n| x.checked_add(n))
            })
            .and_then(|x| {
                x.checked_add(2 * std::mem::size_of::<i32>() + std::mem::size_of::<u64>())
            })
            .ok_or("fixed16 persistent byte overflow")?;
        let dedicated = shared.is_none();
        let workspace = match shared {
            Some(w) => {
                w.validate(&device, geometry, 1)?;
                w
            }
            None => Arc::new(PrefixWorkspace::new(
                device.clone(),
                geometry,
                plan.max_new_rows,
                reserve_bytes,
            )?),
        };
        let allocation_bytes = persistent_bytes
            .checked_add(if dedicated {
                workspace.allocation_bytes
            } else {
                0
            })
            .ok_or("fixed16 accounted payload overflow")?;
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut active = 0;
        status(
            unsafe { sys::lib().cuCtxGetDevice(&mut active) },
            "context device",
        )?;
        let mut major = 0;
        status(
            unsafe {
                sys::lib().cuDeviceGetAttribute(
                    &mut major,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    active,
                )
            },
            "compute capability",
        )?;
        if major < 8 {
            return Err("fixed16 diagnostic requires Ampere or later".into());
        }
        let attribute = |kind| -> Result<i32, String> {
            let mut value = 0;
            status(
                unsafe { sys::lib().cuDeviceGetAttribute(&mut value, kind, active) },
                "device launch limit",
            )?;
            Ok(value)
        };
        let max_x =
            attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize;
        let max_y =
            attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize;
        let max_threads =
            attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as usize;
        if padded > max_x
            || geometry.heads > max_x
            || geometry.heads > max_y
            || groups > max_y
            || hd.div_ceil(256) > max_x
            || max_threads < 256
        {
            return Err("fixed16 geometry exceeds measured device launch limits".into());
        }
        let (mut free, mut total) = (0, 0);
        status(
            unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) },
            "allocation memory",
        )?;
        if free
            < persistent_bytes
                .checked_add(reserve_bytes)
                .ok_or("fixed16 reserve overflow")?
        {
            return Err(format!(
                "fixed16 owner allocation {} plus reserve {} exceeds measured free {}",
                persistent_bytes, reserve_bytes, free
            ));
        }
        let bytes = std::fs::read(&plan.ptx).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != plan.ptx_sha256 {
            return Err("fixed16 PTX identity mismatch".into());
        }
        let ptx = CString::new(bytes).map_err(|e| e.to_string())?;
        let mut module = std::ptr::null_mut();
        status(
            unsafe { sys::lib().cuModuleLoadData(&mut module, ptx.as_ptr().cast()) },
            "module load",
        )?;
        // Allocate under an owned guard so subsequent load/allocation errors
        // release the module rather than leaving hidden context allocations.
        let result = (|| {
            let mut functions = BTreeMap::new();
            for &name in SYMBOLS {
                let symbol = CString::new(name).map_err(|e| e.to_string())?;
                let mut function = std::ptr::null_mut();
                status(
                    unsafe {
                        sys::lib().cuModuleGetFunction(&mut function, module, symbol.as_ptr())
                    },
                    "function load",
                )?;
                let mut threads_limit = 0;
                let mut shared = 0;
                status(
                    unsafe {
                        sys::lib().cuFuncGetAttribute(
                            &mut threads_limit,
                            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            function,
                        )
                    },
                    "kernel thread limit",
                )?;
                status(
                    unsafe {
                        sys::lib().cuFuncGetAttribute(
                            &mut shared,
                            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            function,
                        )
                    },
                    "kernel shared memory",
                )?;
                let block = if matches!(name, "raw_prepare" | "cursor_raw_prepare") {
                    geometry.dim.next_power_of_two().max(32)
                } else {
                    256
                };
                let dynamic = if name == "batch_inverse" {
                    3 * BLOCK_ROWS * BLOCK_ROWS * 4
                } else if matches!(name, "raw_prepare" | "cursor_raw_prepare") {
                    block * 4
                } else {
                    0
                };
                let max_shared = attribute(
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                )? as usize;
                if threads_limit < (block as i32)
                    || shared < 0
                    || (shared as usize)
                        .checked_add(dynamic)
                        .is_none_or(|x| x > max_shared)
                {
                    return Err(format!("fixed16 kernel {name} exceeds measured resources"));
                }
                functions.insert(name, function);
            }
            let alloc32 = |n| device.alloc_zeros::<f32>(n).map_err(|e| e.to_string());
            let raw_pending = (0..5)
                .map(|i| {
                    device
                        .alloc_zeros::<u16>(BLOCK_ROWS * if i == 4 { geometry.heads } else { hd })
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let committed_incoming = alloc32(st)?;
            let complete = alloc32(st)?;
            let latest = alloc32(st)?;
            let pending = device.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
            let error = device.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
            let absolute = device.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
            device.synchronize().map_err(|e| e.to_string())?;
            Ok(Self {
                device: device.clone(),
                module,
                functions,
                geometry,
                max_new_rows: plan.max_new_rows,
                workspace,
                committed_incoming,
                complete,
                latest,
                raw_pending,
                pending,
                error,
                absolute,
                allocation_bytes,
                persistent_bytes,
                arithmetic_sha256: format!(
                    "{:x}",
                    Sha256::digest(
                        format!("{MODULE_NAME}:{}:{}", plan.source_sha256, plan.ptx_sha256)
                            .as_bytes()
                    )
                ),
            })
        })();
        if result.is_err() {
            unsafe {
                let _ = sys::lib().cuModuleUnload(module);
            }
        }
        result
    }
    unsafe fn launch(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        stream: sys::CUstream,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        let function = *self.functions.get(name).ok_or("fixed16 missing function")?;
        status(
            sys::lib().cuLaunchKernel(
                function,
                grid.0,
                grid.1,
                grid.2,
                block,
                1,
                1,
                shared,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            name,
        )
    }
    fn grid(n: usize) -> Result<(u32, u32, u32), String> {
        Ok((
            u32::try_from(n.div_ceil(256)).map_err(|_| "fixed16 grid overflow")?,
            1,
            1,
        ))
    }
    pub unsafe fn reset(&self, stream: sys::CUstream) -> Result<(), String> {
        for carry in [&self.committed_incoming, &self.complete, &self.latest] {
            status(
                sys::lib().cuMemsetD32Async(ptr(carry), 0, carry.len(), stream),
                "carry reset",
            )?;
        }
        for raw in &self.raw_pending {
            status(
                sys::lib().cuMemsetD16Async(ptr(raw), 0x7fc1, raw.len(), stream),
                "raw pending reset",
            )?;
        }
        for scalar in [&self.pending, &self.error] {
            status(
                sys::lib().cuMemsetD32Async(ptr(scalar), 0, 1, stream),
                "scalar reset",
            )?;
        }
        status(
            sys::lib().cuMemsetD8Async(ptr(&self.absolute), 0, 8, stream),
            "absolute reset",
        )
    }
    pub fn register(
        &self,
        registry: &mut crate::session_cache::SequenceStateRegistry,
        layer: usize,
    ) -> Result<(), String> {
        use crate::session_cache::{SequenceStateAllocation, SequenceStateGrowth};
        let mut add =
            |name: &str, address: u64, element_size: usize, dtype: &str, shape: Vec<usize>| {
                let mut bytes = element_size;
                let mut strides = vec![0; shape.len()];
                for i in (0..shape.len()).rev() {
                    strides[i] = bytes;
                    bytes = bytes
                        .checked_mul(shape[i])
                        .ok_or("fixed16 inventory overflow")?;
                }
                registry.register(SequenceStateAllocation {
                    name: format!("layer{layer}.fixed16.{name}"),
                    kind: format!("fixed16_{name}"),
                    layer_idx: Some(layer),
                    device_ordinal: self.device.ordinal(),
                    ptr: address,
                    storage_bytes: bytes,
                    dtype: dtype.into(),
                    element_size,
                    shape,
                    strides_bytes: strides,
                    growth: SequenceStateGrowth::Fixed,
                })
            };
        for (name, value) in [
            ("committed", &self.committed_incoming),
            ("complete", &self.complete),
            ("latest", &self.latest),
        ] {
            add(
                name,
                ptr(value),
                4,
                "float32",
                vec![self.geometry.heads, self.geometry.dim, self.geometry.dim],
            )?;
        }
        for (i, raw) in self.raw_pending.iter().enumerate() {
            add(
                &format!("raw{i}"),
                ptr(raw),
                2,
                "bfloat16",
                if i == 4 {
                    vec![BLOCK_ROWS, self.geometry.heads]
                } else {
                    vec![BLOCK_ROWS, self.geometry.heads, self.geometry.dim]
                },
            )?;
        }
        add("pending", ptr(&self.pending), 4, "int32", vec![1])?;
        add("error", ptr(&self.error), 4, "int32", vec![1])?;
        add("absolute", ptr(&self.absolute), 8, "uint64", vec![1])
    }
}

impl PrefixRecurrence {
    /// Capture-time dispatch: caller owns stable live operands and output on
    /// this context. Runtime graph replay performs no Rust per-token work.
    pub unsafe fn decode(
        &self,
        live: [u64; 5],
        output: u64,
        displayed: u64,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if live.contains(&0) || output == 0 || displayed == 0 {
            return Err("fixed16 null live descriptor before dispatch".into());
        }
        let (h, d, b) = (
            self.geometry.heads as i32,
            self.geometry.dim as i32,
            BLOCK_ROWS as i32,
        );
        let hd = self.geometry.heads * self.geometry.dim;
        let st = hd * self.geometry.dim;
        let w = &self.workspace;
        let cursor = ptr(&self.pending);
        let error = ptr(&self.error);
        let absolute = ptr(&self.absolute);
        self.launch(
            "cursor_advance",
            (1, 1, 1),
            256,
            0,
            stream,
            &mut arguments(&mut [cursor, error, absolute], &mut [], &mut []),
        )?;
        let mut p = live.to_vec();
        p.extend(self.raw_pending.iter().map(ptr));
        p.extend([cursor, error]);
        self.launch(
            "cursor_append",
            Self::grid(BLOCK_ROWS * hd)?,
            256,
            0,
            stream,
            &mut arguments(&mut p, &mut [h, d], &mut []),
        )?;
        self.launch(
            "cursor_output_reset",
            Self::grid(BLOCK_ROWS * hd)?,
            256,
            0,
            stream,
            &mut arguments(&mut [ptr(&w.out)], &mut [0, hd as i32], &mut []),
        )?;
        let mut p = self.raw_pending.iter().map(ptr).collect::<Vec<_>>();
        p.extend([
            self.geometry.a_log,
            self.geometry.dt_bias,
            ptr(&w.q),
            ptr(&w.k),
            ptr(&w.v),
            ptr(&w.g),
            ptr(&w.beta),
            cursor,
        ]);
        let threads = self.geometry.dim.next_power_of_two().max(32) as u32;
        self.launch(
            "cursor_raw_prepare",
            (BLOCK_ROWS as u32, self.geometry.heads as u32, 1),
            threads,
            threads * 4,
            stream,
            &mut arguments(
                &mut p,
                &mut [h, d],
                &mut [self.geometry.gate_lower_bound, 1., 1e-6],
            ),
        )?;
        let mut p = [ptr(&w.g), ptr(&w.gs)];
        let mut dimensions = [h, d, b];
        let mut args = arguments(&mut p, &mut dimensions, &mut []);
        let mut c = cursor;
        args.push((&mut c as *mut u64).cast());
        self.launch("cursor_batch_g", Self::grid(hd)?, 256, 0, stream, &mut args)?;
        let mut p = [
            ptr(&w.gs),
            ptr(&w.k),
            ptr(&w.q),
            ptr(&w.kg),
            ptr(&w.qg),
            ptr(&w.pair),
        ];
        let mut dimensions = [h, d, b];
        let mut args = arguments(&mut p, &mut dimensions, &mut []);
        args.push((&mut c as *mut u64).cast());
        self.launch(
            "cursor_batch_precompute_compact",
            Self::grid(hd)?,
            256,
            0,
            stream,
            &mut args,
        )?;
        let mut p = [
            ptr(&w.q),
            ptr(&w.k),
            ptr(&w.pair),
            ptr(&w.beta),
            ptr(&w.gs),
            ptr(&w.a),
            ptr(&w.qk),
        ];
        let mut dimensions = [h, d, b];
        let mut args = arguments(&mut p, &mut dimensions, &mut []);
        args.push((&mut c as *mut u64).cast());
        self.launch(
            "cursor_batch_coeff_compact",
            (self.geometry.heads as u32, 1, 1),
            256,
            0,
            stream,
            &mut args,
        )?;
        self.launch(
            "batch_inverse",
            (self.geometry.heads as u32, 1, 1),
            256,
            (3 * BLOCK_ROWS * BLOCK_ROWS * 4) as u32,
            stream,
            &mut arguments(&mut [ptr(&w.a), ptr(&w.inverse)], &mut [b], &mut []),
        )?;
        let mut p = [
            ptr(&w.v),
            ptr(&w.beta),
            ptr(&w.gs),
            ptr(&w.kg),
            ptr(&w.qg),
            ptr(&w.pair),
            ptr(&w.k),
            ptr(&w.qk),
            ptr(&w.inverse),
            ptr(&self.committed_incoming),
            ptr(&w.out),
            ptr(&self.latest),
            ptr(&self.latest),
            ptr(&self.complete),
        ];
        let mut dimensions = [h, d, b];
        let mut tails = [0i32, 0i32];
        let mut scale = (self.geometry.dim as f32).sqrt().recip();
        let mut args = arguments(&mut p, &mut dimensions, &mut []);
        args.push((&mut c as *mut u64).cast());
        args.extend(tails.iter_mut().map(|x| (x as *mut i32).cast()));
        args.push((&mut scale as *mut f32).cast());
        let mut carry_error = error;
        args.push((&mut carry_error as *mut u64).cast());
        self.launch(
            "cursor_batch_carry_compact",
            (
                self.geometry.heads as u32,
                self.geometry.dim.div_ceil(16) as u32,
                1,
            ),
            256,
            0,
            stream,
            &mut args,
        )?;
        self.launch(
            "cursor_emit_commit",
            Self::grid(st)?,
            256,
            0,
            stream,
            &mut arguments(
                &mut [
                    ptr(&w.out),
                    ptr(&self.complete),
                    ptr(&self.committed_incoming),
                    output,
                    cursor,
                    error,
                ],
                &mut [h, d, 0],
                &mut [],
            ),
        )?;
        let mut p = self.raw_pending.iter().map(ptr).collect::<Vec<_>>();
        p.extend([cursor, error]);
        self.launch(
            "cursor_clear_completed_raw",
            Self::grid(BLOCK_ROWS * hd)?,
            256,
            0,
            stream,
            &mut arguments(&mut p, &mut [h, d], &mut []),
        )?;
        self.launch(
            "cursor_finish",
            (1, 1, 1),
            256,
            0,
            stream,
            &mut arguments(&mut [cursor, error, absolute], &mut [], &mut []),
        )?;
        self.launch(
            "prefix_copy_display",
            Self::grid(st)?,
            256,
            0,
            stream,
            &mut arguments(
                &mut [ptr(&self.latest), displayed, error],
                &mut [st as i32],
                &mut [],
            ),
        )
    }
    /// Batch prefill uses the actual consumed absolute start supplied by the
    /// Rust request state. The GPU prepend additionally checks owned metadata.
    pub unsafe fn prefill(
        &self,
        live: [u64; 5],
        output: u64,
        displayed: u64,
        start: u64,
        rows: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        self.prefill_in_workspace(
            &self.workspace,
            live,
            output,
            displayed,
            start,
            rows,
            stream,
        )
    }
    /// Batch arena ownership is explicit. The caller must keep this arena and
    /// all operands alive and exclusively ordered for the entire queued kernel
    /// interval, including graph replay. A host borrow alone is insufficient.
    pub unsafe fn prefill_in_workspace(
        &self,
        workspace: &PrefixWorkspace,
        live: [u64; 5],
        output: u64,
        displayed: u64,
        start: u64,
        rows: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        workspace.validate(&self.device, self.geometry, rows)?;
        let mut stream_context = std::ptr::null_mut();
        status(
            sys::lib().cuStreamGetCtx(stream, &mut stream_context),
            "batch arena stream context",
        )?;
        if stream_context != *self.device.cu_primary_ctx() {
            return Err("fixed16 batch arena stream belongs to another context".into());
        }
        if live.contains(&0)
            || output == 0
            || displayed == 0
            || rows == 0
            || rows > self.max_new_rows
        {
            return Err("fixed16 prefill descriptor/capacity admission failed".into());
        }
        start
            .checked_add(rows as u64)
            .ok_or("fixed16 absolute end overflow")?;
        let partial = (start % BLOCK_ROWS as u64) as usize;
        let count = partial + rows;
        let (h, d, b, t) = (
            self.geometry.heads as i32,
            self.geometry.dim as i32,
            BLOCK_ROWS as i32,
            count as i32,
        );
        let hd = self.geometry.heads * self.geometry.dim;
        let st = hd * self.geometry.dim;
        let groups = count.div_ceil(BLOCK_ROWS) as u32;
        let w = workspace;
        let cursor = ptr(&self.pending);
        let error = ptr(&self.error);
        let absolute = ptr(&self.absolute);
        self.launch(
            "prefix_validate",
            (1, 1, 1),
            1,
            0,
            stream,
            &mut arguments(
                &mut [cursor, error, absolute, start],
                &mut [rows as i32],
                &mut [],
            ),
        )?;
        let mut p = live.to_vec();
        p.extend(self.raw_pending.iter().map(ptr));
        p.extend(w.raw.iter().map(ptr));
        p.extend([cursor, error, absolute, start]);
        self.launch(
            "prefix_prepend",
            Self::grid(count * hd)?,
            256,
            0,
            stream,
            &mut arguments(&mut p, &mut [rows as i32, h, d], &mut []),
        )?;
        let mut p = w.raw.iter().map(ptr).collect::<Vec<_>>();
        p.extend([
            self.geometry.a_log,
            self.geometry.dt_bias,
            ptr(&w.q),
            ptr(&w.k),
            ptr(&w.v),
            ptr(&w.g),
            ptr(&w.beta),
        ]);
        let threads = self.geometry.dim.next_power_of_two().max(32) as u32;
        self.launch(
            "raw_prepare",
            (count as u32, self.geometry.heads as u32, 1),
            threads,
            threads * 4,
            stream,
            &mut arguments(
                &mut p,
                &mut [t, h, d],
                &mut [self.geometry.gate_lower_bound, 1., 1e-6],
            ),
        )?;
        self.launch(
            "batch_g",
            (
                u32::try_from(hd.div_ceil(256)).map_err(|_| "fixed16 HD grid")?,
                groups,
                1,
            ),
            256,
            0,
            stream,
            &mut arguments(&mut [ptr(&w.g), ptr(&w.gs)], &mut [h, d, b, t], &mut []),
        )?;
        self.launch(
            "batch_precompute_compact",
            (
                u32::try_from(hd.div_ceil(256)).map_err(|_| "fixed16 HD grid")?,
                groups,
                1,
            ),
            256,
            0,
            stream,
            &mut arguments(
                &mut [
                    ptr(&w.gs),
                    ptr(&w.k),
                    ptr(&w.q),
                    ptr(&w.kg),
                    ptr(&w.qg),
                    ptr(&w.pair),
                ],
                &mut [h, d, b, t],
                &mut [],
            ),
        )?;
        self.launch(
            "batch_coeff_compact",
            (self.geometry.heads as u32, groups, 1),
            256,
            0,
            stream,
            &mut arguments(
                &mut [
                    ptr(&w.q),
                    ptr(&w.k),
                    ptr(&w.pair),
                    ptr(&w.beta),
                    ptr(&w.gs),
                    ptr(&w.a),
                    ptr(&w.qk),
                ],
                &mut [h, d, b, t],
                &mut [],
            ),
        )?;
        self.launch(
            "batch_inverse",
            (self.geometry.heads as u32, groups, 1),
            256,
            (3 * BLOCK_ROWS * BLOCK_ROWS * 4) as u32,
            stream,
            &mut arguments(&mut [ptr(&w.a), ptr(&w.inverse)], &mut [b], &mut []),
        )?;
        let mut carry_p = [
            ptr(&w.v),
            ptr(&w.beta),
            ptr(&w.gs),
            ptr(&w.kg),
            ptr(&w.qg),
            ptr(&w.pair),
            ptr(&w.k),
            ptr(&w.qk),
            ptr(&w.inverse),
            ptr(&self.committed_incoming),
            ptr(&w.out),
            ptr(&self.latest),
            ptr(&self.latest),
            ptr(&self.complete),
        ];
        let mut carry_d = [h, d, b, t, 0, 0];
        let mut carry_scale = [(self.geometry.dim as f32).sqrt().recip()];
        let mut carry_error = error;
        let mut carry_args = arguments(&mut carry_p, &mut carry_d, &mut carry_scale);
        carry_args.push((&mut carry_error as *mut u64).cast());
        self.launch(
            "batch_carry_compact",
            (
                self.geometry.heads as u32,
                self.geometry.dim.div_ceil(16) as u32,
                1,
            ),
            256,
            0,
            stream,
            &mut carry_args,
        )?;
        let mut p = w.raw.iter().map(ptr).collect::<Vec<_>>();
        p.extend(self.raw_pending.iter().map(ptr));
        p.extend([
            ptr(&w.out),
            output,
            ptr(&self.latest),
            displayed,
            ptr(&self.complete),
            ptr(&self.committed_incoming),
            cursor,
            error,
            absolute,
        ]);
        self.launch(
            "prefix_finalize",
            Self::grid((rows * hd).max(st).max(BLOCK_ROWS * hd))?,
            256,
            0,
            stream,
            &mut arguments(&mut p, &mut [partial as i32, rows as i32, h, d], &mut []),
        )
    }
}

pub fn configured_plan() -> Result<Option<Plan>, String> {
    let file = match std::env::var("KRASIS_PREFIX_REPAIR_PLAN") {
        Ok(x) => x,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    crate::matched_probe::require_model_diagnostic_probe()?;
    let expected = std::env::var("KRASIS_PREFIX_REPAIR_PLAN_SHA256")
        .map_err(|_| "fixed16 diagnostic requires frozen plan SHA256")?;
    let bytes = std::fs::read(&file).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err("fixed16 selected-layer plan identity mismatch".into());
    }
    let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if crate::gpu_prefill::glm53_kda_recurrent_oracle_enabled()? {
        return Err(
            "fixed16 selected contract conflicts with recurrent oracle before owner load".into(),
        );
    }
    Ok(Some(plan))
}
impl PrefixRecurrence {
    pub(crate) fn checkpoint_context(&self) -> sys::CUcontext {
        *self.device.cu_primary_ctx()
    }
    pub fn validate_setup(&self, geometry: Geometry, plan: &Plan) -> Result<(), String> {
        let digest = format!(
            "{:x}",
            Sha256::digest(
                format!("{MODULE_NAME}:{}:{}", plan.source_sha256, plan.ptx_sha256).as_bytes()
            )
        );
        if geometry != self.geometry
            || plan.contract != MODULE_NAME
            || plan.max_new_rows != self.max_new_rows
            || digest != self.arithmetic_sha256
        {
            return Err("fixed16 registered owner geometry/arithmetic/capacity changed; create a fresh diagnostic store".into());
        }
        Ok(())
    }
    pub fn required_pointers(&self) -> Vec<u64> {
        let mut values = vec![
            ptr(&self.committed_incoming),
            ptr(&self.complete),
            ptr(&self.latest),
            ptr(&self.pending),
            ptr(&self.error),
            ptr(&self.absolute),
        ];
        values.extend(self.raw_pending.iter().map(ptr));
        values
    }
    /// Validate persisted logical metadata before the generic restore writes
    /// any device allocation. All required bytes still undergo registry checks.
    pub fn validate_restore(
        &self,
        layer: usize,
        logical_tokens: usize,
        blobs: &[&crate::session_cache::SequenceStateBlob],
    ) -> Result<(), String> {
        let bytes = |suffix: &str| -> Result<&[u8], String> {
            let name = format!("layer{layer}.fixed16.{suffix}");
            let mut found = blobs.iter().filter(|x| x.allocation_name == name);
            let first = found
                .next()
                .ok_or_else(|| format!("missing fixed16 snapshot {name}"))?;
            if found.next().is_some() {
                return Err(format!("duplicate fixed16 snapshot {name}"));
            }
            Ok(&first.bytes)
        };
        let pending = i32::from_le_bytes(
            bytes("pending")?
                .try_into()
                .map_err(|_| "fixed16 pending extent")?,
        );
        let absolute = u64::from_le_bytes(
            bytes("absolute")?
                .try_into()
                .map_err(|_| "fixed16 absolute extent")?,
        );
        let error = i32::from_le_bytes(
            bytes("error")?
                .try_into()
                .map_err(|_| "fixed16 error extent")?,
        );
        if absolute != u64::try_from(logical_tokens).map_err(|_| "fixed16 token count overflow")?
            || pending != (logical_tokens % BLOCK_ROWS) as i32
            || error != 0
        {
            return Err(
                "fixed16 restored consumed position/pending/error mismatch before writes".into(),
            );
        }
        Ok(())
    }
}

impl PrefixRecurrence {
    /// Used only by cold diagnostic capture; never by model graph replay.
    pub fn batch_capacity(&self) -> usize {
        self.max_new_rows
    }
    pub fn diagnostic_output_ptr(&self) -> u64 {
        ptr(&self.workspace.out)
    }
    pub fn diagnostic_metadata(&self) -> (u64, u64, u64) {
        (ptr(&self.pending), ptr(&self.absolute), ptr(&self.error))
    }
}
