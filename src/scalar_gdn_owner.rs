//! Scalar native64 owner implementation under qualification. Not wired to serving.
use crate::scalar_gdn_phase::{Event, Stream};
use crate::shared_gdn_workspace::{Lease, Program, Workspace};
use crate::{
    scalar_gdn_layout::{Layout, NATIVE_BLOCK_ROWS as BT},
    scalar_gdn_native::Native,
};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::{collections::BTreeMap, ffi::c_void, sync::Arc};
pub(crate) const FUNCTIONS: &[&str] = &[
    "scalar_gdn_begin",
    "scalar_gdn_stage_bf16",
    "scalar_gdn_stage_gate",
    "scalar_gdn_cumsum",
    "scalar_gdn_retain_bf16",
    "scalar_gdn_retain_gate",
    "scalar_gdn_emit",
    "scalar_gdn_commit_carry",
    "scalar_gdn_commit",
    "scalar_gdn_collect_health",
];
enum Arg {
    P(u64),
    I(i32),
}
impl Arg {
    fn address(&mut self) -> *mut c_void {
        match self {
            Self::P(x) => (x as *mut u64).cast(),
            Self::I(x) => (x as *mut i32).cast(),
        }
    }
}
fn check(rc: sys::CUresult, label: &str) -> Result<(), String> {
    rc.result().map_err(|e| format!("{label}: {e}"))
}
fn native(rc: u32, label: &str) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("scalar FLA {label} returned {rc}"))
    }
}
pub(crate) fn drain_or_fatal(device: &Arc<CudaDevice>, label: &str) {
    if let Err(e) = device.bind_to_thread().and_then(|_| device.synchronize()) {
        eprintln!(
            "KRASIS FATAL: {label} completion unknown: {e}; owners cannot be safely released"
        );
        std::process::exit(crate::cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
    }
}
pub struct Plane<'a, T> {
    owner: &'a dyn crate::cuda_region::CudaRegion,
    ptr: u64,
    elements: usize,
    _dtype: std::marker::PhantomData<T>,
}
impl<'a, T: Copy + Send + Sync> Plane<'a, T> {
    pub(crate) fn pointer(&self) -> u64 {
        self.ptr
    }
    pub(crate) fn require_context_extent(
        &self,
        context: sys::CUcontext,
        elements: usize,
    ) -> Result<(), String> {
        if self.owner.context() != context || self.elements != elements {
            return Err("scalar actual output context/extent differs".into());
        }
        Ok(())
    }
    pub fn checked(
        owner: &'a CudaSlice<T>,
        offset: usize,
        elements: usize,
    ) -> Result<Self, String> {
        // The concrete allocation element type proves this view's dtype.
        unsafe { Self::from_region(owner, offset, elements) }
    }
    /// # Safety
    /// Caller proves the registered real allocation's element type matches T;
    /// the sealed owner itself supplies allocation extent and CUDA context.
    pub unsafe fn from_region(
        owner: &'a dyn crate::cuda_region::CudaRegion,
        offset: usize,
        elements: usize,
    ) -> Result<Self, String> {
        if elements == 0 {
            return Err("empty scalar GDN input view".into());
        }
        let (start, end) = owner.extent()?;
        let offset = offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("input view offset overflow")?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("input view bytes overflow")?;
        let ptr = start
            .checked_add(offset as u64)
            .ok_or("input view address overflow")?;
        let stop = ptr
            .checked_add(bytes as u64)
            .ok_or("input view end overflow")?;
        if start == 0 || ptr < start || stop > end {
            return Err("scalar GDN input view exceeds retained real allocation".into());
        }
        Ok(Self {
            owner,
            ptr,
            elements,
            _dtype: std::marker::PhantomData,
        })
    }
}
pub struct Inputs<'a> {
    pub q: Plane<'a, u16>,
    pub k: Plane<'a, u16>,
    pub v: Plane<'a, u16>,
    pub beta: Plane<'a, u16>,
    pub gate: Plane<'a, f32>,
}
/// Device views retain their actual allocation borrow and remain dynamic in a
/// graph. Host positions are explicit eager/batched inputs and cannot be captured.
pub(crate) enum DecodePosition<'a> {
    Device(Plane<'a, i32>),
    Host(u64),
}

/// Typed latest-state resource, backed by its real allocation owner.
pub(crate) struct LatestState {
    owner: Arc<dyn crate::cuda_region::CudaRegion>,
    pointer: u64,
    bytes: usize,
}
impl LatestState {
    pub(crate) fn cuda(owner: Arc<CudaSlice<f32>>) -> Result<Self, String> {
        use crate::cuda_region::CudaRegion;
        let (start, end) = owner.extent()?;
        Ok(Self {
            owner,
            pointer: start,
            bytes: usize::try_from(end - start).map_err(|e| e.to_string())?,
        })
    }
    pub(crate) fn model(
        owner: Arc<crate::model_tensor_owner::ModelTensorOwner>,
    ) -> Result<Self, String> {
        use crate::cuda_region::CudaRegion;
        if owner.element != crate::stable_mla_gemm::Element::F32 {
            return Err("actual scalar latest tensor must be FP32".into());
        }
        // One-time cold readiness. No Python or per-token/per-layer runtime wait.
        owner.complete_setup_source()?;
        let (start, end) = owner.extent()?;
        Ok(Self {
            owner,
            pointer: start,
            bytes: usize::try_from(end - start).map_err(|e| e.to_string())?,
        })
    }
}

pub struct Owner {
    pub(crate) layout: Layout,
    decode_layout: Layout,
    pending: BTreeMap<&'static str, CudaSlice<u8>>,
    latest: LatestState,
    poisoned: bool,
    fla: Native,
    // Cold-admitted actual stream owners remain alive through every state use.
    streams: Vec<Stream>,
    readiness: Vec<Event>,
    started: bool,
    // Last: the actual program/context outlives all state/native bindings.
    program: Arc<Program>,
}
impl Owner {
    pub(crate) fn has_started(&self) -> bool {
        self.started
    }
    pub fn new(program: Arc<Program>, layout: Layout, latest: LatestState) -> Result<Self, String> {
        let device = &program.device;
        device.bind_to_thread().map_err(|e| e.to_string())?;
        if latest.owner.context() != *device.cu_primary_ctx()
            || latest.bytes != layout.persistent["boundary-state.f32"]
            || latest.pointer == 0
            || latest.pointer % std::mem::align_of::<f32>() as u64 != 0
        {
            return Err("scalar latest real owner/context/extent differs".into());
        }
        let fla = program.binding(&layout)?;
        let decode_layout = Layout::checked(layout.heads, layout.key_dim, layout.value_dim, 1)?;
        let mut pending = BTreeMap::new();
        let result = (|| -> Result<(), String> {
            for (&name, &bytes) in &layout.persistent {
                pending.insert(
                    name,
                    device.alloc_zeros::<u8>(bytes).map_err(|e| e.to_string())?,
                );
            }
            check(
                unsafe {
                    sys::lib().cuMemcpyDtoDAsync_v2(
                        *pending["boundary-state.f32"].device_ptr(),
                        latest.pointer,
                        layout.persistent["boundary-state.f32"],
                        *device.cu_stream(),
                    )
                },
                "scalar initial actual state copy",
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            drop(pending);
            drain_or_fatal(device, "scalar persistent state construction cleanup");
            return Err(error);
        }
        Ok(Self {
            layout,
            decode_layout,
            pending,
            latest,
            poisoned: false,
            fla,
            streams: vec![Stream::Decode(program.device.clone())],
            readiness: Vec::new(),
            started: false,
            program,
        })
    }
    fn p(&self, name: &str) -> u64 {
        *self.pending[name].device_ptr()
    }
    pub(crate) fn admit_prefill(
        &mut self,
        owner: Arc<crate::gpu_prefill::PrefillComputeStream>,
    ) -> Result<(), String> {
        if self.started {
            return Err("scalar state phase must be admitted before execution/calibration".into());
        }
        let stream = Stream::Prefill(owner);
        stream.admit(&self.program.device)?;
        if self.streams.iter().any(|s| s.same(&stream)) {
            return Err("scalar state phase already admitted".into());
        }
        // Pending planes and the initial true carry were initialized on the
        // retained decode allocation stream. Publish readiness to actual prefill.
        let event = Event::new(self.program.device.clone())?;
        if let Err(e) = event
            .record(&Stream::Decode(self.program.device.clone()))
            .and_then(|_| event.wait(&stream))
        {
            stream.drain_or_fatal("scalar failed state readiness use-stream");
            drain_or_fatal(
                &self.program.device,
                "scalar failed state readiness allocation stream",
            );
            return Err(e);
        }
        self.readiness.push(event);
        self.streams.push(stream);
        Ok(())
    }
    fn validate_lease(&self, lease: &Lease<'_>) -> Result<(), String> {
        if !self.streams.iter().any(|s| s.same(&lease.workspace.phase)) {
            return Err("scalar state actual phase owner not cold-admitted".into());
        }
        if !Arc::ptr_eq(&self.program, &lease.workspace.program) {
            return Err(
                "scalar state and workspace do not share the admitted real program/context".into(),
            );
        }
        Ok(())
    }
    fn validate_inputs(&self, x: &Inputs<'_>, rows: usize) -> Result<(), String> {
        let h = self.layout.heads;
        for (context, elements, expected) in [
            (
                x.q.owner.context(),
                x.q.elements,
                rows * h * self.layout.key_dim,
            ),
            (
                x.k.owner.context(),
                x.k.elements,
                rows * h * self.layout.key_dim,
            ),
            (
                x.v.owner.context(),
                x.v.elements,
                rows * h * self.layout.value_dim,
            ),
            (x.beta.owner.context(), x.beta.elements, rows * h),
            (x.gate.owner.context(), x.gate.elements, rows * h),
        ] {
            if context != *self.program.device.cu_primary_ctx() || elements != expected {
                return Err("scalar GDN actual input owner/context/view differs".into());
            }
        }
        Ok(())
    }
    /// # Safety
    /// Retain the actual input/phase owners through queued completion, including
    /// errors. The external lease serializes a whole phase on its owned stream.
    pub unsafe fn enqueue_prefill(
        &mut self,
        lease: &Lease<'_>,
        start: u64,
        rows: usize,
        x: &Inputs<'_>,
    ) -> Result<(), String> {
        if self.poisoned {
            return Err("scalar owner is poisoned; verified restore/reset required".into());
        }
        self.validate_lease(lease)?;
        let shape = Layout::checked(
            self.layout.heads,
            self.layout.key_dim,
            self.layout.value_dim,
            rows,
        )?;
        let (partial, combined, padded) = shape.prefill_shape(start, rows)?;
        lease.workspace.supports(&shape)?;
        self.validate_inputs(x, rows)?;
        self.program
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        self.started = true;
        let result = Operation {
            state: self,
            workspace: lease.workspace,
        }
        .enqueue(start, rows, partial, combined, padded, x, None, false);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    /// Same lifetime/serialization contract; the model's real device position
    /// is consumed dynamically, including captured CUDA graph replay.
    pub unsafe fn enqueue_decode(
        &mut self,
        lease: &Lease<'_>,
        x: &Inputs<'_>,
        position: &CudaSlice<i32>,
    ) -> Result<(), String> {
        if position.len() != 1 {
            return Err("scalar real position owner/context/extent differs".into());
        }
        self.enqueue_decode_position(
            lease,
            x,
            &DecodePosition::Device(Plane::checked(position, 0, 1)?),
        )
    }
    /// Same actual one-row recurrence and dynamic pending/carry semantics for
    /// device graph positions, bounded batch-position views and explicit eager
    /// positions. Host capture is rejected before enqueue, never frozen silently.
    /// # Safety
    /// Retain actual inputs, position owner and whole-phase owners through all
    /// queued work, errors and unwinds, as required by enqueue_decode.
    pub(crate) unsafe fn enqueue_decode_position(
        &mut self,
        lease: &Lease<'_>,
        x: &Inputs<'_>,
        position: &DecodePosition<'_>,
    ) -> Result<(), String> {
        if self.poisoned {
            return Err("scalar owner is poisoned; verified restore/reset required".into());
        }
        self.validate_lease(lease)?;
        lease.workspace.supports(&self.decode_layout)?;
        self.validate_inputs(x, 1)?;
        self.program
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let (start, pointer, host_decode) = match position {
            DecodePosition::Device(view) => {
                view.require_context_extent(*self.program.device.cu_primary_ctx(), 1)?;
                (0, Some(view.pointer()), false)
            }
            DecodePosition::Host(start) => {
                let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                check(
                    sys::lib().cuStreamIsCapturing(lease.workspace.stream(), &mut status),
                    "scalar host-position capture admission",
                )?;
                if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    return Err("scalar host decode position cannot be captured; use the actual dynamic device position view".into());
                }
                (*start, None, true)
            }
        };
        self.started = true;
        let result = Operation {
            state: self,
            workspace: lease.workspace,
        }
        .enqueue(start, 1, 0, 0, BT, x, pointer, host_decode);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub(crate) fn latest_region(&self) -> &dyn crate::cuda_region::CudaRegion {
        self.latest.owner.as_ref()
    }
    pub(crate) fn latest_pointer(&self) -> u64 {
        self.latest.pointer
    }
    pub fn persistent(&self) -> &BTreeMap<&'static str, CudaSlice<u8>> {
        &self.pending
    }
    /// Descriptors come from the real allocations, not estimates. The latest
    /// state is already registered by model loading and must not be duplicated.
    pub(crate) fn registry(
        &self,
        layer: usize,
    ) -> Result<crate::session_cache::SequenceStateRegistry, String> {
        use crate::session_cache::{
            SequenceStateAllocation, SequenceStateGrowth, SequenceStateRegistry,
        };
        let mut registry = SequenceStateRegistry::default();
        let h = self.layout.heads;
        let k = self.layout.key_dim;
        let v = self.layout.value_dim;
        for (&role, allocation) in &self.pending {
            let (dtype, element_size, shape) = match role {
                "q.bf16" | "k.bf16" => ("bfloat16", 2, vec![BT, h, k]),
                "v.bf16" => ("bfloat16", 2, vec![BT, h, v]),
                "beta.bf16" => ("bfloat16", 2, vec![BT, h]),
                "gate.f32" => ("float32", 4, vec![BT, h]),
                "boundary-state.f32" => ("float32", 4, vec![h, k, v]),
                "metadata.u64" => ("uint64", 8, vec![crate::scalar_gdn_layout::METADATA_WORDS]),
                _ => return Err(format!("unregistered scalar persistent role:{role}")),
            };
            let mut bytes: usize = element_size;
            let mut strides = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                strides[i] = bytes;
                bytes = bytes
                    .checked_mul(shape[i])
                    .ok_or("scalar state descriptor byte overflow")?;
            }
            if bytes != allocation.len() || Some(&bytes) != self.layout.persistent.get(role) {
                return Err(format!("scalar actual state extent differs for {role}"));
            }
            registry.register(SequenceStateAllocation {
                name: format!(
                    "gpu{}.layer{layer}.linear.native64.{role}",
                    self.program.device.ordinal()
                ),
                kind: format!("scalar_gdn_native64_{role}"),
                layer_idx: Some(layer),
                device_ordinal: self.program.device.ordinal(),
                ptr: *allocation.device_ptr(),
                storage_bytes: bytes,
                dtype: dtype.into(),
                element_size,
                shape,
                strides_bytes: strides,
                growth: SequenceStateGrowth::Fixed,
            })?;
        }
        if registry.allocations().len() != self.layout.persistent.len() {
            return Err("scalar state inventory incomplete".into());
        }
        Ok(registry)
    }
    /// # Safety
    /// Health allocation and actual state remain owned through this phase.
    /// Called after commit on the same real stream, before any publication.
    pub(crate) unsafe fn collect_health(
        &self,
        lease: &Lease<'_>,
        health: &CudaSlice<u64>,
    ) -> Result<(), String> {
        self.validate_lease(lease)?;
        if health.len() != 1
            || *health.device().cu_primary_ctx() != *self.program.device.cu_primary_ctx()
        {
            return Err("scalar actual semantic-health allocation differs".into());
        }
        Operation {
            state: self,
            workspace: lease.workspace,
        }
        .kernel(
            "scalar_gdn_collect_health",
            (1, 1, 1),
            1,
            &mut [Arg::P(self.p("metadata.u64")), Arg::P(*health.device_ptr())],
        )
    }
    /// # Safety
    /// Hold the complete quiescent execution/recovery phase and all real state
    /// owners. Drain all transfer/use streams before entry. No graph/consumer
    /// may access this sequence until the entire family reset succeeds.
    pub(crate) unsafe fn reset_completed(&mut self) -> Result<(), String> {
        self.poisoned = true;
        self.program
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let stream = *self.program.device.cu_stream();
        let result = (|| {
            check(
                sys::lib().cuMemsetD8Async(self.latest.pointer, 0, self.latest.bytes, stream),
                "scalar reset actual latest",
            )?;
            for (&role, allocation) in &self.pending {
                if role != "metadata.u64" {
                    check(
                        sys::lib().cuMemsetD8Async(
                            *allocation.device_ptr(),
                            0,
                            allocation.len(),
                            stream,
                        ),
                        "scalar reset persistent region",
                    )?;
                }
            }
            let metadata = &self.pending["metadata.u64"];
            check(
                sys::lib().cuMemsetD8Async(*metadata.device_ptr(), 0, metadata.len(), stream),
                "scalar reset metadata",
            )
        })();
        drain_or_fatal(&self.program.device, "scalar reset queued completion");
        result?;
        self.poisoned = false;
        Ok(())
    }
    /// # Safety
    /// The complete family restore is already verified for all registered
    /// regions and real source identity, and all transfer streams completed.
    /// Hold the same recovery phase until every layer has validated. A failure
    /// leaves the enclosing execution poisoned, never a partially valid family.
    pub(crate) unsafe fn validate_restored_completed(
        &mut self,
        cursor: usize,
    ) -> Result<(), String> {
        self.poisoned = true;
        self.validate_completed_metadata(cursor)?;
        self.poisoned = false;
        Ok(())
    }
    /// Caller holds a completed whole-family quiescent phase. Reading these
    /// actual GPU words is a snapshot-boundary check, never a per-token wait.
    pub(crate) unsafe fn require_snapshot_completed(&self, cursor: usize) -> Result<(), String> {
        if self.poisoned {
            return Err("scalar snapshot rejects host-poisoned owner".into());
        }
        self.validate_completed_metadata(cursor)
    }
    fn validate_completed_metadata(&self, cursor: usize) -> Result<(), String> {
        self.program
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let bytes = self
            .program
            .device
            .dtoh_sync_copy(&self.pending["metadata.u64"])
            .map_err(|e| e.to_string())?;
        let metadata = bytes
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        if metadata.len() != crate::scalar_gdn_layout::METADATA_WORDS
            || metadata[0] != u64::try_from(cursor).map_err(|e| e.to_string())?
            || metadata[1] != (cursor % BT) as u64
            || metadata[3] != 0
            || metadata[4] != 0
            || metadata[7] != 0
        {
            return Err(format!("restored scalar state cursor/pending/poison/inflight/reserved invalid:{metadata:?}"));
        }
        Ok(())
    }
    pub fn host_poisoned(&self) -> bool {
        self.poisoned
    }
}
struct Operation<'a> {
    state: &'a Owner,
    workspace: &'a Workspace,
}
impl Operation<'_> {
    fn p(&self, name: &str) -> u64 {
        self.state.p(name)
    }
    fn a(&self, name: &str) -> u64 {
        self.workspace.ptr(name)
    }
    fn stream(&self) -> sys::CUstream {
        self.workspace.stream()
    }
    fn kernel<const N: usize>(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        threads: u32,
        args: &mut [Arg; N],
    ) -> Result<(), String> {
        if [grid.0, grid.1, grid.2]
            .iter()
            .enumerate()
            .any(|(i, &x)| x == 0 || x as usize > self.state.program.limits[i])
        {
            return Err("scalar GDN launch exceeds actual device grid limit".into());
        }
        let mut addresses = std::array::from_fn::<_, N, _>(|index| args[index].address());
        check(
            unsafe {
                sys::lib().cuLaunchKernel(
                    self.state.program.functions[name].cu_function(),
                    grid.0,
                    grid.1,
                    grid.2,
                    threads,
                    1,
                    1,
                    0,
                    self.stream(),
                    addresses.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            name,
        )
    }
    fn linear<const N: usize>(
        &self,
        name: &str,
        cells: usize,
        args: &mut [Arg; N],
    ) -> Result<(), String> {
        let grid = u32::try_from(cells.div_ceil(256)).map_err(|_| "scalar GDN grid overflow")?;
        self.kernel(name, (grid, 1, 1), 256, args)
    }
    unsafe fn enqueue(
        &self,
        start: u64,
        rows: usize,
        partial: usize,
        combined: usize,
        padded: usize,
        x: &Inputs<'_>,
        position: Option<u64>,
        host_decode: bool,
    ) -> Result<(), String> {
        use Arg::{I, P};
        let h = self.state.layout.heads;
        let k = self.state.layout.key_dim;
        let v = self.state.layout.value_dim;
        let meta = self.p("metadata.u64");
        let decode = position.is_some() || host_decode;
        self.kernel(
            "scalar_gdn_begin",
            (1, 1, 1),
            1,
            &mut [
                P(meta),
                P(start),
                P(position.unwrap_or(0)),
                I(rows as i32),
                I(padded as i32),
                I(if host_decode { 2 } else { decode as i32 }),
            ],
        )?;
        for (name, ptr, width) in [
            ("q.bf16", x.q.ptr, h * k),
            ("k.bf16", x.k.ptr, h * k),
            ("v.bf16", x.v.ptr, h * v),
            ("beta.bf16", x.beta.ptr, h),
        ] {
            self.linear(
                "scalar_gdn_stage_bf16",
                padded * width,
                &mut [
                    P(self.a(name)),
                    P(self.p(name)),
                    P(ptr),
                    P(meta),
                    I(width as i32),
                    I(padded as i32),
                ],
            )?;
        }
        self.linear(
            "scalar_gdn_stage_gate",
            padded * h,
            &mut [
                P(self.a("gate.f32")),
                P(self.p("gate.f32")),
                P(x.gate.ptr),
                P(meta),
                I(h as i32),
                I(padded as i32),
            ],
        )?;
        self.kernel(
            "scalar_gdn_cumsum",
            (h as u32, (padded / BT) as u32, 1),
            1,
            &mut [
                P(self.a("cumulative-gate.f32")),
                P(self.a("gate.f32")),
                P(meta),
                I(h as i32),
                I(padded as i32),
            ],
        )?;
        for (name, bytes) in [
            ("A.f32", padded * h * BT * 4),
            ("Ai.bf16", padded * h * BT * 2),
            ("chunk-state.bf16", padded / BT * h * k * v * 2),
        ] {
            check(
                sys::lib().cuMemsetD8Async(self.a(name), 0, bytes, self.stream()),
                "scalar GDN clear actual native workspace span",
            )?;
        }
        let s = self.stream() as *mut c_void;
        let nt = (padded / BT) as u32;
        let t = padded as i32;
        native(
            (self.state.fla.kkt)(
                self.state.fla.handle(),
                self.a("k.bf16"),
                self.a("cumulative-gate.f32"),
                self.a("beta.bf16"),
                self.a("A.f32"),
                0,
                0,
                t,
                nt,
                h as u32,
                1,
                s,
            ),
            "KKT",
        )?;
        native(
            (self.state.fla.solve)(
                self.state.fla.handle(),
                self.a("A.f32"),
                self.a("Ai.bf16"),
                0,
                0,
                t,
                nt,
                h as u32,
                1,
                s,
            ),
            "inverse",
        )?;
        native(
            (self.state.fla.wy)(
                self.state.fla.handle(),
                self.a("k.bf16"),
                self.a("v.bf16"),
                self.a("beta.bf16"),
                self.a("w.bf16"),
                self.a("u.bf16"),
                self.a("Ai.bf16"),
                self.a("cumulative-gate.f32"),
                0,
                0,
                t,
                nt,
                h as u32,
                1,
                s,
            ),
            "WY",
        )?;
        let complete = if decode { 0 } else { combined / BT * BT };
        let latest = self.state.latest.pointer;
        if decode {
            self.state(0, BT, self.p("boundary-state.f32"), latest)?;
        } else {
            if complete > 0 {
                self.state(
                    0,
                    complete,
                    self.p("boundary-state.f32"),
                    self.a("completed-state.f32"),
                )?;
            }
            if complete < combined {
                self.state(
                    complete,
                    BT,
                    if complete > 0 {
                        self.a("completed-state.f32")
                    } else {
                        self.p("boundary-state.f32")
                    },
                    latest,
                )?;
            } else {
                check(
                    sys::lib().cuMemcpyDtoDAsync_v2(
                        latest,
                        self.a("completed-state.f32"),
                        self.state.layout.persistent["boundary-state.f32"],
                        self.stream(),
                    ),
                    "scalar GDN latest complete state copy",
                )?;
            }
        }
        native(
            (self.state.fla.output)(
                self.state.fla.handle(),
                self.a("q.bf16"),
                self.a("k.bf16"),
                self.a("v-new.bf16"),
                self.a("chunk-state.bf16"),
                self.a("cumulative-gate.f32"),
                0,
                self.a("output.bf16"),
                0,
                0,
                1.0,
                t,
                v.div_ceil(128) as u32,
                nt,
                h as u32,
                s,
            ),
            "output",
        )?;
        self.linear(
            "scalar_gdn_emit",
            rows * h * v,
            &mut [
                P(self.a("emitted.bf16")),
                P(self.a("output.bf16")),
                P(meta),
                I((h * v) as i32),
                I(rows as i32),
            ],
        )?;
        for (name, width) in [
            ("q.bf16", h * k),
            ("k.bf16", h * k),
            ("v.bf16", h * v),
            ("beta.bf16", h),
        ] {
            self.linear(
                "scalar_gdn_retain_bf16",
                BT * width,
                &mut [P(self.p(name)), P(self.a(name)), P(meta), I(width as i32)],
            )?;
        }
        self.linear(
            "scalar_gdn_retain_gate",
            BT * h,
            &mut [
                P(self.p("gate.f32")),
                P(self.a("gate.f32")),
                P(meta),
                I(h as i32),
            ],
        )?;
        self.linear(
            "scalar_gdn_commit_carry",
            h * k * v,
            &mut [
                P(self.p("boundary-state.f32")),
                P(if decode {
                    latest
                } else {
                    self.a("completed-state.f32")
                }),
                P(meta),
                P((h * k * v) as u64),
                I(decode as i32),
            ],
        )?;
        self.kernel("scalar_gdn_commit", (1, 1, 1), 1, &mut [P(meta)])?;
        let _ = partial; // Actual emit offset is device-owned, including graph replay.
        Ok(())
    }
    unsafe fn state(
        &self,
        row: usize,
        rows: usize,
        incoming: u64,
        final_state: u64,
    ) -> Result<(), String> {
        let h = self.state.layout.heads;
        let k = self.state.layout.key_dim;
        let v = self.state.layout.value_dim;
        let qstride = (row * h * k * 2) as u64;
        let vstride = (row * h * v * 2) as u64;
        let gstride = (row * h * 4) as u64;
        let hstride = (row / BT * h * k * v * 2) as u64;
        native(
            (self.state.fla.state)(
                self.state.fla.handle(),
                self.a("k.bf16") + qstride,
                self.a("u.bf16") + vstride,
                self.a("w.bf16") + qstride,
                self.a("v-new.bf16") + vstride,
                self.a("cumulative-gate.f32") + gstride,
                0,
                self.a("chunk-state.bf16") + hstride,
                incoming,
                final_state,
                0,
                0,
                rows as i32,
                v.div_ceil(32) as u32,
                h as u32,
                1,
                self.stream() as *mut c_void,
            ),
            "state recurrence",
        )
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        for stream in &self.streams {
            stream.drain_or_fatal("scalar persistent state actual-use completion");
        }
        drain_or_fatal(
            &self.program.device,
            "scalar persistent state allocation stream",
        );
    }
}
