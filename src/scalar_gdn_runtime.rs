//! Complete per-device scalar-GDN state and phase storage under qualification.
//! No activation/default change; model registration and dispatch are explicit.
use crate::{
    gpu_prefill::PrefillComputeStream,
    scalar_gdn_layout::Layout,
    scalar_gdn_owner::{Inputs, LatestState, Owner, Plane},
    scalar_gdn_phase::{Execution, Stream},
    session_cache::SequenceStateRegistry,
    shared_gdn_workspace::{Lease, Program, Workspace},
};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

pub(crate) struct ColdLayer {
    pub layer: usize,
    pub layout: Layout,
    pub latest: LatestState,
    pub conv: Arc<crate::model_tensor_owner::ModelTensorOwner>,
}
struct State {
    backup: Option<Backup>,
    layers: BTreeMap<usize, Owner>,
    active: Option<Stream>,
    prefill: Option<Arc<Workspace>>,
    prefill_capacity: usize,
    decode: Arc<Workspace>,
    layouts: Vec<Layout>,
    program: Arc<Program>,
    health: CudaSlice<u64>,
    health_host: crate::gpu_decode::SequenceStatePinnedBuffer,
    producer: Option<Arc<crate::scalar_gdn_producer::Producer>>,
}
pub(crate) struct Storage {
    // Drain submitted phases before state/stream owners can be dropped.
    execution: Execution,
    inner: Mutex<State>,
    prefill: Option<Stream>,
    decode: Stream,
    registry: SequenceStateRegistry,
    conv: BTreeMap<usize, Arc<crate::model_tensor_owner::ModelTensorOwner>>,
}
pub(crate) struct Phase<'s> {
    state: MutexGuard<'s, State>,
    prefill: Option<Stream>,
    decode: bool,
}
/// Real workspace owner keeps the emitted BF16 allocation and its actual bounds
/// available to native consumers; the enclosing execution owns GPU completion.
pub(crate) struct Emitted {
    workspace: Arc<Workspace>,
    elements: usize,
}
impl Emitted {
    pub(crate) fn plane(&self) -> Result<Plane<'_, u16>, String> {
        // The qualified native recurrence writes this concrete byte allocation
        // as BF16. No owner, dtype or extent is synthesized from a raw pointer.
        unsafe { Plane::from_region(self.workspace.buffer("emitted.bf16")?, 0, self.elements) }
    }
}
impl Storage {
    pub(crate) fn complete_before_release(&self) {
        self.execution.complete_before_release();
    }
    pub(crate) fn new(
        device: Arc<CudaDevice>,
        prefill: Option<Arc<PrefillComputeStream>>,
        layers: Vec<ColdLayer>,
        sidecar: &Path,
        ptx: &str,
    ) -> Result<Arc<Self>, String> {
        if layers.is_empty() {
            return Err("scalar serving requires actual loaded scalar-GDN layers".into());
        }
        let layouts = layers.iter().map(|x| x.layout.clone()).collect::<Vec<_>>();
        let program = Program::new(device.clone(), &layouts, sidecar, ptx)?;
        // Allocate before actual owners record prefill readiness, so their
        // allocation-stream events also cover this real zero initialization.
        let health = device.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
        let health_host =
            crate::gpu_decode::SequenceStatePinnedBuffer::new(std::mem::size_of::<u64>())?;
        let mut execution = Execution::new(device.clone())?;
        if let Some(prefill) = &prefill {
            execution.admit_prefill(prefill.clone())?;
        }
        let mut owners = BTreeMap::new();
        let mut conv = BTreeMap::new();
        let mut registry = SequenceStateRegistry::default();
        registry.register(crate::session_cache::SequenceStateAllocation {
            name: format!("gpu{}.linear.native64.health", device.ordinal()),
            kind: "scalar_gdn_native64_health".into(),
            layer_idx: None,
            device_ordinal: device.ordinal(),
            ptr: *health.device_ptr(),
            storage_bytes: health.len() * std::mem::size_of::<u64>(),
            dtype: "uint64".into(),
            element_size: std::mem::size_of::<u64>(),
            shape: vec![health.len()],
            strides_bytes: vec![std::mem::size_of::<u64>()],
            growth: crate::session_cache::SequenceStateGrowth::Fixed,
        })?;
        for layer in layers {
            if owners.contains_key(&layer.layer) {
                return Err("duplicate actual scalar layer registration".into());
            }
            let mut owner = Owner::new(program.clone(), layer.layout, layer.latest)?;
            if let Some(prefill) = &prefill {
                owner.admit_prefill(prefill.clone())?;
            }
            for region in owner.registry(layer.layer)?.allocations() {
                registry.register(region.clone())?;
            }
            conv.insert(layer.layer, layer.conv);
            owners.insert(layer.layer, owner);
        }
        // Actual prefill rows are chosen by the measured runtime chunk planner.
        // Allocate this shared arena lazily at that real size, inside calibration
        // and request low-water measurement, never at a fixed prompt capacity.
        let decode_layouts = layouts
            .iter()
            .map(|x| Layout::checked(x.heads, x.key_dim, x.value_dim, 1))
            .collect::<Result<Vec<_>, _>>()?;
        let decode_workspace = Workspace::new(program.clone(), &decode_layouts)?;
        Ok(Arc::new(Self {
            execution,
            inner: Mutex::new(State {
                backup: None,
                layers: owners,
                active: None,
                prefill: None,
                prefill_capacity: 0,
                decode: decode_workspace,
                layouts,
                program,
                health,
                health_host,
                producer: None,
            }),
            prefill: prefill.map(Stream::Prefill),
            decode: Stream::Decode(device),
            registry,
            conv,
        }))
    }
    pub(crate) fn attach_prefill(
        &mut self,
        owner: Arc<PrefillComputeStream>,
    ) -> Result<(), String> {
        if self.prefill.is_some() {
            return Err("scalar prefill stream already attached".into());
        }
        let state = self
            .inner
            .get_mut()
            .map_err(|_| "scalar cold state poisoned")?;
        self.execution.admit_prefill(owner.clone())?;
        for layer in state.layers.values_mut() {
            layer.admit_prefill(owner.clone())?;
        }
        self.prefill = Some(Stream::Prefill(owner));
        Ok(())
    }
    /// Attach before calibration/execution. No allocation owner, layer or
    /// execution device may be invented to make a partial producer appear ready.
    pub(crate) fn attach_producer(
        &mut self,
        producer: Arc<crate::scalar_gdn_producer::Producer>,
    ) -> Result<(), String> {
        let state = self
            .inner
            .get_mut()
            .map_err(|_| "scalar producer cold state poisoned")?;
        if state.producer.is_some()
            || state.active.is_some()
            || state.layers.values().any(Owner::has_started)
        {
            return Err("scalar producer must be attached exactly once before execution".into());
        }
        if producer.context() != state.program.context()
            || !state.layers.keys().copied().eq(producer.layer_ids())
        {
            return Err(
                "scalar producer actual context/layer inventory differs from native state".into(),
            );
        }
        state.producer = Some(producer);
        Ok(())
    }
    pub(crate) fn producer(&self) -> Result<Arc<crate::scalar_gdn_producer::Producer>, String> {
        self.inner
            .try_lock()
            .map_err(|_| "scalar producer state held/poisoned")?
            .producer
            .clone()
            .ok_or("scalar serving producer was not cold-attached".into())
    }
    pub(crate) fn semantic_identity(&self) -> Result<serde_json::Value, String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar identity state held/poisoned")?;
        let producer = state
            .producer
            .as_ref()
            .ok_or("scalar identity requires the cold-attached producer")?;
        Ok(serde_json::json!({
            "format":"native64-f32-carry-pending-bf16-prepared-canonical-prefill-v2",
            "native":state.program.identity,
            "producer":producer.identity,
        }))
    }
    pub(crate) fn registry(&self) -> &SequenceStateRegistry {
        &self.registry
    }
    /// Enclose the entire producer/recurrence/consumer invocation, including
    /// capture+actual replay. Phase work borrows existing model allocations;
    /// closure-local buffers need their own completion-aware owner guards.
    ///
    /// # Safety
    /// All submitted model resources remain alive through this whole invocation
    /// and errors/unwind. Work owns capture cleanup and transfer-local guards.
    pub(crate) unsafe fn with_prefill<T>(
        &self,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let stream = self
            .prefill
            .as_ref()
            .ok_or("scalar actual prefill stream not attached")?;
        self.with_invocation(stream, false, work)
    }
    /// Same lifetime obligations as with_prefill. Retire the large prefill arena
    /// before decode/capture/replay, outside capture and measured by calibration.
    pub(crate) unsafe fn with_decode<T>(
        &self,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.with_invocation(&self.decode, true, work)
    }
    unsafe fn with_invocation<T>(
        &self,
        stream: &Stream,
        decode: bool,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.execution.execute(stream, || {
            {
                let mut state = self
                    .inner
                    .try_lock()
                    .map_err(|_| "scalar invocation state held/poisoned")?;
                if state.active.is_some() {
                    return Err("nested scalar whole invocation".into());
                }
                if decode {
                    drop(state.prefill.take());
                    state.prefill_capacity = 0;
                }
                state.active = Some(stream.clone());
            }
            // State/arenas remain owned by self throughout callback and outer
            // completion, including unwinds. Per-layer host leases never wait
            // for the GPU; phase ordering stays in the outer execution domain.
            let result = work();
            let mut state = self
                .inner
                .try_lock()
                .map_err(|_| "scalar invocation exit state held/poisoned")?;
            state.active = None;
            result
        })
    }
    unsafe fn phase(&self, decode: bool) -> Result<Phase<'_>, String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar layer state held/poisoned")?;
        let expected = if decode {
            &self.decode
        } else {
            self.prefill
                .as_ref()
                .ok_or("scalar prefill stream absent")?
        };
        if !state
            .active
            .as_ref()
            .is_some_and(|stream| stream.same(expected))
        {
            return Err("scalar layer submitted outside its actual whole invocation".into());
        }
        Ok(Phase {
            state,
            prefill: if decode { None } else { self.prefill.clone() },
            decode,
        })
    }
    /// # Safety
    /// Actual producer/output owners outlive the enclosing with_prefill call.
    pub(crate) unsafe fn enqueue_prefill(
        &self,
        layer: usize,
        start: u64,
        rows: usize,
        inputs: &Inputs<'_>,
        output: &Plane<'_, u16>,
    ) -> Result<(), String> {
        self.phase(false)?
            .prefill(layer, start, rows, inputs, output)
    }
    /// # Safety
    /// Actual producer/output/position owners outlive capture and all replays.
    pub(crate) unsafe fn enqueue_decode(
        &self,
        layer: usize,
        inputs: &Inputs<'_>,
        position: &CudaSlice<i32>,
        output: &Plane<'_, u16>,
    ) -> Result<(), String> {
        self.phase(true)?.decode(layer, inputs, position, output)
    }
    pub(crate) unsafe fn prepare_prefill(
        &self,
        layer: usize,
        rows: usize,
    ) -> Result<Arc<Workspace>, String> {
        let mut phase = self.phase(false)?;
        if !phase.state.layers.contains_key(&layer) {
            return Err("scalar preparation actual layer absent".into());
        }
        phase.prefill_workspace(rows)
    }
    /// # Safety
    /// Real prepared input allocations outlive the enclosing whole invocation.
    /// Returned output is owned by this storage's shared arena until its next
    /// phase/chunk reuse. Its consumer must enqueue before that reuse.
    pub(crate) unsafe fn prefill_output(
        &self,
        layer: usize,
        start: u64,
        rows: usize,
        inputs: &Inputs<'_>,
    ) -> Result<u64, String> {
        let mut phase = self.phase(false)?;
        let workspace = phase.prefill_workspace(rows)?;
        let lease = workspace.lease()?;
        let State { layers, health, .. } = &mut *phase.state;
        let owner = layers
            .get_mut(&layer)
            .ok_or("scalar prefill actual layer absent")?;
        owner.enqueue_prefill(&lease, start, rows, inputs)?;
        owner.collect_health(&lease, health)?;
        Ok(*workspace.buffer("emitted.bf16")?.device_ptr())
    }
    /// # Safety
    /// Same arena/consumer contract; actual position survives captured replay.
    pub(crate) unsafe fn decode_output(
        &self,
        layer: usize,
        inputs: &Inputs<'_>,
        position: &CudaSlice<i32>,
    ) -> Result<u64, String> {
        let mut phase = self.phase(true)?;
        let workspace = phase.state.decode.clone();
        let lease = workspace.lease()?;
        let State { layers, health, .. } = &mut *phase.state;
        let owner = layers
            .get_mut(&layer)
            .ok_or("scalar decode actual layer absent")?;
        owner.enqueue_decode(&lease, inputs, position)?;
        owner.collect_health(&lease, health)?;
        Ok(*workspace.buffer("emitted.bf16")?.device_ptr())
    }
    /// # Safety
    /// All actual producer/position owners and this emitted consumer remain
    /// inside the enclosing whole decode invocation, including error paths.
    pub(crate) unsafe fn decode_emitted(
        &self,
        layer: usize,
        inputs: &Inputs<'_>,
        position: &crate::scalar_gdn_owner::DecodePosition<'_>,
    ) -> Result<Emitted, String> {
        let mut phase = self.phase(true)?;
        let workspace = phase.state.decode.clone();
        let lease = workspace.lease()?;
        let State { layers, health, .. } = &mut *phase.state;
        let owner = layers
            .get_mut(&layer)
            .ok_or("scalar decode actual layer absent")?;
        owner.enqueue_decode_position(&lease, inputs, position)?;
        let elements = owner
            .layout
            .heads
            .checked_mul(owner.layout.value_dim)
            .ok_or("scalar emitted output extent overflow")?;
        owner.collect_health(&lease, health)?;
        drop(lease);
        Ok(Emitted {
            workspace,
            elements,
        })
    }
    fn read_health_completed(state: &State, stream: &Stream) -> Result<(), String> {
        let device = &state.program.device;
        // Reuse the installed541 copy-completion guard. Real pinned host and
        // device allocations remain borrowed even on enqueue failure/unwind.
        unsafe {
            crate::sequence_copy_completion::with_copy_completion(
                device,
                stream.handle(),
                &(state, &state.health_host),
                |_| {
                    sys::lib()
                        .cuMemcpyDtoHAsync_v2(
                            state.health_host.ptr.cast(),
                            *state.health.device_ptr(),
                            std::mem::size_of::<u64>(),
                            stream.handle(),
                        )
                        .result()
                        .map_err(|e| format!("scalar semantic health read:{e}"))
                },
            )?;
            if state.health_host.ptr.cast::<u64>().read() != 0 {
                return Err("scalar GDN device semantic state is invalid; verified reset/restore required before publication".into());
            }
        }
        Ok(())
    }
    /// Check at an existing CPU logit boundary inside the owned invocation.
    /// The outer execution guard owns ordering and failure cleanup; entering
    /// publication here would recursively acquire its timeline lock.
    pub(crate) fn require_active_health(&self) -> Result<(), String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar active health state held/poisoned")?;
        let stream = state
            .active
            .as_ref()
            .ok_or("scalar active health requires an owned invocation")?;
        Self::read_health_completed(&state, stream)
    }
    /// Called before exposing sampled tokens or logits, after the real phase.
    /// One checked status read per participating device, never per layer.
    pub(crate) fn require_publication_health(&self) -> Result<(), String> {
        unsafe {
            self.execution.publication(&self.decode, || {
                let state = self
                    .inner
                    .try_lock()
                    .map_err(|_| "scalar publication state held/poisoned")?;
                if state.active.is_some() {
                    return Err("scalar publication attempted inside unfinished invocation".into());
                }
                Self::read_health_completed(&state, &self.decode)
            })
        }
    }
    /// Prefix checkpoints occur inside a running prefill invocation. Complete
    /// its actual stream before inspecting state, while retaining the outer
    /// execution guard; do not recursively acquire the phase timeline.
    ///
    /// # Safety
    /// Work retains every transfer owner and completes its copy stream on all
    /// exits, just as quiescent(). No layer submission may overlap this call.
    pub(crate) unsafe fn snapshot_boundary<T>(
        &self,
        cursor: usize,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "scalar snapshot boundary state held/poisoned")?;
        if let Some(stream) = state.active.as_ref() {
            Self::read_health_completed(&state, stream)?;
            for owner in state.layers.values() {
                owner.require_snapshot_completed(cursor)?;
            }
            return work();
        }
        drop(state);
        self.quiescent(cursor, work)
    }
    /// # Safety
    /// Work retains and completes its real transfer owners/streams. This
    /// protects ALL scalar phases around generic full/incremental snapshots.
    pub(crate) unsafe fn quiescent<T>(
        &self,
        cursor: usize,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.execution.quiescent(|| {
            let state = self
                .inner
                .try_lock()
                .map_err(|_| "scalar snapshot state held/poisoned")?;
            for owner in state.layers.values() {
                owner.require_snapshot_completed(cursor)?;
            }
            Self::read_health_completed(&state, &self.decode)?;
            work()
        })
    }
    /// Reset the entire owned scalar family, including the original actual
    /// convolution and latest tensors. Other model families keep their existing
    /// reset contract; this execution domain never clears their validity flags.
    pub(crate) fn reset(&self) -> Result<(), String> {
        use crate::cuda_region::CudaRegion;
        // SAFETY: all actual scalar state owners are retained by self, and the
        // domain drains both phases before entry and on every exit/error/unwind.
        unsafe {
            self.execution.recover(|| {
                let mut state = self
                    .inner
                    .try_lock()
                    .map_err(|_| "scalar reset state held/poisoned")?;
                state.active = None;
                if let Some(backup) = state.backup.as_mut() {
                    backup.cursor = None;
                }
                let device = state.program.device.clone();
                device.bind_to_thread().map_err(|e| e.to_string())?;
                for conv in self.conv.values() {
                    let (pointer, end) = conv.extent()?;
                    let bytes = usize::try_from(end - pointer).map_err(|e| e.to_string())?;
                    sys::lib()
                        .cuMemsetD8Async(pointer, 0, bytes, *device.cu_stream())
                        .result()
                        .map_err(|e| format!("scalar actual convolution reset:{e}"))?;
                }
                for owner in state.layers.values_mut() {
                    owner.reset_completed()?;
                }
                sys::lib()
                    .cuMemsetD8Async(
                        *state.health.device_ptr(),
                        0,
                        state.health.len() * std::mem::size_of::<u64>(),
                        *device.cu_stream(),
                    )
                    .result()
                    .map_err(|e| format!("scalar semantic health reset:{e}"))?;
                Ok(())
            })
        }
    }
    /// # Safety
    /// Work restores EVERY region in registry() plus the model's registered
    /// conv/latest and host positions using the validated generic inventory.
    /// It completes transfer streams even on error/unwind. No partial restore
    /// or different source/layout may report success. Logical cursor is actual
    /// consumed tokens, not generated text or an estimate.
    pub(crate) unsafe fn restore<T>(
        &self,
        cursor: usize,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.execution.recover(|| {
            let mut state = self
                .inner
                .try_lock()
                .map_err(|_| "scalar restore state held/poisoned")?;
            state.active = None;
            if let Some(backup) = state.backup.as_mut() {
                backup.cursor = None;
            }
            let value = work()?;
            for owner in state.layers.values_mut() {
                owner.validate_restored_completed(cursor)?;
            }
            Self::read_health_completed(&state, &self.decode)?;
            Ok(value)
        })
    }
}
impl Phase<'_> {
    fn prefill_workspace(&mut self, rows: usize) -> Result<Arc<Workspace>, String> {
        if self.decode {
            return Err("scalar prefill submitted in decode phase".into());
        }
        if rows == 0 {
            return Err("empty scalar actual prefill chunk".into());
        }
        if self.state.prefill.is_none() || rows > self.state.prefill_capacity {
            drop(self.state.prefill.take());
            self.state.prefill_capacity = 0;
            let layouts = self
                .state
                .layouts
                .iter()
                .map(|x| Layout::checked(x.heads, x.key_dim, x.value_dim, rows))
                .collect::<Result<Vec<_>, _>>()?;
            let Some(Stream::Prefill(owner)) = &self.prefill else {
                return Err("scalar actual prefill stream absent".into());
            };
            let workspace =
                Workspace::on_prefill(self.state.program.clone(), &layouts, owner.clone())?;
            self.state.prefill = Some(workspace);
            self.state.prefill_capacity = rows;
        }
        self.state
            .prefill
            .as_ref()
            .cloned()
            .ok_or("scalar actual prefill workspace absent".into())
    }
    /// # Safety
    /// Actual typed producer views and writable output remain alive through the
    /// enclosing phase. Output may overlap staged inputs only after their final
    /// use. No other consumer may write the same storage concurrently.
    pub(crate) unsafe fn prefill(
        &mut self,
        layer: usize,
        start: u64,
        rows: usize,
        inputs: &Inputs<'_>,
        output: &Plane<'_, u16>,
    ) -> Result<(), String> {
        if self.decode {
            return Err("scalar prefill submitted in decode phase".into());
        }
        let workspace = self.prefill_workspace(rows)?;
        let lease = workspace.lease()?;
        let State { layers, health, .. } = &mut *self.state;
        let owner = layers
            .get_mut(&layer)
            .ok_or("scalar prefill actual layer absent")?;
        owner.enqueue_prefill(&lease, start, rows, inputs)?;
        owner.collect_health(&lease, health)?;
        Self::copy_output(&lease, owner, rows, output)
    }
    /// # Safety
    /// Same typed producer/output/lifetime obligations as prefill; actual
    /// device position must be retained through captured graph execution.
    pub(crate) unsafe fn decode(
        &mut self,
        layer: usize,
        inputs: &Inputs<'_>,
        position: &CudaSlice<i32>,
        output: &Plane<'_, u16>,
    ) -> Result<(), String> {
        if !self.decode {
            return Err("scalar decode submitted in prefill phase".into());
        }
        let workspace = self.state.decode.clone();
        let lease = workspace.lease()?;
        let State { layers, health, .. } = &mut *self.state;
        let owner = layers
            .get_mut(&layer)
            .ok_or("scalar decode actual layer absent")?;
        owner.enqueue_decode(&lease, inputs, position)?;
        owner.collect_health(&lease, health)?;
        Self::copy_output(&lease, owner, 1, output)
    }
    unsafe fn copy_output(
        lease: &Lease<'_>,
        owner: &Owner,
        rows: usize,
        output: &Plane<'_, u16>,
    ) -> Result<(), String> {
        let elements = rows
            .checked_mul(owner.layout.heads)
            .and_then(|n| n.checked_mul(owner.layout.value_dim))
            .ok_or("scalar emitted extent overflow")?;
        output.require_context_extent(lease.workspace.program.context(), elements)?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or("scalar emitted byte overflow")?;
        sys::lib()
            .cuMemcpyDtoDAsync_v2(
                output.pointer(),
                *lease.workspace.buffer("emitted.bf16")?.device_ptr(),
                bytes,
                lease.workspace.stream(),
            )
            .result()
            .map_err(|e| format!("scalar emitted actual output copy:{e}"))
    }
}

const _: fn() = || {
    fn require<T: Send + Sync>() {}
    require::<Storage>();
};

include!("scalar_gdn_backup.rs");
