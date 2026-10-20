// Private diagnostic ordering contract; no serving call site is admitted.
struct StateTimeline {
    event: sys::CUevent,
    device: Arc<CudaDevice>,
    recorded: bool,
    poisoned: bool,
    phases: u64,
    streams: Vec<Arc<NativeStream>>,
}
// SAFETY: completion event and retained stream/device owners transfer only under
// the state-family Mutex. All event updates/drop bind the retained device.
// No Sync implementation: event access is externally serialized by the Mutex.
unsafe impl Send for StateTimeline {}
impl Drop for StateTimeline {
    fn drop(&mut self) {
        if let Err(e) = self.device.bind_to_thread() {
            crate::vram_monitor::fatal_cuda_context_error(
                "state timeline teardown context",
                &e.to_string(),
            );
        }
        // Teardown only; phases never use a host wait to establish ordering.
        if self.recorded {
            let rc = unsafe { sys::lib().cuEventSynchronize(self.event) };
            if rc != sys::CUresult::CUDA_SUCCESS {
                crate::vram_monitor::fatal_cuda_context_error(
                    "state timeline final completion",
                    &format!("{rc:?}"),
                );
            }
        }
        for stream in &self.streams {
            if let Err(e) = stream.drain() {
                crate::vram_monitor::fatal_cuda_context_error(
                    "state timeline retained stream completion",
                    &e,
                );
            }
        }
        let rc = unsafe { sys::lib().cuEventDestroy_v2(self.event) };
        if rc != sys::CUresult::CUDA_SUCCESS {
            crate::vram_monitor::fatal_cuda_context_error(
                "state timeline event release",
                &format!("{rc:?}"),
            );
        }
    }
}
struct StatePhase<'a> {
    owner: &'a PreparedOwner,
    stream: &'a NativeStream,
}
impl PreparedOwner {
    fn initialize_timeline(&self, streams: &[Arc<NativeStream>]) -> Result<(), String> {
        if streams.is_empty() {
            return Err("timeline requires explicit retained streams".into());
        }
        for stream in streams {
            stream.validate()?;
            if !Arc::ptr_eq(&self.device, &stream.device) {
                return Err("timeline stream foreign context".into());
            }
        }
        self.stream.validate()?;
        let mut slot = self
            .state
            .timeline
            .try_lock()
            .map_err(|_| "state family concurrently leased/poisoned")?;
        if slot.is_some() {
            return Err("state timeline already initialized".into());
        }
        let mut event = std::ptr::null_mut();
        check(
            unsafe {
                sys::lib().cuEventCreate(
                    &mut event,
                    sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                )
            },
            "state completion event create",
        )?;
        *slot = Some(StateTimeline {
            event,
            device: self.device.clone(),
            recorded: false,
            poisoned: false,
            phases: 0,
            streams: streams.to_vec(),
        });
        Ok(())
    }
    fn bind_ordered(arena_owner: &Self, state_owner: &Self) -> Result<Self, String> {
        arena_owner.stream.validate()?;
        state_owner.stream.validate()?;
        if !Arc::ptr_eq(&arena_owner.device, &state_owner.device)
            || arena_owner.heads != state_owner.heads
            || arena_owner.dim != state_owner.dim
            || arena_owner.arithmetic_sha256 != state_owner.arithmetic_sha256
            || arena_owner.persistent_bytes != state_owner.persistent_bytes
        {
            return Err("ordered binding geometry/context/arithmetic/layout mismatch".into());
        }
        let slot = state_owner
            .state
            .timeline
            .try_lock()
            .map_err(|_| "state family concurrently leased/poisoned")?;
        let timeline = slot
            .as_ref()
            .ok_or("state timeline must be initialized before binding")?;
        if timeline.poisoned {
            return Err("state timeline poisoned".into());
        }
        if !timeline
            .streams
            .iter()
            .any(|s| s.handle == arena_owner.stream.handle)
        {
            return Err("binding stream unadmitted".into());
        }
        for x in &state_owner.native_admissions {
            native_pointer_admission(&arena_owner.stream, x.ptr, x.requested_bytes)?;
        }
        for (desc, ptr) in state_owner.pending_raw.inventory() {
            native_pointer_admission(&arena_owner.stream, ptr, desc.storage_bytes)?;
        }
        Ok(Self {
            device: arena_owner.device.clone(),
            heads: arena_owner.heads,
            dim: arena_owner.dim,
            max_new_rows: arena_owner.max_new_rows,
            padded: arena_owner.padded,
            arena: arena_owner.arena.clone(),
            state: state_owner.state.clone(),
            arithmetic_sha256: arena_owner.arithmetic_sha256.clone(),
            persistent_bytes: state_owner.persistent_bytes,
            allocation_bytes: 0,
            stream: arena_owner.stream.clone(),
        })
    }
    fn ordered_phase<T>(
        &self,
        stream: &NativeStream,
        enqueue: impl FnOnce(&StatePhase<'_>) -> Result<T, String>,
    ) -> Result<T, String> {
        stream.validate()?;
        if !Arc::ptr_eq(&self.device, &stream.device) {
            return Err("state phase foreign context".into());
        }
        let mut slot = self
            .state
            .timeline
            .try_lock()
            .map_err(|_| "state family concurrently leased/poisoned")?;
        let timeline = slot.as_mut().ok_or("state timeline uninitialized")?;
        if timeline.poisoned {
            return Err("state timeline poisoned".into());
        }
        if !timeline
            .streams
            .iter()
            .any(|s| s.handle == stream.handle && Arc::ptr_eq(&s.device, &stream.device))
        {
            return Err("unadmitted phase stream".into());
        }
        let next = timeline
            .phases
            .checked_add(1)
            .ok_or("state phase count overflow")?;
        if timeline.recorded {
            if let Err(e) = check(
                unsafe { sys::lib().cuStreamWaitEvent(stream.handle, timeline.event, 0) },
                "state consumer event wait",
            ) {
                timeline.poisoned = true;
                return Err(e);
            }
        }
        // Any failed submission invalidates future ownership; never continue with partial work.
        timeline.poisoned = true;
        let result = enqueue(&StatePhase {
            owner: self,
            stream,
        });
        let record = check(
            unsafe { sys::lib().cuEventRecord(timeline.event, stream.handle) },
            "state producer completion record",
        );
        if record.is_ok() {
            timeline.recorded = true;
        }
        record?;
        match result {
            Ok(value) => {
                timeline.phases = next;
                timeline.poisoned = false;
                Ok(value)
            }
            Err(e) => Err(e),
        }
    }
    fn ordered_graph(&self, phase: &StatePhase<'_>, graph: &OwnerGraph<'_>) -> Result<(), String> {
        if !std::ptr::eq(self, phase.owner)
            || !std::ptr::eq(self, graph.owner)
            || graph.exec.is_null()
        {
            return Err("foreign phase/graph owner".into());
        }
        let lease = self.arena.lease()?;
        self.ordered_graph_in_lease(phase, graph, &lease)
    }
    fn ordered_graph_in_lease(
        &self,
        phase: &StatePhase<'_>,
        graph: &OwnerGraph<'_>,
        lease: &EnqueueLease<'_>,
    ) -> Result<(), String> {
        if !std::ptr::eq(self, phase.owner)
            || !std::ptr::eq(self, graph.owner)
            || graph.exec.is_null()
        {
            return Err("foreign phase/graph owner".into());
        }
        self.validate_lease(lease)?;
        phase.stream.validate()?;
        if !Arc::ptr_eq(&self.device, &phase.stream.device) {
            return Err("foreign graph execution device".into());
        }
        check(
            unsafe { sys::lib().cuGraphLaunch(graph.exec, phase.stream.handle) },
            "ordered state graph launch",
        )
    }
}

impl StatePhase<'_> {
    fn prefill(&self, producer: &KdaRawProducer, first: usize, rows: usize) -> Result<(), String> {
        if self.stream.handle != self.owner.stream.handle {
            return Err("batch phase execution stream mismatch".into());
        }
        let lease = self.owner.arena.lease()?;
        producer.enqueue(first, rows, self.owner)?;
        self.owner
            .prefill_in(&lease, &producer.view(rows)?, first as u64, rows)
    }
    fn reset(&self) -> Result<(), String> {
        if self.stream.handle != self.owner.stream.handle {
            return Err("reset phase execution stream mismatch".into());
        }
        self.owner.reset_in_phase(self)
    }
}
