// Complete current-storage quiescent phase. This is not whole-model cache
// publication: callers must separately admit operational inventory/signature.
struct KdaCurrentSequenceResources<'a> {
    groups: Vec<std::sync::MutexGuard<'a, StateTimeline>>,
}
impl<'a> KdaCurrentSequenceResources<'a> {
    fn checked(domain: &'a ProducerResourceDomain) -> Result<Self, String> {
        let mut groups = Vec::with_capacity(domain.groups.len());
        for g in &domain.groups {
            let t = g
                .timeline
                .try_lock()
                .map_err(|_| "current sequence resource held/poisoned")?;
            if t.poisoned {
                return Err("current sequence resource poisoned".into());
            }
            t.phases
                .checked_add(1)
                .ok_or("current sequence resource phase overflow")?;
            for s in &t.streams {
                s.validate()?;
            }
            groups.push(t);
        }
        if groups.is_empty() {
            return Err("current sequence resource inventory empty".into());
        }
        Ok(Self { groups })
    }
    fn drain_all(&self) -> Result<(), String> {
        let mut error = None;
        for t in &self.groups {
            if let Err(e) = t.device.bind_to_thread().map_err(|e| e.to_string()) {
                error.get_or_insert(e);
                continue;
            }
            if t.recorded {
                if let Err(e) = check(
                    unsafe { sys::lib().cuEventSynchronize(t.event) },
                    "current sequence resource completion",
                ) {
                    error.get_or_insert(e);
                }
            }
            for s in &t.streams {
                if let Err(e) = s.drain() {
                    error.get_or_insert(e);
                }
            }
        }
        error.map_or(Ok(()), Err)
    }
    fn execute<T>(&mut self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        for t in &mut self.groups {
            t.poisoned = true;
        }
        self.drain_all()?;
        let result = work();
        // Complete every submitted stream even when the callback failed. The
        // retained owners/guards cannot be released ahead of failed DMA/reset.
        let completion = self.drain_all();
        completion?;
        let value = result?;
        for t in &mut self.groups {
            t.recorded = false;
            t.phases += 1;
            t.poisoned = false;
        }
        Ok(value)
    }
}
fn current_sequence_owners(
    life: &KdaPrefillStorageLifetime,
) -> Result<Vec<(&PreparedOwner, &PreparedOwner)>, String> {
    match &life.state {
        KdaPrefillStorageState::Bound(f) => Ok(f.iter().map(|f| (&f.prefill, &f.decode)).collect()),
        KdaPrefillStorageState::Retired(f) => {
            Ok(f.iter().map(|f| (&f.prefill, &f.decode)).collect())
        }
        _ => Err("current sequence lifetime poisoned/extracted".into()),
    }
}
impl KdaEngineRequestStorage {
    fn current_sequence_transfer<T>(
        &self,
        expected: &serde_json::Value,
        expected_generation: u64,
        expected_epoch: u64,
        stores: &mut [&mut crate::gpu_decode::GpuDecodeStore],
        work: impl FnOnce(
            &KdaPrefillStorageLifetime,
            &mut [&mut crate::gpu_decode::GpuDecodeStore],
        ) -> Result<T, String>,
    ) -> Result<(T, u64), String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "current sequence storage held/poisoned")?;
        if state.poisoned
            || state.domain_generation != expected_generation
            || state.live_sequence_epoch != expected_epoch
            || &state.lifetime.storage_identity()? != expected
            || request_storage_static_identity(&state.lifetime)? != state.static_identity
        {
            return Err("current sequence stale source/inventory/live epoch".into());
        }
        let next = expected_epoch
            .checked_add(1)
            .ok_or("current sequence live epoch overflow")?;
        let members = current_resource_members(&state.lifetime)?;
        if members.len() != state.resource_domain.producers.len() {
            return Err("current sequence domain role inventory".into());
        }
        for (want, have) in members.iter().zip(&state.resource_domain.producers) {
            match (want, have) {
                (OwnedProducerResourceMember::Bound(w), OwnedProducerResourceMember::Bound(h)) => {
                    if !Arc::ptr_eq(&w.raw, &h.raw)
                        || !Arc::ptr_eq(&w.prepared, &h.prepared)
                        || !Arc::ptr_eq(&w.a_log, &h.a_log)
                        || !Arc::ptr_eq(&w.bias, &h.bias)
                        || !Arc::ptr_eq(&w.kernels, &h.kernels)
                        || w.settings != h.settings
                        || w.arithmetic != h.arithmetic
                    {
                        return Err("current sequence stale bound domain role".into());
                    }
                }
                (
                    OwnedProducerResourceMember::RetiredPreparation(w),
                    OwnedProducerResourceMember::RetiredPreparation(h),
                ) => {
                    if !Arc::ptr_eq(&w.prepared, &h.prepared)
                        || !Arc::ptr_eq(&w.a_log, &h.a_log)
                        || !Arc::ptr_eq(&w.bias, &h.bias)
                        || !Arc::ptr_eq(&w.kernels, &h.kernels)
                        || w.source_epoch != h.source_epoch
                        || w.heads != h.heads
                        || w.dim != h.dim
                        || w.capacity != h.capacity
                        || w.lower_bits != h.lower_bits
                        || w.scale_bits != h.scale_bits
                        || w.epsilon_bits != h.epsilon_bits
                        || w.arithmetic != h.arithmetic
                    {
                        return Err("current sequence stale retired domain role".into());
                    }
                }
                _ => return Err("current sequence domain capability differs".into()),
            }
        }
        let owners = current_sequence_owners(&state.lifetime)?;
        let witnesses = KdaQuiescentStorageWitnesses::checked(&owners)?;
        let device = &owners
            .first()
            .ok_or("current sequence owners empty")?
            .0
            .device;
        // This bounded current storage admits one actual context explicitly.
        if stores.len() != 1
            || !stores[0].session_dma_matches_device_rust(
                device.ordinal(),
                *device.cu_primary_ctx() as usize,
            )?
        {
            return Err("current sequence complete DMA context not admitted".into());
        }
        let started = std::cell::Cell::new(false);
        let result = {
            let mut resources = KdaCurrentSequenceResources::checked(&state.resource_domain)?;
            witnesses.execute_marked(&started, || {
                resources.execute(|| {
                    // The Store retains its separate copy stream throughout the
                    // callback. Complete it before writes and on every exit,
                    // while all resource/family/arena guards remain held.
                    for store in stores.iter() {
                        store.complete_session_dma_stream_rust()?;
                    }
                    let result = work(&state.lifetime, stores);
                    let mut completion_error = None;
                    for store in stores.iter() {
                        if let Err(e) = store.complete_session_dma_stream_rust() {
                            completion_error.get_or_insert(e);
                        }
                    }
                    if let Some(e) = completion_error {
                        return Err(e);
                    }
                    result
                })
            })
        };
        match result {
            Ok(value) => {
                state.live_sequence_epoch = next;
                Ok((value, next))
            }
            Err(e) => {
                if started.get() {
                    current_storage_poison(&mut state);
                }
                Err(e)
            }
        }
    }
}
