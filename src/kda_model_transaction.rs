// Private complete-graph family admission. No serving selector is admitted.
struct ModelFamilyBinding<'a> {
    layer: usize,
    owner: &'a PreparedOwner,
    producer: &'a KdaRawProducer,
}
fn model_family_identity(bindings: &[ModelFamilyBinding<'_>]) -> Result<serde_json::Value, String> {
    let mut identities = Vec::new();
    let mut complete_registry = crate::session_cache::SequenceStateRegistry::default();
    let mut last = None;
    let mut state_addresses = std::collections::BTreeSet::new();
    if bindings.is_empty() {
        return Err("complete graph has no admitted families".into());
    }
    for b in bindings {
        if last.is_some_and(|x| x >= b.layer) {
            return Err("complete graph families must be unique and canonical".into());
        }
        last = Some(b.layer);
        if !state_addresses.insert(Arc::as_ptr(&b.owner.state) as usize) {
            return Err("complete graph family aliases persistent state".into());
        }
        if b.owner.heads != b.producer.heads
            || b.owner.dim != b.producer.dim
            || !Arc::ptr_eq(&b.owner.device, &b.producer.raw.device)
            || b.owner.stream.handle != b.producer.raw.stream.handle
        {
            return Err("complete graph producer/family geometry or stream".into());
        }
        b.owner.register(&mut complete_registry, b.layer)?;
        let mut registry = crate::session_cache::SequenceStateRegistry::default();
        b.owner.register(&mut registry, b.layer)?;
        identities.push(serde_json::json!({"layer":b.layer,"owner_arithmetic":b.owner.arithmetic_sha256,"producer_arithmetic":b.producer.arithmetic_sha256,"heads":b.owner.heads,"dim":b.owner.dim,"layout":registry.compatibility_layout_value(),"required_allocations":registry.allocations(),"captured_workspace_binding":model_workspace_identity(b)?}));
    }
    Ok(serde_json::json!(identities))
}
struct ModelFamilyAdmission<'a> {
    stream: &'a NativeStream,
    slots: Vec<std::sync::MutexGuard<'a, Option<StateTimeline>>>,
}
impl<'a> ModelFamilyAdmission<'a> {
    fn checked(
        bindings: &'a [ModelFamilyBinding<'a>],
        stream: &'a NativeStream,
        expected: &serde_json::Value,
    ) -> Result<Self, String> {
        stream.validate()?;
        if &model_family_identity(bindings)? != expected {
            return Err("complete graph family identity or inventory incomplete".into());
        }
        Self::checked_after_identity(bindings, stream)
    }
    fn from_proof(proof: &ModelBindingProof<'a>, stream: &'a NativeStream) -> Result<Self, String> {
        stream.validate()?;
        Self::checked_after_identity(proof.bindings, stream)
    }
    fn checked_after_identity(
        bindings: &'a [ModelFamilyBinding<'a>],
        stream: &'a NativeStream,
    ) -> Result<Self, String> {
        let mut slots = Vec::with_capacity(bindings.len());
        for b in bindings {
            if !Arc::ptr_eq(&b.owner.device, &stream.device) {
                return Err("complete graph foreign family device".into());
            }
            let slot = b
                .owner
                .state
                .timeline
                .try_lock()
                .map_err(|_| "complete graph family concurrently leased/poisoned")?;
            let timeline = slot
                .as_ref()
                .ok_or("complete graph family timeline missing")?;
            if timeline.poisoned {
                return Err("complete graph family poisoned".into());
            }
            if !timeline
                .streams
                .iter()
                .any(|x| x.handle == stream.handle && Arc::ptr_eq(&x.device, &stream.device))
            {
                return Err("complete graph execution stream unadmitted".into());
            }
            timeline
                .phases
                .checked_add(1)
                .ok_or("complete graph phase overflow")?;
            slots.push(slot);
        }
        Ok(Self { stream, slots })
    }
    fn validate_ready(&self) -> Result<(), String> {
        self.stream.validate()?;
        for slot in &self.slots {
            let t = slot
                .as_ref()
                .ok_or("complete graph family timeline missing")?;
            if t.poisoned {
                return Err("complete graph family poisoned".into());
            }
            t.phases
                .checked_add(1)
                .ok_or("complete graph phase overflow")?;
        }
        Ok(())
    }
    fn capture_only<T>(&mut self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        self.validate_ready()?;
        // No event wait/record or host phase mutation belongs inside capture.
        // All admitted family mutexes remain held; arena leases are taken by work.
        let before = self
            .slots
            .iter()
            .map(|s| {
                let t = s.as_ref().expect("checked family");
                (t.phases, t.recorded, t.poisoned)
            })
            .collect::<Vec<_>>();
        let result = work();
        for (slot, want) in self.slots.iter().zip(before) {
            let t = slot.as_ref().ok_or("capture removed timeline")?;
            if (t.phases, t.recorded, t.poisoned) != want {
                return Err("capture changed committed host family timeline".into());
            }
        }
        result
    }
    fn execute<T>(&mut self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        self.validate_ready()?;
        // All families have passed admission before any GPU submission.
        // Mark the entire transaction poisoned until every completion records.
        for slot in &mut self.slots {
            slot.as_mut().ok_or("family vanished")?.poisoned = true;
        }
        for slot in &self.slots {
            let t = slot.as_ref().ok_or("family vanished")?;
            if t.recorded {
                check(
                    unsafe { sys::lib().cuStreamWaitEvent(self.stream.handle, t.event, 0) },
                    "complete graph preceding family wait",
                )?;
            }
        }
        let value = work();
        let mut failure = None;
        for slot in &mut self.slots {
            let t = slot.as_mut().ok_or("family vanished")?;
            match check(
                unsafe { sys::lib().cuEventRecord(t.event, self.stream.handle) },
                "complete graph family completion record",
            ) {
                Ok(()) => t.recorded = true,
                Err(error) => {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        match value {
            Ok(value) => {
                for slot in &mut self.slots {
                    let t = slot.as_mut().ok_or("family vanished")?;
                    t.phases = t
                        .phases
                        .checked_add(1)
                        .ok_or("complete graph phase overflow")?;
                    t.poisoned = false;
                }
                Ok(value)
            }
            Err(error) => Err(error),
        }
    }
}
