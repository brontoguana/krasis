// cfgtest-only exact307 identity/admission bodies, names renewed only.
fn model_family_identity_frozen307(bindings: &[ModelFamilyBinding<'_>]) -> Result<serde_json::Value, String> {
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
        identities.push(serde_json::json!({"layer":b.layer,"owner_arithmetic":b.owner.arithmetic_sha256,"producer_arithmetic":b.producer.arithmetic_sha256,"heads":b.owner.heads,"dim":b.owner.dim,"layout":registry.compatibility_layout_value(),"required_allocations":registry.allocations()}));
    }
    Ok(serde_json::json!(identities))
}
impl<'a> ModelFamilyAdmission<'a> {
    fn checked_frozen307(
        bindings: &'a [ModelFamilyBinding<'a>],
        stream: &'a NativeStream,
        expected: &serde_json::Value,
    ) -> Result<Self, String> {
        stream.validate()?;
        if &model_family_identity_frozen307(bindings)? != expected {
            return Err("complete graph family identity or inventory incomplete".into());
        }
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
}
