// Own exact allocations/parameters/kernels; no GPU data is cloned. Proofs still
// validate current family/source epochs before this immutable membership check.
#[derive(Clone)]
struct OwnedBoundProducerResourceMember {
    raw: Arc<NativeRawOwner>,
    prepared: Arc<NativeRawOwner>,
    a_log: Arc<GpuBuf<f32>>,
    bias: Arc<GpuBuf<f32>>,
    kernels: Arc<KdaRawKernels>,
    settings: (usize, usize, usize, usize, u64, u32, u32, u32),
    arithmetic: String,
}
impl OwnedBoundProducerResourceMember {
    fn settings(p: &KdaRawProducer) -> (usize, usize, usize, usize, u64, u32, u32, u32) {
        (
            p.heads,
            p.dim,
            p.capacity,
            p.input_capacity,
            p.source_epoch,
            p.lower.to_bits(),
            p.scale.to_bits(),
            p.epsilon.to_bits(),
        )
    }
    fn from_producer(p: &KdaRawProducer) -> Self {
        Self {
            raw: p.raw.clone(),
            prepared: p.prepared.clone(),
            a_log: p.a_log.clone(),
            bias: p.bias.clone(),
            kernels: p.kernels.clone(),
            settings: Self::settings(p),
            arithmetic: p.arithmetic_sha256.clone(),
        }
    }
    fn matches(&self, p: &KdaRawProducer) -> bool {
        Arc::ptr_eq(&self.raw, &p.raw)
            && Arc::ptr_eq(&self.prepared, &p.prepared)
            && Arc::ptr_eq(&self.a_log, &p.a_log)
            && Arc::ptr_eq(&self.bias, &p.bias)
            && Arc::ptr_eq(&self.kernels, &p.kernels)
            && self.settings == Self::settings(p)
            && self.arithmetic == p.arithmetic_sha256
    }
}
#[cfg(test)]
fn owned_producer_domain_lifetime_controls(
    root: &std::path::Path,
    source: &KdaRawProducer,
    reserve: usize,
) -> Result<(), String> {
    let stream = source.raw.stream.clone();
    let raw = Arc::new(NativeRawOwner::allocate(
        stream.clone(),
        source.raw.schema.clone(),
        reserve,
    )?);
    let a_bytes = shadow_read(
        pointer(&source.a_log),
        source.a_log.len().checked_mul(4).ok_or("owned A bytes")?,
    )?;
    let b_bytes = shadow_read(
        pointer(&source.bias),
        source.bias.len().checked_mul(4).ok_or("owned bias bytes")?,
    )?;
    let mut p = KdaRawProducer::from_live_raw_owner(
        raw.clone(),
        source.heads,
        source.dim,
        source.capacity,
        reserve,
        source.lower,
        source.scale,
        source.epsilon,
        source.kernels.clone(),
        &a_bytes,
        &b_bytes,
    )?;
    let weak_raw = Arc::downgrade(&p.raw);
    let weak_prepared = Arc::downgrade(&p.prepared);
    let weak_a = Arc::downgrade(&p.a_log);
    let weak_bias = Arc::downgrade(&p.bias);
    let domain = ProducerResourceDomain::checked(&[&p])?;
    let weak_registration = Arc::downgrade(&domain._registration);
    if !domain.producers[0].matches(&p) || ProducerResourceDomain::checked(&[&p]).is_ok() {
        return Err("owned resource original/duplicate member admission".into());
    }
    let mut rejected = Vec::new();
    let epoch = p.source_epoch;
    p.source_epoch = epoch.checked_add(1).ok_or("owned test epoch overflow")?;
    if domain.producers[0].matches(&p) {
        return Err("owned resource stale source epoch admitted".into());
    }
    rejected.push("source_epoch");
    p.source_epoch = epoch;
    let capacity = p.capacity;
    p.capacity = capacity
        .checked_add(1)
        .ok_or("owned test capacity overflow")?;
    if domain.producers[0].matches(&p) {
        return Err("owned resource changed capacity admitted".into());
    }
    rejected.push("capacity");
    p.capacity = capacity;
    let epsilon = p.epsilon;
    p.epsilon = f32::from_bits(epsilon.to_bits() ^ 1);
    if domain.producers[0].matches(&p) {
        return Err("owned resource changed epsilon admitted".into());
    }
    rejected.push("epsilon");
    p.epsilon = epsilon;
    let arithmetic = p.arithmetic_sha256.clone();
    p.arithmetic_sha256.push('0');
    if domain.producers[0].matches(&p) {
        return Err("owned resource changed arithmetic admitted".into());
    }
    rejected.push("arithmetic");
    p.arithmetic_sha256 = arithmetic;
    // Independently allocated preparation/parameters around the same actual raw
    // owner are different members even when scalar geometry is identical.
    let other = KdaRawProducer::from_live_raw_owner(
        raw.clone(),
        source.heads,
        source.dim,
        source.capacity,
        reserve,
        source.lower,
        source.scale,
        source.epsilon,
        source.kernels.clone(),
        &a_bytes,
        &b_bytes,
    )?;
    if domain.producers[0].matches(&other) {
        return Err("owned resource independent allocations admitted".into());
    }
    rejected.push("independent_preparation_parameters");
    let saved_a = p.a_log.clone();
    p.a_log = other.a_log.clone();
    if domain.producers[0].matches(&p) {
        return Err("owned resource changed A allocation admitted".into());
    }
    rejected.push("a_log_owner");
    p.a_log = saved_a;
    let saved_prepared = p.prepared.clone();
    p.prepared = other.prepared.clone();
    if domain.producers[0].matches(&p) {
        return Err("owned resource changed preparation allocation admitted".into());
    }
    rejected.push("prepared_owner");
    p.prepared = saved_prepared;
    let saved_kernels = p.kernels.clone();
    p.kernels = Arc::new(KdaRawKernels {
        device: saved_kernels.device.clone(),
        domain: saved_kernels.domain.clone(),
        prepare: saved_kernels.prepare.clone(),
        ptx_sha256: saved_kernels.ptx_sha256.clone(),
    });
    if domain.producers[0].matches(&p) {
        return Err("owned resource changed kernel owner admitted".into());
    }
    rejected.push("kernel_owner");
    p.kernels = saved_kernels;
    drop(other);
    let event = {
        let timeline = domain.groups[0]
            .timeline
            .try_lock()
            .map_err(|_| "owned resource timeline held")?;
        timeline.event as usize
    };
    for _ in 0..13 {
        if !domain.producers[0].matches(&p)
            || domain.groups[0]
                .timeline
                .try_lock()
                .map_err(|_| "owned resource repeat timeline held")?
                .event as usize
                != event
        {
            return Err("owned resource repeated membership changed event/domain".into());
        }
    }
    let raw_inventory = p.raw.inventory();
    let prepared_inventory = p.prepared.inventory();
    let a_pointer = pointer(&p.a_log);
    let b_pointer = pointer(&p.bias);
    drop(raw);
    drop(p);
    if weak_raw.upgrade().is_none()
        || weak_prepared.upgrade().is_none()
        || weak_a.upgrade().is_none()
        || weak_bias.upgrade().is_none()
    {
        return Err("owned resource domain lost producer allocation owners".into());
    }
    for (plane, ptr) in raw_inventory.iter().chain(&prepared_inventory) {
        native_pointer_admission(&stream, *ptr, plane.storage_bytes)?;
    }
    owner_compare(
        &shadow_read(a_pointer, a_bytes.len())?,
        &a_bytes,
        "owned resource A after producer drop",
    )?;
    owner_compare(
        &shadow_read(b_pointer, b_bytes.len())?,
        &b_bytes,
        "owned resource bias after producer drop",
    )?;
    let a_record = owner_save(root, "owned-a-after-producer-drop.f32", &a_bytes)?;
    let bias_record = owner_save(root, "owned-bias-after-producer-drop.f32", &b_bytes)?;
    drop(domain);
    if weak_raw.upgrade().is_some()
        || weak_prepared.upgrade().is_some()
        || weak_a.upgrade().is_some()
        || weak_bias.upgrade().is_some()
        || weak_registration.upgrade().is_some()
    {
        return Err("owned resource domain leaked allocation/registration owners".into());
    }
    owner_save(root, "OWNED_LIFETIME.json", &serde_json::to_vec_pretty(&serde_json::json!({
        "passed":true,"retained_after_producer_drop":true,"native_extent_admissions":raw_inventory.len()+prepared_inventory.len(),
        "parameter_bytes_exact":true,"parameters":[a_record,bias_record],"invalid_member_cases":rejected,
        "repeated_membership_checks":13,"same_event_across_membership_checks":true,
        "allocation_registration_weak_released":true,"teardown_field_order":"groups before retained producer members before registration",
        "scope":"owned domain lifetime/identity controls; graph/family/resource ordering separately renewed in same native fixture; no selected engine installation or speed claim"
    })).map_err(|e|e.to_string())?)?;
    Ok(())
}
