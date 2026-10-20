// Source lifecycle only. Rust's exclusive mutable borrow forbids outstanding
// domain proofs/admissions; an Arc-installed caller must obtain the same
// exclusivity before invoking this method. No decode-token call site.
impl ProducerResourceDomain {
    fn replace_checked(&mut self, producers: &[&KdaRawProducer]) -> Result<(), String> {
        {
            if producers.len() != self.producers.len() || producers.is_empty() {
                return Err("resource handoff complete ordered inventory required".into());
            }
            for (member, next) in self.producers.iter().zip(producers) {
                let old = member.bound()?;
                let settings = OwnedProducerResourceMember::settings(next);
                if settings.4
                    != old
                        .settings
                        .4
                        .checked_add(1)
                        .ok_or("resource handoff source epoch overflow")?
                    || settings.0 != old.settings.0
                    || settings.1 != old.settings.1
                    || settings.5 != old.settings.5
                    || settings.6 != old.settings.6
                    || settings.7 != old.settings.7
                    || !Arc::ptr_eq(&old.a_log, &next.a_log)
                    || !Arc::ptr_eq(&old.bias, &next.bias)
                    || !Arc::ptr_eq(&old.kernels, &next.kernels)
                    || old.arithmetic != next.arithmetic_sha256
                {
                    return Err(
                        "resource handoff stale epoch or changed ordered member contract".into(),
                    );
                }
            }
            let mut old = Vec::with_capacity(self.groups.len());
            for group in &self.groups {
                let timeline = group
                    .timeline
                    .try_lock()
                    .map_err(|_| "resource handoff old timeline held/poisoned")?;
                if timeline.poisoned {
                    return Err("resource handoff old domain poisoned".into());
                }
                for stream in &timeline.streams {
                    stream.validate()?;
                }
                old.push(timeline);
            }
            // A prospective domain retains every allocation and owns all new
            // completion events. Registration excludes only our exact token;
            // every other live overlap is rejected. No old membership changes.
            let next = Self::build_checked(producers, Some(&self._registration))?;
            let mut failure = None;
            for timeline in &mut old {
                if let Err(e) = timeline.device.bind_to_thread().map_err(|e| e.to_string()) {
                    failure.get_or_insert(e);
                    continue;
                }
                if timeline.recorded {
                    if let Err(e) = check(
                        unsafe { sys::lib().cuEventSynchronize(timeline.event) },
                        "resource handoff old completion wait",
                    ) {
                        failure.get_or_insert(e);
                    }
                }
                for stream in &timeline.streams {
                    if let Err(e) = stream.drain() {
                        failure.get_or_insert(e);
                    }
                }
            }
            if let Some(e) = failure {
                for timeline in &mut old {
                    timeline.poisoned = true;
                }
                return Err(format!(
                    "resource handoff drain failed; ALL old groups poisoned:{e}"
                ));
            }
            // All old work completed. Release timeline guards before swapping;
            // exclusive self access prevents another submission in this gap.
            drop(old);
            let retired = std::mem::replace(self, next);
            // Groups drop first, while their members and registration live.
            drop(retired);
        }
        Ok(())
    }
}
#[cfg(test)]
fn resource_epoch_handoff_controls(
    root: &Path,
    source: &KdaRawProducer,
    reserve: usize,
) -> Result<(), String> {
    let stream = source.raw.stream.clone();
    let old_raw = Arc::new(NativeRawOwner::allocate(
        stream.clone(),
        source.raw.schema.clone(),
        reserve,
    )?);
    let new_raw = Arc::new(NativeRawOwner::allocate(
        stream.clone(),
        source.raw.schema.clone(),
        reserve,
    )?);
    let a_bytes = shadow_read(
        pointer(&source.a_log),
        source.a_log.len().checked_mul(4).ok_or("handoff A bytes")?,
    )?;
    let bias_bytes = shadow_read(
        pointer(&source.bias),
        source
            .bias
            .len()
            .checked_mul(4)
            .ok_or("handoff bias bytes")?,
    )?;
    let old_producer = KdaRawProducer::from_live_raw_owner(
        old_raw.clone(),
        source.heads,
        source.dim,
        source.capacity,
        reserve,
        source.lower,
        source.scale,
        source.epsilon,
        source.kernels.clone(),
        &a_bytes,
        &bias_bytes,
    )?;
    let mut next_producer = KdaRawProducer {
        raw: new_raw.clone(),
        prepared: old_producer.prepared.clone(),
        a_log: old_producer.a_log.clone(),
        bias: old_producer.bias.clone(),
        kernels: old_producer.kernels.clone(),
        heads: old_producer.heads,
        dim: old_producer.dim,
        capacity: old_producer.capacity,
        input_capacity: old_producer.input_capacity,
        source_epoch: old_producer
            .source_epoch
            .checked_add(1)
            .ok_or("handoff epoch overflow")?,
        lower: old_producer.lower,
        scale: old_producer.scale,
        epsilon: old_producer.epsilon,
        parameter_admissions: vec![
            native_pointer_admission(
                &stream,
                pointer(&old_producer.a_log),
                old_producer
                    .a_log
                    .len()
                    .checked_mul(4)
                    .ok_or("handoff parameter A extent")?,
            )?,
            native_pointer_admission(
                &stream,
                pointer(&old_producer.bias),
                old_producer
                    .bias
                    .len()
                    .checked_mul(4)
                    .ok_or("handoff parameter bias extent")?,
            )?,
        ],
        requested_bytes: old_producer.requested_bytes,
        allocated_bytes: old_producer.allocated_bytes,
        arithmetic_sha256: old_producer.arithmetic_sha256.clone(),
    };
    let mut domain = ProducerResourceDomain::checked(&[&old_producer])?;
    let weak_raw = Arc::downgrade(&old_raw);
    let weak_registration = Arc::downgrade(&domain._registration);
    let prepared = Arc::downgrade(&old_producer.prepared);
    let old_epoch = old_producer.source_epoch;
    let new_epoch = next_producer.source_epoch;
    if domain.replace_checked(&[]).is_ok() || !domain.producers[0].matches(&old_producer) {
        return Err("handoff empty inventory changed old domain".into());
    }
    let retained_token = domain._registration.clone();
    if domain.replace_checked(&[&next_producer]).is_ok()
        || !domain.producers[0].matches(&old_producer)
    {
        return Err("handoff retained registration admitted".into());
    }
    drop(retained_token);
    next_producer.source_epoch = old_epoch;
    if domain.replace_checked(&[&next_producer]).is_ok()
        || !domain.producers[0].matches(&old_producer)
    {
        return Err("handoff stale replacement epoch admitted".into());
    }
    next_producer.source_epoch = new_epoch;
    let external_raw = new_raw.clone();
    let external = KdaRawProducer::from_live_raw_owner(
        external_raw,
        source.heads,
        source.dim,
        source.capacity,
        reserve,
        source.lower,
        source.scale,
        source.epsilon,
        source.kernels.clone(),
        &a_bytes,
        &bias_bytes,
    )?;
    let external_domain = ProducerResourceDomain::checked(&[&external])?;
    if domain.replace_checked(&[&next_producer]).is_ok()
        || !domain.producers[0].matches(&old_producer)
        || weak_registration.upgrade().is_none()
    {
        return Err("handoff third-party overlap altered old ownership".into());
    }
    drop(external_domain);
    drop(external);
    let references = source
        .raw
        .inventory()
        .iter()
        .map(|(plane, ptr)| shadow_read(*ptr, plane.storage_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    {
        let mut timeline = domain.groups[0]
            .timeline
            .try_lock()
            .map_err(|_| "handoff test timeline held")?;
        resource_copy_input(&stream, &source.raw, &old_raw)?;
        resource_copy_input(&stream, &old_raw, &new_raw)?;
        check(
            unsafe { sys::lib().cuEventRecord(timeline.event, stream.handle) },
            "handoff test queued completion",
        )?;
        timeline.recorded = true;
        timeline.phases = 1;
    }
    drop(old_raw);
    drop(old_producer);
    if weak_raw.upgrade().is_none() {
        return Err("handoff old raw released before completion".into());
    }
    domain.replace_checked(&[&next_producer])?;
    if weak_raw.upgrade().is_some()
        || weak_registration.upgrade().is_some()
        || prepared.upgrade().is_none()
        || !domain.producers[0].matches(&next_producer)
    {
        return Err("handoff owner release/new membership".into());
    }
    let mut stale = KdaRawProducer {
        raw: next_producer.raw.clone(),
        prepared: next_producer.prepared.clone(),
        a_log: next_producer.a_log.clone(),
        bias: next_producer.bias.clone(),
        kernels: next_producer.kernels.clone(),
        heads: next_producer.heads,
        dim: next_producer.dim,
        capacity: next_producer.capacity,
        input_capacity: next_producer.input_capacity,
        source_epoch: old_epoch,
        lower: next_producer.lower,
        scale: next_producer.scale,
        epsilon: next_producer.epsilon,
        parameter_admissions: vec![
            native_pointer_admission(
                &stream,
                pointer(&next_producer.a_log),
                next_producer
                    .a_log
                    .len()
                    .checked_mul(4)
                    .ok_or("handoff parameter A extent")?,
            )?,
            native_pointer_admission(
                &stream,
                pointer(&next_producer.bias),
                next_producer
                    .bias
                    .len()
                    .checked_mul(4)
                    .ok_or("handoff parameter bias extent")?,
            )?,
        ],
        requested_bytes: next_producer.requested_bytes,
        allocated_bytes: next_producer.allocated_bytes,
        arithmetic_sha256: next_producer.arithmetic_sha256.clone(),
    };
    if domain.producers[0].matches(&stale) {
        return Err("handoff stale source admitted".into());
    }
    stale.source_epoch = new_epoch;
    if !domain.producers[0].matches(&stale) {
        return Err("handoff exact shared owners rejected".into());
    }
    let event = domain.groups[0]
        .timeline
        .try_lock()
        .map_err(|_| "handoff new timeline held")?
        .event as usize;
    for _ in 0..13 {
        if !domain.producers[0].matches(&next_producer)
            || domain.groups[0]
                .timeline
                .try_lock()
                .map_err(|_| "handoff repeat held")?
                .event as usize
                != event
        {
            return Err("handoff event changed across memberships".into());
        }
    }
    let mut records = Vec::new();
    for ((plane, ptr), bytes) in new_raw.inventory().iter().zip(&references) {
        let got = shadow_read(*ptr, plane.storage_bytes)?;
        owner_compare(&got, bytes, "handoff complete queued raw plane")?;
        records.push(owner_save(
            root,
            &format!("handoff-{}.bin", plane.role),
            &got,
        )?);
    }
    owner_compare(
        &shadow_read(pointer(&next_producer.a_log), a_bytes.len())?,
        &a_bytes,
        "handoff retained A",
    )?;
    owner_compare(
        &shadow_read(pointer(&next_producer.bias), bias_bytes.len())?,
        &bias_bytes,
        "handoff retained bias",
    )?;
    let new_registration = Arc::downgrade(&domain._registration);
    drop(stale);
    drop(next_producer);
    drop(new_raw);
    if prepared.upgrade().is_none() {
        return Err("handoff next membership lost prepared owner".into());
    }
    drop(domain);
    if prepared.upgrade().is_some() || new_registration.upgrade().is_some() {
        return Err("handoff new domain leaked owners".into());
    }
    owner_save(root,"EPOCH_HANDOFF.json",&serde_json::to_vec_pretty(&json!({"passed":true,"old_epoch":old_epoch,"new_epoch":new_epoch,"negative_cases":["empty_inventory","retained_old_registration","third_party_overlap","stale_replacement_epoch","stale_member_epoch"],"old_raw_retained_before_drain":true,"old_raw_and_registration_released_after_drain":true,"prepared_parameters_retained_exact":true,"complete_queued_raw_planes":records,"repeated_membership_checks":13,"new_event_stable":true,"weak_owners_released":true,"scope":"exclusive domain handoff fixture; no installed current-storage/session or selected execution"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
