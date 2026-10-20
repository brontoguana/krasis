// Current shared storage: required domain lifecycle; no optional bypass.
struct KdaCurrentResourceTransition<'a> {
    next: Option<ProducerResourceDomain>,
    old_groups: Vec<std::sync::MutexGuard<'a, StateTimeline>>,
}
impl<'a> KdaCurrentResourceTransition<'a> {
    fn checked(
        domain: &'a ProducerResourceDomain,
        members: Vec<OwnedProducerResourceMember>,
        streams: &[Arc<NativeStream>],
    ) -> Result<Self, String> {
        if members.len() != domain.producers.len() {
            return Err("current source domain incomplete prospective roles".into());
        }
        let mut old_groups = Vec::with_capacity(domain.groups.len());
        for group in &domain.groups {
            let timeline = group
                .timeline
                .try_lock()
                .map_err(|_| "current source resource held/poisoned")?;
            if timeline.poisoned {
                return Err("current source resource poisoned".into());
            }
            for stream in &timeline.streams {
                stream.validate()?;
            }
            old_groups.push(timeline);
        }
        let mut next = ProducerResourceDomain::build_members(members, Some(&domain._registration))?;
        current_domain_admit_streams(&mut next, streams)?;
        Ok(Self {
            next: Some(next),
            old_groups,
        })
    }
    fn execute<T>(&mut self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        for timeline in &mut self.old_groups {
            timeline.poisoned = true;
        }
        let mut failure = None;
        for t in &mut self.old_groups {
            if let Err(e) = t.device.bind_to_thread().map_err(|e| e.to_string()) {
                failure.get_or_insert(e);
                continue;
            }
            if t.recorded {
                if let Err(e) = check(
                    unsafe { sys::lib().cuEventSynchronize(t.event) },
                    "current source resource prior completion",
                ) {
                    failure.get_or_insert(e);
                }
            }
            for s in &t.streams {
                if let Err(e) = s.drain() {
                    failure.get_or_insert(e);
                }
            }
        }
        if let Some(e) = failure {
            for t in &mut self.old_groups {
                t.poisoned = true;
            }
            return Err(format!("current source complete resource drain failed:{e}"));
        }
        let result = work();
        if result.is_err() {
            for t in &mut self.old_groups {
                t.poisoned = true;
            }
        }
        result
    }
    fn finish(mut self) -> Result<ProducerResourceDomain, String> {
        self.next
            .take()
            .ok_or_else(|| "current source prospective domain missing".to_string())
    }
}
fn current_domain_admit_streams(
    domain: &mut ProducerResourceDomain,
    streams: &[Arc<NativeStream>],
) -> Result<(), String> {
    if streams.is_empty() {
        return Err("current domain stream inventory empty".into());
    }
    for stream in streams {
        stream.validate()?;
        if !Arc::ptr_eq(&stream.device, &domain._registration._device) {
            return Err("current domain foreign participating stream".into());
        }
    }
    for group in &domain.groups {
        let mut t = group
            .timeline
            .try_lock()
            .map_err(|_| "prospective current resource held")?;
        for stream in streams {
            if !t.streams.iter().any(|s| s.handle == stream.handle) {
                t.streams.push(stream.clone());
            }
        }
    }
    Ok(())
}
fn current_lifetime_streams(
    life: &KdaPrefillStorageLifetime,
) -> Result<Vec<Arc<NativeStream>>, String> {
    let owners = match &life.state {
        KdaPrefillStorageState::Bound(f) => f
            .iter()
            .map(|f| (&f.prefill, &f.decode))
            .collect::<Vec<_>>(),
        KdaPrefillStorageState::Retired(f) => f
            .iter()
            .map(|f| (&f.prefill, &f.decode))
            .collect::<Vec<_>>(),
        _ => return Err("current domain lifetime poisoned".into()),
    };
    let mut streams = Vec::new();
    for (a, b) in owners {
        for s in [&a.stream, &b.stream] {
            s.validate()?;
            if !streams
                .iter()
                .any(|v: &Arc<NativeStream>| v.handle == s.handle)
            {
                streams.push(s.clone());
            }
        }
    }
    Ok(streams)
}
fn current_resource_members(
    life: &KdaPrefillStorageLifetime,
) -> Result<Vec<OwnedProducerResourceMember>, String> {
    let mut members = Vec::new();
    match &life.state {
        KdaPrefillStorageState::Bound(f) => {
            for f in f {
                members.push(OwnedProducerResourceMember::from_producer(
                    &f.prefill_producer,
                ));
                members.push(OwnedProducerResourceMember::from_producer(
                    &f.decode_producer,
                ));
            }
        }
        KdaPrefillStorageState::Retired(f) => {
            for f in f {
                let p = &f.preparation;
                members.push(OwnedProducerResourceMember::RetiredPreparation(
                    OwnedRetiredPreparationResource {
                        prepared: p.prepared.clone(),
                        a_log: p.a_log.clone(),
                        bias: p.bias.clone(),
                        kernels: p.kernels.clone(),
                        heads: p.heads,
                        dim: p.dim,
                        capacity: p.capacity,
                        source_epoch: p.source_epoch,
                        lower_bits: p.lower.to_bits(),
                        scale_bits: p.scale.to_bits(),
                        epsilon_bits: p.epsilon.to_bits(),
                        arithmetic: p.arithmetic_sha256.clone(),
                    },
                ));
                members.push(OwnedProducerResourceMember::from_producer(
                    &f.decode_producer,
                ));
            }
        }
        _ => return Err("current resource lifetime poisoned/extracted".into()),
    }
    Ok(members)
}
fn current_domain_audit(domain: &ProducerResourceDomain) -> Result<serde_json::Value, String> {
    let mut groups = Vec::new();
    for g in &domain.groups {
        let t = g
            .timeline
            .try_lock()
            .map_err(|_| "current domain audit held/poisoned")?;
        groups.push(serde_json::json!({"event":t.event as usize,"recorded":t.recorded,"phases":t.phases,"poisoned":t.poisoned,"streams":t.streams.iter().map(|s|s.handle as usize).collect::<Vec<_>>()}));
    }
    let roles=domain.producers.iter().enumerate().map(|(index,m)|match m{OwnedProducerResourceMember::Bound(b)=>serde_json::json!({"index":index,"phase":"bound","source_epoch":b.settings.4,"raw":b.raw.inventory(),"prepared":b.prepared.inventory(),"a_log":pointer(&b.a_log),"bias":pointer(&b.bias),"group":domain.producer_groups[index]}),OwnedProducerResourceMember::RetiredPreparation(r)=>serde_json::json!({"index":index,"phase":"retired_preparation","source_epoch":r.source_epoch,"raw_capability":false,"prepared":r.prepared.inventory(),"a_log":pointer(&r.a_log),"bias":pointer(&r.bias),"group":domain.producer_groups[index]})}).collect::<Vec<_>>();
    Ok(
        serde_json::json!({"roles":roles,"groups":groups,"workspaces":domain._registration.workspaces.iter().map(|w|serde_json::json!({"context":w.context,"spans":w.spans})).collect::<Vec<_>>(),"retained_between_submissions":true}),
    )
}
impl KdaPrefillStorageLifetime {
    fn retirement_resource_members(
        &self,
        domain: &ProducerResourceDomain,
        epochs: &[u64],
    ) -> Result<Vec<OwnedProducerResourceMember>, String> {
        let current = current_resource_members(self)?;
        if current.len() != domain.producers.len() {
            return Err("current retire domain role count".into());
        }
        let families = self.bound()?;
        if epochs.len() != families.len() {
            return Err("current retire epoch inventory".into());
        }
        // Owner counts derive from the complete lifetime/domain roles. Reject an
        // external raw-wrapper lease before modifying either lifetime or domain.
        let mut counts = std::collections::BTreeMap::<usize, usize>::new();
        for f in families {
            for p in [&f.prefill_producer, &f.decode_producer] {
                *counts.entry(Arc::as_ptr(&p.raw) as usize).or_default() += 1;
            }
        }
        for m in &domain.producers {
            if let OwnedProducerResourceMember::Bound(b) = m {
                *counts.entry(Arc::as_ptr(&b.raw) as usize).or_default() += 1;
            }
        }
        // The prospective `current` snapshots also retain each Bound raw once.
        for m in &current {
            if let OwnedProducerResourceMember::Bound(b) = m {
                *counts.entry(Arc::as_ptr(&b.raw) as usize).or_default() += 1;
            }
        }
        for f in families {
            let raw = &f.prefill_producer.raw;
            let expected = counts
                .get(&(Arc::as_ptr(raw) as usize))
                .copied()
                .ok_or("current raw owner accounting")?;
            if Arc::strong_count(raw) != expected {
                return Err("current prefill raw source still leased externally".into());
            }
        }
        let mut next = Vec::with_capacity(current.len());
        let mut index = 0;
        for (f, &epoch) in families.iter().zip(epochs) {
            if !domain.producers[index].matches(&f.prefill_producer)
                || !domain.producers[index + 1].matches(&f.decode_producer)
            {
                return Err("current retirement stale ordered member".into());
            }
            next.push(OwnedProducerResourceMember::RetiredPreparation(
                OwnedRetiredPreparationResource::from_bound(
                    domain.producers[index].bound()?,
                    epoch,
                )?,
            ));
            next.push(domain.producers[index + 1].clone());
            index += 2;
        }
        Ok(next)
    }
    fn attachment_resource_members(
        &self,
        domain: &ProducerResourceDomain,
        plan: &KdaAttachmentPlan,
    ) -> Result<Vec<OwnedProducerResourceMember>, String> {
        let families = match &self.state {
            KdaPrefillStorageState::Retired(f) => f,
            _ => return Err("current attach needs retired lifetime".into()),
        };
        if families.len() != plan.sources_rows_costs_epochs.len()
            || domain.producers.len()
                != families
                    .len()
                    .checked_mul(2)
                    .ok_or("current attach role extent")?
        {
            return Err("current attach incomplete ordered roles".into());
        }
        let mut next = Vec::with_capacity(domain.producers.len());
        let mut index = 0;
        for (f, (source, rows, _, epoch)) in families.iter().zip(&plan.sources_rows_costs_epochs) {
            let p = &f.preparation;
            let old = match &domain.producers[index] {
                OwnedProducerResourceMember::RetiredPreparation(r) => r,
                _ => return Err("current attach domain raw capability still bound".into()),
            };
            if old.source_epoch != p.source_epoch
                || *epoch
                    != old
                        .source_epoch
                        .checked_add(1)
                        .ok_or("current attach epoch overflow")?
                || !Arc::ptr_eq(&old.prepared, &p.prepared)
                || !Arc::ptr_eq(&old.a_log, &p.a_log)
                || !Arc::ptr_eq(&old.bias, &p.bias)
                || !Arc::ptr_eq(&old.kernels, &p.kernels)
                || old.arithmetic != p.arithmetic_sha256
                || !domain.producers[index + 1].matches(&f.decode_producer)
            {
                return Err("current attach stale preparation/decode contract".into());
            }
            next.push(OwnedProducerResourceMember::Bound(
                OwnedBoundProducerResourceMember {
                    raw: source.inner.clone(),
                    prepared: p.prepared.clone(),
                    a_log: p.a_log.clone(),
                    bias: p.bias.clone(),
                    kernels: p.kernels.clone(),
                    settings: (
                        p.heads,
                        p.dim,
                        p.capacity,
                        *rows,
                        *epoch,
                        p.lower.to_bits(),
                        p.scale.to_bits(),
                        p.epsilon.to_bits(),
                    ),
                    arithmetic: p.arithmetic_sha256.clone(),
                },
            ));
            next.push(domain.producers[index + 1].clone());
            index += 2;
        }
        Ok(next)
    }
}
fn current_storage_poison(state: &mut KdaEngineRequestStorageState) {
    state.poisoned = true;
    for group in &state.resource_domain.groups {
        if let Ok(mut t) = group.timeline.try_lock() {
            t.poisoned = true;
        }
    }
    let owners = match &state.lifetime.state {
        KdaPrefillStorageState::Bound(f) => f
            .iter()
            .map(|f| (&f.prefill, &f.decode))
            .collect::<Vec<_>>(),
        KdaPrefillStorageState::Retired(f) => f
            .iter()
            .map(|f| (&f.prefill, &f.decode))
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    for (a, b) in owners {
        for owner in [a, b] {
            if let Ok(mut slot) = owner.state.timeline.try_lock() {
                if let Some(t) = slot.as_mut() {
                    t.poisoned = true;
                }
            }
        }
    }
}
fn current_storage_retire(
    state: &mut KdaEngineRequestStorageState,
    expected: &serde_json::Value,
    epochs: &[u64],
) -> Result<(Vec<u64>, usize), String> {
    let generation = state
        .domain_generation
        .checked_add(1)
        .ok_or("current domain generation overflow")?;
    let plan = state.lifetime.prepare_retirement_plan(expected, epochs)?;
    let members = state
        .lifetime
        .retirement_resource_members(&state.resource_domain, epochs)?;
    let weak_sources = state
        .lifetime
        .bound()?
        .iter()
        .map(|f| {
            let raw = &f.prefill_producer.raw;
            let retained_by_decode = state
                .lifetime
                .bound()?
                .iter()
                .any(|g| Arc::ptr_eq(&g.decode_producer.raw, raw));
            Ok((Arc::downgrade(raw), retained_by_decode))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let streams = current_lifetime_streams(&state.lifetime)?;
    let owners = state
        .lifetime
        .bound()?
        .iter()
        .map(|f| (&f.prefill, &f.decode))
        .collect::<Vec<_>>();
    let witnesses = KdaQuiescentStorageWitnesses::checked(&owners)?;
    let started = std::cell::Cell::new(false);
    let result = {
        let lifetime = &mut state.lifetime;
        let domain = &state.resource_domain;
        let mut transition = KdaCurrentResourceTransition::checked(domain, members, &streams)?;
        witnesses
            .execute_marked(&started, || {
                transition.execute(|| lifetime.apply_retirement_plan(plan))
            })
            .and_then(|epochs| transition.finish().map(|domain| (epochs, domain)))
    };
    match result {
        Ok((epochs, domain)) => {
            let old = std::mem::replace(&mut state.resource_domain, domain);
            drop(old);
            state.domain_generation = generation;
            if weak_sources
                .iter()
                .any(|(w, retained)| !retained && w.upgrade().is_some())
            {
                current_storage_poison(state);
                return Err("current retired prefill raw wrapper remained leased after complete domain commit".into());
            }
            Ok((
                epochs,
                weak_sources
                    .iter()
                    .filter(|(_, retained)| *retained)
                    .count(),
            ))
        }
        Err(e) => {
            if started.get() {
                current_storage_poison(state);
            }
            Err(e)
        }
    }
}
fn current_storage_attach(
    state: &mut KdaEngineRequestStorageState,
    sources: Vec<RetainedPrefillKdaSource>,
    epochs: &[u64],
) -> Result<Vec<u64>, String> {
    let generation = state
        .domain_generation
        .checked_add(1)
        .ok_or("current domain generation overflow")?;
    let plan = state.lifetime.prepare_attachment_plan(sources, epochs)?;
    let members = state
        .lifetime
        .attachment_resource_members(&state.resource_domain, &plan)?;
    let streams = current_lifetime_streams(&state.lifetime)?;
    let families = match &state.lifetime.state {
        KdaPrefillStorageState::Retired(f) => f,
        _ => return Err("current attach needs retired storage".into()),
    };
    let owners = families
        .iter()
        .map(|f| (&f.prefill, &f.decode))
        .collect::<Vec<_>>();
    let witnesses = KdaQuiescentStorageWitnesses::checked(&owners)?;
    let started = std::cell::Cell::new(false);
    let result = {
        let lifetime = &mut state.lifetime;
        let domain = &state.resource_domain;
        let mut transition = KdaCurrentResourceTransition::checked(domain, members, &streams)?;
        witnesses
            .execute_marked(&started, || {
                transition.execute(|| lifetime.apply_attachment_plan(plan))
            })
            .and_then(|epochs| transition.finish().map(|domain| (epochs, domain)))
    };
    match result {
        Ok((epochs, domain)) => {
            let old = std::mem::replace(&mut state.resource_domain, domain);
            drop(old);
            state.domain_generation = generation;
            Ok(epochs)
        }
        Err(e) => {
            if started.get() {
                current_storage_poison(state);
            }
            Err(e)
        }
    }
}
