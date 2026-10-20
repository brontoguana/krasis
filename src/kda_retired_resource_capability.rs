// Typed preparation-only capability: no raw allocation or execution producer.
#[derive(Clone)]
struct OwnedRetiredPreparationResource {
    prepared: Arc<NativeRawOwner>,
    a_log: Arc<GpuBuf<f32>>,
    bias: Arc<GpuBuf<f32>>,
    kernels: Arc<KdaRawKernels>,
    heads: usize,
    dim: usize,
    capacity: usize,
    source_epoch: u64,
    lower_bits: u32,
    scale_bits: u32,
    epsilon_bits: u32,
    arithmetic: String,
}
#[derive(Clone)]
enum OwnedProducerResourceMember {
    Bound(OwnedBoundProducerResourceMember),
    RetiredPreparation(OwnedRetiredPreparationResource),
}
impl OwnedProducerResourceMember {
    fn settings(p: &KdaRawProducer) -> (usize, usize, usize, usize, u64, u32, u32, u32) {
        OwnedBoundProducerResourceMember::settings(p)
    }
    fn from_producer(p: &KdaRawProducer) -> Self {
        Self::Bound(OwnedBoundProducerResourceMember::from_producer(p))
    }
    fn bound(&self) -> Result<&OwnedBoundProducerResourceMember, String> {
        match self {
            Self::Bound(p) => Ok(p),
            Self::RetiredPreparation(_) => {
                Err("retired preparation has no raw producer capability".into())
            }
        }
    }
    fn matches(&self, p: &KdaRawProducer) -> bool {
        match self {
            Self::Bound(b) => b.matches(p),
            Self::RetiredPreparation(_) => false,
        }
    }
    fn prepared(&self) -> &Arc<NativeRawOwner> {
        match self {
            Self::Bound(b) => &b.prepared,
            Self::RetiredPreparation(r) => &r.prepared,
        }
    }
    fn device(&self) -> &Arc<CudaDevice> {
        &self.prepared().device
    }
    fn stream(&self) -> &Arc<NativeStream> {
        &self.prepared().stream
    }
    fn checked_workspace(
        &self,
        device: &Arc<CudaDevice>,
    ) -> Result<ProducerWorkspaceSpans, String> {
        self.stream().validate()?;
        if !Arc::ptr_eq(self.device(), device) {
            return Err("typed resource foreign device".into());
        }
        let (heads, dim, capacity, a, b, k) = match self {
            Self::Bound(p) => {
                p.raw.stream.validate()?;
                if !Arc::ptr_eq(&p.raw.device, device)
                    || p.raw.stream.handle != self.stream().handle
                    || p.raw.schema
                        != Schema::checked(
                            RawContract::KdaBf16VectorGate,
                            p.settings.3,
                            p.settings.0,
                            p.settings.1,
                            p.settings.1,
                        )?
                {
                    return Err("typed bound raw contract/context/stream".into());
                }
                (
                    p.settings.0,
                    p.settings.1,
                    p.settings.2,
                    &p.a_log,
                    &p.bias,
                    &p.kernels,
                )
            }
            Self::RetiredPreparation(p) => {
                (p.heads, p.dim, p.capacity, &p.a_log, &p.bias, &p.kernels)
            }
        };
        let prep = self.prepared();
        if prep.schema
            != Schema::checked(
                RawContract::KdaPreparedF32VectorGate,
                capacity,
                heads,
                dim,
                dim,
            )?
            || a.len() != heads
            || b.len() != heads.checked_mul(dim).ok_or("typed bias geometry")?
            || !Arc::ptr_eq(&k.device, device)
        {
            return Err("typed preparation/parameter/kernel contract".into());
        }
        let mut spans = Vec::new();
        if let Self::Bound(p) = self {
            for (plane, ptr) in p.raw.inventory() {
                native_pointer_admission(self.stream(), ptr, plane.storage_bytes)?;
                spans.push((ptr, plane.storage_bytes));
            }
        }
        for (plane, ptr) in prep.inventory() {
            native_pointer_admission(self.stream(), ptr, plane.storage_bytes)?;
            spans.push((ptr, plane.storage_bytes));
        }
        native_pointer_admission(
            self.stream(),
            pointer(a),
            a.len().checked_mul(4).ok_or("typed A extent")?,
        )?;
        native_pointer_admission(
            self.stream(),
            pointer(b),
            b.len().checked_mul(4).ok_or("typed bias extent")?,
        )?;
        Ok(ProducerWorkspaceSpans {
            context: *device.cu_primary_ctx() as usize,
            spans,
        })
    }
}
impl OwnedRetiredPreparationResource {
    fn from_bound(b: &OwnedBoundProducerResourceMember, epoch: u64) -> Result<Self, String> {
        if b.settings.4 != epoch {
            return Err("retired resource stale source epoch".into());
        }
        Ok(Self {
            prepared: b.prepared.clone(),
            a_log: b.a_log.clone(),
            bias: b.bias.clone(),
            kernels: b.kernels.clone(),
            heads: b.settings.0,
            dim: b.settings.1,
            capacity: b.settings.2,
            source_epoch: epoch
                .checked_add(1)
                .ok_or("retired source epoch overflow")?,
            lower_bits: b.settings.5,
            scale_bits: b.settings.6,
            epsilon_bits: b.settings.7,
            arithmetic: b.arithmetic.clone(),
        })
    }
    fn admits_next(&self, p: &KdaRawProducer) -> Result<(), String> {
        if p.source_epoch
            != self
                .source_epoch
                .checked_add(1)
                .ok_or("reattach source epoch overflow")?
            || p.heads != self.heads
            || p.dim != self.dim
            || p.capacity != self.capacity
            || p.lower.to_bits() != self.lower_bits
            || p.scale.to_bits() != self.scale_bits
            || p.epsilon.to_bits() != self.epsilon_bits
            || p.arithmetic_sha256 != self.arithmetic
            || !Arc::ptr_eq(&p.prepared, &self.prepared)
            || !Arc::ptr_eq(&p.a_log, &self.a_log)
            || !Arc::ptr_eq(&p.bias, &self.bias)
            || !Arc::ptr_eq(&p.kernels, &self.kernels)
        {
            return Err("retired resource changed preparation or stale attachment epoch".into());
        }
        Ok(())
    }
}
impl ProducerResourceDomain {
    // Cold source transitions only. Complete family/arena transactions belong
    // to the caller's current-storage coordinator, not this domain primitive.
    fn commit_typed_members(
        &mut self,
        members: Vec<OwnedProducerResourceMember>,
    ) -> Result<(), String> {
        if members.len() != self.producers.len() || members.is_empty() {
            return Err("typed source complete domain inventory required".into());
        }
        let mut guards = Vec::with_capacity(self.groups.len());
        for g in &self.groups {
            let t = g
                .timeline
                .try_lock()
                .map_err(|_| "typed source group held/poisoned")?;
            if t.poisoned {
                return Err("typed source domain poisoned".into());
            }
            for s in &t.streams {
                s.validate()?;
            }
            guards.push(t);
        }
        let next = Self::build_members(members, Some(&self._registration))?;
        let mut failure = None;
        for t in &mut guards {
            if let Err(e) = t.device.bind_to_thread().map_err(|e| e.to_string()) {
                failure.get_or_insert(e);
                continue;
            }
            if t.recorded {
                if let Err(e) = check(
                    unsafe { sys::lib().cuEventSynchronize(t.event) },
                    "typed source prior completion",
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
            for t in &mut guards {
                t.poisoned = true;
            }
            return Err(format!(
                "typed source drain failed; ALL resource groups poisoned:{e}"
            ));
        }
        drop(guards);
        let old = std::mem::replace(self, next);
        drop(old);
        Ok(())
    }
    fn retire_members_checked(&mut self, indices: &[(usize, u64)]) -> Result<(), String> {
        if indices.is_empty() {
            return Err("retired resource transition empty".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut next = self.producers.clone();
        for &(index, epoch) in indices {
            if !seen.insert(index) {
                return Err("retired resource duplicate role".into());
            }
            let b = self
                .producers
                .get(index)
                .ok_or("retired resource missing role")?
                .bound()?;
            next[index] = OwnedProducerResourceMember::RetiredPreparation(
                OwnedRetiredPreparationResource::from_bound(b, epoch)?,
            );
        }
        self.commit_typed_members(next)
    }
    fn attach_members_checked(
        &mut self,
        updates: &[(usize, &KdaRawProducer)],
    ) -> Result<(), String> {
        if updates.is_empty() {
            return Err("attached resource transition empty".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut next = self.producers.clone();
        for &(index, p) in updates {
            if !seen.insert(index) {
                return Err("attached resource duplicate role".into());
            }
            let old = match self
                .producers
                .get(index)
                .ok_or("attached resource missing role")?
            {
                OwnedProducerResourceMember::RetiredPreparation(r) => r,
                _ => return Err("attached resource needs retired preparation".into()),
            };
            old.admits_next(p)?;
            next[index] = OwnedProducerResourceMember::from_producer(p);
        }
        self.commit_typed_members(next)
    }
}
#[cfg(test)]
fn retired_resource_capability_controls(
    root: &Path,
    source: &KdaRawProducer,
    reserve: usize,
) -> Result<(), String> {
    let stream = source.raw.stream.clone();
    let a = shadow_read(
        pointer(&source.a_log),
        source
            .a_log
            .len()
            .checked_mul(4)
            .ok_or("retired fixture A")?,
    )?;
    let b = shadow_read(
        pointer(&source.bias),
        source
            .bias
            .len()
            .checked_mul(4)
            .ok_or("retired fixture bias")?,
    )?;
    let allocate = || {
        NativeRawOwner::allocate(stream.clone(), source.raw.schema.clone(), reserve).map(Arc::new)
    };
    let producer = |raw: Arc<NativeRawOwner>| {
        KdaRawProducer::from_live_raw_owner(
            raw,
            source.heads,
            source.dim,
            source.capacity,
            reserve,
            source.lower,
            source.scale,
            source.epsilon,
            source.kernels.clone(),
            &a,
            &b,
        )
    };
    let pref_raw = allocate()?;
    let decode_raw = allocate()?;
    let new_raw = allocate()?;
    let pref = producer(pref_raw.clone())?;
    let decode = producer(decode_raw.clone())?;
    let original_epoch = pref.source_epoch;
    let weak_raw = Arc::downgrade(&pref_raw);
    let weak_prepared = Arc::downgrade(&pref.prepared);
    let mut domain = ProducerResourceDomain::checked(&[&pref, &decode])?;
    let original_registration = Arc::downgrade(&domain._registration);
    for bad in [
        vec![],
        vec![(2, original_epoch)],
        vec![(0, original_epoch), (0, original_epoch)],
        vec![(
            0,
            original_epoch
                .checked_add(1)
                .ok_or("fixture source epoch overflow")?,
        )],
    ] {
        if domain.retire_members_checked(&bad).is_ok()
            || !domain.producers[0].matches(&pref)
            || !domain.producers[1].matches(&decode)
        {
            return Err("retired negative changed bound domain".into());
        }
    }
    let reference = source
        .raw
        .inventory()
        .iter()
        .map(|(plane, ptr)| shadow_read(*ptr, plane.storage_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    {
        let mut groups = domain
            .groups
            .iter()
            .map(|g| {
                g.timeline
                    .try_lock()
                    .map_err(|_| "retired fixture group held")
            })
            .collect::<Result<Vec<_>, _>>()?;
        resource_copy_input(&stream, &source.raw, &pref_raw)?;
        resource_copy_input(&stream, &pref_raw, &decode_raw)?;
        for t in &mut groups {
            check(
                unsafe { sys::lib().cuEventRecord(t.event, stream.handle) },
                "retired fixture completion record",
            )?;
            t.recorded = true;
            t.phases = 1;
        }
    }
    drop(pref_raw);
    drop(pref);
    if weak_raw.upgrade().is_none() {
        return Err("retired old raw released before admission/drain".into());
    }
    domain.retire_members_checked(&[(0, original_epoch)])?;
    if weak_raw.upgrade().is_some()
        || original_registration.upgrade().is_some()
        || weak_prepared.upgrade().is_none()
        || domain.producers[0].bound().is_ok()
        || domain.producers[0].matches(&decode)
        || !domain.producers[1].matches(&decode)
    {
        return Err("retired raw capability/release or decode membership".into());
    }
    if domain
        .retire_members_checked(&[(0, original_epoch + 1)])
        .is_ok()
    {
        return Err("retired capability accepted repeated retirement".into());
    }
    let retired = match &domain.producers[0] {
        OwnedProducerResourceMember::RetiredPreparation(r) => r.clone(),
        _ => return Err("retired role missing".into()),
    };
    let private_planes = retired.prepared.inventory().len();
    if domain._registration.workspaces[0].spans.len() != private_planes {
        return Err("retired registration retains fictitious raw spans".into());
    }
    let owned_bytes = retired
        .prepared
        .schema
        .requested_bytes
        .checked_add(a.len())
        .and_then(|n| n.checked_add(b.len()))
        .ok_or("retired prepared accounting")?;
    let mut attached = KdaRawProducer {
        raw: new_raw.clone(),
        prepared: retired.prepared.clone(),
        a_log: retired.a_log.clone(),
        bias: retired.bias.clone(),
        kernels: retired.kernels.clone(),
        heads: retired.heads,
        dim: retired.dim,
        capacity: retired.capacity,
        input_capacity: source.input_capacity,
        source_epoch: retired
            .source_epoch
            .checked_add(1)
            .ok_or("retired attach epoch overflow")?,
        lower: f32::from_bits(retired.lower_bits),
        scale: f32::from_bits(retired.scale_bits),
        epsilon: f32::from_bits(retired.epsilon_bits),
        parameter_admissions: vec![
            native_pointer_admission(&stream, pointer(&retired.a_log), a.len())?,
            native_pointer_admission(&stream, pointer(&retired.bias), b.len())?,
        ],
        requested_bytes: owned_bytes
            .checked_add(new_raw.schema.requested_bytes)
            .ok_or("retired attached accounting")?,
        allocated_bytes: owned_bytes,
        arithmetic_sha256: retired.arithmetic.clone(),
    };
    let next_epoch = attached.source_epoch;
    attached.source_epoch = retired.source_epoch;
    if domain.attach_members_checked(&[(0, &attached)]).is_ok() {
        return Err("retired stale attach admitted".into());
    }
    attached.source_epoch = next_epoch;
    let actual_prepared = attached.prepared.clone();
    attached.prepared = decode.prepared.clone();
    if domain.attach_members_checked(&[(0, &attached)]).is_ok() {
        return Err("retired foreign preparation admitted".into());
    }
    attached.prepared = actual_prepared;
    for bad in [
        vec![],
        vec![(2, &attached)],
        vec![(0, &attached), (0, &attached)],
        vec![(1, &attached)],
    ] {
        if domain.attach_members_checked(&bad).is_ok()
            || domain.producers[0].bound().is_ok()
            || !domain.producers[1].matches(&decode)
        {
            return Err("retired attach negative changed current domain".into());
        }
    }
    resource_copy_input(&stream, &decode_raw, &new_raw)?;
    let retired_registration = Arc::downgrade(&domain._registration);
    domain.attach_members_checked(&[(0, &attached)])?;
    if retired_registration.upgrade().is_some()
        || !domain.producers[0].matches(&attached)
        || !domain.producers[1].matches(&decode)
        || decode.source_epoch != original_epoch
    {
        return Err("retired attachment/decode epoch preservation".into());
    }
    let event = domain.groups[0]
        .timeline
        .try_lock()
        .map_err(|_| "attached event held")?
        .event as usize;
    for _ in 0..13 {
        if !domain.producers[0].matches(&attached)
            || domain.groups[0]
                .timeline
                .try_lock()
                .map_err(|_| "attached repeat held")?
                .event as usize
                != event
        {
            return Err("attached domain repeated event changed".into());
        }
    }
    let mut records = Vec::new();
    for ((plane, ptr), bytes) in new_raw.inventory().iter().zip(&reference) {
        let got = shadow_read(*ptr, plane.storage_bytes)?;
        owner_compare(&got, bytes, "retired attach complete raw bytes")?;
        records.push(owner_save(
            root,
            &format!("retired-attach-{}.bin", plane.role),
            &got,
        )?);
    }
    owner_compare(
        &shadow_read(pointer(&attached.a_log), a.len())?,
        &a,
        "retired retained A",
    )?;
    owner_compare(
        &shadow_read(pointer(&attached.bias), b.len())?,
        &b,
        "retired retained bias",
    )?;
    let final_registration = Arc::downgrade(&domain._registration);
    drop(retired);
    drop(attached);
    drop(decode);
    drop(decode_raw);
    drop(new_raw);
    drop(domain);
    if weak_prepared.upgrade().is_some() || final_registration.upgrade().is_some() {
        return Err("retired resource final owner leak".into());
    }
    owner_save(root,"RETIRED_CAPABILITY.json",&serde_json::to_vec_pretty(&json!({"passed":true,"source_epochs":[original_epoch,original_epoch+1,next_epoch],"decode_epoch_unchanged":original_epoch,"raw_absent_in_retired":true,"private_prepared_planes":private_planes,"old_raw_released_after_drain":true,"retained_preparation_parameters_exact":true,"new_bound_member_exact":true,"repeat_event_stable":true,"repeated_membership_checks":13,"weak_owners_released":true,"negative_cases":["retire_empty","retire_missing","retire_duplicate","retire_stale","retire_already_retired","attach_stale","attach_foreign_preparation","attach_empty","attach_missing","attach_duplicate","attach_bound_role"],"raw_planes":records,"scope":"typed resource member transitions only; no coupled current EngineArc storage/lifetime or serving acceptance"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
