// Explicit diagnostic workspace transactions; serving inventory/policy are not admitted.
struct PrefillWorkspacePlan {
    source: Arc<NativeRawOwner>,
    prepared_schema: Schema,
    capacity: usize,
    input_rows: usize,
    requested: usize,
    allocated: usize,
    epoch: u64,
}
impl PrefillWorkspacePlan {
    fn checked(
        f: &PreparedKdaModelFamily,
        source: RetainedPrefillKdaSource,
        capacity: usize,
        expected_epoch: u64,
    ) -> Result<Self, String> {
        f.borrowed()?;
        if expected_epoch != f.prefill_producer.source_epoch || capacity == 0 {
            return Err("workspace replacement stale epoch/zero capacity".into());
        }
        let admitted = f
            .prefill_producer
            .validate_replacement(&source.inner, false)?;
        if capacity > admitted.0 {
            return Err("workspace capacity exceeds actual retained rows".into());
        }
        let (_, prepared_schema, _) =
            KdaRawProducer::allocation_blueprint(f.prefill.heads, f.prefill.dim, capacity)?;
        let old = f.prefill_producer.prepared.schema.requested_bytes;
        let requested = admitted
            .1
            .checked_sub(old)
            .and_then(|x| x.checked_add(prepared_schema.requested_bytes))
            .ok_or("workspace producer logical accounting")?;
        let allocated = f
            .prefill_producer
            .allocated_bytes
            .checked_sub(old)
            .and_then(|x| x.checked_add(prepared_schema.requested_bytes))
            .ok_or("workspace producer owned accounting")?;
        Ok(Self {
            source: source.inner,
            prepared_schema,
            capacity,
            input_rows: admitted.0,
            requested,
            allocated,
            epoch: admitted.2,
        })
    }
    fn publish(
        self,
        f: &mut PreparedKdaModelFamily,
        arena: Arc<PreparedArena>,
        prepared: NativeRawOwner,
    ) -> u64 {
        f.prefill_producer.raw = self.source;
        f.prefill_producer.prepared = Arc::new(prepared);
        f.prefill_producer.capacity = self.capacity;
        f.prefill_producer.input_capacity = self.input_rows;
        f.prefill_producer.requested_bytes = self.requested;
        f.prefill_producer.allocated_bytes = self.allocated;
        f.prefill_producer.source_epoch = self.epoch;
        f.prefill.max_new_rows = self.capacity;
        f.prefill.padded = arena.padded;
        f.prefill.allocation_bytes = arena.requested_bytes;
        f.prefill.arena = arena;
        self.epoch
    }
}
fn workspace_group_key(f: &PreparedKdaModelFamily, capacity: usize) -> String {
    let p = &f.prefill_producer;
    let a = &f.prefill.arena;
    // Parameters have independent storage per layer; workspace requires matching
    // operations/scalars, not identical parameter values or allocation addresses.
    serde_json::json!({"context":*a.device.cu_primary_ctx() as usize,"stream":a.stream.handle as usize,"heads":a.heads,"dim":a.dim,"capacity":capacity,"arithmetic":a.arithmetic_sha256,"functions":a.functions.iter().map(|(n,fun)|(*n,fun.cu_function() as usize)).collect::<Vec<_>>(),"producer_ptx":p.kernels.ptx_sha256,"domain":p.kernels.domain.cu_function() as usize,"prepare":p.kernels.prepare.cu_function() as usize,"lower":p.lower.to_bits(),"scale":p.scale.to_bits(),"epsilon":p.epsilon.to_bits()}).to_string()
}
pub(crate) fn workspace_inventory_identity(
    families: &[PreparedKdaModelFamily],
    layers: &[usize],
) -> Result<serde_json::Value, String> {
    if families.is_empty()
        || families.len() != layers.len()
        || layers.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("workspace inventory empty/incomplete/noncanonical layer keys".into());
    }
    Ok(serde_json::Value::Array(
        families
            .iter()
            .zip(layers)
            .map(|(f, &layer)| f.identity(layer))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
pub(crate) fn replace_grouped_prefill_workspaces_quiescent(
    families: &mut [PreparedKdaModelFamily],
    layers: &[usize],
    sources: Vec<RetainedPrefillKdaSource>,
    capacities: &[usize],
    epochs: &[u64],
    expected_inventory: &serde_json::Value,
    reserve: usize,
) -> Result<Vec<u64>, String> {
    let identity = workspace_inventory_identity(families, layers)?;
    if &identity != expected_inventory
        || sources.len() != families.len()
        || capacities.len() != families.len()
        || epochs.len() != families.len()
    {
        return Err("workspace complete supplied inventory/epoch admission".into());
    }
    let plans = families
        .iter()
        .zip(sources)
        .zip(capacities)
        .zip(epochs)
        .map(|(((f, source), &cap), &epoch)| PrefillWorkspacePlan::checked(f, source, cap, epoch))
        .collect::<Result<Vec<_>, _>>()?;
    let mut states = Vec::new();
    let mut arenas: Vec<Arc<PreparedArena>> = Vec::new();
    for f in families.iter() {
        if states.iter().any(|s| Arc::ptr_eq(s, &f.prefill.state)) {
            return Err("workspace duplicate persistent family".into());
        }
        states.push(f.prefill.state.clone());
        for owner in [&f.prefill, &f.decode] {
            if !arenas.iter().any(|a| Arc::ptr_eq(a, &owner.arena)) {
                arenas.push(owner.arena.clone());
            }
        }
        f.prefill
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        f.prefill.stream.validate()?;
        f.decode.stream.validate()?;
    }
    // Arc witnesses decouple lock lifetimes from mutable family publication.
    // All guards are acquired and checked before any waiting/allocation/enqueue.
    let mut slots = states
        .iter()
        .map(|s| {
            s.timeline
                .try_lock()
                .map_err(|_| "workspace family concurrently leased/poisoned".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    for slot in &slots {
        let t = slot.as_ref().ok_or("workspace family timeline missing")?;
        if t.poisoned {
            return Err("workspace family poisoned".into());
        }
    }
    let leases = arenas
        .iter()
        .map(|a| a.lease())
        .collect::<Result<Vec<_>, _>>()?;
    let keys = families
        .iter()
        .zip(capacities)
        .map(|(f, &cap)| workspace_group_key(f, cap))
        .collect::<Vec<_>>();
    for slot in &mut slots {
        slot.as_mut().unwrap().poisoned = true;
    }
    // Any later failure leaves all timelines poisoned and publishes no new source.
    for slot in &mut slots {
        let t = slot.as_mut().unwrap();
        t.device.bind_to_thread().map_err(|e| e.to_string())?;
        if t.recorded {
            check(
                unsafe { sys::lib().cuEventSynchronize(t.event) },
                "workspace prior family completion",
            )?;
        }
        for stream in &t.streams {
            stream.drain()?;
        }
    }
    let mut groups: BTreeMap<String, Arc<PreparedArena>> = BTreeMap::new();
    let mut staged = Vec::new();
    for ((f, plan), key) in families.iter().zip(&plans).zip(&keys) {
        f.prefill
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let arena = if let Some(arena) = groups.get(key) {
            arena.clone()
        } else {
            let arena = PreparedArena::allocate_like(&f.prefill.arena, plan.capacity, reserve)?;
            groups.insert(key.clone(), arena.clone());
            arena
        };
        let prepared = NativeRawOwner::allocate(
            f.prefill.stream.clone(),
            plan.prepared_schema.clone(),
            reserve,
        )?;
        f.prefill.stream.drain()?;
        if memory()?[0] < reserve {
            return Err("observed workspace replacement allocation floor".into());
        }
        staged.push((arena, prepared));
    }
    // Publication has no fallible operations or CUDA submissions. Old guards and
    // owned arenas remain retained throughout every member update.
    let new_epochs = families
        .iter_mut()
        .zip(plans)
        .zip(staged)
        .map(|((f, plan), (arena, prepared))| plan.publish(f, arena, prepared))
        .collect();
    for slot in &mut slots {
        slot.as_mut().unwrap().poisoned = false;
    }
    drop(leases);
    Ok(new_epochs)
}
impl PreparedKdaModelFamily {
    pub(crate) fn replace_prefill_workspace_quiescent(
        &mut self,
        source: RetainedPrefillKdaSource,
        measured_capacity: usize,
        expected_epoch: u64,
        reserve: usize,
    ) -> Result<u64, String> {
        let families = std::slice::from_mut(self);
        let layers = [0];
        let identity = workspace_inventory_identity(families, &layers)?;
        let epochs = replace_grouped_prefill_workspaces_quiescent(
            families,
            &layers,
            vec![source],
            &[measured_capacity],
            &[expected_epoch],
            &identity,
            reserve,
        )?;
        Ok(epochs[0])
    }
}
