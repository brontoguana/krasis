struct KdaRetirementPlan {
    costs_epochs: Vec<(usize, u64)>,
}
struct KdaAttachmentPlan {
    sources_rows_costs_epochs: Vec<(RetainedPrefillKdaSource, usize, usize, u64)>,
}
// Explicit lifecycle state: retired prefills retain preparation resources but
// have no raw input allocation or prefill execution capability.
struct RetiredKdaPreparation {
    prepared: Arc<NativeRawOwner>,
    a_log: Arc<GpuBuf<f32>>,
    bias: Arc<GpuBuf<f32>>,
    kernels: Arc<KdaRawKernels>,
    heads: usize,
    dim: usize,
    capacity: usize,
    source_epoch: u64,
    lower: f32,
    scale: f32,
    epsilon: f32,
    parameter_admissions: Vec<NativePointerAdmission>,
    owned_bytes: usize,
    arithmetic_sha256: String,
}
struct RetiredPrefillKdaFamily {
    prefill: PreparedOwner,
    decode: PreparedOwner,
    preparation: RetiredKdaPreparation,
    decode_producer: KdaRawProducer,
}
enum KdaPrefillStorageState {
    Bound(Vec<PreparedKdaModelFamily>),
    Retired(Vec<RetiredPrefillKdaFamily>),
    Poisoned,
}
pub(crate) struct KdaPrefillStorageLifetime {
    layers: Vec<usize>,
    state: KdaPrefillStorageState,
}
struct KdaQuiescentStorageWitnesses {
    states: Vec<Arc<PreparedPersistentState>>,
    arenas: Vec<Arc<PreparedArena>>,
}
impl KdaQuiescentStorageWitnesses {
    fn checked(owners: &[(&PreparedOwner, &PreparedOwner)]) -> Result<Self, String> {
        let first = owners.first().ok_or("source lifetime empty inventory")?;
        let context = *first.0.device.cu_primary_ctx();
        first.0.device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut states = Vec::new();
        let mut arenas = Vec::new();
        for &(prefill, decode) in owners {
            if !Arc::ptr_eq(&prefill.state, &decode.state)
                || *prefill.device.cu_primary_ctx() != context
                || *decode.device.cu_primary_ctx() != context
            {
                return Err(
                    "source lifetime needs complete outer admission for foreign/multiple contexts"
                        .into(),
                );
            }
            if states.iter().any(|s| Arc::ptr_eq(s, &prefill.state)) {
                return Err("source lifetime duplicate family".into());
            }
            states.push(prefill.state.clone());
            for owner in [prefill, decode] {
                owner.stream.validate()?;
                if !arenas.iter().any(|a| Arc::ptr_eq(a, &owner.arena)) {
                    arenas.push(owner.arena.clone());
                }
            }
        }
        Ok(Self { states, arenas })
    }
    fn execute<T>(&self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        self.execute_marked(&std::cell::Cell::new(false), work)
    }
    fn execute_marked<T>(
        &self,
        started: &std::cell::Cell<bool>,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut slots = self
            .states
            .iter()
            .map(|s| {
                s.timeline
                    .try_lock()
                    .map_err(|_| "source lifetime family concurrently leased/poisoned".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        for slot in &slots {
            let t = slot.as_ref().ok_or("source lifetime timeline absent")?;
            if t.poisoned {
                return Err("source lifetime family poisoned".into());
            }
            t.phases
                .checked_add(1)
                .ok_or("source lifetime phase overflow")?;
            for s in &t.streams {
                s.validate()?;
            }
        }
        let _leases = self
            .arenas
            .iter()
            .map(|a| a.lease())
            .collect::<Result<Vec<_>, _>>()?;
        started.set(true);
        for slot in &mut slots {
            slot.as_mut().expect("checked timeline").poisoned = true;
        }
        for slot in &slots {
            let t = slot.as_ref().expect("checked timeline");
            if t.recorded {
                check(
                    unsafe { sys::lib().cuEventSynchronize(t.event) },
                    "source lifetime prior completion",
                )?;
            }
            for s in &t.streams {
                s.drain()?;
            }
        }
        let value = work()?;
        for slot in &mut slots {
            let t = slot.as_mut().expect("checked timeline");
            t.phases += 1;
            t.recorded = false;
            t.poisoned = false;
        }
        Ok(value)
    }
}
impl KdaPrefillStorageLifetime {
    pub(crate) fn from_bound(
        families: Vec<PreparedKdaModelFamily>,
        layers: Vec<usize>,
    ) -> Result<Self, String> {
        workspace_inventory_identity(&families, &layers)?;
        Ok(Self {
            layers,
            state: KdaPrefillStorageState::Bound(families),
        })
    }
    pub(crate) fn bound(&self) -> Result<&[PreparedKdaModelFamily], String> {
        match &self.state {
            KdaPrefillStorageState::Bound(f) => Ok(f),
            KdaPrefillStorageState::Retired(_) => {
                Err("prefill inputs explicitly retired; current source admission required".into())
            }
            KdaPrefillStorageState::Poisoned => {
                Err("prefill storage lifetime poisoned/extracted".into())
            }
        }
    }
    pub(crate) fn decode_bindings(&self) -> Result<Vec<ModelFamilyBinding<'_>>, String> {
        let pairs = match &self.state {
            KdaPrefillStorageState::Bound(f) => f
                .iter()
                .map(|f| (&f.decode, &f.decode_producer))
                .collect::<Vec<_>>(),
            KdaPrefillStorageState::Retired(f) => f
                .iter()
                .map(|f| (&f.decode, &f.decode_producer))
                .collect::<Vec<_>>(),
            KdaPrefillStorageState::Poisoned => {
                return Err("decode lifetime poisoned/extracted".into())
            }
        };
        let bindings = pairs
            .into_iter()
            .zip(&self.layers)
            .map(|((owner, producer), &layer)| ModelFamilyBinding {
                layer,
                owner,
                producer,
            })
            .collect::<Vec<_>>();
        model_family_identity(&bindings)?;
        Ok(bindings)
    }
    pub(crate) fn registry(&self) -> Result<crate::session_cache::SequenceStateRegistry, String> {
        let mut registry = crate::session_cache::SequenceStateRegistry::default();
        match &self.state {
            KdaPrefillStorageState::Bound(families) => {
                for (f, &layer) in families.iter().zip(&self.layers) {
                    f.register(&mut registry, layer)?;
                }
            }
            KdaPrefillStorageState::Retired(families) => {
                for (f, &layer) in families.iter().zip(&self.layers) {
                    f.prefill.register(&mut registry, layer)?;
                }
            }
            KdaPrefillStorageState::Poisoned => {
                return Err("source lifetime registry poisoned/extracted".into())
            }
        }
        Ok(registry)
    }
    fn prepare_retirement_plan(
        &self,
        expected_inventory: &serde_json::Value,
        expected_epochs: &[u64],
    ) -> Result<KdaRetirementPlan, String> {
        let families = self.bound()?;
        if &workspace_inventory_identity(families, &self.layers)? != expected_inventory
            || expected_epochs.len() != families.len()
        {
            return Err("source retirement stale/incomplete inventory".into());
        }
        let plans = families
            .iter()
            .zip(expected_epochs)
            .map(|(f, &epoch)| {
                let p = &f.prefill_producer;
                if p.source_epoch != epoch {
                    return Err("source retirement stale member epoch".to_string());
                }
                let owned = p
                    .requested_bytes
                    .checked_sub(p.raw.schema.requested_bytes)
                    .ok_or("source retirement accounting")?;
                if owned != p.allocated_bytes {
                    return Err(
                        "source retirement requires model-retained source allocation accounting"
                            .into(),
                    );
                }
                Ok((
                    owned,
                    epoch
                        .checked_add(1)
                        .ok_or("source retirement epoch overflow")?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(KdaRetirementPlan {
            costs_epochs: plans,
        })
    }
    pub(crate) fn retire_prefill_quiescent(
        &mut self,
        expected_inventory: &serde_json::Value,
        expected_epochs: &[u64],
    ) -> Result<Vec<u64>, String> {
        let plan = self.prepare_retirement_plan(expected_inventory, expected_epochs)?;
        let owners = self
            .bound()?
            .iter()
            .map(|f| (&f.prefill, &f.decode))
            .collect::<Vec<_>>();
        let admission = KdaQuiescentStorageWitnesses::checked(&owners)?;
        admission.execute(|| self.apply_retirement_plan(plan))
    }
    fn apply_retirement_plan(&mut self, plan: KdaRetirementPlan) -> Result<Vec<u64>, String> {
        let bound = match std::mem::replace(&mut self.state, KdaPrefillStorageState::Poisoned) {
            KdaPrefillStorageState::Bound(f) => f,
            _ => return Err("source retirement state changed".into()),
        };
        let mut retired = Vec::with_capacity(bound.len());
        let mut epochs = Vec::with_capacity(bound.len());
        for (f, (owned_bytes, source_epoch)) in bound.into_iter().zip(plan.costs_epochs) {
            let PreparedKdaModelFamily {
                prefill,
                decode,
                prefill_producer,
                decode_producer,
            } = f;
            let KdaRawProducer {
                raw,
                prepared,
                a_log,
                bias,
                kernels,
                heads,
                dim,
                capacity,
                input_capacity: _,
                source_epoch: _,
                lower,
                scale,
                epsilon,
                parameter_admissions,
                requested_bytes: _,
                allocated_bytes: _,
                arithmetic_sha256,
            } = prefill_producer;
            // Every prior reader completed while all family/arena guards are held.
            drop(raw);
            retired.push(RetiredPrefillKdaFamily {
                prefill,
                decode,
                decode_producer,
                preparation: RetiredKdaPreparation {
                    prepared,
                    a_log,
                    bias,
                    kernels,
                    heads,
                    dim,
                    capacity,
                    source_epoch,
                    lower,
                    scale,
                    epsilon,
                    parameter_admissions,
                    owned_bytes,
                    arithmetic_sha256,
                },
            });
            epochs.push(source_epoch);
        }
        self.state = KdaPrefillStorageState::Retired(retired);
        Ok(epochs)
    }
    fn prepare_attachment_plan(
        &self,
        sources: Vec<RetainedPrefillKdaSource>,
        expected_epochs: &[u64],
    ) -> Result<KdaAttachmentPlan, String> {
        let families = match &self.state {
            KdaPrefillStorageState::Retired(f) => f,
            _ => return Err("source attachment requires explicitly retired prefills".into()),
        };
        if sources.len() != families.len() || expected_epochs.len() != families.len() {
            return Err("source attachment incomplete inventory".into());
        }
        let plans = families
            .iter()
            .zip(sources)
            .zip(expected_epochs)
            .map(|((f, source), &epoch)| {
                let p = &f.preparation;
                let raw = &source.inner;
                if p.source_epoch != epoch {
                    return Err("source attachment stale member epoch".to_string());
                }
                raw.schema.validate()?;
                raw.stream.validate()?;
                let rows = *raw
                    .schema
                    .planes
                    .first()
                    .and_then(|p| p.shape.first())
                    .ok_or("source attachment rows")?;
                let schema =
                    Schema::checked(RawContract::KdaBf16VectorGate, rows, p.heads, p.dim, p.dim)?;
                if rows == 0
                    || raw.schema != schema
                    || !Arc::ptr_eq(&raw.device, &p.prepared.device)
                    || raw.stream.handle != p.prepared.stream.handle
                    || raw.buffers.len() != schema.planes.len()
                    || raw.admissions.len() != schema.planes.len()
                {
                    return Err("source attachment typed extent/context/stream".into());
                }
                for ((plane, buffer), admitted) in
                    schema.planes.iter().zip(&raw.buffers).zip(&raw.admissions)
                {
                    if buffer.dtype() != plane.dtype
                        || buffer.elements() != plane.elements
                        || admitted.ptr != buffer.address()
                        || admitted.requested_bytes != plane.storage_bytes
                    {
                        return Err("source attachment typed span".into());
                    }
                    native_pointer_admission(&raw.stream, buffer.address(), plane.storage_bytes)?;
                }
                let requested = p
                    .owned_bytes
                    .checked_add(schema.requested_bytes)
                    .ok_or("source attachment accounting")?;
                let next = epoch
                    .checked_add(1)
                    .ok_or("source attachment epoch overflow")?;
                Ok((source, rows, requested, next))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(KdaAttachmentPlan {
            sources_rows_costs_epochs: plans,
        })
    }
    pub(crate) fn attach_prefill_quiescent(
        &mut self,
        sources: Vec<RetainedPrefillKdaSource>,
        expected_epochs: &[u64],
    ) -> Result<Vec<u64>, String> {
        let plan = self.prepare_attachment_plan(sources, expected_epochs)?;
        let families = match &self.state {
            KdaPrefillStorageState::Retired(f) => f,
            _ => return Err("source attachment state changed".into()),
        };
        let owners = families
            .iter()
            .map(|f| (&f.prefill, &f.decode))
            .collect::<Vec<_>>();
        let admission = KdaQuiescentStorageWitnesses::checked(&owners)?;
        admission.execute(|| self.apply_attachment_plan(plan))
    }
    fn apply_attachment_plan(&mut self, plan: KdaAttachmentPlan) -> Result<Vec<u64>, String> {
        let retired = match std::mem::replace(&mut self.state, KdaPrefillStorageState::Poisoned) {
            KdaPrefillStorageState::Retired(f) => f,
            _ => return Err("source attachment state changed".into()),
        };
        let mut bound = Vec::with_capacity(retired.len());
        let mut epochs = Vec::with_capacity(retired.len());
        for (f, (source, input_capacity, requested_bytes, source_epoch)) in
            retired.into_iter().zip(plan.sources_rows_costs_epochs)
        {
            let RetiredPrefillKdaFamily {
                prefill,
                decode,
                preparation,
                decode_producer,
            } = f;
            let RetiredKdaPreparation {
                prepared,
                a_log,
                bias,
                kernels,
                heads,
                dim,
                capacity,
                source_epoch: _,
                lower,
                scale,
                epsilon,
                parameter_admissions,
                owned_bytes,
                arithmetic_sha256,
            } = preparation;
            let prefill_producer = KdaRawProducer {
                raw: source.inner,
                prepared,
                a_log,
                bias,
                kernels,
                heads,
                dim,
                capacity,
                input_capacity,
                source_epoch,
                lower,
                scale,
                epsilon,
                parameter_admissions,
                requested_bytes,
                allocated_bytes: owned_bytes,
                arithmetic_sha256,
            };
            bound.push(PreparedKdaModelFamily {
                prefill,
                decode,
                prefill_producer,
                decode_producer,
            });
            epochs.push(source_epoch);
        }
        self.state = KdaPrefillStorageState::Bound(bound);
        Ok(epochs)
    }
    pub(crate) fn retire_and_verify_input_release(
        &mut self,
        expected: &serde_json::Value,
        epochs: &[u64],
    ) -> Result<Vec<u64>, String> {
        let weak = self
            .bound()?
            .iter()
            .map(|f| Arc::downgrade(&f.prefill_producer.raw))
            .collect::<Vec<_>>();
        let next = self.retire_prefill_quiescent(expected, epochs)?;
        if weak.iter().any(|w| w.upgrade().is_some()) {
            return Err("retired actual prefill source wrapper still leased externally".into());
        }
        Ok(next)
    }
    pub(crate) fn explicitly_retired(&self) -> Result<bool, String> {
        match &self.state {
            KdaPrefillStorageState::Bound(_) => Ok(false),
            KdaPrefillStorageState::Retired(_) => Ok(true),
            KdaPrefillStorageState::Poisoned => {
                Err("prefill storage phase poisoned/extracted".into())
            }
        }
    }
    pub(crate) fn current_prefill_epochs(&self) -> Result<Vec<u64>, String> {
        match &self.state {
            KdaPrefillStorageState::Bound(f) => {
                Ok(f.iter().map(|f| f.prefill_producer.source_epoch).collect())
            }
            KdaPrefillStorageState::Retired(f) => {
                Ok(f.iter().map(|f| f.preparation.source_epoch).collect())
            }
            KdaPrefillStorageState::Poisoned => Err("prefill epochs poisoned/extracted".into()),
        }
    }
    pub(crate) fn storage_identity(&self) -> Result<serde_json::Value, String> {
        match &self.state {
            KdaPrefillStorageState::Bound(f) => Ok(
                serde_json::json!({"phase":"bound","families":workspace_inventory_identity(f,&self.layers)?}),
            ),
            KdaPrefillStorageState::Retired(f) => {
                let mut rows = Vec::new();
                for (f, &layer) in f.iter().zip(&self.layers) {
                    let p = &f.preparation;
                    p.prepared.schema.validate()?;
                    p.prepared.stream.validate()?;
                    let decoder = ModelFamilyBinding {
                        layer,
                        owner: &f.decode,
                        producer: &f.decode_producer,
                    };
                    let arena = model_workspace_identity(&ModelFamilyBinding {
                        layer,
                        owner: &f.prefill,
                        producer: &f.decode_producer,
                    })?["recurrence_arena"]
                        .clone();
                    rows.push(serde_json::json!({"layer":layer,"prefill_source":"explicitly_absent","source_epoch":p.source_epoch,"prepared_schema":p.prepared.schema,"prepared_planes":p.prepared.inventory(),"prepared_admissions":p.prepared.admissions,"parameter_spans":p.parameter_admissions,"a_log":{"ptr":pointer(&p.a_log),"elements":p.a_log.len()},"bias":{"ptr":pointer(&p.bias),"elements":p.bias.len()},"lower":p.lower.to_bits(),"scale":p.scale.to_bits(),"epsilon":p.epsilon.to_bits(),"arithmetic":p.arithmetic_sha256,"kernel_ptx":p.kernels.ptx_sha256,"heads":p.heads,"dim":p.dim,"capacity":p.capacity,"owned_preparation_bytes":p.owned_bytes,"recurrence_arena":arena,"decode":model_family_identity(&[decoder])?}));
                }
                let registry = self.registry()?;
                Ok(
                    serde_json::json!({"phase":"retired","families":rows,"complete_registry":registry.inventory(0)}),
                )
            }
            KdaPrefillStorageState::Poisoned => Err("storage identity poisoned/extracted".into()),
        }
    }
    pub(crate) fn take_bound(&mut self) -> Result<Vec<PreparedKdaModelFamily>, String> {
        self.bound()?;
        match std::mem::replace(&mut self.state, KdaPrefillStorageState::Poisoned) {
            KdaPrefillStorageState::Bound(f) => Ok(f),
            _ => Err("source extraction state changed".into()),
        }
    }
}
