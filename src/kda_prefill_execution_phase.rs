// Borrowed complete actual-prefill resource phase. Does not select arithmetic,
// admit a decode graph, or attest whole-model weight/graph epochs.
pub(crate) struct KdaPrefillExecutionPhase<'owner, 'borrow> {
    bindings: &'borrow [ModelFamilyBinding<'owner>],
    _leases: &'borrow [EnqueueLease<'owner>],
    stream: &'borrow NativeStream,
    shadow: Option<&'borrow KdaActualPrefillShadowPlan>,
    shadow_records: &'borrow std::cell::RefCell<Vec<serde_json::Value>>,
}
impl KdaPrefillExecutionPhase<'_, '_> {
    pub(crate) fn validate_layer(
        &self,
        layer: usize,
        rows: usize,
        stream: sys::CUstream,
        heads: usize,
        dim: usize,
    ) -> Result<(), String> {
        let binding = self
            .bindings
            .iter()
            .find(|b| b.layer == layer)
            .ok_or("actual prefill phase omitted KDA layer")?;
        if rows == 0
            || rows > binding.producer.input_capacity
            || heads != binding.owner.heads
            || dim != binding.owner.dim
            || stream != self.stream.handle
            || binding.producer.raw.stream.handle != stream
        {
            return Err("actual prefill layer source geometry/extent/stream changed".into());
        }
        binding
            .owner
            .validate_lease(model_arena_lease(binding.owner, self._leases)?)?;
        Ok(())
    }
}
impl KdaEngineRequestStorage {
    pub(crate) fn guards_prefill_writers(&self) -> bool {
        self.guard_prefill_writers
    }
    pub(crate) fn with_prefill_phase<T>(
        &self,
        token_ids: &[u32],
        sequence_start: usize,
        work: impl FnOnce(&KdaPrefillExecutionPhase<'_, '_>) -> Result<T, String>,
    ) -> Result<T, String> {
        if !self.guard_prefill_writers || token_ids.is_empty() {
            return Err("actual prefill writer phase not explicitly admitted or empty".into());
        }
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "actual prefill source storage concurrently leased/poisoned")?;
        if state.poisoned {
            return Err("current engine resource storage poisoned".into());
        }
        if request_storage_static_identity(&state.lifetime)? != state.static_identity {
            return Err("actual prefill phase retained static identity changed".into());
        }
        let next_phase = state
            .prefill_writer_phases
            .checked_add(1)
            .ok_or("actual prefill writer phase count overflow")?;
        let shadow = self
            .shadow
            .as_ref()
            .filter(|p| p.rows == token_ids.len() && sequence_start == 0 && !state.shadow_started);
        let shadow_records = std::cell::RefCell::new(Vec::new());
        if let Some(plan) = shadow {
            let tokens = token_ids
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>();
            let input = shadow_save(&plan.output, "actual-input-ids.u32", &tokens)?;
            shadow_records.borrow_mut().push(input);
            state.shadow_started = true;
        }
        let result = {
            let families = state.lifetime.bound()?;
            if families.len() != self.layers.len() {
                return Err("actual prefill phase family inventory incomplete".into());
            }
            let bindings = families
                .iter()
                .zip(&self.layers)
                .map(|(f, &layer)| ModelFamilyBinding {
                    layer,
                    owner: &f.prefill,
                    producer: &f.prefill_producer,
                })
                .collect::<Vec<_>>();
            let expected = model_family_identity(&bindings)?;
            let stream = bindings
                .first()
                .ok_or("actual prefill phase empty")?
                .owner
                .stream
                .clone();
            stream.validate()?;
            let domain = &state.resource_domain;
            let mut proof = ProducerResourceProof::checked(domain, &bindings, &expected)?;
            proof.groups = (0..domain.groups.len()).collect();
            let mut admission = ProducerResourceAdmission::checked(&proof, &stream)?;
            // Acquire every distinct decode arena as well before any writer.
            let mut inactive_arenas = Vec::new();
            for family in families {
                let arena = family.decode.arena.as_ref();
                if !admission
                    .arena
                    .arenas
                    .iter()
                    .chain(inactive_arenas.iter())
                    .any(|lease: &EnqueueLease<'_>| std::ptr::eq(lease.arena, arena))
                {
                    inactive_arenas.push(arena.lease()?);
                }
                let decode_lease = inactive_arenas
                    .iter()
                    .chain(admission.arena.arenas.iter())
                    .find(|lease| std::ptr::eq(lease.arena, arena))
                    .ok_or("complete actual prefill phase omitted decode arena")?;
                family.decode.validate_lease(decode_lease)?;
            }
            admission.execute(|leases| {
                let phase = KdaPrefillExecutionPhase {
                    bindings: &bindings,
                    _leases: leases,
                    stream: &stream,
                    shadow,
                    shadow_records: &shadow_records,
                };
                let value = work(&phase)?;
                phase.complete_actual_shadow()?;
                Ok(value)
            })
        };
        state.shadow_records.extend(shadow_records.into_inner());
        if result.is_ok() {
            state.prefill_writer_phases = next_phase;
        }
        result
    }
}
