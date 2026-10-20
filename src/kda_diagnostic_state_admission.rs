// Cold diagnostic candidate only. Existing operational state remains owned by
// GpuDecodeStore; this draft is never published to its active session registry.
pub(crate) struct KdaDiagnosticStateAdmission {
    original: crate::session_cache::SequenceStateRegistry,
    added: crate::session_cache::SequenceStateRegistry,
    draft: crate::session_cache::SequenceStateRegistry,
    guard: PreparedLogitGuard,
    expected_error_pointers: Vec<u64>,
}
impl KdaDiagnosticStateAdmission {
    pub(crate) fn new(
        families: &[PreparedKdaModelFamily],
        layers: &[usize],
        operational: &crate::session_cache::SequenceStateRegistry,
        reserve: usize,
    ) -> Result<Self, String> {
        if families.is_empty()
            || families.len() != layers.len()
            || operational.allocations().is_empty()
        {
            return Err("diagnostic union requires complete existing state and new owners".into());
        }
        let mut added = crate::session_cache::SequenceStateRegistry::default();
        let mut streams: Vec<Arc<NativeStream>> = Vec::new();
        for (family, &layer) in families.iter().zip(layers) {
            BorrowedKdaModelFamily::checked(
                &family.prefill,
                &family.decode,
                &family.prefill_producer,
                &family.decode_producer,
            )?;
            family.register(&mut added, layer)?;
            for stream in [&family.prefill.stream, &family.decode.stream] {
                if let Some(previous) = streams.iter().find(|x| x.handle == stream.handle) {
                    if !NativeStream::same_backing_owner(previous, stream)? {
                        return Err("diagnostic union stream handle has different owner".into());
                    }
                } else {
                    streams.push(stream.clone());
                }
            }
        }
        let required: Vec<_> = operational
            .allocations()
            .iter()
            .chain(added.allocations())
            .cloned()
            .collect();
        let draft = operational.with_validated_inventory(added.allocations(), &required)?;
        let owners: Vec<_> = families.iter().map(|f| &f.prefill).collect();
        let guard = PreparedLogitGuard::new(&owners, streams, reserve)?;
        guard.validate_owners(&owners)?;
        let expected_error_pointers = owners.iter().map(|o| pointer(&o.error)).collect();
        let result = Self {
            original: operational.clone(),
            added,
            draft,
            guard,
            expected_error_pointers,
        };
        result.audit(operational, &result.added)?;
        Ok(result)
    }
    pub(crate) fn audit(
        &self,
        operational: &crate::session_cache::SequenceStateRegistry,
        added: &crate::session_cache::SequenceStateRegistry,
    ) -> Result<serde_json::Value, String> {
        if self.original.allocations() != operational.allocations()
            || self.added.allocations() != added.allocations()
        {
            return Err("diagnostic union existing or new state inventory changed".into());
        }
        let required: Vec<_> = operational
            .allocations()
            .iter()
            .chain(added.allocations())
            .cloned()
            .collect();
        let current = operational.with_validated_inventory(added.allocations(), &required)?;
        if current.allocations() != self.draft.allocations() {
            return Err("diagnostic complete union changed".into());
        }
        let stream = self
            .guard
            .streams
            .first()
            .ok_or("diagnostic guard stream absent")?;
        stream.validate()?;
        self.guard
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        for allocation in self.draft.allocations() {
            if allocation.device_ordinal != self.guard.device.ordinal() {
                return Err("diagnostic union incomplete actual device ownership".into());
            }
            native_pointer_admission(stream, allocation.ptr, allocation.storage_bytes)?;
        }
        for (state, &expected) in self.guard.states.iter().zip(&self.expected_error_pointers) {
            if pointer(&state.error) != expected || !added.contains_ptr(expected) {
                return Err(
                    "diagnostic guard error owner absent from complete new inventory".into(),
                );
            }
        }
        if self.guard.states.len() != self.expected_error_pointers.len() {
            return Err("diagnostic guard complete error inventory changed".into());
        }
        native_pointer_admission(
            stream,
            pointer(&self.guard.table),
            self.guard.allocation_bytes,
        )?;
        let mut actual = vec![0u64; self.expected_error_pointers.len()];
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    actual.as_mut_ptr().cast(),
                    pointer(&self.guard.table),
                    self.guard.allocation_bytes,
                )
            },
            "diagnostic guard complete table read",
        )?;
        if actual != self.expected_error_pointers {
            return Err("diagnostic guard complete pointer table changed".into());
        }
        let total =
            |registry: &crate::session_cache::SequenceStateRegistry| -> Result<usize, String> {
                registry.allocations().iter().try_fold(0usize, |sum, a| {
                    sum.checked_add(a.storage_bytes)
                        .ok_or_else(|| "diagnostic state inventory extent overflow".into())
                })
            };
        Ok(serde_json::json!({
            "passed":true,"original_operational_inventory":operational.inventory(0),
            "new_diagnostic_inventory":added.inventory(0),"draft_complete_inventory":self.draft.inventory(0),
            "original_operational_storage_bytes":total(operational)?,"new_diagnostic_storage_bytes":total(added)?,
            "draft_complete_storage_bytes":total(&self.draft)?,"all_existing_descriptors_preserved":true,
            "all_native_spans_admitted":true,"guard_owner_count":self.guard.states.len(),
            "guard_table_ptr":pointer(&self.guard.table),"guard_table_bytes":self.guard.allocation_bytes,
            "guard_error_pointers":actual,"guard_required_pointers":self.guard.required_pointers(),
            "guard_arithmetic_sha256":self.guard.arithmetic_sha256,
            "guard_streams":self.guard.streams.iter().map(|s| s.handle as usize).collect::<Vec<_>>(),
            "operational_registry_published":false,"guard_selected_for_serving":false,
            "scope":"actual typed same-device diagnostic draft union and retained GPU error table; no active cache/state publication or selected arithmetic"
        }))
    }
}
