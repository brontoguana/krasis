// Explicit compile-feature controls; ordinary serving contains none of this.
impl GpuDecodeStore {
    pub fn installed_copy_lifecycle_rust(
        &mut self,
        parent: &crate::session_cache::SessionSnapshot,
    ) -> Result<serde_json::Value, String> {
        parent.validate()?;
        let tokens = self.sequence_position_rust()?.kv_absolute_position;
        let previous_tokens = parent.consumed_token_ids.len();
        if previous_tokens == 0 || previous_tokens >= tokens {
            return Err("copy control requires a genuine saved proper-prefix parent".into());
        }
        let previous = self
            .sequence_state_registry
            .allocations()
            .iter()
            .map(|allocation| {
                let mut matches = parent.state_blobs.iter().filter(|b| {
                    b.allocation_name == allocation.name
                        && b.device_ordinal == allocation.device_ordinal
                });
                let blob = matches
                    .next()
                    .ok_or("genuine parent missing canonical allocation")?;
                if matches.next().is_some() {
                    return Err("genuine parent duplicates canonical allocation".into());
                }
                Ok(crate::session_cache::SequenceStateBlob {
                    allocation_name: blob.allocation_name.clone(),
                    kind: blob.kind.clone(),
                    layer_idx: blob.layer_idx,
                    device_ordinal: blob.device_ordinal,
                    dtype: blob.dtype.clone(),
                    element_size: blob.element_size,
                    shape: blob.shape.clone(),
                    strides_bytes: blob.strides_bytes.clone(),
                    bytes: blob.bytes.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if let Some(selected) = &self.selected_kda {
            selected
                .validate_serving_snapshot(previous_tokens, &previous.iter().collect::<Vec<_>>())?;
        }
        self.installed_copy_scope_rust(None, tokens, previous_tokens, previous)
    }
    pub(crate) fn installed_external_copy_lifecycle_rust(
        &mut self,
        allocations: &[crate::session_cache::SequenceStateAllocation],
        owners: &[&dyn crate::sequence_external_owner::CudaRegion],
        tokens: usize,
    ) -> Result<serde_json::Value, String> {
        if allocations.iter().any(|a| {
            !matches!(
                a.growth,
                crate::session_cache::SequenceStateGrowth::TokenRows { .. }
            )
        }) {
            return Err(
                "external prefix control requires only actual token-row allocations".into(),
            );
        }
        let previous_tokens = tokens
            .checked_sub(1)
            .filter(|p| *p > 0)
            .ok_or("external copy requires a real nonempty prefix")?;
        let previous = self.lifecycle_snapshot(Some((allocations, owners)), previous_tokens)?;
        self.installed_copy_scope_rust(
            Some((allocations, owners)),
            tokens,
            previous_tokens,
            previous,
        )
    }
    fn lifecycle_snapshot(
        &mut self,
        external: Option<(
            &[crate::session_cache::SequenceStateAllocation],
            &[&dyn crate::sequence_external_owner::CudaRegion],
        )>,
        tokens: usize,
    ) -> Result<Vec<crate::session_cache::SequenceStateBlob>, String> {
        match external {
            Some((a, o)) => self
                .snapshot_external_sequence_state_rust(a, tokens, o)
                .map(|x| x.0),
            None => self.snapshot_sequence_state_rust(tokens).map(|x| x.0),
        }
    }
    fn lifecycle_incremental(
        &mut self,
        external: Option<(
            &[crate::session_cache::SequenceStateAllocation],
            &[&dyn crate::sequence_external_owner::CudaRegion],
        )>,
        previous_tokens: usize,
        tokens: usize,
        previous: &[&crate::session_cache::SequenceStateBlob],
    ) -> Result<Vec<crate::session_cache::SequenceStateBlob>, String> {
        match external {
            Some((a, o)) => self
                .snapshot_external_sequence_state_incremental_rust(
                    a,
                    previous_tokens,
                    tokens,
                    previous,
                    o,
                )
                .map(|x| x.0),
            None => self
                .snapshot_sequence_state_incremental_rust(previous_tokens, tokens, previous)
                .map(|x| x.0),
        }
    }
    fn lifecycle_restore(
        &mut self,
        external: Option<(
            &[crate::session_cache::SequenceStateAllocation],
            &[&dyn crate::sequence_external_owner::CudaRegion],
        )>,
        tokens: usize,
        blobs: &[&crate::session_cache::SequenceStateBlob],
        rope: i32,
    ) -> Result<(), String> {
        match external {
            Some((a, o)) => self
                .restore_external_sequence_state_rust(a, tokens, blobs, o)
                .map(|_| ()),
            None => self
                .restore_sequence_state_rust(tokens, blobs, rope)
                .map(|_| ()),
        }
    }
    fn installed_copy_scope_rust(
        &mut self,
        external: Option<(
            &[crate::session_cache::SequenceStateAllocation],
            &[&dyn crate::sequence_external_owner::CudaRegion],
        )>,
        tokens: usize,
        previous_tokens: usize,
        previous: Vec<crate::session_cache::SequenceStateBlob>,
    ) -> Result<serde_json::Value, String> {
        use crate::copy_fault::{self, Fault};
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let position = self.sequence_position_rust()?;
        if tokens < 2 {
            return Err("copy lifecycle requires at least two real consumed tokens".into());
        }
        let original = self.lifecycle_snapshot(external, tokens)?;
        if previous_tokens == 0 || previous_tokens >= tokens {
            return Err("copy control previous extent must be a real proper prefix".into());
        }
        let previous_refs: Vec<_> = previous.iter().collect();
        let incremental =
            self.lifecycle_incremental(external, previous_tokens, tokens, &previous_refs)?;
        let compare = |a: &[crate::session_cache::SequenceStateBlob],
                       b: &[crate::session_cache::SequenceStateBlob]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| a.allocation_name == b.allocation_name && a.bytes == b.bytes)
        };
        if !compare(&original, &incremental) {
            return Err("installed full/incremental snapshot differs".into());
        }
        let clone_blobs = |blobs: &[crate::session_cache::SequenceStateBlob]| {
            blobs
                .iter()
                .map(|b| crate::session_cache::SequenceStateBlob {
                    allocation_name: b.allocation_name.clone(),
                    kind: b.kind.clone(),
                    layer_idx: b.layer_idx,
                    device_ordinal: b.device_ordinal,
                    dtype: b.dtype.clone(),
                    element_size: b.element_size,
                    shape: b.shape.clone(),
                    strides_bytes: b.strides_bytes.clone(),
                    bytes: b.bytes.clone(),
                })
                .collect::<Vec<_>>()
        };
        let original_refs: Vec<_> = original.iter().collect();
        let indices = if original.len() > self.sequence_state_staging.len() {
            vec![0, self.sequence_state_staging.len()]
        } else {
            vec![0]
        };
        let mut cases = Vec::new();
        let mut partial_reuse_rejected = true;
        for phase in ["full_snapshot", "incremental_snapshot", "restore"] {
            for &index in &indices {
                for fault in [Fault::EventError, Fault::Panic] {
                    let mut changed = clone_blobs(&previous);
                    // Change finite payload, preserving genuine cursor/error/inactive-row metadata.
                    for b in &mut changed {
                        let role = b
                            .allocation_name
                            .rsplit('.')
                            .next()
                            .ok_or("copy control role")?;
                        if matches!(role, "absolute" | "pending" | "error") {
                            continue;
                        }
                        if role.starts_with("raw") {
                            let rows = *b.shape.first().ok_or("copy control raw schema")?;
                            if rows == 0 {
                                return Err("copy control raw row extent zero".into());
                            }
                            if previous_tokens % rows == 0 {
                                continue;
                            }
                        }
                        if let Some(byte) = b.bytes.first_mut() {
                            *byte ^= 1;
                        }
                    }
                    let changed_refs: Vec<_> = changed.iter().collect();
                    copy_fault::arm(index, fault);
                    let attempt = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                        match phase {
                            "full_snapshot" => {
                                self.lifecycle_snapshot(external, tokens)?;
                            }
                            "incremental_snapshot" => {
                                self.lifecycle_incremental(
                                    external,
                                    previous_tokens,
                                    tokens,
                                    &previous_refs,
                                )?;
                            }
                            "restore" => {
                                self.lifecycle_restore(
                                    external,
                                    previous_tokens,
                                    &changed_refs,
                                    position.rope_position_delta,
                                )?;
                            }
                            _ => unreachable!(),
                        };
                        Ok(())
                    }));
                    let (enqueues, before) = copy_fault::take();
                    let after = unsafe { cuda_sys::lib().cuStreamQuery(self.copy_stream.0) };
                    let expected = match (&attempt, fault) {
                        (Ok(Err(e)), Fault::EventError) => e.contains("post-enqueue event error"),
                        (Err(_), Fault::Panic) => true,
                        _ => false,
                    };
                    if !expected
                        || enqueues != index + 1
                        || after != cuda_sys::CUresult::CUDA_SUCCESS
                    {
                        return Err(format!("{phase}/{index}/{fault:?} copy contract failed: enqueues={enqueues},after={after:?},result={attempt:?}"));
                    }
                    let unchanged = self.sequence_position_rust()?.kv_absolute_position
                        == position.kv_absolute_position;
                    let rejected = if phase == "restore" {
                        let rejected = self.lifecycle_snapshot(external, tokens).is_err();
                        partial_reuse_rejected &= rejected;
                        rejected
                    } else {
                        false
                    };
                    self.lifecycle_restore(
                        external,
                        tokens,
                        &original_refs,
                        position.rope_position_delta,
                    )?;
                    self.require_sequence_state_ready_rust()?;
                    let roundtrip = self.lifecycle_snapshot(external, tokens)?;
                    if !compare(&original, &roundtrip) || !unchanged {
                        return Err(format!("{phase} cursor/original recovery failed"));
                    }
                    cases.push(serde_json::json!({"phase":phase,"index":index,"fault":format!("{fault:?}"),"enqueues":enqueues,"query_at_injection":before,"query_after_return":format!("{after:?}"),"cursor_unchanged":unchanged,"partial_restore_reuse_rejected":rejected,"original_roundtrip_exact":true}));
                }
            }
        }
        let mut invalid = clone_blobs(&original);
        invalid[0].allocation_name.push_str("-invalid");
        let invalid_refs: Vec<_> = invalid.iter().collect();
        copy_fault::take();
        let rejected = self
            .lifecycle_restore(
                external,
                tokens,
                &invalid_refs,
                position.rope_position_delta,
            )
            .is_err();
        let (enqueues, _) = copy_fault::take();
        if !rejected || enqueues != 0 {
            return Err("invalid restore metadata was not rejected before enqueue".into());
        }
        self.require_sequence_state_ready_rust()?;
        // Exercise actual active D2D helpers only in the canonical scope.
        if external.is_none() {
            let fixed = self
                .sequence_state_registry
                .allocations()
                .iter()
                .filter(|a| {
                    matches!(a.growth, crate::session_cache::SequenceStateGrowth::Fixed)
                        && a.storage_bytes > 0
                })
                .count();
            if fixed > 0 {
                self.capture_pending_active_sequence_checkpoint_rust(tokens)?;
                self.commit_active_sequence_checkpoint_rust(tokens)?;
                self.restore_active_sequence_checkpoint_rust(tokens)?;
                if !compare(&original, &self.lifecycle_snapshot(None, tokens)?) {
                    return Err(
                        "successful active checkpoint roundtrip changed original bytes".into(),
                    );
                }
                let active_indices = if fixed > self.sequence_state_staging.len() {
                    vec![0, self.sequence_state_staging.len()]
                } else {
                    vec![0]
                };
                for phase in ["active_capture", "active_restore"] {
                    for &index in &active_indices {
                        for fault in [Fault::EventError, Fault::Panic] {
                            self.capture_pending_active_sequence_checkpoint_rust(tokens)?;
                            self.commit_active_sequence_checkpoint_rust(tokens)?;
                            copy_fault::arm(index, fault);
                            let attempt =
                                catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                                    if phase == "active_capture" {
                                        self.capture_pending_active_sequence_checkpoint_rust(
                                            tokens,
                                        )?;
                                    } else {
                                        self.restore_active_sequence_checkpoint_rust(tokens)?;
                                    }
                                    Ok(())
                                }));
                            let (enqueues, before) = copy_fault::take();
                            let after =
                                unsafe { cuda_sys::lib().cuStreamQuery(self.copy_stream.0) };
                            let expected = match (&attempt, fault) {
                                (Ok(Err(e)), Fault::EventError) => {
                                    e.contains("post-enqueue event error")
                                }
                                (Err(_), Fault::Panic) => true,
                                _ => false,
                            };
                            if !expected
                                || enqueues != index + 1
                                || after != cuda_sys::CUresult::CUDA_SUCCESS
                            {
                                return Err(format!("{phase} active-copy contract failed"));
                            }
                            let rejected = if phase == "active_restore" {
                                let x = self.require_sequence_state_ready_rust().is_err();
                                partial_reuse_rejected &= x;
                                x
                            } else {
                                false
                            };
                            self.lifecycle_restore(
                                None,
                                tokens,
                                &original_refs,
                                position.rope_position_delta,
                            )?;
                            let exact = compare(&original, &self.lifecycle_snapshot(None, tokens)?);
                            if !exact {
                                return Err("active copy changed original bytes".into());
                            }
                            cases.push(serde_json::json!({"phase":phase,"index":index,"fault":format!("{fault:?}"),"enqueues":enqueues,"query_at_injection":before,"query_after_return":format!("{after:?}"),"cursor_unchanged":self.sequence_position_rust()?.kv_absolute_position==position.kv_absolute_position,"partial_restore_reuse_rejected":rejected,"original_roundtrip_exact":true}));
                        }
                    }
                }
            }
            self.discard_active_sequence_checkpoint_rust();
        }
        Ok(
            serde_json::json!({"copy_completion_passed":true,"partial_restore_reuse_rejected":partial_reuse_rejected,"metadata_rejected_before_enqueue":true,"full_incremental_exact":true,"device":self.device.ordinal(),"tokens":tokens,"previous_tokens":previous_tokens,"genuine_saved_parent":external.is_none(),"allocations":original.len(),"bytes":original.iter().map(|b|b.bytes.len()).sum::<usize>(),"cases":cases,"external_scope":external.is_some(),"scope":"actual installed Store helpers; real event errors/unwinds; no CUDA device failure injected"}),
        )
    }
}
