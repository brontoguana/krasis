// Diagnostic phase completion only. This must never replace ordinary selected
// model reset/cache/restore/epoch admission. All family/resource leases are held.
impl KdaPrefillExecutionPhase<'_, '_> {
    fn complete_actual_shadow(&self) -> Result<(), String> {
        use sha2::Digest;
        let Some(plan) = self.shadow else {
            return Ok(());
        };
        let mut reports = std::collections::BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for receipt in self.shadow_records.borrow().iter() {
            let path = receipt["path"]
                .as_str()
                .ok_or("complete shadow receipt path missing")?;
            if std::path::Path::new(path).file_name() != Some(std::ffi::OsStr::new("REPORT.json")) {
                continue;
            }
            let bytes =
                std::fs::read(path).map_err(|e| format!("complete shadow receipt read:{e}"))?;
            if receipt["bytes"].as_u64() != Some(bytes.len() as u64)
                || receipt["sha256"].as_str()
                    != Some(format!("{:x}", sha2::Sha256::digest(&bytes)).as_str())
            {
                return Err("complete shadow receipt changed".into());
            }
            let report: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            let field = |name: &str| -> Result<usize, String> {
                report[name]
                    .as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .ok_or_else(|| format!("complete shadow invalid receipt field:{name}"))
            };
            if field("complete_request_rows")? != plan.rows
                || report["selected_for_generation"] != false
            {
                return Err("complete shadow request scope changed".into());
            }
            reports
                .entry(field("layer")?)
                .or_default()
                .push((field("actual_chunk_absolute_start")?, field("rows")?));
        }
        if reports.len() != self.bindings.len() {
            return Err("complete shadow missing/extra layer receipt".into());
        }
        let mut registry = crate::session_cache::SequenceStateRegistry::default();
        let mut observed = Vec::with_capacity(self.bindings.len());
        self.stream.drain()?;
        // Complete inventory, all exact receipts/cursors and every arena are
        // verified before the first reset write. A late failure mutates nothing.
        for binding in self.bindings {
            let owner = binding.owner;
            self.validate_layer(
                binding.layer,
                plan.rows,
                self.stream.handle,
                owner.heads,
                owner.dim,
            )?;
            let reset_phase = StatePhase {
                owner,
                stream: self.stream,
            };
            owner
                .pending_raw
                .validate_phase_invalidation(&reset_phase)?;
            let spans = reports
                .get_mut(&binding.layer)
                .ok_or("complete shadow actual layer receipt missing")?;
            spans.sort_unstable();
            let mut cursor = 0usize;
            for &(start, rows) in spans.iter() {
                if rows == 0 || start != cursor {
                    return Err("complete shadow gap/duplicate/partial receipt".into());
                }
                cursor = cursor
                    .checked_add(rows)
                    .ok_or("complete shadow receipt extent overflow")?;
            }
            if cursor != plan.rows {
                return Err("complete shadow incomplete shared prefix".into());
            }
            let absolute = u64::from_le_bytes(
                shadow_read(pointer(&owner.absolute), 8)?
                    .try_into()
                    .map_err(|_| "complete shadow absolute width")?,
            );
            let pending = i32::from_le_bytes(
                shadow_read(pointer(&owner.pending), 4)?
                    .try_into()
                    .map_err(|_| "complete shadow pending width")?,
            );
            let error = i32::from_le_bytes(
                shadow_read(pointer(&owner.error), 4)?
                    .try_into()
                    .map_err(|_| "complete shadow error width")?,
            );
            let block_rows = *owner
                .pending_raw
                .schema
                .planes
                .first()
                .and_then(|p| p.shape.first())
                .ok_or("complete shadow pending schema missing")?;
            if block_rows == 0
                || absolute != plan.rows as u64
                || pending < 0
                || pending as usize != plan.rows % block_rows
                || error != 0
            {
                return Err("complete shadow actual cursor/pending/error mismatch".into());
            }
            owner.register(&mut registry, binding.layer)?;
            let latest = shadow_read(
                owner.p("latest"),
                owner
                    .heads
                    .checked_mul(owner.dim)
                    .and_then(|x| x.checked_mul(owner.dim))
                    .and_then(|x| x.checked_mul(4))
                    .ok_or("complete shadow latest extent")?,
            )?;
            let mut nonzero = 0usize;
            for bytes in latest.chunks_exact(4) {
                let value = f32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| "complete shadow latest width")?,
                );
                if !value.is_finite() {
                    return Err("complete shadow latest nonfinite".into());
                }
                if value != 0.0 {
                    nonzero += 1;
                }
            }
            if nonzero == 0 {
                return Err("complete shadow allzero latest does not exercise reset".into());
            }
            observed.push(serde_json::json!({"layer":binding.layer,"absolute":absolute,"pending":pending,"error":error,"nonzero_latest_elements":nonzero,"complete_chunk_receipts":spans}));
        }
        let hashes = |canonical: bool| -> Result<Vec<serde_json::Value>, String> {
            registry.allocations().iter().map(|a| {
                let bytes = shadow_read(a.ptr, a.storage_bytes)?;
                if canonical {
                    if a.kind.starts_with("kda_fixed16_raw") {
                        if a.dtype != "float32" || a.element_size != 4 || bytes.len() % 4 != 0
                            || bytes.chunks_exact(4).any(|b| b != KDA_PENDING_INVALID_F32_BITS.to_le_bytes()) {
                            return Err(format!("complete shadow pending canonical NaN mismatch:{}", a.name));
                        }
                    } else if bytes.iter().any(|&b| b != 0) { return Err(format!("complete shadow canonical matrix/scalar mismatch:{}", a.name)); }
                }
                Ok(serde_json::json!({"allocation":a,"bytes":bytes.len(),"sha256":format!("{:x}",sha2::Sha256::digest(&bytes))}))
            }).collect()
        };
        let before = hashes(false)?;
        for binding in self.bindings {
            let phase = StatePhase {
                owner: binding.owner,
                stream: self.stream,
            };
            binding.owner.reset_in_phase(&phase)?;
        }
        self.stream.drain()?;
        let after = hashes(true)?;
        let receipt = shadow_save(&plan.output, "COMPLETE_RESET.json", &serde_json::to_vec_pretty(&serde_json::json!({
            "passed":true,"rows":plan.rows,"layers":observed,"complete_inventory":registry.inventory(0),
            "state_hashes_before_reset":before,"state_hashes_after_reset":after,
            "all_cursors_receipts_nonzero_states_verified_before_any_final_shadow_reset":true,
            "all_three_matrices_scalars_zero":true,"all_pending_rows_canonical_invalid_nan":true,
            "all_existing_family_raw_groups_both_arenas_admitted":true,
            "scope":"complete actual nonzero shadow phase canonical reset only; no selected request/cache/restore/epoch integration",
            "selected_for_serving":false,"ordinary_gate":"NOT_RUN"
        })).map_err(|e| e.to_string())?)?;
        self.shadow_records.borrow_mut().push(receipt);
        Ok(())
    }
}
