// Bounded installed-owner diagnostic; never publishes an operational cache.
#[derive(Clone, Copy)]
enum InstalledSequenceAction {
    Snapshot,
    Restore,
    Reset,
}
fn installed_blob_hashes(blobs: &[crate::session_cache::SequenceStateBlob]) -> serde_json::Value {
    serde_json::json!(blobs.iter().map(|b| serde_json::json!({"name":b.allocation_name,"bytes":b.bytes.len(),"sha256":format!("{:x}",Sha256::digest(&b.bytes))})).collect::<Vec<_>>())
}
fn installed_checked_receipt(receipt: &serde_json::Value) -> Result<Vec<u8>, String> {
    let path = receipt["path"]
        .as_str()
        .ok_or("installed sequence receipt path")?;
    if !std::path::Path::new(path).is_absolute() {
        return Err("installed sequence relative receipt".into());
    }
    let bytes = std::fs::read(path).map_err(|e| format!("installed sequence receipt:{e}"))?;
    if receipt["bytes"].as_u64() != Some(bytes.len() as u64)
        || receipt["sha256"].as_str() != Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
    {
        return Err("installed sequence receipt changed".into());
    }
    Ok(bytes)
}
impl KdaEngineRequestStorage {
    fn installed_sequence_action(
        &self,
        store: &mut crate::gpu_decode::GpuDecodeStore,
        action: InstalledSequenceAction,
        tokens: usize,
        input: &[crate::session_cache::SequenceStateBlob],
    ) -> Result<Vec<crate::session_cache::SequenceStateBlob>, String> {
        let (registry, audit) = self.audit()?;
        let allocations = registry.allocations().to_vec();
        // Reject malformed input before acquiring a write phase or queuing DMA.
        if matches!(action, InstalledSequenceAction::Restore) {
            validate_kda_snapshot_semantics(
                &allocations,
                tokens,
                &input.iter().collect::<Vec<_>>(),
            )?;
        }
        let (result, _) = self.current_sequence_transfer(
            &audit["current_identity"],
            audit["domain_generation"]
                .as_u64()
                .ok_or("installed source generation")?,
            audit["live_sequence_epoch"]
                .as_u64()
                .ok_or("installed live epoch")?,
            &mut [store],
            |life, stores| {
                let families = current_sequence_owners(life)?;
                let mut retained = Vec::new();
                for (p, _) in &families {
                    retained.extend(p.external_dma_regions()?);
                }
                let owners = retained
                    .iter()
                    .map(|x| x as &dyn crate::sequence_external_owner::CudaRegion)
                    .collect::<Vec<_>>();
                match action {
                    InstalledSequenceAction::Snapshot => {
                        let (blobs, _) = stores[0].snapshot_external_sequence_state_rust(
                            &allocations,
                            tokens,
                            &owners,
                        )?;
                        validate_kda_snapshot_semantics(
                            &allocations,
                            tokens,
                            &blobs.iter().collect::<Vec<_>>(),
                        )?;
                        Ok(blobs)
                    }
                    InstalledSequenceAction::Restore => {
                        stores[0].restore_external_sequence_state_rust(
                            &allocations,
                            tokens,
                            &input.iter().collect::<Vec<_>>(),
                            &owners,
                        )?;
                        Ok(Vec::new())
                    }
                    InstalledSequenceAction::Reset => {
                        let owners = current_sequence_owners(life)?;
                        for (p, _) in &owners {
                            p.pending_raw.validate_phase_invalidation(&StatePhase {
                                owner: p,
                                stream: &p.stream,
                            })?;
                        }
                        for (p, _) in owners {
                            p.reset_in_phase(&StatePhase {
                                owner: p,
                                stream: &p.stream,
                            })?;
                        }
                        Ok(Vec::new())
                    }
                }
            },
        )?;
        Ok(result)
    }
    pub(crate) fn diagnostic_installed_sequence_roundtrip(
        &self,
        store: &mut crate::gpu_decode::GpuDecodeStore,
    ) -> Result<serde_json::Value, String> {
        let (registry, before) = self.audit()?;
        let records = before["shadow_records"]
            .as_array()
            .ok_or("installed shadow receipts absent")?;
        let mut reports = BTreeMap::new();
        let mut rows = None;
        for receipt in records {
            if std::path::Path::new(receipt["path"].as_str().ok_or("installed receipt path")?)
                .file_name()
                != Some(std::ffi::OsStr::new("REPORT.json"))
            {
                continue;
            }
            let report: serde_json::Value =
                serde_json::from_slice(&installed_checked_receipt(receipt)?)
                    .map_err(|e| e.to_string())?;
            let count = report["complete_request_rows"]
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or("installed shadow request extent")?;
            if rows.replace(count).is_some_and(|old| old != count) || count == 0 {
                return Err("installed shadow histories differ".into());
            }
            let start = report["actual_chunk_absolute_start"]
                .as_u64()
                .ok_or("installed chunk start")?;
            let extent = report["rows"].as_u64().ok_or("installed chunk rows")?;
            if start.checked_add(extent) != Some(count as u64) {
                continue;
            }
            let layer = report["layer"]
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or("installed layer")?;
            if report["selected_for_generation"] != false || reports.insert(layer, report).is_some()
            {
                return Err("installed duplicate/selected final shadow".into());
            }
        }
        let rows = rows.ok_or("installed complete shadow missing")?;
        if reports.keys().copied().collect::<Vec<_>>() != self.layers {
            return Err("installed complete layer coverage".into());
        }
        let mut input = Vec::with_capacity(registry.allocations().len());
        let mut receipts = Vec::with_capacity(registry.allocations().len());
        for allocation in registry.allocations() {
            let report = reports
                .get(
                    &allocation
                        .layer_idx
                        .ok_or("installed allocation lacks layer")?,
                )
                .ok_or("installed actual layer absent")?;
            let original = report["state_inventory"]["allocations"]
                .as_array()
                .ok_or("installed saved inventory")?
                .iter()
                .find(|a| a["name"] == allocation.name)
                .ok_or("installed saved allocation missing")?;
            let mut actual = serde_json::to_value(allocation).map_err(|e| e.to_string())?;
            let mut saved = original.clone();
            actual
                .as_object_mut()
                .ok_or("installed actual descriptor")?
                .remove("ptr");
            saved
                .as_object_mut()
                .ok_or("installed saved descriptor")?
                .remove("ptr");
            if saved["used_bytes"].as_u64() != Some(allocation.used_bytes(rows) as u64) {
                return Err("installed saved used extent changed".into());
            }
            saved
                .as_object_mut()
                .ok_or("installed saved descriptor")?
                .remove("used_bytes");
            if actual != saved {
                return Err(format!(
                    "installed saved layout changed:{}",
                    allocation.name
                ));
            }
            let filename = format!("state-{}.bin", allocation.name);
            let matches = report["files"]
                .as_array()
                .ok_or("installed files")?
                .iter()
                .filter(|f| {
                    f["path"].as_str().is_some_and(|p| {
                        std::path::Path::new(p).file_name() == Some(std::ffi::OsStr::new(&filename))
                    })
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err("installed saved state receipt not unique".into());
            }
            let bytes = installed_checked_receipt(matches[0])?;
            receipts.push(matches[0].clone());
            input.push(crate::session_cache::SequenceStateBlob {
                allocation_name: allocation.name.clone(),
                kind: allocation.kind.clone(),
                layer_idx: allocation.layer_idx,
                device_ordinal: allocation.device_ordinal,
                dtype: allocation.dtype.clone(),
                element_size: allocation.element_size,
                shape: allocation.snapshot_shape(allocation.used_bytes(rows)),
                strides_bytes: allocation.strides_bytes.clone(),
                bytes,
            });
        }
        validate_kda_snapshot_semantics(
            registry.allocations(),
            rows,
            &input.iter().collect::<Vec<_>>(),
        )?;
        let partial = registry
            .allocations()
            .iter()
            .filter(|a| a.name.ends_with(".raw0"))
            .all(|a| {
                a.shape
                    .first()
                    .is_some_and(|&block| block > 0 && rows % block > 0)
            });
        if !partial {
            return Err("installed restore requires actual partial blocks".into());
        }
        let canonical =
            self.installed_sequence_action(store, InstalledSequenceAction::Snapshot, 0, &[])?;
        for blob in &canonical {
            if !blob.kind.starts_with("kda_fixed16_raw") && blob.bytes.iter().any(|&v| v != 0) {
                return Err("installed initial carry not canonical".into());
            }
        }
        self.installed_sequence_action(store, InstalledSequenceAction::Restore, rows, &input)?;
        let restored =
            self.installed_sequence_action(store, InstalledSequenceAction::Snapshot, rows, &[])?;
        if installed_blob_hashes(&restored) != installed_blob_hashes(&input) {
            return Err("installed complete partial restore differs".into());
        }
        self.installed_sequence_action(store, InstalledSequenceAction::Reset, 0, &[])?;
        let reset =
            self.installed_sequence_action(store, InstalledSequenceAction::Snapshot, 0, &[])?;
        if installed_blob_hashes(&reset) != installed_blob_hashes(&canonical) {
            return Err("installed final canonical reset differs".into());
        }
        let after = self.audit()?.1;
        for field in [
            "domain_generation",
            "static_identity",
            "current_identity",
            "events",
            "prefill_writer_phases",
            "shadow_records",
        ] {
            if after[field] != before[field] {
                return Err(format!("installed sequence source changed:{field}"));
            }
        }
        if after["live_sequence_epoch"].as_u64()
            != before["live_sequence_epoch"]
                .as_u64()
                .and_then(|x| x.checked_add(5))
        {
            return Err("installed sequence epoch did not advance five completed phases".into());
        }
        let old_groups = before["resource_domain"]["groups"]
            .as_array()
            .ok_or("installed initial groups")?;
        let new_groups = after["resource_domain"]["groups"]
            .as_array()
            .ok_or("installed final groups")?;
        if old_groups.len() != new_groups.len() {
            return Err("installed sequence resource groups changed".into());
        }
        for (old, new) in old_groups.iter().zip(new_groups) {
            let mut expected = old.clone();
            let phases = old["phases"]
                .as_u64()
                .ok_or("installed initial group phases")?;
            expected["phases"] =
                serde_json::json!(phases.checked_add(5).ok_or("installed phase overflow")?);
            expected["recorded"] = serde_json::json!(false);
            if expected != *new {
                return Err("installed sequence resource identity/event changed".into());
            }
        }
        let restored_bytes = input.iter().try_fold(0usize, |total, blob| {
            total
                .checked_add(blob.bytes.len())
                .ok_or("installed restore byte total overflow")
        })?;
        Ok(
            serde_json::json!({"passed":true,"rows":rows,"layers":self.layers,"allocation_count":input.len(),"restored_bytes":restored_bytes,"actual_partial_blocks":partial,"source_receipts":receipts,"expected_state_hashes":installed_blob_hashes(&input),"restored_state_hashes":installed_blob_hashes(&restored),"canonical_initial_hashes":installed_blob_hashes(&canonical),"canonical_final_hashes":installed_blob_hashes(&reset),"before":before,"after":after,"all_copy_work_completed_under_current_resource_family_arena_guards":true,"selected_for_serving":false,"operational_cache_registered":false,"ordinary_gate":"NOT_RUN"}),
        )
    }
}
