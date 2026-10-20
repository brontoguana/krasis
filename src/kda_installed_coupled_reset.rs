// Explicit startup diagnostic over original state and retained current owners.
// No heterogeneous snapshot publication or serving arithmetic selection.
impl KdaEngineRequestStorage {
    pub(crate) fn diagnostic_installed_coupled_reset(
        &self,
        store: &mut crate::gpu_decode::GpuDecodeStore,
    ) -> Result<serde_json::Value, String> {
        let (_, before) = self.audit()?;
        let canonical =
            self.installed_sequence_action(store, InstalledSequenceAction::Snapshot, 0, &[])?;
        let (_, admitted) = self.audit()?;
        let (original_bytes, _) = self.current_sequence_transfer(
            &admitted["current_identity"],
            admitted["domain_generation"]
                .as_u64()
                .ok_or("coupled source generation")?,
            admitted["live_sequence_epoch"]
                .as_u64()
                .ok_or("coupled live epoch")?,
            &mut [store],
            |life, stores| {
                let owners = current_sequence_owners(life)?;
                // Validate every pending raw plane before the first original write.
                for (p, _) in &owners {
                    p.pending_raw.validate_phase_invalidation(&StatePhase {
                        owner: p,
                        stream: &p.stream,
                    })?;
                }
                let original_bytes = stores[0].reset_sequence_state_for_diagnostic_rust()?;
                for (p, _) in owners {
                    p.reset_in_phase(&StatePhase {
                        owner: p,
                        stream: &p.stream,
                    })?;
                }
                Ok(original_bytes)
            },
        )?;
        let reset =
            self.installed_sequence_action(store, InstalledSequenceAction::Snapshot, 0, &[])?;
        if installed_blob_hashes(&reset) != installed_blob_hashes(&canonical) {
            return Err("coupled reset changed canonical current state".into());
        }
        let (_, after) = self.audit()?;
        for field in [
            "domain_generation",
            "static_identity",
            "current_identity",
            "events",
            "prefill_writer_phases",
            "shadow_records",
        ] {
            if before[field] != after[field] {
                return Err(format!("coupled reset source identity changed:{field}"));
            }
        }
        if after["live_sequence_epoch"].as_u64()
            != before["live_sequence_epoch"]
                .as_u64()
                .and_then(|x| x.checked_add(3))
        {
            return Err("coupled reset completed epoch count".into());
        }
        let old_groups = before["resource_domain"]["groups"]
            .as_array()
            .ok_or("coupled initial groups")?;
        let new_groups = after["resource_domain"]["groups"]
            .as_array()
            .ok_or("coupled final groups")?;
        if old_groups.len() != new_groups.len() {
            return Err("coupled resource group coverage".into());
        }
        for (old, new) in old_groups.iter().zip(new_groups) {
            let mut expected = old.clone();
            expected["phases"] = serde_json::json!(old["phases"]
                .as_u64()
                .and_then(|n| n.checked_add(3))
                .ok_or("coupled phases overflow")?);
            expected["recorded"] = serde_json::json!(false);
            if expected != *new {
                return Err("coupled resource event/stream identity changed".into());
            }
        }
        Ok(
            serde_json::json!({"passed":true,"original_reset_bytes":original_bytes,"canonical_before":installed_blob_hashes(&canonical),"canonical_after":installed_blob_hashes(&reset),"before":before,"after":after,"all_copy_work_completed_under_current_resource_family_arena_guards":true,"operational_cache_registered":false,"selected_for_serving":false}),
        )
    }
}
// Opt-in diagnostic capture over the same retained current storage domain.
impl KdaEngineRequestStorage {
    pub(crate) fn diagnostic_installed_graph_admission(
        &self,
        store: &mut crate::gpu_decode::GpuDecodeStore,
    ) -> Result<serde_json::Value, String> {
        let (_, before) = self.audit()?;
        let (capture, _) = self.current_sequence_transfer(
            &before["current_identity"],
            before["domain_generation"]
                .as_u64()
                .ok_or("graph admission source generation")?,
            before["live_sequence_epoch"]
                .as_u64()
                .ok_or("graph admission live epoch")?,
            &mut [store],
            |_life, stores| stores[0].diagnostic_capture_installed_graphs_rust(),
        )?;
        let (_, after) = self.audit()?;
        for key in [
            "domain_generation",
            "static_identity",
            "current_identity",
            "events",
            "prefill_writer_phases",
            "shadow_records",
        ] {
            if before[key] != after[key] {
                return Err(format!("graph admission identity changed:{key}"));
            }
        }
        if after["live_sequence_epoch"].as_u64()
            != before["live_sequence_epoch"]
                .as_u64()
                .and_then(|x| x.checked_add(1))
        {
            return Err("graph admission completed epoch".into());
        }
        let old = before["resource_domain"]["groups"]
            .as_array()
            .ok_or("graph admission groups")?;
        let new = after["resource_domain"]["groups"]
            .as_array()
            .ok_or("graph admission final groups")?;
        if old.len() != new.len() {
            return Err("graph admission coverage".into());
        }
        for (x, y) in old.iter().zip(new) {
            let mut expected = x.clone();
            expected["phases"] = serde_json::json!(x["phases"]
                .as_u64()
                .and_then(|n| n.checked_add(1))
                .ok_or("graph admission phase overflow")?);
            expected["recorded"] = serde_json::json!(false);
            if expected != *y {
                return Err("graph admission resource metadata".into());
            }
        }
        Ok(
            serde_json::json!({"passed":true,"before":before,"after":after,"capture":capture,"completed_under_resource_family_arena_and_copy_guards":true,"graph_replay_qualified":false}),
        )
    }
}
