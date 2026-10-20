fn session_pinned_replay_controls(
    root: &Path,
    model: &mut RetainedKdaSessionModel,
    pristine: &[Vec<crate::session_cache::SequenceStateBlob>],
    store: &mut crate::gpu_decode::GpuDecodeStore,
    allocations: &[crate::session_cache::SequenceStateAllocation],
    compatibility: &Value,
    saved: &[crate::session_cache::SequenceStateBlob],
    all: &Arc<NativeRawOwner>,
    staging: &Arc<NativeRawOwner>,
    capacity: usize,
    batch_end: usize,
    total: usize,
    reference: &[u8],
    plan: &TilingPlan,
    snapshot_ms: Vec<(usize, f64)>,
    reserve: usize,
) -> Result<(), String> {
    model_source_transaction_controls(root, model, reserve)?;
    prefill_source_lifetime_controls(root, model, reserve)?;
    current_engine_domain_controls(root, model, reserve, store)?;
    let identity = workspace_inventory_identity(&model.families, &model.layers)?;
    if model.compatibility()? != *compatibility {
        return Err("session cache compatibility changed with growth".into());
    }
    let refs = saved.iter().collect::<Vec<_>>();
    let (final_before, _) = model.snapshot_pinned(&identity, total, &mut [store])?;
    let final_bytes = serde_json::to_vec(&final_before).map_err(|e| e.to_string())?;
    owner_save(root, "session-final-before.json", &final_bytes)?;
    let mut rejected = Vec::new();
    for key in [
        "model_sha256",
        "weights_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "source_sha256",
        "binary_sha256",
        "epoch",
    ] {
        let mut bad = compatibility.clone();
        bad["model"][key] = if key == "epoch" {
            json!(1)
        } else {
            json!(format!("{:x}", Sha256::digest(key.as_bytes())))
        };
        if model
            .restore_pinned(&identity, &bad, 1, &refs, &mut [store])
            .is_ok()
        {
            return Err(format!("session wrong model stamp admitted:{key}"));
        }
        rejected.push(key.to_string());
    }
    for key in ["layout", "arithmetic", "physical_devices"] {
        let mut bad = compatibility.clone();
        bad[key] = json!([]);
        if model
            .restore_pinned(&identity, &bad, 1, &refs, &mut [store])
            .is_ok()
        {
            return Err(format!(
                "session incompatible stable material admitted:{key}"
            ));
        }
        rejected.push(key.into());
    }
    let mut missing_called = false;
    if model
        .session_transfer(&identity, &mut [], |_| {
            missing_called = true;
            Ok(())
        })
        .is_ok()
        || missing_called
    {
        return Err("session missing transfer device reached DMA".into());
    }
    rejected.push("missing-transfer-device".into());
    {
        let _held = model
            .families
            .last()
            .ok_or("session family missing")?
            .decode
            .arena
            .lease()?;
        let mut called = false;
        if model
            .session_transfer(&identity, &mut [store], |_| {
                called = true;
                Ok(())
            })
            .is_ok()
            || called
        {
            return Err("session held decode arena reached DMA".into());
        }
        rejected.push("late-decode-arena-guard".into());
    }
    for case in 0..5 {
        let mut bad: Vec<crate::session_cache::SequenceStateBlob> =
            serde_json::from_slice(&serde_json::to_vec(saved).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let last = bad.len().checked_sub(1).ok_or("session no snapshot")?;
        match case {
            0 => {
                bad.pop();
            }
            1 => {
                bad[last].bytes.pop();
            }
            2 => {
                bad[last].allocation_name = bad[0].allocation_name.clone();
            }
            3 => {
                bad[last].strides_bytes[0] += 1;
            }
            4 => {
                bad.iter_mut()
                    .find(|b| b.allocation_name.ends_with(".absolute"))
                    .ok_or("absolute absent")?
                    .bytes = 2u64.to_le_bytes().to_vec();
            }
            _ => unreachable!(),
        }
        let br = bad.iter().collect::<Vec<_>>();
        if model
            .restore_pinned(&identity, compatibility, 1, &br, &mut [store])
            .is_ok()
        {
            return Err(format!("session malformed late blob admitted:{case}"));
        }
        rejected.push(format!("blob-{case}"));
    }
    {
        let held = model
            .families
            .last()
            .ok_or("session family missing")?
            .prefill
            .state
            .timeline
            .try_lock()
            .map_err(|_| "session late guard")?;
        let mut called = false;
        if model
            .session_transfer(&identity, &mut [store], |_| {
                called = true;
                Ok(())
            })
            .is_ok()
            || called
        {
            return Err("session held late family reached DMA".into());
        }
        drop(held);
        rejected.push("late-family-guard".into());
    }
    let mut called = false;
    if model
        .session_transfer(&json!([]), &mut [store], |_| {
            called = true;
            Ok(())
        })
        .is_ok()
        || called
    {
        return Err("session stale proof reached DMA".into());
    }
    rejected.push("stale-workspace-proof".into());
    let (after_reject, _) = model.snapshot_pinned(&identity, total, &mut [store])?;
    owner_compare(
        &serde_json::to_vec(&after_reject).map_err(|e| e.to_string())?,
        &final_bytes,
        "session all rejected admissions left complete state unchanged",
    )?;
    owner_save(
        root,
        "session-after-rejections.json",
        &serde_json::to_vec(&after_reject).map_err(|e| e.to_string())?,
    )?;
    // Reset/cancel is a separate live request epoch; stable cached compatibility stays valid.
    let epoch_before = model.live_sequence_epoch;
    let reset_epoch = model.reset_sequence_quiescent(&identity, epoch_before, &mut [store])?;
    if reset_epoch != epoch_before.checked_add(1).ok_or("test epoch overflow")? {
        return Err("session reset did not renew live request epoch".into());
    }
    let (reset_state, _) = model.snapshot_pinned(&identity, 0, &mut [store])?;
    session_compare_pristine(model, &reset_state, pristine)?;
    owner_save(
        root,
        "session-reset-canonical.json",
        &serde_json::to_vec(&reset_state).map_err(|e| e.to_string())?,
    )?;
    if model
        .reset_sequence_quiescent(&identity, epoch_before, &mut [store])
        .is_ok()
    {
        return Err("session stale live request epoch admitted".into());
    }
    {
        let state = model
            .families
            .last()
            .ok_or("reset late family")?
            .prefill
            .state
            .clone();
        let held = state
            .timeline
            .try_lock()
            .map_err(|_| "reset held control")?;
        if model
            .reset_sequence_quiescent(&identity, reset_epoch, &mut [store])
            .is_ok()
        {
            return Err("reset held late family admitted".into());
        }
        drop(held);
    }
    if model
        .reset_sequence_quiescent(&json!([]), reset_epoch, &mut [store])
        .is_ok()
    {
        return Err("reset stale workspace admitted".into());
    }
    model.live_sequence_epoch = u64::MAX;
    if model
        .reset_sequence_quiescent(&identity, u64::MAX, &mut [store])
        .is_ok()
    {
        return Err("reset overflow admitted".into());
    }
    model.live_sequence_epoch = reset_epoch;
    let (reset_after_reject, _) = model.snapshot_pinned(&identity, 0, &mut [store])?;
    owner_compare(
        &serde_json::to_vec(&reset_after_reject).map_err(|e| e.to_string())?,
        &serde_json::to_vec(&reset_state).map_err(|e| e.to_string())?,
        "reset negatives full bytes unchanged",
    )?;
    model.restore_pinned(&identity, compatibility, 1, &refs, &mut [store])?;
    // Queue real native batch work and immediately cancel/reset. The lifecycle
    // drains submitted work before releasing any state/arena guard.
    for (f, &layer) in model.families.iter().zip(&model.layers) {
        let b = f.borrowed()?;
        let ps = &b.prefill.stream;
        let bindings = [b.prefill_binding(layer)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, ps)?;
        held.execute(|leases| {
            b.prefill_producer.enqueue(1, capacity, b.prefill)?;
            b.prefill.prefill_in(
                model_arena_lease(b.prefill, leases)?,
                &b.prefill_producer.view(capacity)?,
                1,
                capacity,
            )
        })?;
    }
    let cancelled_epoch = model.reset_sequence_quiescent(&identity, reset_epoch, &mut [store])?;
    if model.validate_live_sequence_epoch(reset_epoch).is_ok() {
        return Err("cancelled live epoch retained admission".into());
    }
    let (cancelled_state, _) = model.snapshot_pinned(&identity, 0, &mut [store])?;
    session_compare_pristine(model, &cancelled_state, pristine)?;
    owner_save(
        root,
        "session-cancel-canonical.json",
        &serde_json::to_vec(&cancelled_state).map_err(|e| e.to_string())?,
    )?;
    if model.stable_compatibility != *compatibility || model.compatibility()? != *compatibility {
        return Err("reset/cancel corrupted stable model/cache compatibility".into());
    }
    owner_save(root,"session-reset-cancel-report.json",&serde_json::to_vec_pretty(&json!({"passed":true,"epochs":[epoch_before,reset_epoch,cancelled_epoch],"negative_admissions":["stale-live-epoch","held-late-family","stale-workspace","epoch-overflow"],"complete_canonical_state_after_reset_and_cancel":true,"stable_cache_compatibility_unchanged":true,"queued_real_rows_per_family":capacity,"scope":"native two-family reset/drain; actual HTTPcancel/whole-serving request epoch not integrated"})).map_err(|e|e.to_string())?)?;
    model.validate_snapshot(compatibility, 1, &refs)?;
    let restore_ms = model.restore_pinned(&identity, compatibility, 1, &refs, &mut [store])?;
    let (restored, _) = model.snapshot_pinned(&identity, 1, &mut [store])?;
    owner_compare(
        &serde_json::to_vec(&restored).map_err(|e| e.to_string())?,
        &serde_json::to_vec(saved).map_err(|e| e.to_string())?,
        "session exact full partial-state restore",
    )?;
    owner_save(
        root,
        "session-restored-row1.json",
        &serde_json::to_vec(&restored).map_err(|e| e.to_string())?,
    )?;
    let bf16_reference = fs::read(&plan.bf16_reference.path).map_err(|e| e.to_string())?;
    for (i, f) in model.families.iter().enumerate() {
        let b = f.borrowed()?;
        let ps = &b.prefill.stream;
        {
            let bindings = [b.prefill_binding(model.layers[i])];
            let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
            let proof = ProducerResourceProof::checked(
                &domain,
                &bindings,
                &model_family_identity(&bindings)?,
            )?;
            let mut held = ProducerResourceAdmission::checked(&proof, ps)?;
            held.execute(|leases| {
                b.prefill_producer.enqueue(1, capacity, b.prefill)?;
                b.prefill.prefill_in(
                    model_arena_lease(b.prefill, leases)?,
                    &b.prefill_producer.view(capacity)?,
                    1,
                    capacity,
                )
            })?;
        }
        let width = b.prefill.heads * b.prefill.dim * 4;
        let mut output = owner_array(
            b.prefill,
            "emitted",
            capacity * b.prefill.heads * b.prefill.dim,
        )?;
        owner_compare(
            &output,
            &reference[width..batch_end * width],
            "session replay full parallel prefix",
        )?;
        let mut bf16 = workspace_bf16(b.prefill, capacity)?;
        let bindings = [b.decode_binding(model.layers[i])];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let (graph, inventory) = resource_graph_capture(
            &proof,
            b.decode,
            b.decode_producer,
            staging,
            &b.decode_producer.raw,
        )?;
        owner_save(
            root,
            &format!("session-replay-graph-{i}.json"),
            &serde_json::to_vec(&inventory).map_err(|e| e.to_string())?,
        )?;
        for row in batch_end..total {
            let mut held = ProducerResourceAdmission::checked(&proof, &b.decode.stream)?;
            held.execute(|leases| {
                workspace_row_copy(&b.decode.stream, all, row, staging)?;
                b.decode
                    .launch_graph_in(model_arena_lease(b.decode, leases)?, &graph)
            })?;
            output.extend(resource_output(b.decode)?);
            bf16.extend(workspace_bf16(b.decode, 1)?);
        }
        owner_compare(
            &output,
            &reference[width..total * width],
            "session complete replay F32",
        )?;
        owner_compare(
            &bf16,
            &bf16_reference[width / 2..total * width / 2],
            "session complete replay BF16",
        )?;
        owner_save(root, &format!("session-replay-output-{i}.f32"), &output)?;
        owner_save(root, &format!("session-replay-output-{i}.bf16"), &bf16)?;
    }
    let (final_replayed, _) = model.snapshot_pinned(&identity, total, &mut [store])?;
    owner_compare(
        &serde_json::to_vec(&final_replayed).map_err(|e| e.to_string())?,
        &final_bytes,
        "session complete replay final state",
    )?;
    owner_save(
        root,
        "session-final-replayed.json",
        &serde_json::to_vec(&final_replayed).map_err(|e| e.to_string())?,
    )?;
    // Deliberately fail an admitted lifecycle only after every positive comparison.
    let result: Result<(), String> = model.session_transfer(&identity, &mut [store], |_| {
        Err("injected admitted transfer failure".into())
    });
    if result.is_ok() {
        return Err("session admitted failure swallowed".into());
    }
    for f in &model.families {
        let slot = f
            .prefill
            .state
            .timeline
            .try_lock()
            .map_err(|_| "session poison read")?;
        if !slot.as_ref().ok_or("session timeline missing")?.poisoned {
            return Err("session failure did not poison all".into());
        }
    }
    let mut called = false;
    if model
        .session_transfer(&identity, &mut [store], |_| {
            called = true;
            Ok(())
        })
        .is_ok()
        || called
    {
        return Err("session poisoned family reached DMA".into());
    }
    owner_save(root,"session-lifecycle.json",&serde_json::to_vec_pretty(&json!({
        "passed":true,"families":model.families.len(),"snapshot_tokens":1,"final_tokens":total,
        "all_blobs":allocations.len(),"stable_compatibility":compatibility,"current_inventory":identity,
        "source_growth_preserved_cache_compatibility":true,"exact_all_snapshot_restored_bytes":true,
        "complete_replayed_F32_BF16_and_final_state_equal":true,"negative_admissions":rejected,
        "admitted_failure_poisoned_all":true,"future_transfer_after_failure_rejected":true,
        "snapshot_dma_ms":snapshot_ms,"restore_dma_ms":restore_ms,
        "scope":"supplied native two-family same-context inventory; injected callback failure, not physical DMA fault or serving qualification",
        "adopted":false
    })).map_err(|e|e.to_string())?)?;
    Ok(())
}
fn session_compare_pristine(
    model: &RetainedKdaSessionModel,
    actual: &[crate::session_cache::SequenceStateBlob],
    expected: &[Vec<crate::session_cache::SequenceStateBlob>],
) -> Result<(), String> {
    if expected.len() != model.families.len() {
        return Err("pristine family inventory incomplete".into());
    }
    for (index, &layer) in model.layers.iter().enumerate() {
        let local = actual
            .iter()
            .filter(|b| b.layer_idx == Some(layer))
            .collect::<Vec<_>>();
        if local.len() != expected[index].len() {
            return Err("pristine complete blob inventory".into());
        }
        for (a, e) in local.iter().zip(&expected[index]) {
            if a.allocation_name.rsplit('.').next() != e.allocation_name.rsplit('.').next()
                || a.dtype != e.dtype
                || a.shape != e.shape
                || a.bytes != e.bytes
            {
                return Err(format!(
                    "reset/cancel canonical state mismatch {}",
                    a.allocation_name
                ));
            }
        }
    }
    Ok(())
}
