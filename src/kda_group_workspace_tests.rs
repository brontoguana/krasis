fn grouped_workspace_controls(
    root: &Path,
    pa: &KdaRawProducer,
    parent_i0: &Arc<NativeRawOwner>,
    parent_i1: &Arc<NativeRawOwner>,
    reserve: usize,
    transport: &str,
    math: &str,
    plan: &TilingPlan,
    reference: &[u8],
    initial_output: &[u8],
    initial_state: &OwnerSnapshot,
) -> Result<Value, String> {
    let caller_before = [
        workspace_raw_read(parent_i0)?,
        workspace_raw_read(parent_i1)?,
    ];
    let i0 = Arc::new(NativeRawOwner::allocate(
        parent_i0.stream.clone(),
        parent_i0.schema.clone(),
        reserve,
    )?);
    let i1 = Arc::new(NativeRawOwner::allocate(
        parent_i1.stream.clone(),
        parent_i1.schema.clone(),
        reserve,
    )?);
    resource_copy_input(&i0.stream, parent_i0, &i0)?;
    resource_copy_input(&i1.stream, parent_i1, &i1)?;
    i0.stream.drain()?;
    i1.stream.drain()?;
    let bank = KdaModelKernelBank {
        kernels: pa.kernels.clone(),
        transport: transport.into(),
        math: math.into(),
    };
    let baseline = unsafe {
        PreparedKdaModelFamily::from_model_storage(
            RetainedPrefillKdaSource { inner: i0.clone() },
            RetainedDecodeKdaSource { inner: i1.clone() },
            pa.heads,
            pa.dim,
            1,
            reserve,
            pa.lower,
            pa.scale,
            pa.epsilon,
            &bank,
            pointer(&pa.a_log),
            pa.heads,
            pointer(&pa.bias),
            pa.heads * pa.dim,
        )
    }?;
    let mut families = Vec::new();
    for _ in 0..2 {
        families.push(unsafe {
            PreparedKdaModelFamily::from_model_storage_shared_arenas(
                RetainedPrefillKdaSource { inner: i0.clone() },
                RetainedDecodeKdaSource { inner: i1.clone() },
                pa.heads,
                pa.dim,
                1,
                reserve,
                pa.lower,
                pa.scale,
                pa.epsilon,
                &bank,
                &baseline,
                pointer(&pa.a_log),
                pa.heads,
                pointer(&pa.bias),
                pa.heads * pa.dim,
            )
        }?);
    }
    let ps = i0.stream.clone();
    let ds = i1.stream.clone();
    let capacity = 17usize;
    let batch_end = 1 + capacity;
    let graph_rows = 13usize;
    let total = batch_end + graph_rows;
    let raw = plan
        .native
        .raw_inputs
        .iter()
        .map(|id| fs::read(&id.path).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let all = Arc::new(NativeRawOwner::allocate(
        ps.clone(),
        Schema::checked(
            RawContract::KdaBf16VectorGate,
            total,
            pa.heads,
            pa.dim,
            pa.dim,
        )?,
        reserve,
    )?);
    kda_raw_upload(&all, &kda_raw_slice(&raw, 0, total, pa.heads, pa.dim)?)?;
    let pristine = families
        .iter()
        .map(|f| resource_snapshot(&f.prefill).map(|s| s.blobs))
        .collect::<Result<Vec<_>, _>>()?;
    owner_save(
        root,
        "session-pristine-family-states.json",
        &serde_json::to_vec(&pristine).map_err(|e| e.to_string())?,
    )?;
    for f in std::iter::once(&baseline).chain(families.iter()) {
        let b = f.borrowed()?;
        let bindings = [b.prefill_binding(0)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, &ps)?;
        held.execute(|leases| {
            workspace_row_copy(&ps, &all, 0, &b.prefill_producer.raw)?;
            resource_enqueue(b.prefill, b.prefill_producer, leases)
        })?;
        owner_compare(
            &resource_output(b.prefill)?,
            initial_output,
            "group initial real prefix",
        )?;
        owner_compare(
            &serde_json::to_vec(&resource_snapshot(b.prefill)?.blobs).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&initial_state.blobs).map_err(|e| e.to_string())?,
            "group initial nonzero full carry",
        )?;
    }
    if Arc::ptr_eq(&families[0].prefill.state, &families[1].prefill.state)
        || pointer(&families[0].prefill_producer.a_log)
            == pointer(&families[1].prefill_producer.a_log)
    {
        return Err("group family state/params not independent".into());
    }
    let layers = [0, 1];
    let material = |value: &serde_json::Value| -> Result<String, String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(value).map_err(|e| e.to_string())?)
        ))
    };
    let checked_source = plan
        .session_source_identity
        .as_ref()
        .ok_or("session source identity field absent")?;
    owner_checked(checked_source)?;
    if !plan.native.admission.iter().any(|x| {
        x.path == checked_source.path
            && x.bytes == checked_source.bytes
            && x.sha256 == checked_source.sha256
    }) {
        return Err("session source identity absent from complete admission".into());
    }
    let stamp = KdaSessionModelStamp {
        model_sha256: material(
            &json!({"scope":"frozen native diagnostic fixture","heads":pa.heads,"dim":pa.dim}),
        )?,
        weights_sha256: material(&json!([plan.native.a_log, plan.native.bias]))?,
        checkpoint_sha256: material(&json!(plan.native.raw_inputs))?,
        config_sha256: plan.native.runtime_config.sha256.clone(),
        source_sha256: checked_source.sha256.clone(),
        binary_sha256: plan.native.executable.sha256.clone(),
        epoch: 0,
    };
    let mut session_model = RetainedKdaSessionModel::new(families, layers.to_vec(), stamp)?;
    let compatibility = session_model.stable_compatibility.clone();
    let allocations = session_model.registry()?.allocations().to_vec();
    let mut store =
        crate::gpu_decode::GpuDecodeStore::diagnostic_session_dma_store_rust(ps.device.ordinal())?;
    let first_identity = workspace_inventory_identity(&session_model.families, &layers)?;
    let (saved, snapshot_ms) =
        session_model.snapshot_pinned(&first_identity, 1, &mut [&mut store])?;
    session_model.validate_snapshot(&compatibility, 1, &saved.iter().collect::<Vec<_>>())?;
    owner_save(
        root,
        "session-row1-snapshot.json",
        &serde_json::to_vec(&saved).map_err(|e| e.to_string())?,
    )?;

    let identity = workspace_inventory_identity(&session_model.families, &layers)?;
    let before_states = session_model
        .families
        .iter()
        .map(|f| resource_snapshot(&f.prefill))
        .collect::<Result<Vec<_>, _>>()?;
    let stable = session_model
        .families
        .iter()
        .map(workspace_stable)
        .collect::<Result<Vec<_>, _>>()?;
    let old_pref = Arc::downgrade(&session_model.families[0].prefill.arena);
    let before_memory = memory()?;
    for case in 0..3 {
        let sources = if case == 1 {
            vec![
                RetainedPrefillKdaSource { inner: all.clone() },
                RetainedPrefillKdaSource { inner: i1.clone() },
            ]
        } else {
            vec![
                RetainedPrefillKdaSource { inner: all.clone() },
                RetainedPrefillKdaSource { inner: all.clone() },
            ]
        };
        let epochs = if case == 0 { [0, 1] } else { [0, 0] };
        let expected = if case == 2 {
            json!([])
        } else {
            identity.clone()
        };
        if replace_grouped_prefill_workspaces_quiescent(
            &mut session_model.families,
            &layers,
            sources,
            &[capacity, capacity],
            &epochs,
            &expected,
            reserve,
        )
        .is_ok()
            || workspace_inventory_identity(&session_model.families, &layers)? != identity
            || memory()? != before_memory
        {
            return Err(format!(
                "group late invalid admission changed storage:{case}"
            ));
        }
    }
    {
        let state = session_model.families[1].prefill.state.clone();
        let held = state
            .timeline
            .try_lock()
            .map_err(|_| "group negative lock")?;
        if replace_grouped_prefill_workspaces_quiescent(
            &mut session_model.families,
            &layers,
            vec![
                RetainedPrefillKdaSource { inner: all.clone() },
                RetainedPrefillKdaSource { inner: all.clone() },
            ],
            &[capacity, capacity],
            &[0, 0],
            &identity,
            reserve,
        )
        .is_ok()
        {
            return Err("group late held family admitted".into());
        }
        drop(held);
        if workspace_inventory_identity(&session_model.families, &layers)? != identity
            || memory()? != before_memory
        {
            return Err("group late held family mutated/allocated".into());
        }
    }
    let mut arena_negatives = Vec::new();
    for role in ["prefill", "decode"] {
        let last = session_model
            .families
            .last()
            .ok_or("growth late family absent")?;
        let arena = if role == "prefill" {
            last.prefill.arena.clone()
        } else {
            last.decode.arena.clone()
        };
        let held = arena.lease()?;
        let result = replace_grouped_prefill_workspaces_quiescent(
            &mut session_model.families,
            &layers,
            vec![
                RetainedPrefillKdaSource { inner: all.clone() },
                RetainedPrefillKdaSource { inner: all.clone() },
            ],
            &[capacity, capacity],
            &[0, 0],
            &identity,
            reserve,
        );
        drop(held);
        if result.is_ok()
            || workspace_inventory_identity(&session_model.families, &layers)? != identity
            || memory()? != before_memory
        {
            return Err(format!(
                "growth held {role} arena admitted/mutated/allocated"
            ));
        }
        for (f, want) in session_model.families.iter().zip(&before_states) {
            owner_compare(
                &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                    .map_err(|e| e.to_string())?,
                &serde_json::to_vec(&want.blobs).map_err(|e| e.to_string())?,
                "held growth arena full nonzero state unchanged",
            )?;
        }
        arena_negatives.push(role);
    }
    owner_save(root,"group-both-arena-admission.json",&serde_json::to_vec_pretty(&json!({"passed":true,"rejected_roles":arena_negatives,"full_nonzero_state_unchanged":true,"no_allocation_or_publication":true,"current_binding_identity":identity,"scope":"complete supplied native context; not whole-serving execution"})).map_err(|e|e.to_string())?)?;
    let epochs = replace_grouped_prefill_workspaces_quiescent(
        &mut session_model.families,
        &layers,
        vec![
            RetainedPrefillKdaSource { inner: all.clone() },
            RetainedPrefillKdaSource { inner: all.clone() },
        ],
        &[capacity, capacity],
        &[0, 0],
        &identity,
        reserve,
    )?;
    let after_memory = memory()?;
    if epochs != [1, 1]
        || !Arc::ptr_eq(
            &session_model.families[0].prefill.arena,
            &session_model.families[1].prefill.arena,
        )
        || session_model.families[0].prefill_producer.prepared.buffers[0].address()
            == session_model.families[1].prefill_producer.prepared.buffers[0].address()
    {
        return Err("group shared arena/independent prepared publication".into());
    }
    let width = pa.heads * pa.dim * 4;
    let mut all_outputs = Vec::new();
    let mut all_bf16 = Vec::new();
    for (i, f) in session_model.families.iter().enumerate() {
        if workspace_stable(f)? != stable[i]
            || serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                .map_err(|e| e.to_string())?
                != serde_json::to_vec(&before_states[i].blobs).map_err(|e| e.to_string())?
        {
            return Err("group publication mutated nonzero carry/params".into());
        }
        let b = f.borrowed()?;
        let bindings = [b.prefill_binding(i)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        {
            let mut held = ProducerResourceAdmission::checked(&proof, &ps)?;
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
        ps.drain()?;
        let batch = owner_array(b.prefill, "emitted", capacity * pa.heads * pa.dim)?;
        owner_compare(
            &batch,
            &reference[width..batch_end * width],
            "group complete parallel GPU prefix",
        )?;
        let mut output = initial_output.to_vec();
        output.extend(batch);
        all_outputs.push(output);
        let mut bf16 =
            fs::read(&plan.bf16_reference.path).map_err(|e| e.to_string())?[..width / 2].to_vec();
        bf16.extend(workspace_bf16(b.prefill, capacity)?);
        all_bf16.push(bf16);
    }
    for row in 1..batch_end {
        let b = baseline.borrowed()?;
        let bindings = [b.prefill_binding(0)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, &ps)?;
        held.execute(|leases| {
            workspace_row_copy(&ps, &all, row, &b.prefill_producer.raw)?;
            b.prefill_producer.enqueue(0, 1, b.prefill)?;
            b.prefill.prefill_in(
                model_arena_lease(b.prefill, leases)?,
                &b.prefill_producer.view(1)?,
                row as u64,
                1,
            )
        })?;
    }
    for f in &session_model.families {
        owner_compare(
            &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                .map_err(|e| e.to_string())?,
            &serde_json::to_vec(&resource_snapshot(&baseline.prefill)?.blobs)
                .map_err(|e| e.to_string())?,
            "group complete parallel batch state",
        )?;
    }
    let staging = Arc::new(NativeRawOwner::allocate(
        ds.clone(),
        i1.schema.clone(),
        reserve,
    )?);
    for (i, f) in session_model.families.iter().enumerate() {
        let b = f.borrowed()?;
        let bindings = [b.decode_binding(i)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let (graph, inventory) = resource_graph_capture(
            &proof,
            b.decode,
            b.decode_producer,
            &staging,
            &b.decode_producer.raw,
        )?;
        owner_save(
            root,
            &format!("group-workspace-graph-{i}.json"),
            &serde_json::to_vec(&inventory).map_err(|e| e.to_string())?,
        )?;
        for row in batch_end..total {
            let mut held = ProducerResourceAdmission::checked(&proof, &ds)?;
            held.execute(|leases| {
                workspace_row_copy(&ds, &all, row, &staging)?;
                b.decode
                    .launch_graph_in(model_arena_lease(b.decode, leases)?, &graph)
            })?;
            let out = resource_output(b.decode)?;
            owner_compare(
                &out,
                &reference[row * width..(row + 1) * width],
                "group full graph continuation output",
            )?;
            all_outputs[i].extend(out);
            all_bf16[i].extend(workspace_bf16(b.decode, 1)?);
        }
    }
    for row in batch_end..total {
        let b = baseline.borrowed()?;
        let bindings = [b.decode_binding(0)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, &ds)?;
        held.execute(|leases| {
            workspace_row_copy(&ds, &all, row, &b.decode_producer.raw)?;
            shared_factory_decode(b.decode, b.decode_producer, leases)
        })?;
    }
    let baseline_state = resource_snapshot(&baseline.decode)?;
    owner_save(
        root,
        "group-workspace-baseline-state.json",
        &serde_json::to_vec(&baseline_state).map_err(|e| e.to_string())?,
    )?;
    let bf16_reference = fs::read(&plan.bf16_reference.path).map_err(|e| e.to_string())?;
    for (i, f) in session_model.families.iter().enumerate() {
        let state = resource_snapshot(&f.decode)?;
        owner_compare(
            &serde_json::to_vec(&state.blobs).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&baseline_state.blobs).map_err(|e| e.to_string())?,
            "group complete final state",
        )?;
        owner_compare(
            &all_outputs[i],
            &reference[..total * width],
            "group complete F32 prefix",
        )?;
        owner_compare(
            &all_bf16[i],
            &bf16_reference[..total * width / 2],
            "group complete BF16 prefix",
        )?;
        owner_save(
            root,
            &format!("group-workspace-output-{i}.f32"),
            &all_outputs[i],
        )?;
        owner_save(
            root,
            &format!("group-workspace-output-{i}.bf16"),
            &all_bf16[i],
        )?;
        owner_save(
            root,
            &format!("group-workspace-state-{i}.json"),
            &serde_json::to_vec(&state).map_err(|e| e.to_string())?,
        )?;
    }
    session_pinned_replay_controls(
        root,
        &mut session_model,
        &pristine,
        &mut store,
        &allocations,
        &compatibility,
        &saved,
        &all,
        &staging,
        capacity,
        batch_end,
        total,
        reference,
        plan,
        snapshot_ms,
        reserve,
    )?;
    drop(baseline);
    if old_pref.upgrade().is_some()
        || workspace_inventory_identity(&session_model.families, &layers)? == identity
    {
        return Err("group retired arena/proof not retired".into());
    }
    if caller_before
        != [
            workspace_raw_read(parent_i0)?,
            workspace_raw_read(parent_i1)?,
        ]
    {
        return Err("group fixture mutated parent input".into());
    }
    let result = json!({"passed":true,"families":session_model.families.len(),"groups":1,"capacity":capacity,"rows":total,"graph_rows":graph_rows,"epochs":epochs,"before_memory":before_memory,"after_memory":after_memory,"arena_requested_bytes":session_model.families[0].prefill.arena.requested_bytes,"producer_owned_bytes":session_model.families.iter().map(|f| f.prefill_producer.allocated_bytes).collect::<Vec<_>>(),"late_admission_rejection_before_allocation":true,"nonzero_carry_params_unchanged":true,"one_shared_arena_independent_prepared":true,"complete_F32_BF16_prefix_and_final_state_equal":true,"caller_sources_unchanged":true,"old_arena_retired":true,"scope":"supplied two-family single-device inventory; real full-model/multi-device coordination not qualified","adopted":false});
    owner_save(
        root,
        "group-workspace.json",
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    Ok(result)
}
