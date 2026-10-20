fn workspace_bf16(owner: &PreparedOwner, rows: usize) -> Result<Vec<u8>, String> {
    owner.stream.drain()?;
    let mut bytes = vec![
        0u8;
        rows.checked_mul(owner.heads)
            .and_then(|x| x.checked_mul(owner.dim))
            .and_then(|x| x.checked_mul(2))
            .ok_or("workspace BF16 extent")?
    ];
    check(
        unsafe {
            sys::lib().cuMemcpyDtoH_v2(
                bytes.as_mut_ptr().cast(),
                pointer(&owner.arena.bf16_emitted),
                bytes.len(),
            )
        },
        "workspace complete BF16 read",
    )?;
    Ok(bytes)
}
fn workspace_row_copy(
    stream: &NativeStream,
    all: &NativeRawOwner,
    row: usize,
    dst: &NativeRawOwner,
) -> Result<(), String> {
    if row >= all.schema.planes[0].shape[0] || dst.schema.planes[0].shape[0] != 1 {
        return Err("workspace row copy extent".into());
    }
    for ((a, src), (b, to)) in all.inventory().into_iter().zip(dst.inventory()) {
        let stride = a.storage_bytes / a.shape[0];
        if stride != b.storage_bytes {
            return Err("workspace row copy shape".into());
        }
        timeline_copy(stream, to, src + (row * stride) as u64, stride)?;
    }
    Ok(())
}
fn workspace_stable(f: &PreparedKdaModelFamily) -> Result<Value, String> {
    let id = epoch_stable_identity(f)?;
    Ok(json!(id.as_array().ok_or("workspace stable array")?.iter().map(|i| json!({"state":i["state"],"layout":i["layout"],"arithmetic":i["arithmetic"],"producer_arithmetic":i["producer_arithmetic"],"params":i["params"],"a_log":i["a_log"],"bias":i["bias"]})).collect::<Vec<_>>()))
}
fn workspace_capacity_controls(
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
    let mut target = unsafe {
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
    }?;
    let ps = target.prefill.stream.clone();
    let ds = target.decode.stream.clone();
    let old_pref = Arc::downgrade(&target.prefill.arena);
    let old_dec = Arc::downgrade(&target.decode.arena);
    let capacities = [16 + 1, 2 * 16 - 1];
    let graph_rows = 13usize;
    let batch_end = 1 + capacities.iter().sum::<usize>();
    let total = batch_end + graph_rows;
    if total > plan.native.rows {
        return Err("workspace real corpus too short".into());
    }
    let raw = plan
        .native
        .raw_inputs
        .iter()
        .map(|id| fs::read(&id.path).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let input = kda_raw_slice(&raw, 0, total, pa.heads, pa.dim)?;
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
    kda_raw_upload(&all, &input)?;
    for f in [&baseline, &target] {
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
            "workspace initial prefix",
        )?;
        owner_compare(
            &serde_json::to_vec(&resource_snapshot(b.prefill)?.blobs).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&initial_state.blobs).map_err(|e| e.to_string())?,
            "workspace initial complete state",
        )?;
    }
    let stable = workspace_stable(&target)?;
    let width = pa.heads * pa.dim * 4;
    let before_identity = epoch_identity(&target)?;
    let mut f32_out = initial_output.to_vec();
    let mut bf16_out = workspace_bf16(&target.prefill, 1)?;
    let mut start = 1usize;
    let mut growth = Vec::new();
    for (epoch, &capacity) in capacities.iter().enumerate() {
        let snapshot = resource_snapshot(&target.prefill)?;
        let prior = epoch_identity(&target)?;
        let mem = memory()?;
        if target
            .replace_prefill_workspace_quiescent(
                RetainedPrefillKdaSource { inner: all.clone() },
                0,
                epoch as u64,
                reserve,
            )
            .is_ok()
            || epoch_identity(&target)? != prior
            || memory()? != mem
        {
            return Err("workspace invalid capacity mutated/allocated".into());
        }
        for (label, raw, cap, want_epoch) in [
            ("stale epoch", all.clone(), capacity, epoch as u64 + 1),
            (
                "capacity beyond raw storage",
                all.clone(),
                total + 1,
                epoch as u64,
            ),
            ("foreign prefill stream", i1.clone(), 1, epoch as u64),
        ] {
            if target
                .replace_prefill_workspace_quiescent(
                    RetainedPrefillKdaSource { inner: raw },
                    cap,
                    want_epoch,
                    reserve,
                )
                .is_ok()
                || epoch_identity(&target)? != prior
                || memory()? != mem
            {
                return Err(format!(
                    "workspace invalid contract admitted/mutated {label}"
                ));
            }
        }
        {
            let arena = target.prefill.arena.clone();
            let held = arena.lease()?;
            if target
                .replace_prefill_workspace_quiescent(
                    RetainedPrefillKdaSource { inner: all.clone() },
                    capacity,
                    epoch as u64,
                    reserve,
                )
                .is_ok()
            {
                return Err("workspace held shared arena admitted".into());
            }
            drop(held);
        }
        let new_epoch = target.replace_prefill_workspace_quiescent(
            RetainedPrefillKdaSource { inner: all.clone() },
            capacity,
            epoch as u64,
            reserve,
        )?;
        let after = memory()?;
        if new_epoch != epoch as u64 + 1
            || workspace_stable(&target)? != stable
            || serde_json::to_vec(&resource_snapshot(&target.prefill)?.blobs)
                .map_err(|e| e.to_string())?
                != serde_json::to_vec(&snapshot.blobs).map_err(|e| e.to_string())?
        {
            return Err("workspace growth changed persistent state/params".into());
        }
        let b = target.borrowed()?;
        let bindings = [b.prefill_binding(0)];
        if ModelBindingProof::checked(&bindings, &prior).is_ok()
            || ModelBindingProof::checked(&bindings, &before_identity).is_ok()
        {
            return Err("workspace stale proof admitted".into());
        }
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        {
            let mut held = ProducerResourceAdmission::checked(&proof, &ps)?;
            held.execute(|leases| {
                b.prefill_producer.enqueue(start, capacity, b.prefill)?;
                b.prefill.prefill_in(
                    model_arena_lease(b.prefill, leases)?,
                    &b.prefill_producer.view(capacity)?,
                    start as u64,
                    capacity,
                )
            })?;
        }
        ps.drain()?;
        let output = owner_array(b.prefill, "emitted", capacity * pa.heads * pa.dim)?;
        owner_compare(
            &output,
            &reference[start * width..(start + capacity) * width],
            "workspace grown complete corpus prefix",
        )?;
        f32_out.extend(output);
        bf16_out.extend(workspace_bf16(b.prefill, capacity)?);
        growth.push(json!({"capacity":capacity,"epoch":new_epoch,"before_memory":mem,"after_memory":after,"arena_requested_bytes":b.prefill.arena.requested_bytes,"producer_allocated_bytes":b.prefill_producer.allocated_bytes,"stable_carry_and_params":true,"complete_prefix_exact":true}));
        start += capacity;
    }
    if old_pref.upgrade().is_none() || old_dec.upgrade().is_none() {
        return Err("workspace other old arena owner lost".into());
    }
    // Independent unchanged singleton GPU path consumes the same corpus history.
    {
        let b = baseline.borrowed()?;
        let bindings = [b.prefill_binding(0)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        for row in 1..batch_end {
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
            owner_compare(
                &resource_output(b.prefill)?,
                &reference[row * width..(row + 1) * width],
                "workspace unchanged singleton corpus output",
            )?;
        }
    }
    owner_compare(
        &serde_json::to_vec(&resource_snapshot(&target.prefill)?.blobs)
            .map_err(|e| e.to_string())?,
        &serde_json::to_vec(&resource_snapshot(&baseline.prefill)?.blobs)
            .map_err(|e| e.to_string())?,
        "workspace complete grown batch state",
    )?;
    let staging = Arc::new(NativeRawOwner::allocate(
        ds.clone(),
        i1.schema.clone(),
        reserve,
    )?);
    {
        let b = target.borrowed()?;
        let bindings = [b.decode_binding(0)];
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
            "workspace-continuation-graph.json",
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
                "workspace graph continuation complete corpus output",
            )?;
            f32_out.extend(out);
            bf16_out.extend(workspace_bf16(b.decode, 1)?);
        }
    }
    {
        let b = baseline.borrowed()?;
        let bindings = [b.decode_binding(0)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        for row in batch_end..total {
            let mut held = ProducerResourceAdmission::checked(&proof, &ds)?;
            held.execute(|leases| {
                workspace_row_copy(&ds, &all, row, &b.decode_producer.raw)?;
                shared_factory_decode(b.decode, b.decode_producer, leases)
            })?;
            owner_compare(
                &resource_output(b.decode)?,
                &reference[row * width..(row + 1) * width],
                "workspace unchanged decode corpus output",
            )?;
        }
    }
    let ts = resource_snapshot(&target.decode)?;
    let bs = resource_snapshot(&baseline.decode)?;
    owner_compare(
        &serde_json::to_vec(&ts.blobs).map_err(|e| e.to_string())?,
        &serde_json::to_vec(&bs.blobs).map_err(|e| e.to_string())?,
        "workspace complete final continuation state",
    )?;
    let bf16 = fs::read(&plan.bf16_reference.path).map_err(|e| e.to_string())?;
    owner_compare(
        &bf16_out,
        &bf16[..total * width / 2],
        "workspace entire BF16 corpus prefix",
    )?;
    owner_compare(
        &f32_out,
        &reference[..total * width],
        "workspace entire F32 corpus prefix",
    )?;
    for (name, bytes) in [
        ("workspace-output.f32", f32_out),
        ("workspace-output.bf16", bf16_out),
        (
            "workspace-final-state.json",
            serde_json::to_vec(&ts).map_err(|e| e.to_string())?,
        ),
        (
            "workspace-baseline-final-state.json",
            serde_json::to_vec(&bs).map_err(|e| e.to_string())?,
        ),
    ] {
        owner_save(root, name, &bytes)?;
    }
    drop(baseline);
    if old_pref.upgrade().is_some() || old_dec.upgrade().is_none() {
        return Err("workspace shared retirement lifetime".into());
    }
    let caller_after = [
        workspace_raw_read(parent_i0)?,
        workspace_raw_read(parent_i1)?,
    ];
    if caller_before != caller_after {
        return Err("workspace fixture mutated caller raw inputs".into());
    }
    for (role, planes) in [("before", &caller_before), ("after", &caller_after)] {
        for (mode, raw) in planes.iter().enumerate() {
            for (plane, bytes) in raw.iter().enumerate() {
                owner_save(
                    root,
                    &format!("workspace-parent-{role}-{mode}-plane{plane}.bf16"),
                    bytes,
                )?;
            }
        }
    }
    let result = json!({"passed":true,"caller_sources_unchanged":true,"isolated_fixture_source_owners":true,"rows":total,"batch_prefix_rows":batch_end,"graph_continuation_rows":graph_rows,"growth":growth,"old_shared_arena_retained_then_released":true,"distinct_decode_arena_retained":true,"full_F32_BF16_corpus_prefix_equal":true,"complete_final_state_equal":true,"no_replacement_carry_allocation":true,"scope":"bounded real corpus capacity epochs, unchanged scalar batch baseline plus actual graph continuation; no serving budget or model acceptance","adopted":false});
    owner_save(
        root,
        "workspace-capacity.json",
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    Ok(result)
}

fn workspace_raw_read(owner: &NativeRawOwner) -> Result<Vec<Vec<u8>>, String> {
    owner.stream.drain()?;
    owner
        .inventory()
        .into_iter()
        .map(|(plane, ptr)| {
            let mut b = vec![0u8; plane.storage_bytes];
            check(
                unsafe { sys::lib().cuMemcpyDtoH_v2(b.as_mut_ptr().cast(), ptr, b.len()) },
                "workspace caller raw read",
            )?;
            Ok(b)
        })
        .collect()
}
