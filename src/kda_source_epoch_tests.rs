fn epoch_stable_identity(f: &PreparedKdaModelFamily) -> Result<Value, String> {
    let b = f.borrowed()?;
    let ids = model_family_identity(&[b.prefill_binding(0)])?;
    let dids = model_family_identity(&[b.decode_binding(0)])?;
    let stable = |i: &Value| {
        let w = &i[0]["captured_workspace_binding"];
        json!({"state":i[0]["required_allocations"],"layout":i[0]["layout"],"arithmetic":i[0]["owner_arithmetic"],"producer_arithmetic":i[0]["producer_arithmetic"],"arena":w["recurrence_arena"],"prepared":w["producer"]["prepared_planes"],"params":w["producer"]["parameter_spans"],"a_log":w["producer"]["a_log"],"bias":w["producer"]["bias"]})
    };
    Ok(json!([stable(&ids), stable(&dids)]))
}
fn epoch_identity(f: &PreparedKdaModelFamily) -> Result<Value, String> {
    let b = f.borrowed()?;
    model_family_identity(&[b.prefill_binding(0)])
}
fn source_epoch_controls(
    root: &Path,
    child: &mut PreparedKdaModelFamily,
    fresh: &PreparedKdaModelFamily,
    i0: &Arc<NativeRawOwner>,
    i1: &Arc<NativeRawOwner>,
    reserve: usize,
) -> Result<Value, String> {
    let stable = epoch_stable_identity(child)?;
    let before = resource_snapshot(&child.prefill)?;
    let identity0 = epoch_identity(child)?;
    let memory_before = memory()?;
    let replace = |family: &mut PreparedKdaModelFamily,
                   p: Arc<NativeRawOwner>,
                   d: Arc<NativeRawOwner>,
                   e: [u64; 2]| {
        family.replace_raw_sources_quiescent(
            RetainedPrefillKdaSource { inner: p },
            RetainedDecodeKdaSource { inner: d },
            e,
        )
    };
    let mut rejected = Vec::new();
    for (label, p, d, e) in [
        ("stale epoch", i0.clone(), i1.clone(), [1, 0]),
        ("prefill wrong stream", i1.clone(), i1.clone(), [0, 0]),
        ("decode wrong stream", i0.clone(), i0.clone(), [0, 0]),
    ] {
        if replace(child, p, d, e).is_ok()
            || epoch_identity(child)? != identity0
            || memory()? != memory_before
        {
            return Err(format!("source epoch rejection mutated {label}"));
        }
        rejected.push(label);
    }
    let state_owner = child.prefill.state.clone();
    {
        let held = state_owner
            .timeline
            .lock()
            .map_err(|_| "epoch held fixture")?;
        if replace(child, i0.clone(), i1.clone(), [0, 0]).is_ok() {
            return Err("epoch held family admitted".into());
        }
        drop(held);
    }
    {
        let mut s = state_owner
            .timeline
            .lock()
            .map_err(|_| "epoch poison fixture")?;
        s.as_mut().ok_or("epoch missing fixture")?.poisoned = true;
    }
    if replace(child, i0.clone(), i1.clone(), [0, 0]).is_ok() || epoch_identity(child)? != identity0
    {
        return Err("epoch poisoned family admitted/mutated".into());
    }
    // Only a host flag was injected; no GPU callback ran. Test-local restoration.
    state_owner
        .timeline
        .lock()
        .map_err(|_| "epoch poison fixture")?
        .as_mut()
        .ok_or("epoch missing fixture")?
        .poisoned = false;
    let ps = child.prefill_producer.raw.stream.clone();
    let ds = child.decode_producer.raw.stream.clone();
    let p1 = Arc::new(NativeRawOwner::allocate(
        ps.clone(),
        i0.schema.clone(),
        reserve,
    )?);
    let d1 = Arc::new(NativeRawOwner::allocate(
        ds.clone(),
        i1.schema.clone(),
        reserve,
    )?);
    resource_copy_input(&ps, i0, &p1)?;
    resource_copy_input(&ds, i1, &d1)?;
    let weakp = Arc::downgrade(&p1);
    let weakd = Arc::downgrade(&d1);
    if replace(child, p1.clone(), d1.clone(), [0, 0])? != [1, 1] {
        return Err("epoch first count".into());
    }
    let identity1 = epoch_identity(child)?;
    // Reuse identical pointers: epoch must still distinguish graph/source ABA.
    if replace(child, p1, d1, [1, 1])? != [2, 2] {
        return Err("epoch repeated count".into());
    }
    let identity2 = epoch_identity(child)?;
    if identity1 == identity2 || identity0 == identity1 {
        return Err("epoch binding identity unchanged".into());
    }
    let rows = child
        .prefill_producer
        .capacity
        .checked_add(1)
        .ok_or("epoch grown rows")?;
    let schema = Schema::checked(
        RawContract::KdaBf16VectorGate,
        rows,
        child.prefill.heads,
        child.prefill.dim,
        child.prefill.dim,
    )?;
    let p2 = Arc::new(NativeRawOwner::allocate(ps.clone(), schema, reserve)?);
    let d2 = Arc::new(NativeRawOwner::allocate(
        ds.clone(),
        i1.schema.clone(),
        reserve,
    )?);
    // Real retained row0 copied into each positive row; this is a source
    // lifecycle control, not a new calibration/benchmark corpus or fidelity tier.
    for ((plane, dst), (_, src)) in p2.inventory().into_iter().zip(i0.inventory()) {
        let stride = plane.storage_bytes / rows;
        for row in 0..rows {
            timeline_copy(&ps, dst + (row * stride) as u64, src, stride)?;
        }
    }
    resource_copy_input(&ds, i1, &d2)?;
    let bad_decode = replace(child, p2.clone(), p2.clone(), [2, 2]);
    if bad_decode.is_ok() || epoch_identity(child)? != identity2 {
        return Err("epoch wrong decode extent admitted".into());
    }
    if replace(child, p2, d2, [2, 2])? != [3, 3] {
        return Err("epoch growth count".into());
    }
    if weakp.upgrade().is_some() || weakd.upgrade().is_some() {
        return Err("epoch retired raw owners retained".into());
    }
    if epoch_stable_identity(child)? != stable
        || serde_json::to_vec(&resource_snapshot(&child.prefill)?.blobs)
            .map_err(|e| e.to_string())?
            != serde_json::to_vec(&before.blobs).map_err(|e| e.to_string())?
    {
        return Err("epoch changed stable storage/carry".into());
    }
    {
        let b = child.borrowed()?;
        let bindings = [b.prefill_binding(0)];
        for old in [&identity0, &identity1, &identity2] {
            if ModelBindingProof::checked(&bindings, old).is_ok() {
                return Err("stale epoch proof admitted".into());
            }
        }
    }
    owner_save(
        root,
        "epoch-before-state.json",
        &serde_json::to_vec(&before).map_err(|e| e.to_string())?,
    )?;
    owner_save(
        root,
        "epoch-after-rebind-state.json",
        &serde_json::to_vec(&resource_snapshot(&child.prefill)?).map_err(|e| e.to_string())?,
    )?;
    for family in [&*child, fresh] {
        let b = family.borrowed()?;
        let bindings = [b.decode_binding(0)];
        let domain = ProducerResourceDomain::checked(&[b.prefill_producer, b.decode_producer])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, &ds)?;
        held.execute(|leases| shared_factory_decode(b.decode, b.decode_producer, leases))?;
    }
    let got = resource_output(&child.decode)?;
    let want = resource_output(&fresh.decode)?;
    let gs = resource_snapshot(&child.decode)?;
    let ws = resource_snapshot(&fresh.decode)?;
    owner_compare(&got, &want, "source epoch same-history third output")?;
    if serde_json::to_vec(&gs.blobs).map_err(|e| e.to_string())?
        != serde_json::to_vec(&ws.blobs).map_err(|e| e.to_string())?
    {
        return Err("source epoch complete third state mismatch".into());
    }
    for (name, bytes) in [
        ("epoch-child-output.f32", got),
        ("epoch-baseline-output.f32", want),
        (
            "epoch-child-state.json",
            serde_json::to_vec(&gs).map_err(|e| e.to_string())?,
        ),
        (
            "epoch-baseline-state.json",
            serde_json::to_vec(&ws).map_err(|e| e.to_string())?,
        ),
    ] {
        owner_save(root, name, &bytes)?;
    }
    let result = json!({"passed":true,"epochs":[3,3],"input_capacity":child.prefill_producer.input_capacity,"prepared_capacity":child.prefill_producer.capacity,"rejected":rejected,"held_rejected":true,"poison_rejected":true,"wrong_decode_extent_rejected":true,"same_pointer_epoch_distinct":true,"old_binding_rejected":true,"retired_sources_released":true,"stable_storage_and_carry_unchanged":true,"complete_continuation_equal":true,"identity_before":identity0,"identity_after":epoch_identity(child)?,"stable_identity":stable,"scope":"exclusive quiescent source replacement at unchanged measured arena/prepared capacity, third row repeats retained row1; no workspace growth or serving/model epoch qualification","adopted":false});
    owner_save(
        root,
        "source-epoch.json",
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    Ok(result)
}
