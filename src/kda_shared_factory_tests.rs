// Measured singleton construction and continuation; not serving qualification.
fn shared_factory_controls(
    root: &Path,
    device: &Arc<CudaDevice>,
    pa: &KdaRawProducer,
    i0: &Arc<NativeRawOwner>,
    i1: &Arc<NativeRawOwner>,
    sa: &Arc<NativeStream>,
    sb: &Arc<NativeStream>,
    reserve: usize,
    transport: &str,
    math: &str,
    c: &[u8],
    cs: &OwnerSnapshot,
    second: &[u8],
) -> Result<Value, String> {
    let bank = KdaModelKernelBank {
        kernels: pa.kernels.clone(),
        transport: transport.into(),
        math: math.into(),
    };
    let construct = || unsafe {
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
    };
    let before = memory()?;
    let seed = construct()?;
    let after_seed = memory()?;
    let share = |capacity, lower, scale, epsilon, a_elements| unsafe {
        PreparedKdaModelFamily::from_model_storage_shared_arenas(
            RetainedPrefillKdaSource { inner: i0.clone() },
            RetainedDecodeKdaSource { inner: i1.clone() },
            pa.heads,
            pa.dim,
            capacity,
            reserve,
            lower,
            scale,
            epsilon,
            &bank,
            &seed,
            pointer(&pa.a_log),
            a_elements,
            pointer(&pa.bias),
            pa.heads * pa.dim,
        )
    };
    let mut rejected = Vec::new();
    for (label, capacity, lower, scale, epsilon, a_elements) in [
        ("zero capacity", 0, pa.lower, pa.scale, pa.epsilon, pa.heads),
        (
            "oversized capacity",
            2,
            pa.lower,
            pa.scale,
            pa.epsilon,
            pa.heads,
        ),
        (
            "different gate bound",
            1,
            pa.lower + 1.0,
            pa.scale,
            pa.epsilon,
            pa.heads,
        ),
        (
            "different Q scale",
            1,
            pa.lower,
            pa.scale * 2.0,
            pa.epsilon,
            pa.heads,
        ),
        (
            "different epsilon",
            1,
            pa.lower,
            pa.scale,
            pa.epsilon * 2.0,
            pa.heads,
        ),
        (
            "wrong parameter extent",
            1,
            pa.lower,
            pa.scale,
            pa.epsilon,
            pa.heads + 1,
        ),
    ] {
        if share(capacity, lower, scale, epsilon, a_elements).is_ok() || memory()? != after_seed {
            return Err(format!(
                "shared factory invalid contract allocated/admitted {label}"
            ));
        }
        rejected.push(label);
    }
    {
        let mut slot = seed
            .prefill
            .state
            .timeline
            .lock()
            .map_err(|_| "shared fixture seed mutex")?;
        slot.as_mut()
            .ok_or("shared fixture seed timeline")?
            .poisoned = true;
    }
    let poison = share(1, pa.lower, pa.scale, pa.epsilon, pa.heads);
    if poison.is_ok() || memory()? != after_seed {
        return Err("shared factory poisoned seed admitted/allocated".into());
    }
    // No GPU work was submitted by this negative control; restore the injected
    // host flag solely in this independent diagnostic seed before construction.
    {
        let mut slot = seed
            .prefill
            .state
            .timeline
            .lock()
            .map_err(|_| "shared fixture seed mutex")?;
        slot.as_mut()
            .ok_or("shared fixture seed timeline")?
            .poisoned = false;
    }
    let mut child = share(1, pa.lower, pa.scale, pa.epsilon, pa.heads)?;
    let after_shared = memory()?;
    if !Arc::ptr_eq(&seed.prefill.arena, &child.prefill.arena)
        || !Arc::ptr_eq(&seed.decode.arena, &child.decode.arena)
        || Arc::ptr_eq(&seed.prefill.state, &child.prefill.state)
        || !Arc::ptr_eq(&child.prefill.state, &child.decode.state)
    {
        return Err("shared factory arena/persistent ownership".into());
    }
    let weak_pref = Arc::downgrade(&seed.prefill.arena);
    let weak_dec = Arc::downgrade(&seed.decode.arena);
    let weak_seed_state = Arc::downgrade(&seed.prefill.state);
    let fresh = construct()?;
    let after_fresh = memory()?;
    {
        let b0 = seed.borrowed()?;
        let b1 = child.borrowed()?;
        let bf = fresh.borrowed()?;
        let bindings = [b0.prefill_binding(0), b1.prefill_binding(1)];
        let domain = ProducerResourceDomain::checked(&[
            b0.prefill_producer,
            b0.decode_producer,
            b1.prefill_producer,
            b1.decode_producer,
            bf.prefill_producer,
            bf.decode_producer,
        ])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, sa)?;
        if held.arena.arenas.len() != 1 {
            return Err("shared factory arena not deduplicated".into());
        }
        held.execute(|leases| {
            for b in &bindings {
                resource_enqueue(b.owner, b.producer, leases)?;
            }
            Ok(())
        })?;
        drop(held);
        for (label, owner) in [("seed", &seed.prefill), ("child", &child.prefill)] {
            let output = resource_output(owner)?;
            let state = resource_snapshot(owner)?;
            owner_save(root, &format!("shared-{label}-prefix-output.f32"), &output)?;
            owner_save(
                root,
                &format!("shared-{label}-prefix-state.json"),
                &serde_json::to_vec(&state).map_err(|e| e.to_string())?,
            )?;
            owner_compare(&output, c, "shared factory complete prefix output")?;
            owner_compare(
                &serde_json::to_vec(&state.blobs).map_err(|e| e.to_string())?,
                &serde_json::to_vec(&cs.blobs).map_err(|e| e.to_string())?,
                "shared factory complete prefix state",
            )?;
        }
        let fbindings = [bf.prefill_binding(2)];
        let fproof = ProducerResourceProof::checked(
            &domain,
            &fbindings,
            &model_family_identity(&fbindings)?,
        )?;
        {
            let mut held = ProducerResourceAdmission::checked(&fproof, sa)?;
            held.execute(|leases| {
                resource_enqueue(&fresh.prefill, &fresh.prefill_producer, leases)
            })?;
        }
        let dbindings = [bf.decode_binding(2)];
        let dproof = ProducerResourceProof::checked(
            &domain,
            &dbindings,
            &model_family_identity(&dbindings)?,
        )?;
        {
            let mut held = ProducerResourceAdmission::checked(&dproof, sb)?;
            held.execute(|leases| {
                shared_factory_decode(&fresh.decode, &fresh.decode_producer, leases)
            })?;
        }
    }
    let expected_output = resource_output(&fresh.decode)?;
    let expected_state = resource_snapshot(&fresh.decode)?;
    owner_compare(
        &expected_output,
        second,
        "shared baseline unchanged266 second row",
    )?;
    owner_save(
        root,
        "shared-fresh-continuation-output.f32",
        &expected_output,
    )?;
    owner_save(
        root,
        "shared-fresh-continuation-state.json",
        &serde_json::to_vec(&expected_state).map_err(|e| e.to_string())?,
    )?;
    drop(seed);
    if weak_seed_state.upgrade().is_some()
        || weak_pref.upgrade().is_none()
        || weak_dec.upgrade().is_none()
    {
        return Err("shared arena/seed-state lifetime after teardown".into());
    }
    {
        let borrowed = child.borrowed()?;
        let bindings = [borrowed.decode_binding(1)];
        let domain = ProducerResourceDomain::checked(&[
            borrowed.prefill_producer,
            borrowed.decode_producer,
        ])?;
        let proof =
            ProducerResourceProof::checked(&domain, &bindings, &model_family_identity(&bindings)?)?;
        let mut held = ProducerResourceAdmission::checked(&proof, sb)?;
        held.execute(|leases| {
            shared_factory_decode(&child.decode, &child.decode_producer, leases)
        })?;
    }
    let output = resource_output(&child.decode)?;
    let state = resource_snapshot(&child.decode)?;
    owner_save(root, "shared-child-continuation-output.f32", &output)?;
    owner_save(
        root,
        "shared-child-continuation-state.json",
        &serde_json::to_vec(&state).map_err(|e| e.to_string())?,
    )?;
    owner_compare(
        &output,
        &expected_output,
        "shared continuation after seed teardown",
    )?;
    owner_compare(
        &serde_json::to_vec(&state.blobs).map_err(|e| e.to_string())?,
        &serde_json::to_vec(&expected_state.blobs).map_err(|e| e.to_string())?,
        "shared complete continuation state",
    )?;
    let source_epoch = source_epoch_controls(root, &mut child, &fresh, i0, i1, reserve)?;
    let shared_delta = after_seed[0]
        .checked_sub(after_shared[0])
        .ok_or("shared allocation delta")?;
    let fresh_delta = after_shared[0]
        .checked_sub(after_fresh[0])
        .ok_or("fresh allocation delta")?;
    let result = json!({"passed":true,"source_epoch":source_epoch,"rejected":rejected,"poisoned_seed_rejected":true,"memory_before":before,"memory_after_seed":after_seed,"memory_after_shared":after_shared,"memory_after_fresh":after_fresh,"shared_family_physical_delta":shared_delta,"fresh_family_physical_delta":fresh_delta,"shared_arenas":true,"independent_persistent_carry":true,"all_complete_prefix_and_continuation_equal":true,"seed_teardown_carry_released_arenas_retained":true,"shared_identity":child.identity(1)?,"scope":"cold singleton explicit constructor and exact two-row continuation only; not long-capacity or model serving","adopted":false});
    owner_save(
        root,
        "shared-factory.json",
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    Ok(result)
}

fn shared_factory_decode(
    owner: &PreparedOwner,
    producer: &KdaRawProducer,
    leases: &[EnqueueLease<'_>],
) -> Result<(), String> {
    producer.enqueue(0, 1, owner)?;
    owner.decode_capture_in(model_arena_lease(owner, leases)?, &producer.view(1)?)
}
