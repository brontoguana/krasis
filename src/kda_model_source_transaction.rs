// Quiescent complete supplied-context source publication, before request work.
// External graph/resource proofs borrow these families and must retire first.
pub(crate) fn replace_model_raw_sources_quiescent(
    families: &mut [PreparedKdaModelFamily],
    layers: &[usize],
    replacements: Vec<(RetainedPrefillKdaSource, RetainedDecodeKdaSource)>,
    expected_epochs: &[[u64; 2]],
    expected_inventory: &serde_json::Value,
) -> Result<Vec<[u64; 2]>, String> {
    if families.is_empty()
        || families.len() != layers.len()
        || replacements.len() != families.len()
        || expected_epochs.len() != families.len()
        || layers.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("source transaction incomplete/noncanonical inventory".into());
    }
    let context = *families[0].prefill.device.cu_primary_ctx();
    if families
        .iter()
        .any(|f| *f.prefill.device.cu_primary_ctx() != context)
    {
        return Err(
            "source transaction needs complete outer admission for multiple contexts".into(),
        );
    }
    families[0]
        .prefill
        .device
        .bind_to_thread()
        .map_err(|e| e.to_string())?;
    if &workspace_inventory_identity(families, layers)? != expected_inventory {
        return Err("source transaction stale current inventory".into());
    }
    // Stage every validation before acquiring guards or submitting any GPU work.
    let plans = families
        .iter()
        .zip(replacements)
        .zip(expected_epochs)
        .map(|((f, (prefill, decode)), epochs)| {
            f.borrowed()?;
            if *epochs
                != [
                    f.prefill_producer.source_epoch,
                    f.decode_producer.source_epoch,
                ]
            {
                return Err("source transaction stale member epoch".to_string());
            }
            let p = f
                .prefill_producer
                .validate_replacement(&prefill.inner, false)?;
            let d = f
                .decode_producer
                .validate_replacement(&decode.inner, true)?;
            Ok((prefill, decode, p, d))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut states = Vec::new();
    let mut arenas: Vec<Arc<PreparedArena>> = Vec::new();
    for f in families.iter() {
        if states.iter().any(|s| Arc::ptr_eq(s, &f.prefill.state)) {
            return Err("source transaction duplicate persistent family".into());
        }
        states.push(f.prefill.state.clone());
        for o in [&f.prefill, &f.decode] {
            o.stream.validate()?;
            if !arenas.iter().any(|a| Arc::ptr_eq(a, &o.arena)) {
                arenas.push(o.arena.clone());
            }
        }
    }
    let mut slots = states
        .iter()
        .map(|s| {
            s.timeline
                .try_lock()
                .map_err(|_| "source transaction family concurrently leased/poisoned".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    for slot in &slots {
        let t = slot.as_ref().ok_or("source transaction timeline absent")?;
        if t.poisoned {
            return Err("source transaction family poisoned".into());
        }
        t.phases
            .checked_add(1)
            .ok_or("source transaction phase overflow")?;
        for stream in &t.streams {
            stream.validate()?;
        }
    }
    let _leases = arenas
        .iter()
        .map(|a| a.lease())
        .collect::<Result<Vec<_>, _>>()?;
    for slot in &mut slots {
        slot.as_mut().expect("validated timeline").poisoned = true;
    }
    // Any failed completion poisons the entire inventory and publishes no source.
    for slot in &slots {
        let t = slot.as_ref().expect("validated timeline");
        if t.recorded {
            check(
                unsafe { sys::lib().cuEventSynchronize(t.event) },
                "source transaction prior completion",
            )?;
        }
        for stream in &t.streams {
            stream.drain()?;
        }
    }
    // All replacements are validated and all prior users completed. Publication
    // consists only of owned Arc moves and already-checked integer assignments.
    let mut epochs = Vec::with_capacity(families.len());
    for (f, (prefill, decode, p, d)) in families.iter_mut().zip(plans) {
        f.prefill_producer.raw = prefill.inner;
        f.prefill_producer.input_capacity = p.0;
        f.prefill_producer.requested_bytes = p.1;
        f.prefill_producer.source_epoch = p.2;
        f.decode_producer.raw = decode.inner;
        f.decode_producer.input_capacity = d.0;
        f.decode_producer.requested_bytes = d.1;
        f.decode_producer.source_epoch = d.2;
        epochs.push([p.2, d.2]);
    }
    for slot in &mut slots {
        let t = slot.as_mut().expect("validated timeline");
        t.recorded = false;
        t.phases += 1;
        t.poisoned = false;
    }
    Ok(epochs)
}
