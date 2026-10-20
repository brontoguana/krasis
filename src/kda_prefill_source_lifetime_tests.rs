fn prefill_source_lifetime_controls(
    root: &Path,
    model: &mut RetainedKdaSessionModel,
    reserve: usize,
) -> Result<(), String> {
    let stable = model.compatibility()?;
    let before = workspace_inventory_identity(&model.families, &model.layers)?;
    let registry_before = model.registry()?.allocations().to_vec();
    let states = model
        .families
        .iter()
        .map(|f| resource_snapshot(&f.prefill).map(|s| s.blobs))
        .collect::<Result<Vec<_>, _>>()?;
    let witnesses = model
        .families
        .iter()
        .map(|f| {
            (
                f.prefill.arena.clone(),
                f.decode.arena.clone(),
                f.prefill_producer.prepared.inventory(),
                f.decode_producer.prepared.inventory(),
                f.decode_producer.raw.clone(),
                pointer(&f.prefill_producer.a_log),
                pointer(&f.prefill_producer.bias),
            )
        })
        .collect::<Vec<_>>();
    let weak = model
        .families
        .iter()
        .map(|f| Arc::downgrade(&f.prefill_producer.raw))
        .collect::<Vec<_>>();
    let epochs = model
        .families
        .iter()
        .map(|f| f.prefill_producer.source_epoch)
        .collect::<Vec<_>>();
    let mut sources = Vec::new();
    for f in &model.families {
        let raw = Arc::new(NativeRawOwner::allocate(
            f.prefill_producer.raw.stream.clone(),
            f.prefill_producer.raw.schema.clone(),
            reserve,
        )?);
        resource_copy_input(&raw.stream, &f.prefill_producer.raw, &raw)?;
        raw.stream.drain()?;
        owner_compare(
            &serde_json::to_vec(&workspace_raw_read(&raw)?).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&workspace_raw_read(&f.prefill_producer.raw)?)
                .map_err(|e| e.to_string())?,
            "lifetime exact real replacement inputs",
        )?;
        sources.push(raw);
    }
    let mut life = KdaPrefillStorageLifetime::from_bound(
        std::mem::take(&mut model.families),
        model.layers.clone(),
    )?;
    let mut rejected = Vec::new();
    for case in [
        "retire-stale-proof",
        "retire-late-epoch",
        "retire-held-family",
        "retire-held-decode-arena",
    ] {
        let last = life.bound()?.len().checked_sub(1).ok_or("lifetime empty")?;
        let state = life.bound()?[last].prefill.state.clone();
        let arena = life.bound()?[last].decode.arena.clone();
        let held = if case == "retire-held-family" {
            Some(
                state
                    .timeline
                    .try_lock()
                    .map_err(|_| "lifetime held family")?,
            )
        } else {
            None
        };
        let lease = if case == "retire-held-decode-arena" {
            Some(arena.lease()?)
        } else {
            None
        };
        let mut wanted = epochs.clone();
        if case == "retire-late-epoch" {
            wanted[last] = wanted[last]
                .checked_add(1)
                .ok_or("lifetime negative epoch")?;
        }
        let proof = if case == "retire-stale-proof" {
            json!([])
        } else {
            before.clone()
        };
        if life.retire_prefill_quiescent(&proof, &wanted).is_ok() {
            return Err(format!("lifetime invalid retirement admitted:{case}"));
        }
        drop(lease);
        drop(held);
        if workspace_inventory_identity(life.bound()?, &life.layers)? != before {
            return Err("lifetime failed retirement published".into());
        }
        for (f, want) in life.bound()?.iter().zip(&states) {
            owner_compare(
                &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                    .map_err(|e| e.to_string())?,
                &serde_json::to_vec(want).map_err(|e| e.to_string())?,
                "lifetime retirement rejection full state",
            )?;
        }
        rejected.push(case);
    }
    let retired_epochs = life.retire_prefill_quiescent(&before, &epochs)?;
    if life.bound().is_ok() || weak.iter().any(|w| w.upgrade().is_some()) {
        return Err("lifetime retired prefill executable or source lease remains".into());
    }
    let retired_before = lifetime_retired_witness(&life)?;
    if life.registry()?.allocations() != registry_before.as_slice() {
        return Err("lifetime retirement changed registry".into());
    }
    let bindings = life.decode_bindings()?;
    model_family_identity(&bindings)?;
    drop(bindings);
    let families = match &life.state {
        KdaPrefillStorageState::Retired(f) => f,
        _ => return Err("lifetime retirement missing".into()),
    };
    for (((f, (pa, da, pp, dp, raw, a, b)), want), epoch) in families
        .iter()
        .zip(&witnesses)
        .zip(&states)
        .zip(&retired_epochs)
    {
        if !Arc::ptr_eq(&f.prefill.arena, pa)
            || !Arc::ptr_eq(&f.decode.arena, da)
            || f.preparation.prepared.inventory() != *pp
            || f.decode_producer.prepared.inventory() != *dp
            || !Arc::ptr_eq(&f.decode_producer.raw, raw)
            || pointer(&f.preparation.a_log) != *a
            || pointer(&f.preparation.bias) != *b
            || f.preparation.source_epoch != *epoch
        {
            return Err("lifetime retirement changed retained resources".into());
        }
        owner_compare(
            &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                .map_err(|e| e.to_string())?,
            &serde_json::to_vec(want).map_err(|e| e.to_string())?,
            "lifetime retirement full nonzero state",
        )?;
    }
    let replacements = || {
        sources
            .iter()
            .map(|r| RetainedPrefillKdaSource { inner: r.clone() })
            .collect::<Vec<_>>()
    };
    for case in [
        "attach-incomplete",
        "attach-late-extent",
        "attach-late-epoch",
        "attach-held-family",
        "attach-held-decode-arena",
    ] {
        let last = sources.len().checked_sub(1).ok_or("lifetime no sources")?;
        let families = match &life.state {
            KdaPrefillStorageState::Retired(f) => f,
            _ => return Err("lifetime detached phase lost".into()),
        };
        let state = families[last].prefill.state.clone();
        let arena = families[last].decode.arena.clone();
        let held = if case == "attach-held-family" {
            Some(
                state
                    .timeline
                    .try_lock()
                    .map_err(|_| "lifetime held state")?,
            )
        } else {
            None
        };
        let lease = if case == "attach-held-decode-arena" {
            Some(arena.lease()?)
        } else {
            None
        };
        let mut repl = replacements();
        let mut wanted = retired_epochs.clone();
        match case {
            "attach-incomplete" => {
                repl.pop();
            }
            "attach-late-extent" => {
                repl[last] = RetainedPrefillKdaSource {
                    inner: witnesses[last].4.clone(),
                }
            }
            "attach-late-epoch" => {
                wanted[last] = wanted[last]
                    .checked_add(1)
                    .ok_or("lifetime negative epoch")?
            }
            _ => {}
        }
        if life.attach_prefill_quiescent(repl, &wanted).is_ok() {
            return Err(format!("lifetime invalid attachment admitted:{case}"));
        }
        drop(lease);
        drop(held);
        if lifetime_retired_witness(&life)? != retired_before {
            return Err(format!("lifetime failed attachment mutated:{case}"));
        }
        rejected.push(case);
    }
    let attached_epochs = life.attach_prefill_quiescent(replacements(), &retired_epochs)?;
    model.families = life.take_bound()?;
    if model.compatibility()? != stable
        || model.registry()?.allocations() != registry_before.as_slice()
    {
        return Err("lifetime reattachment changed stable cache identity".into());
    }
    for (((f, raw), want), i) in model.families.iter().zip(&sources).zip(&states).zip(0..) {
        if !Arc::ptr_eq(&f.prefill_producer.raw, raw)
            || retired_epochs[i] != epochs[i] + 1
            || attached_epochs[i] != epochs[i] + 2
        {
            return Err("lifetime incomplete epoch/source publication".into());
        }
        owner_compare(
            &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                .map_err(|e| e.to_string())?,
            &serde_json::to_vec(want).map_err(|e| e.to_string())?,
            "lifetime reattachment full nonzero state",
        )?;
    }
    owner_save(root,"prefill-source-lifetime.json",&serde_json::to_vec_pretty(&json!({"passed":true,"layers":model.layers,"negative_admissions":rejected,"epochs_before":epochs,"epochs_retired":retired_epochs,"epochs_attached":attached_epochs,"all_former_prefill_sources_released":true,"retired_prefill_execution_rejected":true,"decode_bindings_and_complete_registry_retained":true,"full_nonzero_state_and_preparation_params_unchanged":true,"stable_cache_compatibility_unchanged":true,"replacement_real_input_contents_exact":true,"scope":"complete supplied two-family one actual context lifecycle; subsequent parallel prefix/graph/session gates mandatory; no engine scratch hooks or serving qualification"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
fn lifetime_retired_witness(life: &KdaPrefillStorageLifetime) -> Result<Value, String> {
    let families = match &life.state {
        KdaPrefillStorageState::Retired(f) => f,
        _ => return Err("lifetime expected retired state".into()),
    };
    let mut rows = Vec::new();
    for f in families {
        rows.push(json!({"prefill_state":resource_snapshot(&f.prefill)?.blobs,"prepared":f.preparation.prepared.inventory(),"params":[pointer(&f.preparation.a_log),pointer(&f.preparation.bias)],"source_epoch":f.preparation.source_epoch,"decode":model_family_identity(&[ModelFamilyBinding{layer:0,owner:&f.decode,producer:&f.decode_producer}])?}));
    }
    Ok(json!(rows))
}
