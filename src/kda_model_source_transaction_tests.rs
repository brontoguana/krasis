fn model_source_transaction_controls(
    root: &Path,
    model: &mut RetainedKdaSessionModel,
    reserve: usize,
) -> Result<(), String> {
    let before = workspace_inventory_identity(&model.families, &model.layers)?;
    let states = model
        .families
        .iter()
        .map(|f| resource_snapshot(&f.prefill).map(|s| s.blobs))
        .collect::<Result<Vec<_>, _>>()?;
    let stable = model.compatibility()?;
    let stable_bindings = model
        .families
        .iter()
        .map(workspace_stable)
        .collect::<Result<Vec<_>, _>>()?;
    let workspaces = model
        .families
        .iter()
        .map(|f| {
            (
                f.prefill.arena.clone(),
                f.decode.arena.clone(),
                f.prefill_producer.prepared.inventory(),
                f.decode_producer.prepared.inventory(),
            )
        })
        .collect::<Vec<_>>();
    let epochs = model
        .families
        .iter()
        .map(|f| {
            [
                f.prefill_producer.source_epoch,
                f.decode_producer.source_epoch,
            ]
        })
        .collect::<Vec<_>>();
    let mut sources = Vec::new();
    for f in &model.families {
        let mut fresh = Vec::new();
        for p in [&f.prefill_producer, &f.decode_producer] {
            let raw = Arc::new(NativeRawOwner::allocate(
                p.raw.stream.clone(),
                p.raw.schema.clone(),
                reserve,
            )?);
            resource_copy_input(&raw.stream, &p.raw, &raw)?;
            raw.stream.drain()?;
            owner_compare(
                &serde_json::to_vec(&workspace_raw_read(&raw)?).map_err(|e| e.to_string())?,
                &serde_json::to_vec(&workspace_raw_read(&p.raw)?).map_err(|e| e.to_string())?,
                "source transaction exact real raw copy",
            )?;
            fresh.push(raw);
        }
        sources.push((fresh.remove(0), fresh.remove(0)));
    }
    let replacements = || {
        sources
            .iter()
            .map(|(p, d)| {
                (
                    RetainedPrefillKdaSource { inner: p.clone() },
                    RetainedDecodeKdaSource { inner: d.clone() },
                )
            })
            .collect::<Vec<_>>()
    };
    let mut rejected = Vec::new();
    for case in [
        "incomplete",
        "late-invalid-source",
        "late-epoch",
        "stale-inventory",
        "held-late-family",
        "held-decode-arena",
    ] {
        let mut repl = replacements();
        let mut want_epochs = epochs.clone();
        let mut want = before.clone();
        let last = repl
            .len()
            .checked_sub(1)
            .ok_or("source test inventory empty")?;
        let state = model.families[last].prefill.state.clone();
        let arena = model.families[last].decode.arena.clone();
        let held_state = if case == "held-late-family" {
            Some(
                state
                    .timeline
                    .try_lock()
                    .map_err(|_| "source negative held state")?,
            )
        } else {
            None
        };
        let held_arena = if case == "held-decode-arena" {
            Some(arena.lease()?)
        } else {
            None
        };
        match case {
            "incomplete" => {
                repl.pop();
            }
            "late-invalid-source" => {
                repl[last].0 = RetainedPrefillKdaSource {
                    inner: sources[last].1.clone(),
                };
            }
            "late-epoch" => {
                want_epochs[last][1] = want_epochs[last][1]
                    .checked_add(1)
                    .ok_or("source negative overflow")?
            }
            "stale-inventory" => want = json!([]),
            _ => {}
        }
        if replace_model_raw_sources_quiescent(
            &mut model.families,
            &model.layers,
            repl,
            &want_epochs,
            &want,
        )
        .is_ok()
        {
            return Err(format!("source transaction admitted negative:{case}"));
        }
        drop(held_arena);
        drop(held_state);
        if workspace_inventory_identity(&model.families, &model.layers)? != before {
            return Err(format!(
                "source transaction negative partially published:{case}"
            ));
        }
        for (f, want) in model.families.iter().zip(&states) {
            owner_compare(
                &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                    .map_err(|e| e.to_string())?,
                &serde_json::to_vec(want).map_err(|e| e.to_string())?,
                "source rejection full state unchanged",
            )?;
        }
        rejected.push(case);
    }
    let published = replace_model_raw_sources_quiescent(
        &mut model.families,
        &model.layers,
        replacements(),
        &epochs,
        &before,
    )?;
    let after = workspace_inventory_identity(&model.families, &model.layers)?;
    for (((f, (p, d)), previous), actual) in model
        .families
        .iter()
        .zip(&sources)
        .zip(&epochs)
        .zip(&published)
    {
        if !Arc::ptr_eq(&f.prefill_producer.raw, p)
            || !Arc::ptr_eq(&f.decode_producer.raw, d)
            || *actual
                != [
                    previous[0].checked_add(1).ok_or("source epoch")?,
                    previous[1].checked_add(1).ok_or("source epoch")?,
                ]
        {
            return Err("source transaction incomplete publication".into());
        }
    }
    for ((f, binding), (p, d, pp, dp)) in
        model.families.iter().zip(&stable_bindings).zip(&workspaces)
    {
        if workspace_stable(f)? != *binding
            || !Arc::ptr_eq(&f.prefill.arena, p)
            || !Arc::ptr_eq(&f.decode.arena, d)
            || f.prefill_producer.prepared.inventory() != *pp
            || f.decode_producer.prepared.inventory() != *dp
        {
            return Err(
                "source transaction changed parameters/persistent/prepared/arena binding".into(),
            );
        }
    }
    if model.compatibility()? != stable {
        return Err("source transaction changed stable cache identity".into());
    }
    for (f, want) in model.families.iter().zip(&states) {
        owner_compare(
            &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                .map_err(|e| e.to_string())?,
            &serde_json::to_vec(want).map_err(|e| e.to_string())?,
            "source positive full nonzero state unchanged",
        )?;
    }
    if replace_model_raw_sources_quiescent(
        &mut model.families,
        &model.layers,
        replacements(),
        &published,
        &before,
    )
    .is_ok()
    {
        return Err("source transaction stale prior proof admitted".into());
    }
    if workspace_inventory_identity(&model.families, &model.layers)? != after {
        return Err("source stale proof changed current binding".into());
    }
    rejected.push("retired-prior-inventory");
    owner_save(root,"model-source-transaction.json",&serde_json::to_vec_pretty(&json!({"passed":true,"layers":model.layers,"negative_admissions":rejected,"epochs_before":epochs,"epochs_after":published,"inventory_before":before,"inventory_after":after,"all_nonzero_persistent_state_unchanged":true,"stable_cache_compatibility_unchanged":true,"source_contents_exact":true,"scope":"complete supplied two-family single actual context; current-source primitive, not serving or simultaneous multi-context qualification"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
