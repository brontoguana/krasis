// Current storage type shared through two aliases; concrete Engine/Store
// installation still requires the separate loaded-model gate.
fn current_storage_prepare_native(storage: &Arc<KdaEngineRequestStorage>) -> Result<(), String> {
    let state = storage
        .inner
        .try_lock()
        .map_err(|_| "current native storage held")?;
    if state.poisoned || request_storage_static_identity(&state.lifetime)? != state.static_identity
    {
        return Err("current native storage stale/poisoned".into());
    }
    let families = state.lifetime.bound()?;
    let bindings = families
        .iter()
        .zip(&storage.layers)
        .map(|(f, &layer)| ModelFamilyBinding {
            layer,
            owner: &f.prefill,
            producer: &f.prefill_producer,
        })
        .collect::<Vec<_>>();
    let expected = model_family_identity(&bindings)?;
    let stream = bindings
        .first()
        .ok_or("current native empty")?
        .owner
        .stream
        .clone();
    let mut proof = ProducerResourceProof::checked(&state.resource_domain, &bindings, &expected)?;
    proof.groups = (0..state.resource_domain.groups.len()).collect();
    let mut admission = ProducerResourceAdmission::checked(&proof, &stream)?;
    let mut inactive = Vec::new();
    for f in families {
        if !admission
            .arena
            .arenas
            .iter()
            .chain(inactive.iter())
            .any(|l: &EnqueueLease<'_>| std::ptr::eq(l.arena, f.decode.arena.as_ref()))
        {
            inactive.push(f.decode.arena.lease()?);
        }
        let lease = admission
            .arena
            .arenas
            .iter()
            .chain(inactive.iter())
            .find(|l| std::ptr::eq(l.arena, f.decode.arena.as_ref()))
            .ok_or("current native missing decode arena")?;
        f.decode.validate_lease(lease)?;
    }
    admission.execute(|leases| {
        for binding in &bindings {
            binding
                .owner
                .validate_lease(model_arena_lease(binding.owner, leases)?)?;
            binding.producer.enqueue(0, 1, binding.owner)?;
        }
        Ok(())
    })
}
fn current_engine_domain_controls(
    root: &Path,
    model: &mut RetainedKdaSessionModel,
    reserve: usize,
    dma_store: &mut crate::gpu_decode::GpuDecodeStore,
) -> Result<(), String> {
    let stable = model.compatibility()?;
    let registry = model.registry()?.allocations().to_vec();
    let states = model
        .families
        .iter()
        .map(|f| resource_snapshot(&f.prefill).map(|s| s.blobs))
        .collect::<Result<Vec<_>, _>>()?;
    let mut sources = Vec::new();
    for f in &model.families {
        let raw = Arc::new(NativeRawOwner::allocate(
            f.prefill_producer.raw.stream.clone(),
            f.prefill_producer.raw.schema.clone(),
            reserve,
        )?);
        resource_copy_input(&raw.stream, &f.prefill_producer.raw, &raw)?;
        raw.stream.drain()?;
        sources.push(raw);
    }
    let rows = *sources
        .first()
        .ok_or("current native sources empty")?
        .schema
        .planes
        .first()
        .ok_or("current native plane missing")?
        .shape
        .first()
        .ok_or("current native row extent")?;
    if sources.iter().any(|s| {
        s.schema
            .planes
            .first()
            .and_then(|p| p.shape.first())
            .copied()
            != Some(rows)
    }) {
        return Err("current fixture row extents differ".into());
    }
    let mut life = KdaPrefillStorageLifetime::from_bound(
        std::mem::take(&mut model.families),
        model.layers.clone(),
    )?;
    current_sequence_failure_native(root, life.bound()?, &life.layers, reserve, dma_store)?;
    let expected = workspace_inventory_identity(life.bound()?, &life.layers)?;
    let epochs = life.current_prefill_epochs()?;
    life.retire_and_verify_input_release(&expected, &epochs)?;
    let engine = KdaEngineRequestStorage::new(life, model.layers.clone(), reserve, true, None)?;
    let store = engine.clone();
    if !Arc::ptr_eq(&engine, &store) {
        return Err("current native Engine/Store storage aliases differ".into());
    }
    let cold = engine.audit()?.1;
    let replacements = || {
        sources
            .iter()
            .map(|raw| RetainedPrefillKdaSource { inner: raw.clone() })
            .collect::<Vec<_>>()
    };
    engine.attach_after_prepare(replacements(), rows)?;
    current_sequence_native_controls(root, "bound", &engine, dma_store)?;
    let attached = store.audit()?.1;
    let mut negatives = Vec::new();
    // Raw-wrapper owners held by an external graph/source must reject before
    // any source/domain publication, then permit valid retirement after drop.
    let external = {
        let state = engine
            .inner
            .try_lock()
            .map_err(|_| "current fixture storage held")?;
        state
            .lifetime
            .bound()?
            .last()
            .ok_or("current fixture missing family")?
            .prefill_producer
            .raw
            .clone()
    };
    let source_ptr = Arc::as_ptr(&external) as usize;
    // The fixture's replacement sources are deliberate external leases as well;
    // drop them before the positive release gate, retaining only Weak witnesses.
    let weak = sources.iter().map(Arc::downgrade).collect::<Vec<_>>();
    drop(sources);
    if engine
        .retire_before_release("native-external-source")
        .is_ok()
        || store.audit()?.1 != attached
    {
        return Err("current external source lease retirement published".into());
    }
    drop(external);
    negatives.push("retire-external-raw");
    for case in [
        "retire-held-family",
        "retire-held-decode-arena",
        "retire-held-resource",
    ] {
        let (state_arc, arena, resource_root) = {
            let state = engine
                .inner
                .try_lock()
                .map_err(|_| "current fixture storage held")?;
            let family = state
                .lifetime
                .bound()?
                .last()
                .ok_or("current fixture family empty")?;
            (
                family.prefill.state.clone(),
                family.decode.arena.clone(),
                state
                    .resource_domain
                    .groups
                    .len()
                    .checked_sub(1)
                    .ok_or("current resource empty")?,
            )
        };
        if case == "retire-held-resource" {
            let state = engine
                .inner
                .try_lock()
                .map_err(|_| "current held resource storage")?;
            let _held = state.resource_domain.groups[resource_root]
                .timeline
                .try_lock()
                .map_err(|_| "current resource already held")?;
            // Exercise the actual inner prepared-transition admission while
            // the last resource group is held. It must fail before any callback.
            let epochs = state.lifetime.current_prefill_epochs()?;
            let members = state
                .lifetime
                .retirement_resource_members(&state.resource_domain, &epochs)?;
            let streams = current_lifetime_streams(&state.lifetime)?;
            if KdaCurrentResourceTransition::checked(&state.resource_domain, members, &streams)
                .is_ok()
            {
                return Err("current held resource reached prepared transition admission".into());
            }
            if store.retire_before_release("native-held-resource").is_ok() {
                return Err("current storage held admission succeeded".into());
            }
        } else {
            let held = if case == "retire-held-family" {
                Some(
                    state_arc
                        .timeline
                        .try_lock()
                        .map_err(|_| "current held family")?,
                )
            } else {
                None
            };
            let lease = if case == "retire-held-decode-arena" {
                Some(arena.lease()?)
            } else {
                None
            };
            if engine.retire_before_release(case).is_ok() {
                return Err(format!("current invalid retirement admitted:{case}"));
            }
            drop(lease);
            drop(held);
        }
        if store.audit()?.1 != attached {
            return Err(format!("current failed retirement changed domain:{case}"));
        }
        negatives.push(case);
    }
    let event_before = attached["resource_domain"]["groups"]
        .as_array()
        .ok_or("current event audit missing")?
        .iter()
        .map(|g| g["event"].clone())
        .collect::<Vec<_>>();
    current_storage_prepare_native(&engine)?;
    current_storage_prepare_native(&store)?;
    let prepared = engine.audit()?.1;
    let event_after = prepared["resource_domain"]["groups"]
        .as_array()
        .ok_or("current event audit missing")?
        .iter()
        .map(|g| g["event"].clone())
        .collect::<Vec<_>>();
    if event_before != event_after || prepared["domain_generation"] != attached["domain_generation"]
    {
        return Err("current producer phase rebuilt persistent resource events".into());
    }
    {
        let state = engine
            .inner
            .try_lock()
            .map_err(|_| "current state integrity held")?;
        for (f, want) in state.lifetime.bound()?.iter().zip(&states) {
            owner_compare(
                &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                    .map_err(|e| e.to_string())?,
                &serde_json::to_vec(want).map_err(|e| e.to_string())?,
                "current producer phases changed persistent nonzero state",
            )?;
        }
    }
    // Allocate the next real source copy while current inputs are admitted and
    // retained; no generated text/tokens or synthetic input substitute.
    let mut next = Vec::new();
    {
        let state = engine
            .inner
            .try_lock()
            .map_err(|_| "current next-source storage held")?;
        for f in state.lifetime.bound()? {
            let raw = Arc::new(NativeRawOwner::allocate(
                f.prefill_producer.raw.stream.clone(),
                f.prefill_producer.raw.schema.clone(),
                reserve,
            )?);
            resource_copy_input(&raw.stream, &f.prefill_producer.raw, &raw)?;
            raw.stream.drain()?;
            next.push(raw);
        }
    }
    engine.retire_before_release("native-complete-retirement")?;
    current_sequence_native_controls(root, "retired", &store, dma_store)?;
    let retired = store.audit()?.1;
    if weak.iter().any(|w| w.upgrade().is_some()) {
        return Err("current complete retirement retained old source wrappers".into());
    }
    if retired["resource_domain"]["roles"]
        .as_array()
        .ok_or("current retired roles")?
        .iter()
        .filter(|r| r["phase"] == "retired_preparation")
        .count()
        != model.layers.len()
    {
        return Err("current retired raw capability not absent".into());
    }
    for case in [
        "attach-empty",
        "attach-wrong-rows",
        "attach-held-family",
        "attach-held-decode-arena",
        "attach-held-resource",
    ] {
        let sources = || {
            next.iter()
                .map(|raw| RetainedPrefillKdaSource { inner: raw.clone() })
                .collect::<Vec<_>>()
        };
        let (family_root, arena, group) = {
            let state = engine
                .inner
                .try_lock()
                .map_err(|_| "current attach inspection held")?;
            let families = match &state.lifetime.state {
                KdaPrefillStorageState::Retired(f) => f,
                _ => return Err("current attach inspection not retired".into()),
            };
            let f = families.last().ok_or("current attach inspection empty")?;
            (
                f.prefill.state.clone(),
                f.decode.arena.clone(),
                state
                    .resource_domain
                    .groups
                    .len()
                    .checked_sub(1)
                    .ok_or("current attach resources empty")?,
            )
        };
        if case == "attach-held-resource" {
            let state = engine
                .inner
                .try_lock()
                .map_err(|_| "current attach storage held")?;
            let _held = state.resource_domain.groups[group]
                .timeline
                .try_lock()
                .map_err(|_| "current attach resource already held")?;
            let epochs = state.lifetime.current_prefill_epochs()?;
            let plan = state.lifetime.prepare_attachment_plan(sources(), &epochs)?;
            let members = state
                .lifetime
                .attachment_resource_members(&state.resource_domain, &plan)?;
            let streams = current_lifetime_streams(&state.lifetime)?;
            if KdaCurrentResourceTransition::checked(&state.resource_domain, members, &streams)
                .is_ok()
            {
                return Err("current attach held resource admitted".into());
            }
            if store.attach_after_prepare(sources(), rows).is_ok() {
                return Err("current attach held storage admitted".into());
            }
        } else {
            let held = if case == "attach-held-family" {
                Some(
                    family_root
                        .timeline
                        .try_lock()
                        .map_err(|_| "current attach family already held")?,
                )
            } else {
                None
            };
            let lease = if case == "attach-held-decode-arena" {
                Some(arena.lease()?)
            } else {
                None
            };
            let candidate = if case == "attach-empty" {
                Vec::new()
            } else {
                sources()
            };
            let candidate_rows = if case == "attach-wrong-rows" {
                rows.checked_add(1)
                    .ok_or("current attach wrong rows overflow")?
            } else {
                rows
            };
            if store
                .attach_after_prepare(candidate, candidate_rows)
                .is_ok()
            {
                return Err(format!("current invalid attachment admitted:{case}"));
            }
            drop(lease);
            drop(held);
        }
        if engine.audit()?.1 != retired {
            return Err(format!("current failed attachment changed domain:{case}"));
        }
        negatives.push(case);
    }
    let attached_next = next
        .iter()
        .map(|raw| RetainedPrefillKdaSource { inner: raw.clone() })
        .collect();
    store.attach_after_prepare(attached_next, rows)?;
    let reattached = engine.audit()?.1;
    {
        let mut state = engine
            .inner
            .try_lock()
            .map_err(|_| "current final extract held")?;
        for (f, want) in state.lifetime.bound()?.iter().zip(&states) {
            owner_compare(
                &serde_json::to_vec(&resource_snapshot(&f.prefill)?.blobs)
                    .map_err(|e| e.to_string())?,
                &serde_json::to_vec(want).map_err(|e| e.to_string())?,
                "current complete source lifecycle changed nonzero state",
            )?;
        }
        model.families = state.lifetime.take_bound()?;
    }
    drop(store);
    drop(engine);
    drop(next);
    if model.compatibility() != Ok(stable) || model.registry()?.allocations() != registry.as_slice()
    {
        return Err(
            "current complete lifecycle changed stable session compatibility/registry".into(),
        );
    }
    owner_save(root,"CURRENT_ENGINE_DOMAIN.json",&serde_json::to_vec_pretty(&json!({"passed":true,"same_current_storage_arc":true,"nonzero_states_unchanged":true,"persistent_event_reuse_across_two_gpu_producer_phases":true,"external_source_pointer":source_ptr,"all_old_prefill_wrappers_released":true,"negative_cases":negatives,"cold":cold,"attached":attached,"after_producer_phases":prepared,"retired":retired,"reattached":reattached,"scope":"same current storage class through two aliases/native producers; concrete PrefillEngine/GpuDecodeStore loaded-model gate still required; no selected recurrence/reset/cache qualification"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
