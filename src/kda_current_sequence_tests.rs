fn current_snapshot_tokens(
    blobs: &[crate::session_cache::SequenceStateBlob],
) -> Result<usize, String> {
    let mut tokens = None;
    for b in blobs
        .iter()
        .filter(|b| b.allocation_name.ends_with(".absolute"))
    {
        let value = usize::try_from(u64::from_le_bytes(
            b.bytes
                .as_slice()
                .try_into()
                .map_err(|_| "current absolute width")?,
        ))
        .map_err(|_| "current absolute overflow")?;
        if tokens.replace(value).is_some_and(|old| old != value) {
            return Err("current absolute histories differ".into());
        }
    }
    tokens.ok_or("current absolute history absent".into())
}
fn current_blob_hashes(blobs: &[crate::session_cache::SequenceStateBlob]) -> Value {
    json!(blobs.iter().map(|b|json!({"name":b.allocation_name,"bytes":b.bytes.len(),"sha256":format!("{:x}",Sha256::digest(&b.bytes))})).collect::<Vec<_>>())
}
fn current_snapshot_native(
    engine: &Arc<KdaEngineRequestStorage>,
    store: &mut crate::gpu_decode::GpuDecodeStore,
    tokens: usize,
) -> Result<Vec<crate::session_cache::SequenceStateBlob>, String> {
    let (registry, audit) = engine.audit()?;
    let allocations = registry.allocations().to_vec();
    let (blobs, _) = engine.current_sequence_transfer(
        &audit["current_identity"],
        audit["domain_generation"]
            .as_u64()
            .ok_or("current generation")?,
        audit["live_sequence_epoch"]
            .as_u64()
            .ok_or("current epoch")?,
        &mut [store],
        |life, stores| {
            let mut dma_regions = Vec::new();
            for (prefill, decode) in current_sequence_owners(life)? {
                dma_regions.extend(prefill.external_dma_regions()?);
                dma_regions.extend(decode.external_dma_regions()?);
            }
            let dma_owners: Vec<&dyn crate::sequence_external_owner::CudaRegion> = dma_regions
                .iter().map(|region| region as &dyn crate::sequence_external_owner::CudaRegion).collect();
            let (blobs, _) =
                stores[0].snapshot_external_sequence_state_rust(&allocations, tokens, &dma_owners)?;
            validate_kda_snapshot_semantics(
                &allocations,
                tokens,
                &blobs.iter().collect::<Vec<_>>(),
            )?;
            Ok(blobs)
        },
    )?;
    Ok(blobs)
}
fn current_sequence_native_controls(
    root: &Path,
    label: &str,
    engine: &Arc<KdaEngineRequestStorage>,
    store: &mut crate::gpu_decode::GpuDecodeStore,
) -> Result<(), String> {
    let (registry, initial) = engine.audit()?;
    let allocations = registry.allocations().to_vec();
    let original = {
        let state = engine
            .inner
            .try_lock()
            .map_err(|_| "current capture held")?;
        let owners = current_sequence_owners(&state.lifetime)?;
        for (p, _) in &owners {
            p.stream.drain()?;
        }
        let mut blobs =
            capture_owner(owners.first().ok_or("current empty owners")?.0, &registry)?.blobs;
        blobs.sort_by(|a, b| {
            (a.device_ordinal, a.allocation_name.as_str())
                .cmp(&(b.device_ordinal, b.allocation_name.as_str()))
        });
        blobs
    };
    let tokens = current_snapshot_tokens(&original)?;
    if tokens == 0 {
        return Err("current fixture requires nonzero incoming carry".into());
    }
    validate_kda_snapshot_semantics(&allocations, tokens, &original.iter().collect::<Vec<_>>())?;
    let captured = current_snapshot_native(engine, store, tokens)?;
    if current_blob_hashes(&captured) != current_blob_hashes(&original) {
        return Err("current complete initial pinned state differs".into());
    }
    let baseline = engine.audit()?.1;
    let generation = baseline["domain_generation"]
        .as_u64()
        .ok_or("current generation")?;
    let epoch = baseline["live_sequence_epoch"]
        .as_u64()
        .ok_or("current epoch")?;
    let expected = baseline["current_identity"].clone();
    let mut negatives = Vec::new();
    for case in [
        "stale-generation",
        "stale-live-epoch",
        "wrong-inventory",
        "missing-DMA-context",
    ] {
        let mut called = false;
        let bad_identity = if case == "wrong-inventory" {
            json!({})
        } else {
            expected.clone()
        };
        let result = if case == "missing-DMA-context" {
            engine.current_sequence_transfer(&bad_identity, generation, epoch, &mut [], |_, _| {
                called = true;
                Ok(())
            })
        } else {
            engine.current_sequence_transfer(
                &bad_identity,
                if case == "stale-generation" {
                    generation
                        .checked_add(1)
                        .ok_or("test generation overflow")?
                } else {
                    generation
                },
                if case == "stale-live-epoch" {
                    epoch.checked_add(1).ok_or("test epoch overflow")?
                } else {
                    epoch
                },
                &mut [&mut *store],
                |_, _| {
                    called = true;
                    Ok(())
                },
            )
        };
        if result.is_ok() || called || engine.audit()?.1 != baseline {
            return Err(format!("current pre-DMA negative mutated:{case}"));
        }
        negatives.push(case.to_string());
    }
    for case in [
        "held-family",
        "held-decode-arena",
        "held-prefill-arena",
        "held-resource",
    ] {
        let (family, pref, decode) = {
            let state = engine
                .inner
                .try_lock()
                .map_err(|_| "current held inspection")?;
            let owners = current_sequence_owners(&state.lifetime)?;
            let (p, d) = owners.last().ok_or("current no owners")?;
            (p.state.clone(), p.arena.clone(), d.arena.clone())
        };
        let mut called = false;
        if case == "held-resource" {
            let state = engine
                .inner
                .try_lock()
                .map_err(|_| "current held resource inspection")?;
            let _held = state
                .resource_domain
                .groups
                .last()
                .ok_or("current no resources")?
                .timeline
                .try_lock()
                .map_err(|_| "current resource held")?;
            if KdaCurrentSequenceResources::checked(&state.resource_domain).is_ok() {
                return Err("current held last resource admitted".into());
            }
            if engine
                .current_sequence_transfer(
                    &expected,
                    generation,
                    epoch,
                    &mut [&mut *store],
                    |_, _| {
                        called = true;
                        Ok(())
                    },
                )
                .is_ok()
            {
                return Err("current held storage admitted".into());
            }
        } else {
            let held = if case == "held-family" {
                Some(
                    family
                        .timeline
                        .try_lock()
                        .map_err(|_| "current family already held")?,
                )
            } else {
                None
            };
            let arena = if case == "held-decode-arena" {
                Some(decode.lease()?)
            } else if case == "held-prefill-arena" {
                Some(pref.lease()?)
            } else {
                None
            };
            if engine
                .current_sequence_transfer(
                    &expected,
                    generation,
                    epoch,
                    &mut [&mut *store],
                    |_, _| {
                        called = true;
                        Ok(())
                    },
                )
                .is_ok()
            {
                return Err(format!("current held phase admitted:{case}"));
            }
            drop(arena);
            drop(held);
        }
        if called || engine.audit()?.1 != baseline {
            return Err(format!("current held admission mutated:{case}"));
        }
        negatives.push(case.to_string());
    }
    {
        let mut state = engine
            .inner
            .try_lock()
            .map_err(|_| "current overflow injection held")?;
        state.live_sequence_epoch = u64::MAX;
    }
    let mut called = false;
    let result = engine.current_sequence_transfer(
        &expected,
        generation,
        u64::MAX,
        &mut [&mut *store],
        |_, _| {
            called = true;
            Ok(())
        },
    );
    {
        let mut state = engine
            .inner
            .try_lock()
            .map_err(|_| "current overflow restoration held")?;
        state.live_sequence_epoch = epoch;
    }
    if result.is_ok() || called || engine.audit()?.1 != baseline {
        return Err("current overflow reached callback/mutated".into());
    }
    negatives.push("checked-live-epoch-overflow".into());
    for case in [
        "unused-pending-zero",
        "active-pending-NaN",
        "matrix-infinity",
        "wrong-cursor",
        "wrong-pending",
        "error-status",
    ] {
        let mut bad: Vec<crate::session_cache::SequenceStateBlob> =
            serde_json::from_slice(&serde_json::to_vec(&captured).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        match case {
            "unused-pending-zero" => {
                let index = allocations
                    .iter()
                    .position(|a| a.name.ends_with(".raw0"))
                    .ok_or("raw0 missing")?;
                let a = &allocations[index];
                let offset = (tokens % a.shape[0]) * a.strides_bytes[0];
                if offset >= bad[index].bytes.len() {
                    return Err("unused pending row missing".into());
                }
                bad[index].bytes[offset..offset + 4].copy_from_slice(&0u32.to_le_bytes());
            }
            "active-pending-NaN" => {
                let b = bad
                    .iter_mut()
                    .find(|b| b.allocation_name.ends_with(".raw0"))
                    .ok_or("raw0 missing")?;
                if tokens
                    % allocations
                        .iter()
                        .find(|a| a.name == b.allocation_name)
                        .ok_or("raw descriptor")?
                        .shape[0]
                    == 0
                {
                    return Err("fixture lacks active partial row".into());
                }
                b.bytes[..4].copy_from_slice(&KDA_PENDING_INVALID_F32_BITS.to_le_bytes());
            }
            "matrix-infinity" => bad
                .iter_mut()
                .find(|b| b.allocation_name.ends_with(".latest"))
                .ok_or("latest missing")?
                .bytes[..4]
                .copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes()),
            "wrong-cursor" => bad
                .iter_mut()
                .find(|b| b.allocation_name.ends_with(".absolute"))
                .ok_or("absolute missing")?
                .bytes
                .copy_from_slice(
                    &(u64::try_from(tokens).map_err(|_| "token width")? + 1).to_le_bytes(),
                ),
            "wrong-pending" => bad
                .iter_mut()
                .find(|b| b.allocation_name.ends_with(".pending"))
                .ok_or("pending missing")?
                .bytes
                .copy_from_slice(&(-1i32).to_le_bytes()),
            "error-status" => bad
                .iter_mut()
                .find(|b| b.allocation_name.ends_with(".error"))
                .ok_or("error missing")?
                .bytes
                .copy_from_slice(&1i32.to_le_bytes()),
            _ => return Err("unknown semantic case".into()),
        }
        if validate_kda_snapshot_semantics(&allocations, tokens, &bad.iter().collect::<Vec<_>>())
            .is_ok()
            || engine.audit()?.1 != baseline
        {
            return Err(format!("current invalid snapshot accepted:{case}"));
        }
        negatives.push(case.to_string());
    }
    {
        let state = engine
            .inner
            .try_lock()
            .map_err(|_| "current before-DMA integrity held")?;
        let owners = current_sequence_owners(&state.lifetime)?;
        for (p, _) in &owners {
            p.stream.drain()?;
        }
        let mut blobs =
            capture_owner(owners.first().ok_or("current empty owners")?.0, &registry)?.blobs;
        blobs.sort_by(|a, b| {
            (a.device_ordinal, a.allocation_name.as_str())
                .cmp(&(b.device_ordinal, b.allocation_name.as_str()))
        });
        if current_blob_hashes(&blobs) != current_blob_hashes(&original) {
            return Err("current rejected cases changed full state bytes".into());
        }
    }
    engine.current_sequence_transfer(
        &expected,
        generation,
        epoch,
        &mut [&mut *store],
        |life, _| {
            let owners = current_sequence_owners(life)?;
            for (p, _) in &owners {
                p.pending_raw.validate_phase_invalidation(&StatePhase {
                    owner: p,
                    stream: &p.stream,
                })?;
            }
            for (p, _) in owners {
                p.reset_in_phase(&StatePhase {
                    owner: p,
                    stream: &p.stream,
                })?;
            }
            Ok(())
        },
    )?;
    let reset = current_snapshot_native(engine, store, 0)?;
    for b in &reset {
        if !b
            .allocation_name
            .rsplit('.')
            .next()
            .ok_or("reset role")?
            .starts_with("raw")
            && b.bytes.iter().any(|&v| v != 0)
        {
            return Err("current canonical reset nonzero carry/scalar".into());
        }
    }
    let pre_restore = engine.audit()?.1;
    validate_kda_snapshot_semantics(&allocations, tokens, &captured.iter().collect::<Vec<_>>())?;
    engine.current_sequence_transfer(
        &pre_restore["current_identity"],
        pre_restore["domain_generation"]
            .as_u64()
            .ok_or("restore generation")?,
        pre_restore["live_sequence_epoch"]
            .as_u64()
            .ok_or("restore epoch")?,
        &mut [&mut *store],
        |life, stores| {
            let mut dma_regions = Vec::new();
            for (prefill, decode) in current_sequence_owners(life)? {
                dma_regions.extend(prefill.external_dma_regions()?);
                dma_regions.extend(decode.external_dma_regions()?);
            }
            let dma_owners: Vec<&dyn crate::sequence_external_owner::CudaRegion> = dma_regions
                .iter().map(|region| region as &dyn crate::sequence_external_owner::CudaRegion).collect();
            stores[0].restore_external_sequence_state_rust(
                &allocations,
                tokens,
                &captured.iter().collect::<Vec<_>>(),
                &dma_owners,
            )?;
            Ok(())
        },
    )?;
    let restored = current_snapshot_native(engine, store, tokens)?;
    if current_blob_hashes(&restored) != current_blob_hashes(&original) {
        return Err("current full partial pinned restore differs".into());
    }
    let final_audit = engine.audit()?.1;
    if final_audit["domain_generation"] != initial["domain_generation"]
        || final_audit["static_identity"] != initial["static_identity"]
        || final_audit["current_identity"] != initial["current_identity"]
        || final_audit["live_sequence_epoch"].as_u64() != Some(epoch + 4)
    {
        return Err("current sequence changed source/static inventory or live epoch".into());
    }
    let events = |a: &Value| {
        a["resource_domain"]["groups"]
            .as_array()
            .map(|g| g.iter().map(|g| g["event"].clone()).collect::<Vec<_>>())
    };
    if events(&initial) != events(&final_audit) {
        return Err("current sequence recreated retained events".into());
    }
    owner_save(root,&format!("CURRENT_SEQUENCE_{label}.json"),&serde_json::to_vec_pretty(&json!({"passed":true,"role":label,"tokens":tokens,"all_pre_DMA_negatives":negatives,"initial":initial,"final":final_audit,"snapshot_hashes":current_blob_hashes(&captured),"reset_hashes":current_blob_hashes(&reset),"restored_hashes":current_blob_hashes(&restored),"full_bytes_exact":true,"canonical_reset":true,"retained_events_exact":true,"scope":"native current storage class and actual Rust pinned DMA; original operational union/installed selected cache reset outstanding"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
fn current_sequence_failure_native(
    root: &Path,
    parent: &[PreparedKdaModelFamily],
    layers: &[usize],
    reserve: usize,
    store: &mut crate::gpu_decode::GpuDecodeStore,
) -> Result<(), String> {
    let path = std::env::var("KRASIS_KDA_RESOURCE_PLAN")
        .map_err(|_| "current failure requires frozen plan")?;
    let plan: TilingPlan = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let transport =
        String::from_utf8(owner_checked(&plan.native.transport_ptx)?).map_err(|e| e.to_string())?;
    let math =
        String::from_utf8(owner_checked(&plan.native.math_ptx)?).map_err(|e| e.to_string())?;
    let mut fresh = Vec::new();
    for f in parent {
        let pre = Arc::new(NativeRawOwner::allocate(
            f.prefill_producer.raw.stream.clone(),
            f.prefill_producer.raw.schema.clone(),
            reserve,
        )?);
        let dec = Arc::new(NativeRawOwner::allocate(
            f.decode_producer.raw.stream.clone(),
            f.decode_producer.raw.schema.clone(),
            reserve,
        )?);
        resource_copy_input(&pre.stream, &f.prefill_producer.raw, &pre)?;
        resource_copy_input(&dec.stream, &f.decode_producer.raw, &dec)?;
        let p = &f.prefill_producer;
        let bank = KdaModelKernelBank {
            kernels: p.kernels.clone(),
            transport: transport.clone(),
            math: math.clone(),
        };
        fresh.push(unsafe {
            PreparedKdaModelFamily::from_model_storage(
                RetainedPrefillKdaSource { inner: pre },
                RetainedDecodeKdaSource { inner: dec },
                p.heads,
                p.dim,
                1,
                reserve,
                p.lower,
                p.scale,
                p.epsilon,
                &bank,
                pointer(&p.a_log),
                p.heads,
                pointer(&p.bias),
                p.heads * p.dim,
            )
        }?);
    }
    let weak = fresh
        .iter()
        .map(|f| Arc::downgrade(&f.prefill.state))
        .collect::<Vec<_>>();
    let mut life = KdaPrefillStorageLifetime::from_bound(fresh, layers.to_vec())?;
    let expected = workspace_inventory_identity(life.bound()?, layers)?;
    let epochs = life.current_prefill_epochs()?;
    life.retire_and_verify_input_release(&expected, &epochs)?;
    let storage = KdaEngineRequestStorage::new(life, layers.to_vec(), reserve, true, None)?;
    let (registry, before) = storage.audit()?;
    let allocations = registry.allocations().to_vec();
    let zeros = current_snapshot_native(&storage, store, 0)?;
    let (_, admission) = storage.audit()?;
    let mut called = false;
    let result: Result<((), u64), String> = storage.current_sequence_transfer(
        &admission["current_identity"],
        admission["domain_generation"]
            .as_u64()
            .ok_or("fault generation")?,
        admission["live_sequence_epoch"]
            .as_u64()
            .ok_or("fault epoch")?,
        &mut [store],
        |life, stores| {
            called = true;
            let mut dma_regions = Vec::new();
            for (prefill, decode) in current_sequence_owners(life)? {
                dma_regions.extend(prefill.external_dma_regions()?);
                dma_regions.extend(decode.external_dma_regions()?);
            }
            let dma_owners: Vec<&dyn crate::sequence_external_owner::CudaRegion> = dma_regions
                .iter().map(|region| region as &dyn crate::sequence_external_owner::CudaRegion).collect();
            stores[0].restore_external_sequence_state_rust(
                &allocations,
                0,
                &zeros.iter().collect::<Vec<_>>(),
                &dma_owners,
            )?;
            let owners = current_sequence_owners(life)?;
            for (p, _) in &owners {
                p.pending_raw.validate_phase_invalidation(&StatePhase {
                    owner: p,
                    stream: &p.stream,
                })?;
            }
            for (p, _) in owners {
                p.reset_in_phase(&StatePhase {
                    owner: p,
                    stream: &p.stream,
                })?;
            }
            Err("admitted current sequence forced callback failure after DMA/reset enqueue".into())
        },
    );
    if !called
        || !result.as_ref().err().is_some_and(|e| {
            e == "admitted current sequence forced callback failure after DMA/reset enqueue"
        })
        || storage.audit().is_ok()
    {
        return Err("current callback failure was hidden/not poisoned".into());
    }
    let (groups, owners_retained) = {
        let state = storage
            .inner
            .try_lock()
            .map_err(|_| "current failure storage mutex")?;
        if !state.poisoned
            || state.live_sequence_epoch
                != admission["live_sequence_epoch"]
                    .as_u64()
                    .ok_or("fault epoch")?
            || state.domain_generation
                != before["domain_generation"]
                    .as_u64()
                    .ok_or("fault generation")?
        {
            return Err("current failed sequence published generation/epoch".into());
        }
        for g in &state.resource_domain.groups {
            let t = g
                .timeline
                .try_lock()
                .map_err(|_| "current fault resource guard leaked")?;
            if !t.poisoned {
                return Err("current fault missing resource poison".into());
            }
            for stream in &t.streams {
                stream.drain()?;
            }
        }
        for w in &weak {
            let owner = w.upgrade().ok_or("current fault released owner early")?;
            let t = owner
                .timeline
                .try_lock()
                .map_err(|_| "current fault family guard leaked")?;
            if !t
                .as_ref()
                .ok_or("current fault family timeline absent")?
                .poisoned
            {
                return Err("current fault family not poisoned".into());
            }
        }
        (state.resource_domain.groups.len(), weak.len())
    };
    drop(storage);
    if weak.iter().any(|w| w.upgrade().is_some()) {
        return Err("current failed state owner leaked after teardown".into());
    }
    owner_save(root,"CURRENT_SEQUENCE_FAILURE.json",&serde_json::to_vec_pretty(&json!({"passed":true,"callback_entered":called,"all_resource_groups_poisoned":groups,"all_family_owners_retained_until_teardown":owners_retained,"failed_generation_and_epoch_unpublished":true,"owners_released_after_teardown":true,"scope":"injected callback error after actual pinned restore/reset enqueue; actual CUDA completion/destruction errors remain unqualified"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
