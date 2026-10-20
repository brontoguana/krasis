fn typed_model_family_chronology(
    root: &Path,
    prefill: &Arc<NativeRawOwner>,
    decode: &Arc<NativeRawOwner>,
    parameters: &KdaRawProducer,
    kernels: &Arc<KdaRawKernels>,
    transport: &str,
    math: &str,
    qp: &TilingPlan,
    reserve: usize,
) -> Result<(), String> {
    let p = &qp.native;
    let rows = p.rows;
    let prefix = rows
        .checked_sub(13)
        .ok_or("typed chronology suffix extent")?;
    if prefix == 0 {
        return Err("typed chronology requires nonempty shared prefix".into());
    }
    let hd = p.heads.checked_mul(p.dim).ok_or("typed chronology width")?;
    let want_f = owner_checked(&qp.output_reference)?;
    let want_b = owner_checked(&qp.bf16_reference)?;
    if want_f.len()
        != rows
            .checked_mul(hd)
            .and_then(|x| x.checked_mul(4))
            .ok_or("typed output span")?
        || want_b.len()
            != rows
                .checked_mul(hd)
                .and_then(|x| x.checked_mul(2))
                .ok_or("typed BF16 span")?
    {
        return Err("typed chronology complete reference extent".into());
    }
    let prefix_want: OwnerSnapshot =
        serde_json::from_slice(&owner_checked(&qp.prefix_reference)?).map_err(|e| e.to_string())?;
    let final_want: OwnerSnapshot =
        serde_json::from_slice(&owner_checked(&qp.final_reference)?).map_err(|e| e.to_string())?;
    let bank = KdaModelKernelBank {
        kernels: kernels.clone(),
        transport: transport.into(),
        math: math.into(),
    };
    let window = crate::vram_monitor::begin_precision_vram_window(prefill.device.ordinal() as i32)?
        .ok_or("typed chronology precision window required")?;
    let before = memory()?;
    let family = unsafe {
        PreparedKdaModelFamily::from_model_storage(
            RetainedPrefillKdaSource {
                inner: prefill.clone(),
            },
            RetainedDecodeKdaSource {
                inner: decode.clone(),
            },
            p.heads,
            p.dim,
            rows,
            reserve,
            p.lower,
            p.scale,
            p.epsilon,
            &bank,
            pointer(&parameters.a_log),
            parameters.a_log.len(),
            pointer(&parameters.bias),
            parameters.bias.len(),
        )
    }?;
    let after = memory()?;
    let physical_delta = before[0]
        .checked_sub(after[0])
        .ok_or("typed factory measured allocation delta invalid")?;
    let borrowed = family.borrowed()?;
    let ps = family.prefill.stream.clone();
    let ds = family.decode.stream.clone();
    let output = native_zeroed::<f32>(
        &ps,
        rows.checked_mul(hd)
            .ok_or("typed complete output elements")?,
    )?;
    let bf16 = native_zeroed::<u16>(
        &ps,
        rows.checked_mul(hd).ok_or("typed complete BF16 elements")?,
    )?;
    let mut registry = crate::session_cache::SequenceStateRegistry::default();
    family.register(&mut registry, 0)?;
    let prefix_gpu = TimelineSnapshot::new(&ps, &prefix_want)?;
    let final_gpu = TimelineSnapshot::new(&ps, &final_want)?;
    ps.drain()?;
    let emissions = [ModelFamilyEmission {
        float: native_zeroed::<f32>(&ds, hd)?,
        bf16: native_zeroed::<u16>(&ds, hd)?,
    }];
    ds.drain()?;
    let bindings = [borrowed.decode_binding(0)];
    let graph = capture_model_families(&bindings, &emissions, &ds)?;
    let inventory = model_graph_inventory(&graph)?;
    let proof = ModelBindingProof::checked(&bindings, &graph.identity)?;
    let mut cases = Vec::new();
    for (label, end, quantum) in [
        ("whole GPU batch", rows, rows),
        ("prefix batch and graph continuation", prefix, prefix),
        (
            "partitioned prefix and graph continuation",
            prefix,
            (prefix / 2).max(1),
        ),
    ] {
        family.prefill.ordered_phase(&ps, |phase| phase.reset())?;
        let mut at = 0;
        while at < end {
            let count = quantum.min(end - at);
            family.prefill.ordered_phase(&ps, |phase| {
                phase.prefill(&family.prefill_producer, at, count)?;
                timeline_copy(
                    &ps,
                    pointer(&output) + (at * hd * 4) as u64,
                    family.prefill.p("emitted"),
                    count * hd * 4,
                )?;
                timeline_copy(
                    &ps,
                    pointer(&bf16) + (at * hd * 2) as u64,
                    pointer(&family.prefill.arena.bf16_emitted),
                    count * hd * 2,
                )?;
                if at + count == prefix {
                    prefix_gpu.enqueue(phase, &registry)?;
                }
                if at + count == rows {
                    final_gpu.enqueue(phase, &registry)?;
                }
                Ok(())
            })?;
            at += count;
        }
        for row in end..rows {
            let mut transaction = ModelArenaAdmission::from_proof(&proof, &ds)?;
            transaction.execute(|leases| {
                let phase = StatePhase {
                    owner: &family.decode,
                    stream: &ds,
                };
                for ((desc, src), (dst_desc, dst)) in family
                    .prefill_producer
                    .raw
                    .inventory()
                    .into_iter()
                    .zip(family.decode_producer.raw.inventory())
                {
                    let stride = desc.strides_bytes[0];
                    if stride != dst_desc.storage_bytes {
                        return Err("typed chronology row source extent".into());
                    }
                    timeline_copy(&ds, dst, src + (row * stride) as u64, stride)?;
                }
                family.decode.ordered_graph_in_lease(
                    &phase,
                    &graph.inner,
                    model_arena_lease(&family.decode, leases)?,
                )?;
                timeline_copy(
                    &ds,
                    pointer(&output) + (row * hd * 4) as u64,
                    pointer(&emissions[0].float),
                    hd * 4,
                )?;
                timeline_copy(
                    &ds,
                    pointer(&bf16) + (row * hd * 2) as u64,
                    pointer(&emissions[0].bf16),
                    hd * 2,
                )?;
                if row + 1 == rows {
                    final_gpu.enqueue(&phase, &registry)?;
                }
                Ok(())
            })?;
        }
        ps.drain()?;
        ds.drain()?; // Explicit diagnostic comparison only, after queued chronology.
        let read = |ptr: u64, bytes: usize| -> Result<Vec<u8>, String> {
            let mut out = vec![0u8; bytes];
            check(
                unsafe { sys::lib().cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) },
                "typed chronology complete output read",
            )?;
            Ok(out)
        };
        let f = read(pointer(&output), want_f.len())?;
        let b = read(pointer(&bf16), want_b.len())?;
        owner_compare(&f, &want_f, "typed constructor every F32 row")?;
        owner_compare(&b, &want_b, "typed constructor every BF16 row")?;
        let final_state = final_gpu.read(&family.decode.arithmetic_sha256)?;
        if final_state.layout != final_want.layout
            || serde_json::to_value(&final_state.blobs).map_err(|e| e.to_string())?
                != serde_json::to_value(&final_want.blobs).map_err(|e| e.to_string())?
        {
            return Err("typed constructor complete final state mismatch".into());
        }
        let dir = root.join(format!("typed-chronology-case-{}", cases.len()));
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        owner_save(&dir, "output.f32", &f)?;
        owner_save(&dir, "output.bf16", &b)?;
        owner_save(
            &dir,
            "final-state.json",
            &serde_json::to_vec_pretty(&final_state).map_err(|e| e.to_string())?,
        )?;
        if end == prefix {
            let state = prefix_gpu.read(&family.prefill.arithmetic_sha256)?;
            if state.layout != prefix_want.layout
                || serde_json::to_value(&state.blobs).map_err(|e| e.to_string())?
                    != serde_json::to_value(&prefix_want.blobs).map_err(|e| e.to_string())?
            {
                return Err("typed constructor complete shared prefix state mismatch".into());
            }
            owner_save(
                &dir,
                "prefix-state.json",
                &serde_json::to_vec_pretty(&state).map_err(|e| e.to_string())?,
            )?;
        }
        cases.push(json!({"label":label,"prefix_end":end,"quantum":quantum,"graph_suffix":rows-end,
            "all_F32_rows_exact":true,"all_BF16_rows_exact":true,"complete_final_state_exact":true,
            "complete_shared_prefix_state_exact":end==prefix,"initial_carry":"admitted zero control, no nonzero claim"}));
    }
    drop(graph);
    drop(proof);
    drop(bindings);
    drop(borrowed);
    drop(family);
    let min_free = window.finish()?;
    if min_free < reserve as u64 {
        return Err("typed constructor chronology precision floor below reserve".into());
    }
    owner_save(root,"typed-constructor-chronology.json",&serde_json::to_vec_pretty(&json!({"passed":true,"cases":cases,"graph_inventory":inventory,
        "source_rows":rows,"factory_prefill_capacity":rows,"parameter_owner_arithmetic":parameters.arithmetic_sha256,
        "memory_before":before,"memory_after_factory":after,"factory_physical_delta":physical_delta,
        "precision_min_free_bytes":min_free,"reserve_bytes":reserve,"all_execution_uses_factory_owners":true,
        "scope":"unpressured owning-constructor whole chronology; not full-model HCS calibration/performance/ordinary acceptance"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
