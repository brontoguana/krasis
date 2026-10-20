fn typed_model_family_setup_controls(
    root: &Path,
    device: &Arc<CudaDevice>,
    prefill: &Arc<NativeRawOwner>,
    decode: &Arc<NativeRawOwner>,
    parameter_owner: &KdaRawProducer,
    kernels: &Arc<KdaRawKernels>,
    transport: &str,
    math: &str,
    reserve: usize,
) -> Result<(), String> {
    let heads = parameter_owner.heads;
    let dim = parameter_owner.dim;
    let bank = KdaModelKernelBank {
        kernels: kernels.clone(),
        transport: transport.into(),
        math: math.into(),
    };
    let construct = |capacity, a_elements, a_ptr| unsafe {
        PreparedKdaModelFamily::from_model_storage(
            RetainedPrefillKdaSource {
                inner: prefill.clone(),
            },
            RetainedDecodeKdaSource {
                inner: decode.clone(),
            },
            heads,
            dim,
            capacity,
            reserve,
            parameter_owner.lower,
            parameter_owner.scale,
            parameter_owner.epsilon,
            &bank,
            a_ptr,
            a_elements,
            pointer(&parameter_owner.bias),
            parameter_owner.bias.len(),
        )
    };
    // Singleton capacity is the existing measured short-probe/decode geometry.
    // This cold control is not a long-prefill capacity or serving benchmark.
    let before = memory()?;
    let mut rejected = Vec::new();
    for (label, capacity, a_elements, a_ptr) in [
        ("zero capacity", 0, heads, pointer(&parameter_owner.a_log)),
        (
            "capacity exceeds retained source",
            prefill.schema.planes[0].shape[0]
                .checked_add(1)
                .ok_or("fixture oversized capacity")?,
            heads,
            pointer(&parameter_owner.a_log),
        ),
        (
            "wrong A-log extent",
            1,
            heads.checked_add(1).ok_or("fixture oversized heads")?,
            pointer(&parameter_owner.a_log),
        ),
        ("null loaded A-log", 1, heads, 0),
    ] {
        if construct(capacity, a_elements, a_ptr).is_ok() || memory() != Ok(before) {
            return Err(format!("typed setup negative admitted/allocated:{label}"));
        }
        rejected.push(label);
    }
    let weak_prefill = Arc::downgrade(prefill);
    let weak_decode = Arc::downgrade(decode);
    let count_prefill = weak_prefill.strong_count();
    let count_decode = weak_decode.strong_count();
    let setup = construct(1, heads, pointer(&parameter_owner.a_log))?;
    if weak_prefill.strong_count() != count_prefill + 1
        || weak_decode.strong_count() != count_decode + 1
        || !Arc::ptr_eq(&setup.prefill_producer.raw, prefill)
        || !Arc::ptr_eq(&setup.decode_producer.raw, decode)
        || !Arc::ptr_eq(&setup.prefill.state, &setup.decode.state)
        || Arc::ptr_eq(&setup.prefill.arena, &setup.decode.arena)
        || setup.prefill_producer.arithmetic_sha256 != parameter_owner.arithmetic_sha256
        || setup.decode_producer.arithmetic_sha256 != parameter_owner.arithmetic_sha256
    {
        return Err("typed model setup ownership/arithmetic mismatch".into());
    }
    let mut params = Vec::new();
    for (label, original, a, b) in [
        (
            "a_log",
            &parameter_owner.a_log,
            &setup.prefill_producer.a_log,
            &setup.decode_producer.a_log,
        ),
        (
            "bias",
            &parameter_owner.bias,
            &setup.prefill_producer.bias,
            &setup.decode_producer.bias,
        ),
    ] {
        let read = |x: &GpuBuf<f32>| -> Result<Vec<u8>, String> {
            let mut values = vec![0u8; x.len() * std::mem::size_of::<f32>()];
            check(
                unsafe {
                    sys::lib().cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), pointer(x), values.len())
                },
                "typed setup actual parameter bytes",
            )?;
            Ok(values)
        };
        let bytes = read(original)?;
        if read(a)? != bytes || read(b)? != bytes {
            return Err("loaded preparation parameters changed by setup".into());
        }
        params.push(json!({"role":label,"source_ptr":pointer(original),"bytes":bytes,"prefill_ptr":pointer(a),"decode_ptr":pointer(b)}));
    }
    let mut registry = crate::session_cache::SequenceStateRegistry::default();
    setup.register(&mut registry, 0)?;
    let identity = setup.identity(0)?;
    let borrowed = setup.borrowed()?;
    // Equivalent preparation arithmetic permits a different producer at setup,
    // but a captured immutable proof must reject its different actual storage.
    let original_binding = [borrowed.decode_binding(0)];
    let expected = model_family_identity(&original_binding)?;
    let substituted = [ModelFamilyBinding {
        layer: 0,
        owner: &setup.decode,
        producer: parameter_owner,
    }];
    if ModelBindingProof::checked(&substituted, &expected).is_ok() {
        return Err("typed captured source substitution proof admitted".into());
    }
    let pf = borrowed.prefill_binding(0);
    let df = borrowed.decode_binding(0);
    if pf
        .owner
        .register(
            &mut crate::session_cache::SequenceStateRegistry::default(),
            0,
        )
        .is_err()
        || !Arc::ptr_eq(&df.owner.state, &pf.owner.state)
    {
        return Err("typed family persistent admissions mismatch".into());
    }
    drop(borrowed);
    let floor = memory()?[0];
    if floor < reserve {
        return Err("typed cold model setup reserve floor".into());
    }
    owner_save(root,"typed-model-family-setup.json",&serde_json::to_vec_pretty(&json!({
        "scope":"cold source/parameter/setup API plus full fixture borrowed pair; not loaded model or long capacity qualification",
        "cold_capacity":1,"rejected":rejected,"source_ptrs":{"prefill":prefill.addresses(),"decode":decode.addresses()},
        "identity":identity,"persistent_registry":registry.inventory(1),"parameter_bytes":params,
        "shared_persistent_state":true,"distinct_arenas":true,"sources_retained":true,"floor_bytes":floor,
        "reserve_bytes":reserve,"gpu_recurrence_submitted_in_cold_control":false
    })).map_err(|e|e.to_string())?)?;
    drop(setup);
    if weak_prefill.strong_count() != count_prefill || weak_decode.strong_count() != count_decode {
        return Err("typed model setup source lifetime did not release".into());
    }
    Ok(())
}
