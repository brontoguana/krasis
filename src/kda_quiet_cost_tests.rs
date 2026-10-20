#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuietCostPlan {
    native: LongChronologyPlan,
    warmups: usize,
    samples: usize,
    output_reference: OwnerId,
    bf16_reference: OwnerId,
    prefix_reference: OwnerId,
    final_reference: OwnerId,
}
fn quiet_native_cost_run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev kda-quiet-native".into());
    }
    let compiled_source = option_env!("KRASIS_ARITHMETIC_SOURCE_SHA256")
        .ok_or("quiet fixture requires a build bound to KRASIS_ARITHMETIC_SOURCE_SHA256")?;
    let path = std::env::var("KRASIS_KDA_QUIET_PLAN").map_err(|_| "quiet frozen plan")?;
    let output = std::env::var("KRASIS_KDA_QUIET_OUTPUT").map_err(|_| "quiet fresh output")?;
    let qp: QuietCostPlan = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let p = &qp.native;
    if !p.diagnostic_only
        || p.qualifies_candidate
        || p.raw_inputs.len() != 5
        || qp.samples == 0
        || qp.warmups == 0
    {
        return Err("quiet scope/sample admission".into());
    }
    for (name, value) in std::env::vars() {
        let expected_metadata = match name.as_str() {
            "KRASIS_ARITHMETIC_SOURCE_SHA256" => Some(compiled_source),
            "KRASIS_SIDECAR_ABI_VERSION" => Some(env!("KRASIS_SIDECAR_ABI_VERSION")),
            _ => None,
        };
        if let Some(expected) = expected_metadata {
            if value != expected {
                return Err(format!("compiler identity metadata mismatch: {name}"));
            }
            continue;
        }
        if name.starts_with("KRASIS_")
            && ![
                "KRASIS_DEV_SCRIPT",
                "KRASIS_KDA_QUIET_PLAN",
                "KRASIS_KDA_QUIET_OUTPUT",
            ]
            .contains(&name.as_str())
        {
            return Err(format!(
                "optional instrumentation/control forbidden: {name}"
            ));
        }
    }
    for x in &p.admission {
        owner_checked(x)?;
    }
    owner_checked(&p.executable)?;
    owner_checked(&p.driver)?;
    if fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        != fs::canonicalize(&p.executable.path).map_err(|e| e.to_string())?
    {
        return Err("quiet actualexe".into());
    }
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| e.to_string())?
        .trim()
        != p.boot_id
    {
        return Err("quiet actualboot".into());
    }
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("configured safety")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("reserveoverflow")?;
    let (h, d, t) = (p.heads, p.dim, p.rows);
    let hd = h.checked_mul(d).ok_or("hdoverflow")?;
    let raw = p
        .raw_inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    for (v, stride) in raw.iter().zip([hd * 2, hd * 2, hd * 2, hd * 2, h * 2]) {
        if v.len() != t.checked_mul(stride).ok_or("rawextentoverflow")?
            || v.chunks_exact(2)
                .any(|x| u16::from_le_bytes(x.try_into().unwrap()) & 0x7f80 == 0x7f80)
        {
            return Err("quiet rawinput extent/nonfinite".into());
        }
    }
    let initial = owner_checked(&p.initial)?;
    if initial.len() != h * d * d * 4 || initial.iter().any(|x| *x != 0) {
        return Err("quiet actualemptyinitial".into());
    }
    let expected = owner_checked(&qp.output_reference)?;
    let expected_bf = owner_checked(&qp.bf16_reference)?;
    if expected.len() != t * hd * 4 || expected_bf.len() != t * hd * 2 {
        return Err("quiet referenceextent".into());
    }
    let prefix: OwnerSnapshot =
        serde_json::from_slice(&owner_checked(&qp.prefix_reference)?).map_err(|e| e.to_string())?;
    let final_state: OwnerSnapshot =
        serde_json::from_slice(&owner_checked(&qp.final_reference)?).map_err(|e| e.to_string())?;
    let start = prefix
        .blobs
        .iter()
        .find(|x| x.allocation_name == "layer0.kda_fixed16.absolute")
        .ok_or("quiet genuineprefixcursor")?
        .bytes
        .as_slice();
    let start = usize::try_from(u64::from_le_bytes(
        start.try_into().map_err(|_| "cursorbytes")?,
    ))
    .map_err(|_| "cursoroverflow")?;
    if start >= t {
        return Err("quiet suffixbounds".into());
    }
    let device = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let mut ordinal = 0;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    check(
        unsafe { sys::lib().cuCtxGetDevice(&mut ordinal) },
        "quiet device",
    )?;
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, ordinal) },
        "quiet uuid",
    )?;
    let u = uuid
        .bytes
        .iter()
        .map(|x| format!("{:02x}", *x as u8))
        .collect::<String>();
    let actual_uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &u[..8],
        &u[8..12],
        &u[12..16],
        &u[16..20],
        &u[20..]
    );
    let cc = [
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?,
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?,
    ];
    if actual_uuid != p.uuid || cc != p.compute {
        return Err("quiet actualGPU".into());
    }
    let root = Path::new(&output);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    owner_save(
        root,
        "ACTUAL_PLAN.json",
        &fs::read(&path).map_err(|e| e.to_string())?,
    )?;
    let stream = NativeStream::create(device.clone())?;
    let before = memory()?;
    let transport =
        String::from_utf8(owner_checked(&p.transport_ptx)?).map_err(|e| e.to_string())?;
    let math = String::from_utf8(owner_checked(&p.math_ptx)?).map_err(|e| e.to_string())?;
    let producer_ptx =
        String::from_utf8(owner_checked(&p.producer_ptx)?).map_err(|e| e.to_string())?;
    let a = PreparedOwner::new_on_stream(
        device.clone(),
        stream.clone(),
        h,
        d,
        t,
        reserve,
        &transport,
        &math,
    )?;
    let singleton = PreparedOwner::new_on_stream(
        device.clone(),
        stream.clone(),
        h,
        d,
        1,
        reserve,
        &transport,
        &math,
    )?;
    let mut full_registry = crate::session_cache::SequenceStateRegistry::default();
    a.register(&mut full_registry, 0)?;
    let mut registry = crate::session_cache::SequenceStateRegistry::default();
    singleton.register(&mut registry, 0)?;
    let a_log = owner_checked(&p.a_log)?;
    let bias = owner_checked(&p.bias)?;
    let params = [a_log.as_slice(), bias.as_slice()];
    let kernels = KdaRawKernels::load(device.clone(), &producer_ptx)?;
    let full = KdaRawProducer::new(
        stream.clone(),
        h,
        d,
        t,
        reserve,
        p.lower,
        p.scale,
        p.epsilon,
        kernels.clone(),
        &a_log,
        &bias,
    )?;
    let one = KdaRawProducer::new(
        stream.clone(),
        h,
        d,
        1,
        reserve,
        p.lower,
        p.scale,
        p.epsilon,
        kernels,
        &a_log,
        &bias,
    )?;
    kda_raw_upload(&full.raw, &raw)?;
    let graph = kda_raw_capture(&singleton, &one)?;
    let inventory = kda_raw_graph_inventory(&singleton, &one, &graph)?;
    let after_alloc = memory()?;
    stream.drain()?;
    let enqueue = || -> Result<(), String> {
        let lease = a.arena.lease()?;
        full.enqueue(0, t, &a)?;
        a.prefill_in(&lease, &full.view(t)?, 0, t)
    };
    let check_full = || -> Result<(), String> {
        long_check_cursor(&a, t)?;
        owner_compare(
            &owner_array(&a, "emitted", t * hd)?,
            &expected,
            "quiet whole emitted correctness",
        )?;
        owner_compare(
            &vector_bf16_array(&a, t * hd)?,
            &expected_bf,
            "quiet wholeBF16correctness",
        )?;
        native_snapshot_equal(
            &a,
            &full_registry,
            &final_state,
            "quiet complete full final state",
        )
    };
    let mut parameter_files = Vec::new();
    long_parameters_save(
        root,
        "full-before-measurement",
        &full,
        &params,
        &mut parameter_files,
    )?;
    long_parameters_save(
        root,
        "one-before-measurement",
        &one,
        &params,
        &mut parameter_files,
    )?;
    a.reset()?;
    enqueue()?;
    stream.drain()?;
    check_full()?;
    for _ in 0..qp.warmups {
        a.reset()?;
        stream.drain()?;
        enqueue()?;
        stream.drain()?;
    }
    let mut full_seconds = Vec::new();
    for _ in 0..qp.samples {
        a.reset()?;
        stream.drain()?;
        let began = std::time::Instant::now();
        enqueue()?;
        stream.drain()?;
        full_seconds.push(began.elapsed().as_secs_f64());
    }
    check_full()?;
    long_parameters_save(
        root,
        "full-after-measurement",
        &full,
        &params,
        &mut parameter_files,
    )?;
    let check_graph = || -> Result<(), String> {
        restore_owner(&singleton, &registry, &prefix)?;
        stream.drain()?;
        for row in start..t {
            kda_raw_upload(&one.raw, &kda_raw_slice(&raw, row, 1, h, d)?)?;
            {
                let lease = singleton.arena.lease()?;
                singleton.launch_graph_in(&lease, &graph)?;
            }
            stream.drain()?;
            long_check_cursor(&singleton, row + 1)?;
            owner_compare(
                &owner_array(&singleton, "emitted", hd)?,
                &expected[row * hd * 4..(row + 1) * hd * 4],
                "quiet graph correctness",
            )?;
            owner_compare(
                &vector_bf16_array(&singleton, hd)?,
                &expected_bf[row * hd * 2..(row + 1) * hd * 2],
                "quiet graphBF16correctness",
            )?;
        }
        native_snapshot_equal(
            &singleton,
            &registry,
            &final_state,
            "quiet complete graph final state",
        )
    };
    check_graph()?;
    let mut continuation_seconds = Vec::new();
    for pass in 0..qp.warmups + qp.samples {
        restore_owner(&singleton, &registry, &prefix)?;
        stream.drain()?;
        let mut seconds = Vec::new();
        for row in start..t {
            kda_raw_upload(&one.raw, &kda_raw_slice(&raw, row, 1, h, d)?)?;
            stream.drain()?;
            let began = std::time::Instant::now();
            {
                let lease = singleton.arena.lease()?;
                singleton.launch_graph_in(&lease, &graph)?;
            }
            stream.drain()?;
            seconds.push(began.elapsed().as_secs_f64());
        }
        if pass >= qp.warmups {
            continuation_seconds.push(seconds);
        }
    }
    check_graph()?;
    long_parameters_save(
        root,
        "one-after-measurement",
        &one,
        &params,
        &mut parameter_files,
    )?;
    for x in &p.admission {
        owner_checked(x)?;
    }
    owner_checked(&p.executable)?;
    let report = json!({"passed":true,"actual_uuid":actual_uuid,"compute":cc,"rows":t,"prefix_rows":start,"graph_rows":t-start,"warmups":qp.warmups,"samples":qp.samples,"full_raw_producer_and_batch_recurrence_seconds":full_seconds,"graph_singleton_seconds":continuation_seconds,"memory_before":before,"memory_after_allocations":after_alloc,"memory_final":memory()?,"reserve_bytes":reserve,"arena_bytes":a.arena.requested_bytes,"persistent_bytes_each":a.persistent_bytes,"producer_bytes":full.requested_bytes,"graph_inventory":inventory,"parameters":parameter_files,"measurement":"Rust enqueue plus ownedstream completion; resets/restores/uploads/correctness outside timing; optional instrumentation disabled","scope":"isolated operator cost only, not model speedregression/HTTP/ordinary gate","adopted":false});
    owner_save(
        root,
        "REPORT.json",
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    Ok(())
}
#[test]
fn actual_native_quiet_raw_cost() {
    quiet_native_cost_run().unwrap()
}
