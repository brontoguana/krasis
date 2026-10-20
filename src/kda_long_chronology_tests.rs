#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LongChronologyPlan {
    diagnostic_only: bool,
    qualifies_candidate: bool,
    executable: OwnerId,
    driver: OwnerId,
    runtime_config: OwnerId,
    admission: Vec<OwnerId>,
    boot_id: String,
    uuid: String,
    compute: [i32; 2],
    heads: usize,
    dim: usize,
    rows: usize,
    raw_inputs: Vec<OwnerId>,
    initial: OwnerId,
    a_log: OwnerId,
    bias: OwnerId,
    transport_ptx: OwnerId,
    math_ptx: OwnerId,
    producer_ptx: OwnerId,
    lower: f32,
    scale: f32,
    epsilon: f32,
    prefix_ends: Vec<usize>,
    graph_suffix_starts: Vec<usize>,
}
fn long_snapshot_save(root: &Path, label: &str, s: &OwnerSnapshot) -> Result<OwnerId, String> {
    owner_save(
        root,
        label,
        &serde_json::to_vec(s).map_err(|e| e.to_string())?,
    )
}

fn long_parameters_save(
    root: &Path,
    label: &str,
    p: &KdaRawProducer,
    expected: &[&[u8]],
    retained: &mut Vec<OwnerId>,
) -> Result<(), String> {
    p.prepared.stream.validate()?;
    for (i, (ptr, want)) in [pointer(&p.a_log), pointer(&p.bias)]
        .into_iter()
        .zip(expected)
        .enumerate()
    {
        let mut actual = vec![0u8; want.len()];
        check(
            unsafe {
                sys::lib().cuMemcpyDtoHAsync_v2(
                    actual.as_mut_ptr().cast(),
                    ptr,
                    actual.len(),
                    p.prepared.stream.handle,
                )
            },
            "diagnostic owned actual parameter read",
        )?;
        p.prepared.stream.drain()?;
        retained.push(owner_save(
            root,
            &format!("{label}-parameter{i}.f32"),
            &actual,
        )?);
        owner_compare(&actual, want, "actual complete parameters before consumer")?;
    }
    Ok(())
}

fn long_check_cursor(o: &PreparedOwner, end: usize) -> Result<(), String> {
    if owner_metadata(o)? != ((end % 16) as i32, 0, end as u64) {
        return Err(format!("long chronology cursor at {end}"));
    }
    Ok(())
}
fn long_read_save(
    root: &Path,
    label: &str,
    o: &PreparedOwner,
    producer: &KdaRawProducer,
    rows: usize,
    retained: &mut Vec<OwnerId>,
) -> Result<(Vec<u8>, Vec<u8>, Vec<Vec<u8>>), String> {
    o.stream.drain()?;
    let f = owner_array(o, "emitted", rows * o.heads * o.dim)?;
    let bf = vector_bf16_array(o, rows * o.heads * o.dim)?;
    let planes = kda_prepared_read(producer, rows)?;
    // Retain produced operands/results before comparison can reject an attempt.
    retained.push(owner_save(root, &format!("{label}-output.f32"), &f)?);
    retained.push(owner_save(root, &format!("{label}-output.bf16"), &bf)?);
    for (i, v) in planes.iter().enumerate() {
        retained.push(owner_save(root, &format!("{label}-prepared{i}.f32"), v)?);
        if v.chunks_exact(4)
            .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
        {
            return Err(format!("nonfinite actual prepared plane {i}"));
        }
    }
    owner_compare(&bf, &vector_rne_view(&f)?, "long actual RNE view")?;
    Ok((f, bf, planes))
}
fn long_partition(
    root: &Path,
    label: &str,
    owner: &PreparedOwner,
    producer: &KdaRawProducer,
    raw: &[Vec<u8>],
    first: usize,
    end: usize,
    quantum: usize,
    want_f: &[u8],
    want_b: &[u8],
    want_p: &[Vec<u8>],
    parameters: &[&[u8]],
    retained: &mut Vec<OwnerId>,
) -> Result<(), String> {
    let (h, d) = (owner.heads, owner.dim);
    let mut at = first;
    while at < end {
        let count = quantum.min(end - at);
        long_parameters_save(
            root,
            &format!("{label}-at{at}"),
            producer,
            parameters,
            retained,
        )?;
        let mut data = kda_raw_slice(raw, at, count, h, d)?;
        // Producers have fixed storage extent. Inactive capacity never enters view(count).
        for (x, (desc, _)) in data.iter_mut().zip(producer.raw.inventory()) {
            x.resize(desc.storage_bytes, 0);
        }
        kda_raw_upload(&producer.raw, &data)?;
        {
            let lease = owner.arena.lease()?;
            producer.enqueue(0, count, owner)?;
            owner.prefill_in(&lease, &producer.view(count)?, at as u64, count)?;
        }
        long_check_cursor(owner, at + count)?;
        let f = owner_array(owner, "emitted", count * h * d)?;
        let bf = vector_bf16_array(owner, count * h * d)?;
        let planes = kda_prepared_read(producer, count)?;
        let row_stride = h * d;
        let checks = (|| {
            owner_compare(
                &f,
                &want_f[at * row_stride * 4..(at + count) * row_stride * 4],
                "long partition complete F32 rows",
            )?;
            owner_compare(
                &bf,
                &want_b[at * row_stride * 2..(at + count) * row_stride * 2],
                "long partition complete BF16 rows",
            )?;
            for (i, v) in planes.iter().enumerate() {
                let stride = if i == 4 { h * 4 } else { h * d * 4 };
                owner_compare(
                    v,
                    &want_p[i][at * stride..(at + count) * stride],
                    "long partition prepared operands",
                )?;
            }
            Ok::<(), String>(())
        })();
        if let Err(e) = checks {
            retained.push(owner_save(
                root,
                &format!("{label}-fault-at{at}-output.f32"),
                &f,
            )?);
            retained.push(owner_save(
                root,
                &format!("{label}-fault-at{at}-output.bf16"),
                &bf,
            )?);
            for (i, v) in planes.iter().enumerate() {
                retained.push(owner_save(
                    root,
                    &format!("{label}-fault-at{at}-prepared{i}.f32"),
                    v,
                )?);
            }
            return Err(e);
        }
        at += count;
    }
    Ok(())
}
fn run_native_long_raw_chronology() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use built ./dev kda-long-native".into());
    }
    let plan_path = std::env::var("KRASIS_KDA_LONG_PLAN").map_err(|_| "long frozen plan")?;
    let output = std::env::var("KRASIS_KDA_LONG_OUTPUT").map_err(|_| "long fresh output")?;
    let p: LongChronologyPlan =
        serde_json::from_slice(&fs::read(&plan_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if !p.diagnostic_only
        || p.qualifies_candidate
        || p.raw_inputs.len() != 5
        || p.rows < 33
        || p.prefix_ends.is_empty()
        || p.prefix_ends.iter().any(|&n| n == 0 || n >= p.rows)
        || p.graph_suffix_starts.iter().any(|&n| n == 0 || n >= p.rows)
    {
        return Err("long diagnostic scope/complete endpoint admission".into());
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
        return Err("long actual executable".into());
    }
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| e.to_string())?
        .trim()
        != p.boot_id
    {
        return Err("long actual boot".into());
    }
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("long configured reserve")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("long reserve overflow")?;
    let (h, d, t) = (p.heads, p.dim, p.rows);
    let hd = h.checked_mul(d).ok_or("long hd overflow")?;
    let st = hd.checked_mul(d).ok_or("long state overflow")?;
    let raw = p
        .raw_inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    for (v, stride) in raw.iter().zip([hd * 2, hd * 2, hd * 2, hd * 2, h * 2]) {
        if v.len() != t.checked_mul(stride).ok_or("long raw overflow")? {
            return Err("long raw extent".into());
        }
        if v.chunks_exact(2)
            .any(|x| u16::from_le_bytes(x.try_into().unwrap()) & 0x7f80 == 0x7f80)
        {
            return Err("long raw nonfinite".into());
        }
    }
    let initial = owner_checked(&p.initial)?;
    if initial.len() != st * 4 || initial.iter().any(|&x| x != 0) {
        return Err("long verified empty initial".into());
    }
    let root = Path::new(&output);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    owner_save(
        root,
        "ACTUAL_PLAN.json",
        &fs::read(&plan_path).map_err(|e| e.to_string())?,
    )?;
    let device = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let mut ordinal = 0;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    check(
        unsafe { sys::lib().cuCtxGetDevice(&mut ordinal) },
        "long context device",
    )?;
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, ordinal) },
        "long actual uuid",
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
    let actual_cc = [
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?,
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?,
    ];
    if actual_uuid != p.uuid || actual_cc != p.compute {
        return Err("long actual architecture".into());
    }
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
    let block = a.pending_raw.schema.planes[0].shape[0];
    let bounded = block
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .ok_or("long bounded capacity")?;
    let b = PreparedOwner::new_on_stream(
        device.clone(),
        stream.clone(),
        h,
        d,
        bounded,
        reserve,
        &transport,
        &math,
    )?;
    let c = PreparedOwner::from_shared(&b, reserve)?;
    let mut ra = crate::session_cache::SequenceStateRegistry::default();
    a.register(&mut ra, 0)?;
    let mut rb = crate::session_cache::SequenceStateRegistry::default();
    b.register(&mut rb, 0)?;
    let mut rc = crate::session_cache::SequenceStateRegistry::default();
    c.register(&mut rc, 0)?;
    let a_log = owner_checked(&p.a_log)?;
    let bias = owner_checked(&p.bias)?;
    let parameters = [a_log.as_slice(), bias.as_slice()];
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
    let chunks = KdaRawProducer::new(
        stream.clone(),
        h,
        d,
        bounded,
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
    let graph = kda_raw_capture(&c, &one)?;
    let graph_inventory = kda_raw_graph_inventory(&c, &one, &graph)?;
    let after = memory()?;
    if after[0] < reserve {
        return Err("long actual allocation reserve".into());
    }
    let mut retained = Vec::new();
    let mut endpoints = Vec::new();
    let work = (|| {
        for (label, producer) in [("full", &full), ("bounded", &chunks), ("singleton", &one)] {
            long_parameters_save(
                root,
                &format!("before-first-{label}"),
                producer,
                &parameters,
                &mut retained,
            )?;
        }
        vector_seed(&a, &initial)?;
        {
            let lease = a.arena.lease()?;
            full.enqueue(0, t, &a)?;
            a.prefill_in(&lease, &full.view(t)?, 0, t)?;
        }
        long_check_cursor(&a, t)?;
        let (want_f, want_b, want_p) = long_read_save(root, "whole", &a, &full, t, &mut retained)?;
        let final_state = capture_owner(&a, &ra)?;
        retained.push(long_snapshot_save(
            root,
            "whole-final-snapshot.json",
            &final_state,
        )?);
        for &end in &p.prefix_ends {
            long_parameters_save(
                root,
                &format!("prefix{end}-before-full"),
                &full,
                &parameters,
                &mut retained,
            )?;
            vector_seed(&a, &initial)?;
            {
                let lease = a.arena.lease()?;
                full.enqueue(0, end, &a)?;
                a.prefill_in(&lease, &full.view(end)?, 0, end)?;
            }
            long_check_cursor(&a, end)?;
            let prefix = capture_owner(&a, &ra)?;
            retained.push(long_snapshot_save(
                root,
                &format!("prefix{end}-full-snapshot.json"),
                &prefix,
            )?);
            let (f, bf, planes) =
                long_read_save(root, &format!("prefix{end}"), &a, &full, end, &mut retained)?;
            owner_compare(&f, &want_f[..end * hd * 4], "long complete full-prefix F32")?;
            owner_compare(
                &bf,
                &want_b[..end * hd * 2],
                "long complete full-prefix BF16",
            )?;
            for (i, v) in planes.iter().enumerate() {
                let s = if i == 4 { h * 4 } else { hd * 4 };
                owner_compare(
                    v,
                    &want_p[i][..end * s],
                    "long complete full-prefix prepared",
                )?;
            }
            vector_seed(&b, &initial)?;
            long_partition(
                root,
                &format!("prefix{end}-bounded"),
                &b,
                &chunks,
                &raw,
                0,
                end,
                bounded,
                &want_f,
                &want_b,
                &want_p,
                &parameters,
                &mut retained,
            )?;
            let actual = capture_owner(&b, &rb)?;
            retained.push(long_snapshot_save(
                root,
                &format!("prefix{end}-bounded-snapshot.json"),
                &actual,
            )?);
            native_snapshot_equal(&b, &rb, &prefix, "long whole-prefix persistent state")?;
            restore_owner(&c, &rc, &prefix)?;
            native_snapshot_equal(&c, &rc, &prefix, "long fresh complete restore")?;
            long_partition(
                root,
                &format!("prefix{end}-suffix"),
                &c,
                &chunks,
                &raw,
                end,
                t,
                bounded,
                &want_f,
                &want_b,
                &want_p,
                &parameters,
                &mut retained,
            )?;
            let actual = capture_owner(&c, &rc)?;
            retained.push(long_snapshot_save(
                root,
                &format!("prefix{end}-suffix-snapshot.json"),
                &actual,
            )?);
            native_snapshot_equal(&c, &rc, &final_state, "long full restored suffix state")?;
            // Compare a single full GPU suffix batch separately from bounded execution.
            restore_owner(&a, &ra, &prefix)?;
            long_parameters_save(
                root,
                &format!("split{end}-before-full"),
                &full,
                &parameters,
                &mut retained,
            )?;
            {
                let lease = a.arena.lease()?;
                full.enqueue(end, t - end, &a)?;
                a.prefill_in(&lease, &full.view(t - end)?, end as u64, t - end)?;
            }
            long_check_cursor(&a, t)?;
            let (f, bf, planes) = long_read_save(
                root,
                &format!("split{end}"),
                &a,
                &full,
                t - end,
                &mut retained,
            )?;
            owner_compare(&f, &want_f[end * hd * 4..], "long one-batch suffix F32")?;
            owner_compare(&bf, &want_b[end * hd * 2..], "long one-batch suffix BF16")?;
            for (i, v) in planes.iter().enumerate() {
                let s = if i == 4 { h * 4 } else { hd * 4 };
                owner_compare(v, &want_p[i][end * s..], "long one-batch suffix prepared")?;
            }
            let actual = capture_owner(&a, &ra)?;
            retained.push(long_snapshot_save(
                root,
                &format!("split{end}-snapshot.json"),
                &actual,
            )?);
            native_snapshot_equal(&a, &ra, &final_state, "long one-batch suffix state")?;
            let graph_rows = if p.graph_suffix_starts.contains(&end) {
                restore_owner(&c, &rc, &prefix)?;
                for row in end..t {
                    long_parameters_save(
                        root,
                        &format!("graph{end}-row{row}-before"),
                        &one,
                        &parameters,
                        &mut retained,
                    )?;
                    kda_raw_upload(&one.raw, &kda_raw_slice(&raw, row, 1, h, d)?)?;
                    {
                        let lease = c.arena.lease()?;
                        c.launch_graph_in(&lease, &graph)?;
                    }
                    long_check_cursor(&c, row + 1)?;
                    let (f, bf, planes) = long_read_save(
                        root,
                        &format!("graph{end}-row{row}"),
                        &c,
                        &one,
                        1,
                        &mut retained,
                    )?;
                    owner_compare(
                        &f,
                        &want_f[row * hd * 4..(row + 1) * hd * 4],
                        "long singleton F32",
                    )?;
                    owner_compare(
                        &bf,
                        &want_b[row * hd * 2..(row + 1) * hd * 2],
                        "long singleton BF16",
                    )?;
                    for (i, v) in planes.iter().enumerate() {
                        let s = if i == 4 { h * 4 } else { hd * 4 };
                        owner_compare(
                            v,
                            &want_p[i][row * s..(row + 1) * s],
                            "long singleton operands",
                        )?;
                    }
                }
                retained.push(long_snapshot_save(
                    root,
                    &format!("graph{end}-final-snapshot.json"),
                    &capture_owner(&c, &rc)?,
                )?);
                native_snapshot_equal(
                    &c,
                    &rc,
                    &final_state,
                    "long captured restored suffix state",
                )?;
                t - end
            } else {
                0
            };
            endpoints.push(json!({"end":end,"prefix_rows_exact":end,"bounded_capacity":bounded,"suffix_rows_exact":t-end,"graph_suffix_rows_exact":graph_rows,"persistent_complete_exact":true}));
            owner_save(
                root,
                &format!("ENDPOINT{end}.json"),
                &serde_json::to_vec_pretty(endpoints.last().unwrap()).map_err(|e| e.to_string())?,
            )?;
        }
        Ok::<(), String>(())
    })();
    retained.push(owner_save(
        root,
        "process.maps",
        &fs::read("/proc/self/maps").map_err(|e| e.to_string())?,
    )?);
    let report = json!({"passed":work.is_ok(),"failure":work.as_ref().err(),"actual_uuid":actual_uuid,"compute":actual_cc,"rows":t,"endpoints":endpoints,"retained":retained,"owner_arithmetic":a.arithmetic_sha256,"producer_arithmetic":full.arithmetic_sha256,"graph_inventory":graph_inventory,"memory_before":before,"memory_after":after,"memory_final":memory()?,"reserve_bytes":reserve,"owner_full_arena_bytes":a.arena.requested_bytes,"owner_bounded_arena_bytes":b.arena.requested_bytes,"owner_persistent_bytes_each":a.persistent_bytes,"producer_full_bytes":full.requested_bytes,"producer_bounded_bytes":chunks.requested_bytes,"producer_singleton_bytes":one.requested_bytes,"A_inventory":ra.inventory(t),"B_inventory":rb.inventory(t),"C_inventory":rc.inventory(t),"adopted":false,"ordinary_gate":"FAILED_UNCHANGED","references":"NEW_F64_AND_WITNESS_REQUIRED","performance":"NOT_MEASURED","native_windows":"OUTSTANDING","scope":"one frozen authentic long raw input; independent full batch, prefix, bounded partition, fresh restore and selected authentic nonzero graph suffix; no actual full/cache cause attribution or serving qualification"});
    owner_save(
        root,
        "REPORT.json",
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    work
}
#[test]
fn actual_native_long_raw_chronology() {
    run_native_long_raw_chronology().unwrap();
}

include!("kda_quiet_cost_tests.rs");

include!("kda_tiling_tests.rs");
