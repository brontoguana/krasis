// Authentic original source9 coverage for the current retained owner. Diagnostic only.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalLongCurrentPlan {
    diagnostic_only: bool,
    qualifies_candidate: bool,
    executable: OwnerId,
    cuda_runtime: OwnerId,
    admission: Vec<OwnerId>,
    runtime_config: OwnerId,
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
    stream_ends: Vec<usize>,
    restore_at: usize,
}
fn original_snapshot_save(root: &Path, label: &str, s: &OwnerSnapshot) -> Result<Value, String> {
    let mut blobs = Vec::new();
    for (i, b) in s.blobs.iter().enumerate() {
        let data = owner_save(root, &format!("{label}-blob{i}.bin"), &b.bytes)?;
        blobs.push(json!({"name":b.allocation_name,"kind":b.kind,"layer":b.layer_idx,"dtype":b.dtype,"element_size":b.element_size,"shape":b.shape,"strides_bytes":b.strides_bytes,"data":data}));
    }
    let v = json!({"arithmetic_sha256":s.arithmetic_sha256,"layout":s.layout,"blobs":blobs});
    owner_save(
        root,
        &format!("{label}-snapshot.json"),
        &serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?,
    )?;
    Ok(v)
}
fn original_complete_equal(
    root: &Path,
    label: &str,
    got: &[u8],
    want: &[u8],
) -> Result<(), String> {
    if got != want {
        owner_save(root, &format!("{label}-failure.bin"), got)?;
        let differing_bytes =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        owner_save(root,&format!("{label}-failure.json"),&serde_json::to_vec(&json!({"differing_bytes":differing_bytes,"got_bytes":got.len(),"want_bytes":want.len(),"first_differing_byte":got.iter().zip(want).position(|(a,b)|a!=b)})).map_err(|e|e.to_string())?)?;
        return owner_compare(got, want, label);
    }
    Ok(())
}
fn original_read_compare(
    root: &Path,
    label: &str,
    o: &PreparedOwner,
    producer: &KdaRawProducer,
    first: usize,
    rows: usize,
    f: &[u8],
    bf: &[u8],
    prepared: &[Vec<u8>],
) -> Result<Value, String> {
    o.stream.drain()?;
    let hd = o.heads * o.dim;
    let out = owner_array(o, "emitted", rows * hd)?;
    let view = vector_bf16_array(o, rows * hd)?;
    original_complete_equal(
        root,
        &format!("{label}-output"),
        &out,
        &f[first * hd * 4..(first + rows) * hd * 4],
    )?;
    original_complete_equal(
        root,
        &format!("{label}-bf16"),
        &view,
        &bf[first * hd * 2..(first + rows) * hd * 2],
    )?;
    let planes = kda_prepared_read(producer, rows)?;
    let mut hashes = Vec::new();
    for (i, v) in planes.iter().enumerate() {
        let stride = if i == 4 { o.heads * 4 } else { hd * 4 };
        original_complete_equal(
            root,
            &format!("{label}-prepared{i}"),
            v,
            &prepared[i][first * stride..(first + rows) * stride],
        )?;
        hashes.push(format!("{:x}", Sha256::digest(v)));
    }
    Ok(
        json!({"first":first,"rows":rows,"output_sha256":format!("{:x}",Sha256::digest(&out)),"bf16_sha256":format!("{:x}",Sha256::digest(&view)),"prepared_sha256":hashes,"complete_bytes_exact":true}),
    )
}
fn original_batch(
    o: &PreparedOwner,
    producer: &KdaRawProducer,
    raw_first: usize,
    first: usize,
    rows: usize,
) -> Result<(), String> {
    let lease = o.arena.lease()?;
    producer.enqueue(raw_first, rows, o)?;
    o.prefill_in(&lease, &producer.view(rows)?, first as u64, rows)?;
    o.stream.drain()?;
    long_check_cursor(o, first + rows)
}
fn run_original_long_current_owner() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev original-long-native".into());
    }
    let plan_path = std::env::var("KRASIS_KDA_ORIGINAL_LONG_PLAN")
        .map_err(|_| "original current frozen plan")?;
    let output = std::env::var("KRASIS_KDA_ORIGINAL_LONG_OUTPUT")
        .map_err(|_| "original current fresh output")?;
    let p: OriginalLongCurrentPlan =
        serde_json::from_slice(&fs::read(&plan_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if !p.diagnostic_only
        || p.qualifies_candidate
        || p.raw_inputs.len() != 5
        || p.prefix_ends.is_empty()
        || p.prefix_ends.iter().any(|&x| x == 0 || x > p.rows)
        || !p.prefix_ends.contains(&p.rows)
        || !p.prefix_ends.contains(&p.restore_at)
        || p.restore_at == 0
        || p.restore_at >= p.rows
        || p.prefix_ends.windows(2).any(|w| w[0] >= w[1])
        || p.stream_ends.iter().any(|x| !p.prefix_ends.contains(x))
        || (p.restore_at + 1..=p.rows).any(|x| !p.prefix_ends.contains(&x))
        || p.stream_ends.windows(2).any(|w| w[0] >= w[1])
        || p.stream_ends.first() == Some(&0)
        || p.stream_ends.last() != Some(&p.rows)
    {
        return Err("original current complete diagnostic scope".into());
    }
    for x in p.admission.iter().chain([&p.executable, &p.runtime_config]) {
        owner_checked(x)?;
    }
    if fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        != fs::canonicalize(&p.executable.path).map_err(|e| e.to_string())?
    {
        return Err("original current actual executable".into());
    }
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| e.to_string())?
        .trim()
        != p.boot_id
    {
        return Err("original current boot identity".into());
    }
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("original current configured reserve")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("original current reserve overflow")?;
    let (h, d, t) = (p.heads, p.dim, p.rows);
    let hd = h.checked_mul(d).ok_or("original current hd")?;
    let st = hd.checked_mul(d).ok_or("original current state")?;
    PreparedLayout::checked(h, d, t)?;
    let raw = p
        .raw_inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    for (v, width) in raw.iter().zip([hd * 2, hd * 2, hd * 2, hd * 2, h * 2]) {
        if v.len() != t.checked_mul(width).ok_or("original current raw size")?
            || v.chunks_exact(2)
                .any(|x| u16::from_le_bytes(x.try_into().unwrap()) & 0x7f80 == 0x7f80)
        {
            return Err("original current raw extent/range".into());
        }
    }
    let initial = owner_checked(&p.initial)?;
    if initial.len() != st * 4 || initial.iter().any(|&x| x != 0) {
        return Err("original current actual cold initial".into());
    }
    let a_log = owner_checked(&p.a_log)?;
    let bias = owner_checked(&p.bias)?;
    let transport =
        String::from_utf8(owner_checked(&p.transport_ptx)?).map_err(|e| e.to_string())?;
    let math = String::from_utf8(owner_checked(&p.math_ptx)?).map_err(|e| e.to_string())?;
    let producer_ptx =
        String::from_utf8(owner_checked(&p.producer_ptx)?).map_err(|e| e.to_string())?;
    let root = Path::new(&output);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    owner_save(
        root,
        "ACTUAL_PLAN.json",
        &fs::read(&plan_path).map_err(|e| e.to_string())?,
    )?;
    owner_checked(&p.cuda_runtime)?;
    let _runtime =
        unsafe { libloading::Library::new(&p.cuda_runtime.path) }.map_err(|e| e.to_string())?;
    let device = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let mut native_uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(&mut native_uuid, device.ordinal() as i32) },
        "original current uuid",
    )?;
    let hex = native_uuid
        .bytes
        .iter()
        .map(|x| format!("{:02x}", *x as u8))
        .collect::<String>();
    let uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    let cc = [
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?,
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?,
    ];
    if uuid != p.uuid || cc != p.compute {
        return Err("original current actual architecture".into());
    }
    let _monitor = crate::vram_monitor::native_calibration_monitor(
        vec![device.ordinal() as i32],
        u64::try_from(reserve / (1024 * 1024)).map_err(|_| "original current monitor reserve")?,
    )?;
    let window = crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?
        .ok_or("original current precision disabled")?;
    let before = memory()?;
    let stream = NativeStream::create(device.clone())?;
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
    let c = PreparedOwner::from_shared(&a, reserve)?;
    let block = a.pending_raw.schema.planes[0].shape[0];
    let bounded = block
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .ok_or("original bounded capacity")?;
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
    let mut ra = crate::session_cache::SequenceStateRegistry::default();
    a.register(&mut ra, 0)?;
    let mut rb = crate::session_cache::SequenceStateRegistry::default();
    b.register(&mut rb, 0)?;
    let mut rc = crate::session_cache::SequenceStateRegistry::default();
    c.register(&mut rc, 0)?;
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
        kernels,
        &a_log,
        &bias,
    )?;
    let allocated = memory()?;
    if allocated[0] < reserve {
        return Err("original current actual allocation margin".into());
    }
    let mut endpoints = Vec::new();
    let mut partitions = Vec::new();
    let mut restores = Vec::new();
    let mut retained = Vec::new();
    let work = (|| {
        long_parameters_save(
            root,
            "full-parameters",
            &full,
            &[a_log.as_slice(), bias.as_slice()],
            &mut retained,
        )?;
        long_parameters_save(
            root,
            "bounded-parameters",
            &chunks,
            &[a_log.as_slice(), bias.as_slice()],
            &mut retained,
        )?;
        kda_raw_upload(&full.raw, &raw)?;
        vector_seed(&a, &initial)?;
        original_batch(&a, &full, 0, 0, t)?;
        let want_f = owner_array(&a, "emitted", t * hd)?;
        let want_b = vector_bf16_array(&a, t * hd)?;
        let want_p = kda_prepared_read(&full, t)?;
        if want_f
            .chunks_exact(4)
            .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
            || want_p.iter().any(|v| {
                v.chunks_exact(4)
                    .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
            })
        {
            return Err("original current baseline nonfinite".into());
        }
        owner_compare(
            &want_b,
            &vector_rne_view(&want_f)?,
            "original current full RNE",
        )?;
        retained.push(owner_save(root, "whole-output.f32", &want_f)?);
        retained.push(owner_save(root, "whole-output.bf16", &want_b)?);
        for (i, v) in want_p.iter().enumerate() {
            retained.push(owner_save(root, &format!("whole-prepared{i}.f32"), v)?);
        }
        let mut snapshots = BTreeMap::new();
        snapshots.insert(t, capture_owner(&a, &ra)?);
        original_snapshot_save(root, "whole", snapshots.get(&t).unwrap())?;
        for &end in &p.prefix_ends {
            if end == t {
                continue;
            }
            let mut poisoned = raw.clone();
            for (v, stride) in poisoned
                .iter_mut()
                .zip([hd * 2, hd * 2, hd * 2, hd * 2, h * 2])
            {
                for x in v[end * stride..].chunks_exact_mut(2) {
                    x.copy_from_slice(&0x7fc1u16.to_le_bytes());
                }
            }
            kda_raw_upload(&full.raw, &poisoned)?;
            vector_seed(&a, &initial)?;
            original_batch(&a, &full, 0, 0, end)?;
            let mut result = original_read_compare(
                root,
                &format!("prefix{end}"),
                &a,
                &full,
                0,
                end,
                &want_f,
                &want_b,
                &want_p,
            )?;
            let state = capture_owner(&a, &ra)?;
            result["snapshot"] = original_snapshot_save(root, &format!("prefix{end}"), &state)?;
            snapshots.insert(end, state);
            endpoints.push(result);
            owner_save(
                root,
                &format!("PREFIX{end}.json"),
                &serde_json::to_vec_pretty(endpoints.last().unwrap()).map_err(|e| e.to_string())?,
            )?;
        }
        kda_raw_upload(&full.raw, &raw)?;
        vector_seed(&a, &initial)?;
        let mut first = 0;
        for &end in &p.stream_ends {
            original_batch(&a, &full, first, first, end - first)?;
            let result = original_read_compare(
                root,
                &format!("stream{first}-{end}"),
                &a,
                &full,
                first,
                end - first,
                &want_f,
                &want_b,
                &want_p,
            )?;
            let state = capture_owner(&a, &ra)?;
            original_snapshot_save(root, &format!("stream{end}"), &state)?;
            native_snapshot_equal(
                &a,
                &ra,
                snapshots
                    .get(&end)
                    .ok_or("original current stream reference endpoint")?,
                "original current all partition persistent blobs",
            )?;
            partitions.push(result);
            first = end;
        }
        let saved = snapshots
            .get(&p.restore_at)
            .ok_or("original current nonzero snapshot")?;
        let nonzero = owner_array(&a, "latest", st)?
            .chunks_exact(4)
            .filter(|x| f32::from_le_bytes((*x).try_into().unwrap()) != 0.0)
            .count();
        // Count the actual retained partial snapshot, not the later whole-model state.
        restore_owner(&c, &rc, saved)?;
        native_snapshot_equal(&c, &rc, saved, "original current fresh complete restore")?;
        let partial_nonzero = owner_array(&c, "incoming", st)?
            .chunks_exact(4)
            .filter(|x| f32::from_le_bytes((*x).try_into().unwrap()) != 0.0)
            .count();
        if partial_nonzero == 0 {
            return Err("original current restore has no nonzero incoming carry".into());
        }
        for end in p.restore_at + 1..=t {
            restore_owner(&b, &rb, saved)?;
            native_snapshot_equal(&b, &rb, saved, "original current bounded restore all blobs")?;
            let mut data = kda_raw_slice(&raw, p.restore_at, end - p.restore_at, h, d)?;
            if end - p.restore_at > bounded {
                return Err("original current admitted suffix exceeds bounded layout".into());
            }
            for (v, (desc, _)) in data.iter_mut().zip(chunks.raw.inventory()) {
                while v.len() < desc.storage_bytes {
                    v.extend_from_slice(&0x7fc1u16.to_le_bytes());
                }
            }
            kda_raw_upload(&chunks.raw, &data)?;
            original_batch(&b, &chunks, 0, p.restore_at, end - p.restore_at)?;
            let result = original_read_compare(
                root,
                &format!("restore{end}"),
                &b,
                &chunks,
                p.restore_at,
                end - p.restore_at,
                &want_f,
                &want_b,
                &want_p,
            )?;
            original_snapshot_save(root, &format!("restore{end}"), &capture_owner(&b, &rb)?)?;
            native_snapshot_equal(
                &b,
                &rb,
                snapshots
                    .get(&end)
                    .ok_or("original current suffix endpoint")?,
                "original current restored all terminal blobs",
            )?;
            // The same restored partial state with a whole suffix GPU batch.
            restore_owner(&c, &rc, saved)?;
            original_batch(&c, &full, p.restore_at, p.restore_at, end - p.restore_at)?;
            original_read_compare(
                root,
                &format!("shared-restore{end}"),
                &c,
                &full,
                p.restore_at,
                end - p.restore_at,
                &want_f,
                &want_b,
                &want_p,
            )?;
            native_snapshot_equal(
                &c,
                &rc,
                snapshots.get(&end).unwrap(),
                "original current shared restored all terminal blobs",
            )?;
            restores.push(result);
        }
        owner_save(
            root,
            "NONZERO.json",
            &serde_json::to_vec(
                &json!({"partial_incoming_nonzero":partial_nonzero,"whole_latest_nonzero":nonzero}),
            )
            .map_err(|e| e.to_string())?,
        )?;
        Ok::<(), String>(())
    })();
    stream.drain()?;
    let minimum = window.finish()?;
    let report = json!({"passed":work.is_ok()&&minimum>=reserve as u64,"failure":work.as_ref().err(),"actual_uuid":uuid,"compute":cc,"rows":t,"prefixes":endpoints,"partitions":partitions,"restores":restores,"retained":retained,"owner_arithmetic_sha256":a.arithmetic_sha256,"producer_arithmetic_sha256":full.arithmetic_sha256,"memory_before":before,"memory_after_allocation":allocated,"memory_final":memory()?,"precision_min_free_bytes":minimum,"reserve_bytes":reserve,"full_arena_bytes":a.arena.requested_bytes,"bounded_arena_bytes":b.arena.requested_bytes,"persistent_bytes_each":a.persistent_bytes,"producer_full_bytes":full.requested_bytes,"producer_bounded_bytes":chunks.requested_bytes,"A_inventory":ra.inventory(t),"B_inventory":rb.inventory(t),"C_inventory":rc.inventory(t),"adopted":false,"independent_references":"REQUIRED_FOR_CURRENT_PREPARED_OPERANDS","ordinary_gate":"NOT_RUN","performance":"NOT_MEASURED","native_windows":"OUTSTANDING"});
    owner_save(
        root,
        "REPORT.json",
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    work?;
    if minimum < reserve as u64 {
        return Err("original current measured precision margin violated".into());
    }
    Ok(())
}
#[test]
#[ignore = "Requires frozen original long inputs and explicit GPU admission"]
fn actual_native_original_long_current_owner() {
    run_original_long_current_owner().unwrap();
}
