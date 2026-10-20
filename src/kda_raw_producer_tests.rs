#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KdaRawPlan {
    owner: OwnerPlan,
    raw_inputs: Vec<OwnerId>,
    a_log: OwnerId,
    bias: OwnerId,
    producer_ptx: OwnerId,
    lower: f32,
    scale: f32,
    epsilon: f32,
}
fn kda_raw_upload(o: &NativeRawOwner, b: &[Vec<u8>]) -> Result<(), String> {
    if b.len() != o.schema.planes.len() {
        return Err("raw producer upload plane count".into());
    }
    for ((desc, ptr), v) in o.inventory().into_iter().zip(b) {
        if v.len() != desc.storage_bytes {
            return Err("raw producer upload extent".into());
        }
        check(
            unsafe {
                sys::lib().cuMemcpyHtoDAsync_v2(ptr, v.as_ptr().cast(), v.len(), o.stream.handle)
            },
            "raw producer diagnostic upload",
        )?;
    }
    o.stream.drain()
}
fn kda_raw_slice(
    v: &[Vec<u8>],
    first: usize,
    rows: usize,
    h: usize,
    d: usize,
) -> Result<Vec<Vec<u8>>, String> {
    if v.len() != 5 {
        return Err("raw producer slice planes".into());
    }
    v.iter()
        .zip([h * d * 2, h * d * 2, h * d * 2, h * d * 2, h * 2])
        .map(|(v, stride)| {
            let start = first.checked_mul(stride).ok_or("raw first overflow")?;
            let end = first
                .checked_add(rows)
                .and_then(|x| x.checked_mul(stride))
                .ok_or("raw end overflow")?;
            Ok(v.get(start..end).ok_or("raw slice bounds")?.to_vec())
        })
        .collect()
}
fn kda_prepared_read(p: &KdaRawProducer, rows: usize) -> Result<Vec<Vec<u8>>, String> {
    p.prepared.stream.drain()?;
    p.prepared
        .inventory()
        .into_iter()
        .map(|(desc, ptr)| {
            let n = rows
                .checked_mul(desc.strides_bytes[0])
                .ok_or("prepared read bytes")?;
            if n > desc.storage_bytes {
                return Err("prepared read bounds".into());
            }
            let mut v = vec![0u8; n];
            check(
                unsafe { sys::lib().cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), ptr, n) },
                "raw producer prepared diagnostic read",
            )?;
            Ok(v)
        })
        .collect()
}
fn kda_raw_graph_inventory(
    o: &PreparedOwner,
    p: &KdaRawProducer,
    g: &OwnerGraph<'_>,
) -> Result<Value, String> {
    let mut n = 0;
    check(
        unsafe { sys::lib().cuGraphGetNodes(g.graph, std::ptr::null_mut(), &mut n) },
        "raw graph count",
    )?;
    let mut nodes = vec![std::ptr::null_mut(); n];
    check(
        unsafe { sys::lib().cuGraphGetNodes(g.graph, nodes.as_mut_ptr(), &mut n) },
        "raw graph nodes",
    )?;
    if n != 17 || n != nodes.len() {
        return Err("raw graph expected seventeen kernels".into());
    }
    let mut inventory = Vec::new();
    for &node in &nodes {
        let mut a: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        check(
            unsafe { sys::lib().cuGraphKernelNodeGetParams_v2(node, &mut a) },
            "raw graph kernel params",
        )?;
        let name = if a.func == p.kernels.domain.cu_function() {
            "kda_raw_domain"
        } else if a.func == p.kernels.prepare.cu_function() {
            "kda_raw_prepare"
        } else {
            o.arena
                .functions
                .iter()
                .find_map(|(n, f)| (f.cu_function() == a.func).then_some(*n))
                .ok_or("raw graph unknown kernel")?
        };
        inventory.push(json!({"node":node as usize,"name":name,"function":a.func as usize,"grid":[a.gridDimX,a.gridDimY,a.gridDimZ],"block":[a.blockDimX,a.blockDimY,a.blockDimZ],"shared_bytes":a.sharedMemBytes}));
    }
    let mut e = 0;
    check(
        unsafe {
            sys::lib().cuGraphGetEdges(g.graph, std::ptr::null_mut(), std::ptr::null_mut(), &mut e)
        },
        "raw graph edge count",
    )?;
    let (mut from, mut to) = (vec![std::ptr::null_mut(); e], vec![std::ptr::null_mut(); e]);
    check(
        unsafe { sys::lib().cuGraphGetEdges(g.graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut e) },
        "raw graph edges",
    )?;
    if e != 16 || e != from.len() || e != to.len() {
        return Err("raw graph chain edges".into());
    }
    for (&f, &t) in from.iter().zip(&to) {
        if !nodes.contains(&f) || !nodes.contains(&t) {
            return Err("raw graph foreign edge".into());
        }
    }
    Ok(
        json!({"nodes":inventory,"edges":from.iter().zip(&to).map(|(f,t)|json!([*f as usize,*t as usize])).collect::<Vec<_>>(),"complete":true}),
    )
}
fn kda_raw_owner_check(
    o: &PreparedOwner,
    initial: &[u8],
    states: &[u8],
    len: usize,
) -> Result<(), String> {
    o.stream.drain()?;
    let st = o.heads * o.dim * o.dim;
    owner_compare(
        &owner_array(o, "latest", st)?,
        &states[(len - 1) * st * 4..len * st * 4],
        "raw producer latest state",
    )?;
    let at = len / 16 * 16;
    let carry = if at == 0 {
        initial
    } else {
        &states[(at - 1) * st * 4..at * st * 4]
    };
    for role in ["incoming", "complete"] {
        owner_compare(
            &owner_array(o, role, st)?,
            carry,
            "raw producer committed carry",
        )?;
    }
    if owner_metadata(o)? != ((len % 16) as i32, 0, len as u64) {
        return Err("raw producer consumed cursor".into());
    }
    Ok(())
}
fn run_native_raw_vector() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use built ./dev".into());
    }
    let input = std::env::var("KRASIS_KDA_RAW_PLAN").map_err(|_| "raw frozen plan")?;
    let output = std::env::var("KRASIS_KDA_RAW_OUTPUT").map_err(|_| "raw fresh output")?;
    let p: KdaRawPlan = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let op = &p.owner;
    if !op.diagnostic_only
        || op.qualifies_candidate
        || op.inputs.len() != 5
        || p.raw_inputs.len() != 5
        || op.rows < 17
    {
        return Err("raw diagnostic scope".into());
    }
    for x in &op.admission {
        owner_checked(x)?;
    }
    owner_checked(&op.driver)?;
    owner_checked(&op.executable)?;
    if fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        != fs::canonicalize(&op.executable.path).map_err(|e| e.to_string())?
    {
        return Err("raw actual executable identity".into());
    }
    let cfg = String::from_utf8(owner_checked(&op.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("raw configured reserve")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("raw reserve overflow")?;
    let raw = p
        .raw_inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    let expected = op
        .inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    let initial = owner_checked(
        op.references
            .get("initial")
            .ok_or("raw initial reference")?,
    )?;
    let out = owner_checked(op.references.get("output").ok_or("raw output reference")?)?;
    let states = owner_checked(op.references.get("states").ok_or("raw state reference")?)?;
    let (h, d, t) = (op.heads, op.dim, op.rows);
    let hd = h.checked_mul(d).ok_or("raw HD extent")?;
    let st = hd.checked_mul(d).ok_or("raw state extent")?;
    if initial.len() != st * 4 || out.len() != t * hd * 4 || states.len() != t * st * 4 {
        return Err("raw complete reference extent".into());
    }
    let root = Path::new(&output);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let device = CudaDevice::new_with_stream(op.cuda_index).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let (mut ordinal, mut uuid) = (0, unsafe { std::mem::zeroed::<sys::CUuuid>() });
    check(
        unsafe { sys::lib().cuCtxGetDevice(&mut ordinal) },
        "raw context ordinal",
    )?;
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, ordinal) },
        "raw actual UUID",
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
    if actual_uuid != op.uuid
        || device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?
            != op.compute[0]
        || device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?
            != op.compute[1]
    {
        return Err("raw actual architecture".into());
    }
    let stream = NativeStream::create(device.clone())?;
    let before = memory()?;
    let transport =
        String::from_utf8(owner_checked(&op.transport_ptx)?).map_err(|e| e.to_string())?;
    let math = String::from_utf8(owner_checked(&op.math_ptx)?).map_err(|e| e.to_string())?;
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
    let b = PreparedOwner::from_shared(&a, reserve)?;
    let mut ra = crate::session_cache::SequenceStateRegistry::default();
    let mut rb = crate::session_cache::SequenceStateRegistry::default();
    a.register(&mut ra, 0)?;
    b.register(&mut rb, 0)?;
    let ptx = String::from_utf8(owner_checked(&p.producer_ptx)?).map_err(|e| e.to_string())?;
    let a_log = owner_checked(&p.a_log)?;
    let bias = owner_checked(&p.bias)?;
    let kernels = KdaRawKernels::load(device.clone(), &ptx)?;
    let producer = KdaRawProducer::new(
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
    let singleton = KdaRawProducer::new(
        stream.clone(),
        h,
        d,
        1,
        reserve,
        p.lower,
        p.scale,
        p.epsilon,
        kernels.clone(),
        &a_log,
        &bias,
    )?;
    kda_raw_upload(&producer.raw, &raw)?;
    let graph = kda_raw_capture(&a, &singleton)?;
    let graph_inventory = kda_raw_graph_inventory(&a, &singleton, &graph)?;
    let after = memory()?;
    if after[0] < reserve {
        return Err("raw observed reserve floor".into());
    }
    let mut retained = Vec::new();
    let mut partitions = Vec::new();
    for len in 1..=t {
        vector_seed(&a, &initial)?;
        {
            let lease = a.arena.lease()?;
            producer.enqueue(0, len, &a)?;
            a.prefill_in(&lease, &producer.view(len)?, 0, len)?;
        }
        a.stream.drain()?;
        let prepared = kda_prepared_read(&producer, len)?;
        let want = raw_slice(&expected, 0, len, h, d);
        for i in 0..5 {
            owner_compare(&prepared[i], &want[i], "raw actual prepared prefix")?;
            retained.push(owner_save(
                root,
                &format!("prefix{len}-prepared{i}.f32"),
                &prepared[i],
            )?);
        }
        let emitted = owner_array(&a, "emitted", len * hd)?;
        owner_compare(&emitted, &out[..len * hd * 4], "raw actual prefix output")?;
        kda_raw_owner_check(&a, &initial, &states, len)?;
        let bf16 = vector_bf16_array(&a, len * hd)?;
        owner_compare(&bf16, &vector_rne_view(&emitted)?, "raw actual prefix BF16")?;
        for (name, v) in [
            ("output.f32", emitted),
            ("output.bf16", bf16),
            ("latest.f32", owner_array(&a, "latest", st)?),
            ("committed.f32", owner_array(&a, "incoming", st)?),
        ] {
            retained.push(owner_save(root, &format!("prefix{len}-{name}"), &v)?);
        }
        let snapshot = capture_owner(&a, &ra)?;
        retained.push(owner_save(
            root,
            &format!("prefix{len}-snapshot.json"),
            &serde_json::to_vec(&snapshot).map_err(|e| e.to_string())?,
        )?);
        if len < t {
            vector_seed(&b, &initial)?;
            restore_owner(&b, &rb, &snapshot)?;
            native_snapshot_equal(&b, &rb, &snapshot, "raw fresh owner restore")?;
            {
                let lease = b.arena.lease()?;
                producer.enqueue(len, t - len, &b)?;
                b.prefill_in(&lease, &producer.view(t - len)?, len as u64, t - len)?;
            }
            b.stream.drain()?;
            let prepared = kda_prepared_read(&producer, t - len)?;
            let want = raw_slice(&expected, len, t - len, h, d);
            for i in 0..5 {
                owner_compare(&prepared[i], &want[i], "raw actual prepared suffix")?;
                retained.push(owner_save(
                    root,
                    &format!("split{len}-prepared{i}.f32"),
                    &prepared[i],
                )?);
            }
            let emitted = owner_array(&b, "emitted", (t - len) * hd)?;
            owner_compare(&emitted, &out[len * hd * 4..], "raw restored suffix output")?;
            kda_raw_owner_check(&b, &initial, &states, t)?;
            retained.push(owner_save(
                root,
                &format!("split{len}-output.f32"),
                &emitted,
            )?);
            retained.push(owner_save(
                root,
                &format!("split{len}-snapshot.json"),
                &serde_json::to_vec(&capture_owner(&b, &rb)?).map_err(|e| e.to_string())?,
            )?);
            restore_owner(&a, &ra, &snapshot)?;
            for row in len..t {
                kda_raw_upload(&singleton.raw, &kda_raw_slice(&raw, row, 1, h, d)?)?;
                {
                    let lease = a.arena.lease()?;
                    a.launch_graph_in(&lease, &graph)?;
                }
                a.stream.drain()?;
                let prepared = kda_prepared_read(&singleton, 1)?;
                let want = raw_slice(&expected, row, 1, h, d);
                for i in 0..5 {
                    owner_compare(&prepared[i], &want[i], "raw graph prepared row")?;
                }
                let emitted = owner_array(&a, "emitted", hd)?;
                owner_compare(
                    &emitted,
                    &out[row * hd * 4..(row + 1) * hd * 4],
                    "raw graph continuation output",
                )?;
                owner_compare(
                    &vector_bf16_array(&a, hd)?,
                    &vector_rne_view(&emitted)?,
                    "raw graph BF16 view",
                )?;
                kda_raw_owner_check(&a, &initial, &states, row + 1)?;
            }
            partitions.push(json!({"cut":len,"prepared_suffix_exact":true,"batch_output_state_exact":true,"all_graph_tail_rows_exact":t-len}));
        }
        println!("raw producer complete prefix PASS {len}");
    }
    vector_seed(&a, &initial)?;
    for row in 0..t {
        kda_raw_upload(&singleton.raw, &kda_raw_slice(&raw, row, 1, h, d)?)?;
        {
            let lease = a.arena.lease()?;
            a.launch_graph_in(&lease, &graph)?;
        }
        a.stream.drain()?;
        owner_compare(
            &owner_array(&a, "emitted", hd)?,
            &out[row * hd * 4..(row + 1) * hd * 4],
            "raw all singleton output",
        )?;
        kda_raw_owner_check(&a, &initial, &states, row + 1)?;
    }
    let good = capture_owner(&a, &ra)?;
    let other = capture_owner(&b, &rb)?;
    let output_before = owner_array(&a, "emitted", hd)?;
    let bf16_before = vector_bf16_array(&a, hd)?;
    let mut faults = Vec::new();
    for (slot, label) in [
        (0, "raw-Q-NaN"),
        (3, "raw-gate-infinity"),
        (4, "raw-beta-NaN"),
        (5, "A-log-NaN"),
        (6, "bias-infinity"),
    ] {
        let mut bad = kda_raw_slice(&raw, 0, 1, h, d)?;
        if slot < 5 {
            bad[slot][..2]
                .copy_from_slice(&(if slot == 3 { 0x7f80u16 } else { 0x7fc1u16 }).to_le_bytes());
        }
        kda_raw_upload(&singleton.raw, &bad)?;
        if slot >= 5 {
            let bits = if slot == 5 {
                0x7fc1a123u32
            } else {
                0x7f800000u32
            };
            let bytes = bits.to_le_bytes();
            let destination = if slot == 5 {
                pointer(&singleton.a_log)
            } else {
                pointer(&singleton.bias)
            };
            check(
                unsafe {
                    sys::lib().cuMemcpyHtoDAsync_v2(
                        destination,
                        bytes.as_ptr().cast(),
                        4,
                        a.stream.handle,
                    )
                },
                "raw diagnostic invalid parameter upload",
            )?;
            a.stream.drain()?;
        }
        restore_owner(&a, &ra, &good)?;
        {
            let lease = a.arena.lease()?;
            a.launch_graph_in(&lease, &graph)?;
        }
        a.stream.drain()?;
        if owner_metadata(&a)?.1 != 4 {
            return Err("raw invalid input first-fault".into());
        }
        let mut actual = capture_owner(&a, &ra)?;
        for blob in &mut actual.blobs {
            if blob.allocation_name.ends_with(".error") {
                blob.bytes = good
                    .blobs
                    .iter()
                    .find(|x| x.allocation_name == blob.allocation_name)
                    .ok_or("raw error snapshot")?
                    .bytes
                    .clone();
            }
        }
        owner_compare(
            &serde_json::to_vec(&actual).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&good).map_err(|e| e.to_string())?,
            "raw invalid persistent no-write",
        )?;
        owner_compare(
            &owner_array(&a, "emitted", hd)?,
            &output_before,
            "raw invalid output no-write",
        )?;
        owner_compare(
            &vector_bf16_array(&a, hd)?,
            &bf16_before,
            "raw invalid BF16 no-write",
        )?;
        native_snapshot_equal(&b, &rb, &other, "raw unrelated owner isolation")?;
        retained.push(owner_save(
            root,
            &format!("fault-{label}-snapshot.json"),
            &serde_json::to_vec(&capture_owner(&a, &ra)?).map_err(|e| e.to_string())?,
        )?);
        faults.push(json!({"label":label,"first_fault":4,"persistent_and_outputs_unchanged":true,"other_owner_exact":true}));
        if slot >= 5 {
            let bytes = if slot == 5 { &a_log[..4] } else { &bias[..4] };
            let destination = if slot == 5 {
                pointer(&singleton.a_log)
            } else {
                pointer(&singleton.bias)
            };
            check(
                unsafe {
                    sys::lib().cuMemcpyHtoDAsync_v2(
                        destination,
                        bytes.as_ptr().cast(),
                        4,
                        a.stream.handle,
                    )
                },
                "raw diagnostic restore parameter",
            )?;
            a.stream.drain()?;
        }
    }
    restore_owner(&a, &ra, &good)?;
    for label in ["arithmetic", "missing-blob", "wrong-dtype"] {
        let mut bad: OwnerSnapshot =
            serde_json::from_slice(&serde_json::to_vec(&good).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        match label {
            "arithmetic" => bad.arithmetic_sha256.push('0'),
            "missing-blob" => {
                bad.blobs.pop();
            }
            _ => bad.blobs.last_mut().ok_or("raw restore last blob")?.dtype = "bfloat16".into(),
        };
        if restore_owner(&a, &ra, &bad).is_ok() {
            return Err("raw malformed snapshot accepted".into());
        }
        native_snapshot_equal(&a, &ra, &good, "raw rejected restore full no-write")?;
        faults.push(json!({"label":label,"restore_rejected_before_write":true}));
    }
    retained.push(owner_save(
        root,
        "loaded-maps.txt",
        &fs::read("/proc/self/maps").map_err(|e| e.to_string())?,
    )?);
    let report = json!({"actual_uuid":actual_uuid,"compute":op.compute,"raw_producer_arithmetic":producer.arithmetic_sha256,"owner_arithmetic":a.arithmetic_sha256,"prefixes_exact":t,"partitions":partitions,"all_singletons_exact":true,"faults":faults,"retained":retained,"graph_inventory":graph_inventory,"raw_admissions":producer.raw.admissions,"prepared_admissions":producer.prepared.admissions,"parameter_admissions":producer.parameter_admissions,"singleton_raw_admissions":singleton.raw.admissions,"singleton_prepared_admissions":singleton.prepared.admissions,"singleton_parameter_admissions":singleton.parameter_admissions,"A_inventory":ra.inventory(t),"B_inventory":rb.inventory(t),"owner_arena_bytes":a.arena.requested_bytes,"owner_persistent_bytes_each":a.persistent_bytes,"producer_requested_bytes":producer.requested_bytes,"singleton_requested_bytes":singleton.requested_bytes,"reserve_bytes":reserve,"memory_before":before,"memory_after":after,"adopted":false,"ordinary_gate":"FAILED_UNCHANGED","performance":"NOT_MEASURED","native_windows":"OUTSTANDING","scope":"raw post-convolution fixture producer to actual core vector owner; no live serving qualification"});
    owner_save(
        root,
        "REPORT.json",
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    Ok(())
}
#[test]
fn actual_native_vector_raw_producer() {
    run_native_raw_vector().unwrap();
}
