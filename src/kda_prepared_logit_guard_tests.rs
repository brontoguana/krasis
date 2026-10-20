fn actual_native_prepared_logit_guard_run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev prepared-logit-guard-native".into());
    }
    let path = std::env::var("KRASIS_KDA_RESOURCE_PLAN").map_err(|_| "frozen guard plan")?;
    let out = std::env::var("KRASIS_KDA_RESOURCE_OUTPUT").map_err(|_| "fresh guard output")?;
    let qp: TilingPlan = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let p = &qp.native;
    if !p.diagnostic_only || p.qualifies_candidate {
        return Err("guard diagnostic scope".into());
    }
    for x in p.admission.iter().chain([
        &p.driver,
        &p.executable,
        &qp.cuda_runtime,
        &p.runtime_config,
        &p.transport_ptx,
        &p.math_ptx,
    ]) {
        owner_checked(x)?;
    }
    if fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        != fs::canonicalize(&p.executable.path).map_err(|e| e.to_string())?
    {
        return Err("guard actual executable".into());
    }
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| e.to_string())?
        .trim()
        != p.boot_id
    {
        return Err("guard boot identity".into());
    }
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("guard config safety")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1048576)
        .ok_or("guard reserve extent")?;
    if reserve != 600 * 1048576 {
        return Err("guard unchanged margin".into());
    }
    let _runtime =
        unsafe { libloading::Library::new(&qp.cuda_runtime.path) }.map_err(|e| e.to_string())?;
    let device = CudaDevice::new(0).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let mut uuid = std::mem::MaybeUninit::<sys::CUuuid>::uninit();
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(uuid.as_mut_ptr(), device.ordinal() as i32) },
        "guard actualuuid",
    )?;
    let hex = unsafe { uuid.assume_init() }
        .bytes
        .iter()
        .map(|v| format!("{:02x}", *v as u8))
        .collect::<String>();
    let actual_uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    let compute = [
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?,
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?,
    ];
    if actual_uuid != p.uuid || compute != p.compute {
        return Err("guard actual architecture".into());
    }
    let _monitor = crate::vram_monitor::native_calibration_monitor(
        vec![device.ordinal() as i32],
        (reserve / 1048576) as u64,
    )?;
    let precision = crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?
        .ok_or("guard precision disabled")?;
    let before = memory()?;
    let root = Path::new(&out);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let transport =
        String::from_utf8(owner_checked(&p.transport_ptx)?).map_err(|e| e.to_string())?;
    let math = String::from_utf8(owner_checked(&p.math_ptx)?).map_err(|e| e.to_string())?;
    let stream = NativeStream::create(device.clone())?;
    let other = NativeStream::create(device.clone())?;
    let foreign = NativeStream::create(device.clone())?;
    let device_a = NativeStream::borrowed_device(device.clone())?;
    let device_b = NativeStream::borrowed_device(device.clone())?;
    let prefill_backing = crate::gpu_prefill::PrefillComputeStream::create(device.clone())?;
    let prefill_a = NativeStream::from_prefill_owner(prefill_backing.clone())?;
    let prefill_b = NativeStream::from_prefill_owner(prefill_backing.clone())?;
    let bare_prefill =
        unsafe { NativeStream::borrow_handle(device.clone(), prefill_backing.handle()) }?;
    let bare_owned = unsafe { NativeStream::borrow_handle(device.clone(), stream.handle) }?;
    let checks = [
        (
            "same_arc_owned",
            NativeStream::same_backing_owner(&stream, &stream)?,
            true,
        ),
        (
            "distinct_owned",
            NativeStream::same_backing_owner(&stream, &other)?,
            false,
        ),
        (
            "device_stream_wrappers",
            NativeStream::same_backing_owner(&device_a, &device_b)?,
            true,
        ),
        (
            "prefill_owner_wrappers",
            NativeStream::same_backing_owner(&prefill_a, &prefill_b)?,
            true,
        ),
        (
            "prefill_bare_handle",
            NativeStream::same_backing_owner(&prefill_a, &bare_prefill)?,
            false,
        ),
        (
            "owned_bare_handle",
            NativeStream::same_backing_owner(&stream, &bare_owned)?,
            false,
        ),
        (
            "device_prefill",
            NativeStream::same_backing_owner(&device_a, &prefill_a)?,
            false,
        ),
    ];
    for (name, actual, expected) in &checks {
        if actual != expected {
            return Err(format!("backing stream owner admission mismatch:{name}"));
        }
    }
    let backing_owner_admissions = checks
        .iter()
        .map(|(name, actual, expected)| json!({"name":name,"admitted":actual,"expected":expected}))
        .collect::<Vec<_>>();
    drop(bare_owned);
    drop(bare_prefill);
    drop(prefill_a);
    drop(prefill_b);
    drop(prefill_backing);
    drop(device_a);
    drop(device_b);
    let first = PreparedOwner::new_on_stream(
        device.clone(),
        stream.clone(),
        p.heads,
        p.dim,
        1,
        reserve,
        &transport,
        &math,
    )?;
    let second = PreparedOwner::from_shared(&first, reserve)?;
    let stale = PreparedOwner::from_shared(&first, reserve)?;
    if PreparedLogitGuard::new(&[], vec![stream.clone()], reserve).is_ok()
        || PreparedLogitGuard::new(&[&first, &first], vec![stream.clone()], reserve).is_ok()
        || PreparedLogitGuard::new(&[&first], vec![other.clone()], reserve).is_ok()
        || PreparedLogitGuard::new(&[&first], vec![stream.clone(), stream.clone()], reserve).is_ok()
    {
        return Err("guard invalid constructor admission".into());
    }
    let guard = PreparedLogitGuard::new(
        &[&first, &second],
        vec![stream.clone(), other.clone()],
        reserve,
    )?;
    guard.validate_owners(&[&first, &second])?;
    if guard.validate_owners(&[&first, &stale]).is_ok()
        || guard.validate_owners(&[&second, &first]).is_ok()
        || guard.validate_owners(&[&first]).is_ok()
    {
        return Err("guard stale selected owners accepted".into());
    }
    let bits = [
        0x3f800000u32,
        0x80000000,
        0x00000001,
        0x7f800000,
        0xff800000,
        0x7fc01234,
    ];
    let logits = native_zeroed::<f32>(&stream, bits.len())?;
    stream.drain()?;
    if unsafe { guard.launch(&foreign, &logits) }.is_ok() {
        return Err("guard unadmitted stream accepted".into());
    }
    let empty = native_zeroed::<f32>(&stream, 0)?;
    if unsafe { guard.launch(&stream, &empty) }.is_ok() {
        return Err("guard empty logits accepted".into());
    }
    let upload = |stream: &Arc<NativeStream>| -> Result<(), String> {
        stream.drain()?;
        check(
            unsafe {
                sys::lib().cuMemcpyHtoD_v2(pointer(&logits), bits.as_ptr().cast(), bits.len() * 4)
            },
            "guard logits upload",
        )
    };
    let read = |stream: &Arc<NativeStream>| -> Result<Vec<u32>, String> {
        stream.drain()?;
        let mut out = vec![0u32; bits.len()];
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), pointer(&logits), out.len() * 4)
            },
            "guard complete logits read",
        )?;
        Ok(out)
    };
    upload(&stream)?;
    unsafe { guard.launch(&stream, &logits) }?;
    if read(&stream)? != bits {
        return Err("guard changed valid logits".into());
    }
    let weak = [Arc::downgrade(&first.state), Arc::downgrade(&second.state)];
    drop(first);
    drop(second);
    drop(stale);
    if weak.iter().any(|x| x.upgrade().is_none()) {
        return Err("guard lost retained actual state owners".into());
    }
    let graph = unsafe { PreparedGuardGraph::capture(&guard, &stream, &logits) }?;
    if unsafe { graph.replay(&foreign) }.is_ok() {
        return Err("guard unadmitted graph stream accepted".into());
    }
    let mut records = Vec::new();
    for (ordinal, state) in guard.states.iter().enumerate() {
        for error in [0u32, 1, 2, 3] {
            for s in &guard.states {
                check(
                    unsafe { sys::lib().cuMemsetD32Async(pointer(&s.error), 0, 1, other.handle) },
                    "guard error reset",
                )?;
            }
            check(
                unsafe {
                    sys::lib().cuMemsetD32Async(pointer(&state.error), error, 1, other.handle)
                },
                "guard live error write",
            )?;
            upload(&other)?;
            unsafe { graph.replay(&other) }?;
            let observed = read(&other)?;
            let mut expected = bits.to_vec();
            if error != 0 {
                expected[0] = crate::prefix_recurrence::PREFIX_FAILURE_MARKER;
            }
            if observed != expected {
                return Err("guard live captured failure transport mismatch".into());
            }
            owner_save(
                root,
                &format!("owner{ordinal}-error{error}-logits.bin"),
                &observed
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            records.push(json!({"owner":ordinal,"error":error,"observed":observed}));
        }
    }
    drop(graph);
    let pointers = guard.required_pointers();
    let allocation_bytes = guard.allocation_bytes;
    drop(guard);
    if weak.iter().any(|x| x.upgrade().is_some()) {
        return Err("guard retained owner leak after graph teardown".into());
    }
    let minimum = precision.finish()?;
    if minimum < reserve as u64 {
        return Err("guard precision floor below margin".into());
    }
    owner_save(root,"REPORT.json",&serde_json::to_vec_pretty(&json!({"passed":true,"stream_backing_owner_admissions":backing_owner_admissions,"uuid":actual_uuid,"compute":compute,"heads":p.heads,"dim":p.dim,"reserve_bytes":reserve,"memory_before":before,"precision_min_free_bytes":minimum,"error_table_bytes":allocation_bytes,"retained_required_pointers":pointers,"complete_logits":records,"valid_bits_preserved":true,"every_owner_error_guarded":true,"actual_graph_capture_replay":true,"retained_lifetime_and_teardown":true,"invalid_admissions_rejected":true,"scope":"GPU failure transport only; caller must perform complete all-owner execution admission", "selected_for_serving":false,"adopted":false})).map_err(|e|e.to_string())?)?;
    Ok(())
}
#[test]
fn actual_native_prepared_logit_guard() {
    if let Err(error) = actual_native_prepared_logit_guard_run() {
        panic!("{error}");
    }
}
