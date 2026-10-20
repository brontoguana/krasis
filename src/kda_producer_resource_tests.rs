// Bounded native resource controls. Existing arithmetic is executed unchanged.
fn resource_output(owner: &PreparedOwner) -> Result<Vec<u8>, String> {
    owner.stream.drain()?;
    let elements = owner
        .heads
        .checked_mul(owner.dim)
        .ok_or("resource output geometry")?;
    let mut bytes = vec![0u8; elements.checked_mul(4).ok_or("resource output extent")?];
    check(
        unsafe {
            sys::lib().cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), owner.p("emitted"), bytes.len())
        },
        "resource complete output read",
    )?;
    Ok(bytes)
}
fn resource_snapshot(owner: &PreparedOwner) -> Result<OwnerSnapshot, String> {
    let mut registry = crate::session_cache::SequenceStateRegistry::default();
    owner.register(&mut registry, 0)?;
    capture_owner(owner, &registry)
}
fn resource_copy_input(
    stream: &NativeStream,
    input: &NativeRawOwner,
    raw: &NativeRawOwner,
) -> Result<(), String> {
    raw.schema.admits(&input.schema)?;
    for ((plane, to), (_, from)) in raw.inventory().into_iter().zip(input.inventory()) {
        timeline_copy(stream, to, from, plane.storage_bytes)?;
    }
    Ok(())
}
fn resource_enqueue(
    owner: &PreparedOwner,
    producer: &KdaRawProducer,
    leases: &[EnqueueLease<'_>],
) -> Result<(), String> {
    producer.enqueue(0, 1, owner)?;
    owner.prefill_in(model_arena_lease(owner, leases)?, &producer.view(1)?, 0, 1)
}
fn actual_native_producer_resource_kda_run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev producer-domain-native".into());
    }
    let path = std::env::var("KRASIS_KDA_RESOURCE_PLAN").map_err(|_| "frozen resource plan")?;
    let out = std::env::var("KRASIS_KDA_RESOURCE_OUTPUT").map_err(|_| "fresh resource output")?;
    let qp: TilingPlan = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let p = &qp.native;
    if !p.diagnostic_only || p.qualifies_candidate || p.rows < 2 {
        return Err("resource fixture scope".into());
    }
    for x in &p.admission {
        owner_checked(x)?;
    }
    owner_checked(&p.driver)?;
    owner_checked(&p.executable)?;
    if fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        != fs::canonicalize(&p.executable.path).map_err(|e| e.to_string())?
    {
        return Err("resource actual executable".into());
    }
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| e.to_string())?
        .trim()
        != p.boot_id
    {
        return Err("resource boot identity".into());
    }
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("resource safety config")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("resource reserve overflow")?;
    if reserve != 600 * 1024 * 1024 {
        return Err("resource unchanged600MiB required".into());
    }
    let all = p
        .raw_inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    let input0 = kda_raw_slice(&all, 0, 1, p.heads, p.dim)?;
    let input1 = kda_raw_slice(&all, 1, 1, p.heads, p.dim)?;
    if input0 == input1 {
        return Err("real input discriminator identical".into());
    }
    let expected = owner_checked(&qp.output_reference)?;
    let width = p
        .heads
        .checked_mul(p.dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or("resource reference width")?;
    if expected.len()
        != p.rows
            .checked_mul(width)
            .ok_or("resource reference extent")?
    {
        return Err("resource complete reference extent".into());
    }
    let transport =
        String::from_utf8(owner_checked(&p.transport_ptx)?).map_err(|e| e.to_string())?;
    let math = String::from_utf8(owner_checked(&p.math_ptx)?).map_err(|e| e.to_string())?;
    let producer_ptx =
        String::from_utf8(owner_checked(&p.producer_ptx)?).map_err(|e| e.to_string())?;
    let a_log = owner_checked(&p.a_log)?;
    let bias = owner_checked(&p.bias)?;
    owner_checked(&qp.cuda_runtime)?;
    let _runtime =
        unsafe { libloading::Library::new(&qp.cuda_runtime.path) }.map_err(|e| e.to_string())?;
    let device = CudaDevice::new(0).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let mut uuid = std::mem::MaybeUninit::<sys::CUuuid>::uninit();
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(uuid.as_mut_ptr(), device.ordinal() as i32) },
        "resource actual uuid",
    )?;
    let u = unsafe { uuid.assume_init() }
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
    let compute = [
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?,
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?,
    ];
    if actual_uuid != p.uuid || compute != p.compute {
        return Err("resource actual architecture".into());
    }
    let _monitor = crate::vram_monitor::native_calibration_monitor(
        vec![device.ordinal() as i32],
        (reserve / 1048576) as u64,
    )?;
    let precision = crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?
        .ok_or("resource precision window disabled")?;
    let before = memory()?;
    let root = Path::new(&out);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let sa = NativeStream::create(device.clone())?;
    let sb = NativeStream::create(device.clone())?;
    let foreign = NativeStream::create(device.clone())?;
    let schema = Schema::checked(RawContract::KdaBf16VectorGate, 1, p.heads, p.dim, p.dim)?;
    let packed = Arc::new(native_zeroed::<u16>(&sa, schema.requested_bytes / 2)?);
    let allocation = NativeBf16Allocation::GpuBuf(packed.clone());
    let ra = Arc::new(packed_raw_views(
        sa.clone(),
        schema.clone(),
        &allocation,
        reserve,
    )?);
    let rb = Arc::new(packed_raw_views(
        sb.clone(),
        schema.clone(),
        &allocation,
        reserve,
    )?);
    let i0 = Arc::new(NativeRawOwner::allocate(
        sa.clone(),
        schema.clone(),
        reserve,
    )?);
    let i1 = Arc::new(NativeRawOwner::allocate(sb.clone(), schema, reserve)?);
    kda_raw_upload(&i0, &input0)?;
    kda_raw_upload(&i1, &input1)?;
    let kernels = KdaRawKernels::load(device.clone(), &producer_ptx)?;
    let producer = |raw: Arc<NativeRawOwner>| {
        KdaRawProducer::from_live_raw_owner(
            raw,
            p.heads,
            p.dim,
            1,
            reserve,
            p.lower,
            p.scale,
            p.epsilon,
            kernels.clone(),
            &a_log,
            &bias,
        )
    };
    let pa = producer(ra.clone())?;
    let pb = producer(rb.clone())?;
    let pc = producer(i0.clone())?;
    let pd = producer(i1.clone())?;
    let make = |stream: Arc<NativeStream>| {
        PreparedOwner::new_on_stream(
            device.clone(),
            stream,
            p.heads,
            p.dim,
            1,
            reserve,
            &transport,
            &math,
        )
    };
    let oa = make(sa.clone())?;
    let ob = make(sb.clone())?;
    let oc = make(sa.clone())?;
    let od = make(sb.clone())?;
    oa.initialize_timeline(&[sa.clone(), sb.clone()])?;
    ob.initialize_timeline(&[sa.clone(), sb.clone()])?;
    owned_producer_domain_lifetime_controls(root, &pa, reserve)?;
    resource_epoch_handoff_controls(root, &pc, reserve)?;
    retired_resource_capability_controls(root, &pc, reserve)?;
    let domain = ProducerResourceDomain::checked(&[&pa, &pb])?;
    if domain.groups.len() != 1 || ProducerResourceDomain::checked(&[&pa]).is_ok() {
        return Err("duplicate producer domain/group not rejected".into());
    }
    let independent = ProducerResourceDomain::checked(&[&pc, &pd])?;
    if independent.groups.len() != 2 {
        return Err("disjoint producer domains merged".into());
    }
    if ProducerResourceDomain::checked(&[]).is_ok() {
        return Err("empty producer inventory admitted".into());
    }
    let ba = [ModelFamilyBinding {
        layer: 0,
        owner: &oa,
        producer: &pa,
    }];
    let bb = [ModelFamilyBinding {
        layer: 1,
        owner: &ob,
        producer: &pb,
    }];
    let both = [
        ModelFamilyBinding {
            layer: 0,
            owner: &oa,
            producer: &pa,
        },
        ModelFamilyBinding {
            layer: 1,
            owner: &ob,
            producer: &pb,
        },
    ];
    let ida = model_family_identity(&ba)?;
    let idb = model_family_identity(&bb)?;
    let ids = model_family_identity(&both)?;
    let proofa = ProducerResourceProof::checked(&domain, &ba, &ida)?;
    let proofb = ProducerResourceProof::checked(&domain, &bb, &idb)?;
    let proofall = ProducerResourceProof::checked(&domain, &both, &ids)?;
    if ProducerResourceProof::checked(&independent, &ba, &ida).is_ok()
        || ProducerResourceProof::checked(&domain, &ba, &json!([])).is_ok()
        || ProducerResourceAdmission::checked(&proofa, &foreign).is_ok()
    {
        return Err("missing/stale/unadmitted resource accepted".into());
    }
    // Partial acquisition must release resource guards when family acquisition fails.
    {
        let storage = ModelBindingProof::checked(&ba, &ida)?;
        let held = ModelArenaAdmission::from_proof(&storage, &sa)?;
        if ProducerResourceAdmission::checked(&proofall, &sa).is_ok() {
            return Err("busy family admitted".into());
        }
        if domain.groups[0]
            .timeline
            .try_lock()
            .map_err(|_| "partial resource guard leaked")?
            .phases
            != 0
        {
            return Err("partial rejection changed timeline".into());
        }
        drop(held);
    }
    // Independent same-operand GPU references, retaining every state byte.
    for (owner, prod) in [(&oc, &pc), (&od, &pd)] {
        let lease = owner.arena.lease()?;
        prod.enqueue(0, 1, owner)?;
        owner.prefill_in(&lease, &prod.view(1)?, 0, 1)?;
        owner.stream.drain()?;
    }
    let c = resource_output(&oc)?;
    let d = resource_output(&od)?;
    owner_compare(&c, &expected[..width], "resource row0 unchanged266 output")?;
    owner_save(root, "reference-row0-output.f32", &c)?;
    owner_save(root, "reference-row1-output.f32", &d)?;
    let cs = resource_snapshot(&oc)?;
    let ds = resource_snapshot(&od)?;
    owner_save(
        root,
        "reference-row0-state.json",
        &serde_json::to_vec(&cs).map_err(|e| e.to_string())?,
    )?;
    owner_save(
        root,
        "reference-row1-state.json",
        &serde_json::to_vec(&ds).map_err(|e| e.to_string())?,
    )?;
    // Constructed unprotected interleaving: complete raw row0 then row1 overwrite.
    resource_copy_input(&sa, &i0, &ra)?;
    sa.drain()?;
    resource_copy_input(&sb, &i1, &rb)?;
    sb.drain()?;
    let hazard = make(sa.clone())?;
    {
        let lease = hazard.arena.lease()?;
        pa.enqueue(0, 1, &hazard)?;
        hazard.prefill_in(&lease, &pa.view(1)?, 0, 1)?;
    }
    let bad = resource_output(&hazard)?;
    let hs = resource_snapshot(&hazard)?;
    owner_save(root, "unprotected-row0-output.f32", &bad)?;
    owner_save(
        root,
        "unprotected-row0-state.json",
        &serde_json::to_vec(&hs).map_err(|e| e.to_string())?,
    )?;
    if bad == c || bad != d {
        return Err("controlled source overwrite did not reproduce exact row1 result".into());
    }
    let mut callbacks = 0usize;
    {
        let mut held = ProducerResourceAdmission::checked(&proofa, &sa)?;
        if ProducerResourceAdmission::checked(&proofb, &sb).is_ok() {
            return Err("independent family proof bypassed shared source lease".into());
        }
        let phases = held
            .resources
            .iter()
            .map(|x| (x.phases, x.recorded, x.poisoned))
            .collect::<Vec<_>>();
        held.capture_only(|_| Ok(()))?;
        if held
            .resources
            .iter()
            .map(|x| (x.phases, x.recorded, x.poisoned))
            .collect::<Vec<_>>()
            != phases
        {
            return Err("capture host timeline mutation".into());
        }
        held.execute(|leases| {
            callbacks += 1;
            resource_copy_input(&sa, &i0, &ra)?;
            resource_enqueue(&oa, &pa, leases)
        })?;
    }
    {
        let mut held = ProducerResourceAdmission::checked(&proofb, &sb)?;
        held.execute(|leases| {
            callbacks += 1;
            resource_copy_input(&sb, &i1, &rb)?;
            resource_enqueue(&ob, &pb, leases)
        })?;
    }
    let gooda = resource_output(&oa)?;
    let goodb = resource_output(&ob)?;
    let asnap = resource_snapshot(&oa)?;
    let bsnap = resource_snapshot(&ob)?;
    owner_save(root, "protected-row0-output.f32", &gooda)?;
    owner_save(root, "protected-row1-output.f32", &goodb)?;
    owner_save(
        root,
        "protected-row0-state.json",
        &serde_json::to_vec(&asnap).map_err(|e| e.to_string())?,
    )?;
    owner_save(
        root,
        "protected-row1-state.json",
        &serde_json::to_vec(&bsnap).map_err(|e| e.to_string())?,
    )?;
    owner_compare(&gooda, &c, "resource protected row0")?;
    owner_compare(&goodb, &d, "resource protected row1")?;
    owner_compare(
        &serde_json::to_vec(&asnap.blobs).map_err(|e| e.to_string())?,
        &serde_json::to_vec(&cs.blobs).map_err(|e| e.to_string())?,
        "complete resource row0 state",
    )?;
    owner_compare(
        &serde_json::to_vec(&bsnap.blobs).map_err(|e| e.to_string())?,
        &serde_json::to_vec(&ds.blobs).map_err(|e| e.to_string())?,
        "complete resource row1 state",
    )?;
    if callbacks != 2 {
        return Err("resource callback count".into());
    }
    drop(independent); // Independent reference-domain lifetime ended before reuse of its sources.
    let shared_factory = shared_factory_controls(
        root,
        &device,
        &pa,
        &i0,
        &i1,
        &sa,
        &sb,
        reserve,
        &transport,
        &math,
        &c,
        &cs,
        &expected[width..width * 2],
    )?;
    let workspace_capacity = workspace_capacity_controls(
        root, &pa, &i0, &i1, reserve, &transport, &math, &qp, &expected, &c, &cs,
    )?;
    let group_workspace = grouped_workspace_controls(
        root, &pa, &i0, &i1, reserve, &transport, &math, &qp, &expected, &c, &cs,
    )?;
    let extended = resource_extended_controls(
        root, &device, &domain, &pa, &pb, &i0, &i1, &sa, &sb, reserve, &transport, &math, &c, &d,
        &cs, &ds,
    )?;
    let phases = domain.groups[0]
        .timeline
        .lock()
        .map_err(|_| "resource timeline poisoned mutex")?
        .phases;
    if phases != 4 {
        return Err("resource completion phases".into());
    }
    {
        let mut held = ProducerResourceAdmission::checked(&proofall, &sa)?;
        let failure: Result<(), String> =
            held.execute(|_| Err("explicit resource callback failure".into()));
        if !matches!(&failure, Err(error) if error == "explicit resource callback failure") {
            return Err("resource failure contract".into());
        }
        if held.validate_ready().is_ok() {
            return Err("held resource poison ignored".into());
        }
    }
    if ProducerResourceAdmission::checked(&proofa, &sa).is_ok()
        || ProducerResourceAdmission::checked(&proofb, &sb).is_ok()
    {
        return Err("failed shared resource admitted".into());
    }
    for o in [&oa, &ob] {
        if !o
            .state
            .timeline
            .lock()
            .map_err(|_| "family poison mutex")?
            .as_ref()
            .ok_or("family missing")?
            .poisoned
        {
            return Err("resource failure did not poison every family".into());
        }
    }
    sa.drain()?;
    sb.drain()?;
    let resident = memory()?;
    let minimum = precision.finish()?;
    if minimum < reserve as u64 {
        return Err("resource precision safety breach".into());
    }
    owner_save(root,"REPORT.json",&serde_json::to_vec_pretty(&json!({"passed":true,"uuid":actual_uuid,"compute":compute,"heads":p.heads,"dim":p.dim,"capacity":1,"memory_before":before,"memory_resident":resident,"precision_min_free_bytes":minimum,"reserve_bytes":reserve,"protected_callbacks":callbacks,"completion_phases_before_fault":phases,"all_family_resource_poison":true,"duplicate_domain_rejected":true,"missing_stale_unadmitted_rejected":true,"partial_rejection_released_guards":true,"complete_output_and_state_equal":true,"unprotected_exactly_row1":true,"host_capture_timeline_unchanged":true,"extended":extended,"workspace_capacity":workspace_capacity,"group_workspace":group_workspace,"shared_factory":shared_factory,"actual_graph_capture_replay":"PASS","partial_overlap_native":"PASS","cross_thread_domain":"PASS","scope":"bounded identical-source two-stream singleton producer exclusion/order; constructed hazard only, not model attribution","adopted":false})).map_err(|e|e.to_string())?)?;
    Ok(())
}
#[test]
#[ignore]
fn actual_native_producer_resource_kda() {
    if let Err(e) = actual_native_producer_resource_kda_run() {
        panic!("{e}");
    }
}
