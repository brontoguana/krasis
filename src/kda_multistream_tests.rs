fn request_worker<T: Send>(
    device: &Arc<CudaDevice>,
    label: &str,
    work: impl FnOnce() -> Result<T, String> + Send,
) -> Result<(T, serde_json::Value), String> {
    std::thread::scope(|scope| {
        scope.spawn(move ||{
            let mut before=std::ptr::null_mut();
            check(unsafe{sys::lib().cuCtxGetCurrent(&mut before)},"fresh worker context query")?;
            if !before.is_null(){return Err("fresh request worker inherited CUDA context".into());}
            device.bind_to_thread().map_err(|e|e.to_string())?;
            let mut current=std::ptr::null_mut();check(unsafe{sys::lib().cuCtxGetCurrent(&mut current)},"bound worker context query")?;
            if current!=*device.cu_primary_ctx(){return Err("request worker primary context mismatch".into());}
            let value=work()?;
            Ok((value,json!({"label":label,"thread_id":format!("{:?}",std::thread::current().id()),"initial_current_context":before as usize,"bound_current_context":current as usize,"expected_primary_context":*device.cu_primary_ctx() as usize,"gpu_wait_before_work":false})))
        }).join().map_err(|_|"request worker panicked".to_string())?
    })
}
fn timeline_copy(stream: &NativeStream, dst: u64, src: u64, bytes: usize) -> Result<(), String> {
    native_pointer_admission(stream, dst, bytes)?;
    native_pointer_admission(stream, src, bytes)?;
    check(
        unsafe { sys::lib().cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle) },
        "timeline diagnostic device copy",
    )
}
struct TimelineSnapshot {
    storage: GpuBuf<u8>,
    template: OwnerSnapshot,
}
impl TimelineSnapshot {
    fn new(stream: &NativeStream, want: &OwnerSnapshot) -> Result<Self, String> {
        let total = want.blobs.iter().try_fold(0usize, |n, x| {
            n.checked_add(x.bytes.len()).ok_or("snapshot byte overflow")
        })?;
        let storage = native_zeroed::<u8>(stream, total)?;
        let template =
            serde_json::from_slice(&serde_json::to_vec(want).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        Ok(Self { storage, template })
    }
    fn enqueue(
        &self,
        phase: &StatePhase<'_>,
        registry: &crate::session_cache::SequenceStateRegistry,
    ) -> Result<(), String> {
        if registry.compatibility_layout_value() != self.template.layout
            || registry.allocations().len() != self.template.blobs.len()
        {
            return Err("timeline snapshot full layout".into());
        }
        let mut offset = 0;
        for (a, b) in registry.allocations().iter().zip(&self.template.blobs) {
            if !crate::session_cache::snapshot_blob_can_restore(a, a.storage_bytes, b) {
                return Err("timeline snapshot full blob schema".into());
            }
            timeline_copy(
                phase.stream,
                pointer(&self.storage) + offset as u64,
                a.ptr,
                a.storage_bytes,
            )?;
            offset += a.storage_bytes;
        }
        Ok(())
    }
    fn read(&self, arithmetic: &str) -> Result<OwnerSnapshot, String> {
        let mut bytes = vec![0u8; self.storage.len()];
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    bytes.as_mut_ptr().cast(),
                    pointer(&self.storage),
                    bytes.len(),
                )
            },
            "timeline final snapshot read",
        )?;
        let mut s: OwnerSnapshot =
            serde_json::from_slice(&serde_json::to_vec(&self.template).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let mut at = 0;
        for b in &mut s.blobs {
            let end = at + b.bytes.len();
            b.bytes = bytes[at..end].to_vec();
            at = end;
        }
        s.arithmetic_sha256 = arithmetic.into();
        Ok(s)
    }
}
fn packed_raw_views(
    stream: Arc<NativeStream>,
    schema: Schema,
    allocation: &NativeBf16Allocation,
    reserve: usize,
) -> Result<NativeRawOwner, String> {
    let mut first = 0usize;
    let mut buffers = Vec::new();
    for plane in &schema.planes {
        let owner = match allocation {
            NativeBf16Allocation::GpuBuf(x) => NativeBf16Allocation::GpuBuf(x.clone()),
            NativeBf16Allocation::CudaSlice(x) => NativeBf16Allocation::CudaSlice(x.clone()),
            NativeBf16Allocation::CudaF32Bf16Storage(x) => {
                NativeBf16Allocation::cuda_f32_bf16_storage(x.owner.clone())?
            }
        };
        buffers.push(NativeBuffer::Bf16View(NativeBf16View::checked(
            owner,
            first,
            plane.elements,
            &stream,
        )?));
        first = first
            .checked_add(plane.elements)
            .ok_or("packed raw element overflow")?;
    }
    if first != allocation.elements() {
        return Err("packed raw total typed extent".into());
    }
    NativeRawOwner::from_owned_buffers(stream, schema, buffers, reserve)
}
fn actual_native_model_transaction_kda_run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev kda-model-transaction-native".into());
    }
    let path =
        std::env::var("KRASIS_KDA_MODEL_TRANSACTION_PLAN").map_err(|_| "frozen tiling plan")?;
    let output =
        std::env::var("KRASIS_KDA_MODEL_TRANSACTION_OUTPUT").map_err(|_| "fresh tiling output")?;
    let qp: TilingPlan = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let p = &qp.native;
    if !p.diagnostic_only
        || p.qualifies_candidate
        || p.rows == 0
        || qp.pressure_fractions.is_empty()
    {
        return Err("tiling diagnostic scope".into());
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
        return Err("actual tiling executable".into());
    }
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| e.to_string())?
        .trim()
        != p.boot_id
    {
        return Err("tiling actual boot".into());
    }
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("config safety")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("reserve overflow")?;
    let (h, d, t) = (p.heads, p.dim, p.rows);
    let hd = h.checked_mul(d).ok_or("HD")?;
    let raw = p
        .raw_inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    if raw.len() != 5 {
        return Err("raw plane count".into());
    }
    for (v, stride) in raw.iter().zip([hd * 2, hd * 2, hd * 2, hd * 2, h * 2]) {
        if v.len() != t.checked_mul(stride).ok_or("raw extent")? {
            return Err("raw extent mismatch".into());
        }
    }
    if owner_checked(&p.initial)?.iter().any(|x| *x != 0) {
        return Err("this frozen zero-start corpus requires exact zero incoming".into());
    }
    let expected = owner_checked(&qp.output_reference)?;
    let expected_bf = owner_checked(&qp.bf16_reference)?;
    if expected.len()
        != t.checked_mul(hd)
            .and_then(|v| v.checked_mul(4))
            .ok_or("F32 output extent")?
        || expected_bf.len() != t * hd * 2
    {
        return Err("expected complete output extent".into());
    }
    let prefix: OwnerSnapshot =
        serde_json::from_slice(&owner_checked(&qp.prefix_reference)?).map_err(|e| e.to_string())?;
    let final_state: OwnerSnapshot =
        serde_json::from_slice(&owner_checked(&qp.final_reference)?).map_err(|e| e.to_string())?;
    let start = p
        .graph_suffix_starts
        .iter()
        .copied()
        .max()
        .ok_or("graph suffix start")?;
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
    let _precision_monitor = crate::vram_monitor::native_calibration_monitor(
        vec![device.ordinal() as i32],
        u64::try_from(reserve / (1024 * 1024)).map_err(|_| "monitor safety extent")?,
    )?;
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
    let cc = [
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?,
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?,
    ];
    if actual_uuid != p.uuid || cc != p.compute {
        return Err("actual tiling architecture".into());
    }
    let root = Path::new(&output);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    if qp.pressure_fractions.len() != 1
        || qp.pressure_fractions[0][1] == 0
        || qp.pressure_fractions[0][0] >= qp.pressure_fractions[0][1]
    {
        return Err("one explicit bounded pressure fraction required".into());
    }
    let before = memory()?;
    let actual_prefill_owner = crate::gpu_prefill::PrefillComputeStream::create(device.clone())?;
    let prefill_owner_weak = Arc::downgrade(&actual_prefill_owner);
    let prefill_stream = NativeStream::from_prefill_owner(actual_prefill_owner.clone())?;
    drop(actual_prefill_owner);
    if prefill_owner_weak.strong_count() != 1 {
        return Err("actual prefill stream owner not exclusively retained".into());
    }
    let decode_stream = NativeStream::create(device.clone())?;
    let legacy = unsafe { NativeStream::borrow_handle(device.clone(), std::ptr::null_mut()) }?;
    if *device.cu_stream() != legacy.handle || prefill_stream.handle == decode_stream.handle {
        return Err("actual distinct/default stream admission".into());
    }
    let singleton = PreparedOwner::new_on_stream(
        device.clone(),
        decode_stream.clone(),
        h,
        d,
        1,
        reserve,
        &transport,
        &math,
    )?;
    singleton.initialize_timeline(&[
        prefill_stream.clone(),
        decode_stream.clone(),
        legacy.clone(),
    ])?;
    let other_state = PreparedOwner::new_on_stream(
        device.clone(),
        decode_stream.clone(),
        h,
        d,
        1,
        reserve,
        &transport,
        &math,
    )?;
    other_state.initialize_timeline(&[
        prefill_stream.clone(),
        decode_stream.clone(),
        legacy.clone(),
    ])?;
    let other_singleton = other_state;
    if Arc::ptr_eq(&singleton.state, &other_singleton.state)
        || Arc::ptr_eq(&singleton.arena, &other_singleton.arena)
    {
        return Err("two-family distinct arena/independent state contract".into());
    }
    let mut thread_records = Vec::new();
    let kernels = KdaRawKernels::load(device.clone(), &producer_ptx)?;
    let one_schema = Schema::checked(RawContract::KdaBf16VectorGate, 1, h, d, d)?;
    if memory()?[0]
        < one_schema
            .requested_bytes
            .checked_add(reserve)
            .ok_or("packed singleton reserve overflow")?
    {
        return Err("packed singleton admission".into());
    }
    let storage_bytes = one_schema
        .planes
        .iter()
        .take(4)
        .try_fold(0usize, |sum, p| {
            sum.checked_add(
                p.elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or("model source storage extent")?,
            )
            .ok_or("model source storage total")
        })?
        .checked_add(
            one_schema
                .planes
                .last()
                .ok_or("model beta schema")?
                .storage_bytes,
        )
        .ok_or("model total source extent")?;
    if memory()?[0]
        < storage_bytes
            .checked_add(reserve)
            .ok_or("model source reserve overflow")?
    {
        return Err("model source physical admission".into());
    }
    let source_vector = one_schema
        .planes
        .iter()
        .take(4)
        .map(|p| {
            device
                .alloc_zeros::<f32>(p.elements)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_four: [Arc<CudaSlice<f32>>; 4] = source_vector
        .try_into()
        .map_err(|_| "four model vector sources")?;
    let source_beta = Arc::new(
        device
            .alloc_zeros::<u16>(one_schema.planes.last().ok_or("beta")?.elements)
            .map_err(|e| e.to_string())?,
    );
    device.synchronize().map_err(|e| e.to_string())?; // Setup-only initialization on actual model CudaDevice stream.
    let one_vector_weaks = source_four.iter().map(Arc::downgrade).collect::<Vec<_>>();
    let one_beta_weak = Arc::downgrade(&source_beta);
    let true_source_addresses = source_four
        .iter()
        .map(|x| *x.device_ptr())
        .chain(std::iter::once(*source_beta.device_ptr()))
        .collect::<Vec<_>>();
    let wrapper_before = memory()?;
    let actual_model_factory = RetainedDecodeKdaSource::from_model_storage(
        device.clone(),
        source_four.clone(),
        source_beta.clone(),
        h,
        d,
        reserve,
    )?;
    if actual_model_factory.addresses() != true_source_addresses || memory()? != wrapper_before {
        return Err("actual model factory pointer/allocation mismatch".into());
    }
    drop(actual_model_factory); // Setup-only default-stream source lease proof; no positive phase is active yet.
    let source = RetainedDecodeKdaSource::from_storage_on_stream(
        decode_stream.clone(),
        source_four.clone(),
        source_beta.clone(),
        h,
        d,
        reserve,
    )?;
    let external_one = source.inner.clone();
    drop(source);
    if memory()? != wrapper_before {
        return Err("explicit-stream model source wrapper allocated GPU memory".into());
    }
    let one_packed_admissions =
        serde_json::to_value(&external_one.admissions).map_err(|e| e.to_string())?;
    let source_storage=source_four.iter().map(|x|json!({"storage_dtype":"float32","view_dtype":"bfloat16","source_base":*x.device_ptr(),"storage_bytes":x.len()*std::mem::size_of::<f32>()})).chain(std::iter::once(json!({"storage_dtype":"bfloat16","view_dtype":"bfloat16","source_base":*source_beta.device_ptr(),"storage_bytes":source_beta.len()*std::mem::size_of::<u16>()}))).collect::<Vec<_>>();
    drop(source_four);
    drop(source_beta);
    if one_vector_weaks.iter().any(|x| x.strong_count() != 1) || one_beta_weak.strong_count() != 1 {
        return Err("actual model typed source owners not retained by views".into());
    }
    let one_weak = Arc::downgrade(&external_one);
    let one_addresses = external_one.addresses();
    let one = KdaRawProducer::from_raw_owner(
        external_one.clone(),
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
    if !Arc::ptr_eq(&external_one, &one.raw)
        || one.raw.addresses() != one_addresses
        || one
            .allocated_bytes
            .checked_add(external_one.schema.requested_bytes)
            != Some(one.requested_bytes)
    {
        return Err("singleton retained raw pointer/accounting".into());
    }
    drop(external_one);
    if one_weak.strong_count() != 1 {
        return Err("singleton source not exclusively retained by producer".into());
    }
    let emissions = (0..2)
        .map(|_| {
            Ok(ModelFamilyEmission {
                float: native_zeroed::<f32>(&decode_stream, hd)?,
                bf16: native_zeroed::<u16>(&decode_stream, hd)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    decode_stream.drain()?;
    let bindings = [
        ModelFamilyBinding {
            layer: 0,
            owner: &singleton,
            producer: &one,
        },
        ModelFamilyBinding {
            layer: 1,
            owner: &other_singleton,
            producer: &one,
        },
    ];
    let family_identity = model_family_identity(&bindings)?;
    let proof_start = std::time::Instant::now();
    let binding_proof = ModelBindingProof::checked(&bindings, &family_identity)?;
    let proof_ns = proof_start.elapsed().as_nanos();
    owner_save(root,"cold-binding-proof.json",&serde_json::to_vec_pretty(&json!({"construction_ns":proof_ns,"identity":binding_proof.identity,"scope":"cold complete immutable KDA storage witness; no model or GPU ordering waiver"})).map_err(|e|e.to_string())?)?;
    let host_before = bindings
        .iter()
        .map(|b| {
            let s = b
                .owner
                .state
                .timeline
                .lock()
                .map_err(|_| "before capture family lock")?;
            let t = s.as_ref().ok_or("before capture missing family")?;
            Ok((t.phases, t.recorded, t.poisoned))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let graph = capture_model_families(&bindings, &emissions, &decode_stream)?;
    let host_after = bindings
        .iter()
        .map(|b| {
            let s = b
                .owner
                .state
                .timeline
                .lock()
                .map_err(|_| "after capture family lock")?;
            let t = s.as_ref().ok_or("after capture missing family")?;
            Ok((t.phases, t.recorded, t.poisoned))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if host_before != host_after {
        return Err("capture advanced committed timelines".into());
    }
    owner_save(root,"capture-timeline.json",&serde_json::to_vec_pretty(&json!({"before":host_before,"after":host_after,"unchanged":true,"family_identity":family_identity})).map_err(|e|e.to_string())?)?;
    let graph_inventory = model_graph_inventory(&graph)?;
    let quiet_graph = capture_quiet_model_families(&bindings, &emissions, &decode_stream)?;
    let quiet_inventory = model_graph_inventory(&quiet_graph)?;
    owner_save(
        root,
        "quiet-graph-inventory.json",
        &serde_json::to_vec_pretty(&quiet_inventory).map_err(|e| e.to_string())?,
    )?;
    let quiet_events = QuietBurstEvents::new()?;
    let twin = KdaRawProducer::from_raw_owner(
        one.raw.clone(),
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
    let rebound = PreparedOwner::bind_ordered(&singleton, &other_singleton)?;
    capture_storage_identity_controls(
        root,
        &device,
        &bindings,
        &decode_stream,
        &graph.identity,
        &twin,
        &rebound,
    )?;
    drop(rebound);
    drop(twin);
    let mut graph = Some(graph);
    let batch_schema = Schema::checked(RawContract::KdaBf16VectorGate, t, h, d, d)?;
    if memory()?[0]
        < batch_schema
            .requested_bytes
            .checked_add(reserve)
            .ok_or("packed batch reserve overflow")?
    {
        return Err("packed batch admission".into());
    }
    let batch_source_vector = batch_schema
        .planes
        .iter()
        .map(|p| native_zeroed::<u16>(&prefill_stream, p.elements).map(Arc::new))
        .collect::<Result<Vec<_>, _>>()?;
    let batch_sources: [Arc<GpuBuf<u16>>; 5] = batch_source_vector
        .try_into()
        .map_err(|_| "five model BF16 prefill sources")?;
    prefill_stream.drain()?; // Setup only, before positive phases.
    let batch_vector_weaks = batch_sources.iter().map(Arc::downgrade).collect::<Vec<_>>();
    let batch_allocation = NativeBf16Allocation::GpuBuf(batch_sources[0].clone());
    let wrapper_before = memory()?;
    let actual_stream_owner = prefill_stream
        .prefill_owner
        .as_ref()
        .ok_or("actual prefill stream lease missing")?
        .clone();
    let source = RetainedPrefillKdaSource::from_model_storage(
        actual_stream_owner,
        batch_sources.clone(),
        t,
        h,
        d,
        reserve,
    )?;
    let external_batch = source.inner.clone();
    drop(source);
    if memory()? != wrapper_before {
        return Err("actual prefill source wrapper allocated GPU memory".into());
    }
    typed_model_family_setup_controls(
        root,
        &device,
        &external_batch,
        &one.raw,
        &one,
        &kernels,
        &transport,
        &math,
        reserve,
    )?;
    let batch_packed_admissions =
        serde_json::to_value(&external_batch.admissions).map_err(|e| e.to_string())?;
    let batch_source_storage=batch_sources.iter().map(|x|json!({"storage_dtype":"bfloat16","view_dtype":"bfloat16","source_base":*x.device_ptr(),"storage_bytes":x.len()*std::mem::size_of::<u16>()})).collect::<Vec<_>>();
    let mut view_rejections = Vec::new();
    for (first, elements, label) in [
        (0, 0, "zero view"),
        (batch_allocation.elements(), 1, "view past typed allocation"),
        (usize::MAX, 1, "view extent overflow"),
    ] {
        let observed = memory()?;
        if NativeBf16View::checked(
            NativeBf16Allocation::GpuBuf(batch_sources[0].clone()),
            first,
            elements,
            &prefill_stream,
        )
        .is_ok()
            || memory()? != observed
        {
            return Err(format!("owned view guard failed/allocated: {label}"));
        }
        view_rejections.push(label);
    }
    let observed = memory()?;
    if NativeRawOwner::from_owned_buffers(
        prefill_stream.clone(),
        external_batch.schema.clone(),
        Vec::new(),
        reserve,
    )
    .is_ok()
        || memory()? != observed
    {
        return Err("owned source plane count guard failed/allocated".into());
    }
    view_rejections.push("source plane count");
    let mut aliased = Vec::new();
    for plane in &external_batch.schema.planes {
        aliased.push(NativeBuffer::Bf16View(NativeBf16View::checked(
            NativeBf16Allocation::GpuBuf(batch_sources[0].clone()),
            0,
            plane.elements,
            &prefill_stream,
        )?));
    }
    let observed = memory()?;
    if NativeRawOwner::from_owned_buffers(
        prefill_stream.clone(),
        external_batch.schema.clone(),
        aliased,
        reserve,
    )
    .is_ok()
        || memory()? != observed
    {
        return Err("owned source overlap guard failed/allocated".into());
    }
    view_rejections.push("source planes overlap");
    owner_save(
        root,
        "packed-view-guards.json",
        &serde_json::to_vec_pretty(&view_rejections).map_err(|e| e.to_string())?,
    )?;
    drop(batch_allocation);
    drop(batch_sources);
    if batch_vector_weaks.iter().any(|x| x.strong_count() != 1) {
        return Err("batch typed allocation owner not retained by all views".into());
    }
    let batch_weak = Arc::downgrade(&external_batch);
    let batch_addresses = external_batch.addresses();
    kda_raw_upload(&external_batch, &raw)?;
    typed_model_family_chronology(
        root,
        &external_batch,
        &one.raw,
        &one,
        &kernels,
        &transport,
        &math,
        &qp,
        reserve,
    )?;
    let wrong_schema = Arc::new(NativeRawOwner::allocate(
        prefill_stream.clone(),
        Schema::checked(RawContract::KdaPreparedF32VectorGate, 1, h, d, d)?,
        reserve,
    )?);
    let mut bad_span = NativeRawOwner::allocate(
        prefill_stream.clone(),
        Schema::checked(RawContract::KdaBf16VectorGate, 1, h, d, d)?,
        reserve,
    )?;
    bad_span.admissions[0].requested_bytes = 1;
    let bad_span = Arc::new(bad_span);
    prefill_stream.drain()?;
    let output_gpu = native_zeroed::<u8>(&prefill_stream, expected.len())?;
    let bf16_gpu = native_zeroed::<u8>(&prefill_stream, expected_bf.len())?;
    let prefix_gpu = TimelineSnapshot::new(&prefill_stream, &prefix)?;
    let final_gpu = TimelineSnapshot::new(&prefill_stream, &final_state)?;
    let other_output_gpu = native_zeroed::<u8>(&prefill_stream, expected.len())?;
    let other_bf16_gpu = native_zeroed::<u8>(&prefill_stream, expected_bf.len())?;
    let other_prefix_gpu = TimelineSnapshot::new(&prefill_stream, &prefix)?;
    let other_final_gpu = TimelineSnapshot::new(&prefill_stream, &final_state)?;
    prefill_stream.drain()?;
    decode_stream.drain()?; // Setup only; not phase ordering.
    if memory()?[0] < reserve {
        return Err("timeline setup reserve violated".into());
    }
    let negative = PreparedOwner::from_shared(&singleton, reserve)?;
    negative.initialize_timeline(&[
        prefill_stream.clone(),
        decode_stream.clone(),
        legacy.clone(),
    ])?;

    let fault_a = PreparedOwner::from_shared(&singleton, reserve)?;
    let fault_b = PreparedOwner::from_shared(&singleton, reserve)?;
    for owner in [&fault_a, &fault_b] {
        owner.initialize_timeline(&[
            prefill_stream.clone(),
            decode_stream.clone(),
            legacy.clone(),
        ])?;
    }
    let baseline_a = PreparedOwner::from_shared(&singleton, reserve)?;
    let baseline_b = PreparedOwner::from_shared(&singleton, reserve)?;
    for owner in [&baseline_a, &baseline_b] {
        owner.initialize_timeline(&[
            prefill_stream.clone(),
            decode_stream.clone(),
            legacy.clone(),
        ])?;
    }
    let fraction = qp.pressure_fractions[0];
    let free_before_pressure = memory()?[0];
    let fraction_bytes = |free: usize| -> Result<usize, String> {
        free.checked_sub(reserve)
            .ok_or("live calibration reserve")?
            .checked_mul(fraction[0])
            .and_then(|x| x.checked_div(fraction[1]))
            .ok_or("live pressure extent".into())
    };
    let initial_pressure = fraction_bytes(free_before_pressure)?;
    let pressure = if initial_pressure > 0 {
        Some(native_zeroed::<u8>(&prefill_stream, initial_pressure)?)
    } else {
        None
    };
    prefill_stream.drain()?;
    let target_free = memory()?[0];
    let target_budget = target_free
        .checked_sub(reserve)
        .ok_or("live target safety")?;
    drop(pressure);
    prefill_stream.drain()?;
    let maximum = live_tiling_calibration::tiled_capacity(h, d, t, memory()?[0], reserve, &device)?;
    let mut probes = std::collections::BTreeMap::<usize, serde_json::Value>::new();
    for n in [1, maximum] {
        if probes.contains_key(&n) {
            continue;
        }
        let value = live_tiling_calibration::calibrated_allocation_probe(
            prefill_stream.clone(),
            device.clone(),
            kernels.clone(),
            p,
            n,
            reserve,
            &transport,
            &math,
            &a_log,
            &bias,
            external_batch.clone(),
            &expected,
            &expected_bf,
        )?;
        owner_save(
            root,
            &format!("allocation-calibration-{n}.json"),
            &serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
        )?;
        probes.insert(n, value);
    }
    let cost = |v: &serde_json::Value| -> Result<usize, String> {
        usize::try_from(
            v["measured_bytes"]
                .as_u64()
                .ok_or("real physical footprint absent")?,
        )
        .map_err(|_| "physical footprint extent".into())
    };
    if cost(probes.get(&1).ok_or("short probe absent")?)? > target_budget {
        return Err("measured short exceeds target; no fallback".into());
    }
    let mut low = 1;
    let mut high = maximum;
    if cost(probes.get(&maximum).ok_or("long probe absent")?)? <= target_budget {
        low = maximum;
    }
    while low < high {
        let mid = low + (high - low + 1) / 2;
        if !probes.contains_key(&mid) {
            let v = live_tiling_calibration::calibrated_allocation_probe(
                prefill_stream.clone(),
                device.clone(),
                kernels.clone(),
                p,
                mid,
                reserve,
                &transport,
                &math,
                &a_log,
                &bias,
                external_batch.clone(),
                &expected,
                &expected_bf,
            )?;
            owner_save(
                root,
                &format!("allocation-calibration-{mid}.json"),
                &serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?,
            )?;
            probes.insert(mid, v);
        }
        if cost(probes.get(&mid).ok_or("adaptive probe absent")?)? <= target_budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let capacity = low;
    let measured_selected = cost(probes.get(&capacity).ok_or("selected probe absent")?)?;
    let final_pressure = fraction_bytes(memory()?[0])?;
    let pressure = if final_pressure > 0 {
        Some(native_zeroed::<u8>(&prefill_stream, final_pressure)?)
    } else {
        None
    };
    prefill_stream.drain()?;
    let actual_final_free = memory()?[0];
    if measured_selected
        > actual_final_free
            .checked_sub(reserve)
            .ok_or("final live reserve")?
    {
        return Err("postcleanup free invalidates measured capacity; no fallback".into());
    }
    owner_save(root,"calibrated-capacity.json",&serde_json::to_vec_pretty(&json!({"capacity":capacity,"short":1,"long":maximum,"target_free":target_free,"target_budget":target_budget,"actual_final_free":actual_final_free,"selected_measured_bytes":measured_selected,"reserve_bytes":reserve,"pressure_fraction":fraction,"initial_pressure_bytes":initial_pressure,"final_pressure_bytes":final_pressure,"retained_live_input_bytes":external_batch.schema.requested_bytes,"new_payload_excludes_retained_input":true,"probes":probes})).map_err(|e|e.to_string())?)?;
    let case_precision = crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?
        .ok_or("required final live precision window disabled")?;
    let template = PreparedOwner::new_on_stream(
        device.clone(),
        prefill_stream.clone(),
        h,
        d,
        capacity,
        reserve,
        &transport,
        &math,
    )?;
    let batch = PreparedOwner::bind_ordered(&template, &singleton)?;
    let other_batch = PreparedOwner::bind_ordered(&template, &other_singleton)?;

    let producer = KdaRawProducer::from_live_raw_owner(
        external_batch.clone(),
        h,
        d,
        capacity,
        reserve,
        p.lower,
        p.scale,
        p.epsilon,
        kernels,
        &a_log,
        &bias,
    )?;
    if !Arc::ptr_eq(&external_batch, &producer.raw)
        || producer.raw.addresses() != batch_addresses
        || producer
            .allocated_bytes
            .checked_add(external_batch.schema.requested_bytes)
            != Some(producer.requested_bytes)
    {
        return Err("batch retained raw pointer/accounting".into());
    }
    let model_family = BorrowedKdaModelFamily::checked(&batch, &singleton, &producer, &one)?;
    let other_model_family =
        BorrowedKdaModelFamily::checked(&other_batch, &other_singleton, &producer, &one)?;
    owner_save(root,"calibrated-model-family-pairs.json",&serde_json::to_vec_pretty(&json!({"prefill":model_family_identity(&[model_family.prefill_binding(0),other_model_family.prefill_binding(1)])?,"decode":model_family_identity(&[model_family.decode_binding(0),other_model_family.decode_binding(1)])?,"capacity":capacity,"one_persistent_family_per_pair":true,"constructed_from_actual_calibrated_owners":true})).map_err(|e|e.to_string())?)?;
    drop(external_batch);
    if batch_weak.strong_count() != 1 {
        return Err("batch source not exclusively retained by producer".into());
    }
    // Authentic source uploaded once externally; no producer raw allocation/copy.
    let mut raw_rejections = Vec::new();
    for (source, heads, rows, a_bytes, label) in [
        (
            wrong_schema.clone(),
            h,
            1,
            a_log.as_slice(),
            "prepared source contract",
        ),
        (
            bad_span.clone(),
            h,
            1,
            a_log.as_slice(),
            "raw admission span",
        ),
        (
            producer.raw.clone(),
            h,
            1,
            a_log.as_slice(),
            "raw row extent",
        ),
        (
            producer.raw.clone(),
            h,
            t,
            &a_log[..a_log.len() - 1],
            "parameter byte extent",
        ),
    ] {
        let before_rejection = memory()?;
        let result = KdaRawProducer::from_raw_owner(
            source,
            heads,
            d,
            rows,
            reserve,
            p.lower,
            p.scale,
            p.epsilon,
            producer.kernels.clone(),
            a_bytes,
            &bias,
        );
        if result.is_ok() || memory()? != before_rejection {
            return Err(format!("retained raw guard failed/allocated: {label}"));
        }
        raw_rejections.push(label);
    }
    drop(wrong_schema);
    drop(bad_span);
    owner_save(
        root,
        "raw-source-guards.json",
        &serde_json::to_vec_pretty(&raw_rejections).map_err(|e| e.to_string())?,
    )?;
    let mut extent_rejections = Vec::new();
    for (first, rows, label) in [
        (t, 1, "input past live end"),
        (usize::MAX, 1, "input offset overflow"),
        (
            0,
            capacity.checked_add(1).ok_or("negative capacity extent")?,
            "prepared output past tile",
        ),
        (0, 0, "zero live rows"),
    ] {
        let before = memory()?;
        if producer.enqueue(first, rows, &batch).is_ok() || memory()? != before {
            return Err(format!("live extent guard failed/allocated:{label}"));
        }
        extent_rejections.push(label);
    }
    owner_save(
        root,
        "live-extent-guards.json",
        &serde_json::to_vec_pretty(&extent_rejections).map_err(|e| e.to_string())?,
    )?;
    let mut registry = crate::session_cache::SequenceStateRegistry::default();
    batch.register(&mut registry, 0)?;
    let mut single_registry = crate::session_cache::SequenceStateRegistry::default();
    singleton.register(&mut single_registry, 0)?;
    if registry.allocations() != single_registry.allocations() {
        return Err("timeline same persistent pointer family".into());
    }
    let mut other_registry = crate::session_cache::SequenceStateRegistry::default();
    other_singleton.register(&mut other_registry, 0)?;
    // Local per-family registry uses the frozen single-layer reference names; the
    // complete model identity above independently records layer0/layer1 inventory.
    model_arena_admission_controls(
        root,
        &device,
        &bindings,
        &decode_stream,
        &family_identity,
        &fault_a,
        &fault_b,
        &one,
    )?;
    let mut model_rejections = Vec::new();
    let missing = &bindings[..1];
    if ModelBindingProof::checked(missing, &family_identity).is_ok() {
        return Err("missing graph family admitted".into());
    }
    model_rejections.push("missing family");
    let duplicate = [
        ModelFamilyBinding {
            layer: 0,
            owner: &singleton,
            producer: &one,
        },
        ModelFamilyBinding {
            layer: 0,
            owner: &other_singleton,
            producer: &one,
        },
    ];
    if model_family_identity(&duplicate).is_ok() {
        return Err("duplicate graph key admitted".into());
    }
    model_rejections.push("duplicate key");
    let alias = [
        ModelFamilyBinding {
            layer: 0,
            owner: &singleton,
            producer: &one,
        },
        ModelFamilyBinding {
            layer: 1,
            owner: &singleton,
            producer: &one,
        },
    ];
    if model_family_identity(&alias).is_ok() {
        return Err("aliased graph state admitted".into());
    }
    model_rejections.push("aliased state");
    let locked = other_singleton
        .state
        .timeline
        .try_lock()
        .map_err(|_| "other family negative lease")?;
    let (_, record) = request_worker(
        &device,
        "complete-family-lock-rejection",
        || -> Result<(), String> {
            if ModelFamilyAdmission::from_proof(&binding_proof, &decode_stream).is_ok() {
                return Err("concurrent complete graph admitted".into());
            }
            Ok(())
        },
    )?;
    thread_records.push(record);
    drop(locked);
    model_rejections.push("concurrent family");
    let baseline_bindings = [
        ModelFamilyBinding {
            layer: 0,
            owner: &baseline_a,
            producer: &one,
        },
        ModelFamilyBinding {
            layer: 1,
            owner: &baseline_b,
            producer: &one,
        },
    ];
    let baseline_identity = model_family_identity(&baseline_bindings)?;
    let mut old =
        ModelFamilyAdmission::checked(&baseline_bindings, &decode_stream, &baseline_identity)?;
    let error = old
        .execute_frozen296(|| -> Result<(), String> {
            for ((src_desc, src), (dst_desc, dst)) in producer
                .raw
                .inventory()
                .into_iter()
                .zip(one.raw.inventory())
            {
                let stride = src_desc.strides_bytes[0];
                if stride != dst_desc.storage_bytes {
                    return Err("baseline raw extent".into());
                }
                timeline_copy(&decode_stream, dst, src, stride)?;
            }
            let lease = baseline_a.arena.lease()?;
            one.enqueue(0, 1, &baseline_a)?;
            baseline_a.decode_capture_in(&lease, &one.view(1)?)?;
            Err("injected after actual first-family GPU submission".into())
        })
        .err()
        .ok_or("baseline injection unexpectedly succeeded")?;
    if error != "injected after actual first-family GPU submission" {
        return Err("baseline stopped before authentic injection".into());
    }
    let timeline_flags = |tx: &ModelFamilyAdmission<'_>| -> Result<Vec<Value>, String> {
        tx.slots.iter().map(|slot|{let t=slot.as_ref().ok_or("baseline timeline missing")?;Ok(json!({"poisoned":t.poisoned,"recorded":t.recorded,"phases":t.phases,"phases_decimal":t.phases.to_string()}))}).collect()
    };
    let failed_flags = timeline_flags(&old)?;
    let mut old_capture_entered = false;
    old.capture_frozen296(|| {
        old_capture_entered = true;
        Ok(())
    })?;
    let after_capture = timeline_flags(&old)?;
    let mut old_execute_entered = false;
    old.execute_frozen296(|| {
        old_execute_entered = true;
        Ok(())
    })?;
    let reopened_flags = timeline_flags(&old)?;
    if !old_capture_entered
        || !old_execute_entered
        || failed_flags != after_capture
        || failed_flags
            .iter()
            .any(|x| x["poisoned"] != true || x["recorded"] != true || x["phases"] != 0)
        || reopened_flags
            .iter()
            .any(|x| x["poisoned"] != false || x["recorded"] != true || x["phases"] != 1)
    {
        return Err("frozen296 held-transaction mechanism not reproduced".into());
    }
    drop(old);
    let mut overflow =
        ModelFamilyAdmission::checked(&baseline_bindings, &decode_stream, &baseline_identity)?;
    overflow.slots[1]
        .as_mut()
        .ok_or("overflow test timeline")?
        .phases = u64::MAX;
    let overflow_before = timeline_flags(&overflow)?;
    let mut overflow_entered = false;
    let overflow_error = overflow
        .execute(|| {
            overflow_entered = true;
            Ok(())
        })
        .err()
        .ok_or("overflow replay admitted")?;
    let overflow_after = timeline_flags(&overflow)?;
    if overflow_entered
        || overflow_error != "complete graph phase overflow"
        || overflow_before != overflow_after
    {
        return Err("overflow mutated family or submitted callback".into());
    }
    drop(overflow);
    drop(baseline_bindings);
    decode_stream.drain()?; // Isolated fault validation only; never positive phase ordering.
    let mut original_gpu_cursors = Vec::new();
    for owner in [&baseline_a, &baseline_b] {
        let mut absolute = 0u64;
        let mut pending = 0i32;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    (&mut absolute as *mut u64).cast(),
                    pointer(&owner.absolute),
                    std::mem::size_of::<u64>(),
                )
            },
            "baseline actual absolute",
        )?;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    (&mut pending as *mut i32).cast(),
                    pointer(&owner.pending),
                    std::mem::size_of::<i32>(),
                )
            },
            "baseline actual pending",
        )?;
        original_gpu_cursors.push(json!({"absolute":absolute,"pending":pending}));
    }
    if original_gpu_cursors
        != vec![
            json!({"absolute":1,"pending":1}),
            json!({"absolute":0,"pending":0}),
        ]
    {
        return Err("baseline first-family actual cursor mismatch".into());
    }
    owner_save(root,"held-transaction-frozen296-control.json",&serde_json::to_vec_pretty(&json!({"actual_first_family_gpu_submission":true,"gpu_cursors":original_gpu_cursors,"capture_callback_entered":old_capture_entered,"execute_callback_entered":old_execute_entered,"failed_flags":failed_flags,"after_capture":after_capture,"reopened_flags":reopened_flags,"overflow_callback_entered":overflow_entered,"overflow_error":overflow_error,"overflow_before":overflow_before,"overflow_after":overflow_after})).map_err(|e|e.to_string())?)?;
    let fault_bindings = [
        ModelFamilyBinding {
            layer: 0,
            owner: &fault_a,
            producer: &one,
        },
        ModelFamilyBinding {
            layer: 1,
            owner: &fault_b,
            producer: &one,
        },
    ];
    let fault_identity = model_family_identity(&fault_bindings)?;
    let fault_proof = ModelBindingProof::checked(&fault_bindings, &fault_identity)?;
    {
        let mut transaction = ModelFamilyAdmission::from_proof(&fault_proof, &decode_stream)?;
        let failure = transaction
            .execute(|| -> Result<(), String> {
                for ((src_desc, src), (dst_desc, dst)) in producer
                    .raw
                    .inventory()
                    .into_iter()
                    .zip(one.raw.inventory())
                {
                    if src_desc.strides_bytes[0] != dst_desc.storage_bytes {
                        return Err("fault authentic raw row extent".into());
                    }
                    timeline_copy(&decode_stream, dst, src, src_desc.strides_bytes[0])?;
                }
                let lease = fault_a.arena.lease()?;
                one.enqueue(0, 1, &fault_a)?;
                fault_a.decode_capture_in(&lease, &one.view(1)?)?;
                Err("injected after actual first-family GPU submission".into())
            })
            .err()
            .ok_or("partial submitted failure unexpectedly succeeded")?;
        if failure != "injected after actual first-family GPU submission" {
            return Err("partial-submission guard failed earlier than injection".into());
        }
        let mut entered_execute = false;
        let execute_error = transaction
            .execute(|| {
                entered_execute = true;
                Ok(())
            })
            .err()
            .ok_or("poisoned held execute admitted")?;
        let mut entered_capture = false;
        let capture_error = transaction
            .capture_only(|| {
                entered_capture = true;
                Ok(())
            })
            .err()
            .ok_or("poisoned held capture admitted")?;
        if entered_execute
            || entered_capture
            || execute_error != "complete graph family poisoned"
            || capture_error != "complete graph family poisoned"
        {
            return Err("held poison rejection callback/error mismatch".into());
        }
        owner_save(root,"held-transaction-rejection.json",&serde_json::to_vec_pretty(&json!({"execute_callback_entered":entered_execute,"capture_callback_entered":entered_capture,"execute_error":execute_error,"capture_error":capture_error,"families":timeline_flags(&transaction)?})).map_err(|e|e.to_string())?)?;
    }
    decode_stream.drain()?;
    let mut faults = Vec::new();
    for owner in [&fault_a, &fault_b] {
        let slot = owner
            .state
            .timeline
            .lock()
            .map_err(|_| "fault family mutex")?;
        let timeline = slot.as_ref().ok_or("fault family missing")?;
        if !timeline.poisoned || !timeline.recorded || timeline.phases != 0 {
            return Err("partial submitted failure did not poison every family".into());
        }
        let mut absolute = 0u64;
        let mut pending = 0i32;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    (&mut absolute as *mut u64).cast(),
                    pointer(&owner.absolute),
                    std::mem::size_of::<u64>(),
                )
            },
            "fault actual absolute",
        )?;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    (&mut pending as *mut i32).cast(),
                    pointer(&owner.pending),
                    std::mem::size_of::<i32>(),
                )
            },
            "fault actual pending",
        )?;
        faults.push(json!({"absolute":absolute,"pending":pending,"poisoned":timeline.poisoned,"recorded":timeline.recorded,"phases":timeline.phases}));
    }
    if faults[0]["absolute"] != 1
        || faults[0]["pending"] != 1
        || faults[1]["absolute"] != 0
        || faults[1]["pending"] != 0
    {
        return Err("partial GPU submission did not affect exactly first family".into());
    }
    if ModelFamilyAdmission::from_proof(&fault_proof, &decode_stream).is_ok() {
        return Err("poisoned complete family set admitted".into());
    }
    owner_save(root,"partial-submission-poison.json",&serde_json::to_vec_pretty(&json!({"families":faults,"injected_after_actual_gpu_submission":true,"all_families_future_admission_rejected":true})).map_err(|e|e.to_string())?)?;
    drop(fault_bindings);
    model_rejections.push("partial submitted failure poisons all families");
    model_rejections.push("poisoned family set");
    owner_save(
        root,
        "model-family-guards.json",
        &serde_json::to_vec_pretty(&model_rejections).map_err(|e| e.to_string())?,
    )?;
    let mut rejected = Vec::new();
    let unordered = PreparedOwner::bind_ordered(&template, &negative)?;
    if unordered.reset().is_ok() {
        return Err("unordered cross-stream reset admitted".into());
    }
    rejected.push("unordered cross-stream reset");
    drop(unordered);
    if PreparedOwner::bind_ordered(&template, &template).is_ok() {
        return Err("uninitialized foreign state admitted".into());
    }
    rejected.push("uninitialized foreign state");
    let nested = negative.ordered_phase(&decode_stream, |_| {
        negative.ordered_phase(&prefill_stream, |_| Ok(()))
    });
    if nested.is_ok() {
        return Err("concurrent family mutation admitted".into());
    }
    rejected.push("nested state-family lease");
    if negative.ordered_phase(&decode_stream, |_| Ok(())).is_ok() {
        return Err("poisoned family admitted".into());
    }
    rejected.push("poisoned family");
    drop(negative); // Only the isolated negative family is poisoned.
    owner_save(
        root,
        "negative-guards.json",
        &serde_json::to_vec_pretty(&rejected).map_err(|e| e.to_string())?,
    )?;
    let primary = *device.cu_primary_ctx() as usize;
    let mut allocation_domain_rejections = Vec::new();
    for (context, executing, pool, rights, label) in [
        (
            0,
            primary,
            0,
            Some(sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE as u32),
            "NULL context without pool",
        ),
        (0, primary, 1, None, "pool without permissions"),
        (
            0,
            primary,
            1,
            Some(sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ as u32),
            "read-only pool",
        ),
        (
            primary ^ 1,
            primary,
            1,
            Some(sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE as u32),
            "foreign context cannot use pool bypass",
        ),
        (primary, 0, 0, None, "NULL executing context"),
    ] {
        if native_allocation_domain(context, executing, pool, rights).is_ok() {
            return Err(format!("allocation domain guard failed: {label}"));
        }
        allocation_domain_rejections.push(label);
    }
    owner_save(
        root,
        "allocation-domain-guards.json",
        &serde_json::to_vec_pretty(&allocation_domain_rejections).map_err(|e| e.to_string())?,
    )?;
    let mut cases = Vec::new();
    for (case, quantum) in [capacity.min(start), (capacity / 2).max(1).min(start)]
        .into_iter()
        .enumerate()
    {
        let dir = root.join(format!("case-{case}"));
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        // Reset is itself a mutation ordered after the previous case's last event.
        let mut phases = Vec::new();
        let (_, record) = request_worker(&device, "two-family-batch", || -> Result<(), String> {
            batch.ordered_phase(&prefill_stream, |phase| phase.reset())?;
            other_batch.ordered_phase(&prefill_stream, |phase| phase.reset())?;
            let mut at = 0usize;

            while at < start {
                let count = quantum.min(start - at);
                batch.ordered_phase(&prefill_stream, |phase| {
                    phase.prefill(&producer, at, count)?;
                    timeline_copy(
                        phase.stream,
                        pointer(&output_gpu) + (at * hd * 4) as u64,
                        batch.p("emitted"),
                        count * hd * 4,
                    )?;
                    timeline_copy(
                        phase.stream,
                        pointer(&bf16_gpu) + (at * hd * 2) as u64,
                        pointer(&batch.arena.bf16_emitted),
                        count * hd * 2,
                    )?;
                    if at + count == start {
                        prefix_gpu.enqueue(phase, &registry)?;
                    }
                    Ok(())
                })?;
                other_batch.ordered_phase(&prefill_stream, |phase| {
                    phase.prefill(&producer, at, count)?;
                    timeline_copy(
                        phase.stream,
                        pointer(&other_output_gpu) + (at * hd * 4) as u64,
                        other_batch.p("emitted"),
                        count * hd * 4,
                    )?;
                    timeline_copy(
                        phase.stream,
                        pointer(&other_bf16_gpu) + (at * hd * 2) as u64,
                        pointer(&other_batch.arena.bf16_emitted),
                        count * hd * 2,
                    )?;
                    if at + count == start {
                        other_prefix_gpu.enqueue(phase, &other_registry)?;
                    }
                    Ok(())
                })?;
                phases.push(json!({"kind":"GPU batch","first":at,"rows":count,"stream":prefill_stream.handle as usize}));
                at += count;
            }
            Ok(())
        })?;
        thread_records.push(record);
        let thread_graph = &mut graph;
        let (_, record) = request_worker(
            &device,
            "two-family-graph-replay",
            || -> Result<(), String> {
                for row in start..t {
                    let replay = if case == 1 && ((row - start) % 2 == 1 || row + 1 == t) {
                        &legacy
                    } else {
                        &decode_stream
                    };
                    let mut transaction = ModelArenaAdmission::from_proof(&binding_proof, replay)?;
                    transaction.execute(|leases| {
                        let phase = &StatePhase {
                            owner: &singleton,
                            stream: replay,
                        };
                        for ((src_desc, src), (dst_desc, dst)) in producer
                            .raw
                            .inventory()
                            .into_iter()
                            .zip(one.raw.inventory())
                        {
                            let stride = src_desc.strides_bytes[0];
                            if stride != dst_desc.storage_bytes {
                                return Err("timeline actual raw row stride".into());
                            }
                            timeline_copy(phase.stream, dst, src + (row * stride) as u64, stride)?;
                        }
                        let live_graph =
                            thread_graph.as_mut().ok_or("raw graph already destroyed")?;
                        singleton.ordered_graph_in_lease(
                            phase,
                            &live_graph.inner,
                            model_arena_lease(&singleton, leases)?,
                        )?;
                        timeline_copy(
                            phase.stream,
                            pointer(&output_gpu) + (row * hd * 4) as u64,
                            pointer(&emissions[0].float),
                            hd * 4,
                        )?;
                        timeline_copy(
                            phase.stream,
                            pointer(&bf16_gpu) + (row * hd * 2) as u64,
                            pointer(&emissions[0].bf16),
                            hd * 2,
                        )?;
                        timeline_copy(
                            phase.stream,
                            pointer(&other_output_gpu) + (row * hd * 4) as u64,
                            pointer(&emissions[1].float),
                            hd * 4,
                        )?;
                        timeline_copy(
                            phase.stream,
                            pointer(&other_bf16_gpu) + (row * hd * 2) as u64,
                            pointer(&emissions[1].bf16),
                            hd * 2,
                        )?;
                        if row + 1 == t {
                            final_gpu.enqueue(phase, &registry)?;
                            let other_phase = StatePhase {
                                owner: &other_singleton,
                                stream: replay,
                            };
                            other_final_gpu.enqueue(&other_phase, &other_registry)?;
                        }
                        Ok(())
                    })?;
                    phases.push(json!({"kind":"captured GPU singleton","first":row,"rows":1,"stream":replay.handle as usize}));
                }
                if case == 1 {
                    drop(
                        thread_graph
                            .take()
                            .ok_or("final graph missing before teardown")?,
                    );
                    // Destructor completes the final default-stream replay before validation readback.
                }
                Ok(())
            },
        )?;
        thread_records.push(record);
        // Explicit validation wait occurs only after all positive phases were queued.
        {
            let slot = singleton
                .state
                .timeline
                .lock()
                .map_err(|_| "state timeline lock")?;
            let timeline = slot.as_ref().ok_or("missing final timeline")?;
            if timeline.poisoned || !timeline.recorded {
                return Err("final timeline invalid".into());
            }
            check(
                unsafe { sys::lib().cuEventSynchronize(timeline.event) },
                "diagnostic final complete phase wait",
            )?;
        }
        let mut outputs = vec![0u8; expected.len()];
        let mut bf = vec![0u8; expected_bf.len()];
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    outputs.as_mut_ptr().cast(),
                    pointer(&output_gpu),
                    outputs.len(),
                )
            },
            "timeline final full output read",
        )?;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(bf.as_mut_ptr().cast(), pointer(&bf16_gpu), bf.len())
            },
            "timeline final full BF16 read",
        )?;
        let actual_prefix = prefix_gpu.read(&singleton.arithmetic_sha256)?;
        let actual_final = final_gpu.read(&singleton.arithmetic_sha256)?;
        owner_save(&dir, "whole-output.f32", &outputs)?;
        owner_save(&dir, "whole-output.bf16", &bf)?;
        owner_save(
            &dir,
            "prefix-snapshot.json",
            &serde_json::to_vec(&actual_prefix).map_err(|e| e.to_string())?,
        )?;
        owner_save(
            &dir,
            "final-snapshot.json",
            &serde_json::to_vec(&actual_final).map_err(|e| e.to_string())?,
        )?;
        owner_save(
            &dir,
            "phases.json",
            &serde_json::to_vec_pretty(&phases).map_err(|e| e.to_string())?,
        )?;
        owner_compare(&outputs, &expected, "timeline complete F32 outputs")?;
        owner_compare(&bf, &expected_bf, "timeline complete BF16 outputs")?;
        for (actual, want, label) in [
            (&actual_prefix, &prefix, "timeline complete prefix state"),
            (&actual_final, &final_state, "timeline complete final state"),
        ] {
            if actual.layout != want.layout {
                return Err(format!("{label} layout"));
            }
            owner_compare(
                &serde_json::to_vec(&actual.blobs).map_err(|e| e.to_string())?,
                &serde_json::to_vec(&want.blobs).map_err(|e| e.to_string())?,
                label,
            )?;
        }
        let mut other_output = vec![0u8; expected.len()];
        let mut other_bf = vec![0u8; expected_bf.len()];
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    other_output.as_mut_ptr().cast(),
                    pointer(&other_output_gpu),
                    other_output.len(),
                )
            },
            "other family output read",
        )?;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(
                    other_bf.as_mut_ptr().cast(),
                    pointer(&other_bf16_gpu),
                    other_bf.len(),
                )
            },
            "other family BF16 read",
        )?;
        owner_compare(&other_output, &expected, "other family complete F32")?;
        owner_compare(&other_bf, &expected_bf, "other family complete BF16")?;
        let other_prefix = other_prefix_gpu.read(&other_singleton.arithmetic_sha256)?;
        let other_final = other_final_gpu.read(&other_singleton.arithmetic_sha256)?;
        for (actual, want) in [(&other_prefix, &prefix), (&other_final, &final_state)] {
            if actual.layout != want.layout {
                return Err("other local reference registry layout".into());
            }
            owner_compare(
                &serde_json::to_vec(&actual.blobs).map_err(|e| e.to_string())?,
                &serde_json::to_vec(&want.blobs).map_err(|e| e.to_string())?,
                "other independent family complete snapshot",
            )?;
        }
        owner_save(&dir, "other-whole-output.f32", &other_output)?;
        owner_save(&dir, "other-whole-output.bf16", &other_bf)?;
        owner_save(
            &dir,
            "other-prefix-snapshot.json",
            &serde_json::to_vec(&other_prefix).map_err(|e| e.to_string())?,
        )?;
        owner_save(
            &dir,
            "other-final-snapshot.json",
            &serde_json::to_vec(&other_final).map_err(|e| e.to_string())?,
        )?;
        long_check_cursor(&other_singleton, t)?;
        long_check_cursor(&singleton, t)?;
        let observed = memory()?;
        if observed[0] < reserve {
            return Err("timeline final reserve violated".into());
        }
        cases.push(json!({"case":case,"batch_quantum":quantum,"suffix":t-start,"phase_count":phases.len(),"complete_outputs_and_states_exact":true,"observed_final_memory":observed,"graph_teardown_before_validation_wait":case==1}));
    }
    let positive_events = {
        let guard = singleton
            .state
            .timeline
            .lock()
            .map_err(|_| "positive event count")?;
        guard.as_ref().ok_or("positive timeline missing")?.phases
    };
    let case_min_free =
        usize::try_from(case_precision.finish()?).map_err(|_| "case precision extent")?;
    if case_min_free < reserve {
        return Err("actual live precision floor below configured safety".into());
    }
    drop(_precision_monitor);
    quiet_ownership_compare(
        root,
        &bindings,
        &quiet_graph,
        &quiet_events,
        &decode_stream,
        &prefill_stream,
        [&batch, &other_batch],
        &producer,
        [&bf16_gpu, &other_bf16_gpu],
        [&registry, &other_registry],
        start,
        t,
        capacity,
        &expected_bf,
        &final_state,
        reserve,
    )?;
    drop(quiet_graph);
    drop(quiet_events);
    if batch_weak.strong_count() != 1 || one_weak.strong_count() != 1 {
        return Err("raw source ownership changed during phases".into());
    }
    let raw_binding = json!({"same_batch_source_addresses":batch_addresses,"same_singleton_source_addresses":one_addresses,"raw_input_new_allocations":0,"raw_input_copies":0,"batch_new_workspace_bytes":producer.allocated_bytes,"live_input_capacity":producer.input_capacity,"prepared_tile_capacity":producer.capacity,"per_tile_source_uploads":0,"singleton_new_workspace_bytes":one.allocated_bytes,"batch_complete_logical_payload_bytes":producer.requested_bytes,"singleton_complete_logical_payload_bytes":one.requested_bytes,"raw_rejected_guards":raw_rejections,"actual_live_model":"NOT_INTEGRATED"});
    if batch_vector_weaks.iter().any(|x| x.strong_count() != 1)
        || one_vector_weaks.iter().any(|x| x.strong_count() != 1)
        || one_beta_weak.strong_count() != 1
    {
        return Err("packed typed allocation owners changed during phases".into());
    }
    let producer_arithmetic = producer.arithmetic_sha256.clone();
    drop(graph); // Executable graph can no longer use producer raw/prepared/parameter allocations.
    drop(bindings);
    drop(producer);
    drop(one);
    if batch_weak.upgrade().is_some()
        || one_weak.upgrade().is_some()
        || batch_vector_weaks.iter().any(|x| x.upgrade().is_some())
        || one_vector_weaks.iter().any(|x| x.upgrade().is_some())
        || one_beta_weak.upgrade().is_some()
    {
        return Err("retained raw source not released after graph and producer".into());
    }
    owner_save(root,"model-thread-handoff.json",&serde_json::to_vec_pretty(&json!({"workers":thread_records,"exclusive_graph_transfer":true,"gpu_wait_between_batch_and_replay":false})).map_err(|e|e.to_string())?)?;
    let raw_binding = json!({"binding":raw_binding,"caller_owner_dropped_before_phases":true,"retained_until_final_graph_completion":true,"released_after_graph_and_producer_drop":true,"graph_borrows_producer":true,"last_default_replay_drained_before_validation":true});
    owner_save(
        root,
        "raw-source-binding.json",
        &serde_json::to_vec_pretty(&raw_binding).map_err(|e| e.to_string())?,
    )?;
    owner_save(root,"packed-source-binding.json",&serde_json::to_vec_pretty(&json!({"batch_type":"model-disjoint-BF16-prefill-views","batch_storage":batch_source_storage,"actual_prefill_factory_checked_before_positive_phases":true,"singleton_type":"model-disjoint-F32-storage-BF16-views","singleton_storage":source_storage,"actual_model_factory_checked_before_positive_phases":true,"batch_admissions":batch_packed_admissions,"singleton_admissions":one_packed_admissions,"typed_source_callers_dropped_before_phases":true,"typed_sources_released_after_final_graph_and_producer":true,"wrapper_new_gpu_allocations":0,"wrapper_gpu_copies":0,"rejected_guards":view_rejections,"setup_only_cuda_device_stream_barrier":true,"actual_model_integration":"OUTSTANDING"})).map_err(|e|e.to_string())?)?;
    drop(pressure);
    owner_save(root,"live-tiling-memory.json",&serde_json::to_vec_pretty(&json!({"minimum_free_bytes":case_min_free,"reserve_bytes":reserve,"capacity":capacity,"actual_live_input_capacity":t,"new_workspace_excludes_live_source":true,"real16gb_qualification":false})).map_err(|e|e.to_string())?)?;
    {
        let timeline = singleton
            .state
            .timeline
            .lock()
            .map_err(|_| "final family lock")?;
        let timeline = timeline.as_ref().ok_or("timeline absent")?;
        owner_save(root,"REPORT.json",&serde_json::to_vec_pretty(&json!({"passed":true,"cases":cases,"rejected_guards":rejected,"event":timeline.event as usize,"event_record_count":positive_events,"total_family_event_record_count":timeline.phases,"state_registry":registry.inventory(t),"prefill_stream":prefill_stream.handle as usize,"decode_stream":decode_stream.handle as usize,"legacy_default":legacy.handle as usize,"actual_uuid":actual_uuid,"compute":cc,"memory_before":before,"reserve_bytes":reserve,"graph_inventory":graph_inventory,"batch_bind_allocations":batch.allocation_bytes,"arithmetic":singleton.arithmetic_sha256,"producer_arithmetic":producer_arithmetic,"host_waits_between_positive_phases":0,"host_restores":0,"adopted":false,"ordinary_gate":"FAILED_UNCHANGED","performance":"NOT_MEASURED","pressure_calibration":"ACTUAL_LIVE_INPUT_SHORT_LONG_ADAPTIVE","native_windows":"OUTSTANDING"})).map_err(|e|e.to_string())?)?;
    } // Release the family inventory guard before dropping the family.
      // Source/graph drains are complete. Drop every owning state/workspace/stream lease before checking destruction.
    drop(baseline_a);
    drop(baseline_b);
    drop(fault_a);
    drop(fault_b);
    drop(other_batch);
    drop(other_singleton);
    drop(batch);
    drop(template);
    drop(singleton);
    drop(prefill_stream);
    if prefill_owner_weak.upgrade().is_some() {
        return Err("actual prefill stream owner leaked after all family/source leases".into());
    }
    owner_save(root,"prefill-stream-lifetime.json",&serde_json::to_vec_pretty(&json!({"actual_created_prefill_owner":true,"caller_owner_dropped_before_positive_phases":true,"retained_by_source_and_state_family":true,"released_after_sources_graph_and_all_state_leases":true,"new_streams_for_source_wrapper":0,"new_unsafe_thread_impl":true})).map_err(|e|e.to_string())?)?;
    Ok(())
}
#[test]
fn actual_native_model_transaction_kda() {
    if let Err(e) = actual_native_model_transaction_kda_run() {
        panic!("{e}")
    }
}
