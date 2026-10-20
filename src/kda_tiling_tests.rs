#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TilingPlan {
    #[serde(default)]
    session_source_identity: Option<OwnerId>,
    native: LongChronologyPlan,
    cuda_runtime: OwnerId,
    pressure_fractions: Vec<[usize; 2]>,
    output_reference: OwnerId,
    bf16_reference: OwnerId,
    prefix_reference: OwnerId,
    final_reference: OwnerId,
}
fn tiled_external_snapshot(
    o: &PreparedOwner,
    r: &crate::session_cache::SequenceStateRegistry,
    want: &OwnerSnapshot,
    label: &str,
) -> Result<OwnerSnapshot, String> {
    let actual = capture_owner(o, r)?;
    if actual.layout != want.layout {
        return Err(format!("{label}: external layout"));
    }
    owner_compare(
        &serde_json::to_vec(&actual.blobs).map_err(|e| e.to_string())?,
        &serde_json::to_vec(&want.blobs).map_err(|e| e.to_string())?,
        label,
    )?;
    Ok(actual)
}
fn tiled_capacity(
    heads: usize,
    dim: usize,
    rows: usize,
    free: usize,
    reserve: usize,
    device: &CudaDevice,
) -> Result<usize, String> {
    let pair_per_group = heads
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(16 * 16))
        .ok_or("tiling index denominator")?;
    let index_groups = (i32::MAX as usize)
        .checked_div(pair_per_group)
        .ok_or("zero geometry")?;
    let grid_groups = usize::try_from(
        device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
            .map_err(|e| e.to_string())?,
    )
    .map_err(|_| "grid bound")?;
    let max_rows = index_groups
        .min(grid_groups)
        .checked_mul(16)
        .and_then(|v| v.checked_sub(15))
        .ok_or("no signed ABI capacity")?;
    let available = free
        .checked_sub(reserve)
        .ok_or("configured reserve unavailable")?;
    let need = |n| -> Result<usize, String> {
        let a = PreparedLayout::checked(heads, dim, n)?.allocation_bytes;
        let b = KdaRawProducer::allocation_blueprint(heads, dim, n)?.2;
        a.checked_add(b)
            .ok_or("combined allocation blueprint".into())
    };
    if need(1)? > available {
        return Err("no safe single-row footprint; no fallback".into());
    }
    let (mut low, mut high) = (1usize, rows.min(max_rows));
    if high == 0 {
        return Err("zero request/capacity".into());
    }
    while low < high {
        let mid = low + (high - low + 1) / 2;
        if need(mid)? <= available {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}
fn calibrated_allocation_probe(stream:Arc<NativeStream>,device:Arc<CudaDevice>,kernels:Arc<KdaRawKernels>,p:&LongChronologyPlan,capacity:usize,reserve:usize,transport:&str,math:&str,a_log:&[u8],bias:&[u8],raw:&[Vec<u8>],expected:&[u8],expected_bf:&[u8])->Result<serde_json::Value,String>{
    let before=memory()?;
    let precision=crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?.ok_or("required native calibration precision window disabled")?;
    let owner=PreparedOwner::new_on_stream(device,stream.clone(),p.heads,p.dim,capacity,reserve,transport,math)?;
    let producer=KdaRawProducer::new(stream.clone(),p.heads,p.dim,capacity,reserve,p.lower,p.scale,p.epsilon,kernels,a_log,bias)?;
    let mut input=kda_raw_slice(raw,0,capacity,p.heads,p.dim)?;
    for(v,(desc,_))in input.iter_mut().zip(producer.raw.inventory()){v.resize(desc.storage_bytes,0);}
    kda_raw_upload(&producer.raw,&input)?;
    let resident=memory()?;
    {let lease=owner.arena.lease()?;producer.enqueue(0,capacity,&owner)?;owner.prefill_in(&lease,&producer.view(capacity)?,0,capacity)?;}
    stream.drain()?;long_check_cursor(&owner,capacity)?;
    let after_execute=memory()?;let endpoints_min=resident[0].min(after_execute[0]);
    let hd=p.heads.checked_mul(p.dim).ok_or("probe HD")?;
    owner_compare(&owner_array(&owner,"emitted",capacity*hd)?,&expected[..capacity*hd*4],"calibration authentic complete F32 prefix")?;
    owner_compare(&vector_bf16_array(&owner,capacity*hd)?,&expected_bf[..capacity*hd*2],"calibration authentic complete BF16 prefix")?;
    let precision_min=usize::try_from(precision.finish()?).map_err(|_|"precision min extent")?;
    let lowest=endpoints_min.min(precision_min);
    if lowest<reserve{return Err("allocation probe precision floor below configured safety".into());}
    let measured_bytes=before[0].checked_sub(lowest).ok_or("allocation probe free increased unexpectedly")?;
    let requested_bytes=owner.allocation_bytes.checked_add(producer.requested_bytes).ok_or("probe payload overflow")?;
    drop(producer);drop(owner);stream.drain()?;let cleanup=memory()?;
    Ok(json!({"capacity":capacity,"before":before,"resident":resident,"after_execute":after_execute,"lowest":lowest,"precision_min_free_bytes":precision_min,"measured_bytes":measured_bytes,"requested_bytes":requested_bytes,"post_cleanup":cleanup,"complete_prefix_outputs_exact":true}))
}
fn actual_native_runtime_tiled_kda_run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev kda-tiling-native".into());
    }
    let path = std::env::var("KRASIS_KDA_TILING_PLAN").map_err(|_| "frozen tiling plan")?;
    let output = std::env::var("KRASIS_KDA_TILING_OUTPUT").map_err(|_| "fresh tiling output")?;
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
    let params = [a_log.as_slice(), bias.as_slice()];
    owner_checked(&qp.cuda_runtime)?;
    let _runtime=unsafe{libloading::Library::new(&qp.cuda_runtime.path)}.map_err(|e|e.to_string())?;
    let device = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    let _precision_monitor=crate::vram_monitor::native_calibration_monitor(vec![device.ordinal() as i32],u64::try_from(reserve/(1024*1024)).map_err(|_|"monitor safety extent")?)?;
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
    let mut reports = Vec::new();
    for (num, fraction) in qp.pressure_fractions.iter().enumerate() {
        if fraction[1] == 0 || fraction[0] >= fraction[1] {
            return Err("explicit pressure fraction".into());
        }
        let dir = root.join(format!("pressure-{num}"));
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        let stream = NativeStream::create(device.clone())?;
        let before = memory()?;
        // Modules, real parameters, persistent and graph singleton all precede measurement/planning.
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
        let kernels = KdaRawKernels::load(device.clone(), &producer_ptx)?;
        let one = KdaRawProducer::new(
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
        let mut singleton_registry = crate::session_cache::SequenceStateRegistry::default();
        singleton.register(&mut singleton_registry, 0)?;
        let graph = kda_raw_capture(&singleton, &one)?;
        let inventory = kda_raw_graph_inventory(&singleton, &one, &graph)?;
        kda_raw_upload(&one.raw,&kda_raw_slice(&raw,0,1,h,d)?)?;
        {let lease=singleton.arena.lease()?;singleton.launch_graph_in(&lease,&graph)?;}
        stream.drain()?;singleton.reset()?;stream.drain()?;
        let free_before_pressure = memory()?[0];
        let pressure_bytes = free_before_pressure
            .checked_sub(reserve)
            .ok_or("reserve before pressure")?
            .checked_mul(fraction[0])
            .and_then(|v| v.checked_div(fraction[1]))
            .ok_or("pressure extent")?;
        let pressure = if pressure_bytes > 0 {
            Some(native_zeroed::<u8>(&stream, pressure_bytes)?)
        } else {
            None
        };
        stream.drain()?;
        let target_free=memory()?[0];
        let target_budget=target_free.checked_sub(reserve).ok_or("target calibration safety")?;
        drop(pressure);stream.drain()?;
        // Both short and long are mandatory even when the long footprint already fits.
        let maximum=tiled_capacity(h,d,t,memory()?[0],reserve,&device)?;
        let mut probes=std::collections::BTreeMap::<usize,serde_json::Value>::new();
        for n in [1,maximum] {
            if probes.contains_key(&n){continue;}
            let v=calibrated_allocation_probe(stream.clone(),device.clone(),kernels.clone(),p,n,reserve,&transport,&math,&a_log,&bias,&raw,&expected,&expected_bf)?;
            probes.insert(n,v);owner_save(&dir,&format!("allocation-calibration-{n}.json"),&serde_json::to_vec_pretty(probes.get(&n).ok_or("recorded short/long probe absent")?).map_err(|e|e.to_string())?)?;
        }
        let cost=|v:&serde_json::Value|->Result<usize,String>{usize::try_from(v["measured_bytes"].as_u64().ok_or("missing real measured footprint")?).map_err(|_|"measured footprint size".into())};
        if cost(probes.get(&1).ok_or("short probe absent")?)?>target_budget{return Err("measured short footprint exceeds target; no fallback".into());}
        let mut low=1usize;let mut high=maximum;
        if cost(probes.get(&maximum).ok_or("long probe absent")?)?<=target_budget {low=maximum;}
        while low<high {
            let mid=low+(high-low+1)/2;
            if !probes.contains_key(&mid){let v=calibrated_allocation_probe(stream.clone(),device.clone(),kernels.clone(),p,mid,reserve,&transport,&math,&a_log,&bias,&raw,&expected,&expected_bf)?;probes.insert(mid,v);owner_save(&dir,&format!("allocation-calibration-{mid}.json"),&serde_json::to_vec_pretty(probes.get(&mid).ok_or("recorded adaptive probe absent")?).map_err(|e|e.to_string())?)?;}
            if cost(probes.get(&mid).ok_or("probe missing")?)?<=target_budget{low=mid;}else{high=mid-1;}
        }
        owner_save(&dir,"allocation-calibration.json",&serde_json::to_vec_pretty(&probes).map_err(|e|e.to_string())?)?;
        let capacity=low;
        let measured_selected=cost(probes.get(&capacity).ok_or("selected successful probe absent")?)?;
        let final_pressure_bytes=memory()?[0].checked_sub(reserve).ok_or("postcleanup reserve")?.checked_mul(fraction[0]).and_then(|v|v.checked_div(fraction[1])).ok_or("final pressure extent")?;
        let pressure=if final_pressure_bytes>0{Some(native_zeroed::<u8>(&stream,final_pressure_bytes)?)}else{None};stream.drain()?;
        let free_after_pressure=memory()?[0];
        if measured_selected>free_after_pressure.checked_sub(reserve).ok_or("final pressure reserve")?{return Err("actual final budget invalidates measured selected footprint; no fallback".into());}
        let case_window=crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?.ok_or("required final case precision window disabled")?;
        owner_save(&dir,"calibrated-capacity.json",&serde_json::to_vec_pretty(&json!({"target_free":target_free,"target_budget":target_budget,"short":1,"long":maximum,"capacity":capacity,"selected_measured_bytes":measured_selected,"final_pressure_bytes":final_pressure_bytes,"actual_final_free":free_after_pressure,"reserve_bytes":reserve,"probes":probes})).map_err(|e|e.to_string())?)?;

        let blueprint = PreparedLayout::checked(h, d, capacity)?;
        let producer_blueprint = KdaRawProducer::allocation_blueprint(h, d, capacity)?.2;
        owner_save(&dir,"before-batch-allocation.json",&serde_json::to_vec_pretty(&json!({"pressure_fraction":fraction,"pressure_bytes":final_pressure_bytes,"calibration_measured_selected_bytes":measured_selected,"memory_before":before,"free_before_pressure":free_before_pressure,"free_after_pressure":free_after_pressure,"capacity":capacity,"owner_blueprint_bytes":blueprint.allocation_bytes,"producer_blueprint_bytes":producer_blueprint,"reserve_bytes":reserve,"actual_free":memory()?})).map_err(|e|e.to_string())?)?;
        let batch = PreparedOwner::new_on_stream(
            device.clone(),
            stream.clone(),
            h,
            d,
            capacity,
            reserve,
            &transport,
            &math,
        )?;
        owner_save(&dir,"after-batch-allocation.json",&serde_json::to_vec_pretty(&json!({"actual_free":memory()?,"actual_owner_requested_bytes":batch.allocation_bytes,"producer_blueprint_bytes":producer_blueprint,"reserve_bytes":reserve})).map_err(|e|e.to_string())?)?;
        let producer = KdaRawProducer::new(
            stream.clone(),
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
        if batch.allocation_bytes != blueprint.allocation_bytes
            || producer.requested_bytes != producer_blueprint
        {
            return Err("shared constructor blueprint mismatch".into());
        }
        let allocation_owner = batch;
        let batch = PreparedOwner::bind_persistent(&allocation_owner,&singleton)?;
        let mut registry = crate::session_cache::SequenceStateRegistry::default();
        batch.register(&mut registry, 0)?;
        if registry.allocations()!=singleton_registry.allocations(){return Err("binding persistent registry pointers/layout differ".into())}
        owner_save(&dir,"bound-registry.json",&serde_json::to_vec_pretty(&registry.inventory(t)).map_err(|e|e.to_string())?)?;
        let mut retained = Vec::new();
        let mut emitted = Vec::with_capacity(expected.len());
        let mut bf16 = Vec::with_capacity(expected_bf.len());
        let mut windows = Vec::new();
        let mut at = 0usize;
        let mut own_prefix = None;
        while at < start {
            let stop = if at < start { start } else { t };
            let count = (stop - at).min(capacity);
            let before_window = memory()?;
            let mut slice = kda_raw_slice(&raw, at, count, h, d)?;
            // Storage capacity is fixed; inactive upload rows never enter view(count).
            for (v, (descriptor, _)) in slice.iter_mut().zip(producer.raw.inventory()) {
                v.resize(descriptor.storage_bytes, 0);
            }
            kda_raw_upload(&producer.raw, &slice)?;
            long_parameters_save(&dir, &format!("at{at}"), &producer, &params, &mut retained)?;
            {
                let lease = batch.arena.lease()?;
                producer.enqueue(0, count, &batch)?;
                batch.prefill_in(&lease, &producer.view(count)?, at as u64, count)?;
            }
            stream.drain()?;
            long_check_cursor(&batch, at + count)?;
            let after_window = memory()?;
            if after_window[0] < reserve {
                return Err("measured tile floor below safety".into());
            }
            let v = owner_array(&batch, "emitted", count * hd)?;
            let b = vector_bf16_array(&batch, count * hd)?;
            owner_compare(
                &v,
                &expected[at * hd * 4..(at + count) * hd * 4],
                "complete window F32",
            )?;
            owner_compare(
                &b,
                &expected_bf[at * hd * 2..(at + count) * hd * 2],
                "complete window BF16",
            )?;
            emitted.extend_from_slice(&v);
            bf16.extend_from_slice(&b);
            windows.push(json!({"absolute_start":at,"rows":count,"free_before":before_window,"free_after":after_window}));
            at += count;
            if at == start {
                let s = tiled_external_snapshot(
                    &batch,
                    &registry,
                    &prefix,
                    "fresh prefix external bytes",
                )?;
                long_snapshot_save(&dir, "prefix-own-snapshot.json", &s)?;
                own_prefix = Some(s);
            }
        }
        let _own_prefix = own_prefix.ok_or("actual prefix required")?;
        for row in start..t {
            kda_raw_upload(&one.raw,&kda_raw_slice(&raw,row,1,h,d)?)?;
            { let lease=singleton.arena.lease()?; singleton.launch_graph_in(&lease,&graph)?; }
            stream.drain()?; long_check_cursor(&singleton,row+1)?;
            let v=owner_array(&singleton,"emitted",hd)?; let b=vector_bf16_array(&singleton,hd)?;
            owner_compare(&v,&expected[row*hd*4..(row+1)*hd*4],"bound graph F32")?;
            owner_compare(&b,&expected_bf[row*hd*2..(row+1)*hd*2],"bound graph BF16")?;
            emitted.extend_from_slice(&v);bf16.extend_from_slice(&b);
        }
        owner_compare(&emitted,&expected,"complete bound whole F32")?;
        owner_compare(&bf16,&expected_bf,"complete bound whole BF16")?;
        owner_save(&dir,"whole-output.f32",&emitted)?;owner_save(&dir,"whole-output.bf16",&bf16)?;
        let graph_final = tiled_external_snapshot(
            &singleton,
            &singleton_registry,
            &final_state,
            "graph final external bytes",
        )?;
        long_snapshot_save(&dir, "graph-final-own-snapshot.json", &graph_final)?;
        let case_min_free=usize::try_from(case_window.finish()?).map_err(|_|"case min extent")?;
        if case_min_free<reserve{return Err("actual final case precision floor below safety".into());}
        reports.push(json!({"pressure_fraction":fraction,"case_precision_min_free_bytes":case_min_free,"initial_pressure_requested_bytes":pressure_bytes,"pressure_bytes":final_pressure_bytes,"calibration_measured_selected_bytes":measured_selected,"memory_before":before,"free_before_pressure":free_before_pressure,"free_after_pressure":free_after_pressure,"capacity":capacity,"actual_owner_bytes":allocation_owner.allocation_bytes,"bound_owner_new_allocations":batch.allocation_bytes,"persistent_direct_gpu_handoff":true,"actual_producer_bytes":producer.requested_bytes,"reserve_bytes":reserve,"windows":windows,"graph_inventory":inventory,"parameters":retained,"arithmetic_identity":batch.arithmetic_sha256,"ordinary_gate":"FAILED_UNCHANGED","pressure_is_real16gb_qualification":false}));
        drop(pressure);
    }
    for x in &p.admission {
        owner_checked(x)?;
    }
    owner_save(root,"REPORT.json",&serde_json::to_vec_pretty(&json!({"passed":true,"actual_uuid":actual_uuid,"compute":cc,"reports":reports,"instrumentation":"actual free memory + complete readbacks; no speed readings","adopted":false})).map_err(|e|e.to_string())?)?;
    Ok(())
}
#[test]
fn actual_native_runtime_tiled_kda() {
    actual_native_runtime_tiled_kda_run().unwrap();
}
