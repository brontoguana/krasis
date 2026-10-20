use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PhasePlan {
    plan: Plan,
    timed_source: Identity,
    timed_ptx: Identity,
    timed_build: Identity,
}
struct PhaseOwner {
    buffers: Buffers,
    clocks: CudaSlice<u64>,
}
impl Drop for PhaseOwner {
    fn drop(&mut self) {
        if let Err(e) = self.buffers.device.synchronize() {
            eprintln!("FATAL: phase completion unknown; all operand/counter owners retained: {e}");
            std::process::abort();
        }
    }
}
fn attributes(f: &cudarc::driver::CudaFunction) -> Result<Value, String> {
    let mut out = serde_json::Map::new();
    for (name, key) in [
        (
            "registers",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
        ),
        (
            "localBytes",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        ),
        (
            "staticSharedBytes",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        ),
        (
            "maxThreads",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ),
    ] {
        let mut v = 0;
        unsafe {
            sys::lib()
                .cuFuncGetAttribute(&mut v, key, f.cu_function())
                .result()
                .map_err(|e| e.to_string())?;
        }
        out.insert(name.into(), json!(v));
    }
    Ok(Value::Object(out))
}
pub(super) fn run(plan: &Path, out: &Path) -> Result<Value, String> {
    let q: PhasePlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let p = &q.plan;
    p.source.read()?;
    q.timed_source.read()?;
    let build: Value = serde_json::from_slice(&q.timed_build.read()?).map_err(|e| e.to_string())?;
    if build["passed"] != true
        || build["ptx"]["sha256"] != q.timed_ptx.sha256
        || build["source"]["sha256"] != q.timed_source.sha256
    {
        return Err("timed shader build binding invalid".into());
    }
    let state = u16s(&p.state.read()?)?;
    let inverse = f32s(&p.inverse.read()?)?;
    let weight = f32s(&p.weight.read()?)?;
    if p.rows == 0
        || p.cols == 0
        || p.tokens == 0
        || p.rows > i32::MAX as usize
        || p.cols > i32::MAX as usize
        || state.len() != mul(p.tokens, p.cols)?
        || inverse.len() != p.tokens
        || weight.len() != mul(p.rows, p.cols)?
    {
        return Err("phase probe dimensions/owned extents invalid".into());
    }
    let dev = CudaDevice::new_with_stream(p.gpu_ordinal).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe {
        sys::lib()
            .cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device())
            .result()
            .map_err(|e| e.to_string())?;
    }
    let hex = uuid
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    if actual != p.gpu_uuid {
        return Err("phase actualGPU differs plan".into());
    }
    for (module, identity) in [("baseline", &p.ptx), ("timed", &q.timed_ptx)] {
        dev.load_ptx(
            Ptx::from_src(String::from_utf8(identity.read()?).map_err(|e| e.to_string())?),
            module,
            &["ordered_hc_f32_tf32x9"],
        )
        .map_err(|e| e.to_string())?;
    }
    let baseline = dev
        .get_func("baseline", "ordered_hc_f32_tf32x9")
        .ok_or("baseline shader absent")?;
    let timed = dev
        .get_func("timed", "ordered_hc_f32_tf32x9")
        .ok_or("timed shader absent")?;
    let baseline_resources = attributes(&baseline)?;
    let timed_resources = attributes(&timed)?;
    let owner = Buffers {
        device: dev.clone(),
        state: dev.htod_copy(state).map_err(|e| e.to_string())?,
        inverse: dev.htod_copy(inverse).map_err(|e| e.to_string())?,
        weight: dev.htod_copy(weight).map_err(|e| e.to_string())?,
        output: dev
            .alloc_zeros::<f32>(mul(p.tokens, p.rows)?)
            .map_err(|e| e.to_string())?,
    };
    let blocks = mul(p.tokens.div_ceil(32), p.rows.div_ceil(32))?;
    let clocks = dev
        .alloc_zeros::<u64>(mul(blocks, 5)?)
        .map_err(|e| e.to_string())?;
    let owned = PhaseOwner {
        buffers: owner,
        clocks,
    };
    let owner = &owned.buffers;
    let clocks = &owned.clocks;
    if cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0
        < p.safety_margin_bytes
    {
        return Err("phase freeVRAM below requested safety".into());
    }
    let mut points = Vec::new();
    for n in [1, p.tokens] {
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(32) as u32, p.rows.div_ceil(32) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            baseline
                .clone()
                .launch(
                    cfg,
                    (
                        *owner.state.device_ptr(),
                        *owner.inverse.device_ptr(),
                        *owner.weight.device_ptr(),
                        *owner.output.device_ptr(),
                        p.rows as i32,
                        p.cols as i32,
                        n as u64,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;
        let original = dev
            .dtoh_sync_copy(&owner.output)
            .map_err(|e| e.to_string())?;
        unsafe {
            timed
                .clone()
                .launch(
                    cfg,
                    (
                        *owner.state.device_ptr(),
                        *owner.inverse.device_ptr(),
                        *owner.weight.device_ptr(),
                        *owner.output.device_ptr(),
                        p.rows as i32,
                        p.cols as i32,
                        n as u64,
                        *clocks.device_ptr(),
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;
        let observed = dev
            .dtoh_sync_copy(&owner.output)
            .map_err(|e| e.to_string())?;
        let count = mul(n, p.rows)?;
        if !original[..count]
            .iter()
            .zip(&observed[..count])
            .all(|(x, y)| x.to_bits() == y.to_bits())
        {
            return Err("instrumentation changed actual arithmetic result".into());
        }
        let data = dev.dtoh_sync_copy(clocks).map_err(|e| e.to_string())?;
        let active_blocks = mul(n.div_ceil(32), p.rows.div_ceil(32))?;
        if data.len() < mul(active_blocks, 5)?
            || data[..active_blocks * 5]
                .chunks_exact(5)
                .any(|v| v[0] == 0 || v[1] == 0 || v[2] == 0 || v[3] == 0 || v[4] == 0)
        {
            return Err("missing/zero requested phase measurements".into());
        }
        points.push(json!({"tokens":n,"resultBitsExact":true,"blocks":data[..active_blocks*5].chunks_exact(5).enumerate().map(|(block,v)|json!({"block":block,"stagingCycles":v[0],"mmaCarryCycles":v[1],"outputCycles":v[2],"mmaLoadCyclesThread0":v[3],"carryCyclesThread0":v[4]})).collect::<Vec<_>>()}));
    }
    dev.synchronize().map_err(|e| e.to_string())?;
    Ok(
        json!({"passed":true,"gpuUuid":actual,"gpuName":dev.name().map_err(|e|e.to_string())?,"baselineResources":baseline_resources,"timedResources":timed_resources,"points":points,"baselinePtx":p.ptx,"timedPtx":q.timed_ptx,"scope":"clock64stagingvscombinedWMMA/carryphaseandoutput, inclusivebarriers; perCTAphasecyclesandthread0subphasecyclesnotkernelms; intra-clocks/resources perturbation explicit; instrumentationresourcesexplicitlyrecorded"}),
    )
}
