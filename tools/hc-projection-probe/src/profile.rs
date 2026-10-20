// Diagnostic GPU event profiling only; no serving dispatch or budget changes.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeBuild {
    source: Identity,
    header: Identity,
    nvcc: Identity,
    flags: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    plan: Plan,
    native_ptx: Identity,
    native_build: Identity,
    scalar_reference: Identity,
    mult: usize,
    samples: usize,
    target_ms: f32,
}
struct Events {
    device: Arc<CudaDevice>,
    start: sys::CUevent,
    end: sys::CUevent,
}
impl Events {
    fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        let mut owner = Self {
            device,
            start: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
        };
        unsafe {
            sys::lib()
                .cuEventCreate(&mut owner.start, 0)
                .result()
                .map_err(|e| format!("create start event: {e}"))?;
            sys::lib()
                .cuEventCreate(&mut owner.end, 0)
                .result()
                .map_err(|e| format!("create end event: {e}"))?;
        }
        Ok(owner)
    }
    fn sample(
        &self,
        repeats: usize,
        enqueue: &impl Fn() -> Result<(), String>,
    ) -> Result<f32, String> {
        graph_owner::completed_graph_launch(&self.device, || {
            unsafe {
                sys::lib()
                    .cuEventRecordWithFlags(
                        self.start,
                        *self.device.cu_stream(),
                        sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL as u32,
                    )
                    .result()
                    .map_err(|e| format!("capture start event: {e}"))?;
            }
            for _ in 0..repeats {
                enqueue()?;
            }
            unsafe {
                sys::lib()
                    .cuEventRecordWithFlags(
                        self.end,
                        *self.device.cu_stream(),
                        sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL as u32,
                    )
                    .result()
                    .map_err(|e| format!("capture end event: {e}"))?;
            }
            Ok(())
        })?;
        let mut ms = 0.0;
        unsafe {
            sys::lib()
                .cuEventElapsedTime(&mut ms, self.start, self.end)
                .result()
                .map_err(|e| format!("elapsed time after completed graph replay: {e}"))?;
        }
        if !ms.is_finite() || ms < 0.0 {
            return Err("invalid CUDA event interval".into());
        }
        Ok(ms)
    }
}
impl Drop for Events {
    fn drop(&mut self) {
        if let Err(e) = self.device.synchronize() {
            eprintln!("FATAL: event owner completion unknown: {e}");
            std::process::abort();
        }
        unsafe {
            for e in [self.start, self.end] {
                if !e.is_null() {
                    if let Err(e) = sys::lib().cuEventDestroy_v2(e).result() {
                        eprintln!("FATAL: event destruction failed: {e}");
                        std::process::abort();
                    }
                }
            }
        }
    }
}
pub(super) fn compile(plan: &Path, out: &Path) -> Result<Value, String> {
    let p: NativeBuild = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    p.source.read()?;
    p.header.read()?;
    p.nvcc.read()?;
    if p.flags
        != [
            "-ptx",
            "-allow-unsupported-compiler",
            "-arch=sm_80",
            "-O3",
            "--use_fast_math",
        ]
    {
        return Err("native profiling flags differ explicit fast-math contract".into());
    }
    let r = Command::new(&p.nvcc.path)
        .args(&p.flags)
        .arg("-o")
        .arg(out.join("operator.ptx"))
        .arg(&p.source.path)
        .output()
        .map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stdout"), r.stdout).map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stderr"), r.stderr).map_err(|e| e.to_string())?;
    if !r.status.success() {
        return Err(format!("native profiling compile failed {}", r.status));
    }
    p.source.read()?;
    p.header.read()?;
    p.nvcc.read()?;
    let ptx = Identity::of(&out.join("operator.ptx"))?;
    Ok(
        json!({"passed":true,"source":p.source,"header":p.header,"nvcc":p.nvcc,"flags":p.flags,"ptx":ptx,"scope":"isolated unchanged native scalar source with explicit NVCC fast flags; not identical installed NVRTC module or fullmodel performance"}),
    )
}
pub(super) fn run(plan: &Path, out: &Path, kind: Kind) -> Result<Value, String> {
    let q: Profile = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let p = &q.plan;
    p.source.read()?;
    p.input_manifest.read()?;
    p.state.read()?;
    p.inverse.read()?;
    p.weight.read()?;
    p.reference_f32.read()?;
    if q.mult == 0
        || p.cols % q.mult != 0
        || p.rows
            != q.mult
                .checked_mul(q.mult.checked_add(2).ok_or("mult overflow")?)
                .ok_or("mix overflow")?
        || q.samples == 0
        || !q.target_ms.is_finite()
        || q.target_ms <= 0.0
    {
        return Err("invalid original HC/profile geometry".into());
    }
    let native: Value =
        serde_json::from_slice(&q.native_build.read()?).map_err(|e| e.to_string())?;
    if native["passed"] != true || native["ptx"]["sha256"] != q.native_ptx.sha256 {
        return Err("native build identity differs".into());
    }
    let scalar = f32s(&q.scalar_reference.read()?)?;
    if scalar.len() != p.rows {
        return Err("actual native scalar reference extent differs".into());
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
        return Err("profile actual GPU UUID differs".into());
    }
    dev.load_ptx(
        Ptx::from_src(String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?),
        "shared",
        &[kind.name()],
    )
    .map_err(|e| e.to_string())?;
    dev.load_ptx(
        Ptx::from_src(String::from_utf8(q.native_ptx.read()?).map_err(|e| e.to_string())?),
        "native",
        &["deepseek_v4_hc_project_kernel"],
    )
    .map_err(|e| e.to_string())?;
    let common = dev
        .get_func("shared", kind.name())
        .ok_or("common kernel missing")?;
    let old = dev
        .get_func("native", "deepseek_v4_hc_project_kernel")
        .ok_or("native kernel missing")?;
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
        return Err("profile dimensions or owned tensor extents invalid".into());
    }
    use sys::CUdevice_attribute as A;
    let attribute = |a| dev.attribute(a).map_err(|e| e.to_string());
    if p.tokens > attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize
        || p.rows > attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize
        || p.tokens.div_ceil(32) > attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize
        || p.rows.div_ceil(32) > attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize
    {
        return Err("profile grid exceeds measured device limits".into());
    }
    let owner = Buffers {
        device: dev.clone(),
        state: dev.htod_copy(state).map_err(|e| e.to_string())?,
        inverse: dev.htod_copy(inverse).map_err(|e| e.to_string())?,
        weight: dev.htod_copy(weight).map_err(|e| e.to_string())?,
        output: dev
            .alloc_zeros::<f32>(mul(p.tokens, p.rows)?)
            .map_err(|e| e.to_string())?,
    };
    if cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0
        < p.safety_margin_bytes
    {
        return Err("profile freeVRAM below requested policy".into());
    }
    let native_launch = |n: usize| -> Result<(), String> {
        let threads = p.cols.min(1024).next_power_of_two().max(32) as u32;
        unsafe {
            old.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (p.rows as u32, n as u32, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    },
                    (
                        *owner.output.device_ptr(),
                        *owner.state.device_ptr(),
                        *owner.weight.device_ptr(),
                        *owner.inverse.device_ptr(),
                        (p.cols / q.mult) as i32,
                        q.mult as i32,
                        p.rows as i32,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let max_threads = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        .map_err(|e| e.to_string())? as usize;
    let limit = 1usize << (usize::BITS - 1 - max_threads.leading_zeros());
    let block_threads = if matches!(kind, Kind::Block64) {
        p.cols.min(limit).next_power_of_two().max(32)
    } else {
        128
    };
    let block_shared = if matches!(kind, Kind::Block64) {
        block_threads * std::mem::size_of::<f64>()
    } else {
        0
    };
    let common_launch = |n: usize| -> Result<(), String> {
        unsafe {
            common
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (
                            n.div_ceil(kind.tile()) as u32,
                            if matches!(kind, Kind::Block64) {
                                p.rows as u32
                            } else {
                                p.rows.div_ceil(kind.tile()) as u32
                            },
                            1,
                        ),
                        block_dim: (block_threads as u32, 1, 1),
                        shared_mem_bytes: block_shared as u32,
                    },
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
        Ok(())
    };
    eprintln!("profile phase=actual-native-reference");
    native_launch(1)?;
    dev.synchronize().map_err(|e| e.to_string())?;
    let got = dev
        .dtoh_sync_copy(&owner.output)
        .map_err(|e| e.to_string())?;
    if !got[..p.rows]
        .iter()
        .zip(&scalar)
        .all(|(x, y)| x.to_bits() == y.to_bits())
    {
        write_f32(&out.join("native-scalar-mismatch.f32"), &got[..p.rows])?;
        return Err(
            "isolated native scalar differs actual34reference; cost comparison not admitted".into(),
        );
    }
    let mut results = Vec::new();
    for (label, n, is_native) in [
        ("common-single", 1, false),
        ("native-scalar-single", 1, true),
        ("common-batch", p.tokens, false),
        ("native-scalar-batch", p.tokens, true),
    ] {
        let enqueue = || {
            if is_native {
                native_launch(n)
            } else {
                common_launch(n)
            }
        };
        eprintln!("profile phase=warmup label={label} tokens={n}");
        enqueue()?;
        dev.synchronize().map_err(|e| e.to_string())?;
        let events = Events::new(dev.clone())?;
        let mut repeats = 1usize;
        loop {
            eprintln!("profile phase=adaptive-events label={label} repeats={repeats}");
            let ms = events.sample(repeats, &enqueue)?;
            if ms >= q.target_ms {
                break;
            }
            repeats = repeats
                .checked_mul(2)
                .filter(|&n| n <= 1048576)
                .ok_or("profile resolution limit reached; no fabricated timing")?;
        }
        let mut readings = Vec::new();
        for _ in 0..q.samples {
            let ms = events.sample(repeats, &enqueue)?;
            readings.push(ms as f64 / repeats as f64);
        }
        let mut sorted = readings.clone();
        sorted.sort_by(f64::total_cmp);
        results.push(json!({"label":label,"tokensPerCall":n,"repeatsPerSample":repeats,"gpuEventMsPerCall":readings,"medianMs":sorted[sorted.len()/2],"scope":if label=="native-scalar-batch"{"original scalar source repeated acrossbatch, NOT optimized nativeprefill GEMM"}else{"isolated captured CUDA graph GPU event interval peroperatorcall, not HTTP/model/speed-test"}}));
    }
    Ok(
        json!({"passed":true,"profile":true,"kernel":kind.name(),"actualNativeScalarRowExact":true,"gpuUuid":actual,"gpuName":dev.name().map_err(|e|e.to_string())?,"rows":p.rows,"cols":p.cols,"tokens":p.tokens,"results":results,"native":q.native_build,"common":p.ptx,"binary":Identity::of(&std::env::current_exe().map_err(|e|e.to_string())?)?,"scope":"diagnostic cost data for optimization; isolated NVCC native scalar baseline, optimized nativeprefill comparison/installedmodel performance pending"}),
    )
}
