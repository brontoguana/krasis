use super::*;
include!("native_abi.rs");
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CostPlan {
    plan: Plan,
    native: Identity,
    warmups: usize,
    samples: usize,
    shapes: Vec<usize>,
    #[serde(default)]
    phase: bool,
    #[serde(default)]
    phase_baselines: Vec<(Identity, Identity)>,
}
struct Events {
    device: Arc<CudaDevice>,
    start: sys::CUevent,
    end: sys::CUevent,
}
impl Events {
    fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        let mut e = Self {
            device,
            start: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
        };
        unsafe {
            sys::lib()
                .cuEventCreate(&mut e.start, 0)
                .result()
                .map_err(|e| e.to_string())?;
            sys::lib()
                .cuEventCreate(&mut e.end, 0)
                .result()
                .map_err(|e| e.to_string())?
        }
        Ok(e)
    }
    fn sample(&self, f: &impl Fn() -> Result<(), String>) -> Result<Value, String> {
        unsafe {
            sys::lib()
                .cuEventRecord(self.start, *self.device.cu_stream())
                .result()
                .map_err(|e| e.to_string())?
        }
        let t = std::time::Instant::now();
        f()?;
        let cpu = t.elapsed().as_secs_f64() * 1e6;
        unsafe {
            sys::lib()
                .cuEventRecord(self.end, *self.device.cu_stream())
                .result()
                .map_err(|e| e.to_string())?;
            sys::lib()
                .cuEventSynchronize(self.end)
                .result()
                .map_err(|e| e.to_string())?
        }
        let mut ms = 0.;
        unsafe {
            sys::lib()
                .cuEventElapsedTime(&mut ms, self.start, self.end)
                .result()
                .map_err(|e| e.to_string())?
        }
        if !ms.is_finite() || ms < 0. {
            return Err("invalid measured interval".into());
        }
        Ok(json!({"cudaStreamMicroseconds":ms as f64*1000.,"rustEnqueueMicroseconds":cpu}))
    }
}
impl Drop for Events {
    fn drop(&mut self) {
        if let Err(e) = self.device.synchronize() {
            eprintln!("FATAL: event completion unknown: {e}");
            std::process::abort()
        }
        unsafe {
            for event in [self.start, self.end] {
                if !event.is_null() {
                    if let Err(e) = sys::lib().cuEventDestroy_v2(event).result() {
                        eprintln!("event teardown error: {e}")
                    }
                }
            }
        }
    }
}
struct Scratch {
    device: Arc<CudaDevice>,
    ctmp: CudaSlice<f32>,
    workspace: CudaSlice<i32>,
    phase: Option<CudaSlice<u64>>,
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(e) = self.device.synchronize() {
            eprintln!("FATAL: scratch completion unknown: {e}");
            std::process::abort()
        }
    }
}
pub(super) fn execute(path: &Path, out: &Path) -> Result<Value, String> {
    let cp: CostPlan = read_json(path)?;
    let plan = &cp.plan;
    if cp.warmups != 8
        || cp.samples != 32
        || cp.shapes.is_empty()
        || cp.shapes.iter().any(|&m| m == 0 || m > i32::MAX as usize)
    {
        return Err("cost collection contract invalid".into());
    }
    plan.source.read()?;
    plan.ptx.read()?;
    cp.native.read()?;
    let build: Value = serde_json::from_slice(&plan.build.read()?).map_err(|e| e.to_string())?;
    if build["passed"] != true
        || build["source"] != serde_json::to_value(&plan.source).unwrap()
        || build["ptx"] != serde_json::to_value(&plan.ptx).unwrap()
    {
        return Err("build binding mismatch".into());
    }
    let dev = CudaDevice::new_with_stream(plan.gpu_ordinal).map_err(|e| e.to_string())?;
    let mut uuid = sys::CUuuid { bytes: [0; 16] };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let h = uuid
        .bytes
        .iter()
        .map(|v| format!("{:02x}", *v as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    );
    if actual != plan.gpu_uuid {
        return Err("GPU UUID mismatch".into());
    }
    let lib = unsafe { libloading::Library::new(&cp.native.path) }.map_err(|e| e.to_string())?;
    let native: MarlinMmFn = unsafe {
        *lib.get::<MarlinMmFn>(b"krasis_marlin_mm_bf16\0")
            .map_err(|e| e.to_string())?
    };
    dev.load_ptx(
        String::from_utf8(plan.ptx.read()?)
            .map_err(|e| e.to_string())?
            .into(),
        "marlin_cost",
        &["marlin_projection_tensorcomp"],
    )
    .map_err(|e| e.to_string())?;
    let fun = dev
        .get_func("marlin_cost", "marlin_projection_tensorcomp")
        .ok_or("entry missing")?;
    let mut sms = 0;
    unsafe {
        sys::lib().cuDeviceGetAttribute(
            &mut sms,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            *dev.cu_device(),
        )
    }
    .result()
    .map_err(|e| e.to_string())?;
    if sms <= 0 {
        return Err("invalid runtime SM count".into());
    }
    let events = Events::new(dev.clone())?;
    let max_m = *cp.shapes.iter().max().unwrap();
    let mut readings = Vec::new();
    for (case_index, case) in plan.cases.iter().enumerate() {
        if case.n == 0
            || case.k == 0
            || case.n % 64 != 0
            || case.k % 16 != 0
            || case.n > i32::MAX as usize
            || case.k > i32::MAX as usize
            || ![4, 8].contains(&case.bits)
            || case.group == 0
            || case.group > case.k
        {
            return Err("invalid actual geometry".into());
        }
        let all = case.input.read()?;
        let end = case
            .input_offset
            .checked_add(mul(case.k, 2)?)
            .ok_or("input offset overflow")?;
        if end > all.len() {
            return Err("input range outside allocation".into());
        }
        let x = u16s(&all[case.input_offset..end])?;
        let packed = case.packed.read()?;
        let scales = case.scales.read()?;
        if packed.len() != mul(mul(case.n, case.k)?, case.bits)? / 8
            || scales.len() != mul(mul(case.k.div_ceil(case.group), case.n)?, 2)?
        {
            return Err("actual Marlin payload geometry mismatch".into());
        }
        let (iw, is) = inverse_perms(case.bits, case.group < case.k);
        let count = mul(max_m, case.n)?;
        let owner = Owners {
            device: dev.clone(),
            packed: dev
                .htod_copy(
                    packed
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
                .map_err(|e| e.to_string())?,
            scales: dev.htod_copy(u16s(&scales)?).map_err(|e| e.to_string())?,
            iw: dev.htod_copy(iw).map_err(|e| e.to_string())?,
            is: dev.htod_copy(is.clone()).map_err(|e| e.to_string())?,
            input: dev.htod_copy(x.repeat(max_m)).map_err(|e| e.to_string())?,
            output: dev.alloc_zeros::<u16>(count).map_err(|e| e.to_string())?,
            f32out: dev.alloc_zeros::<f32>(count).map_err(|e| e.to_string())?,
        };
        let scratch = Scratch {
            phase: if cp.phase {
                Some(
                    dev.alloc_zeros::<u64>(mul(mul(max_m.div_ceil(32), case.n.div_ceil(64))?, 4)?)
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            },
            device: dev.clone(),
            ctmp: dev.alloc_zeros::<f32>(count).map_err(|e| e.to_string())?,
            workspace: dev
                .alloc_zeros::<i32>(sms as usize * 4)
                .map_err(|e| e.to_string())?,
        };
        dev.synchronize().map_err(|e| e.to_string())?;
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < plan.safety_margin_bytes {
            return Err("measured free below safety policy".into());
        }
        for &tokens in &cp.shapes {
            let common = || -> Result<(), String> {
                let mut pw = *owner.packed.device_ptr();
                let mut ps = *owner.scales.device_ptr();
                let mut piw = *owner.iw.device_ptr();
                let mut pis = *owner.is.device_ptr();
                let mut px = *owner.input.device_ptr();
                let mut py = *owner.output.device_ptr();
                let mut pf = *owner.f32out.device_ptr();
                let mut n = case.n as i32;
                let mut k = case.k as i32;
                let mut g = case.group as i32;
                let mut sl = is.len() as i32;
                let mut bits = case.bits as i32;
                let mut m = tokens as u64;
                let mut phase_ptr = scratch.phase.as_ref().map(|b| *b.device_ptr()).unwrap_or(0);
                let mut params = [
                    (&mut pw as *mut u64).cast(),
                    (&mut ps as *mut u64).cast(),
                    (&mut piw as *mut u64).cast(),
                    (&mut pis as *mut u64).cast(),
                    (&mut px as *mut u64).cast(),
                    (&mut py as *mut u64).cast(),
                    (&mut pf as *mut u64).cast(),
                    (&mut n as *mut i32).cast(),
                    (&mut k as *mut i32).cast(),
                    (&mut g as *mut i32).cast(),
                    (&mut sl as *mut i32).cast(),
                    (&mut bits as *mut i32).cast(),
                    (&mut m as *mut u64).cast(),
                    (&mut phase_ptr as *mut u64).cast(),
                ];
                // CUDA reads the argument count from the loaded function metadata.
                // The ordinary13-argument kernel ignores the final diagnostic slot.
                unsafe {
                    sys::lib().cuLaunchKernel(
                        fun.cu_function(),
                        tokens.div_ceil(32) as u32,
                        case.n.div_ceil(64) as u32,
                        1,
                        256,
                        1,
                        1,
                        0,
                        *dev.cu_stream(),
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                }
                .result()
                .map_err(|e| e.to_string())
            };
            let old = || -> Result<(), String> {
                let scalar = if case.bits == 4 {
                    ScalarType::U4B8
                } else {
                    ScalarType::U8B128
                };
                unsafe {
                    sys::lib()
                        .cuMemsetD32Async(
                            *scratch.workspace.device_ptr(),
                            0,
                            sms as usize * 4,
                            *dev.cu_stream(),
                        )
                        .result()
                        .map_err(|e| e.to_string())?;
                    if marlin_ctmp_clear_required(tokens, case.n) {
                        sys::lib()
                            .cuMemsetD32Async(
                                *scratch.ctmp.device_ptr(),
                                0,
                                mul(tokens, case.n)?,
                                *dev.cu_stream(),
                            )
                            .result()
                            .map_err(|e| e.to_string())?
                    }
                    native(
                        *owner.input.device_ptr() as *const _,
                        *owner.packed.device_ptr() as *const _,
                        *owner.output.device_ptr() as *mut _,
                        *scratch.ctmp.device_ptr() as *mut _,
                        *owner.scales.device_ptr() as *const _,
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        tokens as i32,
                        case.n as i32,
                        case.k as i32,
                        case.k as i32,
                        *scratch.workspace.device_ptr() as *mut _,
                        &scalar,
                        false,
                        true,
                        false,
                        case.k.div_ceil(case.group) as i32,
                        case.group as i32,
                        plan.gpu_ordinal as i32,
                        *dev.cu_stream() as u64,
                        -1,
                        -1,
                        sms,
                        false,
                        true,
                        false,
                    );
                }
                Ok(())
            };
            // Compare actual cold outputs before cost collection; native BF16 weight rounding
            // is a different numerical contract and is reported, never used as a target.
            old()?;
            let native_y = dev
                .dtoh_sync_copy(&owner.output)
                .map_err(|e| e.to_string())?;
            common()?;
            let common_y = dev
                .dtoh_sync_copy(&owner.output)
                .map_err(|e| e.to_string())?;
            if cp.phase {
                let (by, bf) = cp
                    .phase_baselines
                    .get(case_index)
                    .ok_or("missing real phase baseline")?;
                let expected_y = u16s(&by.read()?)?;
                let expected_f = bf.read()?;
                let actual_f = dev
                    .dtoh_sync_copy(&owner.f32out)
                    .map_err(|e| e.to_string())?;
                if expected_y.len() < case.n || expected_f.len() < case.n * 4 {
                    return Err("phase baseline extent invalid".into());
                }
                for m in 0..tokens {
                    for n in 0..case.n {
                        if common_y[m * case.n + n] != expected_y[n]
                            || actual_f[m * case.n + n].to_bits()
                                != u32::from_le_bytes(
                                    expected_f[n * 4..n * 4 + 4].try_into().unwrap(),
                                )
                        {
                            return Err("phase observer changes actual87ordinary arithmetic".into());
                        }
                    }
                }
            }
            let valid = mul(tokens, case.n)?;
            let different = native_y[..valid]
                .iter()
                .zip(&common_y[..valid])
                .filter(|(a, b)| a != b)
                .count();
            for _ in 0..cp.warmups {
                common()?;
                old()?
            }
            dev.synchronize().map_err(|e| e.to_string())?;
            let mut samples = Vec::new();
            for sample in 0..cp.samples {
                let (new, prior) = if sample % 2 == 0 {
                    (events.sample(&common)?, events.sample(&old)?)
                } else {
                    let prior = events.sample(&old)?;
                    (events.sample(&common)?, prior)
                };
                let record = json!({"case":case.name,"tokens":tokens,"sample":sample,"common":new,"nativeMarlin":prior});
                {
                    use std::io::Write;
                    let mut file = fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(out.join("READINGS.jsonl"))
                        .map_err(|e| e.to_string())?;
                    writeln!(file, "{record}").map_err(|e| e.to_string())?;
                    file.sync_data().map_err(|e| e.to_string())?;
                }
                samples.push(record)
            }
            if let Some(buf) = &scratch.phase {
                let ticks = dev.dtoh_sync_copy(buf).map_err(|e| e.to_string())?;
                let valid = mul(mul(tokens.div_ceil(32), case.n.div_ceil(64))?, 4)?;
                let bytes: Vec<u8> = ticks[..valid]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                if bytes.len() > 16000000 {
                    return Err("phase dump above admitted perfile bound".into());
                }
                fs::write(
                    out.join(format!("{}-m{}.phase-u64", case.name, tokens)),
                    bytes,
                )
                .map_err(|e| e.to_string())?;
            }
            readings.push(json!({"case":case.name,"bits":case.bits,"n":case.n,"k":case.k,"tokens":tokens,"freeAfterAllocation":free,"nativeVsCommonBF16Different":different,"warmups":cp.warmups,"samples":samples}));
        }
    }
    cp.native.read()?;
    plan.ptx.read()?;
    Ok(
        json!({"passed":true,"gpuUuid":actual,"readings":readings,"scope":"CUDA event stream spans and Rust CPU enqueue for isolated repeated actual expert input rows; native wrapper uses actual retained sidecar/ABI/workspace policy. Profiling, not model/client throughput or speed-test qualification."}),
    )
}
