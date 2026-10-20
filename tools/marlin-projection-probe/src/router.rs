//! Actual retained BF16 router operands, explicitly F32 output.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterPlan {
    source: Identity,
    ptx: Identity,
    build: Identity,
    input: Identity,
    weight: Identity,
    rows: usize,
    cols: usize,
    gpu_ordinal: usize,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    raw_limit: usize,
}
pub fn execute(path: &Path, out: &Path) -> Result<Value, String> {
    let p: RouterPlan = read_json(path)?;
    p.source.read()?;
    let code = String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?;
    let build: Value = serde_json::from_slice(&p.build.read()?).map_err(|e| e.to_string())?;
    if build["passed"] != true
        || build["source"] != serde_json::to_value(&p.source).unwrap()
        || build["ptx"] != serde_json::to_value(&p.ptx).unwrap()
    {
        return Err("router build identity mismatch".into());
    }
    let weight = u16s(&p.weight.read()?)?;
    let input = u16s(&p.input.read()?)?;
    if p.rows == 0
        || p.cols == 0
        || p.rows > i32::MAX as usize
        || p.cols > i32::MAX as usize
        || weight.len() != mul(p.rows, p.cols)?
        || input.len() != p.cols
        || weight
            .iter()
            .chain(input.iter())
            .any(|v| v & 0x7f80 == 0x7f80)
    {
        return Err("actual router geometry/finite operand admission failed".into());
    }
    let geometries = [
        (p.rows, p.cols),
        (1, 16usize.min(p.cols)),
        (17usize.min(p.rows), 80usize.min(p.cols)),
    ];
    let prospective = geometries
        .iter()
        .map(|&(n, _)| n * 67 * 4 * 4)
        .sum::<usize>();
    if prospective > p.raw_limit {
        return Err("router prospective raw exceeds bound".into());
    }
    let dev = CudaDevice::new_with_stream(p.gpu_ordinal).map_err(|e| e.to_string())?;
    let mut uuid = std::mem::MaybeUninit::uninit();
    unsafe { sys::lib().cuDeviceGetUuid_v2(uuid.as_mut_ptr(), *dev.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let h = unsafe { uuid.assume_init() }
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
    if actual != p.gpu_uuid {
        return Err("router GPU identity mismatch".into());
    }
    dev.load_ptx(
        code.into(),
        "common_router",
        &["bf16_projection_tensorcomp_f32"],
    )
    .map_err(|e| e.to_string())?;
    let f = dev
        .get_func("common_router", "bf16_projection_tensorcomp_f32")
        .ok_or("router symbol absent")?;
    let mut records = Vec::new();
    let mut raw = 0;
    for (n, k) in geometries {
        let w: Vec<_> = (0..n)
            .flat_map(|r| weight[r * p.cols..r * p.cols + k].iter().copied())
            .collect();
        let x = input[..k].to_vec();
        let wd = dev.htod_copy(w.clone()).map_err(|e| e.to_string())?;
        let xd = dev.htod_copy(x.repeat(67)).map_err(|e| e.to_string())?;
        let yd = dev
            .htod_copy(vec![f32::from_bits(0x7fc12345); 67 * n + 32])
            .map_err(|e| e.to_string())?;
        let _completion = Completion(dev.clone());
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < p.safety_margin_bytes {
            return Err("router actual VRAM below safety margin".into());
        }
        let mut refs = vec![0f64; n];
        let mut abs = vec![0f64; n];
        for r in 0..n {
            for j in 0..k {
                let prod = f32::from_bits((w[r * k + j] as u32) << 16) as f64
                    * f32::from_bits((x[j] as u32) << 16) as f64;
                refs[r] += prod;
                abs[r] += prod.abs();
            }
        }
        let launch = |start: usize, tokens: usize| -> Result<(), String> {
            if tokens == 0 || start.checked_add(tokens).filter(|&m| m <= 67).is_none() {
                return Err("router actual allocation bounds".into());
            }
            let mut input = *xd.device_ptr() + (start * k * 2) as u64;
            let mut weight = *wd.device_ptr();
            let mut output = *yd.device_ptr() + (start * n * 4) as u64;
            let mut rows = n as i32;
            let mut cols = k as i32;
            let mut m = tokens as u64;
            let mut args = [
                (&mut input as *mut u64).cast(),
                (&mut weight as *mut u64).cast(),
                (&mut output as *mut u64).cast(),
                (&mut rows as *mut i32).cast(),
                (&mut cols as *mut i32).cast(),
                (&mut m as *mut u64).cast(),
            ];
            unsafe {
                sys::lib().cuLaunchKernel(
                    f.cu_function(),
                    tokens.div_ceil(64) as u32,
                    n.div_ceil(64) as u32,
                    1,
                    512,
                    1,
                    1,
                    0,
                    *dev.cu_stream(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
            .result()
            .map_err(|e| e.to_string())
        };
        let mut baseline = None;
        let mut comparisons = Vec::new();
        for mode in ["full", "singletons", "partition", "graph"] {
            unsafe {
                sys::lib().cuMemsetD32Async(
                    *yd.device_ptr(),
                    0x7fc12345,
                    67 * n + 32,
                    *dev.cu_stream(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            match mode {
                "full" => launch(0, 67)?,
                "singletons" => {
                    for r in 0..67 {
                        launch(r, 1)?;
                    }
                }
                "partition" => {
                    let mut r = 0;
                    for m in [1, 15, 1, 16, 17, 17] {
                        launch(r, m)?;
                        r += m;
                    }
                }
                _ => graph_owner::completed_graph_launch(&dev, || launch(0, 67))?,
            }
            let y = dev.dtoh_sync_copy(&yd).map_err(|e| e.to_string())?;
            if y[67 * n..].iter().any(|v| v.to_bits() != 0x7fc12345) {
                return Err("router guard corrupted".into());
            }
            let mut max_error = 0f64;
            for m in 0..67 {
                for r in 0..n {
                    let v = y[m * n + r];
                    if !v.is_finite() || v.to_bits() != y[r].to_bits() {
                        return Err("router repeated row differs".into());
                    }
                    let err = (v as f64 - refs[r]).abs();
                    let steps = 2 * k + 16;
                    let u = 2f64.powi(-24);
                    let bound = (steps as f64 * u / (1.0 - steps as f64 * u)) * abs[r]
                        + k as f64 * 2f64.powi(-126);
                    if err > bound {
                        return Err("router F64 contraction bound".into());
                    }
                    max_error = max_error.max(err);
                }
            }
            let bits: Vec<_> = y.iter().map(|v| v.to_bits()).collect();
            if let Some(b) = &baseline {
                if *b != bits {
                    return Err(format!("router {mode} differs full"));
                }
            } else {
                baseline = Some(bits);
            }
            let bytes: Vec<_> = y[..67 * n].iter().flat_map(|v| v.to_le_bytes()).collect();
            raw += bytes.len();
            fs::write(out.join(format!("n{n}-k{k}-{mode}.f32")), bytes)
                .map_err(|e| e.to_string())?;
            comparisons.push(json!({"mode":mode,"exactFull":true,"maxAbsoluteF64Error":max_error}));
        }
        if graph_owner::completed_graph_launch(&dev, || {
            launch(0, 67)?;
            Err("intentional router post-enqueue rejection".into())
        })
        .is_ok()
        {
            return Err("router rejected error accepted".into());
        }
        launch(0, 67)?;
        let y = dev.dtoh_sync_copy(&yd).map_err(|e| e.to_string())?;
        if y.iter().map(|v| v.to_bits()).collect::<Vec<_>>() != baseline.unwrap()
            || dev.dtoh_sync_copy(&wd).map_err(|e| e.to_string())? != w
            || dev.dtoh_sync_copy(&xd).map_err(|e| e.to_string())? != x.repeat(67)
        {
            return Err("router recovery/readonly changed".into());
        }
        fs::write(
            out.join(format!("n{n}-k{k}-reference.f64")),
            refs.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            out.join(format!("n{n}-k{k}-abs.f64")),
            abs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        records.push(json!({"rows":n,"cols":k,"comparisons":comparisons,"guardsReadonlyErrorRecoveryExact":true}));
    }
    if raw != prospective {
        return Err("router raw extent reconciliation".into());
    }
    p.source.read()?;
    p.ptx.read()?;
    Ok(
        json!({"passed":true,"servingQualified":false,"rawBytes":raw,"gpuUuid":actual,"records":records,"scope":"Actual retained BF16 router plus coefficient-preserving shape/tail crops; true F32output and wholeF64bounds, no roundedlogits/reinterpretation/modelacceptance."}),
    )
}
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("FATAL router completion unknown:{e}");
            std::process::abort()
        }
    }
}
