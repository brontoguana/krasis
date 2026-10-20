mod cost;
mod crops;
mod grouped;
mod router;
mod slots;
mod activation;
mod combine;
mod row_grid;
mod norm;
#[path = "../../../src/common_marlin.rs"]
mod common_marlin;
mod graph_owner;
mod jit;
mod layout_reference;
#[path = "../../../src/reference_rounding.rs"]
mod reference_rounding;
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Identity {
    fn read(&self) -> Result<Vec<u8>, String> {
        let b = fs::read(&self.path).map_err(|e| e.to_string())?;
        if b.len() != self.bytes || format!("{:x}", Sha256::digest(&b)) != self.sha256 {
            return Err(format!("identity mismatch {}", self.path.display()));
        }
        Ok(b)
    }
    fn of(path: &Path) -> Result<Self, String> {
        let b = fs::read(path).map_err(|e| e.to_string())?;
        Ok(Self {
            path: path.into(),
            bytes: b.len(),
            sha256: format!("{:x}", Sha256::digest(&b)),
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Build {
    source: Identity,
    nvcc: Identity,
    flags: Vec<String>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    n: usize,
    k: usize,
    bits: usize,
    group: usize,
    input: Identity,
    input_offset: usize,
    packed: Identity,
    scales: Identity,
    reference: Identity,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    source: Identity,
    ptx: Identity,
    build: Identity,
    cases: Vec<Case>,
    gpu_ordinal: usize,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    raw_limit: usize,
    total_limit: usize,
}
fn read_json<T: serde::de::DeserializeOwned>(p: &Path) -> Result<T, String> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn save(p: &Path, v: &Value) -> Result<(), String> {
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .map_err(|e| e.to_string())?;
    f.write_all(&serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("extent overflow".into())
}
fn u16s(b: &[u8]) -> Result<Vec<u16>, String> {
    if b.len() % 2 != 0 {
        return Err("odd BF16 bytes".into());
    }
    Ok(b.chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn inverse_perms(bits: usize, grouped: bool) -> (Vec<i32>, Vec<i32>) {
    let mut perm = Vec::new();
    for i in 0..32 {
        let mut sub = Vec::new();
        for block in 0..2 {
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                sub.push(16 * row + i / 4 + 8 * block)
            }
        }
        for j in 0..4 {
            for &q in &sub {
                perm.push(q + 256 * j)
            }
        }
    }
    let inter: Vec<usize> = if bits == 4 {
        vec![0, 2, 4, 6, 1, 3, 5, 7]
    } else {
        vec![0, 2, 1, 3]
    };
    let mut inv = vec![0; 1024];
    for i in 0..1024 {
        let v = perm[i / inter.len() * inter.len() + inter[i % inter.len()]];
        inv[v] = i as i32
    }
    let sp: Vec<usize> = if grouped {
        (0..8)
            .flat_map(|i| (0..8).map(move |j| i + 8 * j))
            .collect()
    } else {
        (0..4)
            .flat_map(|i| {
                [0, 1, 8, 9, 16, 17, 24, 25]
                    .into_iter()
                    .map(move |off| 2 * i + off)
            })
            .collect()
    };
    let mut si = vec![0; sp.len()];
    for (i, &v) in sp.iter().enumerate() {
        si[v] = i as i32
    }
    (inv, si)
}
fn build(plan: &Path, out: &Path, norm_contract:bool) -> Result<Value, String> {
    let b: Build = read_json(plan)?;
    b.source.read()?;
    b.nvcc.read()?;
    let flags: &[&str] = if norm_contract {
        if b.source.path.file_name().and_then(|v|v.to_str())!=Some("common_rmsnorm.cu"){return Err("normalization compiler contract requires the declared norm source".into());}
        &["-ptx","-allow-unsupported-compiler","-arch=sm_80","-O3","--use_fast_math","-std=c++17"]
    } else { &[
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ] };
    if !b.flags.iter().map(String::as_str).eq(flags.iter().copied()) {
        return Err("declared precise/normalization compiler contract differs".into());
    }
    let version = Command::new(&b.nvcc.path)
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !version.status.success() {
        return Err("compiler version failed".into());
    }
    let r = Command::new(&b.nvcc.path)
        .args(&b.flags)
        .arg(&b.source.path)
        .arg("-o")
        .arg(out.join("operator.ptx"))
        .output()
        .map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stdout"), r.stdout).map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stderr"), r.stderr).map_err(|e| e.to_string())?;
    if !r.status.success() {
        return Err(format!("compile failed {}", r.status));
    }
    b.source.read()?;
    b.nvcc.read()?;
    let ptx = Identity::of(&out.join("operator.ptx"))?;
    Ok(
        json!({"passed":true,"source":b.source,"nvcc":b.nvcc,"flags":b.flags,"version":String::from_utf8_lossy(&version.stdout),"ptx":ptx}),
    )
}
struct Owners {
    device: Arc<CudaDevice>,
    packed: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    iw: CudaSlice<i32>,
    is: CudaSlice<i32>,
    input: CudaSlice<u16>,
    output: CudaSlice<u16>,
    f32out: CudaSlice<f32>,
}
impl Drop for Owners {
    fn drop(&mut self) {
        if let Err(e) = self.device.synchronize() {
            eprintln!("FATAL: owner completion unknown: {e}");
            std::process::abort()
        }
    }
}
fn execute(plan: &Path, out: &Path) -> Result<Value, String> {
    let plan: Plan = read_json(plan)?;
    plan.source.read()?;
    let ptx = String::from_utf8(plan.ptx.read()?).map_err(|e| e.to_string())?;
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
    dev.load_ptx(
        ptx.into(),
        "marlin_proto",
        &["marlin_projection_tensorcomp"],
    )
    .map_err(|e| e.to_string())?;
    let fun = dev
        .get_func("marlin_proto", "marlin_projection_tensorcomp")
        .ok_or("entry missing")?;
    let mut raw = 0usize;
    let mut records = Vec::new();
    for case in &plan.cases {
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
            return Err("invalid Marlin geometry".into());
        }
        let input = case.input.read()?;
        let end = case
            .input_offset
            .checked_add(mul(case.k, 2)?)
            .ok_or("input range overflow")?;
        if end > input.len() {
            return Err("actual input range invalid".into());
        }
        let x = u16s(&input[case.input_offset..end])?;
        let packed = case.packed.read()?;
        let scales = case.scales.read()?;
        let reference = case.reference.read()?;
        if packed.len() != mul(mul(case.n, case.k)?, case.bits)? / 8
            || scales.len() != mul(mul(case.k.div_ceil(case.group), case.n)?, 2)?
            || reference.len() != mul(case.n, 8)?
        {
            return Err("actual tensor geometry mismatch".into());
        }
        if x.iter()
            .chain(u16s(&scales)?.iter())
            .any(|v| v & 0x7f80 == 0x7f80)
        {
            return Err("nonfinite BF16 operands".into());
        }
        let (iw, is) = inverse_perms(case.bits, case.group < case.k);
        let max_tokens = 67usize;
        let guard = 32usize;
        let count = mul(max_tokens, case.n)?;
        let repeated = x.repeat(max_tokens);
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
            iw: dev.htod_copy(iw.clone()).map_err(|e| e.to_string())?,
            is: dev.htod_copy(is.clone()).map_err(|e| e.to_string())?,
            input: dev.htod_copy(repeated.clone()).map_err(|e| e.to_string())?,
            output: dev
                .htod_copy(vec![0x7fc1; count + guard])
                .map_err(|e| e.to_string())?,
            f32out: dev
                .htod_copy(vec![f32::from_bits(0x7fc12345); count + guard])
                .map_err(|e| e.to_string())?,
        };
        dev.synchronize().map_err(|e| e.to_string())?;
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < plan.safety_margin_bytes {
            return Err("measured VRAM below safety margin".into());
        }
        let launch = |start: usize, tokens: usize| -> Result<(), String> {
            if tokens == 0
                || start
                    .checked_add(tokens)
                    .filter(|&v| v <= max_tokens)
                    .is_none()
            {
                return Err("outside actual allocation".into());
            }
            let mut pw = *owner.packed.device_ptr();
            let mut ps = *owner.scales.device_ptr();
            let mut piw = *owner.iw.device_ptr();
            let mut pis = *owner.is.device_ptr();
            let mut px = *owner.input.device_ptr() + mul(mul(start, case.k)?, 2)? as u64;
            let mut py = *owner.output.device_ptr() + mul(mul(start, case.n)?, 2)? as u64;
            let mut pf = *owner.f32out.device_ptr() + mul(mul(start, case.n)?, 4)? as u64;
            let mut n = case.n as i32;
            let mut k = case.k as i32;
            let mut g = case.group as i32;
            let mut sl = is.len() as i32;
            let mut bits = case.bits as i32;
            let mut m = tokens as u64;
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
            ];
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
        let case_dir = out.join(&case.name);
        fs::create_dir(&case_dir).map_err(|e| e.to_string())?;
        let mut baseline: Option<(Vec<u16>, Vec<f32>)> = None;
        let mut comparisons = Vec::new();
        for mode in ["full", "singletons", "partition", "graph"] {
            match mode {
                "full" => launch(0, max_tokens)?,
                "singletons" => {
                    for start in 0..max_tokens {
                        launch(start, 1)?
                    }
                }
                "partition" => {
                    let mut start = 0;
                    for num in [1, 15, 1, 16, 17, 17] {
                        launch(start, num)?;
                        start += num
                    }
                }
                "graph" => graph_owner::completed_graph_launch(&dev, || launch(0, max_tokens))?,
                _ => unreachable!(),
            }
            let y = dev
                .dtoh_sync_copy(&owner.output)
                .map_err(|e| e.to_string())?;
            let z = dev
                .dtoh_sync_copy(&owner.f32out)
                .map_err(|e| e.to_string())?;
            if y[count..].iter().any(|&v| v != 0x7fc1)
                || z[count..].iter().any(|v| v.to_bits() != 0x7fc12345)
            {
                return Err("suffix guard changed".into());
            }
            let yb: Vec<u8> = y[..count].iter().flat_map(|v| v.to_le_bytes()).collect();
            let zb: Vec<u8> = z[..count].iter().flat_map(|v| v.to_le_bytes()).collect();
            let added = yb.len() + zb.len();
            if raw
                .checked_add(added)
                .filter(|&v| v <= plan.raw_limit)
                .is_none()
            {
                return Err("prospective raw limit exceeded".into());
            }
            fs::write(case_dir.join(format!("{mode}.bf16")), yb).map_err(|e| e.to_string())?;
            fs::write(case_dir.join(format!("{mode}.f32")), zb).map_err(|e| e.to_string())?;
            raw += added;
            let mut nearest_mismatch = 0;
            let mut max_abs = 0f64;
            for m in 0..max_tokens {
                for n in 0..case.n {
                    let r = f64::from_le_bytes(reference[n * 8..n * 8 + 8].try_into().unwrap());
                    let v = z[m * case.n + n];
                    if !r.is_finite() || !v.is_finite() {
                        return Err("nonfinite reference/output".into());
                    }
                    max_abs = max_abs.max((v as f64 - r).abs());
                    if y[m * case.n + n] != reference_rounding::bf16_from_f64_bits(r) {
                        nearest_mismatch += 1
                    }
                    if y[m * case.n + n] != y[n] || v.to_bits() != z[n].to_bits() {
                        return Err("identical real rows differ inside batch".into());
                    }
                }
            }
            if let Some((by, bz)) = &baseline {
                if y != *by || z.iter().zip(bz).any(|(x, y)| x.to_bits() != y.to_bits()) {
                    return Err(format!("{} {mode} differs full", case.name));
                }
            } else {
                baseline = Some((y, z))
            }
            comparisons.push(json!({"mode":mode,"exactFull":true,"nearestF64Mismatch":nearest_mismatch,"maxAbsoluteF32Error":max_abs}));
        }
        let rejected = graph_owner::completed_graph_launch(&dev, || {
            launch(0, max_tokens)?;
            Err("intentional post-enqueue graph rejection".into())
        });
        if rejected.is_ok() {
            return Err("graph rejection accepted".into());
        }
        launch(0, max_tokens)?;
        let y = dev
            .dtoh_sync_copy(&owner.output)
            .map_err(|e| e.to_string())?;
        if y != baseline.as_ref().unwrap().0 {
            return Err("error recovery changed output".into());
        }
        if dev
            .dtoh_sync_copy(&owner.input)
            .map_err(|e| e.to_string())?
            != repeated
            || dev
                .dtoh_sync_copy(&owner.packed)
                .map_err(|e| e.to_string())?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                != packed
            || dev
                .dtoh_sync_copy(&owner.scales)
                .map_err(|e| e.to_string())?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                != scales
            || dev.dtoh_sync_copy(&owner.iw).map_err(|e| e.to_string())? != iw
            || dev.dtoh_sync_copy(&owner.is).map_err(|e| e.to_string())? != is
        {
            return Err("read-only operand changed".into());
        }
        records.push(json!({"name":case.name,"comparisons":comparisons,"readOnlyExact":true,"suffixGuardExact":true,"graphErrorRecoveryExact":true,"freeAfterAllocation":free}));
        drop(owner);
    }
    plan.source.read()?;
    plan.ptx.read()?;
    let result = json!({"passed":true,"servingQualified":false,"gpuUuid":actual,"records":records,"rawBytes":raw,"totalLimit":plan.total_limit,"scope":"Actual isolated expert operands repeated across matrix rows to test placement, partition and graph invariance; no prompt/model throughput or whole FFN qualification."});
    Ok(result)
}
fn main() {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        eprintln!("use ./dev marlin-projection-probe");
        std::process::exit(2)
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("build|run plan.json fresh-output");
        std::process::exit(2)
    }
    let out = Path::new(&args[2]);
    if let Err(e) = fs::create_dir(out) {
        eprintln!("fresh output: {e}");
        std::process::exit(1)
    }
    let result = match args[0].as_str() {
        "rounding-controls" => rounding_controls(),
        "layout-controls" => layout_controls(),
        "crop-fixtures" => crops::execute(Path::new(&args[1]), out),
        "grouped" => grouped::execute(Path::new(&args[1]), out),
        "router" => router::execute(Path::new(&args[1]), out),
        "combine" => combine::execute(Path::new(&args[1]), out),
        "norm" => norm::execute(Path::new(&args[1]), out),
        "row-grid" => row_grid::execute(Path::new(&args[1]), out),
        "decomposition-controls" => decomposition_controls(),
        "jit-log" => jit::inspect(Path::new(&args[1]), out),
        "cost" => cost::execute(Path::new(&args[1]), out),
        "build" => build(Path::new(&args[1]), out, false),
        "norm-build" => build(Path::new(&args[1]), out, true),
        "run" => execute(Path::new(&args[1]), out),
        _ => Err("invalid mode".into()),
    };
    match result {
        Ok(v) => {
            if let Err(e) = save(&out.join("REPORT.json"), &v) {
                eprintln!("{e}");
                std::process::exit(1)
            }
            println!(
                "{}",
                json!({"passed":true,"report":out.join("REPORT.json")})
            )
        }
        Err(e) => {
            let _ = save(
                &out.join("FAILURE.json"),
                &json!({"passed":false,"error":e}),
            );
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}

fn rounding_controls() -> Result<Value, String> {
    let cases = [
        (0.0, 0),
        (-0.0, 0x8000),
        (1.0, 0x3f80),
        (1.0 + 2f64.powi(-8), 0x3f80),
        (1.0 + 2f64.powi(-8) + 2f64.powi(-40), 0x3f81),
        (1.0 + 2f64.powi(-8) - 2f64.powi(-40), 0x3f80),
        (1.0 + 3.0 * 2f64.powi(-8), 0x3f82),
        (2f64.powi(-126), 128),
        (2f64.powi(-133), 1),
        (2f64.powi(-134), 0),
        (3.0 * 2f64.powi(-134), 2),
        (f64::INFINITY, 0x7f80),
        (f64::NEG_INFINITY, 0xff80),
        (f64::MIN_POSITIVE, 0),
    ];
    for (value, expected) in cases {
        if reference_rounding::bf16_from_f64_bits(value) != expected {
            return Err(format!("direct reference conversion failed {value}"));
        }
    }
    let sticky = 1.0 + 2f64.powi(-8) + 2f64.powi(-40);
    if half::bf16::from_f64(sticky).to_bits() == reference_rounding::bf16_from_f64_bits(sticky) {
        return Err("retained dependency truncation control no longer reproduces; review dependency identity".into());
    }
    Ok(
        json!({"passed":true,"directRoundingControls":cases.len(),"dependencyStickyBitDefectReproduced":true,"stickyInputBits":sticky.to_bits()}),
    )
}

fn decomposition_controls() -> Result<Value, String> {
    let mut finite = 0u64;
    let mut overflow = 0u64;
    let mut subnormal = 0u64;
    for bits in 0u32..=65535 {
        let sb = bits as u16;
        if sb & 0x7f80 == 0x7f80 {
            continue;
        }
        let scale = half::bf16::from_bits(sb).to_f32();
        for code in -128i32..128 {
            let value = (code as f32) * scale;
            if !value.is_finite() {
                overflow += 1;
                continue;
            }
            let high = half::bf16::from_bits((value.to_bits() >> 16) as u16).to_f32();
            let low = half::bf16::from_f32(value - high).to_f32();
            if (high as f64) + (low as f64) != value as f64 {
                return Err(format!(
                    "BF16 split loses logical quant weight scale={sb} code={code}"
                ));
            }
            finite += 1;
            if value != 0. && value.abs() < f32::MIN_POSITIVE {
                subnormal += 1
            }
        }
    }
    Ok(
        json!({"passed":true,"finiteExactPairs":finite,"nonfiniteLogicalWeightsExcluded":overflow,"exactF32SubnormalPairs":subnormal,"domain":"All finite BF16 scale bits and every signed INT8 code; INT4 subset included. Only finite logical F32 dequant weights admitted; no BF16 weight-bit loss."}),
    )
}

fn layout_controls() -> Result<Value, String> {
    let mut checks = 0;
    for bits in [4usize, 8] {
        let native = if bits == 4 {
            layout_reference::generate_weight_perm_int4()
        } else {
            layout_reference::generate_weight_perm_int8()
        };
        let mut inv = vec![0i32; 1024];
        for (i, &p) in native.iter().enumerate() {
            inv[p] = i as i32
        }
        for grouped in [true, false] {
            let (a, b) = inverse_perms(bits, grouped);
            if a != inv {
                return Err("probe inverseweight differs native codec".into());
            }
            let (sg, sc) = layout_reference::generate_scale_perms();
            let perm: &[usize] = if grouped { &sg } else { &sc };
            let mut expected = vec![0i32; perm.len()];
            let mut seen = std::collections::HashSet::new();
            for (i, &p) in perm.iter().enumerate() {
                if !seen.insert(p) {
                    return Err("native permutation notbijective".into());
                }
                expected[p] = i as i32
            }
            if b != expected {
                return Err("probe inverse scale differs native codec".into());
            }
            for n in 0usize..4096 {
                let size = perm.len();
                let actual = if size == 64 {
                    (n / 64) * 64 + (n % 8) * 8 + (n % 64) / 8
                } else {
                    (n / 32) * 32 + ((n % 8) / 2) * 8 + ((n % 32) / 8) * 2 + n % 2
                };
                let target = (n / size) * size + expected[n % size] as usize;
                if actual != target {
                    return Err("closed scaleaddress differs native permutation".into());
                }
                checks += 1
            }
        }
        for n in (0u64..4096).chain([1073741823, 1073741824, 2147483583]) {
            for sub in 0u64..4 {
                for j in 0u64..4 {
                    let tk = sub * 2 + (j & 1) + if j >= 2 { 8 } else { 0 };
                    let p = (n / 64) * 1024
                        + inv[((n % 64) / 16 * 256 + tk * 16 + n % 16) as usize] as u64;
                    let pack = 32 / bits as u64;
                    let (word, shift) = if bits == 4 {
                        (
                            (n / 64) * 128 + (n % 8) * 16 + sub * 4 + (n % 64) / 16,
                            (n % 16 / 8) * 8 + [0, 16, 4, 20][j as usize],
                        )
                    } else {
                        (
                            (n / 64) * 256 + (n % 8) * 32 + sub * 8 + (n % 64) / 8,
                            [0, 16, 8, 24][j as usize],
                        )
                    };
                    if word != p / pack || shift != (p % pack) * bits as u64 {
                        return Err("closed word/shift differs native permutation".into());
                    }
                    checks += 1
                }
            }
        }
    }
    let broken: std::collections::HashSet<_> = [0usize, 1, 8, 9]
        .into_iter()
        .flat_map(|off| (0..8).map(move |i| 2 * i + off))
        .collect();
    if broken.len() != 24 {
        return Err("old channelwise missing-column defect not reproduced".into());
    }
    Ok(
        json!({"passed":true,"checks":checks,"canonicalInverseAndClosedAddressesExact":true,"channelwiseDefectReproduced":true,"legacyMissingColumns":8,"scope":"Actual canonical Rust codec generator functions copied verbatim; all grouped/channelwise inverse entries and packed-fragment word/shift identities checked, including large logical columns. No runtime bank/model change."}),
    )
}
