mod graph_owner;
mod phases;
mod profile;
use cudarc::{
    driver::{sys, CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
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
    fn of(p: &Path) -> Result<Self, String> {
        let b = fs::read(p).map_err(|e| e.to_string())?;
        Ok(Self {
            path: p.to_path_buf(),
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    case_name: String,
    input_manifest: Identity,
    sanitizer: Identity,
    source: Identity,
    ptx: Identity,
    build_report: Identity,
    state: Identity,
    inverse: Identity,
    weight: Identity,
    reference: Identity,
    reference_f32: Identity,
    rows: usize,
    cols: usize,
    tokens: usize,
    gpu_ordinal: usize,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    partitions: Vec<Vec<usize>>,
}
fn save(p: &Path, v: &Value) -> Result<(), String> {
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(p)
        .map_err(|e| e.to_string())?;
    f.write_all(&serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}
fn fresh(p: &Path) -> Result<(), String> {
    fs::create_dir(p).map_err(|e| format!("fresh output directory {}: {e}", p.display()))
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("extent overflow".into())
}
fn f32s(z: &[u8]) -> Result<Vec<f32>, String> {
    if z.len() % 4 != 0 {
        return Err("invalid F32 bytes".into());
    }
    Ok(z.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn u16s(z: &[u8]) -> Result<Vec<u16>, String> {
    if z.len() % 2 != 0 {
        return Err("invalid BF16 bytes".into());
    }
    Ok(z.chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn write_f32(p: &Path, z: &[f32]) -> Result<(), String> {
    let mut b = Vec::with_capacity(z.len() * 4);
    for x in z {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(p, b).map_err(|e| e.to_string())
}
#[derive(Clone, Copy)]
enum Kind {
    Tensor9,
    Simt64,
    Block64,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Tensor9 => "ordered_hc_f32_tf32x9",
            Self::Simt64 => "ordered_hc_f64_simt",
            Self::Block64 => "ordered_hc_f64_block",
        }
    }
    fn tile(self) -> usize {
        match self {
            Self::Tensor9 => 32,
            Self::Simt64 | Self::Block64 => 4,
        }
    }
    fn shared(self) -> i32 {
        match self {
            Self::Tensor9 => 30720,
            Self::Simt64 | Self::Block64 => 0,
        }
    }
}
fn compile(plan: &Path, out: &Path, kind: Kind) -> Result<Value, String> {
    let p: Build = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    p.source.read()?;
    p.nvcc.read()?;
    let expected = [
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ];
    if p.flags != expected {
        return Err("compiler flags differ from admitted precise prototype".into());
    }
    let version = Command::new(&p.nvcc.path)
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !version.status.success() {
        return Err("nvcc version failed".into());
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
        return Err(format!("nvcc failed {}", r.status));
    }
    p.source.read()?;
    p.nvcc.read()?;
    let ptx = Identity::of(&out.join("operator.ptx"))?;
    if !String::from_utf8(ptx.read()?)
        .map_err(|e| e.to_string())?
        .contains(kind.name())
    {
        return Err("compiled entry missing".into());
    }
    Ok(
        json!({"passed":true,"source":p.source,"nvcc":p.nvcc,"compilerVersion":String::from_utf8_lossy(&version.stdout),"flags":p.flags,"ptx":ptx,"adopted":false}),
    )
}
struct Buffers {
    device: Arc<CudaDevice>,
    state: CudaSlice<u16>,
    inverse: CudaSlice<f32>,
    weight: CudaSlice<f32>,
    output: CudaSlice<f32>,
}
impl Drop for Buffers {
    fn drop(&mut self) {
        if let Err(e) = self.device.synchronize() {
            eprintln!("FATAL: completion unknown; buffer owners retained: {e}");
            std::process::abort();
        }
    }
}
fn execute(plan: &Path, out: &Path, kind: Kind) -> Result<Value, String> {
    let p: Plan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let provenance: Value =
        serde_json::from_slice(&p.input_manifest.read()?).map_err(|e| e.to_string())?;
    if provenance["passed"] != true {
        return Err("input reference audit not passed".into());
    }
    let case = provenance["cases"]
        .as_array()
        .ok_or("missing input cases")?
        .iter()
        .find(|r| r["name"] == p.case_name)
        .ok_or("case absent from reference audit")?;
    if case["rows"] != p.rows || case["cols"] != p.cols || case["tokens"] != p.tokens {
        return Err("case dimensions differ reference audit".into());
    }
    for (field, actual) in [
        ("state", &p.state),
        ("inverse", &p.inverse),
        ("weight", &p.weight),
        ("reference", &p.reference),
        ("reference_f32", &p.reference_f32),
    ] {
        if case[field] != serde_json::to_value(actual).map_err(|e| e.to_string())? {
            return Err(format!("case {field} identity differs reference audit"));
        }
    }
    p.source.read()?;
    let ptx = String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?;
    p.build_report.read()?;
    let build: Value =
        serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
    if build["passed"] != true
        || build["ptx"]["sha256"] != p.ptx.sha256
        || build["source"]["sha256"] != p.source.sha256
    {
        return Err("build plan binding invalid".into());
    }
    if p.rows == 0
        || p.cols == 0
        || p.tokens == 0
        || p.rows > i32::MAX as usize
        || p.cols > i32::MAX as usize
    {
        return Err("invalid runtime matrix dimensions".into());
    }
    let state = u16s(&p.state.read()?)?;
    let inv = f32s(&p.inverse.read()?)?;
    let weight = f32s(&p.weight.read()?)?;
    let reference = p.reference.read()?;
    let reference_f32 = f32s(&p.reference_f32.read()?)?;
    if state.len() != mul(p.tokens, p.cols)?
        || inv.len() != p.tokens
        || weight.len() != mul(p.rows, p.cols)?
        || reference.len() != mul(mul(p.tokens, p.rows)?, 8)?
        || reference_f32.len() != mul(p.tokens, p.rows)?
    {
        return Err("tensor extents differ plan".into());
    }
    if state
        .iter()
        .any(|x| !half::bf16::from_bits(*x).to_f32().is_finite())
        || inv.iter().chain(weight.iter()).any(|x| !x.is_finite())
    {
        return Err("nonfinite prototype input".into());
    }
    for q in &p.partitions {
        if q.is_empty()
            || q.iter().any(|&n| n == 0)
            || q.iter().try_fold(0usize, |s, n| s.checked_add(*n)) != Some(p.tokens)
        {
            return Err("invalid partition plan".into());
        }
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
    let actual_uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    if actual_uuid != p.gpu_uuid {
        return Err(format!(
            "actual GPU UUID differs plan: ordinal={} actual={} expected={}",
            p.gpu_ordinal, actual_uuid, p.gpu_uuid
        ));
    }
    use sys::CUdevice_attribute as A;
    let attr = |a| dev.attribute(a).map_err(|e| e.to_string());
    let cc = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
        + attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    if cc < 80 {
        return Err("Ampere or newer required for this explicit prototype".into());
    }
    if attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? < kind.shared()
        || attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 128
    {
        return Err("device limits below fixed algebraic tile".into());
    }
    let max_threads = attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? as usize;
    let max_shared = attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? as usize;
    let width_limit = 1usize << (usize::BITS - 1 - max_threads.leading_zeros());
    let threads = if matches!(kind, Kind::Block64) {
        p.cols.min(width_limit).next_power_of_two().max(32)
    } else {
        128
    };
    let dynamic_shared = if matches!(kind, Kind::Block64) {
        mul(threads, std::mem::size_of::<f64>())?
    } else {
        0
    };
    if threads > max_threads || dynamic_shared > max_shared {
        return Err("block-parallel geometry exceeds actual device limits".into());
    }
    let max_x = attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize;
    let max_y = attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize;
    let max_z = attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)? as usize;
    dev.load_ptx(Ptx::from_src(ptx), "hc_projection", &[kind.name()])
        .map_err(|e| e.to_string())?;
    let fun = dev
        .get_func("hc_projection", kind.name())
        .ok_or("loaded kernel missing")?;
    let count = mul(p.tokens, p.rows)?;
    let guard = 32usize;
    let sentinel = f32::from_bits(0x7fc12345);
    let buffers = Buffers {
        device: dev.clone(),
        state: dev.htod_copy(state.clone()).map_err(|e| e.to_string())?,
        inverse: dev.htod_copy(inv.clone()).map_err(|e| e.to_string())?,
        weight: dev.htod_copy(weight.clone()).map_err(|e| e.to_string())?,
        output: dev
            .htod_copy(vec![sentinel; count + guard])
            .map_err(|e| e.to_string())?,
    };
    let free_after_allocation = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    if free_after_allocation < p.safety_margin_bytes {
        return Err("measured free VRAM below requested safety policy".into());
    }
    let reset = || -> Result<(), String> {
        dev.synchronize().map_err(|e| e.to_string())?;
        let fill = vec![sentinel; count + guard];
        unsafe {
            let copied = sys::lib()
                .cuMemcpyHtoD_v2(
                    *buffers.output.device_ptr(),
                    fill.as_ptr().cast(),
                    mul(fill.len(), 4)?,
                )
                .result();
            // Reset uses the legacy stream: keep the host owner through
            // context completion on success and error before it can drop.
            if let Err(e) = sys::lib().cuCtxSynchronize().result() {
                eprintln!("FATAL: reset completion unknown; owners retained: {e}");
                std::process::abort();
            }
            copied.map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let enqueue = |start: usize, n: usize| -> Result<(), String> {
        if n == 0
            || start
                .checked_add(n)
                .filter(|&end| end <= p.tokens)
                .is_none()
        {
            return Err("launch outside owned tensor".into());
        }
        let tiles = n.div_ceil(kind.tile());
        let x = tiles.min(max_x);
        let z = tiles.div_ceil(x);
        let y = if matches!(kind, Kind::Block64) {
            p.rows
        } else {
            p.rows.div_ceil(kind.tile())
        };
        if y > max_y || z > max_z {
            return Err("launch grid exceeds actual device limits".into());
        }
        let sp = (*buffers.state.device_ptr())
            .checked_add(mul(mul(start, p.cols)?, 2)? as u64)
            .ok_or("state pointer overflow")?;
        let ip = (*buffers.inverse.device_ptr())
            .checked_add(mul(start, 4)? as u64)
            .ok_or("inverse pointer overflow")?;
        let op = (*buffers.output.device_ptr())
            .checked_add(mul(mul(start, p.rows)?, 4)? as u64)
            .ok_or("output pointer overflow")?;
        unsafe {
            fun.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (x as u32, y as u32, z as u32),
                        block_dim: (threads as u32, 1, 1),
                        shared_mem_bytes: dynamic_shared as u32,
                    },
                    (
                        sp,
                        ip,
                        *buffers.weight.device_ptr(),
                        op,
                        p.rows as i32,
                        p.cols as i32,
                        n as u64,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let read = || -> Result<Vec<f32>, String> {
        dev.synchronize().map_err(|e| e.to_string())?;
        let z = dev
            .dtoh_sync_copy(&buffers.output)
            .map_err(|e| e.to_string())?;
        if z[count..].iter().any(|x| x.to_bits() != sentinel.to_bits())
            || z[..count].iter().any(|x| !x.is_finite())
        {
            return Err("output nonfinite or suffix guard changed".into());
        }
        Ok(z)
    };
    enqueue(0, p.tokens)?;
    let full = read()?;
    write_f32(&out.join("full.f32"), &full[..count])?;
    let same = |z: &[f32]| -> bool {
        z[..count]
            .iter()
            .zip(&full[..count])
            .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    let mut controls = Vec::new();
    for (i, parts) in p.partitions.iter().enumerate() {
        reset()?;
        let mut start = 0;
        for &n in parts {
            enqueue(start, n)?;
            start += n;
        }
        let z = read()?;
        write_f32(&out.join(format!("partition-{i}.f32")), &z[..count])?;
        if !same(&z) {
            return Err(format!("partition{i} differs full; data retained"));
        }
        reset()?;
        graph_owner::completed_graph_launch(&dev, || {
            let mut start = 0;
            for &n in parts {
                enqueue(start, n)?;
                start += n;
            }
            Ok(())
        })?;
        let z = read()?;
        write_f32(&out.join(format!("partition-graph-{i}.f32")), &z[..count])?;
        if !same(&z) {
            return Err(format!("partition graph{i} differs full; data retained"));
        }
        controls.push(
            json!({"partition":parts,"exact":true,"graphExact":true,"suffixGuardExact":true}),
        );
    }
    reset()?;
    for i in 0..p.tokens {
        enqueue(i, 1)?;
    }
    let z = read()?;
    write_f32(&out.join("singletons.f32"), &z[..count])?;
    if !same(&z) {
        return Err("singleton outputs differ full".into());
    }
    reset()?;
    graph_owner::completed_graph_launch(&dev, || enqueue(0, p.tokens))?;
    let z = read()?;
    write_f32(&out.join("graph.f32"), &z[..count])?;
    if !same(&z) {
        return Err("graph output differs full".into());
    }
    reset()?;
    let rejected = graph_owner::completed_graph_launch(&dev, || {
        enqueue(0, p.tokens)?;
        Err("intentional after-enqueue rejection".into())
    });
    if rejected.is_ok() {
        return Err("injected graph rejection accepted".into());
    }
    let z = dev
        .dtoh_sync_copy(&buffers.output)
        .map_err(|e| e.to_string())?;
    if z.iter().any(|x| x.to_bits() != sentinel.to_bits()) {
        return Err("unreplayed rejected graph wrote output".into());
    }
    enqueue(0, p.tokens)?;
    let z = read()?;
    if !same(&z) {
        return Err("graph error recovery differs full".into());
    }
    if dev
        .dtoh_sync_copy(&buffers.state)
        .map_err(|e| e.to_string())?
        != state
        || !dev
            .dtoh_sync_copy(&buffers.inverse)
            .map_err(|e| e.to_string())?
            .iter()
            .zip(&inv)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        || !dev
            .dtoh_sync_copy(&buffers.weight)
            .map_err(|e| e.to_string())?
            .iter()
            .zip(&weight)
            .all(|(x, y)| x.to_bits() == y.to_bits())
    {
        return Err("read-only input changed".into());
    }
    let mut numeric = Vec::new();
    for i in 0..count {
        let ref64 = f64::from_le_bytes(reference[i * 8..i * 8 + 8].try_into().unwrap());
        if !ref64.is_finite() {
            return Err("nonfinite reference".into());
        }
        let expected = reference_f32[i];
        if !expected.is_finite() {
            return Err("nonfinite exact F32 reference".into());
        }
        let ordered = |v: f32| -> i64 {
            let b = v.to_bits();
            if b & 0x80000000 != 0 {
                (0x80000000u32 - (b & 0x7fffffff)) as i64
            } else {
                (0x80000000u32 + (b & 0x7fffffff)) as i64
            }
        };
        let ulp = (ordered(full[i]) - ordered(expected)).unsigned_abs();
        numeric.push(json!({"token":i/p.rows,"row":i%p.rows,"actual":full[i],"referenceF64":ref64,"referenceF32":expected,"ulp":ulp,"absoluteError":(full[i] as f64-ref64).abs()}));
    }
    let numeric_pass = numeric.iter().all(|v| v["ulp"].as_u64().unwrap() <= 1);
    let max_ulp = numeric
        .iter()
        .map(|v| v["ulp"].as_u64().unwrap())
        .max()
        .unwrap();
    save(&out.join("numerics.json"), &json!(numeric))?;
    p.source.read()?;
    p.ptx.read()?;
    Ok(
        json!({"collectionPassed":true,"partitionControlsPassed":true,"numericalTargetPassed":numeric_pass,"passed":numeric_pass,"maxUlp":max_ulp,"controls":controls,"singletonsExact":true,"graphExact":true,"suffixGuardsExact":true,"inputsExact":true,"measuredFreeAfterAllocationBytes":free_after_allocation,"safetyMarginBytes":p.safety_margin_bytes,"postEnqueueRejection":rejected.err(),"rejectedGraphUnreplayed":true,"exactRecovery":true,"gpuUuid":actual_uuid,"gpuName":dev.name().map_err(|e|e.to_string())?,"computeCapability":cc,"rows":p.rows,"cols":p.cols,"tokens":p.tokens,"source":p.source,"ptx":p.ptx,"binary":Identity::of(&std::env::current_exe().map_err(|e|e.to_string())?)?,"caseName":p.case_name,"inputManifest":p.input_manifest,"kernel":kind.name(),"scope":"unadopted operator controls; authentic/constructed provenance bound in input manifest, not ordinary serving/Witness/performance/ADQ"}),
    )
}
fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("use ./dev hc-projection-probe; direct execution refused".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "devices" {
        let mut count = 0;
        unsafe {
            sys::lib().cuInit(0).result().map_err(|e| e.to_string())?;
            sys::lib()
                .cuDeviceGetCount(&mut count)
                .result()
                .map_err(|e| e.to_string())?;
        }
        let mut devices = Vec::new();
        for ordinal in 0..count {
            let mut device = 0;
            let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
            unsafe {
                sys::lib()
                    .cuDeviceGet(&mut device, ordinal)
                    .result()
                    .map_err(|e| e.to_string())?;
                sys::lib()
                    .cuDeviceGetUuid_v2(&mut uuid, device)
                    .result()
                    .map_err(|e| e.to_string())?;
            }
            let hex = uuid
                .bytes
                .iter()
                .map(|b| format!("{:02x}", *b as u8))
                .collect::<String>();
            devices.push(json!({"ordinal":ordinal,"uuid":format!("GPU-{}-{}-{}-{}-{}",&hex[..8],&hex[8..12],&hex[12..16],&hex[16..20],&hex[20..])}));
        }
        println!(
            "{}",
            json!({"devices":devices,"cudaVisibleDevices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),"cudaDeviceOrder":std::env::var("CUDA_DEVICE_ORDER").ok()})
        );
        return Ok(());
    }
    if args.len() != 3 {
        return Err("usage: build|run plan.json fresh-output-dir".into());
    }
    if args[0] == "sanitize" || args[0] == "sanitize-simt" || args[0] == "sanitize-block" {
        let plan: Plan = serde_json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        plan.sanitizer.read()?;
        let version = Command::new(&plan.sanitizer.path)
            .arg("--version")
            .output()
            .map_err(|e| e.to_string())?;
        if !version.status.success() {
            return Err("admitted sanitizer version query failed".into());
        }
        let result = Command::new(&plan.sanitizer.path)
            .args([
                "--tool",
                "memcheck",
                "--leak-check",
                "full",
                "--error-exitcode",
                "97",
                "--target-processes",
                "all",
            ])
            .arg(std::env::current_exe().map_err(|e| e.to_string())?)
            .arg(if args[0] == "sanitize-block" {
                "run-block"
            } else if args[0] == "sanitize-simt" {
                "run-simt"
            } else {
                "run"
            })
            .arg(&args[1])
            .arg(&args[2])
            .status()
            .map_err(|e| e.to_string())?;
        save(
            &Path::new(&args[2]).join("SANITIZER.json"),
            &json!({"sanitizer":plan.sanitizer,"version":String::from_utf8_lossy(&version.stdout),"status":result.code(),"tool":"memcheck","leakCheck":"full","errorExitcode":97}),
        )?;
        if !result.success() {
            return Err(format!("sanitized operator failed {result}; data retained"));
        }
        return Ok(());
    }
    let out = Path::new(&args[2]);
    fresh(out)?;
    let result = match args[0].as_str() {
        "build" => compile(Path::new(&args[1]), out, Kind::Tensor9),
        "build-simt" => compile(Path::new(&args[1]), out, Kind::Simt64),
        "build-block" => compile(Path::new(&args[1]), out, Kind::Block64),
        "build-native" => profile::compile(Path::new(&args[1]), out),
        "profile" => profile::run(Path::new(&args[1]), out, Kind::Tensor9),
        "profile-simt" => profile::run(Path::new(&args[1]), out, Kind::Simt64),
        "profile-block" => profile::run(Path::new(&args[1]), out, Kind::Block64),
        "phases" => phases::run(Path::new(&args[1]), out),
        "run" => execute(Path::new(&args[1]), out, Kind::Tensor9),
        "run-simt" => execute(Path::new(&args[1]), out, Kind::Simt64),
        "run-block" => execute(Path::new(&args[1]), out, Kind::Block64),
        _ => Err("unknown operation".into()),
    };
    match result {
        Ok(v) => {
            save(&out.join("REPORT.json"), &v)?;
            println!("{}", v);
            if v["passed"] != true {
                return Err("registered numerical target failed; outputs preserved".into());
            }
            Ok(())
        }
        Err(e) => {
            save(&out.join("FAILED.json"), &json!({"passed":false,"error":e}))?;
            Err(e)
        }
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
