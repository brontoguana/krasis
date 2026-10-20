use cudarc::cublas::{result as br, sys as bs, CudaBlas};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};
mod graph_owner;
mod profile;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Identity {
    fn read(&self) -> Result<Vec<u8>, String> {
        let b = fs::read(&self.path).map_err(|e| e.to_string())?;
        if b.len() != self.bytes || hash(&b) != self.sha256 {
            return Err(format!("identity mismatch {}", self.path.display()));
        }
        Ok(b)
    }
    fn of(p: &Path) -> Result<Self, String> {
        let b = fs::read(p).map_err(|e| e.to_string())?;
        Ok(Self {
            path: p.into(),
            bytes: b.len(),
            sha256: hash(&b),
        })
    }
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
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
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("extent overflow".into())
}
fn bf(b: u16) -> f64 {
    f32::from_bits((b as u32) << 16) as f64
}
fn rne(x: f64) -> Result<u16, String> {
    let f = x as f32;
    if !f.is_finite() {
        return Err("nonfinite reference result".into());
    }
    let b = f.to_bits();
    Ok((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16)
}
fn u16s(b: Vec<u8>) -> Result<Vec<u16>, String> {
    if b.len() % 2 != 0 {
        return Err("odd BF16 bytes".into());
    }
    let v: Vec<_> = b
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect();
    if v.iter().any(|x| !bf(*x).is_finite()) {
        return Err("nonfinite BF16 operand".into());
    }
    Ok(v)
}
fn write(p: &Path, v: &[u16]) -> Result<(), String> {
    fs::write(
        p,
        v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())
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
    schema: usize,
    case_name: String,
    source: Identity,
    ptx: Identity,
    build_report: Identity,
    input: Identity,
    weight: Identity,
    origin: Identity,
    reference: Identity,
    reference_report: Identity,
    sanitizer: Identity,
    rows: usize,
    cols: usize,
    tokens: usize,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    partitions: Vec<Vec<usize>>,
}
fn compile(plan: &Path, out: &Path) -> Result<Value, String> {
    let p: Build = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    p.source.read()?;
    p.nvcc.read()?;
    let flags = [
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ];
    if p.flags != flags {
        return Err("compiler policy mismatch".into());
    }
    let result = Command::new(&p.nvcc.path)
        .args(&p.flags)
        .arg("-o")
        .arg(out.join("operator.ptx"))
        .arg(&p.source.path)
        .output()
        .map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stdout"), result.stdout).map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stderr"), result.stderr).map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!("nvcc failed {}", result.status));
    }
    p.source.read()?;
    let ptx = Identity::of(&out.join("operator.ptx"))?;
    let text = String::from_utf8(ptx.read()?).map_err(|e| e.to_string())?;
    for name in [
        "bf16_projection_wmma",
        "bf16_projection_f64",
        "bf16_projection_compensated",
        "bf16_projection_chunked",
        "bf16_projection_tensorcomp",
    ] {
        if !text.contains(name) {
            return Err("compiled entry missing".into());
        }
    }
    Ok(
        json!({"passed":true,"source":p.source,"nvcc":p.nvcc,"flags":p.flags,"ptx":ptx,"adopted":false}),
    )
}
struct Buffers {
    dev: Arc<CudaDevice>,
    input: CudaSlice<u16>,
    weight: CudaSlice<u16>,
    output: CudaSlice<u16>,
}
impl Drop for Buffers {
    fn drop(&mut self) {
        if let Err(e) = self.dev.synchronize() {
            eprintln!("FATAL: borrowed buffer completion unknown: {e}");
            std::process::abort();
        }
    }
}
fn execute(plan: &Path, out: &Path, kind: &str, profiling: bool) -> Result<Value, String> {
    let p: Plan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if p.schema != 1
        || p.rows == 0
        || p.cols == 0
        || p.tokens == 0
        || p.rows > i32::MAX as usize
        || p.cols > i32::MAX as usize
    {
        return Err("invalid BF16 matrix contract".into());
    }
    let origin: Value = serde_json::from_slice(&p.origin.read()?).map_err(|e| e.to_string())?;
    if origin["caseName"] != p.case_name
        || origin["rows"] != p.rows
        || origin["cols"] != p.cols
        || origin["tokens"] != p.tokens
    {
        return Err("input provenance dimensions differ".into());
    }
    if origin["passed"] != true
        || origin["input"]["sha256"] != p.input.sha256
        || origin["weight"]["sha256"] != p.weight.sha256
        || origin["input"]["bytes"] != p.input.bytes
        || origin["weight"]["bytes"] != p.weight.bytes
    {
        return Err("operand identities differ from admitted provenance".into());
    }
    p.source.read()?;
    let build: Value =
        serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
    if build["passed"] != true
        || build["source"]["sha256"] != p.source.sha256
        || build["ptx"]["sha256"] != p.ptx.sha256
    {
        return Err("compiled source binding differs".into());
    }
    let input = u16s(p.input.read()?)?;
    let weight = u16s(p.weight.read()?)?;
    if input.len() != mul(p.tokens, p.cols)? || weight.len() != mul(p.rows, p.cols)? {
        return Err("operand extent mismatch".into());
    }
    for q in &p.partitions {
        if q.is_empty()
            || q.iter().any(|n| *n == 0)
            || q.iter().try_fold(0usize, |s, n| s.checked_add(*n)) != Some(p.tokens)
        {
            return Err("invalid partitions".into());
        }
    }
    let reference = u16s(p.reference.read()?)?;
    let rr: Value =
        serde_json::from_slice(&p.reference_report.read()?).map_err(|e| e.to_string())?;
    if reference.len() != mul(p.tokens, p.rows)?
        || rr["passed"] != true
        || rr["caseName"] != p.case_name
        || rr["input"]["sha256"] != p.input.sha256
        || rr["weight"]["sha256"] != p.weight.sha256
        || rr["reference"]["sha256"] != p.reference.sha256
        || rr["rows"] != p.rows
        || rr["cols"] != p.cols
        || rr["tokens"] != p.tokens
    {
        return Err("independent reference identity/geometry mismatch".into());
    }
    write(&out.join("reference.bf16"), &reference)?;
    let dev = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
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
        return Err(format!("actual CUDA UUID {actual} != {}", p.gpu_uuid));
    }
    use sys::CUdevice_attribute as A;
    let attr = |a| {
        dev.attribute(a)
            .map_err(|e| e.to_string())
            .and_then(|n| usize::try_from(n).map_err(|e| e.to_string()))
    };
    if attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8 {
        return Err("BF16 prototype requires Ampere or newer".into());
    }
    let mx = attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?;
    let my = attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?;
    let mz = attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?;
    let width_limit = attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
        .min(attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? / 8);
    let mut width = 1usize;
    while width
        .checked_mul(2)
        .is_some_and(|n| n <= width_limit && n <= p.cols)
    {
        width *= 2;
    }
    let ptx = String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?;
    dev.load_ptx(
        ptx.into(),
        "bf16-projection",
        &[
            "bf16_projection_wmma",
            "bf16_projection_f64",
            "bf16_projection_compensated",
        "bf16_projection_chunked",
        "bf16_projection_tensorcomp",
        ],
    )
    .map_err(|e| e.to_string())?;
    let fun = dev
        .get_func(
            "bf16-projection",
            if kind == "f64" {
                "bf16_projection_f64"
            } else if kind == "tensorcomp" {
                "bf16_projection_tensorcomp"
            } else if kind == "chunked" {
                "bf16_projection_chunked"
            } else if kind == "compensated" {
                "bf16_projection_compensated"
            } else {
                "bf16_projection_wmma"
            },
        )
        .ok_or("kernel absent")?;
    let mut function_threads = 0i32;
    unsafe {
        sys::lib().cuFuncGetAttribute(
            &mut function_threads,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            fun.cu_function(),
        )
    }
    .result()
    .map_err(|e| e.to_string())?;
    let function_threads = usize::try_from(function_threads).map_err(|e| e.to_string())?;
    if kind != "native" {
        while width > function_threads {
            width /= 2;
        }
        if width == 0 {
            return Err("loaded function has no admissible block width".into());
        }
    }
    let mut active_blocks = 0i32;
    if matches!(kind, "f64" | "compensated") {
        loop {
            unsafe {
                sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut active_blocks,
                    fun.cu_function(),
                    width as i32,
                    width * 8,
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            if active_blocks > 0 {
                break;
            }
            width /= 2;
            if width == 0 {
                return Err("loaded function has no admissible occupancy".into());
            }
        }
    }
    if kind=="tensorcomp" {
        if function_threads<512 || attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?<512 {
            return Err("loaded tiled function/device cannot admit 512-thread CTA".into());
        }
        unsafe {sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut active_blocks,fun.cu_function(),512,0)}.result().map_err(|e|e.to_string())?;
        if active_blocks<=0{return Err("loaded tiled function occupancy does not admit CTA".into());}
    }
    let count = reference.len();
    let sentinel = 0x55aau16;
    let guard = 16usize;
    let image = vec![sentinel; count.checked_add(guard).ok_or("guard extent overflow")?];
    let required = input
        .len()
        .checked_add(weight.len())
        .and_then(|n| n.checked_add(image.len()))
        .and_then(|n| n.checked_mul(2))
        .ok_or("GPU allocation extent overflow")?;
    let free_before = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    if required
        .checked_add(p.safety_margin_bytes)
        .is_none_or(|n| n > free_before)
    {
        return Err("actual free VRAM does not admit operand/output allocation and margin".into());
    }
    let buffers = Buffers {
        dev: dev.clone(),
        input: dev.htod_copy(input.clone()).map_err(|e| e.to_string())?,
        weight: dev.htod_copy(weight.clone()).map_err(|e| e.to_string())?,
        output: dev.htod_copy(image.clone()).map_err(|e| e.to_string())?,
    };
    let free = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    if free < p.safety_margin_bytes {
        return Err("measured allocation free below admitted margin".into());
    }
    let blas = CudaBlas::new(dev.clone()).map_err(|e| e.to_string())?;
    unsafe { br::set_stream(*blas.handle(), *dev.cu_stream() as bs::cudaStream_t) }
        .map_err(|e| e.to_string())?;
    let enqueue = |start: usize, rows: usize| -> Result<(), String> {
        if rows == 0 || start.checked_add(rows).is_none_or(|n| n > p.tokens) {
            return Err("invalid enqueue extent".into());
        }
        let ip = (*buffers.input.device_ptr())
            .checked_add(mul(mul(start, p.cols)?, 2)? as u64)
            .ok_or("input address overflow")?;
        let op = (*buffers.output.device_ptr())
            .checked_add(mul(mul(start, p.rows)?, 2)? as u64)
            .ok_or("output address overflow")?;
        if kind == "native" {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            unsafe {
                br::gemm_ex(
                    *blas.handle(),
                    bs::cublasOperation_t::CUBLAS_OP_T,
                    bs::cublasOperation_t::CUBLAS_OP_N,
                    p.rows as i32,
                    i32::try_from(rows).map_err(|e| e.to_string())?,
                    p.cols as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    *buffers.weight.device_ptr() as *const std::ffi::c_void,
                    bs::cudaDataType::CUDA_R_16BF,
                    p.cols as i32,
                    ip as *const std::ffi::c_void,
                    bs::cudaDataType::CUDA_R_16BF,
                    p.cols as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    op as *mut std::ffi::c_void,
                    bs::cudaDataType::CUDA_R_16BF,
                    p.rows as i32,
                    bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        let tile = if kind=="tensorcomp" {64} else if matches!(kind, "f64" | "compensated") {
            4
        } else {
            16
        };
        let x = rows.div_ceil(tile);
        let nt = if matches!(kind, "f64" | "compensated") {
            p.rows
        } else {
            p.rows.div_ceil(if kind=="tensorcomp" {64}else{16})
        };
        let y = nt.min(my);
        let z = nt.div_ceil(y);
        if x > mx || z > mz {
            return Err("actual CUDA grid limits exceeded".into());
        }
        unsafe {
            fun.clone().launch(
                LaunchConfig {
                    grid_dim: (x as u32, y as u32, z as u32),
                    block_dim: (
                        if matches!(kind, "f64" | "compensated") {
                            width as u32
                        } else if kind=="tensorcomp" {512} else {
                            32
                        },
                        1,
                        1,
                    ),
                    shared_mem_bytes: if matches!(kind, "f64" | "compensated") {
                        (width * 8) as u32
                    } else {
                        0
                    },
                },
                (
                    ip,
                    *buffers.weight.device_ptr(),
                    op,
                    p.rows as i32,
                    p.cols as i32,
                    rows as u64,
                ),
            )
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    };
    let reset = || -> Result<(), String> {
        dev.synchronize().map_err(|e| e.to_string())?;
        unsafe {
            sys::lib().cuMemcpyHtoD_v2(
                *buffers.output.device_ptr(),
                image.as_ptr() as *const std::ffi::c_void,
                mul(image.len(), 2)?,
            )
        }
        .result()
        .map_err(|e| e.to_string())
    };
    let read = || -> Result<Vec<u16>, String> {
        dev.synchronize().map_err(|e| e.to_string())?;
        let v = dev
            .dtoh_sync_copy(&buffers.output)
            .map_err(|e| e.to_string())?;
        if v[count..] != image[count..] || v[..count].iter().any(|v| !bf(*v).is_finite()) {
            return Err("output nonfinite or suffix guard changed".into());
        }
        Ok(v)
    };
    enqueue(0, p.tokens)?;
    let full = read()?;
    write(&out.join("full.bf16"), &full[..count])?;
    if profiling {
        let timing = profile::samples(&dev, || enqueue(0, p.tokens))?;
        let after = read()?;
        if after != full || dev.dtoh_sync_copy(&buffers.input).map_err(|e| e.to_string())? != input
            || dev.dtoh_sync_copy(&buffers.weight).map_err(|e| e.to_string())? != weight {
            return Err("profiling changed output or read-only operands".into());
        }
        write(&out.join("after-profile.bf16"), &after[..count])?;
        return Ok(json!({"passed":true,"measurementPassed":true,"timing":timing,
            "outputExactBeforeAfter":true,"inputsExact":true,"suffixGuardsExact":true,
            "caseName":p.case_name,"rows":p.rows,"cols":p.cols,"tokens":p.tokens,"kernel":kind,
            "gpuUuid":actual,"gpuName":dev.name().map_err(|e|e.to_string())?,
            "measuredFreeAfterAllocationBytes":free,"runtimeBlockWidth":if kind=="native" {None} else if matches!(kind,"f64"|"compensated"){Some(width)}else{Some(if kind=="tensorcomp" {512}else{32})},"runtimeActiveBlocks":if matches!(kind,"f64"|"compensated"){Some(active_blocks)}else{None},
            "safetyMarginBytes":p.safety_margin_bytes,"source":p.source,"ptx":p.ptx,"origin":p.origin,
            "input":p.input,"weight":p.weight,"reference":p.reference,"referenceReport":p.reference_report,
            "binary":Identity::of(&std::env::current_exe().map_err(|e|e.to_string())?)?,
            "scope":"instrumented operator cost only; partition/numerical acceptance remains separate in frozen50 controls; not serving speed/benchmark/ADQ"}));
    }
    let mut controls = Vec::new();
    for (i, parts) in p.partitions.iter().enumerate() {
        reset()?;
        let mut at = 0;
        for n in parts {
            enqueue(at, *n)?;
            at += n;
        }
        let v = read()?;
        write(&out.join(format!("partition-{i}.bf16")), &v[..count])?;
        let exact = v == full;
        reset()?;
        graph_owner::completed_graph_launch(&dev, || {
            let mut at = 0;
            for n in parts {
                enqueue(at, *n)?;
                at += n;
            }
            Ok(())
        })?;
        let g = read()?;
        write(&out.join(format!("partition-{i}-graph.bf16")), &g[..count])?;
        controls.push(json!({"partition":parts,"exact":exact,"graphExact":g==full}));
    }
    reset()?;
    for t in 0..p.tokens {
        enqueue(t, 1)?;
    }
    let single = read()?;
    write(&out.join("singletons.bf16"), &single[..count])?;
    reset()?;
    graph_owner::completed_graph_launch(&dev, || enqueue(0, p.tokens))?;
    let graph = read()?;
    write(&out.join("graph.bf16"), &graph[..count])?;
    reset()?;
    let rejection = graph_owner::completed_graph_launch(&dev, || {
        enqueue(0, p.tokens)?;
        Err("intentional post-enqueue rejection".into())
    });
    if rejection.is_ok() || read()? != image {
        return Err("rejected graph replayed or changed guard".into());
    }
    enqueue(0, p.tokens)?;
    let recovery = read()?;
    if recovery != full {
        return Err("exact recovery failed".into());
    }
    if dev
        .dtoh_sync_copy(&buffers.input)
        .map_err(|e| e.to_string())?
        != input
        || dev
            .dtoh_sync_copy(&buffers.weight)
            .map_err(|e| e.to_string())?
            != weight
    {
        return Err("read-only operand changed".into());
    }
    let ordered = |v: u16| {
        if v & 0x8000 != 0 {
            (!v) as i32
        } else {
            (v as i32) + 0x8000
        }
    };
    let max_ulp = full[..count]
        .iter()
        .zip(&reference)
        .map(|(x, y)| (ordered(*x) - ordered(*y)).unsigned_abs())
        .max()
        .ok_or("empty result")?;
    let different = full[..count]
        .iter()
        .zip(&reference)
        .filter(|(x, y)| x != y)
        .count();
    let partition = controls
        .iter()
        .all(|v| v["exact"] == true && v["graphExact"] == true)
        && single == full
        && graph == full;
    Ok(
        json!({"passed":partition&&max_ulp<=1,"collectionPassed":true,"partitionControlsPassed":partition,"numericalTargetPassed":max_ulp<=1,"maxBf16Ulp":max_ulp,"differentFromF64Rne":different,"controls":controls,"singletonsExact":single==full,"graphExact":graph==full,"postEnqueueRejection":rejection.err(),"exactRecovery":true,"inputsExact":true,"suffixGuardsExact":true,"caseName":p.case_name,"rows":p.rows,"cols":p.cols,"tokens":p.tokens,"kernel":kind,"gpuUuid":actual,"gpuName":dev.name().map_err(|e|e.to_string())?,"measuredFreeAfterAllocationBytes":free,"runtimeBlockWidth":if kind=="native"{None}else{Some(if matches!(kind,"f64"|"compensated"){width}else if kind=="tensorcomp"{512}else{32})},"runtimeActiveBlocks":if matches!(kind,"f64"|"compensated"){Some(active_blocks)}else{None},"safetyMarginBytes":p.safety_margin_bytes,"source":p.source,"ptx":p.ptx,"origin":p.origin,"input":p.input,"weight":p.weight,"reference":p.reference,"referenceReport":p.reference_report,"binary":Identity::of(&std::env::current_exe().map_err(|e|e.to_string())?)?,"scope":"unadopted captured/constructed BF16 operator controls, not serving/Witness/model speed or ADQ"}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RefPlan {
    case_name: String,
    rows: usize,
    cols: usize,
    tokens: usize,
    input: Identity,
    weight: Identity,
}
fn reference(plan: &Path, out: &Path) -> Result<Value, String> {
    let p: RefPlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if p.rows == 0 || p.cols == 0 || p.tokens == 0 {
        return Err("empty reference matrix".into());
    }
    let input = u16s(p.input.read()?)?;
    let weight = u16s(p.weight.read()?)?;
    if input.len() != mul(p.tokens, p.cols)? || weight.len() != mul(p.rows, p.cols)? {
        return Err("reference operand extent mismatch".into());
    }
    let mut values = vec![0u16; mul(p.tokens, p.rows)?];
    for t in 0..p.tokens {
        for n in 0..p.rows {
            let mut sum = 0.0f64;
            for k in 0..p.cols {
                sum = bf(weight[n * p.cols + k]).mul_add(bf(input[t * p.cols + k]), sum);
            }
            values[t * p.rows + n] = rne(sum)?;
        }
    }
    write(&out.join("reference.bf16"), &values)?;
    Ok(
        json!({"passed":true,"caseName":p.case_name,"rows":p.rows,"cols":p.cols,"tokens":p.tokens,"input":p.input,"weight":p.weight,"reference":Identity::of(&out.join("reference.bf16"))?,"binary":Identity::of(&std::env::current_exe().map_err(|e|e.to_string())?)?,"scope":"CPU F64 mathematical reference of actual/declared constructed BF16 operands, F32 then BF16 rounding; no GPU/model/Witness/performance qualification"}),
    )
}

fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("use ./dev bf16-tensor-comp-probe".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("expected build|wmma|f64|native plan fresh-output".into());
    }
    let out = Path::new(&args[2]);
    if let Some(kind) = args[0].strip_prefix("sanitize-") {
        if !matches!(kind, "wmma" | "f64" | "compensated" | "native" | "chunked" | "tensorcomp") {
            return Err("unknown sanitized operator".into());
        }
        let p: Plan = serde_json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        p.sanitizer.read()?;
        let version = Command::new(&p.sanitizer.path)
            .arg("--version")
            .output()
            .map_err(|e| e.to_string())?;
        if !version.status.success() {
            return Err("sanitizer version check failed".into());
        }
        let status = Command::new(&p.sanitizer.path)
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
            .arg(kind)
            .arg(&args[1])
            .arg(out)
            .status()
            .map_err(|e| e.to_string())?;
        save(
            &out.join("SANITIZER.json"),
            &json!({"sanitizer":p.sanitizer,"version":String::from_utf8_lossy(&version.stdout),"status":status.code(),"tool":"memcheck","leakCheck":"full","errorExitcode":97}),
        )?;
        if !status.success() {
            return Err(format!(
                "sanitized operator failed {status}; evidence retained"
            ));
        }
        return Ok(());
    }
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let result = match args[0].as_str() {
        "build" => compile(Path::new(&args[1]), out),
        "reference" => reference(Path::new(&args[1]), out),
        "wmma" | "f64" | "compensated" | "native" | "chunked" | "tensorcomp" => execute(Path::new(&args[1]), out, &args[0], false),
        mode if mode.starts_with("profile-") && matches!(&mode[8..], "wmma" | "f64" | "compensated" | "native" | "chunked" | "tensorcomp") => execute(Path::new(&args[1]), out, &mode[8..], true),
        _ => Err("unknown explicit operator".into()),
    };
    match result {
        Ok(v) => {
            save(&out.join("REPORT.json"), &v)?;
            if v["passed"] == false {
                return Err("operator controls failed; evidence retained".into());
            }
            println!("{v}");
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
