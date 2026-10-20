//! Bounded native query-projection discriminator. No serving/default change.
use cudarc::{
    cublas::{result as br, sys as bs, CudaBlas},
    driver::{sys, CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
mod exact;
mod graph_owner;
#[rustfmt::skip]
mod native_strided;
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
fn hash(z: &[u8]) -> String {
    format!("{:x}", Sha256::digest(z))
}
impl Identity {
    fn read(&self) -> Result<Vec<u8>, String> {
        let b = fs::read(&self.path).map_err(|e| e.to_string())?;
        if b.len() != self.bytes || hash(&b) != self.sha256 {
            return Err(format!("identity mismatch {}", self.path.display()));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: usize,
    executable: Identity,
    origin: Identity,
    source_admission: Identity,
    input: Identity,
    weight: Identity,
    prefill: Identity,
    decode: Identity,
    rows: usize,
    cols: usize,
    batch_rows: usize,
    target_row: usize,
    gpu_uuid: String,
    source_gpu_uuid: String,
    safety_margin_bytes: usize,
    math_mode: u32,
    native_helper_sha256: String,
    ptx: Identity,
    kernel_source: Identity,
}
fn words(b: Vec<u8>) -> Result<Vec<u16>, String> {
    if b.len() % 2 != 0 {
        return Err("odd BF16 extent".into());
    }
    let v: Vec<_> = b
        .chunks_exact(2)
        .map(|z| u16::from_le_bytes([z[0], z[1]]))
        .collect();
    if v.iter().any(|v| v & 0x7f80 == 0x7f80) {
        return Err("nonfinite BF16 operand".into());
    }
    Ok(v)
}
fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn write(p: &Path, v: &[u16]) -> Result<(), String> {
    fs::write(p, bytes(v)).map_err(|e| e.to_string())
}
fn save(p: &Path, v: &serde_json::Value) -> Result<(), String> {
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
fn compare(x: &[u16], y: &[u16]) -> Result<serde_json::Value, String> {
    if x.len() != y.len() {
        return Err("comparison extent mismatch".into());
    }
    let order = |v: u16| if v & 0x8000 != 0 { !v } else { v | 0x8000 };
    let mut count = 0;
    let mut maximum = 0u16;
    let mut first = None;
    for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
        if x != y {
            count += 1;
            if first.is_none() {
                first = Some(i)
            }
        }
        maximum = maximum.max(order(x).abs_diff(order(y)));
    }
    Ok(
        json!({"elements":x.len(),"bitDifferences":count,"first":first,"maxBf16UlpDistance":maximum,"withinOneBf16Ulp":maximum<=1}),
    )
}
struct Resources {
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    weight: Option<CudaSlice<u16>>,
    single: Option<CudaSlice<u16>>,
    multi: Option<CudaSlice<u16>>,
    out_single: Option<CudaSlice<u16>>,
    out_multi: Option<CudaSlice<u16>>,
}
impl Drop for Resources {
    fn drop(&mut self) {
        if let Err(e) = self.dev.synchronize() {
            eprintln!("FATAL: query probe completion unknown; retaining owners until process termination: {e}");
            std::process::abort();
        }
    }
}
const GUARD_WORDS: usize = 32; // Test canary/alignment, not a runtime memory budget.
const CANARY: u16 = 0xa55a;
fn plain(r: &Resources, input: u64, output: u64, n: usize, k: usize) -> Result<(), String> {
    let n = i32::try_from(n).map_err(|_| "native N exceeds i32")?;
    let k = i32::try_from(k).map_err(|_| "native K exceeds i32")?;
    let (alpha, beta) = (1f32, 0f32);
    unsafe {
        br::set_stream(*r.blas.handle(), *r.dev.cu_stream() as bs::cudaStream_t)
            .map_err(|e| e.to_string())?;
        br::gemm_ex(
            *r.blas.handle(),
            bs::cublasOperation_t::CUBLAS_OP_T,
            bs::cublasOperation_t::CUBLAS_OP_N,
            n,
            1,
            k,
            (&alpha as *const f32).cast(),
            *r.weight.as_ref().ok_or("weight owner absent")?.device_ptr() as *const _,
            bs::cudaDataType::CUDA_R_16BF,
            k,
            input as *const _,
            bs::cudaDataType::CUDA_R_16BF,
            k,
            (&beta as *const f32).cast(),
            output as *mut _,
            bs::cudaDataType::CUDA_R_16BF,
            n,
            bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn run(plan_path: &Path, out: &Path) -> Result<(), String> {
    let raw = fs::read(plan_path).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if p.schema != 1
        || p.rows == 0
        || p.cols == 0
        || p.batch_rows == 0
        || p.target_row >= p.batch_rows
        || p.safety_margin_bytes == 0
        || p.math_mode != 0
    {
        return Err("unsupported explicit native query probe contract".into());
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    if fs::canonicalize(&p.executable.path).map_err(|e| e.to_string())?
        != fs::canonicalize(&executable).map_err(|e| e.to_string())?
    {
        return Err("running executable differs from admitted path".into());
    }
    p.executable.read()?;
    let origin: serde_json::Value =
        serde_json::from_slice(&p.origin.read()?).map_err(|e| e.to_string())?;
    if origin["attributionAllowed"] != true
        || origin["projectionInputsExact"] != true
        || origin["projectionOutputsExact"] != false
    {
        return Err("origin does not establish native query projection mismatch".into());
    }
    p.source_admission.read()?;
    let helper = include_str!("native_strided.rs");
    let needle = "pub(super) fn run_cublas_bf16_row_independent_gemm";
    let start = helper
        .find("#[allow(clippy::too_many_arguments)]")
        .ok_or("native helper prefix missing")?;
    if hash(
        helper[start..]
            .replace(needle, "fn run_cublas_bf16_row_independent_gemm")
            .as_bytes(),
    ) != p.native_helper_sha256
    {
        return Err("native helper differs from frozen serving source".into());
    }
    let input = words(p.input.read()?)?;
    let weight = words(p.weight.read()?)?;
    let observed_full = words(p.prefill.read()?)?;
    let observed_decode = words(p.decode.read()?)?;
    if input.len() != p.cols
        || weight.len() != p.rows.checked_mul(p.cols).ok_or("weight extent overflow")?
        || observed_full.len() != p.rows
        || observed_decode.len() != p.rows
    {
        return Err("captured logical extents mismatch".into());
    }
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let reference = weight
        .chunks_exact(p.cols)
        .map(|w| exact::dot(&input, w))
        .collect::<Result<Vec<_>, _>>()?;
    if reference.iter().any(|v| v & 0x7f80 == 0x7f80) {
        return Err("captured exact dot rounds to nonfinite BF16".into());
    }
    write(&out.join("exact-reference.bf16"), &reference)?;
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
        return Err(format!(
            "actual GPU {actual} differs from selected {}",
            p.gpu_uuid
        ));
    }
    if dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| e.to_string())?
        < 8
    {
        return Err("BF16 projection requires Ampere or later".into());
    }
    let (free_before, total) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    let multi_elements = p
        .batch_rows
        .checked_mul(p.cols)
        .ok_or("batch input overflow")?;
    let multi_output = p
        .batch_rows
        .checked_mul(p.rows)
        .ok_or("batch output overflow")?;
    let single_output = p
        .rows
        .checked_add(2 * GUARD_WORDS)
        .ok_or("single guard overflow")?;
    let multi_output_guarded = multi_output
        .checked_add(2 * GUARD_WORDS)
        .ok_or("batch guard overflow")?;
    let allocated_elements = [
        weight.len(),
        input.len(),
        multi_elements,
        single_output,
        multi_output_guarded,
    ]
    .into_iter()
    .try_fold(0usize, |a, b| {
        a.checked_add(b).ok_or("allocation extent overflow")
    })?;
    let allocation_bytes = allocated_elements
        .checked_mul(2)
        .ok_or("allocation bytes overflow")?;
    if free_before
        < allocation_bytes
            .checked_add(p.safety_margin_bytes)
            .ok_or("admission bytes overflow")?
    {
        return Err(
            "insufficient actual free VRAM for operands plus supplied safety margin".into(),
        );
    }
    let blas = CudaBlas::new(dev.clone()).map_err(|e| e.to_string())?;
    let mut r = Resources {
        dev: dev.clone(),
        blas,
        weight: None,
        single: None,
        multi: None,
        out_single: None,
        out_multi: None,
    };
    r.weight = Some(dev.htod_copy(weight.clone()).map_err(|e| e.to_string())?);
    r.single = Some(dev.htod_copy(input.clone()).map_err(|e| e.to_string())?);
    let mut full_input = vec![0u16; multi_elements];
    let target_at = p
        .target_row
        .checked_mul(p.cols)
        .ok_or("input row overflow")?;
    full_input[target_at..target_at + p.cols].copy_from_slice(&input);
    r.multi = Some(
        dev.htod_copy(full_input.clone())
            .map_err(|e| e.to_string())?,
    );
    r.out_single = Some(
        dev.htod_copy(vec![CANARY; single_output])
            .map_err(|e| e.to_string())?,
    );
    r.out_multi = Some(
        dev.htod_copy(vec![CANARY; multi_output_guarded])
            .map_err(|e| e.to_string())?,
    );
    let status = unsafe {
        bs::lib().cublasSetMathMode(*r.blas.handle(), bs::cublasMath_t::CUBLAS_DEFAULT_MATH)
    };
    if status != bs::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("set explicit observed math mode: {status:?}"));
    }
    p.kernel_source.read()?;
    let ptx = String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?;
    dev.load_ptx(ptx.into(), "query-comp", &["bf16_projection_tensorcomp"])
        .map_err(|e| e.to_string())?;
    let fun = dev
        .get_func("query-comp", "bf16_projection_tensorcomp")
        .ok_or("tensorcomp entry absent")?;
    let attr = |v| dev.attribute(v).map_err(|e| e.to_string());
    use sys::CUdevice_attribute as A;
    let mut function_threads = 0;
    unsafe {
        sys::lib().cuFuncGetAttribute(
            &mut function_threads,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            fun.cu_function(),
        )
    }
    .result()
    .map_err(|e| e.to_string())?;
    if function_threads < 512 || attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 512 {
        return Err("tensorcomp CTA unsupported".into());
    }
    let mut occupancy = 0;
    unsafe {
        sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &mut occupancy,
            fun.cu_function(),
            512,
            0,
        )
    }
    .result()
    .map_err(|e| e.to_string())?;
    if occupancy <= 0 {
        return Err("tensorcomp CTA has no admitted occupancy".into());
    }
    let grid_limits = (
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize,
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize,
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)? as usize,
    );
    let mut outputs = Vec::new();
    for kind in [
        "plain-single",
        "strided-single",
        "strided-full",
        "tensor-single",
        "tensor-full",
        "tensor-single-graph",
        "tensor-full-graph",
        "tensor-partition",
        "tensor-partition-graph",
    ] {
        let (full, input_ptr, output_owner, count) =
            if kind == "strided-full" || kind.contains("full") || kind.contains("partition") {
                (
                    true,
                    *r.multi.as_ref().ok_or("multi input absent")?.device_ptr(),
                    r.out_multi.as_ref().ok_or("multi output absent")?,
                    p.batch_rows,
                )
            } else {
                (
                    false,
                    *r.single.as_ref().ok_or("single input absent")?.device_ptr(),
                    r.out_single.as_ref().ok_or("single output absent")?,
                    1,
                )
            };
        unsafe {
            sys::lib().cuMemsetD16Async(
                *output_owner.device_ptr(),
                CANARY,
                if full {
                    multi_output_guarded
                } else {
                    single_output
                },
                *dev.cu_stream(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        let ptr = (*output_owner.device_ptr())
            .checked_add((GUARD_WORDS * 2) as u64)
            .ok_or("output pointer overflow")?;
        if kind.starts_with("tensor-") {
            let enqueue = |start: usize, rows: usize| -> Result<(), String> {
                if rows == 0 || start.checked_add(rows).is_none_or(|n| n > count) {
                    return Err("invalid tensor partition".into());
                }
                let gx = rows.div_ceil(64);
                let tiles = p.rows.div_ceil(64);
                let gy = tiles.min(grid_limits.1);
                let gz = tiles.div_ceil(gy);
                if gx > grid_limits.0 || gz > grid_limits.2 {
                    return Err("tensorcomp grid exceeds actual device".into());
                }
                let ip = input_ptr
                    .checked_add((start * p.cols * 2) as u64)
                    .ok_or("input address overflow")?;
                let op = ptr
                    .checked_add((start * p.rows * 2) as u64)
                    .ok_or("output address overflow")?;
                unsafe {
                    fun.clone().launch(
                        LaunchConfig {
                            grid_dim: (gx as u32, gy as u32, gz as u32),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            ip,
                            *r.weight.as_ref().ok_or("weight absent")?.device_ptr(),
                            op,
                            i32::try_from(p.rows).map_err(|e| e.to_string())?,
                            i32::try_from(p.cols).map_err(|e| e.to_string())?,
                            rows as u64,
                        ),
                    )
                }
                .map_err(|e| e.to_string())
            };
            let launch = || -> Result<(), String> {
                if kind.contains("partition") {
                    let begin = p
                        .target_row
                        .checked_sub(3)
                        .ok_or("partition needs target >=3")?;
                    let middle = 7usize;
                    if begin + middle >= count {
                        return Err("partition tail absent".into());
                    }
                    enqueue(0, begin)?;
                    enqueue(begin, middle)?;
                    enqueue(begin + middle, count - begin - middle)
                } else {
                    enqueue(0, count)
                }
            };
            if kind.ends_with("graph") {
                graph_owner::completed_graph_launch(&dev, launch)?;
            } else {
                launch()?;
            }
        } else if kind == "plain-single" {
            plain(&r, input_ptr, ptr, p.rows, p.cols)?;
        } else {
            native_strided::run_cublas_bf16_row_independent_gemm(
                *r.blas.handle(),
                *dev.cu_stream(),
                input_ptr,
                *r.weight.as_ref().ok_or("weight absent")?.device_ptr(),
                ptr,
                count,
                p.rows,
                p.cols,
            )?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;
        let raw_output = dev
            .dtoh_sync_copy(output_owner)
            .map_err(|e| e.to_string())?;
        let logical = if full { multi_output } else { p.rows };
        if raw_output[..GUARD_WORDS]
            .iter()
            .chain(raw_output[GUARD_WORDS + logical..].iter())
            .any(|v| *v != CANARY)
        {
            return Err(format!("{kind} changed an output guard"));
        }
        let begin = GUARD_WORDS + if full { p.target_row * p.rows } else { 0 };
        let target = raw_output[begin..begin + p.rows].to_vec();
        if full
            && raw_output[GUARD_WORDS..GUARD_WORDS + logical]
                .chunks_exact(p.rows)
                .enumerate()
                .any(|(row, v)| row != p.target_row && v.iter().any(|x| x & 0x7fff != 0))
        {
            return Err("strided full changed a declared zero control row".into());
        }
        if target.iter().any(|v| v & 0x7f80 == 0x7f80) {
            return Err(format!("{kind} produced nonfinite output"));
        }
        write(&out.join(format!("{kind}.bf16")), &target)?;
        outputs.push((kind, target));
    }
    if dev
        .dtoh_sync_copy(r.weight.as_ref().ok_or("weight absent")?)
        .map_err(|e| e.to_string())?
        != weight
        || dev
            .dtoh_sync_copy(r.single.as_ref().ok_or("single absent")?)
            .map_err(|e| e.to_string())?
            != input
        || dev
            .dtoh_sync_copy(r.multi.as_ref().ok_or("multi absent")?)
            .map_err(|e| e.to_string())?
            != full_input
    {
        return Err("native query probe changed read-only input or weights".into());
    }
    let (free_after, _) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    if free_after < p.safety_margin_bytes {
        return Err("actual free VRAM after completed operations below supplied margin".into());
    }
    let mut mode = 0u32;
    let status = unsafe {
        bs::lib().cublasGetMathMode(
            *r.blas.handle(),
            (&mut mode as *mut u32).cast::<bs::cublasMath_t>(),
        )
    };
    if status != bs::cublasStatus_t::CUBLAS_STATUS_SUCCESS || mode != p.math_mode {
        return Err("actual native math mode changed".into());
    }
    let cases=outputs.iter().map(|(kind,v)|Ok(json!({"kind":kind,"toExact":compare(v,&reference)?,"toObservedFull":compare(v,&observed_full)?,"toObservedDecode":compare(v,&observed_decode)?,"sha256":hash(&bytes(v))}))).collect::<Result<Vec<_>,String>>()?;
    let mut driver_version = 0i32;
    unsafe { sys::lib().cuDriverGetVersion(&mut driver_version) }
        .result()
        .map_err(|e| e.to_string())?;
    let mut cublas_version = 0i32;
    let version_status =
        unsafe { bs::lib().cublasGetVersion_v2(*r.blas.handle(), &mut cublas_version) };
    if version_status != bs::cublasStatus_t::CUBLAS_STATUS_SUCCESS
        || cublas_version <= 0
        || driver_version <= 0
    {
        return Err("actual CUDA/cuBLAS version query failed".into());
    }
    let report = json!({"collectionComplete":true,"cudaDriverVersion":driver_version,"cublasVersion":cublas_version,"planSha256":hash(&raw),"gpuUuid":actual,"sourceGpuUuid":p.source_gpu_uuid,"sameGpuAsCapture":actual==p.source_gpu_uuid,"deviceName":dev.name().map_err(|e|e.to_string())?,"mathMode":mode,"rows":p.rows,"cols":p.cols,"batchRows":p.batch_rows,"targetRow":p.target_row,"allocationBytes":allocation_bytes,"freeBefore":free_before,"freeAfterCompleted":free_after,"totalBytes":total,"safetyMarginBytes":p.safety_margin_bytes,"readonlyExact":true,"outputGuardsExact":true,"zeroControlRowsExact":true,"exactReferenceSha256":hash(&bytes(&reference)),"observedFullToExact":compare(&observed_full,&reference)?,"observedDecodeToExact":compare(&observed_decode,&reference)?,"cases":cases,"stridedBatchSingletonExact":outputs[1].1==outputs[2].1,"tensorAllControlsExact":outputs[3..].iter().all(|x|x.1==outputs[3].1),"tensorAllMatchExactReference":outputs[3..].iter().all(|x|x.1==reference),"kernelSource":p.kernel_source,"ptx":p.ptx,"servingQualified":false,"scope":"Native BF16 cuBLAS entry-point discriminator using exact captured vector/matrix; full batch is explicit zero embedding at original row, not a real prompt corpus. Direct integer exact-real BF16 RNE reference. Completed-boundary VRAM readings are not CUDA low-water samples. Captured graph replay and partition controls for the existing compensated kernel; no sanitizer, timing, throughput or serving acceptance."});
    save(&out.join("REPORT.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        eprintln!("ERROR: use ./dev dsa-query-probe");
        std::process::exit(1)
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = if args.len() != 2 {
        Err("expected plan and fresh output directory".into())
    } else {
        run(Path::new(&args[0]), Path::new(&args[1]))
    };
    if let Err(e) = result {
        eprintln!("ERROR: {e}");
        std::process::exit(1)
    }
}
