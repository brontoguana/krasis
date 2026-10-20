#[path = "../../../src/dsa_score.rs"]
mod dsa_score;
// Retained-row discriminator. Other query rows repeat the captured row; uncaptured
// future keys are zero controls. This is not a full serving-input reconstruction.
mod graph_owner;
use cudarc::{
    cublas::{result as br, sys as bs},
    driver::{sys, CudaDevice, DevicePtr},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
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
            return Err(format!("identity {}", self.path.display()));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    graph: bool,
    name: String,
    rows: usize,
    context: usize,
    row: usize,
    head_batch: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    gpu_uuid: String,
    score_ptx: Identity,
    score_source: Identity,
    query: Identity,
    keys: Identity,
    weights: Identity,
    ptx: Identity,
    provenance: Identity,
    heads: usize,
    dim: usize,
    prefix: usize,
    scale: f32,
    cases: Vec<Case>,
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
fn checked_product(v: &[usize]) -> Result<usize, String> {
    v.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b).ok_or("dimension product overflow".into())
    })
}
struct Blas(bs::cublasHandle_t);
impl Drop for Blas {
    fn drop(&mut self) {
        let _ = unsafe { br::destroy_handle(self.0) };
    }
}
fn run(plan: &Path, out: &Path) -> Result<(), String> {
    let raw = fs::read(plan).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if out.exists() {
        return Err("fresh output required".into());
    }
    p.provenance.read()?;
    let qb = p.query.read()?;
    let kb = p.keys.read()?;
    let wb = p.weights.read()?;
    let ptx = String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?;
    if p.heads == 0
        || p.dim == 0
        || p.prefix == 0
        || !p.scale.is_finite()
        || p.scale <= 0.0
        || p.cases.is_empty()
    {
        return Err("invalid plan geometry".into());
    }
    if qb.len() != checked_product(&[p.heads, p.dim, 2])?
        || kb.len() != checked_product(&[p.prefix, p.dim, 2])?
        || wb.len() != p.heads * 2
    {
        return Err("actual input extent".into());
    }
    for x in &p.cases {
        if x.rows == 0
            || x.row >= x.rows
            || x.context < p.prefix
            || x.head_batch == 0
            || x.head_batch > p.heads
            || !x
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return Err("invalid case".into());
        }
        for v in [x.rows, x.context, p.heads * p.dim, p.dim] {
            i32::try_from(v).map_err(|_| "cuBLAS dimension exceeds i32")?;
        }
        checked_product(&[x.rows, x.context, x.head_batch, 4])?;
    }
    let d = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *d.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let s = uuid
        .bytes
        .iter()
        .map(|x| format!("{:02x}", *x as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &s[..8],
        &s[8..12],
        &s[12..16],
        &s[16..20],
        &s[20..]
    );
    if actual != p.gpu_uuid {
        return Err("GPU identity mismatch".into());
    }
    d.load_ptx(
        ptx.into(),
        "frozen132",
        &["dsa_prefill_accumulate_gemm_scores_kernel"],
    )
    .map_err(|e| e.to_string())?;
    let epilogue = d
        .get_func("frozen132", "dsa_prefill_accumulate_gemm_scores_kernel")
        .ok_or("epilogue absent")?;
    p.score_source.read()?;
    let score_code = String::from_utf8(p.score_ptx.read()?).map_err(|e| e.to_string())?;
    let score_program = dsa_score::Program::load(d.clone(), &score_code)?;
    let score_kernel = score_program.kernel();
    let blas = Blas(br::create_handle().map_err(|e| e.to_string())?);
    unsafe { br::set_stream(blas.0, *d.cu_stream() as bs::cudaStream_t) }
        .map_err(|e| e.to_string())?;
    let guard = d
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        .map_err(|e| e.to_string())? as usize;
    let sentinel = f32::from_bits(0x7fc01234);
    let weight = u16s(&wb);
    let dw = d.htod_sync_copy(&weight).map_err(|e| e.to_string())?;
    let pos = d
        .htod_sync_copy(&[(p.prefix - 1) as i32])
        .map_err(|e| e.to_string())?;
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let mut reports = Vec::new();
    for case in &p.cases {
        let q = u16s(&qb).repeat(case.rows);
        let mut k = u16s(&kb);
        k.resize(case.context * p.dim, 0);
        let elements = checked_product(&[case.rows, case.context, case.head_batch])?;
        let need = checked_product(&[q.len() + k.len(), 2])?
            .checked_add(checked_product(&[elements + 2 * guard, 4])?)
            .and_then(|x| x.checked_add(p.heads * p.prefix * 8))
            .ok_or("peak overflow")?;
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < need + 600 * 1024 * 1024 {
            return Err(
                "actual free VRAM below dimension-derived buffers plus default600MB".into(),
            );
        }
        let dq = d.htod_sync_copy(&q).map_err(|e| e.to_string())?;
        let dk = d.htod_sync_copy(&k).map_err(|e| e.to_string())?;
        let temp = d
            .htod_sync_copy(&vec![sentinel; elements + 2 * guard])
            .map_err(|e| e.to_string())?;
        let mut target = vec![0f32; p.heads * p.prefix];
        for start in (0..p.heads).step_by(case.head_batch) {
            let count = (p.heads - start).min(case.head_batch);
            let launch_score = || unsafe {
                score_kernel.enqueue(
                    *dk.device_ptr(),
                    *dq.device_ptr(),
                    *temp.device_ptr() + (guard * 4) as u64,
                    case.rows,
                    case.context,
                    p.heads,
                    p.dim,
                    start,
                    count,
                    p.scale,
                    *d.cu_stream(),
                )
            };
            if case.graph {
                graph_owner::completed_graph_launch(&d, launch_score)?;
            } else {
                launch_score()?;
            }
            d.synchronize().map_err(|e| e.to_string())?;
            for h in 0..count {
                let ptr = *temp.device_ptr()
                    + ((guard + h * case.rows * case.context + case.row * case.context) * 4) as u64;
                unsafe {
                    sys::lib().cuMemcpyDtoH_v2(
                        target[(start + h) * p.prefix..].as_mut_ptr().cast(),
                        ptr,
                        p.prefix * 4,
                    )
                }
                .result()
                .map_err(|e| e.to_string())?;
            }
        }
        let t = d.dtoh_sync_copy(&temp).map_err(|e| e.to_string())?;
        if t[..guard]
            .iter()
            .chain(&t[guard + elements..])
            .any(|x| x.to_bits() != sentinel.to_bits())
        {
            return Err("GEMM allocation guard changed".into());
        }
        if d.dtoh_sync_copy(&dq).map_err(|e| e.to_string())? != q
            || d.dtoh_sync_copy(&dk).map_err(|e| e.to_string())? != k
        {
            return Err("GEMM readonly input changed".into());
        }
        fs::write(out.join(format!("{}-heads.f32", case.name)), bytes(&target))
            .map_err(|e| e.to_string())?;
        let mut accumulations = Vec::new();
        // Replay frozen production epilogue on the cropped real row, varying only head grouping.
        for grouping in [1, p.heads] {
            let output = d
                .htod_sync_copy(&vec![sentinel; p.prefix + 2 * guard])
                .map_err(|e| e.to_string())?;
            let input = d.htod_sync_copy(&target).map_err(|e| e.to_string())?;
            for start in (0..p.heads).step_by(grouping) {
                let count = (p.heads - start).min(grouping);
                let (mut e0, mut e1, mut e2, mut e3) = (
                    *output.device_ptr() + (guard * 4) as u64,
                    *input.device_ptr() + (start * p.prefix * 4) as u64,
                    *dw.device_ptr(),
                    *pos.device_ptr(),
                );
                let (
                    mut e4,
                    mut e5,
                    mut e6,
                    mut e7,
                    mut e8,
                    mut e9,
                    mut e10,
                    mut e11,
                    mut e12,
                    mut e13,
                ) = (
                    1i32,
                    p.prefix as i32,
                    p.prefix as i32,
                    p.heads as i32,
                    start as i32,
                    count as i32,
                    p.prefix as i32,
                    if start == 0 { 1i32 } else { 0i32 },
                    0i32,
                    1i32,
                );
                macro_rules! arg {
                    ($v:ident) => {
                        &mut $v as *mut _ as *mut std::ffi::c_void
                    };
                }
                unsafe {
                    sys::lib().cuLaunchKernel(
                        epilogue.cu_function(),
                        p.prefix.div_ceil(256) as u32,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        *d.cu_stream(),
                        [
                            arg!(e0),
                            arg!(e1),
                            arg!(e2),
                            arg!(e3),
                            arg!(e4),
                            arg!(e5),
                            arg!(e6),
                            arg!(e7),
                            arg!(e8),
                            arg!(e9),
                            arg!(e10),
                            arg!(e11),
                            arg!(e12),
                            arg!(e13),
                        ]
                        .as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                }
                .result()
                .map_err(|e| e.to_string())?;
            }
            d.synchronize().map_err(|e| e.to_string())?;
            let all = d.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
            if all[..guard]
                .iter()
                .chain(&all[guard + p.prefix..])
                .any(|x| x.to_bits() != sentinel.to_bits())
            {
                return Err("epilogue guard changed".into());
            }
            let b = bytes(&all[guard..guard + p.prefix]);
            let name = format!("{}-group{}.f32", case.name, grouping);
            fs::write(out.join(&name), &b).map_err(|e| e.to_string())?;
            if bytes(&d.dtoh_sync_copy(&input).map_err(|e| e.to_string())?) != bytes(&target) {
                return Err("epilogue readonly target changed".into());
            }
            accumulations.push(json!({"grouping":grouping,"file":name,"sha256":hash(&b)}));
        }
        reports.push(json!({"case":case.name,"rows":case.rows,"context":case.context,"target_row":case.row,"head_batch":case.head_batch,"heads_sha256":hash(&bytes(&target)),"accumulations":accumulations,"guards_and_readonly_passed":true}));
    }
    if d.dtoh_sync_copy(&dw).map_err(|e| e.to_string())? != weight {
        return Err("head weights mutated".into());
    }
    let report = json!({"passed":true,"gpu_uuid":actual,"plan_sha256":hash(&raw),"cases":reports,"scope":"Matching captured causal-row operands in repeated-row/padded-future-key controls. Fixed K16 compensated score contraction with actual strided head input/scaled F32 output; frozen production PTX head epilogue. Not complete original physical inputs, serving correctness, speed, fidelity or ADQ acceptance."});
    fs::write(
        out.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") || args.len() != 3 {
        eprintln!("Use ./dev dsa-score-probe <plan.json> <fresh-output>");
        std::process::exit(2)
    }
    if let Err(e) = run(Path::new(&args[1]), Path::new(&args[2])) {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
