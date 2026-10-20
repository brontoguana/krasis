#[path = "../../../../src/cuda_region.rs"]
mod cuda_region;
#[path = "../../../bf16-projection-probe/src/graph_owner.rs"]
mod graph_owner;
#[path = "../../../../src/stable_mla_gemm.rs"]
mod stable_mla_gemm;
#[path = "../../../../src/stable_mla_ops.rs"]
mod stable_mla_ops;
use cudarc::driver::{sys, CudaDevice, DevicePtr, DeviceSlice};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use stable_mla_gemm::Span;
use stable_mla_ops::Ops;
use std::{fs, path::PathBuf};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
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
            return Err("rotary copy input identity".into());
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    gpu_uuid: String,
    ptx: Identity,
    provenance: Identity,
    query: Identity,
    rows: usize,
    heads: usize,
    nope: usize,
    rope: usize,
    partitions: Vec<usize>,
}
fn serial(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn run(plan: &str, out: &str) -> Result<(), String> {
    let raw = fs::read(plan).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    p.provenance.read()?;
    let root = PathBuf::from(out);
    if root.exists() {
        return Err("fresh rotary preservation output required".into());
    }
    if p.rows == 0 || p.heads == 0 || p.rope == 0 || p.partitions.iter().sum::<usize>() != p.rows {
        return Err("rotary preservation geometry".into());
    }
    let d = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *d.cu_device()) }
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
        return Err("rotary preservation physical GPU".into());
    }
    let ops = Ops::load(
        d.clone(),
        &String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?,
    )?;
    let bytes = p.query.read()?;
    if bytes.len() != p.rows * p.heads * (p.nope + p.rope) * 4 {
        return Err("actual query extent".into());
    }
    let source: Vec<_> = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let guard = usize::try_from(
        d.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let count = p.rows * p.heads * p.rope;
    let nan = f32::from_bits(0x7fc12345);
    let expected: Vec<_> = source
        .chunks_exact(p.nope + p.rope)
        .flat_map(|x| x[p.nope..].iter().copied())
        .collect();
    let needed = source.len() * 4 + (count + 2 * guard) * 4;
    if cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0
        < needed + 600 * 1024 * 1024
    {
        return Err("rotary preservation actual free VRAM below required extent and margin".into());
    }
    let src = d.htod_sync_copy(&source).map_err(|e| e.to_string())?;
    let dst = d
        .htod_sync_copy(&vec![nan; count + 2 * guard])
        .map_err(|e| e.to_string())?;
    let enqueue = |start: usize, rows: usize| -> Result<(), String> {
        let input = Span::f32(
            &src,
            start * p.heads * (p.nope + p.rope),
            rows * p.heads * (p.nope + p.rope),
        )?;
        let output = Span::f32(
            &dst,
            guard + start * p.heads * p.rope,
            rows * p.heads * p.rope,
        )?;
        unsafe {
            ops.pack_rope(
                &input,
                &output,
                rows,
                p.heads,
                p.nope,
                p.rope,
                *d.cu_stream(),
            )
        }
    };
    let verify = || -> Result<(), String> {
        d.synchronize().map_err(|e| e.to_string())?;
        let v = d.dtoh_sync_copy(&dst).map_err(|e| e.to_string())?;
        if serial(&v[guard..guard + count]) != serial(&expected)
            || v[..guard]
                .iter()
                .chain(&v[guard + count..])
                .any(|v| v.to_bits() != nan.to_bits())
        {
            return Err("rotary preservation complete bytes/guard differ".into());
        }
        Ok(())
    };
    let reset = || -> Result<(), String> {
        unsafe {
            sys::lib().cuMemsetD32Async(*dst.device_ptr(), nan.to_bits(), dst.len(), *d.cu_stream())
        }
        .result()
        .map_err(|e| e.to_string())
    };
    graph_owner::completed_eager(&d, &(&src, &dst, &ops), |_| enqueue(0, p.rows))?;
    verify()?;
    reset()?;
    let mut start = 0;
    for &rows in &p.partitions {
        graph_owner::completed_eager(&d, &(&src, &dst, &ops), |_| enqueue(start, rows))?;
        start += rows
    }
    verify()?;
    reset()?;
    for row in 0..p.rows {
        graph_owner::completed_eager(&d, &(&src, &dst, &ops), |_| enqueue(row, 1))?
    }
    verify()?;
    reset()?;
    graph_owner::completed_graph_launch(&d, || enqueue(0, p.rows))?;
    verify()?;
    reset()?;
    let error = graph_owner::completed_eager(&d, &(&src, &dst, &ops), |_| {
        enqueue(0, p.rows)?;
        Err::<(), String>("injected rotary preservation post-enqueue error".into())
    })
    .err()
    .ok_or("post-enqueue error missing")?;
    if !error.contains("injected rotary") {
        return Err(error);
    }
    verify()?;
    reset()?;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = graph_owner::completed_eager(&d, &(&src, &dst, &ops), |_| -> Result<(), String> {
            enqueue(0, p.rows)?;
            panic!("injected rotary preservation eager unwind")
        });
    }));
    if unwind.is_ok() {
        return Err("rotary preservation unwind absent".into());
    }
    verify()?;
    let input = Span::f32(&src, 0, source.len())?;
    let alias = unsafe {
        ops.pack_rope(
            &input,
            &input,
            p.rows,
            p.heads,
            p.nope,
            p.rope,
            *d.cu_stream(),
        )
    }
    .err()
    .ok_or("rotary source alias admitted")?;
    if !alias.contains("aliases") {
        return Err(format!("wrong rotary alias rejection: {alias}"));
    }
    if serial(&d.dtoh_sync_copy(&src).map_err(|e| e.to_string())?) != bytes {
        return Err("rotary source changed".into());
    }
    fs::create_dir(&root).map_err(|e| e.to_string())?;
    let actual = d.dtoh_sync_copy(&dst).map_err(|e| e.to_string())?;
    let payload = serial(&actual[guard..guard + count]);
    if payload.len() != count * 4 {
        return Err("dimension-derived rotary raw bound differs".into());
    }
    fs::write(root.join("rope.f32"), &payload).map_err(|e| e.to_string())?;
    let r = json!({"passed":true,"gpu_uuid":actual_uuid(&uuid),"plan_sha256":hash(&raw),"full_partition_singleton_graph_exact":true,"actual_eager_error_unwind_completion":true,"source_readonly_exact":true,"guards_exact":true,"source_alias_rejected":alias,"raw_bytes":payload.len(),"raw_bound_bytes":count*4,"raw_sha256":hash(&payload),"scope":"Actual F32 rotary slice preservation/owner/completion operator only; no serving qualification"});
    fs::write(
        root.join("REPORT.json"),
        serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{r}");
    Ok(())
}
fn actual_uuid(uuid: &sys::CUuuid) -> String {
    let h = uuid
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect::<String>();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") || a.len() != 3 {
        eprintln!("Use ./dev stable-rope-copy-probe <plan> <fresh-output>");
        std::process::exit(2)
    }
    if let Err(e) = run(&a[1], &a[2]) {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
