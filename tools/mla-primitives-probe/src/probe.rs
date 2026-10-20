use super::{hash, identity, Identity};
use crate::{graph_owner, mla_f32::MlaF32Ops};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr};
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::Path, sync::Arc};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    sanitizer: Identity,
    sanitizer_binary: Identity,
    gpu_uuid: String,
    source: Identity,
    driver: Identity,
    ptx: Identity,
    native_ptx: Identity,
    provenance: Identity,
    rows: usize,
    heads: usize,
    nope: usize,
    rope: usize,
    rank: usize,
    stride: usize,
    ccd: usize,
    norm_width: usize,
    norm_stride: usize,
    base: usize,
    interleaved: bool,
    eps: f32,
    qa: Identity,
    gamma: Identity,
    query: Identity,
    kv: Identity,
    kv_gamma: Identity,
    wkc: Identity,
    cos: Identity,
    sin: Identity,
    positions: Identity,
    table_rows: usize,
    partitions: Vec<usize>,
}
fn f32s(b: Vec<u8>) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err("F32 extent".into());
    }
    let v: Vec<_> = b
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite input".into());
    }
    Ok(v)
}
fn u16s(b: Vec<u8>) -> Result<Vec<u16>, String> {
    if b.len() % 2 != 0 {
        return Err("BF16 extent".into());
    }
    Ok(b.chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect())
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn bf(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
struct Owners {
    dev: Arc<CudaDevice>,
    qa: CudaSlice<f32>,
    kv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    norm_bf: CudaSlice<u16>,
    q_bf: CudaSlice<u16>,
    out: CudaSlice<f32>,
    ckv: CudaSlice<u8>,
    kpe: Option<CudaSlice<u8>>,
    gamma: CudaSlice<f32>,
    kg: CudaSlice<f32>,
    w: CudaSlice<u16>,
    cos: Option<CudaSlice<f32>>,
    sin: Option<CudaSlice<f32>>,
    pos: Option<CudaSlice<i32>>,
}
impl Drop for Owners {
    fn drop(&mut self) {
        if let Err(e) = self
            .dev
            .bind_to_thread()
            .and_then(|_| self.dev.synchronize())
        {
            eprintln!("FATAL: primitive owner completion unknown: {e}");
            std::process::abort()
        }
    }
}
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("FATAL native temporary completion unknown: {e}");
            std::process::abort()
        }
    }
}
fn ptr<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}
unsafe fn launch(
    d: &Arc<CudaDevice>,
    name: &str,
    blocks: usize,
    args: &mut [*mut std::ffi::c_void],
) -> Result<(), String> {
    let f = d
        .get_func("native57", name)
        .ok_or("native function absent")?;
    sys::lib()
        .cuLaunchKernel(
            f.cu_function(),
            blocks as u32,
            1,
            1,
            256,
            1,
            1,
            0,
            *d.cu_stream(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
        .result()
        .map_err(|e| e.to_string())
}
// Explicit typed parameter tuples avoid host pointer lifetime ambiguity.
macro_rules! native {($d:expr,$name:expr,$blocks:expr; $($v:ident),+)=>{launch($d,$name,$blocks,&mut[$(&mut $v as *mut _ as _),+])}}
#[derive(PartialEq)]
struct ResultData {
    qa: Vec<u32>,
    kv: Vec<u32>,
    q: Vec<u32>,
    norm: Vec<u16>,
    qb: Vec<u16>,
    out: Vec<u32>,
    c: Vec<u8>,
    k: Vec<u8>,
}
fn collect(o: &Owners) -> Result<ResultData, String> {
    let d = &o.dev;
    let rd = |x: &CudaSlice<f32>| -> Result<Vec<u32>, String> {
        Ok(d.dtoh_sync_copy(x)
            .map_err(|e| e.to_string())?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    Ok(ResultData {
        qa: rd(&o.qa)?,
        kv: rd(&o.kv)?,
        q: rd(&o.q)?,
        norm: d.dtoh_sync_copy(&o.norm_bf).map_err(|e| e.to_string())?,
        qb: d.dtoh_sync_copy(&o.q_bf).map_err(|e| e.to_string())?,
        out: rd(&o.out)?,
        c: d.dtoh_sync_copy(&o.ckv).map_err(|e| e.to_string())?,
        k: match &o.kpe {
            Some(k) => d.dtoh_sync_copy(k).map_err(|e| e.to_string())?,
            None => vec![],
        },
    })
}
fn save(r: &ResultData, root: &Path, prefix: &str) -> Result<(), String> {
    for (n, b) in [
        (
            "qa",
            r.qa.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        ("kv", r.kv.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ("query", r.q.iter().flat_map(|v| v.to_le_bytes()).collect()),
        (
            "norm",
            r.norm.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        (
            "query-bf",
            r.qb.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        (
            "absorb",
            r.out.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        ("ckv", r.c.clone()),
        ("kpe", r.k.clone()),
    ] {
        fs::write(root.join(format!("{prefix}-{n}.raw")), b).map_err(|e| e.to_string())?;
    }
    Ok(())
}
pub(super) fn run(plan: &str, out: &str) -> Result<(), String> {
    let raw = fs::read(plan).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    p.source.read()?;
    p.driver.read()?;
    p.provenance.read()?;
    let ptx = p.ptx.read()?;
    let native_ptx = p.native_ptx.read()?;
    if p.rows == 0
        || p.partitions.iter().sum::<usize>() != p.rows
        || p.partitions.contains(&0)
        || p.rope % 16 != 0
        || p.rank > p.ccd
        || p.ccd % 16 != 0
        || p.stride < p.rank + p.rope
        || p.norm_stride < p.norm_width
        || p.heads == 0
    {
        return Err("explicit probe geometry".into());
    }
    let qa = f32s(p.qa.read()?)?;
    let kv = f32s(p.kv.read()?)?;
    let q = f32s(p.query.read()?)?;
    let g = f32s(p.gamma.read()?)?;
    let kg = f32s(p.kv_gamma.read()?)?;
    let w = u16s(p.wkc.read()?)?;
    let cs = f32s(p.cos.read()?)?;
    let sn = f32s(p.sin.read()?)?;
    let pb = p.positions.read()?;
    if pb.len() % 4 != 0 {
        return Err("positions extent".into());
    }
    let positions: Vec<i32> = pb
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let qw = p.nope + p.rope;
    let n = p.rows;
    let cap = p.base + n + 2;
    if qa.len() != n * p.norm_stride
        || g.len() != p.norm_width
        || kv.len() != n * p.stride
        || kg.len() != p.rank
        || q.len() != n * p.heads * qw
        || w.len() != p.heads * p.nope * p.ccd
        || cs.len() != p.table_rows * p.rope / 2
        || sn.len() != cs.len()
        || positions.len() != n
        || positions
            .iter()
            .any(|&x| x < 0 || x as usize >= p.table_rows)
    {
        return Err("actual typed operand geometry/position admission".into());
    }
    let dev = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let h = uuid
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
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
        return Err("actual GPU UUID mismatch".into());
    }
    let ops = MlaF32Ops::load(
        dev.clone(),
        std::str::from_utf8(&ptx).map_err(|e| e.to_string())?,
    )?;
    dev.load_ptx(
        std::str::from_utf8(&native_ptx)
            .map_err(|e| e.to_string())?
            .into(),
        "native57",
        &[
            "per_head_rmsnorm",
            "mla_split_q",
            "mla_deinterleave",
            "apply_rope",
            "mla_absorb_wkc",
            "mla_kv_cache_write_k4",
            "mla_kv_cache_write_k6",
        ],
    )
    .map_err(|e| e.to_string())?;
    let free = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    // Account actual peak allocations (owners plus native temporary Q splits), before allocation.
    let required = qa.len() * 4
        + kv.len() * 4
        + q.len() * 4
        + n * p.norm_width * 2
        + n * p.heads * qw * 2
        + n * p.heads * p.ccd * 4
        + cap * (p.ccd + p.rope) / 16 * 14
        + g.len() * 4
        + kg.len() * 4
        + w.len() * 2
        + (cs.len() + sn.len()) * 4
        + if p.rope > 0 { n * 4 } else { 0 }
        + n * p.heads * qw * 4
        + 8 * 4
        + 8 * 2
        + 8 * 2
        + 16
        + if p.rope > 0 { 16 } else { 0 };
    if free < required + 600 * 1024 * 1024 {
        return Err(
            "measured free VRAM cannot cover actual primitive buffers and default safety margin"
                .into(),
        );
    }
    let root = Path::new(out);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let mut results = Vec::new();
    for format in [9u32, 7u32] {
        let bytes = if format == 9 { 10 } else { 14 };
        let csize = cap * p.ccd / 16 * bytes;
        let ksize = cap * p.rope / 16 * bytes;
        let hf = vec![f32::from_bits(0x7fc000a5); n * p.heads * p.ccd + 8];
        let hnb = vec![0xa5a5; n * p.norm_width + 8];
        let hqb = vec![0xa5a5; n * p.heads * qw + 8];
        let hc = vec![0xa5; csize + 16];
        let hk = vec![0xa5; ksize + 16];
        let mut o = Owners {
            dev: dev.clone(),
            qa: dev.htod_sync_copy(&qa).map_err(|e| e.to_string())?,
            kv: dev.htod_sync_copy(&kv).map_err(|e| e.to_string())?,
            q: dev.htod_sync_copy(&q).map_err(|e| e.to_string())?,
            norm_bf: dev.htod_sync_copy(&hnb).map_err(|e| e.to_string())?,
            q_bf: dev.htod_sync_copy(&hqb).map_err(|e| e.to_string())?,
            out: dev.htod_sync_copy(&hf).map_err(|e| e.to_string())?,
            ckv: dev.htod_sync_copy(&hc).map_err(|e| e.to_string())?,
            kpe: if p.rope > 0 {
                Some(dev.htod_sync_copy(&hk).map_err(|e| e.to_string())?)
            } else {
                None
            },
            gamma: dev.htod_sync_copy(&g).map_err(|e| e.to_string())?,
            kg: dev.htod_sync_copy(&kg).map_err(|e| e.to_string())?,
            w: dev.htod_sync_copy(&w).map_err(|e| e.to_string())?,
            cos: if p.rope > 0 {
                Some(dev.htod_sync_copy(&cs).map_err(|e| e.to_string())?)
            } else {
                None
            },
            sin: if p.rope > 0 {
                Some(dev.htod_sync_copy(&sn).map_err(|e| e.to_string())?)
            } else {
                None
            },
            pos: if p.rope > 0 {
                Some(dev.htod_sync_copy(&positions).map_err(|e| e.to_string())?)
            } else {
                None
            },
        };
        let optional = |s: &Option<CudaSlice<f32>>| s.as_ref().map(ptr).unwrap_or(0);
        let enqueue = |o: &Owners, start: usize, count: usize| -> Result<(), String> {
            unsafe {
                let st = *dev.cu_stream();
                ops.norm(
                    ptr(&o.qa) + (start * p.norm_stride * 4) as u64,
                    ptr(&o.gamma),
                    ptr(&o.norm_bf) + (start * p.norm_width * 2) as u64,
                    count,
                    p.norm_width,
                    p.norm_stride,
                    p.norm_width,
                    p.eps,
                    st,
                )?;
                ops.norm(
                    ptr(&o.kv) + (start * p.stride * 4) as u64,
                    ptr(&o.kg),
                    0,
                    count,
                    p.rank,
                    p.stride,
                    0,
                    p.eps,
                    st,
                )?;
                ops.prepare_query(
                    ptr(&o.q) + (start * p.heads * qw * 4) as u64,
                    ptr(&o.kv) + (start * p.stride * 4) as u64,
                    ptr(&o.q_bf) + (start * qw * 2) as u64,
                    n,
                    optional(&o.cos),
                    optional(&o.sin),
                    o.pos
                        .as_ref()
                        .map(|v| ptr(v) + (start * 4) as u64)
                        .unwrap_or(0),
                    p.table_rows,
                    count,
                    p.heads,
                    p.nope,
                    p.rope,
                    p.rank,
                    p.stride,
                    p.interleaved,
                    st,
                )?;
                ops.cache(
                    ptr(&o.ckv),
                    o.kpe.as_ref().map(ptr).unwrap_or(0),
                    ptr(&o.kv) + (start * p.stride * 4) as u64,
                    count,
                    p.base + start,
                    cap,
                    p.rank,
                    p.stride,
                    p.ccd,
                    p.rope,
                    format,
                    st,
                )?;
                ops.absorb(
                    ptr(&o.out) + (start * p.heads * p.ccd * 4) as u64,
                    ptr(&o.q) + (start * p.heads * qw * 4) as u64,
                    ptr(&o.w),
                    count,
                    p.heads,
                    p.nope,
                    p.rope,
                    p.ccd,
                    st,
                )?;
            }
            Ok(())
        };
        enqueue(&o, 0, n)?;
        let full = collect(&o)?;
        save(&full, root, &format!("k{format}-full"))?;
        let reset = |o: &mut Owners| -> Result<(), String> {
            dev.htod_sync_copy_into(&qa, &mut o.qa)
                .map_err(|e| e.to_string())?;
            dev.htod_sync_copy_into(&kv, &mut o.kv)
                .map_err(|e| e.to_string())?;
            dev.htod_sync_copy_into(&q, &mut o.q)
                .map_err(|e| e.to_string())?;
            dev.htod_sync_copy_into(&hnb, &mut o.norm_bf)
                .map_err(|e| e.to_string())?;
            dev.htod_sync_copy_into(&hqb, &mut o.q_bf)
                .map_err(|e| e.to_string())?;
            dev.htod_sync_copy_into(&hf, &mut o.out)
                .map_err(|e| e.to_string())?;
            dev.htod_sync_copy_into(&hc, &mut o.ckv)
                .map_err(|e| e.to_string())?;
            if let Some(k) = &mut o.kpe {
                dev.htod_sync_copy_into(&hk, k).map_err(|e| e.to_string())?;
            }
            Ok(())
        };
        let initial = {
            reset(&mut o)?;
            collect(&o)?
        };
        let prefix_matches = |v: &ResultData, end: usize| -> bool {
            let rowcmp = |a: &[u32], f: &[u32], i: &[u32], stride: usize| {
                a[..end * stride] == f[..end * stride] && a[end * stride..] == i[end * stride..]
            };
            rowcmp(&v.qa, &full.qa, &initial.qa, p.norm_stride)
                && rowcmp(&v.kv, &full.kv, &initial.kv, p.stride)
                && rowcmp(&v.q, &full.q, &initial.q, p.heads * qw)
                && rowcmp(&v.out, &full.out, &initial.out, p.heads * p.ccd)
                && v.norm[..end * p.norm_width] == full.norm[..end * p.norm_width]
                && v.norm[end * p.norm_width..] == initial.norm[end * p.norm_width..]
                && (0..p.heads).all(|h| {
                    let at = h * n * qw;
                    v.qb[at..at + end * qw] == full.qb[at..at + end * qw]
                        && v.qb[at + end * qw..at + n * qw]
                            == initial.qb[at + end * qw..at + n * qw]
                })
                && v.qb[n * p.heads * qw..] == initial.qb[n * p.heads * qw..]
                && [
                    (
                        v.c.as_slice(),
                        full.c.as_slice(),
                        initial.c.as_slice(),
                        p.ccd,
                    ),
                    (
                        v.k.as_slice(),
                        full.k.as_slice(),
                        initial.k.as_slice(),
                        p.rope,
                    ),
                ]
                .iter()
                .all(|(a, f, i, d)| {
                    let from = p.base * d / 16 * bytes;
                    let to = (p.base + end) * d / 16 * bytes;
                    a[..from] == i[..from] && a[from..to] == f[from..to] && a[to..] == i[to..]
                })
        };
        for parts in [p.partitions.clone(), vec![1; n]] {
            reset(&mut o)?;
            let mut start = 0;
            for count in parts {
                enqueue(&o, start, count)?;
                start += count;
                let v = collect(&o)?;
                if !prefix_matches(&v, start) {
                    save(&v, root, "failed-prefix")?;
                    return Err(format!("whole prefix/suffix comparison failed at {start}"));
                }
            }
            if collect(&o)? != full {
                return Err("partition complete equality".into());
            }
        }
        reset(&mut o)?;
        graph_owner::completed_graph_launch(&dev, || enqueue(&o, 0, n))?;
        if collect(&o)? != full {
            return Err("graph differs".into());
        }
        reset(&mut o)?;
        let rejected = graph_owner::completed_graph_launch(&dev, || {
            enqueue(&o, 0, 1)?;
            unsafe {
                ops.cache(
                    ptr(&o.ckv),
                    o.kpe.as_ref().map(ptr).unwrap_or(0),
                    ptr(&o.kv),
                    n,
                    p.base,
                    p.base + n - 1,
                    p.rank,
                    p.stride,
                    p.ccd,
                    p.rope,
                    format,
                    *dev.cu_stream(),
                )
            }
        })
        .is_err();
        if !rejected || collect(&o)? != initial {
            return Err("rejected captured graph was replayed".into());
        }
        enqueue(&o, 0, n)?;
        if collect(&o)? != full {
            return Err("recovery differs".into());
        }
        // Independently run frozen native single-row kernels on identical operands.
        reset(&mut o)?;
        let qn = dev
            .htod_sync_copy(&vec![0f32; n * p.heads * p.nope])
            .map_err(|e| e.to_string())?;
        let mut qp = if p.rope > 0 {
            Some(
                dev.htod_sync_copy(&vec![0f32; n * p.heads * p.rope])
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let _native_completion = Completion(dev.clone());
        for row in 0..n {
            unsafe {
                let mut data = ptr(&o.qa) + (row * p.norm_stride * 4) as u64;
                let mut weight = ptr(&o.gamma);
                let mut eps = p.eps;
                let mut heads = 1i32;
                let mut width = p.norm_width as i32;
                let mut perhead = 0i32;
                native!(&dev,"per_head_rmsnorm",1;data,weight,eps,heads,width,perhead)?;
                data = ptr(&o.kv) + (row * p.stride * 4) as u64;
                weight = ptr(&o.kg);
                width = p.rank as i32;
                native!(&dev,"per_head_rmsnorm",1;data,weight,eps,heads,width,perhead)?;
                let mut q_nope = ptr(&qn) + (row * p.heads * p.nope * 4) as u64;
                let mut q_pe = qp
                    .as_ref()
                    .map(|x| ptr(x) + (row * p.heads * p.rope * 4) as u64)
                    .unwrap_or(0);
                let mut query = ptr(&o.q) + (row * p.heads * qw * 4) as u64;
                heads = p.heads as i32;
                let mut nope = p.nope as i32;
                let mut rope = p.rope as i32;
                native!(&dev,"mla_split_q",(p.heads*qw).div_ceil(256);q_nope,q_pe,query,heads,nope,rope)?;
                let mut k_pe = if p.rope > 0 {
                    ptr(&o.kv) + ((row * p.stride + p.rank) * 4) as u64
                } else {
                    0
                };
                if p.rope > 0 {
                    if p.interleaved {
                        let mut total = (p.heads * p.rope) as i32;
                        native!(&dev,"mla_deinterleave",(p.heads*p.rope).div_ceil(256);q_pe,total,rope)?;
                        total = p.rope as i32;
                        native!(&dev,"mla_deinterleave",p.rope.div_ceil(256);k_pe,total,rope)?;
                    }
                    let mut cos = optional(&o.cos);
                    let mut sin = optional(&o.sin);
                    let mut position = positions[row];
                    let mut kh = 1i32;
                    let mut half = rope / 2;
                    native!(&dev,"apply_rope",((p.heads+1)*p.rope/2).div_ceil(256);q_pe,k_pe,cos,sin,position,heads,kh,rope,half)?;
                }
                let mut absorb = ptr(&o.out) + (row * p.heads * p.ccd * 4) as u64;
                let mut wc = ptr(&o.w);
                let mut ccd = p.ccd as i32;
                native!(&dev,"mla_absorb_wkc",p.heads;absorb,q_nope,wc,heads,nope,ccd)?;
                let mut c = ptr(&o.ckv);
                let mut k = o.kpe.as_ref().map(ptr).unwrap_or(0);
                let mut kc = ptr(&o.kv) + (row * p.stride * 4) as u64;
                let mut position = (p.base + row) as i32;
                let mut rank = p.rank as i32;
                native!(&dev,if format==9{"mla_kv_cache_write_k4"}else{"mla_kv_cache_write_k6"},(p.ccd.max(p.rope)/16).div_ceil(256);c,k,kc,k_pe,position,rank,ccd,rope)?;
            }
        }
        let native_result = collect(&o)?;
        let native_qn = dev.dtoh_sync_copy(&qn).map_err(|e| e.to_string())?;
        let native_qp = match qp.as_ref() {
            Some(v) => dev.dtoh_sync_copy(v).map_err(|e| e.to_string())?,
            None => vec![],
        };
        let mut normalized_native = native_result; // Native pipeline leaves Q split, and has no BF16 co-output.
        for row in 0..n {
            for h in 0..p.heads {
                for j in 0..qw {
                    let val = if j < p.nope {
                        native_qn[(row * p.heads + h) * p.nope + j]
                    } else {
                        native_qp[(row * p.heads + h) * p.rope + j - p.nope]
                    };
                    normalized_native.q[(row * p.heads + h) * qw + j] = val.to_bits();
                    normalized_native.qb[(h * n + row) * qw + j] = bf(val);
                }
            }
            for j in 0..p.norm_width {
                normalized_native.norm[row * p.norm_width + j] = bf(f32::from_bits(
                    normalized_native.qa[row * p.norm_stride + j],
                ));
            }
        }
        save(&normalized_native, root, &format!("k{format}-native"))?;
        if normalized_native != full {
            return Err(format!("native parity failed k{format}: qa={} kv={} query={} norm={} queryBF={} absorb={} ckv={} kpe={}",normalized_native.qa==full.qa,normalized_native.kv==full.kv,normalized_native.q==full.q,normalized_native.norm==full.norm,normalized_native.qb==full.qb,normalized_native.out==full.out,normalized_native.c==full.c,normalized_native.k==full.k));
        }
        let readonly = same(
            &dev.dtoh_sync_copy(&o.gamma).map_err(|e| e.to_string())?,
            &g,
        ) && same(&dev.dtoh_sync_copy(&o.kg).map_err(|e| e.to_string())?, &kg)
            && dev.dtoh_sync_copy(&o.w).map_err(|e| e.to_string())? == w
            && match (&o.cos, &o.sin, &o.pos) {
                (Some(c), Some(s), Some(pos)) => {
                    same(&dev.dtoh_sync_copy(c).map_err(|e| e.to_string())?, &cs)
                        && same(&dev.dtoh_sync_copy(s).map_err(|e| e.to_string())?, &sn)
                        && dev.dtoh_sync_copy(pos).map_err(|e| e.to_string())? == positions
                }
                (None, None, None) => p.rope == 0,
                _ => false,
            };
        let padding = (0..n).all(|r| {
            full.qa[r * p.norm_stride + p.norm_width..(r + 1) * p.norm_stride]
                == initial.qa[r * p.norm_stride + p.norm_width..(r + 1) * p.norm_stride]
                && full.kv[r * p.stride + p.rank + p.rope..(r + 1) * p.stride]
                    == initial.kv[r * p.stride + p.rank + p.rope..(r + 1) * p.stride]
        });
        if !readonly || !padding || !prefix_matches(&full, n) {
            return Err("readonly/padding/guards failed".into());
        }
        drop(_native_completion);
        drop(qp.take()); // all native work completed by downloads, owners remain live.
        results.push(json!({"format":format,"nativeExact":true,"allPrefixesPartitionsSingletonsGraphRecoveryExact":true,"rejectedGraphNotReplayed":true,"readonlyAndPadding":true}));
    }
    let report = json!({"passed":true,"gpuUuid":actual,"planSha256":hash(&raw),"driver":identity(&p.driver.path)?,"source":identity(&p.source.path)?,"ptx":identity(&p.ptx.path)?,"nativePtx":identity(&p.native_ptx.path)?,"binary":identity(&std::env::current_exe().map_err(|e|e.to_string())?)?,"requiredGpuBytes":required,"freeBeforeAllocationBytes":free,"results":results,"scope":"F32 primitive operator controls only. Explicit actual/cropped tensor operands. No serving, model fidelity, speed or ADQ qualification."});
    fs::write(
        root.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}

pub(super) fn sanitize(plan: &str, out: &str) -> Result<(), String> {
    let p: Plan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    p.sanitizer.read()?;
    p.sanitizer_binary.read()?;
    let version = std::process::Command::new(&p.sanitizer.path)
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !version.status.success() {
        return Err("sanitizer version failed".into());
    }
    eprintln!("{}", String::from_utf8_lossy(&version.stdout));
    let status = std::process::Command::new(&p.sanitizer.path)
        .args([
            "--tool",
            "memcheck",
            "--leak-check",
            "full",
            "--error-exitcode",
            "99",
        ])
        .arg(std::env::current_exe().map_err(|e| e.to_string())?)
        .arg("probe")
        .arg(plan)
        .arg(out)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("sanitized primitive controls failed: {status}"));
    }
    Ok(())
}
