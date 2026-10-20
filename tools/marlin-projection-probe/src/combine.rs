//! Retained real expert outputs/route weights; F32 accumulation contract controls.
use super::*;
#[path = "../../../src/common_moe_combine.rs"]
mod common_moe_combine;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CombinePlan {
    source: Identity,
    ptx: Identity,
    build: Identity,
    experts: Vec<Identity>,
    shared: Identity,
    weights: Identity,
    gate: Identity,
    width: usize,
    gpu_ordinal: usize,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    raw_limit: usize,
}
fn save_raw(
    out: &Path,
    name: &str,
    bytes: &[u8],
    raw: &mut usize,
    limit: usize,
) -> Result<(), String> {
    *raw = raw
        .checked_add(bytes.len())
        .filter(|&v| v <= limit)
        .ok_or("combine raw cap exceeded")?;
    fs::write(out.join(name), bytes).map_err(|e| e.to_string())
}
pub fn execute(path: &Path, out: &Path) -> Result<Value, String> {
    let p: CombinePlan = read_json(path)?;
    p.source.read()?;
    let ptx = String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?;
    let b: Value = serde_json::from_slice(&p.build.read()?).map_err(|e| e.to_string())?;
    if b["passed"] != true
        || b["source"] != serde_json::to_value(&p.source).unwrap()
        || b["ptx"] != serde_json::to_value(&p.ptx).unwrap()
    {
        return Err("combine build identity mismatch".into());
    }
    let experts = p
        .experts
        .iter()
        .map(|id| u16s(&id.read()?))
        .collect::<Result<Vec<_>, _>>()?;
    let shared = u16s(&p.shared.read()?)?;
    let wb = p.weights.read()?;
    let gb = p.gate.read()?;
    let topk = experts.len();
    if p.width == 0
        || p.width < 17
        || topk == 0
        || wb.len() != topk * 4
        || gb.len() < 4
        || experts.iter().any(|x| x.len() < p.width)
        || shared.len() < p.width
    {
        return Err("combine actual operand geometry".into());
    }
    let weights: Vec<_> = wb
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let gate = f32::from_le_bytes(gb[..4].try_into().unwrap());
    if !gate.is_finite() || weights.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite combine route data".into());
    }
    let widths = [p.width, 17, 1];
    let rows = 17usize;
    let prospective = widths
        .iter()
        .sum::<usize>()
        .checked_mul((rows * 24 + 4) * 9)
        .ok_or("combine prospective overflow")?;
    if prospective > p.raw_limit {
        return Err("combine prospective cap exceeded".into());
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
        return Err("combine GPU identity mismatch".into());
    }
    let program = common_moe_combine::Program::load(dev.clone(), &ptx)?;
    let mut raw = 0usize;
    let mut records = Vec::new();
    for width in widths {
        for mask in 0..3 {
            for scale in [1f32, 0.5, 1.25] {
                let key = format!("n{width}-mask{mask}-scale{scale}");
                let actual_weights: Vec<_> = weights
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| {
                        if mask == 2 || (mask == 1 && i % 2 == 0) {
                            0.
                        } else {
                            w
                        }
                    })
                    .collect();
                let aw=&actual_weights;
                let e_host: Vec<_> = experts
                    .iter()
                    .enumerate()
                    .flat_map(|(i, v)| {
                        v[..width]
                            .iter()
                            .map(move |&v| if aw[i] == 0. { 0x7fc1 } else { v })
                    })
                    .collect();
                let e = dev.htod_copy(e_host.repeat(rows)).map_err(|e| e.to_string())?;
                let w = dev
                    .htod_copy(actual_weights.repeat(rows))
                    .map_err(|e| e.to_string())?;
                let s = dev
                    .htod_copy(shared[..width].repeat(rows))
                    .map_err(|e| e.to_string())?;
                let g = dev.htod_copy(vec![gate; rows]).map_err(|e| e.to_string())?;
                let r = dev
                    .htod_copy(vec![f32::from_bits(0x7fc12345); rows * width + 32])
                    .map_err(|e| e.to_string())?;
                let mirror = dev
                    .htod_copy(vec![0x7fc1u16; rows * width + 32])
                    .map_err(|e| e.to_string())?;
                let output = dev
                    .htod_copy(vec![0x7fc1u16; rows * width + 32])
                    .map_err(|e| e.to_string())?;
                let combined_f32=dev.htod_copy(vec![f32::from_bits(0x7fc12345);rows*width+32]).map_err(|e|e.to_string())?;
                let _completion = Completion(dev.clone());
                let reference: Vec<_> = (0..width)
                    .map(|n| {
                        let mut sum = 0f32;
                        for i in 0..topk {
                            let weight = actual_weights[i] * scale;
                            if weight != 0. {
                                let v = f32::from_bits((experts[i][n] as u32) << 16);
                                sum = v.mul_add(weight, sum);
                            }
                        }
                        sum
                    })
                    .collect();
                let launch_reduce = |start: usize, tokens: usize| unsafe {
                    program.enqueue_reduce(
                        *e.device_ptr() + (start * topk * width * 2) as u64,
                        *w.device_ptr() + (start * topk * 4) as u64,
                        *r.device_ptr() + (start * width * 4) as u64,
                        *mirror.device_ptr() + (start * width * 2) as u64,
                        width,
                        topk,
                        tokens,
                        scale,
                        *dev.cu_stream(),
                    )
                };
                let mut baseline_r = None;
                let mut baseline_b = None;
                for mode in ["full", "singletons", "partition", "graph", "error"] {
                    unsafe {
                        sys::lib().cuMemsetD32Async(
                            *r.device_ptr(),
                            0x7fc12345,
                            rows * width + 32,
                            *dev.cu_stream(),
                        )
                    }
                    .result()
                    .map_err(|e| e.to_string())?;
                    unsafe {
                        sys::lib().cuMemsetD16Async(
                            *mirror.device_ptr(),
                            0x7fc1,
                            rows * width + 32,
                            *dev.cu_stream(),
                        )
                    }
                    .result()
                    .map_err(|e| e.to_string())?;
                    match mode {
                        "full" => launch_reduce(0, rows)?,
                        "singletons" => {
                            for row in 0..rows {
                                launch_reduce(row, 1)?;
                            }
                        }
                        "partition" => {
                            launch_reduce(0, 1)?;
                            launch_reduce(1, 15)?;
                            launch_reduce(16, 1)?;
                        }
                        "graph" => {
                            graph_owner::completed_graph_launch(&dev, || launch_reduce(0, rows))?
                        }
                        _ => {
                            if graph_owner::completed_graph_launch(&dev, || {
                                launch_reduce(0, rows)?;
                                Err("intentional reduce error".into())
                            })
                            .is_ok()
                            {
                                return Err("reduce error accepted".into());
                            }
                            launch_reduce(0, rows)?;
                        }
                    }
                    let rr = dev.dtoh_sync_copy(&r).map_err(|e| e.to_string())?;
                    let bb = dev.dtoh_sync_copy(&mirror).map_err(|e| e.to_string())?;
                    if rr[rows * width..].iter().any(|v| v.to_bits() != 0x7fc12345)
                        || bb[rows * width..].iter().any(|&v| v != 0x7fc1)
                    {
                        return Err("reduce suffix guard changed".into());
                    }
                    for row in 0..rows {
                        for n in 0..width {
                            if rr[row * width + n].to_bits() != reference[n].to_bits()
                                || bb[row * width + n]
                                    != half::bf16::from_f32(reference[n]).to_bits()
                            {
                                return Err(
                                    "reduce differs exact CPU F32-FMA/BF16 reference".into()
                                );
                            }
                        }
                    }
                    let bits: Vec<_> = rr.iter().map(|v| v.to_bits()).collect();
                    if let Some(old) = &baseline_r {
                        if *old != bits || *baseline_b.as_ref().unwrap() != bb {
                            return Err("reduce batch/graph/error differs".into());
                        }
                    } else {
                        baseline_r = Some(bits);
                        baseline_b = Some(bb);
                    }
                }
                let rr = dev.dtoh_sync_copy(&r).map_err(|e| e.to_string())?;
                let bb = baseline_b.unwrap();
                save_raw(
                    out,
                    &format!("{key}-routed.f32"),
                    &rr[..rows * width]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                    &mut raw,
                    p.raw_limit,
                )?;
                save_raw(
                    out,
                    &format!("{key}-mirror.bf16"),
                    &bb[..rows * width]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                    &mut raw,
                    p.raw_limit,
                )?;
                save_raw(
                    out,
                    &format!("{key}-cpu.f32"),
                    &reference
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                    &mut raw,
                    p.raw_limit,
                )?;
                let mut combined = Vec::new();
                for branch in 0..3 {
                    let launch = |start: usize, tokens: usize| unsafe {
                        program.enqueue_combine(
                            *r.device_ptr() + (start * width * 4) as u64,
                            if branch == 0 {
                                0
                            } else {
                                *s.device_ptr() + (start * width * 2) as u64
                            },
                            if branch == 2 {
                                *g.device_ptr() + (start * 4) as u64
                            } else {
                                0
                            },
                            *output.device_ptr() + (start * width * 2) as u64,
                            *combined_f32.device_ptr()+(start*width*4)as u64,
                            width,
                            tokens,
                            1.,
                            *dev.cu_stream(),
                        )
                    };
                    let mut baseline = None;
                    let mut baseline_c=None;
                    let mut nearest = 0;
                    let mut max_steps = 0;
                    for mode in ["full", "singletons", "graph", "error"] {
                        unsafe {
                            sys::lib().cuMemsetD16Async(
                                *output.device_ptr(),
                                0x7fc1,
                                rows * width + 32,
                                *dev.cu_stream(),
                            )
                        }
                        .result()
                        .map_err(|e| e.to_string())?;
                        unsafe{sys::lib().cuMemsetD32Async(*combined_f32.device_ptr(),0x7fc12345,rows*width+32,*dev.cu_stream())}.result().map_err(|e|e.to_string())?;
                        match mode {
                            "full" => launch(0, rows)?,
                            "singletons" => {
                                for row in 0..rows {
                                    launch(row, 1)?;
                                }
                            }
                            "graph" => {
                                graph_owner::completed_graph_launch(&dev, || launch(0, rows))?
                            }
                            _ => {
                                if graph_owner::completed_graph_launch(&dev, || {
                                    launch(0, rows)?;
                                    Err("intentional combine error".into())
                                })
                                .is_ok()
                                {
                                    return Err("combine error accepted".into());
                                }
                                launch(0, rows)?;
                            }
                        }
                        let y = dev.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
                        let cf=dev.dtoh_sync_copy(&combined_f32).map_err(|e|e.to_string())?;
                        if cf[rows*width..].iter().any(|v|v.to_bits()!=0x7fc12345){return Err("combined F32 guard changed".into());}
                        for (i,&v) in cf[..rows*width].iter().enumerate(){if !v.is_finite()||half::bf16::from_f32(v).to_bits()!=y[i]{return Err("combined F32/BF16 result contract differs".into());}}
                        let cb:Vec<_>=cf.iter().map(|v|v.to_bits()).collect();if let Some(b)=&baseline_c{if *b!=cb{return Err("combined F32 placement differs".into());}}else{baseline_c=Some(cb);}
                        if y[rows * width..].iter().any(|&v| v != 0x7fc1) {
                            return Err("combine suffix guard changed".into());
                        }
                        if let Some(old) = &baseline {
                            if *old != y {
                                return Err("combine placement/graph/error differs".into());
                            }
                        } else {
                            baseline = Some(y);
                        }
                    }
                    let y = baseline.unwrap();
                    let cf=dev.dtoh_sync_copy(&combined_f32).map_err(|e|e.to_string())?;
                    // Production prefill adds shared into its existing F32 accumulator.
                    // Exercise that exact alias and the F32-only output ABI on real owners.
                    for graph in [false,true] {
                        unsafe{sys::lib().cuMemcpyHtoDAsync_v2(*r.device_ptr(),rr.as_ptr().cast(),rr.len()*4,*dev.cu_stream())}.result().map_err(|e|e.to_string())?;
                        let alias=||unsafe{program.enqueue_combine(*r.device_ptr(),if branch==0{0}else{*s.device_ptr()},if branch==2{*g.device_ptr()}else{0},0,*r.device_ptr(),width,rows,1.,*dev.cu_stream())};
                        if graph{graph_owner::completed_graph_launch(&dev,alias)?;}else{alias()?;}
                        if dev.dtoh_sync_copy(&r).map_err(|e|e.to_string())?.iter().zip(&cf).any(|(x,y)|x.to_bits()!=y.to_bits()){return Err("inplace F32 combine differs disjoint output".into());}
                    }
                    unsafe{sys::lib().cuMemcpyHtoDAsync_v2(*r.device_ptr(),rr.as_ptr().cast(),rr.len()*4,*dev.cu_stream())}.result().map_err(|e|e.to_string())?;
                    save_raw(out,&format!("{key}-branch{branch}.f32"),&cf[..rows*width].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>(),&mut raw,p.raw_limit)?;
                    let ordered = |b: u16| {
                        if b & 0x8000 != 0 {
                            0x8000i32 - (b & 0x7fff) as i32
                        } else {
                            0x8000 + b as i32
                        }
                    };
                    for row in 0..rows {
                        for n in 0..width {
                            let ref64 = reference[n] as f64
                                + if branch == 0 {
                                    0.
                                } else {
                                    f32::from_bits((shared[n] as u32) << 16) as f64
                                        * if branch == 2 {
                                            1. / (1. + (-(gate as f64)).exp())
                                        } else {
                                            1.
                                        }
                                };
                            let target = reference_rounding::bf16_from_f64_bits(ref64);
                            let steps = (ordered(y[row * width + n]) - ordered(target)).abs();
                            nearest += usize::from(steps != 0);
                            max_steps = max_steps.max(steps);
                            if branch==1 && y[row*width+n]!=half::bf16::from_f32(f32::from_bits((shared[n]as u32)<<16).mul_add(1.,reference[n])).to_bits(){return Err("combined output differs exact CPU F32-FMA contract".into());}
                            if steps > if branch == 0 { 0 } else { 1 } {
                                return Err(format!("combine independent F64 bound {key} branch{branch} steps{steps}"));
                            }
                        }
                    }
                    save_raw(
                        out,
                        &format!("{key}-branch{branch}.bf16"),
                        &y[..rows * width]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                        &mut raw,
                        p.raw_limit,
                    )?;
                    combined.push(json!({"branch":branch,"fullSingletonGraphErrorExact":true,"inplaceF32OnlyEagerGraphExact":true,"nearestF64Mismatch":nearest,"maxF64BF16Steps":max_steps}));
                }
                if dev.dtoh_sync_copy(&e).map_err(|e|e.to_string())?!=e_host.repeat(rows) || dev.dtoh_sync_copy(&r).map_err(|e|e.to_string())?.iter().zip(&rr).any(|(x,y)|x.to_bits()!=y.to_bits()){return Err("combine changed expert/routed readonly inputs".into());}
                if dev.dtoh_sync_copy(&w).map_err(|e| e.to_string())? != actual_weights.repeat(rows)
                    || dev.dtoh_sync_copy(&s).map_err(|e| e.to_string())?
                        != shared[..width].repeat(rows)
                    || dev.dtoh_sync_copy(&g).map_err(|e| e.to_string())? != vec![gate; rows]
                {
                    return Err("combine readonly operands changed".into());
                }
                let free = cudarc::driver::result::mem_get_info()
                    .map_err(|e| e.to_string())?
                    .0;
                if free < p.safety_margin_bytes {
                    return Err("combine measured VRAM below margin".into());
                }
                records.push(json!({"width":width,"mask":mask,"scale":scale,"exactCpuF32FmaAllCoordinates":true,"placementGraphErrorGuardsExact":true,"combined":combined}));
            }
        }
    }
    if raw != prospective {
        return Err(format!(
            "combine raw extent reconciliation {raw}!={prospective}"
        ));
    }
    p.source.read()?;
    p.ptx.read()?;
    Ok(
        json!({"passed":true,"servingQualified":false,"program":program.identity,"rawBytes":raw,"records":records,"scope":"Retained actual expert output rows/route weights; duplicate real rows for placement only, explicit zero-mask staleNaN negative controls. Exact CPU F32-FMA routed outputs, unshared BF16nearestF64 exact, shared/gated <=1BF16step, ungatedshared exactCPUF32FMA. No model/calibration/performanceacceptance."}),
    )
}
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("FATAL combine completion unknown:{e}");
            std::process::abort()
        }
    }
}
