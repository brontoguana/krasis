//! Actual MLA HQQ6 operands through the real serving dual-output adapter.
//! Repeated actual tensor rows test batching, not text quality or model throughput.
#[path = "../../bf16-projection-probe/src/graph_owner.rs"]
mod graph_owner;
#[allow(dead_code)]
#[path = "../../../src/ordered_projection.rs"]
mod ordered_projection;
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
#[derive(Deserialize, Clone)]
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
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn identity(path: &Path) -> Result<Value, String> {
    let b = fs::read(path).map_err(|e| e.to_string())?;
    Ok(json!({"path":path,"bytes":b.len(),"sha256":hash(&b)}))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: usize,
    case_name: String,
    capture_plan: Identity,
    operand_audit: Identity,
    operand_inventory: Identity,
    weight_layout: String,
    reference_audit: Identity,
    bank_audit: Identity,
    source_freeze: Identity,
    shader_source: Identity,
    build_contract: Identity,
    ptx: Identity,
    bank_rust_source: Identity,
    source_bridge: Identity,
    sanitizer: Identity,
    sanitizer_binary: Identity,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    row_sources: Vec<usize>,
    partitions: Vec<Vec<usize>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Capture {
    schema: usize,
    layout: String,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_stride: usize,
    scales_stride: usize,
    zeros_stride: usize,
    packed: Identity,
    scales: Identity,
    zeros: Identity,
    materialized: Identity,
    prefill_input: Identity,
    decode_input: Identity,
    prefill_output: Identity,
    decode_output: Identity,
    decode_output_dtype: String,
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("operator size overflow".into())
}
fn u16s(b: &[u8]) -> Result<Vec<u16>, String> {
    if b.len() % 2 != 0 {
        return Err("odd BF16 extent".into());
    }
    let v: Vec<_> = b
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter()
        .any(|&x| !f32::from_bits((x as u32) << 16).is_finite())
    {
        return Err("nonfinite BF16 operand".into());
    }
    Ok(v)
}
fn f32s(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err("odd F32 extent".into());
    }
    let v: Vec<_> = b
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    // Captured coefficient padding has no numerical contract; validate logical groups only.
    Ok(v)
}
fn f64s(b: &[u8]) -> Result<Vec<f64>, String> {
    if b.len() % 8 != 0 {
        return Err("odd F64 extent".into());
    }
    let v: Vec<_> = b
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite F64 reference".into());
    }
    Ok(v)
}
fn rne(v: f64) -> u16 {
    let b = (v as f32).to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn ordered_bf(x: u16) -> u16 {
    if x & 0x8000 != 0 {
        !x
    } else {
        x | 0x8000
    }
}
struct Buffers {
    device: Arc<CudaDevice>,
    packed: Option<CudaSlice<u8>>,
    scales: Option<CudaSlice<f32>>,
    zeros: Option<CudaSlice<f32>>,
    input: Option<CudaSlice<u16>>,
    bf: Option<CudaSlice<u16>>,
    fp: Option<CudaSlice<f32>>,
}
impl Drop for Buffers {
    fn drop(&mut self) {
        if let Err(e) = self
            .device
            .bind_to_thread()
            .and_then(|_| self.device.synchronize())
        {
            eprintln!(
                "FATAL: actual dual-output owner completion unknown; retaining allocations: {e}"
            );
            std::process::abort();
        }
    }
}
fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev mla-projection-probe".into());
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected bound plan and fresh output directory".into());
    }
    let raw = fs::read(&args[1]).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if p.schema != 1
        || p.row_sources.is_empty()
        || p.row_sources.iter().any(|&x| x > 1)
        || p.safety_margin_bytes != 600 * 1024 * 1024
    {
        return Err("explicit dual-output schema/actual-row/default-margin admission".into());
    }
    let c: Capture = serde_json::from_slice(&p.capture_plan.read()?).map_err(|e| e.to_string())?;
    if c.schema != 1
        || c.layout != "hqq6_row_major_f32_parameters"
        || c.rows == 0
        || c.cols == 0
        || c.rows > i32::MAX as usize
        || c.cols > i32::MAX as usize
        || c.group_size == 0
        || c.cols % 4 != 0
        || c.group_size % 4 != 0
        || c.cols % c.group_size != 0
        || c.packed_stride != mul(c.cols / 4, 3)?
        || c.scales_stride % 4 != 0
        || c.scales_stride < mul(c.cols / c.group_size, 4)?
        || c.zeros_stride != c.scales_stride
    {
        return Err("actual HQQ6 capture schema/geometry".into());
    }
    let reference_audit: Value =
        serde_json::from_slice(&p.reference_audit.read()?).map_err(|e| e.to_string())?;
    if reference_audit["passed"] != true {
        return Err("independent F64 audit not passed".into());
    }
    let summary = reference_audit["summaries"]
        .as_array()
        .and_then(|v| v.iter().find(|v| v["name"] == p.case_name))
        .ok_or("actual case absent from independent reference audit")?;
    if summary["details"]["plan_sha256"] != p.capture_plan.sha256 {
        return Err("reference/capture provenance mismatch".into());
    }
    let references: Vec<Identity> =
        serde_json::from_value(summary["references"].clone()).map_err(|e| e.to_string())?;
    let find_ref = |name: &str| -> Result<Vec<f64>, String> {
        let r = references
            .iter()
            .find(|x| x.path.file_name().and_then(|x| x.to_str()) == Some(name))
            .ok_or("typed reference absent")?;
        let v = f64s(&r.read()?)?;
        if v.len() != c.rows {
            return Err("typed F64 reference rows mismatch".into());
        }
        Ok(v)
    };
    let refs = [
        find_ref("prefill-quant-f32-reference.f64")?,
        find_ref("decode-quant-f32-reference.f64")?,
    ];
    let shader = p.shader_source.read()?;
    let ptx = p.ptx.read()?;
    let bank: Value = serde_json::from_slice(&p.bank_audit.read()?).map_err(|e| e.to_string())?;
    if bank["passed"] != true
        || !bank["rows"]
            .as_array()
            .ok_or("bank rows absent")?
            .iter()
            .any(|v| v["sha256"] == p.ptx.sha256)
    {
        return Err("PTX not bound to installed57 bank".into());
    }
    let source_freeze: Value =
        serde_json::from_slice(&p.source_freeze.read()?).map_err(|e| e.to_string())?;
    if !source_freeze["files"]
        .as_array()
        .ok_or("source freeze files absent")?
        .iter()
        .any(|v| v["sha256"] == p.shader_source.sha256)
    {
        return Err("shader source not frozen with installed57".into());
    }
    let contract = p.build_contract.read()?;
    if !contract.ends_with(&shader)
        || !String::from_utf8_lossy(&contract).contains("--use_fast_math")
    {
        return Err("actual selected shader compilation contract differs".into());
    }
    p.bank_rust_source.read()?;
    let bridge: Value =
        serde_json::from_slice(&p.source_bridge.read()?).map_err(|e| e.to_string())?;
    if bridge["passed"] != true || bridge["currentSourceSha256"] != p.shader_source.sha256 {
        return Err("retained-to-current shader source delta unbound".into());
    }
    let operand_audit: Value =
        serde_json::from_slice(&p.operand_audit.read()?).map_err(|e| e.to_string())?;
    let inventory: Value =
        serde_json::from_slice(&p.operand_inventory.read()?).map_err(|e| e.to_string())?;
    if operand_audit["collectionPassed"] != true
        || !matches!(p.weight_layout.as_str(), "prefill" | "decode")
    {
        return Err("actual MLA operand/layout admission".into());
    }
    let weight_proof = operand_audit["weights"]
        .as_array()
        .and_then(|v| v.iter().find(|x| x["name"] == p.case_name))
        .ok_or("actual weight proof absent")?;
    if weight_proof["logicalHqqWeightsExact"] != true || weight_proof["materializedBf16"] != true {
        return Err("actual source coefficient/materialization proof missing".into());
    }
    let key = if p.weight_layout == "prefill" {
        "full"
    } else {
        "decode"
    };
    let root = PathBuf::from(
        operand_audit["paths"][key]
            .as_str()
            .ok_or("actual operand root absent")?,
    );
    let capture_root = PathBuf::from(
        inventory["captureRoot"]
            .as_str()
            .ok_or("inventory root absent")?,
    );
    let read_actual = |file: &str| -> Result<Vec<u8>, String> {
        let path = root.join(file);
        let relative = path
            .strip_prefix(&capture_root)
            .map_err(|_| "actual operand outside inventory")?
            .to_str()
            .ok_or("operand filename encoding")?;
        let row = inventory["rows"]
            .as_array()
            .and_then(|v| v.iter().find(|v| v["relative"] == relative))
            .ok_or("actual captured file not inventoried")?;
        let id = Identity {
            path,
            bytes: usize::try_from(row["bytes"].as_u64().ok_or("inventory bytes")?)
                .map_err(|e| e.to_string())?,
            sha256: row["sha256"].as_str().ok_or("inventory hash")?.into(),
        };
        id.read()
    };
    let actual_meta: Value =
        serde_json::from_slice(&read_actual(&format!("{}-weights.json", p.case_name))?)
            .map_err(|e| e.to_string())?;
    for (key, expected) in [
        ("rows", c.rows),
        ("cols", c.cols),
        ("group_size", c.group_size),
    ] {
        if actual_meta[key].as_u64() != Some(expected as u64) {
            return Err(format!("actual layout geometry differs: {key}"));
        }
    }
    let stride = |key: &str| -> Result<usize, String> {
        usize::try_from(actual_meta[key].as_u64().ok_or("actual stride missing")?)
            .map_err(|e| e.to_string())
    };
    let ps = stride("packed_stride")?;
    let ss = stride("scales_stride")?;
    let zs = stride("zeros_stride")?;
    if ps != mul(c.cols / 4, 3)? || ss % 4 != 0 || ss < mul(c.cols / c.group_size, 4)? || zs != ss {
        return Err("actual loaded HQQ6 stride contract invalid".into());
    }
    let packed = read_actual(&format!("{}-weights.packed", p.case_name))?;
    let scales = f32s(&read_actual(&format!(
        "{}-weights.scales.f32",
        p.case_name
    ))?)?;
    let zeros = f32s(&read_actual(&format!("{}-weights.zeros.f32", p.case_name))?)?;
    let input_rows = [
        u16s(&c.prefill_input.read()?)?,
        u16s(&c.decode_input.read()?)?,
    ];
    if input_rows.iter().any(|x| x.len() != c.cols)
        || packed.len() != mul(c.rows, ps)?
        || scales.len() * 4 != mul(c.rows, ss)?
        || zeros.len() * 4 != mul(c.rows, zs)?
    {
        return Err("actual input/weight allocation extent mismatch".into());
    }
    let tokens = p.row_sources.len();
    if p.partitions.is_empty()
        || p.partitions.iter().any(|v| {
            v.is_empty()
                || v.contains(&0)
                || v.iter().try_fold(0usize, |s, &n| s.checked_add(n)) != Some(tokens)
        })
    {
        return Err("incomplete/nonpositive partition coverage".into());
    }
    let input_extent = mul(tokens, c.cols)?;
    let input: Vec<u16> = p
        .row_sources
        .iter()
        .flat_map(|&s| input_rows[s].iter().copied())
        .collect();
    if input.len() != input_extent {
        return Err("constructed actual-row extent differs".into());
    }
    // Conventional FP32 dot-product forward-error target, computed from actual K.
    let nops = mul(c.cols, 2)? as f64;
    let nu = nops * (f32::EPSILON as f64 / 2.0);
    if nu >= 1.0 {
        return Err("FP32 forward-error target undefined for actual K".into());
    }
    let gamma = nu / (1.0 - nu);
    let mut abs_sums = [vec![0.0f64; c.rows], vec![0.0f64; c.rows]];
    for r in 0..c.rows {
        for col in 0..c.cols {
            let off = r * ps + (col / 4) * 3;
            let bits = (packed[off] as u32)
                | ((packed[off + 1] as u32) << 8)
                | ((packed[off + 2] as u32) << 16);
            let q = (bits >> ((col % 4) * 6)) & 63;
            let g = col / c.group_size;
            let weight = (q as f32 - zeros[r * (zs / 4) + g]) * scales[r * (ss / 4) + g];
            if !weight.is_finite() {
                return Err("nonfinite reconstructed weight".into());
            }
            for s in 0..2 {
                let x = f32::from_bits((input_rows[s][col] as u32) << 16);
                abs_sums[s][r] += (weight as f64 * x as f64).abs();
            }
        }
    }
    let dev = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    if dev.cu_stream().is_null() {
        return Err("explicit nondefault stream required".into());
    }
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
        return Err("actual CUDA UUID differs from admitted card".into());
    }
    let bank = ordered_projection::OrderedProjection::load(
        dev.clone(),
        std::str::from_utf8(&ptx).map_err(|e| e.to_string())?,
    )?;
    let count = mul(tokens, c.rows)?;
    let guard = 16usize;
    let total = count.checked_add(guard).ok_or("guard extent overflow")?;
    let bi = vec![0x55aau16; total];
    let fi = vec![f32::from_bits(0x7fc12345); total];
    let required = [
        packed.len(),
        scales.len() * 4,
        zeros.len() * 4,
        input.len() * 2,
        mul(total, 6)?,
    ]
    .into_iter()
    .try_fold(0usize, |a, b| {
        a.checked_add(b).ok_or("required bytes overflow")
    })?;
    let before = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    if before
        < required
            .checked_add(p.safety_margin_bytes)
            .ok_or("reserve overflow")?
    {
        return Err("measured free VRAM cannot cover actual allocations and margin".into());
    }
    let mut owners = Buffers {
        device: dev.clone(),
        packed: None,
        scales: None,
        zeros: None,
        input: None,
        bf: None,
        fp: None,
    };
    owners.packed = Some(dev.htod_copy(packed.clone()).map_err(|e| e.to_string())?);
    owners.scales = Some(dev.htod_copy(scales.clone()).map_err(|e| e.to_string())?);
    owners.zeros = Some(dev.htod_copy(zeros.clone()).map_err(|e| e.to_string())?);
    owners.input = Some(dev.htod_copy(input.clone()).map_err(|e| e.to_string())?);
    owners.bf = Some(dev.htod_copy(bi.clone()).map_err(|e| e.to_string())?);
    owners.fp = Some(dev.htod_copy(fi.clone()).map_err(|e| e.to_string())?);
    let packed_gpu = owners.packed.as_ref().unwrap();
    let scales_gpu = owners.scales.as_ref().unwrap();
    let zeros_gpu = owners.zeros.as_ref().unwrap();
    let input_gpu = owners.input.as_ref().unwrap();
    let bf_gpu = owners.bf.as_ref().unwrap();
    let fp_gpu = owners.fp.as_ref().unwrap();
    let free = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    if free < p.safety_margin_bytes {
        return Err("actual postallocation reserve violated".into());
    }
    let w = ordered_projection::Layout {
        packed: *packed_gpu.device_ptr(),
        scales: *scales_gpu.device_ptr(),
        zeros: *zeros_gpu.device_ptr(),
        rows: c.rows,
        cols: c.cols,
        group: c.group_size,
        packed_stride: ps,
        scales_stride: ss,
        zeros_stride: zs,
    };
    w.validate()?;
    let enqueue = |at: usize, n: usize| -> Result<(), String> {
        if n == 0 || at.checked_add(n).map_or(true, |end| end > tokens) {
            return Err("current prefix outside allocated input/output".into());
        }
        unsafe {
            bank.enqueue_f32(
                &w,
                *input_gpu.device_ptr() + mul(mul(at, c.cols)?, 2)? as u64,
                *bf_gpu.device_ptr() + mul(mul(at, c.rows)?, 2)? as u64,
                *fp_gpu.device_ptr() + mul(mul(at, c.rows)?, 4)? as u64,
                n,
                *dev.cu_stream(),
            )
        }
    };
    let reset = || -> Result<(), String> {
        dev.synchronize().map_err(|e| e.to_string())?;
        for (ptr, src, size) in [
            (
                *bf_gpu.device_ptr(),
                bi.as_ptr().cast::<std::ffi::c_void>(),
                bi.len() * 2,
            ),
            (
                *fp_gpu.device_ptr(),
                fi.as_ptr().cast::<std::ffi::c_void>(),
                fi.len() * 4,
            ),
        ] {
            unsafe { sys::lib().cuMemcpyHtoD_v2(ptr, src, size) }
                .result()
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let read = || -> Result<(Vec<u16>, Vec<u32>), String> {
        let b = dev.dtoh_sync_copy(bf_gpu).map_err(|e| e.to_string())?;
        let f = dev.dtoh_sync_copy(fp_gpu).map_err(|e| e.to_string())?;
        Ok((b, f.iter().map(|x| x.to_bits()).collect()))
    };
    let image = (
        bi.clone(),
        fi.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
    );
    enqueue(0, tokens)?;
    let full = read()?;
    if full.0[count..] != image.0[count..] || full.1[count..] != image.1[count..] {
        return Err("whole output suffix guard modified".into());
    }
    let mut max_bf_ulp = 0u16;
    let mut sse = 0.0f64;
    let mut norm = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut forward_violations = 0usize;
    let mut f32_different = 0usize;
    for t in 0..tokens {
        let source = p.row_sources[t];
        for r in 0..c.rows {
            let i = t * c.rows + r;
            let v = f32::from_bits(full.1[i]);
            let reference = refs[source][r];
            if !v.is_finite() || !(reference as f32).is_finite() {
                return Err("nonfinite or overflowing F32 output/reference".into());
            }
            let error = (v as f64 - reference).abs();
            max_abs = max_abs.max(error);
            sse += error * error;
            norm += reference * reference;
            forward_violations += usize::from(error > gamma * abs_sums[source][r]);
            f32_different += usize::from(v.to_bits() != (reference as f32).to_bits());
            let bf = full.0[i];
            if bf != rne(v as f64) {
                return Err("BF16 output not exact RNE of same actual F32 result".into());
            }
            max_bf_ulp = max_bf_ulp.max(ordered_bf(bf).abs_diff(ordered_bf(rne(reference))));
        }
    }
    let out = PathBuf::from(&args[2]);
    fs::create_dir(&out).map_err(|e| e.to_string())?;
    fs::write(
        out.join("full.bf16"),
        full.0[..count]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        out.join("full.f32"),
        full.1[..count]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    reset()?;
    unsafe {
        bank.enqueue(
            &w,
            *input_gpu.device_ptr(),
            *bf_gpu.device_ptr(),
            tokens,
            *dev.cu_stream(),
        )
    }?;
    let bf_only = read()?;
    let original_bf16_preserved = bf_only.0 == full.0 && bf_only.1 == image.1;
    let mut controls = Vec::new();
    for parts in &p.partitions {
        reset()?;
        let mut at = 0;
        let mut prefix_exact = true;
        for &n in parts {
            enqueue(at, n)?;
            at += n;
            let current = read()?;
            let used = at * c.rows;
            prefix_exact &= current.0[..used] == full.0[..used]
                && current.1[..used] == full.1[..used]
                && current.0[used..] == image.0[used..]
                && current.1[used..] == image.1[used..];
        }
        let exact = read()? == full;
        reset()?;
        graph_owner::completed_graph_launch(&dev, || {
            let mut at = 0;
            for &n in parts {
                enqueue(at, n)?;
                at += n;
            }
            Ok(())
        })?;
        controls.push(json!({"partition":parts,"everyExecutedPrefixAndUntouchedSuffixExact":prefix_exact,"completeExact":exact,"graphExact":read()?==full}));
    }
    reset()?;
    for t in 0..tokens {
        enqueue(t, 1)?;
    }
    let singleton = read()? == full;
    reset()?;
    graph_owner::completed_graph_launch(&dev, || enqueue(0, tokens))?;
    let graph = read()? == full;
    reset()?;
    let rejected = graph_owner::completed_graph_launch(&dev, || {
        enqueue(0, tokens)?;
        Err("intentional post-enqueue rejection before replay".into())
    });
    let not_replayed = rejected.is_err() && read()? == image;
    let invalid = unsafe {
        bank.enqueue_f32(
            &w,
            *input_gpu.device_ptr(),
            *bf_gpu.device_ptr(),
            *bf_gpu.device_ptr(),
            tokens,
            *dev.cu_stream(),
        )
    };
    let overlap_rejected = invalid.is_err() && read()? == image;
    enqueue(0, tokens)?;
    let recovery = read()? == full;
    let readonly = dev.dtoh_sync_copy(packed_gpu).map_err(|e| e.to_string())? == packed
        && dev
            .dtoh_sync_copy(scales_gpu)
            .map_err(|e| e.to_string())?
            .iter()
            .map(|x| x.to_bits())
            .eq(scales.iter().map(|x| x.to_bits()))
        && dev
            .dtoh_sync_copy(zeros_gpu)
            .map_err(|e| e.to_string())?
            .iter()
            .map(|x| x.to_bits())
            .eq(zeros.iter().map(|x| x.to_bits()))
        && dev.dtoh_sync_copy(input_gpu).map_err(|e| e.to_string())? == input;
    let partition = controls.iter().all(|v| {
        v["everyExecutedPrefixAndUntouchedSuffixExact"] == true
            && v["completeExact"] == true
            && v["graphExact"] == true
    });
    let passed = original_bf16_preserved
        && partition
        && singleton
        && graph
        && not_replayed
        && overlap_rejected
        && recovery
        && readonly
        && max_bf_ulp <= 1
        && forward_violations == 0;
    let value = json!({"passed":passed,"collectionPassed":true,"case":p.case_name,"weightLayout":p.weight_layout,"actualWeightStrides":{"packed":ps,"scales":ss,"zeros":zs},"tokens":tokens,"rows":c.rows,"cols":c.cols,"gpuUuid":actual,"gpuName":dev.name().map_err(|e|e.to_string())?,"partitionControlsPassed":partition,"originalBf16OnlyAdapterPreserved":original_bf16_preserved,"controls":controls,"singletonsExact":singleton,"graphExact":graph,"rejectedGraphNotReplayed":not_replayed,"invalidOverlapRejectedWithoutMutation":overlap_rejected,"exactRecovery":recovery,"readOnlyOperandsExact":readonly,"suffixGuardsExact":true,"maxBf16Ulp":max_bf_ulp,"f32DifferentFromF64Rne":f32_different,"f32MaxAbsToF64":max_abs,"f32RelativeL2":if norm>0.0{Some((sse/norm).sqrt())}else{None},"f32ForwardTargetViolations":forward_violations,"f32ForwardTargetGamma":gamma,"numericalTarget":"F32 result satisfies conventional gamma(2K)*sum(abs(w*x)) dot-product target; RNE result <=1BF16ULP from independentF64. Target derives from actual K and IEEEFP32 epsilon, not one model/GPU. This is a preregistered accuracy target, not a formal algorithm error theorem.","requiredGpuBytes":required,"freeBeforeAllocationBytes":before,"freeAfterAllocationBytes":free,"safetyMarginBytes":p.safety_margin_bytes,"planSha256":hash(&raw),"binary":identity(&std::env::current_exe().map_err(|e|e.to_string())?)?,"bankRustSourceSha256":p.bank_rust_source.sha256,"ptxSha256":p.ptx.sha256,"sourceSha256":p.shader_source.sha256,"scope":"Actual MLA tensor rows repeated/alternated explicitly for batch boundary controls; real serving dual-output adapter and exact existing kernel. Not serving correctness, model/Witness fidelity, speed or ADQ."});
    fs::write(
        out.join("REPORT.json"),
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{value}");
    if !passed {
        return Err("dual-output numerical/partition controls failed".into());
    }
    Ok(())
}
fn dispatch() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev mla-projection-probe".into());
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 4 && args[1] == "sanitize" {
        let p: Plan = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        p.sanitizer.read()?;
        p.sanitizer_binary.read()?;
        let version = std::process::Command::new(&p.sanitizer.path)
            .arg("--version")
            .output()
            .map_err(|e| e.to_string())?;
        if !version.status.success() {
            return Err("sanitizer version check failed".into());
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
            .arg(&args[2])
            .arg(&args[3])
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("sanitized dual-output control failed: {status}"));
        }
        return Ok(());
    }
    run()
}
fn main() {
    if let Err(e) = dispatch() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn layout() -> ordered_projection::Layout {
        ordered_projection::Layout {
            packed: 0x1000,
            scales: 0x2000,
            zeros: 0x3000,
            rows: 3,
            cols: 4,
            group: 4,
            packed_stride: 3,
            scales_stride: 4,
            zeros_stride: 4,
        }
    }
    #[test]
    fn output_owners_must_be_distinct_and_aligned() {
        let w = layout();
        assert!(
            ordered_projection::validate_projection_outputs(&w, 0x4000, 0x5000, 0x6000, 2).is_ok()
        );
        for fp in [0x5000, 0x4000, 0x2000, 0x6002] {
            assert!(
                ordered_projection::validate_projection_outputs(&w, 0x4000, 0x5000, fp, 2).is_err()
            );
        }
        assert!(ordered_projection::validate_projection_outputs(
            &w,
            0x4000,
            0x5000,
            0x6000,
            usize::MAX
        )
        .is_err());
    }
    #[test]
    fn original_bf16_only_contract_and_zero_size() {
        let w = layout();
        assert!(ordered_projection::validate_projection_outputs(&w, 0x4000, 0x5000, 0, 2).is_ok());
        assert!(ordered_projection::validate_projection_outputs(&w, 0x4000, 0x5000, 0, 0).is_err());
        assert!(ordered_projection::validate_projection_outputs(&w, 0x4000, 0x1000, 0, 2).is_err());
    }
}
