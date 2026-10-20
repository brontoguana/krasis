// Unadopted fixed tensor batch diagnostic; explicit new arithmetic contract.
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Identity {
    fn read(&self) -> Result<Vec<u8>, String> {
        let z = fs::read(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        if z.len() != self.bytes || format!("{:x}", Sha256::digest(&z)) != self.sha256 {
            return Err(format!("identity mismatch: {}", self.path.display()));
        }
        Ok(z)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildPlan {
    source: Identity,
    nvcc: Identity,
    flags: Vec<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    height: usize,
    partitions: Vec<usize>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbePlan {
    schema: u32,
    source: Identity,
    ptx: Identity,
    build_report: Identity,
    dispatch: Identity,
    capture: BTreeMap<String, Identity>,
    gpu_ordinal: usize,
    gpu_name: String,
    gpu_uuid: String,
    compute_capability: u32,
    measured_decode_warps: usize,
    token_tile: usize,
    kernel_contract: String,
    safety_margin_bytes: usize,
    cases: Vec<Case>,
}
struct Completion<'a>(&'a std::sync::Arc<CudaDevice>);
impl Drop for Completion<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("ERROR: diagnostic teardown completion failed: {e}");
        }
    }
}
fn get(n: &Value, k: &str) -> Result<usize, String> {
    n[k].as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or(format!("invalid {k}"))
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("size overflow".into())
}
fn add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b).ok_or("size overflow".into())
}
fn dims(d: &Value) -> Result<(usize, usize, usize, usize, usize, usize, usize, usize), String> {
    let r = get(d, "rows")?;
    let k = get(d, "cols")?;
    let g = get(d, "group_size")?;
    let ps = get(d, "packed_row_stride_bytes")?;
    let ss = get(d, "scales_row_stride_bytes")?;
    let zs = get(d, "zeros_row_stride_bytes")?;
    let h = get(d, "height")?;
    let row = d["row"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or("invalid selected row")?;
    if d["nbits"] != 6
        || d["axis"] != 1
        || d["layout"] != "row_major_axis1_grouped_uint6_packed"
        || d["packed_dtype"] != "uint8"
        || d["scales_dtype"] != "float32"
        || d["zeros_dtype"] != "float32"
        || row >= h
        || k % 4 != 0
        || g % 4 != 0
        || k % g != 0
        || ps != mul(k / 4, 3)?
        || ss != mul(k / g, 4)?
        || zs != ss
        || [r, k, g, ps, ss, zs].iter().any(|&v| v > i32::MAX as usize)
    {
        return Err("unsupported actual packed HQQ6 geometry/dtype/alignment".into());
    }
    Ok((r, k, g, ps, ss, zs, h, row))
}
fn grid(
    r: usize,
    h: usize,
    warps: usize,
    limits: [usize; 5],
    shared: usize,
) -> Result<LaunchConfig, String> {
    let [max_threads, max_shared, gx, gy, gz] = limits;
    let threads = mul(warps, 32)?;
    if warps == 0 || threads > max_threads || shared > max_shared || gy == 0 || gz == 0 {
        return Err("actual launch limits exceeded".into());
    }
    let x = r.div_ceil(warps);
    let y = h.min(gy);
    if y == 0 || x == 0 || x > gx || mul(x, warps)? > i32::MAX as usize {
        return Err("grid x/y admission failed".into());
    }
    let z = h.div_ceil(y);
    if z > gz {
        return Err("grid z admission failed".into());
    }
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(x).map_err(|_| "grid overflow")?,
            u32::try_from(y).map_err(|_| "grid overflow")?,
            u32::try_from(z).map_err(|_| "grid overflow")?,
        ),
        block_dim: (
            u32::try_from(threads).map_err(|_| "threads overflow")?,
            1,
            1,
        ),
        shared_mem_bytes: u32::try_from(shared).map_err(|_| "shared overflow")?,
    })
}
fn tensor_grid(r: usize, h: usize, limits: [usize; 5]) -> Result<LaunchConfig, String> {
    let shared_floats = mul(add(mul(3, 64)?, 32)?, 40)?;
    let shared_bytes = mul(shared_floats, 4)?;
    let [threads, shared, gx, gy, gz] = limits;
    if threads < 8 * 32 || shared < shared_bytes {
        return Err("runtime thread/shared limits reject tensor layout".into());
    }
    let token_blocks = h.div_ceil(32);
    let x = token_blocks.min(gx);
    let y = r.div_ceil(64);
    if x == 0 || y == 0 || y > gy {
        return Err("runtime tensor grid x/y limits exceeded".into());
    }
    let z = token_blocks.div_ceil(x);
    if z > gz {
        return Err("runtime tensor grid z limit exceeded".into());
    }
    Ok(LaunchConfig {
        grid_dim: (x as u32, y as u32, z as u32),
        block_dim: (8 * 32, 1, 1),
        shared_mem_bytes: 0,
    })
}
fn save_f32(out: &Path, name: &str, z: &[f32]) -> Result<Identity, String> {
    let b: Vec<u8> = z.iter().flat_map(|x| x.to_le_bytes()).collect();
    let path = out.join(name);
    fs::write(&path, &b).map_err(|e| e.to_string())?;
    Ok(Identity {
        path,
        bytes: b.len(),
        sha256: format!("{:x}", Sha256::digest(&b)),
    })
}
fn validate_cases(c: &[Case], height: usize) -> Result<(), String> {
    let mut names = std::collections::BTreeSet::new();
    if c.is_empty() {
        return Err("missing preregistered cases".into());
    }
    for x in c {
        if x.name.is_empty()
            || !x
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            || !names.insert(&x.name)
            || x.height == 0
            || x.height > height
            || x.partitions.is_empty()
            || x.partitions.contains(&0)
            || x.partitions
                .iter()
                .try_fold(0usize, |a, &b| a.checked_add(b))
                != Some(x.height)
        {
            return Err("invalid complete partition coverage".into());
        }
    }
    Ok(())
}
fn save(out: &Path, name: &str, z: &[u16]) -> Result<Identity, String> {
    let path = out.join(name);
    let b = z.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    fs::write(&path, &b).map_err(|e| e.to_string())?;
    Ok(Identity {
        path,
        bytes: b.len(),
        sha256: format!("{:x}", Sha256::digest(&b)),
    })
}
fn u16s(z: &[u8]) -> Result<Vec<u16>, String> {
    if z.len() % 2 != 0 {
        return Err("odd BF16 bytes".into());
    }
    let v = z
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    if v.iter().any(|&v| !bf16::from_bits(v).is_finite()) {
        return Err("nonfinite BF16".into());
    }
    Ok(v)
}
fn f32s(z: &[u8]) -> Result<Vec<f32>, String> {
    if z.len() % 4 != 0 {
        return Err("unaligned F32 bytes".into());
    }
    let v = z
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    if v.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite F32".into());
    }
    Ok(v)
}
fn build(plan: &Path, out: &Path) -> Result<Value, String> {
    let p: BuildPlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    p.source.read()?;
    p.nvcc.read()?;
    if p.flags
        != [
            "-ptx",
            "-allow-unsupported-compiler",
            "-arch=sm_80",
            "-O3",
            "--use_fast_math",
        ]
    {
        return Err("compiler flags differ from admitted production decode flags".into());
    }
    let z = Command::new(&p.nvcc.path)
        .args(&p.flags)
        .arg("-o")
        .arg(out.join("operator.ptx"))
        .arg(&p.source.path)
        .output()
        .map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stdout"), z.stdout).map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stderr"), z.stderr).map_err(|e| e.to_string())?;
    if !z.status.success() {
        return Err(format!("nvcc failed {}", z.status));
    }
    p.source.read()?;
    p.nvcc.read()?;
    let b = fs::read(out.join("operator.ptx")).map_err(|e| e.to_string())?;
    let s = std::str::from_utf8(&b).map_err(|e| e.to_string())?;
    for n in [
        "hqq6_decode_gemv_bf16_vec4",
        "hqq6_batch_gemv_bf16_vec4",
        "hqq6_batch_tile2",
        "hqq6_batch_tile4",
        "hqq6_batch_compensated_tf32x3",
    ] {
        if !s.contains(n) {
            return Err("missing compiled kernel".into());
        }
    }
    Ok(
        json!({"plan":p,"ptx":Identity{path:out.join("operator.ptx"),bytes:b.len(),sha256:format!("{:x}",Sha256::digest(&b))},"diagnostic_only":true,"adopted":false}),
    )
}
fn probe(plan: &Path, out: &Path) -> Result<Value, String> {
    let p: ProbePlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if p.schema != 1 || p.safety_margin_bytes != 600 * 1024 * 1024 {
        return Err("explicit schema and user 600 MiB margin required".into());
    }
    if p.token_tile != 32 || p.kernel_contract != "tf32_three_component_fixed_k8_twosum_v1" {
        return Err("explicit tensor arithmetic admission required".into());
    }
    p.source.read()?;
    let built: Value =
        serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
    if built["plan"]["source"] != serde_json::to_value(&p.source).map_err(|e| e.to_string())?
        || built["ptx"] != serde_json::to_value(&p.ptx).map_err(|e| e.to_string())?
    {
        return Err("PTX/source/build provenance mismatch".into());
    }
    let dispatch: Value = serde_json::from_slice(&p.dispatch.read()?).map_err(|e| e.to_string())?;
    if dispatch["hqq6_selected_vec4_warps"].as_u64() != Some(p.measured_decode_warps as u64)
        || dispatch["runtime_stage"] != "decode"
        || dispatch["kernel_ready"] != true
    {
        return Err("measured dispatch does not match explicit warp admission".into());
    }
    let ptx = p.ptx.read()?;
    let read = |k: &str| {
        p.capture
            .get(k)
            .ok_or(format!("missing identity {k}"))?
            .read()
    };
    let desc: Value = serde_json::from_slice(&read("descriptor")?).map_err(|e| e.to_string())?;
    let (r, k, g, ps, ss, zs, h, row) = dims(&desc)?;
    validate_cases(&p.cases, h)?;
    for (key, value) in [
        ("rows", r),
        ("cols", k),
        ("group_size", g),
        ("packed_row_stride_bytes", ps),
        ("scales_row_stride_bytes", ss),
        ("zeros_row_stride_bytes", zs),
    ] {
        if dispatch[key].as_u64() != Some(value as u64) {
            return Err(format!("dispatch geometry differs: {key}"));
        }
    }
    let packed = read("packed")?;
    let sb = read("scales")?;
    let zb = read("zeros")?;
    let xb = read("full_input")?;
    let selected = read("selected_input")?;
    let nb = read("native_output")?;
    for (actual, expected) in [
        (packed.len(), mul(r, ps)?),
        (sb.len(), mul(r, ss)?),
        (zb.len(), mul(r, zs)?),
        (xb.len(), mul(mul(h, k)?, 2)?),
        (selected.len(), mul(k, 2)?),
        (nb.len(), mul(r, 2)?),
    ] {
        if actual != expected {
            return Err("actual captured allocation extent mismatch".into());
        }
    }
    let start = mul(mul(row, k)?, 2)?;
    if xb[start..start + mul(k, 2)?] != selected {
        return Err("selected row differs from authentic controlled full input".into());
    }
    let scales = f32s(&sb)?;
    let zeros = f32s(&zb)?;
    let x = u16s(&xb)?;
    let native = u16s(&nb)?;
    let dev = CudaDevice::new(p.gpu_ordinal).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<cudarc::driver::sys::CUuuid>() };
    unsafe {
        cudarc::driver::sys::lib()
            .cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device())
            .result()
            .map_err(|e| e.to_string())?;
    }
    let hex = uuid
        .bytes
        .iter()
        .map(|&b| format!("{:02x}", b as u8))
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
        return Err("actual GPU UUID differs from frozen admission".into());
    }
    use cudarc::driver::sys::CUdevice_attribute as A;
    let attr = |a| dev.attribute(a).map_err(|e| e.to_string());
    let cc = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
        + attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    if cc < 80
        || cc as u32 != p.compute_capability
        || dev.name().map_err(|e| e.to_string())? != p.gpu_name
    {
        return Err("actual device does not match explicit admission".into());
    }
    let limits = [
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
        attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
    ]
    .map(|v| v as usize);
    let config = tensor_grid(r, h, limits)?;
    let before = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    dev.load_ptx(
        cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx).map_err(|e| e.to_string())?),
        "row_operator",
        &[
            "hqq6_decode_gemv_bf16_vec4",
            "hqq6_batch_gemv_bf16_vec4",
            "hqq6_batch_tile2",
            "hqq6_batch_tile4",
            "hqq6_batch_compensated_tf32x3",
        ],
    )
    .map_err(|e| e.to_string())?;
    let orig = dev
        .get_func("row_operator", "hqq6_decode_gemv_bf16_vec4")
        .ok_or("original kernel missing")?;
    let batch = dev
        .get_func("row_operator", "hqq6_batch_compensated_tf32x3")
        .ok_or("batch kernel missing")?;
    let after_module = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    let tensor_bytes = [
        packed.len(),
        sb.len(),
        zb.len(),
        xb.len(),
        mul(mul(h, r)?, 6)?,
    ]
    .iter()
    .try_fold(0usize, |a, &b| add(a, b))?;
    if after_module.0 < add(tensor_bytes, p.safety_margin_bytes)? {
        return Err("measured VRAM cannot cover actual tensors plus 600 MiB".into());
    }
    let dp = dev.htod_copy(packed.clone()).map_err(|e| e.to_string())?;
    let ds = dev.htod_copy(scales.clone()).map_err(|e| e.to_string())?;
    let dz = dev.htod_copy(zeros.clone()).map_err(|e| e.to_string())?;
    let dx = dev.htod_copy(x.clone()).map_err(|e| e.to_string())?;
    let mut dy = dev
        .alloc_zeros::<u16>(mul(h, r)?)
        .map_err(|e| e.to_string())?;
    let mut df = dev
        .alloc_zeros::<f32>(mul(h, r)?)
        .map_err(|e| e.to_string())?;
    let _completion = Completion(&dev);
    let floor = cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0;
    if floor < p.safety_margin_bytes {
        return Err("allocated operator tensors breach reserve".into());
    }
    let input_ptr = *dx.device_ptr();
    let output_ptr = *dy.device_ptr();
    let float_ptr = *df.device_ptr();
    let row_in = input_ptr
        .checked_add(mul(mul(row, k)?, 2)? as u64)
        .ok_or("pointer overflow")?;
    let original_config = grid(r, 1, p.measured_decode_warps, limits, mul(k, 2)?)?;
    unsafe {
        orig.launch(
            original_config,
            (
                *dp.device_ptr(),
                *ds.device_ptr(),
                *dz.device_ptr(),
                row_in,
                output_ptr,
                r as i32,
                k as i32,
                g as i32,
                ps as i32,
                ss as i32,
                zs as i32,
            ),
        )
        .map_err(|e| e.to_string())?
    }
    dev.synchronize().map_err(|e| e.to_string())?;
    let original = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
    save(out, "original-selected.bf16", &original[..r])?;
    if original[..r] != native {
        return Err("original single row differs from retained actual native decode".into());
    }
    let launch = |start: usize, count: usize| -> Result<(), String> {
        let cf = tensor_grid(r, count, limits)?;
        let xi = input_ptr
            .checked_add(mul(mul(start, k)?, 2)? as u64)
            .ok_or("pointer overflow")?;
        let yi = output_ptr
            .checked_add(mul(mul(start, r)?, 2)? as u64)
            .ok_or("pointer overflow")?;
        let fi = float_ptr
            .checked_add(mul(mul(start, r)?, 4)? as u64)
            .ok_or("F32 pointer overflow")?;
        unsafe {
            batch
                .clone()
                .launch(
                    cf,
                    (
                        *dp.device_ptr(),
                        *ds.device_ptr(),
                        *dz.device_ptr(),
                        xi,
                        yi,
                        fi,
                        r as i32,
                        k as i32,
                        g as i32,
                        ps as i32,
                        ss as i32,
                        count as u64,
                    ),
                )
                .map_err(|e| e.to_string())?
        }
        Ok(())
    };
    unsafe {
        batch
            .clone()
            .launch(
                tensor_grid(r, 1, limits)?,
                (
                    *dp.device_ptr(),
                    *ds.device_ptr(),
                    *dz.device_ptr(),
                    row_in,
                    output_ptr,
                    float_ptr,
                    r as i32,
                    k as i32,
                    g as i32,
                    ps as i32,
                    ss as i32,
                    1u64,
                ),
            )
            .map_err(|e| e.to_string())?;
    }
    dev.synchronize().map_err(|e| e.to_string())?;
    let one = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
    save(out, "batch-selected-single.bf16", &one[..r])?;
    let one_float = dev.dtoh_sync_copy(&df).map_err(|e| e.to_string())?;
    save_f32(out, "batch-selected-single.f32", &one_float[..r])?;
    let selected_diff = one[..r]
        .iter()
        .zip(native.iter())
        .filter(|(a, b)| a != b)
        .count();
    let full_start = Instant::now();
    launch(0, h)?;
    dev.synchronize().map_err(|e| e.to_string())?;
    let full_seconds = full_start.elapsed().as_secs_f64();
    let full = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
    let full_id = save(out, "batch-full.bf16", &full)?;
    let full_float = dev.dtoh_sync_copy(&df).map_err(|e| e.to_string())?;
    let full_float_id = save_f32(out, "batch-full.f32", &full_float)?;
    if full_float.iter().any(|x| !x.is_finite())
        || full.iter().any(|&x| !bf16::from_bits(x).is_finite())
    {
        return Err("nonfinite tensor output; whole raw outputs retained".into());
    }
    if full[row * r..(row + 1) * r] != one[..r]
        || full_float[row * r..(row + 1) * r]
            .iter()
            .zip(one_float[..r].iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("selected singleton differs full tensor result; raw outputs retained".into());
    }
    let mut results = Vec::new();
    let mut min_free = floor;
    for case in &p.cases {
        let poison = vec![0x7fc1u16; mul(h, r)?];
        dev.htod_sync_copy_into(&poison, &mut dy)
            .map_err(|e| e.to_string())?;
        let float_poison = vec![f32::from_bits(0x7fc1a123); mul(h, r)?];
        dev.htod_sync_copy_into(&float_poison, &mut df)
            .map_err(|e| e.to_string())?;
        let begin = Instant::now();
        let mut offset = 0;
        for &n in &case.partitions {
            launch(offset, n)?;
            offset += n;
        }
        dev.synchronize().map_err(|e| e.to_string())?;
        let seconds = begin.elapsed().as_secs_f64();
        let got = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
        let got_float = dev.dtoh_sync_copy(&df).map_err(|e| e.to_string())?;
        let f32_exact = got_float[..case.height * r]
            .iter()
            .zip(full_float[..case.height * r].iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let f32_guard = got_float[case.height * r..]
            .iter()
            .all(|x| x.to_bits() == 0x7fc1a123);
        let f32_prefix: Vec<u8> = got_float[..case.height * r]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let f32_hash = format!("{:x}", Sha256::digest(&f32_prefix));
        if !f32_exact || !f32_guard {
            save_f32(
                out,
                &format!("case-{}.f32", case.name),
                &got_float[..case.height * r],
            )?;
        }
        let actual = save(
            out,
            &format!("case-{}.bf16", case.name),
            &got[..case.height * r],
        )?;
        let exact = got[..case.height * r] == full[..case.height * r];
        let untouched = got[case.height * r..].iter().all(|&v| v == 0x7fc1);
        results.push(json!({"case":case,"complete_prefix_exact":exact,"unused_output_poison_exact":untouched,"complete_f32_prefix_exact":f32_exact,"unused_f32_poison_exact":f32_guard,"f32_prefix_fingerprint":{"bytes":f32_prefix.len(),"sha256":f32_hash,"raw_saved_on_mismatch":!f32_exact||!f32_guard},"native_launch_and_completion_host_seconds":seconds,"output":actual}));
        fs::write(
            out.join("PARTIAL_RESULTS.json"),
            serde_json::to_vec_pretty(&results).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if !exact || !untouched || !f32_exact || !f32_guard {
            return Err(
                "complete prefix/partition/guard mismatch; partial evidence retained".into(),
            );
        }
        min_free = min_free.min(
            cudarc::driver::result::mem_get_info()
                .map_err(|e| e.to_string())?
                .0,
        );
        if min_free < p.safety_margin_bytes {
            return Err("observed free memory breached reserve".into());
        }
    }
    // Independent F64 mathematical reconstruction, selected authentic operand only.
    let mut refs = Vec::with_capacity(r);
    let mut d2 = 0.;
    let mut n2 = 0.;
    let mut max = 0f64;
    let mut float_d2 = 0f64;
    let mut float_max = 0f64;
    for rr in 0..r {
        let mut sum = 0f64;
        for col in 0..k {
            let byte = rr * ps + (col / 4) * 3;
            let word = (packed[byte] as u32)
                | ((packed[byte + 1] as u32) << 8)
                | ((packed[byte + 2] as u32) << 16);
            let q = ((word >> ((col % 4) * 6)) & 63) as f64;
            let gi = rr * (ss / 4) + col / g;
            let w = (q - zeros[gi] as f64) * scales[gi] as f64;
            sum += w * bf16::from_bits(x[row * k + col]).to_f64();
        }
        if !sum.is_finite() {
            return Err("nonfinite independent reference".into());
        }
        refs.push(sum);
        let d = bf16::from_bits(one[rr]).to_f64() - sum;
        d2 += d * d;
        let fd = one_float[rr] as f64 - sum;
        float_d2 += fd * fd;
        float_max = float_max.max(fd.abs());
        n2 += sum * sum;
        max = max.max(d.abs());
    }
    fs::write(
        out.join("selected-reference.f64"),
        refs.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    for x in p.capture.values() {
        x.read()?;
    }
    p.source.read()?;
    p.ptx.read()?;
    p.build_report.read()?;
    p.dispatch.read()?;
    Ok(
        json!({"passed":true,"diagnostic_only":true,"adopted":false,"plan":p,"actual_limits":limits,"full_grid":config.grid_dim,"memory":{"before_module_free":before.0,"after_module_free":after_module.0,"allocated_tensor_bytes":tensor_bytes,"observed_minimum_free":min_free,"safety_margin_bytes":600*1024*1024},"original_selected_matches_retained_native_exact":true,"batch_selected_matches_retained_native_exact":selected_diff==0,"selected_native_difference_coordinates":selected_diff,"kernel_contract":p.kernel_contract,"diagnostic_dispatch":"fixed CTA64x32,8warps,K32 staging/TF32 K8; not runtime autotuned", "selected_singleton_full_f32_exact":true,"full_f32_output":full_float_id,"full_output":full_id,"cases":results,"independent_selected_f64":{"coordinates":r,"max_abs":max,"difference_l2":d2.sqrt(),"reference_l2":n2.sqrt(),"relative_l2":if n2>0.{Some((d2/n2).sqrt())}else{None}},"selected_f32_reference_error":{"coordinates":r,"max_abs":float_max,"difference_l2":float_d2.sqrt(),"relative_l2":if n2>0.{Some((float_d2/n2).sqrt())}else{None}},"native_launch_and_completion_host_seconds_full":full_seconds,"limits_of_claim":["New tensor accumulation and component arithmetic unadopted; old-native differences quantified rather than assumed equal","Provenance explicitly admitted: actual Q, controlled beta Q-matrix, or synthetic numerical range","Stateless operator only, no integrated recurrence/cache/ordinary qualification","Instrumented synchronized diagnostic host timing is not an uninstrumented engine or HTTP benchmark"]}),
    )
}
fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev row-projection-compensated-probe".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 || !["build", "probe"].contains(&args[0].as_str()) {
        return Err("./dev row-projection-probe <build|probe> <frozen-plan> <fresh-output>".into());
    }
    let out = PathBuf::from(&args[2]);
    fs::create_dir(&out).map_err(|e| format!("fresh output required: {e}"))?;
    let result = match args[0].as_str() {
        "build" => build(Path::new(&args[1]), &out),
        "probe" => probe(Path::new(&args[1]), &out),
        _ => unreachable!(),
    };
    match result {
        Ok(r) => {
            fs::write(
                out.join("REPORT.json"),
                serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            println!("Evidence saved {}", out.display());
            Ok(())
        }
        Err(e) => {
            fs::write(out.join("FAILURE.txt"), &e).map_err(|x| x.to_string())?;
            Err(e)
        }
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_grid_spans_y_and_z_and_rejects_device_limits() {
        let cf = grid(17, 11, 2, [1024, 49152, 100, 4, 3], 8192).unwrap();
        assert_eq!(cf.grid_dim, (9, 4, 3));
        assert!(grid(17, 13, 2, [1024, 49152, 100, 4, 3], 8192).is_err());
        assert!(grid(17, 11, 2, [32, 49152, 100, 4, 3], 8192).is_err());
        assert!(grid(17, 11, 2, [1024, 4000, 100, 4, 3], 8192).is_err());
        assert!(grid(17, 11, 2, [1024, 49152, 8, 4, 3], 8192).is_err());
    }
    #[test]
    fn incomplete_or_overflow_partition_is_rejected_before_cuda() {
        assert!(validate_cases(
            &[Case {
                name: "ok".into(),
                height: 5,
                partitions: vec![2, 3]
            }],
            5
        )
        .is_ok());
        for p in [vec![2, 2], vec![0, 5], vec![usize::MAX, 1]] {
            assert!(validate_cases(
                &[Case {
                    name: "bad".into(),
                    height: 5,
                    partitions: p
                }],
                5
            )
            .is_err())
        }
    }
    #[test]
    fn nonfinite_operands_reject() {
        assert!(u16s(&0x7fc1u16.to_le_bytes()).is_err());
        assert!(f32s(&f32::INFINITY.to_le_bytes()).is_err());
        assert!(mul(usize::MAX, 2).is_err());
    }
}
