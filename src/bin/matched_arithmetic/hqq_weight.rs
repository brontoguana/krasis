// Bounded diagnostic only. The serving HQQ kernels are never replaced here.
use super::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, process::Command};
const ABI: &str = "hqq6-packed3-f32-scales-zeros-bf16-input-scalar-or-vec4-v1";
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Identity {
    fn verify(&self) -> Result<Vec<u8>, String> {
        let data = read(&self.path, self.bytes)?;
        if self.sha256.len() != 64 || format!("{:x}", Sha256::digest(&data)) != self.sha256 {
            return Err(format!("identity mismatch: {}", self.path.display()));
        }
        Ok(data)
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BuildPlan {
    schema_version: u32,
    abi: String,
    source: Identity,
    nvcc: Identity,
    target_compute: u32,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProbePlan {
    schema_version: u32,
    abi: String,
    source: Identity,
    ptx: Identity,
    build_report: Identity,
    capture: BTreeMap<String, Identity>,
    dispatch: Identity,
    gpu_ordinal: usize,
    compute_capability: u32,
    gpu_name: String,
    measured_decode_warps: usize,
    safety_margin_bytes: usize,
}
fn build_validation(plan: &BuildPlan) -> Result<(), String> {
    // This alternative has the same Ampere-minimum ISA contract as decode_kernels.ptx.
    if plan.schema_version != 1 || plan.abi != ABI || plan.target_compute != 80 {
        return Err("unsupported HQQ candidate source/ABI/target contract".into());
    }
    Ok(())
}
pub(super) fn build(input: &Path, out: &Path) -> Result<Value, String> {
    let plan: BuildPlan = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    build_validation(&plan)?;
    plan.source.verify()?;
    plan.nvcc.verify()?;
    let flags = [
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
    ];
    let result = Command::new(&plan.nvcc.path)
        .args(flags)
        .arg("-o")
        .arg(out.join("candidate.ptx"))
        .arg(&plan.source.path)
        .output()
        .map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stdout"), &result.stdout).map_err(|e| e.to_string())?;
    fs::write(out.join("compiler.stderr"), &result.stderr).map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!("candidate CUDA build failed: {}", result.status));
    }
    plan.source.verify()?;
    plan.nvcc.verify()?;
    let ptx = fs::read(out.join("candidate.ptx")).map_err(|e| e.to_string())?;
    let text = std::str::from_utf8(&ptx).map_err(|e| e.to_string())?;
    for name in [
        "hqq6_bf16_weight_gemv_f32",
        "hqq6_bf16_weight_gemv_bf16",
        "hqq6_bf16_weight_gemv_bf16_vec4",
    ] {
        if !text.contains(name) {
            return Err(format!("candidate PTX missing {name}"));
        }
    }
    Ok(
        json!({"diagnostic_only":true,"qualifies_candidate":false,"plan":plan,"compiler_flags":flags,"ptx_sha256":format!("{:x}",Sha256::digest(&ptx)),"ptx_bytes":ptx.len(),"changes_only":"HQQ reconstructed weight BF16 RNE before unchanged F32 dot/reduction; serving kernels unchanged"}),
    )
}
fn geometry(
    desc: &Value,
    warps: usize,
) -> Result<(usize, usize, usize, usize, usize, usize), String> {
    let rows = number(desc, "rows")?;
    let cols = number(desc, "cols")?;
    let group = number(desc, "group_size")?;
    let ps = number(desc, "packed_row_stride_bytes")?;
    let ss = number(desc, "scales_row_stride_bytes")?;
    let zs = number(desc, "zeros_row_stride_bytes")?;
    if desc["nbits"] != 6
        || desc["materialize_bf16"] != true
        || cols % 4 != 0
        || group % 4 != 0
        || cols % group != 0
        || ![1, 2, 4, 8, 16, 32].contains(&warps)
        || [rows, cols, group, ps, ss, zs]
            .iter()
            .any(|&n| n > i32::MAX as usize)
        || ps != cols.checked_mul(3).ok_or("packed size overflow")? / 4
        || ss != cols / group * 4
        || zs != ss
    {
        return Err("unsupported HQQ6 packed/scale/zero layout or warp geometry".into());
    }
    Ok((rows, cols, group, ps, ss, zs))
}
pub(super) fn probe(input: &Path, out: &Path, gpu: usize, warps: usize) -> Result<Value, String> {
    let plan: ProbePlan = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if plan.schema_version != 1
        || plan.abi != ABI
        || plan.gpu_ordinal != gpu
        || plan.measured_decode_warps != warps
    {
        return Err("HQQ probe plan/CLI/ABI mismatch".into());
    }
    let source = plan.source.verify()?;
    let ptx = plan.ptx.verify()?;
    let build: Value =
        serde_json::from_slice(&plan.build_report.verify()?).map_err(|e| e.to_string())?;
    if build["plan"]["abi"] != ABI
        || build["plan"]["source"]["sha256"] != format!("{:x}", Sha256::digest(&source))
        || build["ptx_sha256"] != plan.ptx.sha256
        || build["plan"]["target_compute"] != 80
    {
        return Err("source/PTX/build provenance mismatch".into());
    }
    let names = [
        "descriptor.json",
        "packed.bin",
        "scales.f32",
        "zeros.f32",
        "input.bf16",
        "materialized.bf16",
        "production-prefill.bf16",
    ];
    if plan.capture.len() != names.len()
        || names.iter().any(|name| !plan.capture.contains_key(*name))
    {
        return Err("capture identities incomplete or unexpected".into());
    }
    let mut data = BTreeMap::new();
    for (name, id) in &plan.capture {
        data.insert(name.as_str(), id.verify()?);
    }
    let desc: Value =
        serde_json::from_slice(&data["descriptor.json"]).map_err(|e| e.to_string())?;
    let (rows, cols, group, ps, ss, zs) = geometry(&desc, warps)?;
    let expect = |name: &str, count: usize| -> Result<(), String> {
        if data[name].len() != count {
            return Err(format!("capture shape mismatch {name}"));
        }
        Ok(())
    };
    let size = |a: usize, b: usize| a.checked_mul(b).ok_or("capture byte overflow");
    expect("packed.bin", size(rows, ps)?)?;
    expect("scales.f32", size(rows, ss)?)?;
    expect("zeros.f32", size(rows, zs)?)?;
    expect("input.bf16", size(cols, 2)?)?;
    expect("materialized.bf16", size(size(rows, cols)?, 2)?)?;
    expect("production-prefill.bf16", size(rows, 2)?)?;
    let dispatch: Value =
        serde_json::from_slice(&plan.dispatch.verify()?).map_err(|e| e.to_string())?;
    let matches: Vec<_> = dispatch["selections"]
        .as_array()
        .ok_or("missing measured dispatch")?
        .iter()
        .filter(|d| d["rows"] == rows && d["cols"] == cols && d["group_size"] == group)
        .collect();
    if matches.len() != 1 || matches[0]["family"] != "vec4" || matches[0]["warps"] != warps {
        return Err("geometry not bound to measured original dispatch".into());
    }
    let packed = &data["packed.bin"];
    let scales = f32s(&data["scales.f32"]);
    let zeros = f32s(&data["zeros.f32"]);
    let x = u16s(&data["input.bf16"]);
    let materialized = u16s(&data["materialized.bf16"]);
    let captured = u16s(&data["production-prefill.bf16"]);
    let mut raw = vec![0f32; size(rows, cols)?];
    let mut rounded = raw.clone();
    let mut ideal_raw = vec![0f64; rows];
    let mut ideal_rounded = ideal_raw.clone();
    for r in 0..rows {
        for c in 0..cols {
            let i = r * ps + c / 4 * 3;
            let bits =
                packed[i] as u32 | (packed[i + 1] as u32) << 8 | (packed[i + 2] as u32) << 16;
            let q = ((bits >> ((c % 4) * 6)) & 63) as f32;
            let w = (q - zeros[r * (zs / 4) + c / group]) * scales[r * (ss / 4) + c / group];
            let wb = bf16::from_f32(w);
            let xf = bf16::from_bits(x[c]).to_f32();
            if !w.is_finite() || !xf.is_finite() || !wb.to_f32().is_finite() {
                return Err("nonfinite captured operands".into());
            }
            if wb.to_bits() != materialized[r * cols + c] {
                return Err(
                    "independent reconstructed BF16 weight differs from captured materialization"
                        .into(),
                );
            }
            raw[r * cols + c] = w;
            rounded[r * cols + c] = wb.to_f32();
            ideal_raw[r] += w as f64 * xf as f64;
            ideal_rounded[r] += wb.to_f64() * xf as f64;
        }
    }
    let reference = |w: &[f32], vec4: bool| -> Vec<f32> {
        (0..rows)
            .map(|r| {
                let mut lanes = [0f32; 32];
                for lane in 0..32 {
                    if vec4 {
                        for quartet in (lane..cols / 4).step_by(32) {
                            for off in 0..4 {
                                let c = quartet * 4 + off;
                                lanes[lane] = w[r * cols + c]
                                    .mul_add(bf16::from_bits(x[c]).to_f32(), lanes[lane]);
                            }
                        }
                    } else {
                        for c in (lane..cols).step_by(32) {
                            lanes[lane] = w[r * cols + c]
                                .mul_add(bf16::from_bits(x[c]).to_f32(), lanes[lane]);
                        }
                    }
                }
                for off in [16, 8, 4, 2, 1] {
                    let old = lanes;
                    for lane in 0..32 - off {
                        lanes[lane] = old[lane] + old[lane + off];
                    }
                }
                lanes[0]
            })
            .collect()
    };
    let scalar_raw = reference(&raw, false);
    let vec_raw = reference(&raw, true);
    let scalar_rounded = reference(&rounded, false);
    let vec_rounded = reference(&rounded, true);
    for (name, values) in [
        ("original-scalar", &scalar_raw),
        ("original-vec4", &vec_raw),
        ("bf16-weight-scalar", &scalar_rounded),
        ("bf16-weight-vec4", &vec_rounded),
    ] {
        fs::write(
            out.join(format!("reference-{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        save16(
            &out.join(format!("reference-{name}.bf16")),
            &values
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect::<Vec<_>>(),
        )?;
    }
    save64(&out.join("reference-f32-weight.f64"), &ideal_raw)?;
    save64(&out.join("reference-bf16-weight.f64"), &ideal_rounded)?;
    let dev = CudaDevice::new(gpu).map_err(|e| e.to_string())?;
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let cc = dev
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| e.to_string())?
        * 10
        + dev
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?;
    let name = dev.name().map_err(|e| e.to_string())?;
    if cc as u32 != plan.compute_capability || name != plan.gpu_name || cc < 80 {
        return Err("actual device does not match admitted hardware/ISA".into());
    }
    let before_module = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    let embedded = include_str!(concat!(env!("OUT_DIR"), "/decode_kernels.ptx"));
    dev.load_ptx(
        cudarc::nvrtc::Ptx::from_src(embedded),
        "weight_original",
        &[
            "hqq6_decode_gemv_f32",
            "hqq6_decode_gemv_bf16",
            "hqq6_decode_gemv_bf16_vec4",
        ],
    )
    .map_err(|e| e.to_string())?;
    dev.load_ptx(
        cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx).map_err(|e| e.to_string())?),
        "weight_candidate",
        &[
            "hqq6_bf16_weight_gemv_f32",
            "hqq6_bf16_weight_gemv_bf16",
            "hqq6_bf16_weight_gemv_bf16_vec4",
        ],
    )
    .map_err(|e| e.to_string())?;
    let after_module = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    let tensor_bytes = packed
        .len()
        .checked_add(scales.len().checked_mul(4).ok_or("scale byte overflow")?)
        .and_then(|n| n.checked_add(zeros.len() * 4))
        .and_then(|n| n.checked_add(x.len() * 2))
        .and_then(|n| n.checked_add(rows * 4))
        .ok_or("GPU tensor byte overflow")?;
    if plan.safety_margin_bytes == 0
        || after_module.0
            < tensor_bytes
                .checked_add(plan.safety_margin_bytes)
                .ok_or("GPU admission overflow")?
    {
        return Err(
            "measured free VRAM cannot cover operator tensors plus configured margin".into(),
        );
    }
    let dp = dev.htod_copy(packed.clone()).map_err(|e| e.to_string())?;
    let ds = dev.htod_copy(scales).map_err(|e| e.to_string())?;
    let dz = dev.htod_copy(zeros).map_err(|e| e.to_string())?;
    let dx = dev.htod_copy(x).map_err(|e| e.to_string())?;
    let mut paths = Vec::new();
    let mut minimum_observed_free = after_module.0;
    for (is_candidate, module, prefix) in [
        (false, "weight_original", "hqq6_decode"),
        (true, "weight_candidate", "hqq6_bf16_weight"),
    ] {
        for (suffix, f32_output, quartet) in [
            ("gemv_f32", true, false),
            ("gemv_bf16", false, false),
            ("gemv_bf16_vec4", false, true),
        ] {
            let name = format!("{prefix}_{suffix}");
            let reference = match (is_candidate, quartet) {
                (false, false) => &scalar_raw,
                (false, true) => &vec_raw,
                (true, false) => &scalar_rounded,
                (true, true) => &vec_rounded,
            };
            let threads = if quartet { warps * 32 } else { 256 };
            let config = LaunchConfig {
                grid_dim: (rows.div_ceil(threads / 32) as u32, 1, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: (cols * 2) as u32,
            };
            let func = dev.get_func(module, &name).ok_or("HQQ kernel missing")?;
            let values: Vec<f64>;
            let exact: bool;
            if f32_output {
                let y = dev.alloc_zeros::<f32>(rows).map_err(|e| e.to_string())?;
                unsafe {
                    func.launch(
                        config,
                        (
                            *dp.device_ptr(),
                            *ds.device_ptr(),
                            *dz.device_ptr(),
                            *dx.device_ptr(),
                            *y.device_ptr(),
                            rows as i32,
                            cols as i32,
                            group as i32,
                            ps as i32,
                            ss as i32,
                            zs as i32,
                        ),
                    )
                    .map_err(|e| e.to_string())?;
                }
                let result = dev.dtoh_sync_copy(&y).map_err(|e| e.to_string())?;
                minimum_observed_free = minimum_observed_free.min(
                    cudarc::driver::result::mem_get_info()
                        .map_err(|e| e.to_string())?
                        .0,
                );
                exact = result
                    .iter()
                    .zip(reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                fs::write(
                    out.join(format!("{name}.f32")),
                    result
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?;
                values = result.iter().map(|&v| v as f64).collect();
            } else {
                let y = dev.alloc_zeros::<u16>(rows).map_err(|e| e.to_string())?;
                unsafe {
                    func.launch(
                        config,
                        (
                            *dp.device_ptr(),
                            *ds.device_ptr(),
                            *dz.device_ptr(),
                            *dx.device_ptr(),
                            *y.device_ptr(),
                            rows as i32,
                            cols as i32,
                            group as i32,
                            ps as i32,
                            ss as i32,
                            zs as i32,
                        ),
                    )
                    .map_err(|e| e.to_string())?;
                }
                let result = dev.dtoh_sync_copy(&y).map_err(|e| e.to_string())?;
                minimum_observed_free = minimum_observed_free.min(
                    cudarc::driver::result::mem_get_info()
                        .map_err(|e| e.to_string())?
                        .0,
                );
                exact = result
                    .iter()
                    .zip(reference)
                    .all(|(&a, &b)| a == bf16::from_f32(b).to_bits());
                save16(&out.join(format!("{name}.bf16")), &result)?;
                values = bfvalues(&result);
            }
            let reference_values: Vec<f64> = reference
                .iter()
                .map(|&v| {
                    if f32_output {
                        v as f64
                    } else {
                        bf16::from_f32(v).to_f64()
                    }
                })
                .collect();
            paths.push(json!({"kernel":name,"candidate":is_candidate,"threads":threads,"output_dtype":if f32_output{"f32"}else{"bf16"},"matching_order_reference_exact":exact,"vs_matching_order_reference":metrics(&values,&reference_values)?,"vs_f64_f32_weight_reference":metrics(&values,&ideal_raw)?,"vs_f64_bf16_weight_reference":metrics(&values,&ideal_rounded)?,"vs_original_prefill":metrics(&values,&bfvalues(&captured))?}));
        }
    }
    let after_runs = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
    if minimum_observed_free < plan.safety_margin_bytes {
        return Err("measured operator VRAM free below configured margin".into());
    }
    let all_exact = paths
        .iter()
        .all(|p| p["matching_order_reference_exact"] == true);
    let report = json!({"diagnostic_only":true,"qualifies_candidate":false,"plan":plan,"descriptor":desc,"gpu_name":name,"compute_capability":cc,"embedded_production_ptx_sha256":format!("{:x}",Sha256::digest(embedded.as_bytes())),"independent_materialization_exact":true,"all_six_matching_order_references_exact":all_exact,"memory":{"before_module_free":before_module.0,"after_module_free":after_module.0,"after_runs_free":after_runs.0,"minimum_observed_free":minimum_observed_free,"configured_margin":plan.safety_margin_bytes,"operator_tensor_bytes":tensor_bytes},"paths":paths,"limits":["Single captured input; no full-model or ordinary consistency qualification","Original BF16 safetensor/Witness accuracy not measured; rounding to match a path is not a justified production repair","All serving kernels, autotune, calibration, graphs/cache, defaults and budgets unchanged"]});
    // Save negative evidence before returning the exact-parity failure.
    fs::write(
        out.join("PARITY_REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !all_exact {
        return Err(
            "HQQ matching-order CPU/CUDA parity failed; evidence retained, no whole-model adoption"
                .into(),
        );
    }
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_rejects_unbound_layout_and_launch() {
        let good = json!({"rows":8192,"cols":4096,"group_size":128,"nbits":6,"materialize_bf16":true,"packed_row_stride_bytes":3072,"scales_row_stride_bytes":128,"zeros_row_stride_bytes":128});
        assert!(geometry(&good, 16).is_ok());
        for (key, value) in [
            ("cols", json!(4095)),
            ("group_size", json!(126)),
            ("nbits", json!(4)),
            ("materialize_bf16", json!(false)),
            ("scales_row_stride_bytes", json!(124)),
            ("packed_row_stride_bytes", json!(3071)),
            ("zeros_row_stride_bytes", json!(124)),
            ("rows", json!(0)),
        ] {
            let mut bad = good.clone();
            bad[key] = value;
            assert!(geometry(&bad, 16).is_err(), "{key}")
        }
        assert!(geometry(&good, 3).is_err());
    }
}
