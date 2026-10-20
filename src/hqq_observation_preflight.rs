//! CPU-only admission using the exact production execution-format builder.
use super::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Identity {
    fn read(&self) -> Result<Vec<u8>, String> {
        use sha2::{Digest, Sha256};
        if !self.path.is_absolute() {
            return Err("absolute HQQ source identity required".into());
        }
        let bytes = std::fs::read(&self.path).map_err(|e| e.to_string())?;
        if bytes.len() != self.bytes || format!("{:x}", Sha256::digest(&bytes)) != self.sha256 {
            return Err("HQQ source identity changed before cold preflight".into());
        }
        Ok(bytes)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    cache: Identity,
    rows: usize,
    cols: usize,
    group_size: usize,
    nbits: u8,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema_version: u32,
    model_config: Identity,
    entries: Vec<Entry>,
    admission_files: Vec<Identity>,
}
fn number(t: &safetensors::SafeTensors<'_>, name: &str) -> Result<usize, String> {
    let tensor = t.tensor(name).map_err(|e| e.to_string())?;
    if tensor.shape() != [1] {
        return Err(format!("{name} is not an actual scalar tensor"));
    }
    match tensor.dtype() {
        safetensors::Dtype::I32 => usize::try_from(i32::from_le_bytes(
            tensor.data().try_into().map_err(|_| "invalid i32 scalar")?,
        ))
        .map_err(|e| e.to_string()),
        safetensors::Dtype::I64 => usize::try_from(i64::from_le_bytes(
            tensor.data().try_into().map_err(|_| "invalid i64 scalar")?,
        ))
        .map_err(|e| e.to_string()),
        _ => Err("unsupported actual HQQ scalar representation".into()),
    }
}
fn run(plan_path: &Path, root: &Path, materialize_default: bool) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    if !plan_path.is_absolute() || !root.is_absolute() || root.exists() {
        return Err("absolute frozen plan and fresh host-only output required".into());
    }
    let bytes = std::fs::read(plan_path).map_err(|e| e.to_string())?;
    let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if plan.schema_version != 1 || plan.entries.is_empty() {
        return Err("empty actual HQQ host preflight".into());
    }
    plan.model_config.read()?;
    for file in &plan.admission_files {
        file.read()?;
    }
    let materialize = resolve_bool_env("KRASIS_HQQ_PREFILL_MATERIALIZE_BF16", materialize_default)?;
    // Complete input and output-extent admission precedes all writes; no CUDA API.
    let mut payloads = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for entry in &plan.entries {
        if entry.name.is_empty()
            || entry.name == "."
            || entry.name == ".."
            || entry.name.contains('/')
            || entry.name.contains('\\')
            || !names.insert(entry.name.clone())
            || entry.rows == 0
            || entry.cols == 0
            || entry.group_size == 0
        {
            return Err("invalid actual host-preflight tensor inventory".into());
        }
        let source = entry.cache.read()?;
        let tensors = safetensors::SafeTensors::deserialize(&source).map_err(|e| e.to_string())?;
        if number(&tensors, "group_size")? != entry.group_size
            || number(&tensors, "nbits")? != entry.nbits as usize
            || number(&tensors, "axis")? != 1
        {
            return Err("actual canonical tensor scalar metadata differs".into());
        }
        let shape = tensors.tensor("orig_shape").map_err(|e| e.to_string())?;
        if shape.shape() != [2] {
            return Err("actual original HQQ tensor rank differs".into());
        }
        let shape_values: Vec<usize> = match shape.dtype() {
            safetensors::Dtype::I32 => shape
                .data()
                .chunks_exact(4)
                .map(|b| {
                    usize::try_from(i32::from_le_bytes(b.try_into().unwrap()))
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<_, _>>()?,
            safetensors::Dtype::I64 => shape
                .data()
                .chunks_exact(8)
                .map(|b| {
                    usize::try_from(i64::from_le_bytes(b.try_into().unwrap()))
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("unsupported actual original HQQ shape representation".into()),
        };
        if shape_values != [entry.rows, entry.cols] {
            return Err("actual original HQQ dimensions differ from admitted geometry".into());
        }
        let packed = tensors.tensor("packed").map_err(|e| e.to_string())?;
        let scales = tensors.tensor("scales").map_err(|e| e.to_string())?;
        let zeros = tensors.tensor("zeros").map_err(|e| e.to_string())?;
        let groups = entry.cols.div_ceil(entry.group_size);
        let expected = [entry.rows, groups];
        if packed.dtype() != safetensors::Dtype::U8
            || scales.dtype() != safetensors::Dtype::F32
            || zeros.dtype() != safetensors::Dtype::F32
            || scales.shape() != expected
            || zeros.shape() != expected
        {
            return Err("actual canonical HQQ plane shape/dtype differs".into());
        }
        let actual = GpuDecodeStore::build_hqq_prefill_runtime_format(
            packed.data(),
            scales.data(),
            zeros.data(),
            entry.rows,
            entry.cols,
            entry.group_size,
            entry.nbits as usize,
            materialize,
        )?;
        let total = actual
            .packed_bytes()
            .checked_add(actual.scales_bytes())
            .and_then(|x| x.checked_add(actual.zeros_bytes()))
            .ok_or("actual host execution payload overflow")?;
        let layout = crate::hqq_observation_layout::describe(
            entry.rows,
            entry.cols,
            entry.group_size,
            entry.nbits,
            &actual.layout,
            &actual.packed_dtype,
            &actual.scales_dtype,
            &actual.zeros_dtype,
            total,
        )?;
        if (layout.packed, layout.scales, layout.zeros)
            != (
                actual.packed_bytes(),
                actual.scales_bytes(),
                actual.zeros_bytes(),
            )
        {
            return Err(
                "observer layout disagrees with actual production host execution-plane lengths"
                    .into(),
            );
        }
        let old_assumption = actual.layout == hqq_expected_layout(entry.nbits, "prefill")?
            && actual.packed_dtype == "uint8"
            && actual.scales_dtype == "float32"
            && actual.zeros_dtype == "float32";
        payloads.push((entry, actual, layout, old_assumption));
    }
    std::fs::create_dir(root).map_err(|e| e.to_string())?;
    let mut results = Vec::new();
    for (entry, actual, layout, old_assumption) in payloads {
        let dir = root.join(&entry.name);
        std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        let mut files = Vec::new();
        for (name, data) in [
            ("packed.bin", actual.packed_host.as_slice()),
            (layout.scales_file, actual.scales_host.as_slice()),
            (layout.zeros_file, actual.zeros_host.as_slice()),
        ] {
            let path = dir.join(name);
            std::fs::write(&path, data).map_err(|e| e.to_string())?;
            files.push(serde_json::json!({"path":path,"bytes":data.len(),"sha256":format!("{:x}",Sha256::digest(data))}));
        }
        results.push(serde_json::json!({"name":entry.name,"canonical_source":{"path":entry.cache.path,"bytes":entry.cache.bytes,"sha256":entry.cache.sha256},"actual_production_format":hqq_runtime_format_json(&actual),"actual_observer_payloads":layout.metadata,"original193_row_major_assumption_accepted":old_assumption,"files":files}));
    }
    let report = serde_json::json!({"passed":true,"gpu_initialized":false,"model_loaded":false,"materialize_default_from_actual_model_config":materialize_default,"resolved_materialize_bf16":materialize,"tensors":results,"scope":"Exact production CPU format builder on checksum-bound actual canonical cache, descriptor-specific observer extents versus real output planes. No CUDA or model run, no numerical pipeline comparison or serving qualification."});
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(root.join("REPORT.json"), &text).map_err(|e| e.to_string())?;
    Ok(text)
}
#[pyfunction]
pub(crate) fn hqq_prefill_observation_preflight(
    py: Python<'_>,
    plan_path: String,
    output: String,
    materialize_default: bool,
) -> PyResult<String> {
    py.allow_threads(move || {
        run(
            Path::new(&plan_path),
            Path::new(&output),
            materialize_default,
        )
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}
