use super::*;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
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
        if !self.path.is_absolute() || self.sha256.len() != 64 {
            return Err("invalid accuracy artifact identity".into());
        }
        let b = read(&self.path, self.bytes)?;
        if format!("{:x}", Sha256::digest(&b)) != self.sha256 {
            return Err(format!(
                "accuracy checksum mismatch {}",
                self.path.display()
            ));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: u32,
    index: Identity,
    shard: PathBuf,
    tensor: String,
    header_sha256: String,
    payload_sha256: String,
    rows: usize,
    cols: usize,
    input: Identity,
    contracts: BTreeMap<String, Identity>,
}
pub(super) fn run(input: &Path, out: &Path) -> Result<Value, String> {
    let p: Plan = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if p.schema != 1
        || !p.shard.is_absolute()
        || p.rows == 0
        || p.cols == 0
        || p.contracts.len() != 6
    {
        return Err("invalid original-BF16 accuracy plan".into());
    }
    let index: Value = serde_json::from_slice(&p.index.read()?).map_err(|e| e.to_string())?;
    if index["weight_map"][&p.tensor].as_str() != p.shard.file_name().and_then(|s| s.to_str()) {
        return Err("original tensor index/shard mismatch".into());
    }
    let mut f = fs::File::open(&p.shard).map_err(|e| e.to_string())?;
    let file_bytes = f.metadata().map_err(|e| e.to_string())?.len();
    let mut word = [0u8; 8];
    f.read_exact(&mut word).map_err(|e| e.to_string())?;
    let header_bytes = u64::from_le_bytes(word);
    if header_bytes > file_bytes.saturating_sub(8) {
        return Err("invalid safetensors header length".into());
    }
    let mut header =
        vec![0u8; usize::try_from(header_bytes).map_err(|_| "header exceeds address space")?];
    f.read_exact(&mut header).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&header)) != p.header_sha256 {
        return Err("original BF16 header identity changed".into());
    }
    let metadata: Value = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    let tensor = &metadata[&p.tensor];
    if tensor["dtype"] != "BF16" || tensor["shape"] != json!([p.rows, p.cols]) {
        return Err("original BF16 tensor dtype/shape mismatch".into());
    }
    let start = tensor["data_offsets"][0]
        .as_u64()
        .ok_or("missing tensor start")?;
    let end = tensor["data_offsets"][1]
        .as_u64()
        .ok_or("missing tensor end")?;
    let expected = p
        .rows
        .checked_mul(p.cols)
        .and_then(|x| x.checked_mul(2))
        .ok_or("BF16 tensor size overflow")?;
    let base = 8u64
        .checked_add(header_bytes)
        .ok_or("safetensors base overflow")?;
    if end.checked_sub(start) != Some(expected as u64)
        || base.checked_add(end).map_or(true, |x| x > file_bytes)
    {
        return Err("original BF16 tensor offsets invalid".into());
    }
    f.seek(SeekFrom::Start(base + start))
        .map_err(|e| e.to_string())?;
    let mut weights = vec![0; expected];
    f.read_exact(&mut weights).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&weights)) != p.payload_sha256 {
        return Err("original BF16 tensor identity changed".into());
    }
    let x = p.input.read()?;
    if x.len() != p.cols * 2 {
        return Err("accuracy input shape mismatch".into());
    }
    let xv = bfvalues(&u16s(&x));
    let wv = u16s(&weights);
    let mut reference = vec![0.; p.rows];
    for (r, y) in reference.iter_mut().enumerate() {
        for c in 0..p.cols {
            let w = bf16::from_bits(wv[r * p.cols + c]).to_f64();
            if !w.is_finite() || !xv[c].is_finite() {
                return Err("nonfinite original BF16 operands".into());
            }
            *y += w * xv[c];
        }
    }
    save64(&out.join("original-bf16-weight-dot.f64"), &reference)?;
    let mut results = BTreeMap::new();
    let mut values = BTreeMap::new();
    for (name, id) in &p.contracts {
        let b = id.read()?;
        let actual = if name.ends_with(".f32") {
            if b.len() != p.rows * 4 {
                return Err("F32 output shape mismatch".into());
            }
            f32s(&b).iter().map(|&x| x as f64).collect::<Vec<_>>()
        } else if name.ends_with(".bf16") {
            if b.len() != p.rows * 2 {
                return Err("BF16 output shape mismatch".into());
            }
            bfvalues(&u16s(&b))
        } else {
            return Err("unadmitted output precision".into());
        };
        results.insert(name.clone(), metrics(&actual, &reference)?);
        values.insert(name.clone(), actual);
    }
    let mut comparisons = Vec::new();
    for (native, candidate) in [
        ("hqq6_decode_gemv_f32.f32", "hqq6_bf16_weight_gemv_f32.f32"),
        (
            "hqq6_decode_gemv_bf16.bf16",
            "hqq6_bf16_weight_gemv_bf16.bf16",
        ),
        (
            "hqq6_decode_gemv_bf16_vec4.bf16",
            "hqq6_bf16_weight_gemv_bf16_vec4.bf16",
        ),
    ] {
        let a = values.get(native).ok_or("missing native output")?;
        let b = values.get(candidate).ok_or("missing rounded output")?;
        let mut better = 0;
        let mut worse = 0;
        let mut equal = 0;
        let mut worst_delta = 0f64;
        for ((&a, &b), &r) in a.iter().zip(b).zip(&reference) {
            let ea = (a - r).abs();
            let eb = (b - r).abs();
            if eb < ea {
                better += 1;
            } else if eb > ea {
                worse += 1;
            } else {
                equal += 1;
            }
            worst_delta = worst_delta.max(eb - ea);
        }
        comparisons.push(json!({"native":native,"rounded":candidate,"better_coordinates":better,"worse_coordinates":worse,"equal_coordinates":equal,"largest_absolute_error_increase":worst_delta}));
    }
    Ok(
        json!({"diagnostic_only":true,"qualifies_candidate":false,"reference":"original BF16 safetensors weights, exact same captured BF16 operand, independent F64 dot","tensor":p.tensor,"weight_sha256":p.payload_sha256,"header_sha256":p.header_sha256,"input_sha256":p.input.sha256,"paths":results,"coordinate_regressions_retained":comparisons,"limits":"One original-weight projection comparison; does not establish whole-model decode/cached accuracy or Witness qualification"}),
    )
}
