//! Captured actual HQQ6/F32 and materialized-BF16 mathematical references.
//! Offline CPU F64 reductions; no GPU dispatch, serving or Witness verdict.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
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
        if b.len() != self.bytes || format!("{:x}", Sha256::digest(&b)) != self.sha256 {
            return Err(format!(
                "captured identity mismatch {}",
                self.path.display()
            ));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
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
fn finite(v: f64) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err("nonfinite captured/reference value".into())
    }
}
fn bf(b: &[u8], i: usize) -> Result<f64, String> {
    finite(f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16) as f64)
}
fn f32at(b: &[u8], i: usize) -> Result<f32, String> {
    let v = f32::from_le_bytes(b[i..i + 4].try_into().map_err(|_| "F32 extent")?);
    finite(v as f64)?;
    Ok(v)
}
fn rne(v: f64) -> Result<u16, String> {
    finite(v)?;
    let f = v as f32;
    finite(f as f64)?;
    let b = f.to_bits();
    Ok((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16)
}
fn quant(row: &[u8], k: usize) -> u32 {
    let at = (k / 4) * 3;
    let bits = (row[at] as u32) | ((row[at + 1] as u32) << 8) | ((row[at + 2] as u32) << 16);
    (bits >> ((k % 4) * 6)) & 63
}
fn coefficient(q: u32, zero: f32, scale: f32) -> Result<f32, String> {
    let difference = q as f32 - zero;
    let value = difference * scale;
    finite(value as f64)?;
    Ok(value)
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("reference extent overflow".into())
}
fn validate(p: &Plan) -> Result<(), String> {
    if p.schema != 1
        || p.layout != "hqq6_row_major_f32_parameters"
        || p.rows == 0
        || p.cols == 0
        || p.group_size == 0
        || p.cols % 4 != 0
        || p.group_size % 4 != 0
        || p.cols % p.group_size != 0
        || p.packed_stride != mul(p.cols / 4, 3)?
        || p.scales_stride % 4 != 0
        || p.zeros_stride % 4 != 0
        || p.scales_stride < mul(p.cols / p.group_size, 4)?
        || p.zeros_stride < mul(p.cols / p.group_size, 4)?
        || !matches!(p.decode_output_dtype.as_str(), "f32" | "bf16")
    {
        return Err("unsupported actual HQQ6/F32 reference geometry/dtype".into());
    }
    Ok(())
}
#[derive(Default)]
struct ErrorStats {
    different_f32: usize,
    different_bf16: usize,
    max_abs: f64,
    sse: f64,
    norm: f64,
}
impl ErrorStats {
    fn add(&mut self, actual: f64, reference: f64) -> Result<(), String> {
        finite(actual)?;
        finite(reference)?;
        self.different_f32 +=
            usize::from((actual as f32).to_bits() != (reference as f32).to_bits());
        self.different_bf16 += usize::from(rne(actual)? != rne(reference)?);
        let error = actual - reference;
        self.max_abs = self.max_abs.max(error.abs());
        self.sse += error * error;
        self.norm += reference * reference;
        Ok(())
    }
    fn report(&self) -> serde_json::Value {
        json!({"different_from_f64_rounded_f32":self.different_f32,"different_from_f64_rounded_bf16":self.different_bf16,"max_abs_to_f64":self.max_abs,"relative_l2":if self.norm>0.0{Some((self.sse/self.norm).sqrt())}else{None}})
    }
}
fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev mla-dot-reference".into());
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected identity-bound plan and fresh output directory".into());
    }
    let raw = fs::read(&args[1]).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    validate(&p)?;
    let packed = p.packed.read()?;
    let scales = p.scales.read()?;
    let zeros = p.zeros.read()?;
    let materialized = p.materialized.read()?;
    let pi = p.prefill_input.read()?;
    let di = p.decode_input.read()?;
    let po = p.prefill_output.read()?;
    let dout = p.decode_output.read()?;
    if packed.len() != mul(p.rows, p.packed_stride)?
        || scales.len() != mul(p.rows, p.scales_stride)?
        || zeros.len() != mul(p.rows, p.zeros_stride)?
        || materialized.len() != mul(mul(p.rows, p.cols)?, 2)?
        || pi.len() != mul(p.cols, 2)?
        || di.len() != pi.len()
        || po.len() != mul(p.rows, 2)?
        || dout.len() != mul(p.rows, if p.decode_output_dtype == "f32" { 4 } else { 2 })?
    {
        return Err("actual captured reference extent mismatch".into());
    }
    let pref_input = (0..p.cols)
        .map(|i| bf(&pi, i))
        .collect::<Result<Vec<_>, _>>()?;
    let dec_input = (0..p.cols)
        .map(|i| bf(&di, i))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pref_actual = ErrorStats::default();
    let mut pref_quant = ErrorStats::default();
    let mut dec_quant = ErrorStats::default();
    let mut materialized_diff = 0usize;
    let mut first_materialized_diff = None;
    let mut pref_quant_ref = Vec::with_capacity(mul(p.rows, 8)?);
    let mut dec_quant_ref = Vec::with_capacity(mul(p.rows, 8)?);
    let mut pref_bf_ref = Vec::with_capacity(mul(p.rows, 8)?);
    for row in 0..p.rows {
        let packed_row = &packed[row * p.packed_stride..(row + 1) * p.packed_stride];
        let mut ref_bf = 0.0f64;
        let mut ref_pref_q = 0.0f64;
        let mut ref_dec_q = 0.0f64;
        for col in 0..p.cols {
            let group = col / p.group_size;
            let scale = f32at(&scales, row * p.scales_stride + group * 4)?;
            let zero = f32at(&zeros, row * p.zeros_stride + group * 4)?;
            let weight = coefficient(quant(packed_row, col), zero, scale)? as f64;
            let actual_bf = bf(&materialized, row * p.cols + col)?;
            let bits = u16::from_le_bytes(
                materialized[2 * (row * p.cols + col)..2 * (row * p.cols + col) + 2]
                    .try_into()
                    .unwrap(),
            );
            if rne(weight)? != bits {
                materialized_diff += 1;
                if first_materialized_diff.is_none() {
                    first_materialized_diff = Some([row, col]);
                }
            }
            ref_bf = actual_bf.mul_add(pref_input[col], ref_bf);
            ref_pref_q = weight.mul_add(pref_input[col], ref_pref_q);
            ref_dec_q = weight.mul_add(dec_input[col], ref_dec_q);
        }
        let actual_pref = bf(&po, row)?;
        let actual_dec = if p.decode_output_dtype == "f32" {
            f32at(&dout, row * 4)? as f64
        } else {
            bf(&dout, row)?
        };
        pref_actual.add(actual_pref, ref_bf)?;
        pref_quant.add(actual_pref, ref_pref_q)?;
        dec_quant.add(actual_dec, ref_dec_q)?;
        pref_bf_ref.extend(ref_bf.to_le_bytes());
        pref_quant_ref.extend(ref_pref_q.to_le_bytes());
        dec_quant_ref.extend(ref_dec_q.to_le_bytes());
    }
    let out = PathBuf::from(&args[2]);
    fs::create_dir(&out).map_err(|e| e.to_string())?;
    for (name, bytes) in [
        ("prefill-materialized-reference.f64", pref_bf_ref),
        ("prefill-quant-f32-reference.f64", pref_quant_ref),
        ("decode-quant-f32-reference.f64", dec_quant_ref),
    ] {
        fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    }
    let report = json!({"schema":1,"plan_sha256":format!("{:x}",Sha256::digest(&raw)),"rows":p.rows,"cols":p.cols,"input_rows_exact":pi==di,"coefficient_reconstruction":"F32(q-zero), then F32 multiply scale; F64 dot with captured BF16 inputs; original coefficients and strides identity-bound","materialized_bf16_different_from_f32_coefficient_rne":materialized_diff,"first_materialized_difference":first_materialized_diff,"prefill_vs_actual_bf16_coefficients":pref_actual.report(),"prefill_vs_quant_f32_coefficients":pref_quant.report(),"decode_vs_quant_f32_coefficients":dec_quant.report(),"decode_output_dtype":p.decode_output_dtype,"scope":"Independent offline CPU F64 mathematical reductions of actual two coefficient/input contracts; original-model/Witness fidelity and serving correctness are not qualified. Each path uses its own actual input; only equal-input cases isolate projection representation/arithmetic."});
    fs::write(
        out.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_packed_quartet_and_coefficients() {
        let packed = [0x40, 0x00, 0xfe];
        assert_eq!(
            (0..4).map(|i| quant(&packed, i)).collect::<Vec<_>>(),
            [0, 1, 32, 63]
        );
        assert_eq!(
            (0..4)
                .map(|i| coefficient(quant(&packed, i), 32.0, 0.5).unwrap())
                .collect::<Vec<_>>(),
            [-16.0, -15.5, 0.0, 15.5]
        );
    }
    #[test]
    fn ties_round_even_with_both_signs() {
        assert_eq!(rne(1.00390625).unwrap(), 0x3f80);
        assert_eq!(rne(1.01171875).unwrap(), 0x3f82);
        assert_eq!(rne(-1.00390625).unwrap(), 0xbf80);
        assert_eq!(rne(-1.01171875).unwrap(), 0xbf82);
        assert!(rne(f64::INFINITY).is_err());
        assert!(coefficient(1, f32::NAN, 1.0).is_err());
    }
    #[test]
    fn mathematical_cancellation_retains_small_term() {
        let weights = [134217728.0f64, 1.0, -134217728.0];
        let result = weights.iter().fold(0.0f64, |s, &v| v.mul_add(1.0, s));
        assert_eq!(result, 1.0);
        assert_eq!(weights.iter().fold(0.0f32, |s, &v| s + v as f32), 0.0);
        assert_eq!(rne(result).unwrap(), 0x3f80);
    }
}
