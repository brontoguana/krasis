//! Selected dense FFN readbacks; no diagnostic device allocations or math changes.
use crate::matched_probe::Target;
use cudarc::driver::sys;

pub(crate) fn row(target: &Target, name: &str, ptr: u64, row: usize, width: usize, stream: sys::CUstream) -> Result<(), String> {
    if ptr == 0 || width == 0 { return Err("dense probe requires loaded nonempty tensor".into()); }
    let bytes=width.checked_mul(2).ok_or("dense row bytes overflow")?;
    let offset=row.checked_mul(bytes).ok_or("dense row offset overflow")?;
    target.device(name,ptr.checked_add(offset as u64).ok_or("dense row address overflow")?,bytes,stream)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn marlin(target: &Target, name: &str, packed: u64, scales: u64, rows: usize, cols: usize, groups: usize, group: usize, bits: u8, stream: sys::CUstream) -> Result<(), String> {
    if !matches!(bits,4|8) || packed==0 || scales==0 || rows==0 || cols==0 || group==0 || groups!=cols.div_ceil(group) {
        return Err("dense probe requires actual complete Marlin INT4/INT8 geometry".into());
    }
    let packed_bits=rows.checked_mul(cols).and_then(|n|n.checked_mul(bits as usize)).ok_or("dense packed bytes overflow")?;
    if packed_bits%8!=0 {return Err("dense packed extent is not byte aligned".into());}
    let scale_bytes=rows.checked_mul(groups).and_then(|n|n.checked_mul(2)).ok_or("dense scale bytes overflow")?;
    target.device(&format!("{name}.packed"),packed,packed_bits/8,stream)?;
    target.device(&format!("{name}.scales.bf16"),scales,scale_bytes,stream)?;
    target.write(&format!("{name}.json"),serde_json::to_vec_pretty(&serde_json::json!({"format":"marlin","rows":rows,"cols":cols,"bits":bits,"group_size":group,"groups":groups,"packed_bytes":packed_bits/8,"scale_bytes":scale_bytes,"packed_ptr":packed,"scales_ptr":scales})).map_err(|e|e.to_string())?.as_slice())
}

pub(crate) fn bf16(target: &Target, name: &str, ptr: u64, rows: usize, cols: usize, stream: sys::CUstream) -> Result<(), String> {
    if ptr==0 || rows==0 || cols==0 {return Err("dense BF16 diagnostic requires loaded nonempty weights".into());}
    let bytes=rows.checked_mul(cols).and_then(|n|n.checked_mul(2)).ok_or("dense BF16 weight extent overflow")?;
    target.device(&format!("{name}.weights.bf16"),ptr,bytes,stream)?;
    target.write(&format!("{name}.json"),serde_json::to_vec_pretty(&serde_json::json!({"format":"bf16","rows":rows,"cols":cols,"weight_bytes":bytes,"weight_ptr":ptr})).map_err(|e|e.to_string())?.as_slice())
}
