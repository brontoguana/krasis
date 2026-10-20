//! Explicit actual MLA observations; readbacks only, no device allocation/math change.
use crate::{matched_probe::Target, ordered_projection::Layout};
use cudarc::driver::sys;
fn name_ok(name: &str) -> Result<(), String> {
    if name.is_empty()
        || matches!(name, "." | "..")
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err("invalid MLA observation file component".into());
    }
    Ok(())
}
pub(crate) fn row(
    t: &Target,
    name: &str,
    ptr: u64,
    row: usize,
    width: usize,
    element_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    name_ok(name)?;
    if ptr == 0 || width == 0 || !matches!(element_bytes, 2 | 4) || ptr % element_bytes as u64 != 0
    {
        return Err("invalid actual MLA row".into());
    }
    let bytes = width
        .checked_mul(element_bytes)
        .ok_or("MLA row byte overflow")?;
    let at = ptr
        .checked_add(row.checked_mul(bytes).ok_or("MLA row offset overflow")? as u64)
        .ok_or("MLA row address overflow")?;
    at.checked_add(bytes as u64).ok_or("MLA row end overflow")?;
    t.device(name, at, bytes, stream)
}
pub(crate) fn block(
    t: &Target,
    name: &str,
    ptr: u64,
    elements: usize,
    element_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    name_ok(name)?;
    if ptr == 0 || elements == 0 || element_bytes == 0 {
        return Err("invalid actual MLA block".into());
    }
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or("MLA block byte overflow")?;
    ptr.checked_add(bytes as u64)
        .ok_or("MLA block address overflow")?;
    t.device(name, ptr, bytes, stream)
}
pub(crate) fn weights(
    t: &Target,
    name: &str,
    w: &Layout,
    descriptor: serde_json::Value,
    stream: sys::CUstream,
) -> Result<(), String> {
    name_ok(name)?;
    w.validate()?;
    block(
        t,
        &format!("{name}.packed"),
        w.packed,
        w.rows,
        w.packed_stride,
        stream,
    )?;
    block(
        t,
        &format!("{name}.scales.f32"),
        w.scales,
        w.rows,
        w.scales_stride,
        stream,
    )?;
    block(
        t,
        &format!("{name}.zeros.f32"),
        w.zeros,
        w.rows,
        w.zeros_stride,
        stream,
    )?;
    t.write(&format!("{name}.json"),serde_json::to_vec_pretty(&serde_json::json!({"format":"hqq6_row_major_f32_parameters","rows":w.rows,"cols":w.cols,"group_size":w.group,"packed_stride":w.packed_stride,"scales_stride":w.scales_stride,"zeros_stride":w.zeros_stride,"valid_coefficient_bytes_per_row":w.cols/w.group*4,"parameter_padding":"raw preserved, compare only logical groups when interpreting coefficients","actual_descriptor":descriptor})).map_err(|e|e.to_string())?.as_slice())
}
pub(crate) fn cache(
    t: &Target,
    prefix: usize,
    ccd: usize,
    rope: usize,
    format: u32,
    ckv: u64,
    kpe: u64,
    stream: sys::CUstream,
) -> Result<(), String> {
    if prefix == 0 || ccd == 0 || ccd % 16 != 0 || rope % 16 != 0 {
        return Err("invalid observed MLA cache geometry".into());
    }
    let block_bytes = crate::gpu_prefill::native_mla_block_bytes(format)?;
    let ckv_bytes = prefix
        .checked_mul(ccd / 16)
        .and_then(|n| n.checked_mul(block_bytes))
        .ok_or("MLA cKV prefix overflow")?;
    block(t, "ckv-prefix.raw", ckv, ckv_bytes, 1, stream)?;
    let kpe_bytes = prefix
        .checked_mul(rope / 16)
        .and_then(|n| n.checked_mul(block_bytes))
        .ok_or("MLA KPE prefix overflow")?;
    if rope > 0 {
        block(t, "kpe-prefix.raw", kpe, kpe_bytes, 1, stream)?;
    }
    t.write("cache.json",serde_json::to_vec_pretty(&serde_json::json!({"prefix":prefix,"ckv_dim":ccd,"rope_dim":rope,"kv_format":format,"block_elements":16,"block_bytes":block_bytes,"ckv_bytes":ckv_bytes,"kpe_bytes":kpe_bytes,"zero_rope":rope==0,"kpe_payload_present":rope>0,"scope":"actual bytes through the common current position only; raw padding is not by itself causal evidence"})).map_err(|e|e.to_string())?.as_slice())
}
pub(crate) fn head_rows(
    t: &Target,
    name: &str,
    ptr: u64,
    row: usize,
    tokens: usize,
    heads: usize,
    width: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    name_ok(name)?;
    if row >= tokens || heads == 0 || width == 0 {
        return Err("invalid observed MLA head-major row".into());
    }
    for h in 0..heads {
        let logical = h
            .checked_mul(tokens)
            .and_then(|v| v.checked_add(row))
            .ok_or("MLA head-major offset overflow")?;
        self::row(
            t,
            &format!("{name}-head{h}.bf16"),
            ptr,
            logical,
            width,
            2,
            stream,
        )?;
    }
    Ok(())
}

/// Translate only an admitted global observation row into its actual chunk view.
/// The directory keeps the original global target identity; metadata records both.
pub(crate) fn chunk_row(global: usize, start: usize, rows: usize) -> Result<usize, String> {
    let end = start
        .checked_add(rows)
        .ok_or("MLA observation chunk overflow")?;
    if rows == 0 || global < start || global >= end {
        return Err("MLA observation global row outside actual chunk".into());
    }
    Ok(global - start)
}
pub(crate) fn chunk_target(
    target: Option<Target>,
    start: usize,
    rows: usize,
) -> Result<Option<Target>, String> {
    let Some(t) = target else { return Ok(None) };
    let end = start
        .checked_add(rows)
        .ok_or("MLA observation chunk overflow")?;
    if rows == 0 {
        return Err("MLA observation empty chunk".into());
    }
    // A valid request-global target belongs to exactly one real chunk. Other
    // chunks do not observe it; the collector requires the resulting payload.
    if t.row < start || t.row >= end {
        return Ok(None);
    }
    Ok(Some(Target {
        row: chunk_row(t.row, start, rows)?,
        directory: t.directory,
    }))
}
pub(crate) fn prefill_target(
    invocation: Option<&crate::matched_prefill_capture::Invocation>,
    operation: &str,
    layer: usize,
    start: usize,
    rows: usize,
) -> Result<Option<Target>, String> {
    let Some(invocation) = invocation else {
        return Ok(None);
    };
    let target = crate::matched_prefill_capture::target(
        Some(invocation),
        operation,
        layer,
        invocation.logical_tokens(),
        None,
    )?;
    chunk_target(target, start, rows)
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    #[test]
    fn actual_whole_and_partial_chunk_rows() {
        assert_eq!(chunk_row(10500, 0, 10564).unwrap(), 10500);
        assert_eq!(chunk_row(10500, 9755, 809).unwrap(), 745);
        assert_eq!(chunk_row(10499, 9755, 745).unwrap(), 744);
        assert!(chunk_row(9754, 9755, 809).is_err());
        assert!(chunk_row(10564, 9755, 809).is_err());
        assert!(chunk_row(0, 0, 0).is_err());
        assert!(chunk_row(usize::MAX, usize::MAX, 1).is_err());
    }
}

/// Logical live score rows, including the real empty pooled prefix case.
pub(crate) fn causal_score_count(raw_prefix: usize, divisor: usize) -> Result<usize, String> {
    if divisor == 0 {
        return Err("observed score divisor is zero".into());
    }
    Ok(raw_prefix / divisor)
}
pub(crate) fn logical_block(
    t: &Target,
    name: &str,
    ptr: u64,
    elements: usize,
    bytes: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    if elements == 0 {
        name_ok(name)?;
        t.write(name, &[])
    } else {
        block(t, name, ptr, elements, bytes, stream)
    }
}
#[cfg(test)]
mod pooled_tests {
    use super::*;
    #[test]
    fn actual_raw_and_pooled_counts() {
        assert_eq!(causal_score_count(10501, 1).unwrap(), 10501);
        assert_eq!(causal_score_count(10501, 4).unwrap(), 2625);
        assert_eq!(causal_score_count(10500, 4).unwrap(), 2625);
        assert_eq!(causal_score_count(10504, 4).unwrap(), 2626);
        assert_eq!(causal_score_count(1, 4).unwrap(), 0);
        assert!(causal_score_count(10501, 0).is_err());
    }
}

#[cfg(test)]
mod request_chunk_tests {
    use super::*;
    #[test]
    fn request_target_is_observed_only_in_its_actual_chunk() {
        let make = || {
            Some(Target {
                row: 10500,
                directory: std::path::PathBuf::from("unused-global-target"),
            })
        };
        assert!(chunk_target(make(), 0, 9755).unwrap().is_none());
        let actual = chunk_target(make(), 9755, 809).unwrap().unwrap();
        assert_eq!(actual.row, 745);
        assert_eq!(
            actual.directory,
            std::path::PathBuf::from("unused-global-target")
        );
        assert!(chunk_target(make(), 10501, 63).unwrap().is_none());
        assert!(chunk_target(make(), usize::MAX, 2).is_err());
    }
}
