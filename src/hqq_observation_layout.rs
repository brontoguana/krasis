//! Checked actual execution payload layouts for bounded read-only diagnostics.
use serde_json::{json, Value};
pub(crate) struct Payloads {
    pub packed: usize,
    pub scales: usize,
    pub zeros: usize,
    pub scales_file: &'static str,
    pub zeros_file: &'static str,
    pub metadata: Value,
}
pub(crate) fn describe(
    rows: usize,
    cols: usize,
    group: usize,
    bits: u8,
    layout: &str,
    packed_dtype: &str,
    scales_dtype: &str,
    zeros_dtype: &str,
    tensor_bytes: usize,
) -> Result<Payloads, String> {
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| "observed HQQ payload overflow".to_string())
    };
    if rows == 0 || cols == 0 || group == 0 {
        return Err("observed HQQ execution geometry is empty".into());
    }
    let groups = cols.div_ceil(group);
    let entries = mul(rows, groups)?;
    let row_major = match bits {
        4 => "row_major_axis1_grouped_uint4_packed",
        6 => "row_major_axis1_grouped_uint6_packed",
        8 => "row_major_axis1_grouped_uint8",
        _ => return Err("observed HQQ execution bit width unsupported".into()),
    };
    let (packed, scales, zeros, scales_file, zeros_file, segments) = if layout == row_major {
        if packed_dtype != "uint8" || scales_dtype != "float32" || zeros_dtype != "float32" {
            return Err("observed row-major HQQ payload dtype mismatch".into());
        }
        let padded = mul(groups, group)?;
        let stride = match bits {
            4 => padded.div_ceil(2),
            6 => mul(padded.div_ceil(4), 3)?,
            8 => padded,
            _ => unreachable!(),
        };
        let packed = mul(rows, stride)?;
        let scales = mul(entries, 4)?;
        (
            packed,
            scales,
            scales,
            "scales.f32",
            "zeros.f32",
            json!({"packing":"canonical row-major unsigned groups","packed":[{"offset":0,"bytes":packed,"dtype":"uint8"}],"scales":[{"offset":0,"bytes":scales,"dtype":"float32"}],"zeros":[{"offset":0,"bytes":scales,"dtype":"float32"}]}),
        )
    } else if bits == 4 && layout == "marlin_u4_float_zp_twoscale_intercept_axis1_grouped_uint4" {
        if packed_dtype != "uint32"
            || scales_dtype != "bfloat16_base_delta"
            || zeros_dtype != "bfloat16_plus_float32_twoscale_intercept_correction"
            || rows % 64 != 0
            || cols % 16 != 0
            || cols % group != 0
        {
            return Err("observed native Marlin u4 execution dtype/tiling mismatch".into());
        }
        let packed = mul(rows, cols)? / 2;
        let bf16 = mul(entries, 2)?;
        let fp32 = mul(entries, 4)?;
        let scales = bf16.checked_mul(2).ok_or("observed split scale overflow")?;
        let zeros = bf16
            .checked_add(fp32)
            .ok_or("observed intercept payload overflow")?;
        (
            packed,
            scales,
            zeros,
            "scales.raw.bin",
            "zeros.raw.bin",
            json!({"packing":"actual native Marlin u4 repacked tiles, production kernel permutation","packed":[{"offset":0,"bytes":packed,"dtype":"uint32"}],"scales":[{"offset":0,"bytes":bf16,"dtype":"bfloat16","role":"base"},{"offset":bf16,"bytes":bf16,"dtype":"bfloat16","role":"delta"}],"zeros":[{"offset":0,"bytes":bf16,"dtype":"bfloat16","role":"zero"},{"offset":bf16,"bytes":fp32,"dtype":"float32","role":"twoscale_intercept_correction"}]}),
        )
    } else {
        return Err(format!("bounded HQQ projection observation does not implement actual execution layout {layout} ({packed_dtype}/{scales_dtype}/{zeros_dtype}); no source-format substitution"));
    };
    let total = packed
        .checked_add(scales)
        .and_then(|x| x.checked_add(zeros))
        .ok_or("observed HQQ bundle overflow")?;
    if total > tensor_bytes {
        return Err(format!("observed HQQ execution payload {total} exceeds actual descriptor extent {tensor_bytes}"));
    }
    Ok(Payloads {
        packed,
        scales,
        zeros,
        scales_file,
        zeros_file,
        metadata: json!({"layout":layout,"nbits":bits,"rows":rows,"cols":cols,"group_size":group,"total_read_bytes":total,"descriptor_tensor_bytes":tensor_bytes,"packed_file":"packed.bin","scales_file":scales_file,"zeros_file":zeros_file,"segments":segments,"source":"actual loaded execution descriptor; canonical cache format is not assumed"}),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_marlin_counterexample_has_typed_split_planes() {
        let p = describe(
            12288,
            2048,
            128,
            4,
            "marlin_u4_float_zp_twoscale_intercept_axis1_grouped_uint4",
            "uint32",
            "bfloat16_base_delta",
            "bfloat16_plus_float32_twoscale_intercept_correction",
            14548992,
        )
        .unwrap();
        assert_eq!((p.packed, p.scales, p.zeros), (12582912, 786432, 1179648));
        assert_eq!(p.metadata["segments"]["zeros"][1]["offset"], 393216);
        assert!(describe(
            12288,
            2048,
            128,
            4,
            "marlin_u4_float_zp_twoscale_intercept_axis1_grouped_uint4",
            "uint8",
            "float32",
            "float32",
            14548992
        )
        .is_err());
    }
    #[test]
    fn canonical_odd_geometry_and_bad_extents_are_checked() {
        let p = describe(
            3,
            5,
            4,
            4,
            "row_major_axis1_grouped_uint4_packed",
            "uint8",
            "float32",
            "float32",
            60,
        )
        .unwrap();
        assert_eq!((p.packed, p.scales, p.zeros), (12, 24, 24));
        assert!(describe(
            3,
            5,
            4,
            4,
            "row_major_axis1_grouped_uint4_packed",
            "uint8",
            "float32",
            "float32",
            59
        )
        .is_err());
        assert!(describe(
            usize::MAX,
            5,
            4,
            4,
            "row_major_axis1_grouped_uint4_packed",
            "uint8",
            "float32",
            "float32",
            usize::MAX
        )
        .is_err());
        assert!(describe(
            3,
            5,
            0,
            4,
            "row_major_axis1_grouped_uint4_packed",
            "uint8",
            "float32",
            "float32",
            60
        )
        .is_err());
    }
}
