use half::bf16;
use memmap2::Mmap;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

struct Layer<'a> {
    header: [u32; 7],
    rows: usize,
    columns: [Vec<&'a [u8]>; 4],
}

fn parse(raw: &[u8]) -> Result<Layer<'_>> {
    if raw.len() < 32 || &raw[..4] != b"KTC1" {
        return Err("invalid KTC1 header".into());
    }
    let mut header = [0; 7];
    for (index, value) in header.iter_mut().enumerate() {
        *value = u32::from_le_bytes(raw[4 + index * 4..8 + index * 4].try_into().unwrap());
    }
    // KTC1 version 2 carries the real router-weight column.
    if header[0] != 2 || header[6] != 0 || header[2..6].contains(&0) {
        return Err("unsupported KTC1 version or invalid dimensions".into());
    }
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or("capture dimension overflow");
    let strides = [
        mul(header[2] as usize, 2)?,
        mul(header[4] as usize, 4)?,
        mul(header[4] as usize, 4)?,
        mul(mul(header[4] as usize, header[3] as usize)?, 2)?,
    ];
    let mut columns: [Vec<&[u8]>; 4] = std::array::from_fn(|_| Vec::new());
    let mut offset = 32usize;
    let mut rows = 0usize;
    while offset < raw.len() {
        let end = offset.checked_add(4).ok_or("chunk offset overflow")?;
        let chunk_rows = u32::from_le_bytes(
            raw.get(offset..end)
                .ok_or("truncated chunk rows")?
                .try_into()
                .unwrap(),
        ) as usize;
        if chunk_rows == 0 {
            return Err("empty KTC1 chunk".into());
        }
        rows = rows.checked_add(chunk_rows).ok_or("row count overflow")?;
        offset = end;
        for index in 0..4 {
            let end = offset
                .checked_add(mul(chunk_rows, strides[index])?)
                .ok_or("chunk offset overflow")?;
            columns[index].push(raw.get(offset..end).ok_or("truncated chunk payload")?);
            offset = end;
        }
    }
    if rows == 0 {
        return Err("no KTC1 rows".into());
    }
    Ok(Layer {
        header,
        rows,
        columns,
    })
}

fn elements<'a>(column: &'a [&'a [u8]], width: usize) -> impl Iterator<Item = &'a [u8]> {
    column
        .iter()
        .flat_map(move |chunk| chunk.chunks_exact(width))
}

fn numeric(column: &[&[u8]], other: &[&[u8]], width: usize) -> Result<serde_json::Value> {
    let mut count = 0u64;
    let mut unequal = 0u64;
    let mut ref_sq = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let decode = |bytes: &[u8]| -> Result<f64> {
        let value = if width == 2 {
            bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f64()
        } else {
            f32::from_le_bytes(bytes.try_into().unwrap()) as f64
        };
        if !value.is_finite() {
            return Err("nonfinite captured value".into());
        }
        Ok(value)
    };
    for (a, b) in elements(column, width).zip(elements(other, width)) {
        let reference = decode(a)?;
        let actual = decode(b)?;
        let error = actual - reference;
        count += 1;
        unequal += u64::from(a != b);
        ref_sq += reference * reference;
        err_sq += error * error;
        max_abs = max_abs.max(error.abs());
    }
    Ok(
        json!({"elements":count,"bitwise_unequal":unequal,"max_abs":max_abs,
        "reference_l2":ref_sq.sqrt(),"error_l2":err_sq.sqrt(),
        "relative_l2":if ref_sq > 0.0 {Some((err_sq/ref_sq).sqrt())} else {None}}),
    )
}

fn compare(left: &[u8], right: &[u8]) -> Result<serde_json::Value> {
    let a = parse(left)?;
    let b = parse(right)?;
    if a.header != b.header || a.rows != b.rows {
        return Err("capture layer geometry/row count mismatch".into());
    }
    let experts = a.header[5] as usize;
    let mut counts = BTreeMap::<usize, u64>::new();
    let mut slot_mismatches = 0u64;
    for (left, right) in elements(&a.columns[1], 4).zip(elements(&b.columns[1], 4)) {
        let lhs = i32::from_le_bytes(left.try_into().unwrap());
        let rhs = i32::from_le_bytes(right.try_into().unwrap());
        if lhs < 0 || rhs < 0 || lhs as usize >= experts || rhs as usize >= experts {
            return Err("out-of-range routed expert".into());
        }
        *counts.entry(rhs as usize).or_default() += 1;
        slot_mismatches += u64::from(lhs != rhs);
    }
    for column in [&a.columns[2], &b.columns[2]] {
        for value in elements(column, 4) {
            if f32::from_le_bytes(value.try_into().unwrap()) < 0.0 {
                return Err("negative router weight".into());
            }
        }
    }
    Ok(
        json!({"layer":a.header[1],"rows":a.rows,"route_slot_mismatches":slot_mismatches,
        "expert_count":experts,"candidate_route_counts":counts,
        "route_count_min":if counts.len()==experts {counts.values().min().copied().unwrap_or(0)} else {0},
        "unobserved_experts":experts-counts.len(),
        "input":numeric(&a.columns[0],&b.columns[0],2)?,
        "weights":numeric(&a.columns[2],&b.columns[2],4)?,
        "down_by_slot":numeric(&a.columns[3],&b.columns[3],2)?}),
    )
}

fn map(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // Diagnostic inputs must be closed, published shards; use the supported
    // inspector to verify their enclosing transactions before this comparison.
    unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let result = (|| {
        if args.len() != 2 {
            return Err("usage: capture-layer-compare <reference.ktc> <candidate.ktc>".into());
        }
        compare(&map(Path::new(&args[0]))?, &map(Path::new(&args[1]))?)
    })();
    match result {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("ERROR: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod compare_tests {
    use super::*;
    fn fixture(chunks: &[u32]) -> Vec<u8> {
        let mut bytes = b"KTC1".to_vec();
        for field in [2u32, 3, 1, 1, 1, 2, 0] {
            bytes.extend(field.to_le_bytes());
        }
        for rows in chunks {
            bytes.extend(rows.to_le_bytes());
            for (value, width) in [(0x3f80u32, 2usize), (1, 4), (0x3f000000, 4), (0x4000, 2)] {
                for _ in 0..*rows {
                    bytes.extend(&value.to_le_bytes()[..width]);
                }
            }
        }
        bytes
    }
    #[test]
    fn chunk_independence_and_invalid_inputs() {
        let reference = fixture(&[3]);
        let candidate = fixture(&[1, 2]);
        let result = compare(&reference, &candidate).unwrap();
        assert_eq!(result["input"]["bitwise_unequal"], 0);
        assert_eq!(result["route_slot_mismatches"], 0);
        assert_eq!(result["candidate_route_counts"], json!({"1":3}));
        assert_eq!(result["unobserved_experts"], 1);
        assert!(compare(&reference, &fixture(&[2])).is_err());
        assert!(parse(&candidate[..candidate.len() - 1]).is_err());
        let mut obsolete = reference.clone();
        obsolete[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse(&obsolete).is_err());
        let mut invalid = reference.clone();
        invalid[42..46].copy_from_slice(&2i32.to_le_bytes());
        assert!(compare(&reference, &invalid).is_err());
        let mut changed = reference.clone();
        changed[36..38].copy_from_slice(&0x4000u16.to_le_bytes());
        assert_eq!(
            compare(&reference, &changed).unwrap()["input"]["max_abs"],
            1.0
        );
    }
}
