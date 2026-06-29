/// Phase 0 test: Verify Marlin INT4/INT8 shape compatibility with attention projections.
///
/// This is a standalone binary test (avoids PyO3 linking issues).
/// Run with: cargo test --release --test test_marlin_attn_shapes -- --nocapture
use std::path::Path;

// We can't import from krasis directly due to PyO3, so we duplicate the
// relevant pure-Rust functions here. These are exact copies from src/weights/marlin.rs.

const DEFAULT_GROUP_SIZE: usize = 128;
const PACK_FACTOR: usize = 8;
const PACK_FACTOR_INT8: usize = 4;
const MARLIN_TILE: usize = 16;

#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

struct QuantizedInt4 {
    packed: Vec<u32>,
    scales: Vec<u16>,
    rows: usize,
    cols: usize,
    group_size: usize,
}

struct QuantizedInt8 {
    data: Vec<i8>,
    scales: Vec<u16>,
    rows: usize,
    cols: usize,
    group_size: usize,
}

fn quantize_int4(
    weight_bf16: &[u16],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> QuantizedInt4 {
    assert_eq!(weight_bf16.len(), rows * cols);
    assert!(cols % group_size == 0);
    assert!(cols % PACK_FACTOR == 0);

    let num_groups_per_row = cols / group_size;
    let packed_cols = cols / PACK_FACTOR;
    let mut scales = vec![0u16; rows * num_groups_per_row];
    let mut packed = vec![0u32; rows * packed_cols];

    for row in 0..rows {
        let row_offset = row * cols;
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let mut amax: f32 = 0.0;
            for i in 0..group_size {
                amax = amax.max(bf16_to_f32(weight_bf16[group_start + i]).abs());
            }
            let scale = if amax == 0.0 { 1.0 } else { amax / 7.0 };
            scales[row * num_groups_per_row + g] = f32_to_bf16(scale);
        }
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let scale = bf16_to_f32(scales[row * num_groups_per_row + g]);
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            for i in (0..group_size).step_by(PACK_FACTOR) {
                let mut word: u32 = 0;
                for j in 0..PACK_FACTOR {
                    let val = bf16_to_f32(weight_bf16[group_start + i + j]);
                    let q = (val * inv_scale).round().clamp(-8.0, 7.0) as i8;
                    let u4 = (q + 8) as u8 & 0xF;
                    word |= (u4 as u32) << (j * 4);
                }
                packed[row * packed_cols + (g * group_size + i) / PACK_FACTOR] = word;
            }
        }
    }
    QuantizedInt4 {
        packed,
        scales,
        rows,
        cols,
        group_size,
    }
}

fn dequantize_int4(q: &QuantizedInt4) -> Vec<f32> {
    let num_groups_per_row = q.cols / q.group_size;
    let packed_cols = q.cols / PACK_FACTOR;
    let mut output = vec![0.0f32; q.rows * q.cols];
    for row in 0..q.rows {
        for g in 0..num_groups_per_row {
            let scale = bf16_to_f32(q.scales[row * num_groups_per_row + g]);
            for i in (0..q.group_size).step_by(PACK_FACTOR) {
                let col_in_row = g * q.group_size + i;
                let word = q.packed[row * packed_cols + col_in_row / PACK_FACTOR];
                for j in 0..PACK_FACTOR {
                    let u4 = ((word >> (j * 4)) & 0xF) as i8;
                    output[row * q.cols + col_in_row + j] = (u4 - 8) as f32 * scale;
                }
            }
        }
    }
    output
}

fn quantize_int8(
    weight_bf16: &[u16],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> QuantizedInt8 {
    assert_eq!(weight_bf16.len(), rows * cols);
    assert!(cols % group_size == 0);

    let num_groups_per_row = cols / group_size;
    let mut scales = vec![0u16; rows * num_groups_per_row];
    let mut data = vec![0i8; rows * cols];

    for row in 0..rows {
        let row_offset = row * cols;
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let mut amax: f32 = 0.0;
            for i in 0..group_size {
                amax = amax.max(bf16_to_f32(weight_bf16[group_start + i]).abs());
            }
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales[row * num_groups_per_row + g] = f32_to_bf16(scale);
        }
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let scale = bf16_to_f32(scales[row * num_groups_per_row + g]);
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            for i in 0..group_size {
                let val = bf16_to_f32(weight_bf16[group_start + i]);
                data[group_start + i] = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }
    QuantizedInt8 {
        data,
        scales,
        rows,
        cols,
        group_size,
    }
}

fn dequantize_int8(q: &QuantizedInt8) -> Vec<f32> {
    let num_groups_per_row = q.cols / q.group_size;
    let mut output = vec![0.0f32; q.rows * q.cols];
    for row in 0..q.rows {
        for g in 0..num_groups_per_row {
            let scale = bf16_to_f32(q.scales[row * num_groups_per_row + g]);
            let group_start = row * q.cols + g * q.group_size;
            for i in 0..q.group_size {
                output[group_start + i] = q.data[group_start + i] as f32 * scale;
            }
        }
    }
    output
}

fn generate_weight_perm_int4() -> [usize; 1024] {
    let mut perm = [0usize; 1024];
    let mut idx = 0;
    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = [0usize; 8];
        let mut p1_idx = 0;
        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1[p1_idx] = 16 * row + col + 8 * block;
                p1_idx += 1;
            }
        }
        for j in 0..4 {
            for &p in &perm1 {
                perm[idx] = p + 256 * j;
                idx += 1;
            }
        }
    }
    let interleave = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut result = [0usize; 1024];
    for group in 0..(1024 / 8) {
        for (dest, &src) in interleave.iter().enumerate() {
            result[group * 8 + dest] = perm[group * 8 + src];
        }
    }
    result
}

fn generate_weight_perm_int8() -> [usize; 1024] {
    let mut perm = [0usize; 1024];
    let mut idx = 0;
    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = [0usize; 8];
        let mut p1_idx = 0;
        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1[p1_idx] = 16 * row + col + 8 * block;
                p1_idx += 1;
            }
        }
        for j in 0..4 {
            for &p in &perm1 {
                perm[idx] = p + 256 * j;
                idx += 1;
            }
        }
    }
    let interleave = [0, 2, 1, 3];
    let mut result = [0usize; 1024];
    for group in 0..(1024 / 4) {
        for (dest, &src) in interleave.iter().enumerate() {
            result[group * 4 + dest] = perm[group * 4 + src];
        }
    }
    result
}

fn generate_scale_perms() -> ([usize; 64], [usize; 32]) {
    let mut scale_perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            scale_perm[i * 8 + j] = i + 8 * j;
        }
    }
    let offsets = [0, 1, 8, 9, 16, 17, 24, 25];
    let mut scale_perm_single = [0usize; 32];
    for i in 0..4 {
        for (j, &off) in offsets.iter().enumerate() {
            scale_perm_single[i * 8 + j] = 2 * i + off;
        }
    }
    (scale_perm, scale_perm_single)
}

struct MarlinRepacked {
    packed: Vec<u32>,
    scales: Vec<u16>,
    k: usize,
    n: usize,
    group_size: usize,
}

fn marlin_repack_int4(q: &QuantizedInt4) -> MarlinRepacked {
    let n = q.rows;
    let k = q.cols;
    let group_size = q.group_size;
    assert!(
        k % MARLIN_TILE == 0,
        "K ({k}) must be divisible by {MARLIN_TILE}"
    );
    assert!(n % 64 == 0, "N ({n}) must be divisible by 64");

    let packed_k = k / PACK_FACTOR;
    let mut unpacked = vec![0u8; n * k];
    for row in 0..n {
        for col_pack in 0..packed_k {
            let word = q.packed[row * packed_k + col_pack];
            for j in 0..PACK_FACTOR {
                unpacked[row * k + col_pack * PACK_FACTOR + j] =
                    ((word >> (j as u32 * 4)) & 0xF) as u8;
            }
        }
    }

    let k_tiles = k / MARLIN_TILE;
    let n_tiles = n / MARLIN_TILE;
    let row_len = n * MARLIN_TILE;
    let mut permuted = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for nt in 0..n_tiles {
            for tk in 0..MARLIN_TILE {
                for tn in 0..MARLIN_TILE {
                    let src_k = kt * MARLIN_TILE + tk;
                    let src_n = nt * MARLIN_TILE + tn;
                    let dst_col = nt * MARLIN_TILE * MARLIN_TILE + tk * MARLIN_TILE + tn;
                    permuted[kt * row_len + dst_col] = unpacked[src_n * k + src_k];
                }
            }
        }
    }

    let perm = generate_weight_perm_int4();
    let num_chunks = row_len / 1024;
    let mut perm_applied = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for chunk in 0..num_chunks {
            let base = kt * row_len + chunk * 1024;
            for i in 0..1024 {
                perm_applied[base + i] = permuted[base + perm[i]];
            }
        }
    }

    let out_cols = row_len / PACK_FACTOR;
    let mut out_packed = vec![0u32; k_tiles * out_cols];
    for row in 0..k_tiles {
        for col in 0..out_cols {
            let mut word: u32 = 0;
            for i in 0..PACK_FACTOR {
                word |=
                    (perm_applied[row * row_len + col * PACK_FACTOR + i] as u32) << (i as u32 * 4);
            }
            out_packed[row * out_cols + col] = word;
        }
    }

    let num_groups_k = k / group_size;
    let mut scales_transposed = vec![0u16; num_groups_k * n];
    for row in 0..n {
        for g in 0..num_groups_k {
            scales_transposed[g * n + row] = q.scales[row * num_groups_k + g];
        }
    }

    let (scale_perm, scale_perm_single) = generate_scale_perms();
    let is_grouped = group_size < k;
    let sperm: &[usize] = if is_grouped {
        &scale_perm
    } else {
        &scale_perm_single
    };
    let perm_len = sperm.len();
    let total_scale_vals = num_groups_k * n;
    let num_scale_chunks = total_scale_vals / perm_len;
    let mut scales_permuted = vec![0u16; total_scale_vals];
    for chunk in 0..num_scale_chunks {
        let base = chunk * perm_len;
        for i in 0..perm_len {
            scales_permuted[base + i] = scales_transposed[base + sperm[i]];
        }
    }

    MarlinRepacked {
        packed: out_packed,
        scales: scales_permuted,
        k,
        n,
        group_size,
    }
}

fn dequantize_marlin_int4(m: &MarlinRepacked) -> Vec<f32> {
    let k = m.k;
    let n = m.n;
    let group_size = m.group_size;
    let k_tiles = k / MARLIN_TILE;
    let row_len = n * MARLIN_TILE;
    let out_cols = row_len / PACK_FACTOR;
    let num_groups_k = k / group_size;

    let mut perm_applied = vec![0u8; k_tiles * row_len];
    for row in 0..k_tiles {
        for col in 0..out_cols {
            let word = m.packed[row * out_cols + col];
            for i in 0..PACK_FACTOR {
                perm_applied[row * row_len + col * PACK_FACTOR + i] =
                    ((word >> (i as u32 * 4)) & 0xF) as u8;
            }
        }
    }

    let perm = generate_weight_perm_int4();
    let num_chunks = row_len / 1024;
    let n_tiles = n / MARLIN_TILE;
    let mut permuted = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for chunk in 0..num_chunks {
            let base = kt * row_len + chunk * 1024;
            for i in 0..1024 {
                permuted[base + perm[i]] = perm_applied[base + i];
            }
        }
    }

    let mut transposed = vec![0u8; k * n];
    for kt in 0..k_tiles {
        for nt in 0..n_tiles {
            for tk in 0..MARLIN_TILE {
                for tn in 0..MARLIN_TILE {
                    let src_k = kt * MARLIN_TILE + tk;
                    let src_n = nt * MARLIN_TILE + tn;
                    transposed[src_k * n + src_n] = permuted
                        [kt * row_len + nt * MARLIN_TILE * MARLIN_TILE + tk * MARLIN_TILE + tn];
                }
            }
        }
    }

    let (scale_perm, scale_perm_single) = generate_scale_perms();
    let is_grouped = group_size < k;
    let sperm: &[usize] = if is_grouped {
        &scale_perm
    } else {
        &scale_perm_single
    };
    let perm_len = sperm.len();
    let total_scale_vals = num_groups_k * n;
    let num_scale_chunks = total_scale_vals / perm_len;
    let mut scales_transposed = vec![0u16; total_scale_vals];
    for chunk in 0..num_scale_chunks {
        let base = chunk * perm_len;
        for i in 0..perm_len {
            scales_transposed[base + sperm[i]] = m.scales[base + i];
        }
    }

    let mut output = vec![0.0f32; n * k];
    for ki in 0..k {
        for ni in 0..n {
            let u4 = transposed[ki * n + ni];
            let q_val = (u4 as i8) - 8;
            let scale = bf16_to_f32(scales_transposed[(ki / group_size) * n + ni]);
            output[ni * k + ki] = q_val as f32 * scale;
        }
    }
    output
}

fn marlin_repack_int8(q: &QuantizedInt8) -> MarlinRepacked {
    let n = q.rows;
    let k = q.cols;
    let group_size = q.group_size;
    assert!(
        k % MARLIN_TILE == 0,
        "K ({k}) must be divisible by {MARLIN_TILE}"
    );
    assert!(n % 64 == 0, "N ({n}) must be divisible by 64");

    let mut unsigned = vec![0u8; n * k];
    for i in 0..q.data.len() {
        unsigned[i] = (q.data[i] as i16 + 128) as u8;
    }

    let k_tiles = k / MARLIN_TILE;
    let n_tiles = n / MARLIN_TILE;
    let row_len = n * MARLIN_TILE;
    let mut permuted = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for nt in 0..n_tiles {
            for tk in 0..MARLIN_TILE {
                for tn in 0..MARLIN_TILE {
                    let dst_col = nt * MARLIN_TILE * MARLIN_TILE + tk * MARLIN_TILE + tn;
                    permuted[kt * row_len + dst_col] =
                        unsigned[(nt * MARLIN_TILE + tn) * k + kt * MARLIN_TILE + tk];
                }
            }
        }
    }

    let perm = generate_weight_perm_int8();
    let num_chunks = row_len / 1024;
    let mut perm_applied = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for chunk in 0..num_chunks {
            let base = kt * row_len + chunk * 1024;
            for i in 0..1024 {
                perm_applied[base + i] = permuted[base + perm[i]];
            }
        }
    }

    let out_cols = row_len / PACK_FACTOR_INT8;
    let mut out_packed = vec![0u32; k_tiles * out_cols];
    for row in 0..k_tiles {
        for col in 0..out_cols {
            let mut word: u32 = 0;
            for i in 0..PACK_FACTOR_INT8 {
                word |= (perm_applied[row * row_len + col * PACK_FACTOR_INT8 + i] as u32)
                    << (i as u32 * 8);
            }
            out_packed[row * out_cols + col] = word;
        }
    }

    let num_groups_k = k / group_size;
    let mut scales_transposed = vec![0u16; num_groups_k * n];
    for row in 0..n {
        for g in 0..num_groups_k {
            scales_transposed[g * n + row] = q.scales[row * num_groups_k + g];
        }
    }

    let (scale_perm, scale_perm_single) = generate_scale_perms();
    let is_grouped = group_size < k;
    let sperm: &[usize] = if is_grouped {
        &scale_perm
    } else {
        &scale_perm_single
    };
    let perm_len = sperm.len();
    let total_scale_vals = num_groups_k * n;
    let num_scale_chunks = total_scale_vals / perm_len;
    let mut scales_permuted = vec![0u16; total_scale_vals];
    for chunk in 0..num_scale_chunks {
        let base = chunk * perm_len;
        for i in 0..perm_len {
            scales_permuted[base + i] = scales_transposed[base + sperm[i]];
        }
    }

    MarlinRepacked {
        packed: out_packed,
        scales: scales_permuted,
        k,
        n,
        group_size,
    }
}

fn dequantize_marlin_int8(m: &MarlinRepacked) -> Vec<f32> {
    let k = m.k;
    let n = m.n;
    let group_size = m.group_size;
    let k_tiles = k / MARLIN_TILE;
    let row_len = n * MARLIN_TILE;
    let out_cols = row_len / PACK_FACTOR_INT8;
    let num_groups_k = k / group_size;

    let mut perm_applied = vec![0u8; k_tiles * row_len];
    for row in 0..k_tiles {
        for col in 0..out_cols {
            let word = m.packed[row * out_cols + col];
            for i in 0..PACK_FACTOR_INT8 {
                perm_applied[row * row_len + col * PACK_FACTOR_INT8 + i] =
                    ((word >> (i as u32 * 8)) & 0xFF) as u8;
            }
        }
    }

    let perm = generate_weight_perm_int8();
    let num_chunks = row_len / 1024;
    let n_tiles = n / MARLIN_TILE;
    let mut permuted = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for chunk in 0..num_chunks {
            let base = kt * row_len + chunk * 1024;
            for i in 0..1024 {
                permuted[base + perm[i]] = perm_applied[base + i];
            }
        }
    }

    let mut transposed = vec![0u8; k * n];
    for kt in 0..k_tiles {
        for nt in 0..n_tiles {
            for tk in 0..MARLIN_TILE {
                for tn in 0..MARLIN_TILE {
                    transposed[(kt * MARLIN_TILE + tk) * n + nt * MARLIN_TILE + tn] = permuted
                        [kt * row_len + nt * MARLIN_TILE * MARLIN_TILE + tk * MARLIN_TILE + tn];
                }
            }
        }
    }

    let (scale_perm, scale_perm_single) = generate_scale_perms();
    let is_grouped = group_size < k;
    let sperm: &[usize] = if is_grouped {
        &scale_perm
    } else {
        &scale_perm_single
    };
    let perm_len = sperm.len();
    let total_scale_vals = num_groups_k * n;
    let num_scale_chunks = total_scale_vals / perm_len;
    let mut scales_transposed = vec![0u16; total_scale_vals];
    for chunk in 0..num_scale_chunks {
        let base = chunk * perm_len;
        for i in 0..perm_len {
            scales_transposed[base + sperm[i]] = m.scales[base + i];
        }
    }

    let mut output = vec![0.0f32; n * k];
    for ki in 0..k {
        for ni in 0..n {
            let u8_val = transposed[ki * n + ni];
            let q_val = (u8_val as i16) - 128;
            let scale = bf16_to_f32(scales_transposed[(ki / group_size) * n + ni]);
            output[ni * k + ki] = q_val as f32 * scale;
        }
    }
    output
}

/// Minimal safetensors reader (header-only, mmap data)
fn read_bf16_tensor(path: &Path, tensor_name: &str) -> Option<(Vec<u16>, usize, usize)> {
    use std::io::Read;

    let file = std::fs::File::open(path).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };

    // Header length is first 8 bytes (little-endian u64)
    if mmap.len() < 8 {
        return None;
    }
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header_str = std::str::from_utf8(&mmap[8..8 + header_len]).ok()?;

    // Parse JSON header to find tensor
    // Simple JSON parsing for our needs
    let tensor_key = format!("\"{}\"", tensor_name);
    let pos = header_str.find(&tensor_key)?;
    let after_key = &header_str[pos + tensor_key.len()..];

    // Find shape
    let shape_pos = after_key.find("\"shape\"")?;
    let bracket_start = after_key[shape_pos..].find('[')?;
    let bracket_end = after_key[shape_pos + bracket_start..].find(']')?;
    let shape_str =
        &after_key[shape_pos + bracket_start + 1..shape_pos + bracket_start + bracket_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    if dims.len() != 2 {
        return None;
    }

    // Find data_offsets
    let offsets_pos = after_key.find("\"data_offsets\"")?;
    let ob_start = after_key[offsets_pos..].find('[')?;
    let ob_end = after_key[offsets_pos + ob_start..].find(']')?;
    let offsets_str = &after_key[offsets_pos + ob_start + 1..offsets_pos + ob_start + ob_end];
    let offsets: Vec<usize> = offsets_str
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    let data_start = 8 + header_len + offsets[0];
    let data_end = 8 + header_len + offsets[1];
    let byte_data = &mmap[data_start..data_end];

    // Convert bytes to u16
    let n_elements = dims[0] * dims[1];
    assert_eq!(byte_data.len(), n_elements * 2, "BF16 size mismatch");
    let mut bf16_data = vec![0u16; n_elements];
    for i in 0..n_elements {
        bf16_data[i] = u16::from_le_bytes([byte_data[i * 2], byte_data[i * 2 + 1]]);
    }

    Some((bf16_data, dims[0], dims[1]))
}

fn test_tensor(name: &str, bf16_data: &[u16], n: usize, k: usize) {
    // Check alignment
    assert!(n % 64 == 0, "{name}: N={n} not divisible by 64");
    assert!(
        k % MARLIN_TILE == 0,
        "{name}: K={k} not divisible by {MARLIN_TILE}"
    );
    assert!(
        k % DEFAULT_GROUP_SIZE == 0,
        "{name}: K={k} not divisible by {DEFAULT_GROUP_SIZE}"
    );

    // INT4 roundtrip
    let q4 = quantize_int4(bf16_data, n, k, DEFAULT_GROUP_SIZE);
    let deq4 = dequantize_int4(&q4);
    let m4 = marlin_repack_int4(&q4);
    let deq4m = dequantize_marlin_int4(&m4);

    let mut max_diff4: f32 = 0.0;
    let mut sum_sq_err4: f64 = 0.0;
    let mut sum_sq_orig: f64 = 0.0;
    for i in 0..(n * k) {
        max_diff4 = max_diff4.max((deq4[i] - deq4m[i]).abs());
        let orig = bf16_to_f32(bf16_data[i]);
        sum_sq_err4 += ((orig - deq4m[i]) as f64).powi(2);
        sum_sq_orig += (orig as f64).powi(2);
    }
    let rms_orig = (sum_sq_orig / (n * k) as f64).sqrt();
    let rmse4 = (sum_sq_err4 / (n * k) as f64).sqrt();
    let snr4 = 20.0 * (rms_orig / rmse4).log10();
    assert!(
        max_diff4 == 0.0,
        "{name}: INT4 Marlin repack diff={max_diff4}"
    );

    // INT8 roundtrip
    let q8 = quantize_int8(bf16_data, n, k, DEFAULT_GROUP_SIZE);
    let deq8 = dequantize_int8(&q8);
    let m8 = marlin_repack_int8(&q8);
    let deq8m = dequantize_marlin_int8(&m8);

    let mut max_diff8: f32 = 0.0;
    let mut sum_sq_err8: f64 = 0.0;
    for i in 0..(n * k) {
        max_diff8 = max_diff8.max((deq8[i] - deq8m[i]).abs());
        let orig = bf16_to_f32(bf16_data[i]);
        sum_sq_err8 += ((orig - deq8m[i]) as f64).powi(2);
    }
    let rmse8 = (sum_sq_err8 / (n * k) as f64).sqrt();
    let snr8 = 20.0 * (rms_orig / rmse8).log10();
    assert!(
        max_diff8 == 0.0,
        "{name}: INT8 Marlin repack diff={max_diff8}"
    );

    eprintln!("  {name} [{n}x{k}]: INT4 SNR={snr4:.1}dB  INT8 SNR={snr8:.1}dB  PASS");
}

#[test]
fn test_marlin_attention_shapes_synthetic() {
    // Test all QCN attention projection shapes with synthetic data
    let shapes = [
        ("Q_proj_gqa", 4096, 2048),  // QCN GQA Q
        ("K_proj_gqa", 512, 2048),   // QCN GQA K
        ("V_proj_gqa", 512, 2048),   // QCN GQA V
        ("O_proj_gqa", 2048, 4096),  // QCN GQA O
        ("Q_proj_q235", 8192, 4096), // Q235 Q
        ("K_proj_q235", 512, 4096),  // Q235 K
        ("V_proj_q235", 512, 4096),  // Q235 V
        ("O_proj_q235", 4096, 8192), // Q235 O
    ];

    eprintln!("\n=== Phase 0: Marlin Shape Compatibility (Synthetic) ===");
    for (name, n, k) in &shapes {
        let mut bf16_data = vec![0u16; n * k];
        for i in 0..bf16_data.len() {
            let val = ((i as f32 / bf16_data.len() as f32) - 0.5) * 0.2;
            bf16_data[i] = f32_to_bf16(val);
        }
        test_tensor(name, &bf16_data, *n, *k);
    }
    eprintln!("=== All synthetic shapes PASS ===\n");
}

#[test]
fn test_marlin_attention_shapes_qcn_real() {
    let path =
        Path::new("/home/main/.krasis/models/Qwen3-Coder-Next/model-00010-of-00040.safetensors");
    if !path.exists() {
        eprintln!("Skipping real QCN test — model not downloaded");
        return;
    }

    eprintln!("\n=== Phase 0: Marlin Shape Compatibility (QCN Real Weights) ===");
    let tensors = [
        "model.layers.11.self_attn.q_proj.weight",
        "model.layers.11.self_attn.k_proj.weight",
        "model.layers.11.self_attn.v_proj.weight",
        "model.layers.11.self_attn.o_proj.weight",
    ];

    for tensor_name in &tensors {
        match read_bf16_tensor(path, tensor_name) {
            Some((data, n, k)) => test_tensor(tensor_name, &data, n, k),
            None => eprintln!("  {tensor_name}: not found in shard, skipping"),
        }
    }
    eprintln!("=== QCN real weights PASS ===\n");
}
