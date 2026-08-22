//! Source-bound TileQ-S routed-expert cache format.
//!
//! TileQ is deliberately separate from the `KRAS` Marlin cache.  A TileQ
//! artifact contains an indexed 3-bit residual plus layer-resident 2D-tiled
//! low-rank factors.  The loader validates the complete index before exposing
//! any borrowed payload range; runtime dispatch must never reinterpret an
//! incomplete or mismatched artifact as INT4.

use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const TILEQ_MAGIC: &[u8; 4] = b"KTQ1";
pub const TILEQ_VERSION: u32 = 1;
pub const TILEQ_HEADER_BYTES: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TileQRange {
    /// Byte offset relative to the start of the payload area.
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileQProjectionManifest {
    pub name: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    pub grid_rows: usize,
    pub grid_cols: usize,
    /// Two little-endian u16 values `(row, col)` per expert.
    pub expert_tiles: TileQRange,
    /// BF16 `[experts, input_dim]` activation-scale inverse from Eq. 12.
    pub expert_inverse_scales_bf16: TileQRange,
    /// BF16 `[grid_rows, input_dim, rank]`, with singular values folded in.
    pub left_factors_bf16: TileQRange,
    /// BF16 `[grid_cols, rank, output_dim]`.
    pub right_factors_bf16: TileQRange,
    pub selected_scale_exponent: f32,
    pub heldout_weighted_mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileQLayerManifest {
    pub model_layer: usize,
    pub expert_count: usize,
    /// Packed row-major signed INT3 gate+up residuals for all experts.
    pub w13_packed: TileQRange,
    /// BF16 group scales matching `w13_packed`.
    pub w13_scales: TileQRange,
    /// Packed row-major signed INT3 down residuals for all experts.
    pub w2_packed: TileQRange,
    /// BF16 group scales matching `w2_packed`.
    pub w2_scales: TileQRange,
    pub per_expert_w13_packed: u64,
    pub per_expert_w13_scales: u64,
    pub per_expert_w2_packed: u64,
    pub per_expert_w2_scales: u64,
    pub gate: TileQProjectionManifest,
    pub up: TileQProjectionManifest,
    pub down: TileQProjectionManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileQManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub architecture: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub routed_experts: usize,
    pub routed_layers: usize,
    pub residual_bits: u8,
    pub group_size: usize,
    pub rank: usize,
    pub source_routed_sha256: String,
    pub source_config_sha256: String,
    pub calibration_sha256: String,
    pub heldout_sha256: String,
    pub scale_exponent_candidates: Vec<f32>,
    pub sketch_seed: u64,
    pub sketch_iterations: usize,
    pub clustering_seed: u64,
    /// Exact residual quantizer used by the offline builder. The first
    /// implementation uses diagonal-Hessian scale search; later full GPTQ
    /// artifacts must use a distinct value rather than changing semantics.
    pub residual_quantizer: String,
    pub scale_search_multipliers: Vec<f32>,
    pub gptq_block_size: usize,
    pub payload_bytes: u64,
    pub layers: Vec<TileQLayerManifest>,
}

pub struct TileQCache {
    path: PathBuf,
    mmap: Mmap,
    payload_offset: usize,
    manifest: TileQManifest,
}

impl TileQCache {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| format!("failed to open TileQ cache {}: {e}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| format!("failed to mmap TileQ cache {}: {e}", path.display()))?;
        if mmap.len() < TILEQ_HEADER_BYTES {
            return Err(format!(
                "TileQ cache {} is truncated: {} bytes",
                path.display(),
                mmap.len()
            ));
        }
        if &mmap[0..4] != TILEQ_MAGIC {
            return Err(format!("TileQ cache {} has invalid magic", path.display()));
        }
        let version = read_u32(&mmap, 4)?;
        if version != TILEQ_VERSION {
            return Err(format!(
                "TileQ cache {} has schema version {}, expected {}",
                path.display(),
                version,
                TILEQ_VERSION
            ));
        }
        let manifest_len = usize::try_from(read_u64(&mmap, 8)?)
            .map_err(|_| "TileQ manifest length does not fit usize".to_string())?;
        let payload_offset = usize::try_from(read_u64(&mmap, 16)?)
            .map_err(|_| "TileQ payload offset does not fit usize".to_string())?;
        let payload_len = usize::try_from(read_u64(&mmap, 24)?)
            .map_err(|_| "TileQ payload length does not fit usize".to_string())?;
        let manifest_end = TILEQ_HEADER_BYTES
            .checked_add(manifest_len)
            .ok_or_else(|| "TileQ manifest range overflow".to_string())?;
        let payload_end = payload_offset
            .checked_add(payload_len)
            .ok_or_else(|| "TileQ payload range overflow".to_string())?;
        if manifest_end > mmap.len() || payload_offset < manifest_end || payload_end != mmap.len() {
            return Err(format!(
                "TileQ cache {} has invalid ranges: manifest_end={} payload={}..{} file={}",
                path.display(),
                manifest_end,
                payload_offset,
                payload_end,
                mmap.len()
            ));
        }
        let manifest_bytes = &mmap[TILEQ_HEADER_BYTES..manifest_end];
        let expected_manifest_sha = &mmap[32..64];
        let actual_manifest_sha = Sha256::digest(manifest_bytes);
        if actual_manifest_sha.as_slice() != expected_manifest_sha {
            return Err(format!(
                "TileQ cache {} manifest SHA-256 mismatch",
                path.display()
            ));
        }
        let manifest: TileQManifest = serde_json::from_slice(manifest_bytes)
            .map_err(|e| format!("failed to parse TileQ manifest {}: {e}", path.display()))?;
        validate_manifest(&manifest, payload_len as u64)?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            payload_offset,
            manifest,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn manifest(&self) -> &TileQManifest {
        &self.manifest
    }

    pub fn bytes(&self, range: &TileQRange) -> Result<&[u8], String> {
        validate_range(range, self.manifest.payload_bytes, "requested")?;
        let start = self
            .payload_offset
            .checked_add(range.offset as usize)
            .ok_or_else(|| "TileQ requested range start overflow".to_string())?;
        let end = start
            .checked_add(range.len as usize)
            .ok_or_else(|| "TileQ requested range end overflow".to_string())?;
        Ok(&self.mmap[start..end])
    }

    /// Create a bounded private writable mapping for one payload range.
    ///
    /// CUDA host registration applies to a complete VMA on some driver/kernel
    /// combinations. Mapping the whole TileQ corpus and registering a slice
    /// would therefore make a 100+ GiB registration request. Every component
    /// range is page aligned by the builder, so runtime maps exactly one layer
    /// component per VMA and registers only that bounded allocation.
    pub fn map_private(&self, range: &TileQRange) -> Result<MmapMut, String> {
        validate_range(range, self.manifest.payload_bytes, "private mapping")?;
        let file_offset = self
            .payload_offset
            .checked_add(range.offset as usize)
            .ok_or_else(|| "TileQ private mapping offset overflow".to_string())?;
        if file_offset % 4096 != 0 {
            return Err(format!(
                "TileQ private mapping offset {file_offset} is not 4096-byte aligned"
            ));
        }
        let file = File::open(&self.path).map_err(|e| {
            format!(
                "failed to reopen TileQ cache {} for private mapping: {e}",
                self.path.display()
            )
        })?;
        unsafe {
            MmapOptions::new()
                .offset(file_offset as u64)
                .len(range.len as usize)
                .map_copy(&file)
        }
        .map_err(|e| {
            format!(
                "failed to privately map TileQ cache {} at {}+{}: {e}",
                self.path.display(),
                file_offset,
                range.len
            )
        })
    }
}

/// Match the offline builder's source-identity hash: sorted basename, file
/// length, then complete contents for each input file.
pub fn combined_file_sha256(paths: &[PathBuf]) -> Result<String, String> {
    let mut paths = paths.to_vec();
    paths.sort_by_key(|path| path.file_name().map(|name| name.to_os_string()));
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    for path in paths {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("TileQ source path {} has no UTF-8 basename", path.display()))?;
        let encoded = name.as_bytes();
        digest.update((encoded.len() as u64).to_le_bytes());
        digest.update(encoded);
        let metadata = std::fs::metadata(&path)
            .map_err(|e| format!("failed to stat TileQ source {}: {e}", path.display()))?;
        digest.update(metadata.len().to_le_bytes());
        let mut file = File::open(&path)
            .map_err(|e| format!("failed to open TileQ source {}: {e}", path.display()))?;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|e| format!("failed to hash TileQ source {}: {e}", path.display()))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "TileQ u32 offset overflow".to_string())?;
    let raw: [u8; 4] = bytes
        .get(offset..end)
        .ok_or_else(|| "TileQ header is truncated".to_string())?
        .try_into()
        .map_err(|_| "TileQ u32 conversion failed".to_string())?;
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "TileQ u64 offset overflow".to_string())?;
    let raw: [u8; 8] = bytes
        .get(offset..end)
        .ok_or_else(|| "TileQ header is truncated".to_string())?
        .try_into()
        .map_err(|_| "TileQ u64 conversion failed".to_string())?;
    Ok(u64::from_le_bytes(raw))
}

fn validate_range(range: &TileQRange, payload_bytes: u64, label: &str) -> Result<(), String> {
    let end = range
        .offset
        .checked_add(range.len)
        .ok_or_else(|| format!("TileQ {label} range overflow"))?;
    if end > payload_bytes {
        return Err(format!(
            "TileQ {label} range {}..{} exceeds payload {}",
            range.offset, end, payload_bytes
        ));
    }
    Ok(())
}

fn validate_projection(
    projection: &TileQProjectionManifest,
    expert_count: usize,
    rank: usize,
    payload_bytes: u64,
) -> Result<(), String> {
    if projection.input_dim == 0
        || projection.output_dim == 0
        || projection.rank != rank
        || projection.grid_rows == 0
        || projection.grid_cols == 0
        || projection.grid_rows.saturating_mul(projection.grid_cols) < expert_count
    {
        return Err(format!(
            "TileQ projection {} has invalid geometry input={} output={} rank={} grid={}x{} experts={}",
            projection.name,
            projection.input_dim,
            projection.output_dim,
            projection.rank,
            projection.grid_rows,
            projection.grid_cols,
            expert_count
        ));
    }
    let tile_bytes = expert_count
        .checked_mul(4)
        .ok_or_else(|| format!("TileQ {} tile length overflow", projection.name))?
        as u64;
    if projection.expert_tiles.len != tile_bytes {
        return Err(format!(
            "TileQ projection {} tile table is {} bytes, expected {}",
            projection.name, projection.expert_tiles.len, tile_bytes
        ));
    }
    let left_bytes = projection
        .grid_rows
        .checked_mul(projection.input_dim)
        .and_then(|v| v.checked_mul(rank))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| format!("TileQ {} left factor length overflow", projection.name))?
        as u64;
    let right_bytes = projection
        .grid_cols
        .checked_mul(rank)
        .and_then(|v| v.checked_mul(projection.output_dim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| format!("TileQ {} right factor length overflow", projection.name))?
        as u64;
    let inverse_scale_bytes = expert_count
        .checked_mul(projection.input_dim)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| format!("TileQ {} inverse-scale length overflow", projection.name))?
        as u64;
    if projection.left_factors_bf16.len != left_bytes
        || projection.right_factors_bf16.len != right_bytes
        || projection.expert_inverse_scales_bf16.len != inverse_scale_bytes
    {
        return Err(format!(
            "TileQ projection {} factor lengths inverse={}/{} left={}/{} right={}/{}",
            projection.name,
            projection.expert_inverse_scales_bf16.len,
            inverse_scale_bytes,
            projection.left_factors_bf16.len,
            left_bytes,
            projection.right_factors_bf16.len,
            right_bytes
        ));
    }
    for (label, range) in [
        ("tiles", &projection.expert_tiles),
        ("inverse_scales", &projection.expert_inverse_scales_bf16),
        ("left", &projection.left_factors_bf16),
        ("right", &projection.right_factors_bf16),
    ] {
        validate_range(
            range,
            payload_bytes,
            &format!("{} {label}", projection.name),
        )?;
    }
    if !projection.selected_scale_exponent.is_finite()
        || !projection.heldout_weighted_mse.is_finite()
        || projection.heldout_weighted_mse < 0.0
    {
        return Err(format!(
            "TileQ projection {} has non-finite calibration metadata",
            projection.name
        ));
    }
    Ok(())
}

fn validate_manifest(manifest: &TileQManifest, payload_bytes: u64) -> Result<(), String> {
    if manifest.schema_version != TILEQ_VERSION
        || manifest.residual_bits != 3
        || manifest.group_size == 0
        || manifest.rank == 0
        || manifest.residual_quantizer.is_empty()
        || manifest.scale_search_multipliers.is_empty()
        || manifest
            .scale_search_multipliers
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || manifest.hidden_size == 0
        || manifest.intermediate_size == 0
        || manifest.routed_experts == 0
        || manifest.routed_layers == 0
        || manifest.layers.len() != manifest.routed_layers
        || manifest.payload_bytes != payload_bytes
    {
        return Err("TileQ manifest top-level geometry/version mismatch".to_string());
    }
    for hash in [
        &manifest.source_routed_sha256,
        &manifest.source_config_sha256,
        &manifest.calibration_sha256,
        &manifest.heldout_sha256,
    ] {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("TileQ manifest contains invalid SHA-256 {hash:?}"));
        }
    }
    for (layer_index, layer) in manifest.layers.iter().enumerate() {
        if layer.expert_count != manifest.routed_experts
            || layer.model_layer < layer_index
            || layer.per_expert_w13_packed == 0
            || layer.per_expert_w13_scales == 0
            || layer.per_expert_w2_packed == 0
            || layer.per_expert_w2_scales == 0
        {
            return Err(format!("TileQ layer {layer_index} has invalid geometry"));
        }
        for (label, range, per_expert) in [
            ("w13_packed", &layer.w13_packed, layer.per_expert_w13_packed),
            ("w13_scales", &layer.w13_scales, layer.per_expert_w13_scales),
            ("w2_packed", &layer.w2_packed, layer.per_expert_w2_packed),
            ("w2_scales", &layer.w2_scales, layer.per_expert_w2_scales),
        ] {
            validate_range(range, payload_bytes, label)?;
            let expected = per_expert
                .checked_mul(layer.expert_count as u64)
                .ok_or_else(|| format!("TileQ layer {layer_index} {label} length overflow"))?;
            if range.len != expected {
                return Err(format!(
                    "TileQ layer {layer_index} {label} length {}/{}",
                    range.len, expected
                ));
            }
        }
        validate_projection(
            &layer.gate,
            layer.expert_count,
            manifest.rank,
            payload_bytes,
        )?;
        validate_projection(&layer.up, layer.expert_count, manifest.rank, payload_bytes)?;
        validate_projection(
            &layer.down,
            layer.expert_count,
            manifest.rank,
            payload_bytes,
        )?;
    }
    Ok(())
}

/// Pack signed INT3 values (`-4..=3`) into a dense little-endian bitstream.
/// Rows are independently word-aligned so a CUDA row can be addressed without
/// scanning preceding rows. `cols` must be divisible by 32, which is true for
/// every currently supported routed expert but is validated rather than
/// assumed by the runtime.
pub fn pack_signed_int3_rows(values: &[i8], rows: usize, cols: usize) -> Result<Vec<u32>, String> {
    if rows == 0 || cols == 0 || cols % 32 != 0 || values.len() != rows.saturating_mul(cols) {
        return Err(format!(
            "invalid INT3 matrix rows={} cols={} values={}; require cols divisible by 32",
            rows,
            cols,
            values.len()
        ));
    }
    let words_per_row = cols
        .checked_mul(3)
        .and_then(|v| v.checked_div(32))
        .ok_or_else(|| "INT3 row word count overflow".to_string())?;
    let mut packed = vec![0u32; rows * words_per_row];
    for row in 0..rows {
        let row_base = row * cols;
        let word_base = row * words_per_row;
        for col in 0..cols {
            let value = values[row_base + col];
            if !(-4..=3).contains(&value) {
                return Err(format!(
                    "INT3 value {value} at ({row},{col}) is outside -4..=3"
                ));
            }
            let encoded = (value as i32 & 0x7) as u32;
            let bit = col * 3;
            let word = bit / 32;
            let shift = bit % 32;
            packed[word_base + word] |= encoded << shift;
            if shift > 29 {
                packed[word_base + word + 1] |= encoded >> (32 - shift);
            }
        }
    }
    Ok(packed)
}

pub fn unpack_signed_int3_rows(
    packed: &[u32],
    rows: usize,
    cols: usize,
) -> Result<Vec<i8>, String> {
    if rows == 0 || cols == 0 || cols % 32 != 0 {
        return Err(format!(
            "invalid INT3 unpack geometry rows={rows} cols={cols}"
        ));
    }
    let words_per_row = cols * 3 / 32;
    if packed.len() != rows.saturating_mul(words_per_row) {
        return Err(format!(
            "INT3 packed length {} does not match rows={} cols={} expected={}",
            packed.len(),
            rows,
            cols,
            rows * words_per_row
        ));
    }
    let mut values = vec![0i8; rows * cols];
    for row in 0..rows {
        let word_base = row * words_per_row;
        for col in 0..cols {
            let bit = col * 3;
            let word = bit / 32;
            let shift = bit % 32;
            let mut encoded = packed[word_base + word] >> shift;
            if shift > 29 {
                encoded |= packed[word_base + word + 1] << (32 - shift);
            }
            let encoded = (encoded & 0x7) as i8;
            values[row * cols + col] = if encoded & 0x4 != 0 {
                encoded - 8
            } else {
                encoded
            };
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{pack_signed_int3_rows, unpack_signed_int3_rows};

    #[test]
    fn int3_dense_row_round_trip_crosses_words_exactly() {
        let rows = 3;
        let cols = 128;
        let values = (0..rows * cols)
            .map(|i| ((i % 8) as i8) - 4)
            .collect::<Vec<_>>();
        let packed = pack_signed_int3_rows(&values, rows, cols).unwrap();
        assert_eq!(packed.len() * 4, rows * cols * 3 / 8);
        assert_eq!(
            unpack_signed_int3_rows(&packed, rows, cols).unwrap(),
            values
        );
    }

    #[test]
    fn int3_pack_rejects_non_word_aligned_rows_and_out_of_range_values() {
        assert!(pack_signed_int3_rows(&vec![0; 31], 1, 31).is_err());
        let mut values = vec![0; 32];
        values[17] = 4;
        assert!(pack_signed_int3_rows(&values, 1, 32).is_err());
    }
}
