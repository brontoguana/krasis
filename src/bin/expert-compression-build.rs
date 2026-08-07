//! Build a versioned, expert-contiguous, bit-exact compression sidecar.

use krasis::expert_codec::{
    encode_expert, CodecHistogram, ComponentKind, ExpertComponent,
};
use krasis::expert_sidecar::{
    encode_header, routed_expert_sha256_from_digests, source_header_sha256,
    ExpertSidecarHeader, SIDECAR_HEADER_BYTES,
};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const CACHE_HEADER_BYTES: usize = 64;
const MARLIN_CACHE_VERSION: u32 = 7;
type BuildResult<T> = Result<T, String>;

#[derive(Deserialize)]
struct ModelConfig {
    #[serde(default = "default_experts_gated")]
    experts_gated: bool,
}

fn default_experts_gated() -> bool {
    true
}

#[derive(Clone, Copy)]
struct CacheLayout {
    experts_per_layer: usize,
    moe_layers: usize,
    w13_packed: usize,
    w13_scales: usize,
    w2_packed: usize,
    expert_bytes: usize,
}

struct TempOutput {
    path: PathBuf,
    keep: bool,
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> BuildResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
        return Err(
            "Usage: ./dev expert-compression-build <cache.bin> <config.json> <output.krec> <lane-bytes>"
                .to_string(),
        );
    }
    let cache_path = PathBuf::from(&args[0]);
    let config_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);
    let lane_bytes = args[3]
        .parse::<usize>()
        .map_err(|error| format!("invalid lane-bytes: {error}"))?;
    if lane_bytes == 0 {
        return Err("lane-bytes must be positive".to_string());
    }
    if output_path.exists() {
        return Err(format!(
            "refusing to overwrite existing expert sidecar {}",
            output_path.display(),
        ));
    }

    let config: ModelConfig = serde_json::from_reader(
        File::open(&config_path)
            .map_err(|error| format!("failed to open {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("failed to parse model config: {error}"))?;
    let cache_file = File::open(&cache_path)
        .map_err(|error| format!("failed to open {}: {error}", cache_path.display()))?;
    let source_cache_bytes = cache_file.metadata().map_err(|error| error.to_string())?.len();
    let cache = unsafe { Mmap::map(&cache_file) }.map_err(|error| error.to_string())?;
    let layout = read_layout(&cache, config.experts_gated)?;
    let expert_count = layout
        .experts_per_layer
        .checked_mul(layout.moe_layers)
        .ok_or_else(|| "expert count overflow".to_string())?;
    let routed_end = CACHE_HEADER_BYTES
        .checked_add(
            expert_count
                .checked_mul(layout.expert_bytes)
                .ok_or_else(|| "routed cache size overflow".to_string())?,
        )
        .ok_or_else(|| "routed cache end overflow".to_string())?;
    if routed_end > cache.len() {
        return Err(format!(
            "cache has {} bytes but routed expert layout requires {routed_end}",
            cache.len(),
        ));
    }

    let histogram_start = Instant::now();
    let measured = (0..expert_count)
        .into_par_iter()
        .map(|ordinal| {
            let source = expert_bytes(&cache, &layout, ordinal);
            let mut histogram = CodecHistogram::default();
            for component in components(source, &layout) {
                histogram.observe(component);
            }
            (histogram, source_header_sha256(source))
        })
        .collect::<Vec<_>>();
    let mut histogram = CodecHistogram::default();
    let mut source_digests = Vec::with_capacity(expert_count);
    for (expert_histogram, digest) in measured {
        histogram.merge(&expert_histogram);
        source_digests.push(digest);
    }
    let routed_expert_sha256 =
        routed_expert_sha256_from_digests(layout.expert_bytes, &source_digests)?;
    let tables = histogram.build_tables()?;
    eprintln!(
        "expert compression build: trained tables from {} real experts ({:.3} GiB) in {:.3}s",
        expert_count,
        expert_count as f64 * layout.expert_bytes as f64 / 1024.0_f64.powi(3),
        histogram_start.elapsed().as_secs_f64(),
    );

    let tables_offset = SIDECAR_HEADER_BYTES;
    let index_offset = align_up(
        tables_offset + tables.serialized_frequencies().len(),
        std::mem::align_of::<u64>(),
    )?;
    let index_bytes = (expert_count + 1)
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| "sidecar index size overflow".to_string())?;
    let payload_offset = align_up(index_offset + index_bytes, 4096)?;

    let temporary_path = temporary_path(&output_path)?;
    let mut cleanup = TempOutput {
        path: temporary_path.clone(),
        keep: false,
    };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .map_err(|error| format!("failed to create {}: {error}", temporary_path.display()))?;
    output
        .set_len(payload_offset as u64)
        .map_err(|error| format!("failed to reserve sidecar metadata: {error}"))?;
    output
        .seek(SeekFrom::Start(tables_offset as u64))
        .map_err(|error| error.to_string())?;
    output
        .write_all(&tables.serialized_frequencies())
        .map_err(|error| error.to_string())?;
    output
        .seek(SeekFrom::Start(payload_offset as u64))
        .map_err(|error| error.to_string())?;

    let build_start = Instant::now();
    let batch_size = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let validation_ordinals = [0, expert_count / 2, expert_count - 1];
    let mut offsets = Vec::with_capacity(expert_count + 1);
    offsets.push(payload_offset as u64);
    let mut encoded_bytes = 0_u64;
    for batch_start in (0..expert_count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(expert_count);
        let encoded = (batch_start..batch_end)
            .into_par_iter()
            .map(|ordinal| {
                let source = expert_bytes(&cache, &layout, ordinal);
                let encoded = encode_expert(&components(source, &layout), &tables, lane_bytes)?;
                if validation_ordinals.contains(&ordinal)
                    && encoded.decode_cpu(&tables)? != source
                {
                    return Err(format!("real expert {ordinal} failed CPU byte identity"));
                }
                Ok(encoded)
            })
            .collect::<BuildResult<Vec<_>>>()?;
        for expert in encoded {
            output
                .write_all(&expert.blob)
                .map_err(|error| format!("failed to write sidecar payload: {error}"))?;
            encoded_bytes = encoded_bytes
                .checked_add(expert.blob.len() as u64)
                .ok_or_else(|| "sidecar payload size overflow".to_string())?;
            offsets.push(payload_offset as u64 + encoded_bytes);
        }
        eprintln!(
            "expert compression build: {}/{} experts, {:.3} GiB written, {:.3}s",
            batch_end,
            expert_count,
            encoded_bytes as f64 / 1024.0_f64.powi(3),
            build_start.elapsed().as_secs_f64(),
        );
    }
    let sidecar_bytes = payload_offset
        .checked_add(encoded_bytes as usize)
        .ok_or_else(|| "sidecar length overflow".to_string())?;
    output
        .seek(SeekFrom::Start(index_offset as u64))
        .map_err(|error| error.to_string())?;
    for offset in offsets {
        output
            .write_all(&offset.to_le_bytes())
            .map_err(|error| format!("failed to write sidecar index: {error}"))?;
    }
    let header = ExpertSidecarHeader {
        source_cache_bytes,
        source_header_sha256: source_header_sha256(&cache[..CACHE_HEADER_BYTES]),
        routed_expert_sha256,
        expert_bytes: layout.expert_bytes,
        expert_count,
        lane_bytes,
        tables_offset,
        index_offset,
        payload_offset,
        sidecar_bytes,
    };
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    output
        .write_all(&encode_header(&header)?)
        .map_err(|error| format!("failed to write sidecar header: {error}"))?;
    output
        .sync_all()
        .map_err(|error| format!("failed to sync sidecar: {error}"))?;
    drop(output);
    std::fs::rename(&temporary_path, &output_path).map_err(|error| {
        format!(
            "failed to publish {} as {}: {error}",
            temporary_path.display(),
            output_path.display(),
        )
    })?;
    cleanup.keep = true;
    eprintln!(
        "expert compression build complete: {} -> {} bytes ({:.4}% saved), output={}, elapsed={:.3}s",
        expert_count as u64 * layout.expert_bytes as u64,
        encoded_bytes,
        (1.0 - encoded_bytes as f64 / (expert_count as f64 * layout.expert_bytes as f64)) * 100.0,
        output_path.display(),
        build_start.elapsed().as_secs_f64(),
    );
    Ok(())
}

fn components<'a>(source: &'a [u8], layout: &CacheLayout) -> [ExpertComponent<'a>; 4] {
    let w13_scales = layout.w13_packed;
    let w2_packed = w13_scales + layout.w13_scales;
    let w2_scales = w2_packed + layout.w2_packed;
    [
        ExpertComponent { bytes: &source[..w13_scales], kind: ComponentKind::PackedNibbles },
        ExpertComponent { bytes: &source[w13_scales..w2_packed], kind: ComponentKind::Bf16Scales },
        ExpertComponent { bytes: &source[w2_packed..w2_scales], kind: ComponentKind::PackedNibbles },
        ExpertComponent { bytes: &source[w2_scales..], kind: ComponentKind::Bf16Scales },
    ]
}

fn expert_bytes<'a>(cache: &'a [u8], layout: &CacheLayout, ordinal: usize) -> &'a [u8] {
    let start = CACHE_HEADER_BYTES + ordinal * layout.expert_bytes;
    &cache[start..start + layout.expert_bytes]
}

fn read_layout(bytes: &[u8], gated: bool) -> BuildResult<CacheLayout> {
    if bytes.len() < CACHE_HEADER_BYTES || &bytes[0..4] != b"KRAS" {
        return Err("invalid Marlin cache header".to_string());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != MARLIN_CACHE_VERSION {
        return Err(format!("Marlin cache version {version}, expected {MARLIN_CACHE_VERSION}"));
    }
    let hidden = read_header_usize(bytes, 8)?;
    let intermediate = read_header_usize(bytes, 16)?;
    let experts_per_layer = read_header_usize(bytes, 24)?;
    let moe_layers = read_header_usize(bytes, 32)?;
    let group = read_header_usize(bytes, 40)?;
    if hidden % 8 != 0 || hidden % group != 0 || intermediate % 8 != 0 {
        return Err("cache dimensions do not satisfy Marlin INT4 packing".to_string());
    }
    let padded_hidden = if hidden == intermediate && hidden % 256 != 0 {
        hidden + 64
    } else {
        hidden
    };
    let w13_width = intermediate * if gated { 2 } else { 1 };
    let w13_packed = (hidden / 8) * w13_width * 4;
    let w13_scales = (hidden / group) * w13_width * 2;
    let w2_packed = (intermediate / 8) * padded_hidden * 4;
    let w2_scales = intermediate.div_ceil(group) * padded_hidden * 2;
    Ok(CacheLayout {
        experts_per_layer,
        moe_layers,
        w13_packed,
        w13_scales,
        w2_packed,
        expert_bytes: w13_packed + w13_scales + w2_packed + w2_scales,
    })
}

fn read_header_usize(bytes: &[u8], offset: usize) -> BuildResult<usize> {
    usize::try_from(u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()))
        .map_err(|_| "cache dimension does not fit usize".to_string())
}

fn align_up(value: usize, alignment: usize) -> BuildResult<usize> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| "alignment overflow".to_string())
}

fn temporary_path(output: &Path) -> BuildResult<PathBuf> {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "sidecar output has no UTF-8 filename".to_string())?;
    Ok(output.with_file_name(format!(".{name}.{}.partial", std::process::id())))
}
