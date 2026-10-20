//! Crash-resilient compact mixed INT4/INT8 Marlin cache builder.
//!
//! The builder streams one routed expert from the immutable checkpoint at a
//! time. A progress journal is published only after the record has been
//! flushed and synced; resume rehashes every committed record and truncates
//! any uncommitted tail before continuing.

use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const PLAN_FORMAT: &str = "Krasis compact mixed Marlin build plan";
const PLAN_VERSION: u32 = 1;
const PROGRESS_FORMAT: &str = "Krasis compact mixed Marlin build progress";
const PROGRESS_VERSION: u32 = 1;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompactBuildPlan {
    format: String,
    schema_version: u32,
    manifest: mixed_precision::MixedPrecisionManifest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompletedRecord {
    index: usize,
    offset: u64,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BuildProgress {
    format: String,
    schema_version: u32,
    plan_sha256: String,
    output_basename: String,
    records: Vec<CompletedRecord>,
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value)
        .map_err(|error| format!("serialize compact mixed cache state: {error}"))
}

fn read_canonical<T>(path: &Path) -> Result<(T, Vec<u8>), String>
where
    T: for<'de> Deserialize<'de> + Serialize,
{
    let raw = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let value: T = serde_json::from_slice(&raw)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if canonical_json(&value)? != raw {
        return Err(format!("{} is not canonical compact JSON", path.display()));
    }
    Ok((value, raw))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create {}: {error}", parent.display()))?;
    let tmp = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("state"),
        std::process::id()
    ));
    if tmp.exists() {
        return Err(format!(
            "refusing stale compact mixed cache temporary file {}",
            tmp.display()
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|error| format!("create {}: {error}", tmp.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write {}: {error}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|error| format!("publish {}: {error}", path.display()))?;
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("sync {}: {error}", parent.display()))?;
    Ok(())
}

fn sha256_file_range(file: &mut File, offset: u64, bytes: u64) -> Result<String, String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek compact cache: {error}"))?;
    let mut remaining = bytes;
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut digest = Sha256::new();
    while remaining > 0 {
        let want = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        file.read_exact(&mut buffer[..want])
            .map_err(|error| format!("read compact cache record: {error}"))?;
        digest.update(&buffer[..want]);
        remaining -= want as u64;
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn decode_sha256(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err("SHA-256 must contain 64 lowercase hexadecimal characters".to_string());
    }
    let mut result = [0u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| "SHA-256 contains non-hexadecimal data".to_string())?;
    }
    Ok(result)
}

fn memory_hwm_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?.split_whitespace().next()?;
        value.parse::<u64>().ok()?.checked_mul(1024)
    })
}

#[derive(Debug, Serialize)]
struct SystemMemorySnapshot {
    process_vm_hwm_bytes: Option<u64>,
    mem_available_bytes: Option<u64>,
    cached_bytes: Option<u64>,
    buffers_bytes: Option<u64>,
    reclaimable_bytes: Option<u64>,
}

fn system_memory_snapshot() -> SystemMemorySnapshot {
    let mut snapshot = SystemMemorySnapshot {
        process_vm_hwm_bytes: memory_hwm_bytes(),
        mem_available_bytes: None,
        cached_bytes: None,
        buffers_bytes: None,
        reclaimable_bytes: None,
    };
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            let Some((key, tail)) = line.split_once(':') else {
                continue;
            };
            let Some(bytes) = tail
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .and_then(|value| value.checked_mul(1024))
            else {
                continue;
            };
            match key {
                "MemAvailable" => snapshot.mem_available_bytes = Some(bytes),
                "Cached" => snapshot.cached_bytes = Some(bytes),
                "Buffers" => snapshot.buffers_bytes = Some(bytes),
                "SReclaimable" => snapshot.reclaimable_bytes = Some(bytes),
                _ => {}
            }
        }
    }
    snapshot
}

fn region_bytes(
    region: &mixed_precision::MixedRegion,
    geometry: &mixed_precision::MixedPrecisionGeometry,
) -> usize {
    if region.bits == 4 {
        geometry.int4_expert_bytes
    } else {
        geometry.int8_expert_bytes
    }
}

fn validate_resume(
    partial: &mut File,
    progress: &BuildProgress,
    expected_plan_sha256: &str,
    output_basename: &str,
    manifest: &mixed_precision::MixedPrecisionManifest,
    geometry: &mixed_precision::MixedPrecisionGeometry,
) -> Result<u64, String> {
    if progress.format != PROGRESS_FORMAT
        || progress.schema_version != PROGRESS_VERSION
        || progress.plan_sha256 != expected_plan_sha256
        || progress.output_basename != output_basename
        || progress.records.len() > manifest.regions.len()
    {
        return Err("compact mixed cache progress identity mismatch".to_string());
    }
    let mut expected_offset = mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64;
    for (index, record) in progress.records.iter().enumerate() {
        let expected_bytes = region_bytes(&manifest.regions[index], geometry) as u64;
        if record.index != index
            || record.offset != expected_offset
            || record.bytes != expected_bytes
            || record.sha256.len() != 64
            || record.sha256.bytes().any(|byte| !byte.is_ascii_hexdigit())
            || sha256_file_range(partial, record.offset, record.bytes)? != record.sha256
        {
            return Err(format!(
                "compact mixed cache committed record {index} failed resume verification"
            ));
        }
        expected_offset = expected_offset
            .checked_add(record.bytes)
            .ok_or_else(|| "compact mixed cache resume offset overflow".to_string())?;
    }
    partial
        .set_len(expected_offset)
        .map_err(|error| format!("truncate uncommitted compact cache tail: {error}"))?;
    partial
        .seek(SeekFrom::Start(expected_offset))
        .map_err(|error| format!("seek compact cache append position: {error}"))?;
    Ok(expected_offset)
}

pub(super) fn build_compact_mixed_cache(
    model_dir: &Path,
    plan_path: &Path,
    calibration_path: &Path,
    output_cache: &Path,
    output_manifest: &Path,
) -> Result<String, String> {
    build_compact_mixed_cache_impl(
        model_dir,
        plan_path,
        calibration_path,
        output_cache,
        output_manifest,
        None,
    )
}

pub(super) fn assemble_compact_mixed_cache(
    model_dir: &Path,
    plan_path: &Path,
    calibration_path: &Path,
    output_cache: &Path,
    output_manifest: &Path,
    int4_manifest: &Path,
    int8_manifest: &Path,
) -> Result<String, String> {
    build_compact_mixed_cache_impl(
        model_dir,
        plan_path,
        calibration_path,
        output_cache,
        output_manifest,
        Some([int4_manifest, int8_manifest]),
    )
}

/// Reuse only an immutable, fully verified packed bank with the same measured
/// calibration. Keep file reads bounded to one expert and reject changed files.
struct ReuseSource {
    manifest: mixed_precision::MixedPrecisionManifest,
    manifest_path: std::path::PathBuf,
    manifest_raw: Vec<u8>,
    manifest_metadata: std::fs::Metadata,
    _manifest_lock: File,
    path: std::path::PathBuf,
    file: File,
    metadata: std::fs::Metadata,
    offsets: Vec<u64>,
}

impl ReuseSource {
    fn open(
        path: &Path,
        target: &mixed_precision::MixedPrecisionManifest,
        geometry: &mixed_precision::MixedPrecisionGeometry,
        ranking_raw: &[u8],
    ) -> Result<Self, String> {
        let manifest_lock = File::open(path).map_err(|e| e.to_string())?;
        manifest_lock
            .try_lock_shared()
            .map_err(|e| format!("lock reuse manifest: {e}"))?;
        let (manifest, manifest_raw): (mixed_precision::MixedPrecisionManifest, Vec<u8>) =
            read_canonical(path)?;
        mixed_precision::validate_manifest(
            &manifest,
            geometry,
            &target.checkpoint_sha256,
            &target.cache_namespace,
            &target.config_sha256,
        )?;
        mixed_precision::validate_ranking_evidence_bytes(&manifest, ranking_raw)?;
        let mut calibration = manifest.calibration.clone();
        // Ordering chooses records; it never changes their quantization.
        calibration.ranking_method = target.calibration.ranking_method.clone();
        if calibration != target.calibration {
            return Err("reuse source calibration/ranking identity differs from target".into());
        }
        let identity = manifest
            .compact_cache
            .as_ref()
            .ok_or("reuse requires a compact source cache")?;
        let cache_path = path
            .parent()
            .ok_or("reuse manifest has no parent")?
            .join(&identity.basename);
        let mut file = File::open(&cache_path).map_err(|e| format!("open reuse source: {e}"))?;
        file.try_lock_shared()
            .map_err(|e| format!("lock reuse source: {e}"))?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if metadata.len() != identity.bytes {
            return Err("reuse source byte length mismatch".into());
        }
        let mut header = [0u8; mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES];
        file.read_exact(&mut header)
            .map_err(|e| format!("read reuse header: {e}"))?;
        let mut offsets = Vec::with_capacity(manifest.regions.len());
        let mut end = header.len() as u64;
        for region in &manifest.regions {
            offsets.push(end);
            end = end
                .checked_add(region_bytes(region, geometry) as u64)
                .ok_or("reuse offset overflow")?;
        }
        if &header[..4] != mixed_precision::MIXED_COMPACT_CACHE_MAGIC
            || u32::from_le_bytes(header[4..8].try_into().unwrap())
                != mixed_precision::MIXED_COMPACT_CACHE_VERSION
            || u64::from_le_bytes(header[8..16].try_into().unwrap()) != header.len() as u64
            || u64::from_le_bytes(header[16..24].try_into().unwrap()) != end - header.len() as u64
            || header[24..56] != decode_sha256(&identity.payload_sha256)?
            || header[56..88] != decode_sha256(&identity.region_map_sha256)?
            || header[88..120] != decode_sha256(&target.checkpoint_sha256)?
            || identity.region_map_sha256 != mixed_precision::region_map_sha256(&manifest.regions)?
            || identity.header_sha256 != mixed_precision::sha256_hex(&header)
            || end != metadata.len()
            || sha256_file_range(&mut file, 0, metadata.len())? != identity.sha256
        {
            return Err("reuse source header/payload/geometry identity mismatch".into());
        }
        let manifest_metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
        let source = Self {
            manifest,
            manifest_path: path.to_path_buf(),
            manifest_raw,
            manifest_metadata,
            _manifest_lock: manifest_lock,
            path: cache_path,
            file,
            metadata,
            offsets,
        };
        source.check_unchanged()?;
        Ok(source)
    }

    fn reject_output_aliases(&self, outputs: &[&Path]) -> Result<(), String> {
        for input in [&self.path, &self.manifest_path] {
            let canonical = std::fs::canonicalize(input).map_err(|e| e.to_string())?;
            for output in outputs {
                if output.exists()
                    && std::fs::canonicalize(output).map_err(|e| e.to_string())? == canonical
                {
                    return Err("reuse output must be distinct from source cache/manifest".into());
                }
            }
        }
        Ok(())
    }

    fn check_unchanged(&self) -> Result<(), String> {
        for current in [self.file.metadata(), std::fs::metadata(&self.path)] {
            let current = current.map_err(|e| format!("stat reuse source: {e}"))?;
            if current.len() != self.metadata.len()
                || current.modified().map_err(|e| e.to_string())?
                    != self.metadata.modified().map_err(|e| e.to_string())?
            {
                return Err("reuse source changed during assembly".into());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if current.dev() != self.metadata.dev() || current.ino() != self.metadata.ino() {
                    return Err("reuse source replaced during assembly".into());
                }
            }
        }
        let current = std::fs::metadata(&self.manifest_path).map_err(|e| e.to_string())?;
        if current.len() != self.manifest_metadata.len()
            || current.modified().map_err(|e| e.to_string())?
                != self
                    .manifest_metadata
                    .modified()
                    .map_err(|e| e.to_string())?
        {
            return Err("reuse source manifest changed during assembly".into());
        }
        Ok(())
    }

    fn check_final(&self) -> Result<(), String> {
        self.check_unchanged()?;
        if std::fs::read(&self.manifest_path).map_err(|e| e.to_string())? != self.manifest_raw {
            return Err("reuse source manifest changed during assembly".into());
        }
        Ok(())
    }

    fn record(
        &mut self,
        index: usize,
        target: &mixed_precision::MixedRegion,
        geometry: &mixed_precision::MixedPrecisionGeometry,
    ) -> Result<Vec<u8>, String> {
        self.check_unchanged()?;
        let region = self
            .manifest
            .regions
            .get(index)
            .ok_or("reuse source region missing")?;
        if (
            region.model_layer,
            region.moe_layer,
            region.expert,
            region.bits,
        ) != (
            target.model_layer,
            target.moe_layer,
            target.expert,
            target.bits,
        ) {
            return Err(format!(
                "reuse source lacks requested precision at record {index}"
            ));
        }
        let mut record = vec![0u8; region_bytes(target, geometry)];
        self.file
            .seek(SeekFrom::Start(self.offsets[index]))
            .and_then(|_| self.file.read_exact(&mut record))
            .map_err(|e| format!("read reuse record {index}: {e}"))?;
        self.check_unchanged()?;
        Ok(record)
    }
}

fn build_compact_mixed_cache_impl(
    model_dir: &Path,
    plan_path: &Path,
    calibration_path: &Path,
    output_cache: &Path,
    output_manifest: &Path,
    reuse_manifests: Option<[&Path; 2]>,
) -> Result<String, String> {
    let prepare_started = std::time::Instant::now();
    let memory_start = system_memory_snapshot();
    let plan_raw = std::fs::read(plan_path).map_err(|error| format!("read build plan: {error}"))?;
    let plan_json: serde_json::Value =
        serde_json::from_slice(&plan_raw).map_err(|error| format!("parse build plan: {error}"))?;
    if serde_json::to_vec(&plan_json).map_err(|error| error.to_string())? != plan_raw {
        return Err("build plan must use sorted canonical compact JSON".to_string());
    }
    let mut plan: CompactBuildPlan = serde_json::from_value(plan_json)
        .map_err(|error| format!("parse build plan contract: {error}"))?;
    if plan.format != PLAN_FORMAT || plan.schema_version != PLAN_VERSION {
        return Err("unsupported compact mixed cache build-plan contract".to_string());
    }
    if !matches!(
        plan.manifest.schema_version,
        mixed_precision::MIXED_MANIFEST_VERSION | mixed_precision::MIXED_COVERAGE_MANIFEST_VERSION
    ) || plan.manifest.compact_cache.is_some()
        || plan.manifest.source_int4.is_some()
        || plan.manifest.source_int8.is_some()
    {
        return Err(
            "compact mixed cache build plan must contain an unpublished version-2 manifest"
                .to_string(),
        );
    }
    if output_manifest.exists() {
        return Err("refusing to overwrite a published compact mixed manifest".to_string());
    }
    let output_dir = output_manifest
        .parent()
        .ok_or_else(|| "compact mixed manifest has no parent directory".to_string())?;
    if output_cache.parent() != Some(output_dir) || plan_path.parent() != Some(output_dir) {
        return Err("compact mixed plan, cache, and manifest must share one directory".to_string());
    }
    let output_basename = output_cache
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "compact mixed cache basename is not valid UTF-8".to_string())?;
    if Path::new(output_basename)
        .file_name()
        .and_then(|value| value.to_str())
        != Some(output_basename)
    {
        return Err("compact mixed cache basename is unsafe".to_string());
    }
    let partial_path = output_cache.with_file_name(format!(".{output_basename}.partial"));
    let progress_path = output_cache.with_file_name(format!(".{output_basename}.progress.json"));

    let config_path = model_dir.join("config.json");
    let config_raw = std::fs::read(&config_path)
        .map_err(|error| format!("read {}: {error}", config_path.display()))?;
    let raw_json: serde_json::Value = serde_json::from_slice(&config_raw)
        .map_err(|error| format!("parse {}: {error}", config_path.display()))?;
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_raw = std::fs::read(&index_path)
        .map_err(|error| format!("read {}: {error}", index_path.display()))?;
    let index_json: serde_json::Value = serde_json::from_slice(&index_raw)
        .map_err(|error| format!("parse {}: {error}", index_path.display()))?;
    let safetensors_index: SafetensorsIndex = serde_json::from_slice(&index_raw)
        .map_err(|error| format!("parse safetensors index: {error}"))?;
    let config = ModelConfig::from_json_with_index(&raw_json, Some(&index_json))?;
    let checkpoint_sha256 = crate::checkpoint_identity::checkpoint_identity_sha256(model_dir)?;
    let cache_namespace = crate::checkpoint_identity::cache_namespace(model_dir)?;
    let config_sha256 = format!("{:x}", Sha256::digest(&config_raw));
    let model_layers = (0..config.num_moe_layers())
        .map(|moe_layer| {
            u32::try_from(config.moe_abs_layer(moe_layer))
                .map_err(|_| "model layer exceeds u32".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let int4_sizes = marlin_expert_byte_sizes(&config, plan.manifest.group_size as usize, 4);
    let int8_sizes = marlin_expert_byte_sizes(&config, plan.manifest.group_size as usize, 8);
    let geometry = mixed_precision::MixedPrecisionGeometry {
        group_size: plan.manifest.group_size as usize,
        hidden_size: config.routed_expert_hidden_size(),
        intermediate_size: config.moe_intermediate_size,
        routed_layers: config.num_moe_layers(),
        routed_experts_per_layer: config.n_routed_experts,
        experts_gated: config.experts_gated,
        model_layers,
        int4_expert_bytes: int4_sizes.0 + int4_sizes.1 + int4_sizes.2 + int4_sizes.3,
        int8_expert_bytes: int8_sizes.0 + int8_sizes.1 + int8_sizes.2 + int8_sizes.3,
    };
    let calibration_raw = std::fs::read(calibration_path)
        .map_err(|error| format!("read {}: {error}", calibration_path.display()))?;
    let calibration_sha256 = format!("{:x}", Sha256::digest(&calibration_raw));
    if plan
        .manifest
        .calibration
        .search_rmse_calibration_sha256
        .as_deref()
        != Some(calibration_sha256.as_str())
    {
        return Err("compact mixed cache search-RMSE calibration identity mismatch".to_string());
    }
    let calibration =
        ExpertInt4CalibData::from_bytes(calibration_path.to_path_buf(), calibration_raw)?;
    calibration.validate_for_model(&config)?;
    let ranking_path = output_dir.join("mixed-expert-ranking.json");
    let ranking_raw = std::fs::read(&ranking_path)
        .map_err(|error| format!("read {}: {error}", ranking_path.display()))?;
    mixed_precision::validate_ranking_evidence_bytes(&plan.manifest, &ranking_raw)?;
    let ranking_json: Option<serde_json::Value> =
        if plan.manifest.schema_version == mixed_precision::MIXED_COVERAGE_MANIFEST_VERSION {
            Some(
                serde_json::from_slice(&ranking_raw)
                    .map_err(|error| format!("parse ranking projections: {error}"))?,
            )
        } else {
            None
        };

    // Validate every non-cache aspect before touching checkpoint payloads.
    plan.manifest.compact_cache = Some(mixed_precision::MixedCompactCacheIdentity {
        basename: output_basename.to_string(),
        bytes: 1,
        sha256: "00".repeat(32),
        header_sha256: "00".repeat(32),
        payload_sha256: "00".repeat(32),
        region_map_sha256: mixed_precision::region_map_sha256(&plan.manifest.regions)?,
    });
    mixed_precision::validate_manifest(
        &plan.manifest,
        &geometry,
        &checkpoint_sha256,
        &cache_namespace,
        &config_sha256,
    )?;
    plan.manifest.compact_cache = None;

    let reuse_verify_started = std::time::Instant::now();
    let mut reuse_sources = match reuse_manifests {
        Some(paths) => Some([
            ReuseSource::open(paths[0], &plan.manifest, &geometry, &ranking_raw)?,
            ReuseSource::open(paths[1], &plan.manifest, &geometry, &ranking_raw)?,
        ]),
        None => None,
    };
    let reuse_verification_seconds = reuse_verify_started.elapsed().as_secs_f64();
    if let Some(sources) = &reuse_sources {
        for source in sources {
            source.reject_output_aliases(&[
                output_cache,
                output_manifest,
                &partial_path,
                &progress_path,
            ])?;
        }
        for (index, region) in plan.manifest.regions.iter().enumerate() {
            let source = &sources[usize::from(region.bits == 8)].manifest.regions[index];
            if source.bits != region.bits {
                return Err(format!(
                    "reuse source lacks requested precision at record {index}"
                ));
            }
        }
    }

    if is_stacked_experts(&safetensors_index.weight_map)
        || is_separate_stacked_experts(&safetensors_index.weight_map)
        || is_mxfp4(&safetensors_index.weight_map)
        || is_prequantized(&safetensors_index.weight_map)
    {
        return Err("compact mixed cache preparation currently requires independently addressable source experts; this checkpoint layout is unsupported".to_string());
    }
    if !has_gate_proj_experts(&safetensors_index.weight_map) || !config.experts_gated {
        return Err(
            "compact mixed cache preparation currently requires gated routed experts".to_string(),
        );
    }

    let needed_layers: std::collections::HashSet<usize> = (0..config.num_moe_layers())
        .map(|index| config.moe_abs_layer(index))
        .collect();
    let mut shard_names = safetensors_index
        .weight_map
        .iter()
        .filter_map(|(tensor, shard)| {
            parse_layer_number(tensor)
                .filter(|layer| needed_layers.contains(layer))
                .map(|_| shard.clone())
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    shard_names.sort();
    let mut shards = HashMap::new();
    for shard_name in shard_names {
        shards.insert(
            shard_name.clone(),
            MmapSafetensors::open(&model_dir.join(&shard_name))
                .map_err(|error| format!("open checkpoint shard {shard_name}: {error}"))?,
        );
    }
    let layers_prefix = detect_expert_prefix(&safetensors_index.weight_map)?;
    let expert_sublayer = detect_expert_sublayer(&safetensors_index.weight_map);
    let deepseek_v4_fp4 = is_deepseek_v4_fp4(&safetensors_index.weight_map);
    let plan_sha256 = format!("{:x}", Sha256::digest(&plan_raw));
    let cache_already_published = output_cache.exists();
    if cache_already_published && (partial_path.exists() || !progress_path.exists()) {
        return Err(
            "published compact cache recovery requires its complete progress journal and no partial file"
                .to_string(),
        );
    }
    let mut progress = if progress_path.exists() {
        let (value, _): (BuildProgress, Vec<u8>) = read_canonical(&progress_path)?;
        value
    } else {
        BuildProgress {
            format: PROGRESS_FORMAT.to_string(),
            schema_version: PROGRESS_VERSION,
            plan_sha256: plan_sha256.clone(),
            output_basename: output_basename.to_string(),
            records: Vec::new(),
        }
    };
    let mut peak_progress_journal_bytes = canonical_json(&progress)?.len() as u64;
    let work_path = if cache_already_published {
        output_cache
    } else {
        &partial_path
    };
    let mut partial = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(work_path)
        .map_err(|error| format!("open {}: {error}", work_path.display()))?;
    // Shared source locks also reject hard-linked work files on Linux/Windows.
    // Lock before any header write or resume truncation; never wait on an alias.
    partial
        .try_lock()
        .map_err(|e| format!("lock compact cache output: {e}"))?;
    if progress.records.is_empty()
        && partial.metadata().map_err(|error| error.to_string())?.len() == 0
    {
        partial
            .write_all(&vec![
                0u8;
                mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES
            ])
            .and_then(|_| partial.sync_all())
            .map_err(|error| format!("initialize compact cache: {error}"))?;
        let progress_bytes = canonical_json(&progress)?;
        peak_progress_journal_bytes = peak_progress_journal_bytes.max(progress_bytes.len() as u64);
        atomic_write(&progress_path, &progress_bytes)?;
    }
    let mut append_offset = validate_resume(
        &mut partial,
        &progress,
        &plan_sha256,
        output_basename,
        &plan.manifest,
        &geometry,
    )?;
    crate::syscheck::log_memory_usage("before compact mixed cache build");
    let started = std::time::Instant::now();
    let start_index = progress.records.len();
    for index in start_index..plan.manifest.regions.len() {
        let region = &plan.manifest.regions[index];
        let record = if let Some(sources) = &mut reuse_sources {
            sources[usize::from(region.bits == 8)].record(index, region, &geometry)?
        } else {
            let layer_idx = usize::try_from(region.model_layer)
                .map_err(|_| "mixed region model layer exceeds usize".to_string())?;
            let expert_idx = usize::try_from(region.expert)
                .map_err(|_| "mixed region expert exceeds usize".to_string())?;
            let prefix = if deepseek_v4_fp4 {
                MarlinExpertNamespace::Main.routed_prefix(layer_idx, expert_idx)
            } else {
                format!("{layers_prefix}.layers.{layer_idx}.{expert_sublayer}.experts.{expert_idx}")
            };
            let (gate, up, down) = if deepseek_v4_fp4 {
                load_deepseek_v4_fp4_expert(
                    layer_idx,
                    expert_idx,
                    &prefix,
                    &safetensors_index.weight_map,
                    &shards,
                    geometry.group_size,
                    region.bits,
                    ExpertInt4CalibMode::SearchRmse,
                    Some(&calibration),
                )?
            } else {
                load_and_quantize_expert(
                    layer_idx,
                    expert_idx,
                    &prefix,
                    &safetensors_index.weight_map,
                    &shards,
                    geometry.group_size,
                    region.bits,
                    ExpertInt4CalibMode::SearchRmse,
                    Some(&calibration),
                )?
            };
            if let Some(ranking) = &ranking_json {
                if region.quality_gain_units.is_some() {
                    let candidate = &ranking["candidates"][index];
                    let key = if region.bits == 4 {
                        "int4_projection_sha256"
                    } else {
                        "int8_projection_sha256"
                    };
                    let expected = candidate[key]
                        .as_array()
                        .ok_or("measured ranking lacks materialized projection identities")?;
                    if expected.len() != 3 {
                        return Err("ranking projection identity count mismatch".to_string());
                    }
                    for (projection, weight) in [&gate, &up, &down].into_iter().enumerate() {
                        let values = match weight {
                            QuantWeight::Int4(q) => marlin::dequantize_int4(q),
                            QuantWeight::Int8(q) => marlin::dequantize_int8(q),
                            QuantWeight::Bf16(_) => {
                                return Err("mixed cache has BF16 routed projection".to_string())
                            }
                        };
                        let bytes: Vec<u8> = values
                            .iter()
                            .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
                            .collect();
                        if expected[projection].as_str()
                            != Some(mixed_precision::sha256_hex(&bytes).as_str())
                        {
                            return Err(format!("materialized projection differs from scored weights at record {index}, projection {projection}"));
                        }
                    }
                }
            }
            let marlin = UnifiedExpertWeights::from_expert_weights_marlin(
                &ExpertWeights { gate, up, down },
                region.bits,
            );
            let expected_bytes = region_bytes(region, &geometry);
            let mut record = Vec::with_capacity(expected_bytes);
            record.extend(
                marlin
                    .w13_packed
                    .iter()
                    .flat_map(|value| value.to_le_bytes()),
            );
            record.extend(
                marlin
                    .w13_scales
                    .iter()
                    .flat_map(|value| value.to_le_bytes()),
            );
            record.extend(
                marlin
                    .w2_packed
                    .iter()
                    .flat_map(|value| value.to_le_bytes()),
            );
            record.extend(
                marlin
                    .w2_scales
                    .iter()
                    .flat_map(|value| value.to_le_bytes()),
            );
            record
        };
        let expected_bytes = region_bytes(region, &geometry);
        if record.len() != expected_bytes {
            return Err(format!(
                "compact mixed cache L{}E{} record size {} != expected {expected_bytes}",
                region.model_layer,
                region.expert,
                record.len()
            ));
        }
        let record_sha256 = format!("{:x}", Sha256::digest(&record));
        partial
            .seek(SeekFrom::Start(append_offset))
            .and_then(|_| partial.write_all(&record))
            .and_then(|_| partial.sync_data())
            .map_err(|error| format!("write compact mixed cache record {index}: {error}"))?;
        progress.records.push(CompletedRecord {
            index,
            offset: append_offset,
            bytes: record.len() as u64,
            sha256: record_sha256,
        });
        append_offset += record.len() as u64;
        let progress_bytes = canonical_json(&progress)?;
        peak_progress_journal_bytes = peak_progress_journal_bytes.max(progress_bytes.len() as u64);
        atomic_write(&progress_path, &progress_bytes)?;
        if (index + 1) % config.n_routed_experts == 0 || index + 1 == plan.manifest.regions.len() {
            crate::syscheck::log_memory_usage(&format!(
                "compact mixed cache records {}/{}",
                index + 1,
                plan.manifest.regions.len()
            ));
            eprintln!(
                "    Compact mixed cache: {}/{} experts ({:.0}s elapsed)",
                index + 1,
                plan.manifest.regions.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }

    if let Some(sources) = &reuse_sources {
        for source in sources {
            source.check_unchanged()?;
        }
    }

    let payload_bytes = append_offset - mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64;
    let payload_sha256 = sha256_file_range(
        &mut partial,
        mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64,
        payload_bytes,
    )?;
    let region_map_sha256 = mixed_precision::region_map_sha256(&plan.manifest.regions)?;
    let mut header = [0u8; mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES];
    header[..4].copy_from_slice(mixed_precision::MIXED_COMPACT_CACHE_MAGIC);
    header[4..8].copy_from_slice(&mixed_precision::MIXED_COMPACT_CACHE_VERSION.to_le_bytes());
    header[8..16]
        .copy_from_slice(&(mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64).to_le_bytes());
    header[16..24].copy_from_slice(&payload_bytes.to_le_bytes());
    header[24..56].copy_from_slice(&decode_sha256(&payload_sha256)?);
    header[56..88].copy_from_slice(&decode_sha256(&region_map_sha256)?);
    header[88..120].copy_from_slice(&decode_sha256(&checkpoint_sha256)?);
    partial
        .seek(SeekFrom::Start(0))
        .and_then(|_| partial.write_all(&header))
        .and_then(|_| partial.sync_all())
        .map_err(|error| format!("finalize compact mixed cache header: {error}"))?;
    let final_bytes = partial.metadata().map_err(|error| error.to_string())?.len();
    let full_sha256 = sha256_file_range(&mut partial, 0, final_bytes)?;
    let header_sha256 = format!("{:x}", Sha256::digest(header));
    drop(partial);
    if let Some(sources) = &reuse_sources {
        for source in sources {
            source.check_final()?;
        }
    }
    if !cache_already_published {
        std::fs::rename(&partial_path, output_cache)
            .map_err(|error| format!("publish {}: {error}", output_cache.display()))?;
    }
    File::open(output_dir)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("sync {}: {error}", output_dir.display()))?;

    plan.manifest.compact_cache = Some(mixed_precision::MixedCompactCacheIdentity {
        basename: output_basename.to_string(),
        bytes: final_bytes,
        sha256: full_sha256,
        header_sha256,
        payload_sha256,
        region_map_sha256,
    });
    mixed_precision::validate_manifest(
        &plan.manifest,
        &geometry,
        &checkpoint_sha256,
        &cache_namespace,
        &config_sha256,
    )?;
    atomic_write(output_manifest, &canonical_json(&plan.manifest)?)?;
    std::fs::remove_file(&progress_path).map_err(|error| {
        format!(
            "remove completed build journal {}: {error}",
            progress_path.display()
        )
    })?;
    let memory_end = system_memory_snapshot();
    let report = serde_json::json!({
        "cache": output_cache,
        "manifest": output_manifest,
        "records": plan.manifest.regions.len(),
        "bytes": final_bytes,
        "payload_bytes": payload_bytes,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "total_elapsed_seconds": prepare_started.elapsed().as_secs_f64(),
        "reuse_verification_seconds": reuse_verification_seconds,
        "memory_start": memory_start,
        "memory_end": memory_end,
        "process_vm_hwm_bytes": memory_hwm_bytes(),
        "peak_progress_journal_bytes": peak_progress_journal_bytes,
        "peak_output_plus_journal_bytes": final_bytes.saturating_add(peak_progress_journal_bytes),
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "reuse_sources": reuse_sources.as_ref().map(|sources| sources.iter().map(|source| serde_json::json!({
            "manifest": source.manifest_path,
            "manifest_sha256": mixed_precision::sha256_hex(&source.manifest_raw),
            "cache_sha256": source.manifest.compact_cache.as_ref().unwrap().sha256,
        })).collect::<Vec<_>>()),
    });
    Ok(serde_json::to_string(&report).map_err(|error| error.to_string())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_manifest() -> mixed_precision::MixedPrecisionManifest {
        let capture_hashes = vec!["33".repeat(32)];
        mixed_precision::MixedPrecisionManifest {
            scales_only: None,
            format: mixed_precision::MIXED_MANIFEST_FORMAT.to_string(),
            schema_version: mixed_precision::MIXED_MANIFEST_VERSION,
            checkpoint_sha256: "11".repeat(32),
            cache_namespace: "fixture--identity".to_string(),
            config_sha256: "22".repeat(32),
            group_size: 8,
            hidden_size: 8,
            intermediate_size: 8,
            routed_layers: 1,
            routed_experts_per_layer: 2,
            experts_gated: true,
            default_bits: 4,
            source_int4: None,
            source_int8: None,
            compact_cache: None,
            calibration: mixed_precision::MixedCalibrationIdentity {
                artifact_sha256: mixed_precision::sha256_hex(
                    &serde_json::to_vec(&capture_hashes).unwrap(),
                ),
                ranking_evidence_sha256: "44".repeat(32),
                ranking_method: mixed_precision::MIXED_RANKING_EXPECTED_V2.to_string(),
                quality_gain_scale_exponent: -12,
                corpus_sha256: vec!["55".repeat(32)],
                capture_artifact_sha256: capture_hashes,
                excluded_input_set_sha256: "66".repeat(32),
                search_rmse_calibration_sha256: Some("77".repeat(32)),
                coverage_policy: None,
            },
            budget: mixed_precision::MixedBudget {
                requested_basis_points: 10_000,
                baseline_int4_routed_bytes: 24,
                maximum_added_bytes: 24,
                achieved_added_bytes: 24,
                achieved_percent_millionths: 100_000_000,
            },
            regions: vec![
                mixed_precision::MixedRegion {
                    model_layer: 3,
                    moe_layer: 0,
                    expert: 0,
                    bits: 4,
                    int4_bytes: 12,
                    int8_bytes: 24,
                    quality_gain_units: Some(2),
                    quality_rank: Some(0),
                    coverage: None,
                },
                mixed_precision::MixedRegion {
                    model_layer: 3,
                    moe_layer: 0,
                    expert: 1,
                    bits: 8,
                    int4_bytes: 12,
                    int8_bytes: 24,
                    quality_gain_units: Some(1),
                    quality_rank: Some(1),
                    coverage: None,
                },
            ],
        }
    }

    fn fixture_geometry() -> mixed_precision::MixedPrecisionGeometry {
        mixed_precision::MixedPrecisionGeometry {
            group_size: 8,
            hidden_size: 8,
            intermediate_size: 8,
            routed_layers: 1,
            routed_experts_per_layer: 2,
            experts_gated: true,
            model_layers: vec![3],
            int4_expert_bytes: 12,
            int8_expert_bytes: 24,
        }
    }

    fn reuse_fixture() -> (
        mixed_precision::MixedPrecisionManifest,
        mixed_precision::MixedPrecisionGeometry,
        Vec<u8>,
    ) {
        let mut manifest = fixture_manifest();
        manifest.schema_version = mixed_precision::MIXED_COVERAGE_MANIFEST_VERSION;
        manifest.routed_experts_per_layer = 20;
        manifest.calibration.ranking_method =
            mixed_precision::MIXED_RANKING_EXPECTED_EXPOSURE_V3.into();
        manifest.calibration.coverage_policy = Some(mixed_precision::MixedCoveragePolicy {
            method: "underobserved_int8_v1".into(),
            minimum_routes_per_expert: 500,
            required_corpora: 1,
        });
        let template = manifest.regions[0].clone();
        manifest.regions = (0..20)
            .map(|expert| {
                let mut region = template.clone();
                region.expert = expert;
                let required = expert == 19;
                region.bits = if required { 8 } else { 4 };
                region.quality_gain_units = (!required).then_some(i64::from(20 - expert));
                region.quality_rank = (!required).then_some(expert);
                region.coverage = Some(mixed_precision::MixedRegionCoverage {
                    route_count: if required { 1 } else { 500 },
                    corpus_route_counts: vec![if required { 1 } else { 500 }],
                    selection_reason: if required {
                        "coverage_required_int8"
                    } else {
                        "measured_int4"
                    }
                    .into(),
                });
                region
            })
            .collect();
        manifest.budget = mixed_precision::MixedBudget {
            requested_basis_points: 0,
            baseline_int4_routed_bytes: 240,
            maximum_added_bytes: 12,
            achieved_added_bytes: 12,
            achieved_percent_millionths: 5_000_000,
        };
        let candidates: Vec<_> = manifest.regions.iter().map(|region| {
            let coverage = region.coverage.as_ref().unwrap();
            let mut candidate = serde_json::json!({"model_layer":3,"moe_layer":0,"expert":region.expert,
                "int4_bytes":12,"int8_bytes":24,"route_count":coverage.route_count,
                "corpus_route_counts":coverage.corpus_route_counts,"coverage_required_int8":region.expert==19});
            if let Some(gain) = region.quality_gain_units {
                candidate["expected_gain_units"] = gain.into();
                candidate["expected_rank"] = region.quality_rank.unwrap().into();
            }
            candidate
        }).collect();
        let ranking = serde_json::to_vec(
            &serde_json::json!({"coverage_policy":"underobserved_int8_v1",
            "minimum_routes_per_expert":500,"tail_fraction":0.01,"minimum_tail_observations":5,
            "search_rmse_calibration_sha256":manifest.calibration.search_rmse_calibration_sha256,
            "candidates":candidates}),
        )
        .unwrap();
        manifest.calibration.ranking_evidence_sha256 = mixed_precision::sha256_hex(&ranking);
        let mut geometry = fixture_geometry();
        geometry.routed_experts_per_layer = 20;
        (manifest, geometry, ranking)
    }

    fn write_reuse_fixture(
        dir: &Path,
        name: &str,
        mut manifest: mixed_precision::MixedPrecisionManifest,
        geometry: &mixed_precision::MixedPrecisionGeometry,
        full_int8: bool,
    ) -> std::path::PathBuf {
        if full_int8 {
            for region in &mut manifest.regions {
                region.bits = 8;
                if region.quality_gain_units.is_some() {
                    region.coverage.as_mut().unwrap().selection_reason =
                        "measured_promotion".into();
                }
            }
            manifest.budget.requested_basis_points = 10_000;
            manifest.budget.maximum_added_bytes = 240;
            manifest.budget.achieved_added_bytes = 240;
            manifest.budget.achieved_percent_millionths = 100_000_000;
        }
        let payload: Vec<u8> = manifest
            .regions
            .iter()
            .flat_map(|region| {
                vec![
                    (if region.bits == 4 { 0x20 } else { 0x80 }) + region.expert as u8;
                    region_bytes(region, geometry)
                ]
            })
            .collect();
        let payload_sha = mixed_precision::sha256_hex(&payload);
        let region_sha = mixed_precision::region_map_sha256(&manifest.regions).unwrap();
        let mut header = [0u8; mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES];
        header[..4].copy_from_slice(mixed_precision::MIXED_COMPACT_CACHE_MAGIC);
        header[4..8].copy_from_slice(&mixed_precision::MIXED_COMPACT_CACHE_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(
            &(mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64).to_le_bytes(),
        );
        header[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        header[24..56].copy_from_slice(&decode_sha256(&payload_sha).unwrap());
        header[56..88].copy_from_slice(&decode_sha256(&region_sha).unwrap());
        header[88..120].copy_from_slice(&decode_sha256(&manifest.checkpoint_sha256).unwrap());
        let header_sha = mixed_precision::sha256_hex(&header);
        let mut raw = header.to_vec();
        raw.extend(&payload);
        let basename = format!("{name}.krmx");
        std::fs::write(dir.join(&basename), &raw).unwrap();
        manifest.compact_cache = Some(mixed_precision::MixedCompactCacheIdentity {
            basename,
            bytes: raw.len() as u64,
            sha256: mixed_precision::sha256_hex(&raw),
            header_sha256: header_sha,
            payload_sha256: payload_sha,
            region_map_sha256: region_sha,
        });
        let path = dir.join(format!("{name}.json"));
        std::fs::write(&path, canonical_json(&manifest).unwrap()).unwrap();
        path
    }

    #[test]
    fn reuse_preserves_selected_bytes_including_protected_experts() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("krasis-reuse-{nonce}"));
        std::fs::create_dir(&dir).unwrap();
        let (mut target, geometry, ranking) = reuse_fixture();
        let low = write_reuse_fixture(&dir, "low", target.clone(), &geometry, false);
        let high = write_reuse_fixture(&dir, "high", target.clone(), &geometry, true);
        target.regions[0].bits = 8;
        target.regions[0]
            .coverage
            .as_mut()
            .unwrap()
            .selection_reason = "measured_promotion".into();
        target.budget.requested_basis_points = 1000;
        target.budget.maximum_added_bytes = 24;
        target.budget.achieved_added_bytes = 24;
        target.budget.achieved_percent_millionths = 10_000_000;
        let mut sources = [
            ReuseSource::open(&low, &target, &geometry, &ranking).unwrap(),
            ReuseSource::open(&high, &target, &geometry, &ranking).unwrap(),
        ];
        let mut actual = Vec::new();
        for (index, region) in target.regions.iter().enumerate() {
            actual.extend(
                sources[usize::from(region.bits == 8)]
                    .record(index, region, &geometry)
                    .unwrap(),
            );
        }
        let expected: Vec<u8> = target
            .regions
            .iter()
            .flat_map(|region| {
                vec![
                    (if region.bits == 4 { 0x20 } else { 0x80 }) + region.expert as u8;
                    region_bytes(region, &geometry)
                ]
            })
            .collect();
        assert_eq!(actual, expected);
        assert!(sources[0]
            .record(0, &target.regions[0], &geometry)
            .unwrap_err()
            .contains("requested precision"));
        let mut other = target.clone();
        other.calibration.search_rmse_calibration_sha256 = Some("88".repeat(32));
        assert!(ReuseSource::open(&low, &other, &geometry, &ranking)
            .err()
            .unwrap()
            .contains("calibration/ranking"));
        sources[0].check_final().unwrap();
        sources[1].check_final().unwrap();
        drop(sources);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reuse_rejects_source_aliases_before_output_mutation() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("krasis-reuse-alias-{nonce}"));
        std::fs::create_dir(&dir).unwrap();
        let (target, geometry, ranking) = reuse_fixture();
        let manifest = write_reuse_fixture(&dir, "low", target.clone(), &geometry, false);
        let source = ReuseSource::open(&manifest, &target, &geometry, &ranking).unwrap();
        assert!(source
            .reject_output_aliases(&[&dir.join("low.krmx")])
            .is_err());
        assert!(source.reject_output_aliases(&[&manifest]).is_err());
        assert!(source
            .reject_output_aliases(&[&dir.join("new.krmx")])
            .is_ok());
        let alias = dir.join(".new.krmx.partial");
        std::fs::hard_link(&source.path, &alias).unwrap();
        let original = std::fs::read(&source.path).unwrap();
        let writer = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&alias)
            .unwrap();
        assert!(writer.try_lock().is_err());
        assert_eq!(std::fs::read(&source.path).unwrap(), original);
        drop(writer);
        #[cfg(unix)]
        {
            let symlink = dir.join(".other.krmx.partial");
            std::os::unix::fs::symlink(&source.path, &symlink).unwrap();
            assert!(source.reject_output_aliases(&[&symlink]).is_err());
        }
        drop(source);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reuse_rejects_corruption_and_changed_sources() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("krasis-reuse-corrupt-{nonce}"));
        std::fs::create_dir(&dir).unwrap();
        let (target, geometry, ranking) = reuse_fixture();
        let path = write_reuse_fixture(&dir, "low", target.clone(), &geometry, false);
        let source = ReuseSource::open(&path, &target, &geometry, &ranking).unwrap();
        let cache = dir.join("low.krmx");
        let mut raw = std::fs::read(&cache).unwrap();
        let original = raw.clone();
        *raw.last_mut().unwrap() ^= 1;
        if std::fs::write(&cache, &raw).is_ok() {
            assert!(source.check_unchanged().unwrap_err().contains("changed"));
        } else if source.check_unchanged().is_ok() {
            // Mandatory source locks may reject the attempted write outright.
            assert_eq!(std::fs::read(&cache).unwrap(), original);
        }
        drop(source);
        std::fs::write(&cache, &raw).unwrap();
        assert!(ReuseSource::open(&path, &target, &geometry, &ranking)
            .err()
            .unwrap()
            .contains("identity mismatch"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_rehashes_committed_records_and_discards_only_uncommitted_tail() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("krasis-mixed-resume-{nonce}.bin"));
        let mut bytes = vec![0u8; mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES];
        bytes.extend_from_slice(&[9u8; 12]);
        bytes.extend_from_slice(&[8u8; 7]);
        std::fs::write(&path, bytes).unwrap();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let progress = BuildProgress {
            format: PROGRESS_FORMAT.to_string(),
            schema_version: PROGRESS_VERSION,
            plan_sha256: "aa".repeat(32),
            output_basename: "fixture.krmx".to_string(),
            records: vec![CompletedRecord {
                index: 0,
                offset: mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64,
                bytes: 12,
                sha256: format!("{:x}", Sha256::digest([9u8; 12])),
            }],
        };
        let offset = validate_resume(
            &mut file,
            &progress,
            &"aa".repeat(32),
            "fixture.krmx",
            &fixture_manifest(),
            &fixture_geometry(),
        )
        .unwrap();
        assert_eq!(
            offset,
            (mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES + 12) as u64
        );
        assert_eq!(file.metadata().unwrap().len(), offset);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn resume_rejects_corrupt_committed_record() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("krasis-mixed-corrupt-{nonce}.bin"));
        let mut bytes = vec![0u8; mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES];
        bytes.extend_from_slice(&[9u8; 12]);
        std::fs::write(&path, bytes).unwrap();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let progress = BuildProgress {
            format: PROGRESS_FORMAT.to_string(),
            schema_version: PROGRESS_VERSION,
            plan_sha256: "aa".repeat(32),
            output_basename: "fixture.krmx".to_string(),
            records: vec![CompletedRecord {
                index: 0,
                offset: mixed_precision::MIXED_COMPACT_CACHE_HEADER_BYTES as u64,
                bytes: 12,
                sha256: "00".repeat(32),
            }],
        };
        assert!(validate_resume(
            &mut file,
            &progress,
            &"aa".repeat(32),
            "fixture.krmx",
            &fixture_manifest(),
            &fixture_geometry(),
        )
        .unwrap_err()
        .contains("failed resume verification"));
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
