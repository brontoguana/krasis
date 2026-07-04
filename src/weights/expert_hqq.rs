//! Expert-HQQ cache metadata and fail-closed read/write plumbing.
//!
//! This module deliberately does not register runtime kernels or production
//! dispatch paths. It defines the standalone cache/header contract needed
//! before expert-HQQ W13/W2 execution can be added in a later gate.

use crate::weights::marlin::{bf16_to_f32, f32_to_bf16};
use crate::weights::safetensors_io::{Dtype, MmapSafetensors};

#[cfg(all(test, has_prefill_kernels))]
use cudarc::driver::{sys as cuda_sys, CudaDevice, CudaFunction, CudaSlice, DevicePtr};
use half::f16;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
#[cfg(all(test, has_prefill_kernels))]
use std::sync::Arc;
use std::time::Instant;

use super::ModelConfig;

pub const EXPERT_HQQ_CACHE_MAGIC: &[u8; 4] = b"KRHQ";
pub const EXPERT_HQQ_CACHE_VERSION: u32 = 1;
pub const EXPERT_HQQ_HEADER_SIZE: usize = 64;
pub const EXPERT_HQQ_TENSOR_DESCRIPTOR_SIZE: usize = 88;
pub const EXPERT_HQQ_AXIS: usize = 1;
pub const EXPERT_HQQ_PACKED_DTYPE: &str = "uint8";
pub const EXPERT_HQQ_SCALES_DTYPE: &str = "float32";
pub const EXPERT_HQQ_ZEROS_DTYPE: &str = "float32";
const EXPERT_HQQ_GENERATION_HQQ6_GROUP_SIZES: &[usize] = &[16, 32, 64];
const EXPERT_HQQ_GENERATION_HQQ8_GROUP_SIZES: &[usize] = &[64];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertHqqTensorRole {
    W13,
    W2,
}

impl ExpertHqqTensorRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::W13 => "w13",
            Self::W2 => "w2",
        }
    }

    fn tag(self) -> u32 {
        match self {
            Self::W13 => 1,
            Self::W2 => 2,
        }
    }

    fn from_tag(tag: u32) -> Result<Self, String> {
        match tag {
            1 => Ok(Self::W13),
            2 => Ok(Self::W2),
            other => Err(format!("Unsupported expert-HQQ tensor role tag {other}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqCacheHeader {
    pub version: u32,
    pub hidden_size: usize,
    pub routed_hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_moe_layers: usize,
    pub config_hash: u64,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqCacheExpectation {
    pub hidden_size: usize,
    pub routed_hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_moe_layers: usize,
    pub config_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqTensorDescriptor {
    pub role: ExpertHqqTensorRole,
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub nbits: u8,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub axis: usize,
    pub layout: String,
    pub packed_dtype: String,
    pub scales_dtype: String,
    pub zeros_dtype: String,
    pub packed_bytes: usize,
    pub scales_bytes: usize,
    pub zeros_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertHqqTensorKey {
    pub role: ExpertHqqTensorRole,
    pub layer_idx: usize,
    pub expert_idx: usize,
}

impl ExpertHqqTensorKey {
    pub fn new(role: ExpertHqqTensorRole, layer_idx: usize, expert_idx: usize) -> Self {
        Self {
            role,
            layer_idx,
            expert_idx,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqTensorRecord {
    pub descriptor: ExpertHqqTensorDescriptor,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub zeros: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqTensorInput {
    pub descriptor: ExpertHqqTensorDescriptor,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub zeros: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertHqqRuntimeDiagnosticModelShape {
    pub hidden_size: usize,
    pub routed_hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_hidden_layers: usize,
    pub experts_gated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertHqqRuntimeDiagnosticRequirement {
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub nbits: u8,
    pub group_size: usize,
}

impl ExpertHqqRuntimeDiagnosticRequirement {
    pub fn new(layer_idx: usize, expert_idx: usize, nbits: u8, group_size: usize) -> Self {
        Self {
            layer_idx,
            expert_idx,
            nbits,
            group_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqRuntimeDiagnosticTensorSummary {
    pub role: ExpertHqqTensorRole,
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub nbits: u8,
    pub group_size: usize,
    pub axis: usize,
    pub layout: String,
    pub rows: usize,
    pub cols: usize,
    pub packed_bytes: usize,
    pub scales_bytes: usize,
    pub zeros_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqRuntimeDiagnosticReport {
    pub checked_experts: usize,
    pub tensor_records: usize,
    pub total_payload_bytes: usize,
    pub tensors: Vec<ExpertHqqRuntimeDiagnosticTensorSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqSafetensorsTensorSpec {
    pub path: PathBuf,
    pub key: String,
    pub role: ExpertHqqTensorRole,
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub nbits: u8,
    pub group_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqCacheGenerationPlan {
    pub manifest_path: PathBuf,
    pub model_dir: PathBuf,
    pub output_cache_path: PathBuf,
    pub diagnostic_spec_path: PathBuf,
    pub header: ExpertHqqCacheHeader,
    pub layer_idx: usize,
    pub layers: Vec<usize>,
    pub experts: Vec<usize>,
    pub nbits: u8,
    pub group_size: usize,
    pub layout: String,
    pub specs: Vec<ExpertHqqSafetensorsTensorSpec>,
    pub required_tensors: Vec<ExpertHqqTensorKey>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqCacheGenerationReport {
    pub manifest_path: PathBuf,
    pub cache_path: PathBuf,
    pub diagnostic_spec_path: PathBuf,
    pub layer_idx: usize,
    pub layers: Vec<usize>,
    pub expert_count: usize,
    pub tensor_records: usize,
    pub total_payload_bytes: usize,
    pub cache_file_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertHqqPrefillWork {
    pub expert_idx: usize,
    pub row_offset: usize,
    pub row_count: usize,
}

impl ExpertHqqPrefillWork {
    pub fn new(expert_idx: usize, row_offset: usize, row_count: usize) -> Self {
        Self {
            expert_idx,
            row_offset,
            row_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqPrefillDispatchEntry {
    pub expert_idx: usize,
    pub row_offset: usize,
    pub row_count: usize,
    pub w13_key: ExpertHqqTensorKey,
    pub w2_key: ExpertHqqTensorKey,
    pub w13_rows: usize,
    pub w13_cols: usize,
    pub w2_rows: usize,
    pub w2_cols: usize,
    pub w13_nbits: u8,
    pub w2_nbits: u8,
    pub w13_group_size: usize,
    pub w2_group_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqPrefillDispatchPlan {
    pub layer_idx: usize,
    pub experts_gated: bool,
    pub input_layout: &'static str,
    pub w13_dequant_layout: &'static str,
    pub w13_output_layout: &'static str,
    pub activation_output_layout: &'static str,
    pub w2_dequant_layout: &'static str,
    pub w2_output_layout: &'static str,
    pub entries: Vec<ExpertHqqPrefillDispatchEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertHqqPrefillReferenceOutput {
    pub sorted_row_count: usize,
    pub routed_hidden_size: usize,
    pub w13_rows: usize,
    pub moe_intermediate_size: usize,
    pub w13_preactivation: Vec<f32>,
    pub activation: Vec<f32>,
    pub values: Vec<f32>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertHqqPrefillTestDispatchOutput {
    pub sorted_row_count: usize,
    pub routed_hidden_size: usize,
    pub values: Vec<f32>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertHqqPrefillGpuPrototypeOutput {
    pub sorted_row_count: usize,
    pub routed_hidden_size: usize,
    pub w13_rows: usize,
    pub moe_intermediate_size: usize,
    pub w13_preactivation: Vec<f32>,
    pub activation: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertHqqBf16PathOracleMetadata {
    pub sorted_row_count: usize,
    pub routed_hidden_size: usize,
    pub w13_rows: usize,
    pub moe_intermediate_size: usize,
    pub input_bf16_values: usize,
    pub w13_preactivation_values: usize,
    pub activation_values: usize,
    pub output_values: usize,
    pub input_layout: &'static str,
    pub w13_output_layout: &'static str,
    pub activation_output_layout: &'static str,
    pub w2_output_layout: &'static str,
    pub correctness_oracle: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertHqqPrefillBf16PathOracleOutput {
    pub sorted_row_count: usize,
    pub routed_hidden_size: usize,
    pub w13_rows: usize,
    pub moe_intermediate_size: usize,
    pub input_bf16: Vec<f32>,
    pub w13_preactivation: Vec<f32>,
    pub activation: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertHqqRuntimePrefillBlock {
    pub expert_idx: usize,
    pub absolute_row_offset: usize,
    pub row_count: usize,
}

impl ExpertHqqRuntimePrefillBlock {
    pub fn new(expert_idx: usize, absolute_row_offset: usize, row_count: usize) -> Self {
        Self {
            expert_idx,
            absolute_row_offset,
            row_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertHqqRuntimePrefillBufferShape {
    pub total_sorted_rows: usize,
    pub input_row_stride: usize,
    pub w13_row_stride: usize,
    pub activation_row_stride: usize,
    pub output_row_stride: usize,
}

impl ExpertHqqRuntimePrefillBufferShape {
    pub fn contiguous_for_cache(
        cache: &ExpertHqqCache,
        experts_gated: bool,
        total_sorted_rows: usize,
    ) -> Result<Self, String> {
        let w13_rows = if experts_gated {
            checked_mul(
                2,
                cache.header.moe_intermediate_size,
                "runtime-shaped gated W13 rows",
            )?
        } else {
            cache.header.moe_intermediate_size
        };
        Ok(Self {
            total_sorted_rows,
            input_row_stride: cache.header.routed_hidden_size,
            w13_row_stride: w13_rows,
            activation_row_stride: cache.header.moe_intermediate_size,
            output_row_stride: cache.header.routed_hidden_size,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertHqqRuntimePrefillBufferLengths {
    pub input_values: usize,
    pub w13_values: usize,
    pub activation_values: usize,
    pub output_values: usize,
}

impl ExpertHqqRuntimePrefillBufferLengths {
    pub fn required(
        model: ExpertHqqRuntimeDiagnosticModelShape,
        shape: ExpertHqqRuntimePrefillBufferShape,
    ) -> Result<Self, String> {
        let w13_rows = expected_runtime_w13_rows(model.experts_gated, model.moe_intermediate_size)?;
        Ok(Self {
            input_values: runtime_buffer_len(
                shape.total_sorted_rows,
                shape.input_row_stride,
                model.routed_hidden_size,
            )?,
            w13_values: runtime_buffer_len(
                shape.total_sorted_rows,
                shape.w13_row_stride,
                w13_rows,
            )?,
            activation_values: runtime_buffer_len(
                shape.total_sorted_rows,
                shape.activation_row_stride,
                model.moe_intermediate_size,
            )?,
            output_values: runtime_buffer_len(
                shape.total_sorted_rows,
                shape.output_row_stride,
                model.routed_hidden_size,
            )?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqRuntimePrefillDiagnosticReport {
    pub layer_idx: usize,
    pub experts_gated: bool,
    pub nbits: u8,
    pub group_size: usize,
    pub total_sorted_rows: usize,
    pub claimed_rows: usize,
    pub padding_rows: usize,
    pub plan_entries: usize,
    pub input_row_stride: usize,
    pub w13_row_stride: usize,
    pub activation_row_stride: usize,
    pub output_row_stride: usize,
    pub buffer_lengths: ExpertHqqRuntimePrefillBufferLengths,
    pub oracle: ExpertHqqBf16PathOracleMetadata,
    pub availability: ExpertHqqRuntimeDiagnosticReport,
}

#[cfg(all(test, has_prefill_kernels))]
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertHqqRuntimeShapedPrefillGpuOutput {
    pub total_sorted_rows: usize,
    pub compact_row_count: usize,
    pub routed_hidden_size: usize,
    pub w13_rows: usize,
    pub moe_intermediate_size: usize,
    pub input_row_stride: usize,
    pub w13_row_stride: usize,
    pub activation_row_stride: usize,
    pub output_row_stride: usize,
    pub claimed_rows: Vec<bool>,
    pub w13_preactivation: Vec<f32>,
    pub activation: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqCache {
    pub header: ExpertHqqCacheHeader,
    pub tensors: Vec<ExpertHqqTensorRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertHqqDiagnosticCacheSpec {
    pub spec_path: PathBuf,
    pub cache_path: PathBuf,
    pub requirements: Vec<ExpertHqqDiagnosticCacheRequirement>,
    pub required_tensors: Vec<ExpertHqqTensorKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertHqqDiagnosticCacheRequirement {
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub nbits: u8,
    pub group_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertHqqDiagnosticCacheSpecJson {
    purpose: String,
    cache_path: String,
    requirements: Vec<ExpertHqqDiagnosticCacheRequirementJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertHqqDiagnosticCacheRequirementJson {
    layer_idx: usize,
    experts: Vec<usize>,
    roles: Vec<String>,
    nbits: u8,
    group_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertHqqCacheGenerationManifestJson {
    purpose: String,
    model_dir: String,
    config_hash: String,
    #[serde(default)]
    layer_idx: Option<usize>,
    #[serde(default)]
    layers: Option<Vec<usize>>,
    experts: Vec<usize>,
    roles: Vec<String>,
    nbits: u8,
    group_size: usize,
    axis: usize,
    layout: String,
    output_cache_path: String,
    diagnostic_spec_path: String,
    tensors: Vec<ExpertHqqCacheGenerationTensorJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertHqqCacheGenerationTensorJson {
    layer_idx: usize,
    expert_idx: usize,
    role: String,
    tensor_key: String,
    shard_path: String,
    expected_rows: usize,
    expected_cols: usize,
}

pub fn expert_hqq_config_hash(config_json_bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in config_json_bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

pub fn load_expert_hqq_diagnostic_cache_spec(
    spec_path: &Path,
) -> Result<ExpertHqqDiagnosticCacheSpec, String> {
    let spec_path = std::fs::canonicalize(spec_path).map_err(|e| {
        format!(
            "failed to resolve expert-HQQ diagnostic cache spec {}: {e}",
            spec_path.display()
        )
    })?;
    let raw = std::fs::read_to_string(&spec_path).map_err(|e| {
        format!(
            "failed to read expert-HQQ diagnostic cache spec {}: {e}",
            spec_path.display()
        )
    })?;
    let parsed: ExpertHqqDiagnosticCacheSpecJson = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "malformed expert-HQQ diagnostic cache spec {}: {e}",
            spec_path.display()
        )
    })?;
    if parsed.purpose != "runtime_prefill_diagnostic" {
        return Err(format!(
            "expert-HQQ diagnostic cache spec purpose must be runtime_prefill_diagnostic, got {:?}",
            parsed.purpose
        ));
    }
    let cache_raw = parsed.cache_path.trim();
    if cache_raw.is_empty() {
        return Err("expert-HQQ diagnostic cache spec cache_path must be non-empty".to_string());
    }
    let cache_candidate = PathBuf::from(cache_raw);
    let cache_path = if cache_candidate.is_absolute() {
        cache_candidate
    } else {
        spec_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(cache_candidate)
    };
    let cache_path = std::fs::canonicalize(&cache_path).map_err(|e| {
        format!(
            "failed to resolve expert-HQQ diagnostic cache_path {} from spec {}: {e}",
            cache_path.display(),
            spec_path.display()
        )
    })?;
    if !cache_path.is_file() {
        return Err(format!(
            "expert-HQQ diagnostic cache_path is not a file: {}",
            cache_path.display()
        ));
    }
    if parsed.requirements.is_empty() {
        return Err("expert-HQQ diagnostic cache spec requirements must be non-empty".to_string());
    }

    let mut seen_keys = HashSet::new();
    let mut requirements = Vec::new();
    let mut required_tensors = Vec::new();
    for (req_idx, req) in parsed.requirements.iter().enumerate() {
        if req.experts.is_empty() {
            return Err(format!(
                "expert-HQQ diagnostic cache spec requirements[{req_idx}].experts must be non-empty"
            ));
        }
        if req.group_size == 0 {
            return Err(format!(
                "expert-HQQ diagnostic cache spec requirements[{req_idx}].group_size must be > 0"
            ));
        }
        expert_hqq_layout_for_nbits(req.nbits).map_err(|e| {
            format!("expert-HQQ diagnostic cache spec requirements[{req_idx}] invalid nbits: {e}")
        })?;
        let mut roles = HashSet::new();
        for role in &req.roles {
            let parsed_role = match role.as_str() {
                "w13" => ExpertHqqTensorRole::W13,
                "w2" => ExpertHqqTensorRole::W2,
                other => {
                    return Err(format!(
                        "expert-HQQ diagnostic cache spec requirements[{req_idx}] has invalid role {other:?}; expected w13 and w2"
                    ));
                }
            };
            if !roles.insert(parsed_role) {
                return Err(format!(
                    "expert-HQQ diagnostic cache spec requirements[{req_idx}] duplicates role {}",
                    parsed_role.as_str()
                ));
            }
        }
        if roles.len() != 2
            || !roles.contains(&ExpertHqqTensorRole::W13)
            || !roles.contains(&ExpertHqqTensorRole::W2)
        {
            return Err(format!(
                "expert-HQQ diagnostic cache spec requirements[{req_idx}].roles must contain exactly w13 and w2"
            ));
        }

        for &expert_idx in &req.experts {
            let requirement = ExpertHqqDiagnosticCacheRequirement {
                layer_idx: req.layer_idx,
                expert_idx,
                nbits: req.nbits,
                group_size: req.group_size,
            };
            requirements.push(requirement);
            for role in [ExpertHqqTensorRole::W13, ExpertHqqTensorRole::W2] {
                let key = ExpertHqqTensorKey::new(role, req.layer_idx, expert_idx);
                if !seen_keys.insert(key) {
                    return Err(format!(
                        "duplicate expert-HQQ diagnostic cache spec requirement for layer={} expert={} role={}",
                        key.layer_idx,
                        key.expert_idx,
                        key.role.as_str()
                    ));
                }
                required_tensors.push(key);
            }
        }
    }

    Ok(ExpertHqqDiagnosticCacheSpec {
        spec_path,
        cache_path,
        requirements,
        required_tensors,
    })
}

pub fn plan_expert_hqq_cache_generation_from_manifest_path(
    manifest_path: &Path,
) -> Result<ExpertHqqCacheGenerationPlan, String> {
    let manifest_path = std::fs::canonicalize(manifest_path).map_err(|e| {
        format!(
            "failed to resolve expert-HQQ cache generation manifest {}: {e}",
            manifest_path.display()
        )
    })?;
    let manifest_dir = manifest_path
        .parent()
        .ok_or_else(|| {
            format!(
                "expert-HQQ cache generation manifest {} has no parent directory",
                manifest_path.display()
            )
        })?
        .to_path_buf();
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| {
        format!(
            "failed to read expert-HQQ cache generation manifest {}: {e}",
            manifest_path.display()
        )
    })?;
    let parsed: ExpertHqqCacheGenerationManifestJson = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "malformed expert-HQQ cache generation manifest {}: {e}",
            manifest_path.display()
        )
    })?;
    if parsed.purpose != "expert_hqq_cache_generation" {
        return Err(format!(
            "expert-HQQ cache generation manifest purpose must be expert_hqq_cache_generation, got {:?}",
            parsed.purpose
        ));
    }
    validate_expert_hqq_generation_variant(parsed.nbits, parsed.group_size)?;
    if parsed.axis != EXPERT_HQQ_AXIS {
        return Err(format!(
            "expert-HQQ cache generation manifest requires axis={}, got {}",
            EXPERT_HQQ_AXIS, parsed.axis
        ));
    }
    let expected_layout = expert_hqq_layout_for_nbits(parsed.nbits)?;
    if parsed.layout != expected_layout {
        return Err(format!(
            "expert-HQQ cache generation manifest layout {:?} does not match nbits={} expected {:?}",
            parsed.layout, parsed.nbits, expected_layout
        ));
    }

    let model_dir = resolve_manifest_dir(&parsed.model_dir, &manifest_dir, "expert-HQQ model_dir")?;
    let config_path = model_dir.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .map_err(|e| format!("failed to read model config {}: {e}", config_path.display()))?;
    let actual_config_hash = expert_hqq_config_hash(&config_bytes);
    let manifest_config_hash = parse_manifest_u64(&parsed.config_hash, "config_hash")?;
    if manifest_config_hash != actual_config_hash {
        return Err(format!(
            "expert-HQQ cache generation manifest config_hash mismatch: manifest={:016x} actual={:016x}",
            manifest_config_hash, actual_config_hash
        ));
    }
    let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|e| {
        format!(
            "failed to parse model config {} for expert-HQQ cache generation: {e}",
            config_path.display()
        )
    })?;
    let model_config = ModelConfig::from_json(&config_json).map_err(|e| {
        format!("failed to parse model config for expert-HQQ cache generation: {e}")
    })?;
    let mut layers = match (parsed.layer_idx, parsed.layers.clone()) {
        (Some(layer_idx), None) => vec![layer_idx],
        (None, Some(layers)) => layers,
        (Some(_), Some(_)) => {
            return Err(
                "expert-HQQ cache generation manifest must specify exactly one of layer_idx or layers"
                    .to_string(),
            );
        }
        (None, None) => {
            return Err(
                "expert-HQQ cache generation manifest must specify layer_idx or layers".to_string(),
            );
        }
    };
    if layers.is_empty() {
        return Err("expert-HQQ cache generation manifest layers must be non-empty".to_string());
    }
    layers.sort_unstable();
    if layers.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("expert-HQQ cache generation manifest layers must be unique".to_string());
    }
    for &layer_idx in &layers {
        if layer_idx >= model_config.num_hidden_layers {
            return Err(format!(
                "expert-HQQ cache generation layer_idx {} out of range {}",
                layer_idx, model_config.num_hidden_layers
            ));
        }
        if !model_config.moe_layer_indices.contains(&layer_idx) {
            return Err(format!(
                "expert-HQQ cache generation layer_idx {} is not a model MoE layer",
                layer_idx
            ));
        }
    }
    if parsed.layers.is_some() && layers != model_config.moe_layer_indices {
        return Err(format!(
            "expert-HQQ cache generation all-layer manifest layers must exactly match model MoE layers {:?}, got {:?}",
            model_config.moe_layer_indices, layers
        ));
    }
    let routed_hidden_size = model_config.routed_expert_hidden_size();
    if routed_hidden_size == 0 {
        return Err("expert-HQQ cache generation routed_hidden_size must be > 0".to_string());
    }

    let mut experts = parsed.experts.clone();
    experts.sort_unstable();
    if experts.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("expert-HQQ cache generation manifest experts must be unique".to_string());
    }
    let expected_experts: Vec<usize> = (0..model_config.n_routed_experts).collect();
    if experts != expected_experts {
        return Err(format!(
            "expert-HQQ cache generation manifest experts must exactly match model expert range 0..{}, got {} entries",
            model_config.n_routed_experts.saturating_sub(1),
            experts.len()
        ));
    }

    let roles = parse_exact_w13_w2_roles(&parsed.roles, "expert-HQQ cache generation manifest")?;
    let expected_tensor_count = layers.len() * experts.len() * roles.len();
    if parsed.tensors.len() != expected_tensor_count {
        return Err(format!(
            "expert-HQQ cache generation manifest tensors must contain complete W13/W2 pairs for {} layers and {} experts, got {} records",
            layers.len(),
            experts.len(),
            parsed.tensors.len()
        ));
    }

    let output_cache_path = resolve_manifest_output_path(
        &parsed.output_cache_path,
        &manifest_dir,
        "expert-HQQ output_cache_path",
    )?;
    let diagnostic_spec_path = resolve_manifest_output_path(
        &parsed.diagnostic_spec_path,
        &manifest_dir,
        "expert-HQQ diagnostic_spec_path",
    )?;

    let mut seen = HashSet::new();
    let mut keyed_specs = Vec::with_capacity(parsed.tensors.len());
    for (idx, tensor) in parsed.tensors.iter().enumerate() {
        if !layers.binary_search(&tensor.layer_idx).is_ok() {
            return Err(format!(
                "expert-HQQ cache generation tensors[{idx}].layer_idx {} is not in the manifest layer set {:?}",
                tensor.layer_idx, layers
            ));
        }
        if !experts.binary_search(&tensor.expert_idx).is_ok() {
            return Err(format!(
                "expert-HQQ cache generation tensors[{idx}].expert_idx {} is not in the manifest expert set",
                tensor.expert_idx
            ));
        }
        let role = parse_expert_hqq_role(
            &tensor.role,
            &format!("expert-HQQ cache generation tensors[{idx}].role"),
        )?;
        let key = ExpertHqqTensorKey::new(role, tensor.layer_idx, tensor.expert_idx);
        if !seen.insert(key) {
            return Err(format!(
                "duplicate expert-HQQ cache generation tensor for layer={} expert={} role={}",
                key.layer_idx,
                key.expert_idx,
                key.role.as_str()
            ));
        }
        let (expected_rows, expected_cols) = expected_generation_tensor_shape(&model_config, role)?;
        if tensor.expected_rows != expected_rows || tensor.expected_cols != expected_cols {
            return Err(format!(
                "expert-HQQ cache generation tensors[{idx}] shape metadata mismatch for layer={} expert={} role={}: manifest={}x{} expected={}x{}",
                tensor.layer_idx,
                tensor.expert_idx,
                role.as_str(),
                tensor.expected_rows,
                tensor.expected_cols,
                expected_rows,
                expected_cols
            ));
        }
        let shard_path = resolve_manifest_file(
            &tensor.shard_path,
            &model_dir,
            "expert-HQQ safetensors shard_path",
        )?;
        validate_generation_safetensors_entry(
            &shard_path,
            &tensor.tensor_key,
            tensor.expected_rows,
            tensor.expected_cols,
        )
        .map_err(|e| {
            format!("expert-HQQ cache generation tensors[{idx}] source validation failed: {e}")
        })?;
        keyed_specs.push((
            key,
            ExpertHqqSafetensorsTensorSpec::new(
                shard_path,
                &tensor.tensor_key,
                role,
                tensor.layer_idx,
                tensor.expert_idx,
                parsed.nbits,
                parsed.group_size,
            )?,
        ));
    }
    for &layer_idx in &layers {
        for &expert in &experts {
            for role in [ExpertHqqTensorRole::W13, ExpertHqqTensorRole::W2] {
                let key = ExpertHqqTensorKey::new(role, layer_idx, expert);
                if !seen.contains(&key) {
                    return Err(format!(
                        "expert-HQQ cache generation manifest missing required tensor for layer={} expert={} role={}",
                        key.layer_idx,
                        key.expert_idx,
                        key.role.as_str()
                    ));
                }
            }
        }
    }
    keyed_specs.sort_by_key(|(key, _)| {
        (
            key.layer_idx,
            key.expert_idx,
            match key.role {
                ExpertHqqTensorRole::W13 => 0usize,
                ExpertHqqTensorRole::W2 => 1usize,
            },
        )
    });
    let specs: Vec<_> = keyed_specs.into_iter().map(|(_, spec)| spec).collect();
    let required_tensors = layers
        .iter()
        .flat_map(|&layer_idx| {
            experts.iter().flat_map(move |&expert| {
                [
                    ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, layer_idx, expert),
                    ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, layer_idx, expert),
                ]
            })
        })
        .collect::<Vec<_>>();
    let header = ExpertHqqCacheHeader::new(
        model_config.hidden_size,
        routed_hidden_size,
        model_config.moe_intermediate_size,
        model_config.n_routed_experts,
        model_config.num_hidden_layers,
        actual_config_hash,
        specs.len(),
    )?;

    Ok(ExpertHqqCacheGenerationPlan {
        manifest_path,
        model_dir,
        output_cache_path,
        diagnostic_spec_path,
        header,
        layer_idx: layers[0],
        layers,
        experts,
        nbits: parsed.nbits,
        group_size: parsed.group_size,
        layout: parsed.layout,
        specs,
        required_tensors,
    })
}

pub fn generate_expert_hqq_cache_from_manifest_path(
    manifest_path: &Path,
) -> Result<ExpertHqqCacheGenerationReport, String> {
    let plan = plan_expert_hqq_cache_generation_from_manifest_path(manifest_path)?;
    let cache = write_expert_hqq_cache_from_safetensors(
        &plan.output_cache_path,
        plan.header.clone(),
        &plan.specs,
    )?;
    let loaded = load_expert_hqq_cache(&plan.output_cache_path, &cache.header.expectation())?;
    if loaded != cache {
        return Err(format!(
            "expert-HQQ cache generation readback mismatch after writing {}",
            plan.output_cache_path.display()
        ));
    }
    loaded.validate_required_tensors(&plan.required_tensors)?;
    write_generation_diagnostic_cache_spec(&plan, &loaded)?;
    let diagnostic_spec = load_expert_hqq_diagnostic_cache_spec(&plan.diagnostic_spec_path)?;
    diagnostic_spec
        .validate_model_bounds(plan.header.num_moe_layers, plan.header.n_routed_experts)?;
    loaded.validate_required_tensors(&diagnostic_spec.required_tensors)?;
    diagnostic_spec.validate_cache_descriptors(&loaded)?;
    let total_payload_bytes = loaded
        .tensors
        .iter()
        .map(|record| record.packed.len() + record.scales.len() + record.zeros.len())
        .sum();
    let cache_file_bytes = std::fs::metadata(&plan.output_cache_path)
        .map_err(|e| {
            format!(
                "failed to stat generated expert-HQQ cache {}: {e}",
                plan.output_cache_path.display()
            )
        })?
        .len();
    Ok(ExpertHqqCacheGenerationReport {
        manifest_path: plan.manifest_path,
        cache_path: plan.output_cache_path,
        diagnostic_spec_path: plan.diagnostic_spec_path,
        layer_idx: plan.layer_idx,
        layers: plan.layers,
        expert_count: plan.experts.len(),
        tensor_records: loaded.tensors.len(),
        total_payload_bytes,
        cache_file_bytes,
    })
}

fn parse_expert_hqq_role(value: &str, label: &str) -> Result<ExpertHqqTensorRole, String> {
    match value {
        "w13" => Ok(ExpertHqqTensorRole::W13),
        "w2" => Ok(ExpertHqqTensorRole::W2),
        other => Err(format!("{label} must be w13 or w2, got {other:?}")),
    }
}

fn parse_exact_w13_w2_roles(
    values: &[String],
    label: &str,
) -> Result<Vec<ExpertHqqTensorRole>, String> {
    if values.len() != 2 {
        return Err(format!("{label} roles must contain exactly w13 and w2"));
    }
    let mut roles = HashSet::new();
    for (idx, value) in values.iter().enumerate() {
        let role = parse_expert_hqq_role(value, &format!("{label} roles[{idx}]"))?;
        if !roles.insert(role) {
            return Err(format!("{label} roles duplicate {}", role.as_str()));
        }
    }
    if !roles.contains(&ExpertHqqTensorRole::W13) || !roles.contains(&ExpertHqqTensorRole::W2) {
        return Err(format!("{label} roles must contain exactly w13 and w2"));
    }
    Ok(vec![ExpertHqqTensorRole::W13, ExpertHqqTensorRole::W2])
}

fn parse_manifest_u64(value: &str, label: &str) -> Result<u64, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "expert-HQQ cache generation manifest {label} must be non-empty"
        ));
    }
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|e| {
            format!(
                "expert-HQQ cache generation manifest {label} is not valid hex u64 {trimmed:?}: {e}"
            )
        })
    } else {
        trimmed.parse::<u64>().map_err(|e| {
            format!(
                "expert-HQQ cache generation manifest {label} is not valid u64 {trimmed:?}: {e}"
            )
        })
    }
}

fn validate_expert_hqq_generation_variant(nbits: u8, group_size: usize) -> Result<(), String> {
    let allowed_groups = match nbits {
        6 => EXPERT_HQQ_GENERATION_HQQ6_GROUP_SIZES,
        8 => EXPERT_HQQ_GENERATION_HQQ8_GROUP_SIZES,
        other => {
            return Err(format!(
                "expert-HQQ cache generation manifest requires nbits in [6, 8], got {other}"
            ))
        }
    };
    if allowed_groups.contains(&group_size) {
        return Ok(());
    }
    Err(format!(
        "expert-HQQ cache generation manifest requires group_size in {:?} for HQQ{}, got {}",
        allowed_groups, nbits, group_size
    ))
}

fn resolve_manifest_dir(raw: &str, base: &Path, label: &str) -> Result<PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(format!("{label} must be non-empty"));
    }
    let candidate = PathBuf::from(raw);
    let candidate = if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    };
    let resolved = std::fs::canonicalize(&candidate)
        .map_err(|e| format!("failed to resolve {label} {}: {e}", candidate.display()))?;
    if !resolved.is_dir() {
        return Err(format!(
            "{label} is not a directory: {}",
            resolved.display()
        ));
    }
    Ok(resolved)
}

fn resolve_manifest_file(raw: &str, base: &Path, label: &str) -> Result<PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(format!("{label} must be non-empty"));
    }
    let candidate = PathBuf::from(raw);
    let candidate = if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    };
    let resolved = std::fs::canonicalize(&candidate)
        .map_err(|e| format!("failed to resolve {label} {}: {e}", candidate.display()))?;
    if !resolved.is_file() {
        return Err(format!("{label} is not a file: {}", resolved.display()));
    }
    Ok(resolved)
}

fn resolve_manifest_output_path(raw: &str, base: &Path, label: &str) -> Result<PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(format!("{label} must be non-empty"));
    }
    let candidate = PathBuf::from(raw);
    let candidate = if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    };
    let parent = candidate
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| format!("{label} must include a parent directory"))?;
    let parent = std::fs::canonicalize(parent)
        .map_err(|e| format!("failed to resolve {label} parent {}: {e}", parent.display()))?;
    let file_name = candidate
        .file_name()
        .ok_or_else(|| format!("{label} must include a file name"))?;
    Ok(parent.join(file_name))
}

fn expected_generation_tensor_shape(
    config: &ModelConfig,
    role: ExpertHqqTensorRole,
) -> Result<(usize, usize), String> {
    let routed_hidden_size = config.routed_expert_hidden_size();
    if routed_hidden_size == 0 {
        return Err("expert-HQQ cache generation routed_hidden_size must be > 0".to_string());
    }
    match role {
        ExpertHqqTensorRole::W13 => {
            let rows = if config.experts_gated {
                checked_mul(2, config.moe_intermediate_size, "gated W13 rows")?
            } else {
                config.moe_intermediate_size
            };
            Ok((rows, routed_hidden_size))
        }
        ExpertHqqTensorRole::W2 => Ok((routed_hidden_size, config.moe_intermediate_size)),
    }
}

fn validate_generation_safetensors_entry(
    shard_path: &Path,
    tensor_key: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), String> {
    let safetensors = MmapSafetensors::open(shard_path)
        .map_err(|e| format!("failed to open safetensors {}: {e}", shard_path.display()))?;
    let info = safetensors
        .tensor_info(tensor_key)
        .ok_or_else(|| format!("Tensor not found: {tensor_key} in {}", shard_path.display()))?;
    if info.shape.as_slice() != [expected_rows, expected_cols] {
        return Err(format!(
            "safetensors tensor {tensor_key} shape {:?} != manifest {}x{}",
            info.shape, expected_rows, expected_cols
        ));
    }
    if !matches!(&info.dtype, Dtype::F32 | Dtype::Bf16 | Dtype::F16) {
        return Err(format!(
            "safetensors tensor {tensor_key} dtype {:?} is unsupported; expected BF16, F16, or F32",
            info.dtype
        ));
    }
    Ok(())
}

fn write_generation_diagnostic_cache_spec(
    plan: &ExpertHqqCacheGenerationPlan,
    cache: &ExpertHqqCache,
) -> Result<(), String> {
    if cache.tensors.len() != plan.required_tensors.len() {
        return Err(format!(
            "expert-HQQ generated cache tensor count {} != required {}",
            cache.tensors.len(),
            plan.required_tensors.len()
        ));
    }
    let spec_parent = plan.diagnostic_spec_path.parent().ok_or_else(|| {
        format!(
            "expert-HQQ diagnostic spec path {} has no parent directory",
            plan.diagnostic_spec_path.display()
        )
    })?;
    let cache_path_for_spec = if plan.output_cache_path.parent() == Some(spec_parent) {
        plan.output_cache_path
            .file_name()
            .ok_or_else(|| {
                format!(
                    "expert-HQQ output cache path {} has no file name",
                    plan.output_cache_path.display()
                )
            })?
            .to_string_lossy()
            .to_string()
    } else {
        plan.output_cache_path.display().to_string()
    };
    let requirements = plan
        .layers
        .iter()
        .map(|&layer_idx| {
            serde_json::json!({
                "layer_idx": layer_idx,
                "experts": plan.experts.clone(),
                "roles": ["w13", "w2"],
                "nbits": plan.nbits,
                "group_size": plan.group_size
            })
        })
        .collect::<Vec<_>>();
    let spec_json = serde_json::json!({
        "purpose": "runtime_prefill_diagnostic",
        "cache_path": cache_path_for_spec,
        "requirements": requirements
    });
    let encoded = serde_json::to_string_pretty(&spec_json)
        .map_err(|e| format!("failed to encode expert-HQQ diagnostic cache spec: {e}"))?;
    std::fs::write(&plan.diagnostic_spec_path, format!("{encoded}\n")).map_err(|e| {
        format!(
            "failed to write expert-HQQ diagnostic cache spec {}: {e}",
            plan.diagnostic_spec_path.display()
        )
    })
}

impl ExpertHqqDiagnosticCacheSpec {
    pub fn validate_model_bounds(
        &self,
        num_hidden_layers: usize,
        n_routed_experts: usize,
    ) -> Result<(), String> {
        for req in &self.requirements {
            if req.layer_idx >= num_hidden_layers {
                return Err(format!(
                    "expert-HQQ diagnostic cache spec layer_idx {} out of range {}",
                    req.layer_idx, num_hidden_layers
                ));
            }
            if req.expert_idx >= n_routed_experts {
                return Err(format!(
                    "expert-HQQ diagnostic cache spec expert_idx {} out of range {}",
                    req.expert_idx, n_routed_experts
                ));
            }
        }
        Ok(())
    }

    pub fn validate_cache_descriptors(&self, cache: &ExpertHqqCache) -> Result<(), String> {
        for req in &self.requirements {
            let expected_layout = expert_hqq_layout_for_nbits(req.nbits)?;
            for role in [ExpertHqqTensorRole::W13, ExpertHqqTensorRole::W2] {
                let key = ExpertHqqTensorKey::new(role, req.layer_idx, req.expert_idx);
                let record = cache.require_tensor_record(key)?;
                let desc = &record.descriptor;
                if desc.nbits != req.nbits {
                    return Err(format!(
                        "expert-HQQ diagnostic cache descriptor nbits mismatch for layer={} expert={} role={}: cache={} spec={}",
                        req.layer_idx,
                        req.expert_idx,
                        role.as_str(),
                        desc.nbits,
                        req.nbits
                    ));
                }
                if desc.group_size != req.group_size {
                    return Err(format!(
                        "expert-HQQ diagnostic cache descriptor group_size mismatch for layer={} expert={} role={}: cache={} spec={}",
                        req.layer_idx,
                        req.expert_idx,
                        role.as_str(),
                        desc.group_size,
                        req.group_size
                    ));
                }
                if desc.layout != expected_layout {
                    return Err(format!(
                        "expert-HQQ diagnostic cache descriptor layout mismatch for layer={} expert={} role={}: cache={} spec={}",
                        req.layer_idx,
                        req.expert_idx,
                        role.as_str(),
                        desc.layout,
                        expected_layout
                    ));
                }
            }
        }
        Ok(())
    }
}

pub fn expert_hqq_layout_for_nbits(nbits: u8) -> Result<&'static str, String> {
    match nbits {
        4 => Ok("row_major_axis1_grouped_uint4_packed"),
        6 => Ok("row_major_axis1_grouped_uint6_packed"),
        8 => Ok("row_major_axis1_grouped_uint8"),
        other => Err(format!(
            "Unsupported expert-HQQ nbits {other}; expected 4, 6, or 8"
        )),
    }
}

fn layout_code_for_nbits(nbits: u8) -> Result<u32, String> {
    match nbits {
        4 => Ok(4),
        6 => Ok(6),
        8 => Ok(8),
        other => Err(format!(
            "Unsupported expert-HQQ nbits {other}; expected 4, 6, or 8"
        )),
    }
}

fn nbits_from_layout_code(code: u32) -> Result<u8, String> {
    match code {
        4 => Ok(4),
        6 => Ok(6),
        8 => Ok(8),
        other => Err(format!(
            "Unsupported expert-HQQ layout code {other}; expected 4, 6, or 8"
        )),
    }
}

#[inline]
fn checked_mul(a: usize, b: usize, label: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("expert-HQQ {label} byte count overflow"))
}

#[inline]
fn group_count(cols: usize, group_size: usize) -> Result<usize, String> {
    if group_size == 0 {
        return Err("expert-HQQ group_size must be > 0".to_string());
    }
    Ok(cols.div_ceil(group_size))
}

fn padded_cols(cols: usize, group_size: usize) -> Result<usize, String> {
    checked_mul(group_count(cols, group_size)?, group_size, "padded cols")
}

pub fn expert_hqq_component_sizes(
    rows: usize,
    cols: usize,
    nbits: u8,
    group_size: usize,
) -> Result<(usize, usize, usize), String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "expert-HQQ tensors require positive rows/cols, got {rows}x{cols}"
        ));
    }
    let padded = padded_cols(cols, group_size)?;
    let packed_cols = match nbits {
        4 => padded.div_ceil(2),
        6 => padded
            .div_ceil(4)
            .checked_mul(3)
            .ok_or_else(|| "expert-HQQ uint6 packed columns overflow".to_string())?,
        8 => padded,
        other => {
            return Err(format!(
                "Unsupported expert-HQQ nbits {other}; expected 4, 6, or 8"
            ))
        }
    };
    let packed_bytes = checked_mul(rows, packed_cols, "packed")?;
    let groups = group_count(cols, group_size)?;
    let meta_entries = checked_mul(rows, groups, "scale/zero entries")?;
    let scales_bytes = checked_mul(meta_entries, std::mem::size_of::<f32>(), "scales")?;
    let zeros_bytes = checked_mul(meta_entries, std::mem::size_of::<f32>(), "zeros")?;
    Ok((packed_bytes, scales_bytes, zeros_bytes))
}

impl ExpertHqqCacheHeader {
    pub fn new(
        hidden_size: usize,
        routed_hidden_size: usize,
        moe_intermediate_size: usize,
        n_routed_experts: usize,
        num_moe_layers: usize,
        config_hash: u64,
        tensor_count: usize,
    ) -> Result<Self, String> {
        let header = Self {
            version: EXPERT_HQQ_CACHE_VERSION,
            hidden_size,
            routed_hidden_size,
            moe_intermediate_size,
            n_routed_experts,
            num_moe_layers,
            config_hash,
            tensor_count,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn expectation(&self) -> ExpertHqqCacheExpectation {
        ExpertHqqCacheExpectation {
            hidden_size: self.hidden_size,
            routed_hidden_size: self.routed_hidden_size,
            moe_intermediate_size: self.moe_intermediate_size,
            n_routed_experts: self.n_routed_experts,
            num_moe_layers: self.num_moe_layers,
            config_hash: self.config_hash,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != EXPERT_HQQ_CACHE_VERSION {
            return Err(format!(
                "expert-HQQ cache version {} != {}",
                self.version, EXPERT_HQQ_CACHE_VERSION
            ));
        }
        if self.hidden_size == 0
            || self.routed_hidden_size == 0
            || self.moe_intermediate_size == 0
            || self.n_routed_experts == 0
            || self.num_moe_layers == 0
        {
            return Err(format!(
                "expert-HQQ header has invalid dimensions h={} rh={} m={} experts={} layers={}",
                self.hidden_size,
                self.routed_hidden_size,
                self.moe_intermediate_size,
                self.n_routed_experts,
                self.num_moe_layers,
            ));
        }
        if self.tensor_count == 0 {
            return Err("expert-HQQ cache must contain at least one tensor descriptor".to_string());
        }
        Ok(())
    }

    pub fn validate_against(&self, expected: &ExpertHqqCacheExpectation) -> Result<(), String> {
        self.validate()?;
        let mut errors = Vec::new();
        if self.hidden_size != expected.hidden_size {
            errors.push(format!(
                "hidden_size={} expected={}",
                self.hidden_size, expected.hidden_size
            ));
        }
        if self.routed_hidden_size != expected.routed_hidden_size {
            errors.push(format!(
                "routed_hidden_size={} expected={}",
                self.routed_hidden_size, expected.routed_hidden_size
            ));
        }
        if self.moe_intermediate_size != expected.moe_intermediate_size {
            errors.push(format!(
                "moe_intermediate_size={} expected={}",
                self.moe_intermediate_size, expected.moe_intermediate_size
            ));
        }
        if self.n_routed_experts != expected.n_routed_experts {
            errors.push(format!(
                "n_routed_experts={} expected={}",
                self.n_routed_experts, expected.n_routed_experts
            ));
        }
        if self.num_moe_layers != expected.num_moe_layers {
            errors.push(format!(
                "num_moe_layers={} expected={}",
                self.num_moe_layers, expected.num_moe_layers
            ));
        }
        if self.config_hash != expected.config_hash {
            errors.push(format!(
                "config_hash={:016x} expected={:016x}",
                self.config_hash, expected.config_hash
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "expert-HQQ cache header mismatch: {}",
                errors.join(", ")
            ))
        }
    }
}

impl ExpertHqqTensorDescriptor {
    pub fn new(
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        rows: usize,
        cols: usize,
        nbits: u8,
        group_size: usize,
    ) -> Result<Self, String> {
        let (packed_bytes, scales_bytes, zeros_bytes) =
            expert_hqq_component_sizes(rows, cols, nbits, group_size)?;
        Ok(Self {
            role,
            layer_idx,
            expert_idx,
            nbits,
            rows,
            cols,
            group_size,
            axis: EXPERT_HQQ_AXIS,
            layout: expert_hqq_layout_for_nbits(nbits)?.to_string(),
            packed_dtype: EXPERT_HQQ_PACKED_DTYPE.to_string(),
            scales_dtype: EXPERT_HQQ_SCALES_DTYPE.to_string(),
            zeros_dtype: EXPERT_HQQ_ZEROS_DTYPE.to_string(),
            packed_bytes,
            scales_bytes,
            zeros_bytes,
        })
    }

    pub fn validate(&self, header: &ExpertHqqCacheHeader) -> Result<(), String> {
        if self.layer_idx >= header.num_moe_layers {
            return Err(format!(
                "expert-HQQ {} layer_idx {} out of range {}",
                self.role.as_str(),
                self.layer_idx,
                header.num_moe_layers
            ));
        }
        if self.expert_idx >= header.n_routed_experts {
            return Err(format!(
                "expert-HQQ {} expert_idx {} out of range {}",
                self.role.as_str(),
                self.expert_idx,
                header.n_routed_experts
            ));
        }
        if self.axis != EXPERT_HQQ_AXIS {
            return Err(format!(
                "expert-HQQ {} uses axis {} (expected axis={})",
                self.role.as_str(),
                self.axis,
                EXPERT_HQQ_AXIS
            ));
        }
        let expected_layout = expert_hqq_layout_for_nbits(self.nbits)?;
        if self.layout != expected_layout {
            return Err(format!(
                "expert-HQQ {} layout '{}' does not match nbits {} expected '{}'",
                self.role.as_str(),
                self.layout,
                self.nbits,
                expected_layout
            ));
        }
        if self.packed_dtype != EXPERT_HQQ_PACKED_DTYPE
            || self.scales_dtype != EXPERT_HQQ_SCALES_DTYPE
            || self.zeros_dtype != EXPERT_HQQ_ZEROS_DTYPE
        {
            return Err(format!(
                "expert-HQQ {} dtype mismatch packed/scales/zeros={}/{}/{}",
                self.role.as_str(),
                self.packed_dtype,
                self.scales_dtype,
                self.zeros_dtype
            ));
        }
        match self.role {
            ExpertHqqTensorRole::W13 => {
                if self.cols != header.routed_hidden_size {
                    return Err(format!(
                        "expert-HQQ W13 cols {} != routed_hidden_size {}",
                        self.cols, header.routed_hidden_size
                    ));
                }
                if self.rows != header.moe_intermediate_size
                    && self.rows != 2 * header.moe_intermediate_size
                {
                    return Err(format!(
                        "expert-HQQ W13 rows {} must be intermediate {} or gated 2*intermediate {}",
                        self.rows,
                        header.moe_intermediate_size,
                        2 * header.moe_intermediate_size
                    ));
                }
            }
            ExpertHqqTensorRole::W2 => {
                if self.cols != header.moe_intermediate_size {
                    return Err(format!(
                        "expert-HQQ W2 cols {} != moe_intermediate_size {}",
                        self.cols, header.moe_intermediate_size
                    ));
                }
                if self.rows != header.routed_hidden_size {
                    return Err(format!(
                        "expert-HQQ W2 rows {} != routed_hidden_size {}",
                        self.rows, header.routed_hidden_size
                    ));
                }
            }
        }

        let (packed_bytes, scales_bytes, zeros_bytes) =
            expert_hqq_component_sizes(self.rows, self.cols, self.nbits, self.group_size)?;
        if self.packed_bytes != packed_bytes
            || self.scales_bytes != scales_bytes
            || self.zeros_bytes != zeros_bytes
        {
            return Err(format!(
                "expert-HQQ {} component bytes mismatch: packed/scales/zeros={}/{}/{} expected {}/{}/{}",
                self.role.as_str(),
                self.packed_bytes,
                self.scales_bytes,
                self.zeros_bytes,
                packed_bytes,
                scales_bytes,
                zeros_bytes
            ));
        }
        Ok(())
    }
}

impl ExpertHqqTensorRecord {
    pub fn new(
        descriptor: ExpertHqqTensorDescriptor,
        packed: Vec<u8>,
        scales: Vec<u8>,
        zeros: Vec<u8>,
    ) -> Result<Self, String> {
        let record = Self {
            descriptor,
            packed,
            scales,
            zeros,
        };
        record.validate_payload_lengths()?;
        Ok(record)
    }

    fn validate_payload_lengths(&self) -> Result<(), String> {
        validate_component_payload_lengths(
            self.descriptor.role,
            self.packed.len(),
            self.scales.len(),
            self.zeros.len(),
            self.descriptor.packed_bytes,
            self.descriptor.scales_bytes,
            self.descriptor.zeros_bytes,
        )
    }
}

impl ExpertHqqTensorInput {
    pub fn new(
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        rows: usize,
        cols: usize,
        nbits: u8,
        group_size: usize,
        packed: Vec<u8>,
        scales: Vec<u8>,
        zeros: Vec<u8>,
    ) -> Result<Self, String> {
        let descriptor = ExpertHqqTensorDescriptor::new(
            role, layer_idx, expert_idx, rows, cols, nbits, group_size,
        )?;
        let input = Self {
            descriptor,
            packed,
            scales,
            zeros,
        };
        input.validate_payload_lengths()?;
        Ok(input)
    }

    pub fn validate_payload_lengths(&self) -> Result<(), String> {
        validate_component_payload_lengths(
            self.descriptor.role,
            self.packed.len(),
            self.scales.len(),
            self.zeros.len(),
            self.descriptor.packed_bytes,
            self.descriptor.scales_bytes,
            self.descriptor.zeros_bytes,
        )
    }

    fn into_record(self) -> Result<ExpertHqqTensorRecord, String> {
        ExpertHqqTensorRecord::new(self.descriptor, self.packed, self.scales, self.zeros)
    }
}

impl ExpertHqqSafetensorsTensorSpec {
    pub fn new<P: Into<PathBuf>, S: Into<String>>(
        path: P,
        key: S,
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        nbits: u8,
        group_size: usize,
    ) -> Result<Self, String> {
        if group_size == 0 {
            return Err("expert-HQQ safetensors spec group_size must be > 0".to_string());
        }
        if nbits != 4 && nbits != 6 && nbits != 8 {
            return Err(format!(
                "expert-HQQ safetensors builder supports HQQ4/HQQ6/HQQ8 only, got nbits={nbits}"
            ));
        }
        Ok(Self {
            path: path.into(),
            key: key.into(),
            role,
            layer_idx,
            expert_idx,
            nbits,
            group_size,
        })
    }
}

fn validate_component_payload_lengths(
    role: ExpertHqqTensorRole,
    packed_len: usize,
    scales_len: usize,
    zeros_len: usize,
    packed_bytes: usize,
    scales_bytes: usize,
    zeros_bytes: usize,
) -> Result<(), String> {
    if packed_len != packed_bytes {
        return Err(format!(
            "expert-HQQ {} packed length {} != descriptor {}",
            role.as_str(),
            packed_len,
            packed_bytes
        ));
    }
    if scales_len != scales_bytes {
        return Err(format!(
            "expert-HQQ {} scales length {} != descriptor {}",
            role.as_str(),
            scales_len,
            scales_bytes
        ));
    }
    if zeros_len != zeros_bytes {
        return Err(format!(
            "expert-HQQ {} zeros length {} != descriptor {}",
            role.as_str(),
            zeros_len,
            zeros_bytes
        ));
    }
    Ok(())
}

impl ExpertHqqCache {
    pub fn new(
        header: ExpertHqqCacheHeader,
        tensors: Vec<ExpertHqqTensorRecord>,
    ) -> Result<Self, String> {
        let cache = Self { header, tensors };
        cache.validate()?;
        Ok(cache)
    }

    pub fn from_inputs(
        header: ExpertHqqCacheHeader,
        tensors: Vec<ExpertHqqTensorInput>,
    ) -> Result<Self, String> {
        let records: Result<Vec<_>, _> = tensors
            .into_iter()
            .map(ExpertHqqTensorInput::into_record)
            .collect();
        Self::new(header, records?)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.header.validate()?;
        if self.header.tensor_count != self.tensors.len() {
            return Err(format!(
                "expert-HQQ tensor_count {} != records {}",
                self.header.tensor_count,
                self.tensors.len()
            ));
        }
        let mut seen = HashSet::new();
        for record in &self.tensors {
            record.descriptor.validate(&self.header)?;
            record.validate_payload_lengths()?;
            let key = (
                record.descriptor.layer_idx,
                record.descriptor.expert_idx,
                record.descriptor.role,
            );
            if !seen.insert(key) {
                return Err(format!(
                    "duplicate expert-HQQ descriptor for layer={} expert={} role={}",
                    record.descriptor.layer_idx,
                    record.descriptor.expert_idx,
                    record.descriptor.role.as_str()
                ));
            }
        }
        Ok(())
    }

    pub fn validate_against(&self, expected: &ExpertHqqCacheExpectation) -> Result<(), String> {
        self.header.validate_against(expected)?;
        self.validate()
    }

    pub fn tensor_record(&self, key: ExpertHqqTensorKey) -> Option<&ExpertHqqTensorRecord> {
        self.tensors.iter().find(|record| {
            record.descriptor.role == key.role
                && record.descriptor.layer_idx == key.layer_idx
                && record.descriptor.expert_idx == key.expert_idx
        })
    }

    pub fn require_tensor_record(
        &self,
        key: ExpertHqqTensorKey,
    ) -> Result<&ExpertHqqTensorRecord, String> {
        self.tensor_record(key).ok_or_else(|| {
            format!(
                "missing required expert-HQQ descriptor for layer={} expert={} role={}",
                key.layer_idx,
                key.expert_idx,
                key.role.as_str()
            )
        })
    }

    pub fn validate_required_tensors(&self, required: &[ExpertHqqTensorKey]) -> Result<(), String> {
        let mut seen = HashSet::new();
        for &key in required {
            if !seen.insert(key) {
                return Err(format!(
                    "duplicate expert-HQQ registration requirement for layer={} expert={} role={}",
                    key.layer_idx,
                    key.expert_idx,
                    key.role.as_str()
                ));
            }
            self.require_tensor_record(key)?;
        }
        Ok(())
    }

    pub fn prefill_dispatch_plan(
        &self,
        layer_idx: usize,
        experts_gated: bool,
        works: &[ExpertHqqPrefillWork],
    ) -> Result<ExpertHqqPrefillDispatchPlan, String> {
        self.validate()?;
        if layer_idx >= self.header.num_moe_layers {
            return Err(format!(
                "expert-HQQ prefill dispatch layer_idx {} out of range {}",
                layer_idx, self.header.num_moe_layers
            ));
        }
        if works.is_empty() {
            return Err(
                "expert-HQQ prefill dispatch requires at least one selected expert".to_string(),
            );
        }
        let expected_w13_rows = if experts_gated {
            checked_mul(
                2,
                self.header.moe_intermediate_size,
                "prefill dispatch gated W13 rows",
            )?
        } else {
            self.header.moe_intermediate_size
        };
        let mut seen_experts = HashSet::new();
        let mut entries = Vec::with_capacity(works.len());
        for work in works {
            if work.row_count == 0 {
                return Err(format!(
                    "expert-HQQ prefill dispatch expert {} has zero selected rows",
                    work.expert_idx
                ));
            }
            work.row_offset.checked_add(work.row_count).ok_or_else(|| {
                format!(
                    "expert-HQQ prefill dispatch row range overflow for expert {}",
                    work.expert_idx
                )
            })?;
            if work.expert_idx >= self.header.n_routed_experts {
                return Err(format!(
                    "expert-HQQ prefill dispatch expert_idx {} out of range {}",
                    work.expert_idx, self.header.n_routed_experts
                ));
            }
            if !seen_experts.insert(work.expert_idx) {
                return Err(format!(
                    "expert-HQQ prefill dispatch duplicate selected expert {}",
                    work.expert_idx
                ));
            }
            let w13_key =
                ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, layer_idx, work.expert_idx);
            let w2_key =
                ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, layer_idx, work.expert_idx);
            let w13 = self.require_tensor_record(w13_key)?;
            let w2 = self.require_tensor_record(w2_key)?;
            w13.validate_payload_lengths()?;
            w2.validate_payload_lengths()?;

            let w13_desc = &w13.descriptor;
            let w2_desc = &w2.descriptor;
            if w13_desc.rows != expected_w13_rows {
                return Err(format!(
                    "expert-HQQ prefill dispatch W13 rows {} != expected {} for layer={} expert={} experts_gated={}",
                    w13_desc.rows, expected_w13_rows, layer_idx, work.expert_idx, experts_gated
                ));
            }
            if w13_desc.cols != self.header.routed_hidden_size {
                return Err(format!(
                    "expert-HQQ prefill dispatch W13 cols {} != routed_hidden_size {} for layer={} expert={}",
                    w13_desc.cols, self.header.routed_hidden_size, layer_idx, work.expert_idx
                ));
            }
            if w2_desc.rows != self.header.routed_hidden_size {
                return Err(format!(
                    "expert-HQQ prefill dispatch W2 rows {} != routed_hidden_size {} for layer={} expert={}",
                    w2_desc.rows, self.header.routed_hidden_size, layer_idx, work.expert_idx
                ));
            }
            if w2_desc.cols != self.header.moe_intermediate_size {
                return Err(format!(
                    "expert-HQQ prefill dispatch W2 cols {} != moe_intermediate_size {} for layer={} expert={}",
                    w2_desc.cols, self.header.moe_intermediate_size, layer_idx, work.expert_idx
                ));
            }
            if w13_desc.axis != EXPERT_HQQ_AXIS || w2_desc.axis != EXPERT_HQQ_AXIS {
                return Err(format!(
                    "expert-HQQ prefill dispatch requires axis={} for W13/W2, got {}/{} for layer={} expert={}",
                    EXPERT_HQQ_AXIS, w13_desc.axis, w2_desc.axis, layer_idx, work.expert_idx
                ));
            }
            if w13_desc.layout != expert_hqq_layout_for_nbits(w13_desc.nbits)?
                || w2_desc.layout != expert_hqq_layout_for_nbits(w2_desc.nbits)?
            {
                return Err(format!(
                    "expert-HQQ prefill dispatch layout does not match nbits for layer={} expert={}",
                    layer_idx, work.expert_idx
                ));
            }

            entries.push(ExpertHqqPrefillDispatchEntry {
                expert_idx: work.expert_idx,
                row_offset: work.row_offset,
                row_count: work.row_count,
                w13_key,
                w2_key,
                w13_rows: w13_desc.rows,
                w13_cols: w13_desc.cols,
                w2_rows: w2_desc.rows,
                w2_cols: w2_desc.cols,
                w13_nbits: w13_desc.nbits,
                w2_nbits: w2_desc.nbits,
                w13_group_size: w13_desc.group_size,
                w2_group_size: w2_desc.group_size,
            });
        }
        Ok(ExpertHqqPrefillDispatchPlan {
            layer_idx,
            experts_gated,
            input_layout: "row_major_selected_rows_by_routed_hidden",
            w13_dequant_layout: "row_major_axis1_grouped_rows_by_routed_hidden",
            w13_output_layout: "row_major_selected_rows_by_w13_rows",
            activation_output_layout: "row_major_selected_rows_by_moe_intermediate",
            w2_dequant_layout: "row_major_axis1_grouped_routed_hidden_by_moe_intermediate",
            w2_output_layout: "row_major_selected_rows_by_routed_hidden",
            entries,
        })
    }

    pub fn execute_prefill_reference(
        &self,
        plan: &ExpertHqqPrefillDispatchPlan,
        sorted_routed_inputs: &[f32],
        sorted_row_count: usize,
    ) -> Result<ExpertHqqPrefillReferenceOutput, String> {
        self.validate()?;
        validate_prefill_reference_plan(self, plan, sorted_row_count)?;
        let routed_hidden = self.header.routed_hidden_size;
        let intermediate = self.header.moe_intermediate_size;
        let w13_rows = if plan.experts_gated {
            checked_mul(2, intermediate, "reference gated W13 output rows")?
        } else {
            intermediate
        };
        let expected_input_len = sorted_row_count
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ reference input byte count overflow".to_string())?;
        if sorted_routed_inputs.len() != expected_input_len {
            return Err(format!(
                "expert-HQQ prefill reference input length {} != sorted_row_count*routed_hidden_size {}",
                sorted_routed_inputs.len(),
                expected_input_len
            ));
        }

        let mut row_claimed = vec![false; sorted_row_count];
        let mut output = vec![0.0f32; expected_input_len];
        let mut w13_preactivation = vec![
            0.0f32;
            sorted_row_count.checked_mul(w13_rows).ok_or_else(
                || "expert-HQQ reference W13 output length overflow".to_string()
            )?
        ];
        let mut activation_out = vec![
            0.0f32;
            sorted_row_count.checked_mul(intermediate).ok_or_else(
                || { "expert-HQQ reference activation output length overflow".to_string() }
            )?
        ];
        for entry in &plan.entries {
            let row_end = entry
                .row_offset
                .checked_add(entry.row_count)
                .ok_or_else(|| {
                    format!(
                        "expert-HQQ prefill reference row range overflow for expert {}",
                        entry.expert_idx
                    )
                })?;
            if row_end > sorted_row_count {
                return Err(format!(
                    "expert-HQQ prefill reference row range {}..{} exceeds sorted_row_count {} for expert {}",
                    entry.row_offset, row_end, sorted_row_count, entry.expert_idx
                ));
            }
            for row in entry.row_offset..row_end {
                if row_claimed[row] {
                    return Err(format!(
                        "expert-HQQ prefill reference row {row} is claimed by more than one selected expert"
                    ));
                }
                row_claimed[row] = true;
            }

            let w13 = self.require_tensor_record(entry.w13_key)?;
            let w2 = self.require_tensor_record(entry.w2_key)?;
            validate_prefill_reference_entry(self, plan, entry, w13, w2)?;
            let w13_dequant = dequantize_expert_hqq_record_to_f32(w13)?;
            let w2_dequant = dequantize_expert_hqq_record_to_f32(w2)?;
            let mut activation = vec![0.0f32; intermediate];
            for sorted_row in entry.row_offset..row_end {
                let input_start = sorted_row * routed_hidden;
                let input = &sorted_routed_inputs[input_start..input_start + routed_hidden];
                let w13_start = sorted_row * w13_rows;
                let activation_start = sorted_row * intermediate;
                if plan.experts_gated {
                    for n in 0..intermediate {
                        let gate = dot_f32(
                            &w13_dequant[n * routed_hidden..(n + 1) * routed_hidden],
                            input,
                        );
                        let up_row = intermediate + n;
                        let up = dot_f32(
                            &w13_dequant[up_row * routed_hidden..(up_row + 1) * routed_hidden],
                            input,
                        );
                        let act = silu(gate) * up;
                        w13_preactivation[w13_start + n] = gate;
                        w13_preactivation[w13_start + up_row] = up;
                        activation[n] = act;
                        activation_out[activation_start + n] = act;
                    }
                } else {
                    for n in 0..intermediate {
                        let preact = dot_f32(
                            &w13_dequant[n * routed_hidden..(n + 1) * routed_hidden],
                            input,
                        );
                        let relu = preact.max(0.0);
                        let act = relu * relu;
                        w13_preactivation[w13_start + n] = preact;
                        activation[n] = act;
                        activation_out[activation_start + n] = act;
                    }
                }
                for out_row in 0..routed_hidden {
                    output[input_start + out_row] = dot_f32(
                        &w2_dequant[out_row * intermediate..(out_row + 1) * intermediate],
                        &activation,
                    );
                }
            }
        }
        if let Some(row) = row_claimed.iter().position(|claimed| !*claimed) {
            return Err(format!(
                "expert-HQQ prefill reference row {row} has no selected expert plan entry"
            ));
        }

        Ok(ExpertHqqPrefillReferenceOutput {
            sorted_row_count,
            routed_hidden_size: routed_hidden,
            w13_rows,
            moe_intermediate_size: intermediate,
            w13_preactivation,
            activation: activation_out,
            values: output,
        })
    }

    #[cfg(test)]
    pub fn execute_prefill_test_dispatch(
        &self,
        plan: &ExpertHqqPrefillDispatchPlan,
        sorted_routed_inputs: &[f32],
        sorted_row_count: usize,
    ) -> Result<ExpertHqqPrefillTestDispatchOutput, String> {
        self.validate()?;
        validate_prefill_test_dispatch_plan(self, plan, sorted_row_count)?;
        let routed_hidden = self.header.routed_hidden_size;
        let intermediate = self.header.moe_intermediate_size;
        let expected_input_len = sorted_row_count
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ test dispatch input byte count overflow".to_string())?;
        if sorted_routed_inputs.len() != expected_input_len {
            return Err(format!(
                "expert-HQQ prefill test dispatch input length {} != sorted_row_count*routed_hidden_size {}",
                sorted_routed_inputs.len(),
                expected_input_len
            ));
        }

        let mut row_claimed = vec![false; sorted_row_count];
        let mut output = vec![0.0f32; expected_input_len];
        for entry in &plan.entries {
            let row_end = entry
                .row_offset
                .checked_add(entry.row_count)
                .ok_or_else(|| {
                    format!(
                        "expert-HQQ prefill test dispatch row range overflow for expert {}",
                        entry.expert_idx
                    )
                })?;
            if row_end > sorted_row_count {
                return Err(format!(
                    "expert-HQQ prefill test dispatch row range {}..{} exceeds sorted_row_count {} for expert {}",
                    entry.row_offset, row_end, sorted_row_count, entry.expert_idx
                ));
            }
            for row in entry.row_offset..row_end {
                if row_claimed[row] {
                    return Err(format!(
                        "expert-HQQ prefill test dispatch row {row} is claimed by more than one selected expert"
                    ));
                }
                row_claimed[row] = true;
            }

            let w13 = self.require_tensor_record(entry.w13_key)?;
            let w2 = self.require_tensor_record(entry.w2_key)?;
            validate_prefill_test_dispatch_entry(self, plan, entry, w13, w2)?;
            let w13_dequant = dequantize_expert_hqq_record_to_f32(w13)?;
            let w2_dequant = dequantize_expert_hqq_record_to_f32(w2)?;
            let mut activation = vec![0.0f32; intermediate];
            for sorted_row in entry.row_offset..row_end {
                let input_start = sorted_row * routed_hidden;
                let input = &sorted_routed_inputs[input_start..input_start + routed_hidden];
                if plan.experts_gated {
                    for n in 0..intermediate {
                        let gate = dot_f32(
                            &w13_dequant[n * routed_hidden..(n + 1) * routed_hidden],
                            input,
                        );
                        let up_row = intermediate + n;
                        let up = dot_f32(
                            &w13_dequant[up_row * routed_hidden..(up_row + 1) * routed_hidden],
                            input,
                        );
                        activation[n] = silu(gate) * up;
                    }
                } else {
                    for n in 0..intermediate {
                        let preact = dot_f32(
                            &w13_dequant[n * routed_hidden..(n + 1) * routed_hidden],
                            input,
                        );
                        let relu = preact.max(0.0);
                        activation[n] = relu * relu;
                    }
                }
                for out_row in 0..routed_hidden {
                    output[input_start + out_row] = dot_f32(
                        &w2_dequant[out_row * intermediate..(out_row + 1) * intermediate],
                        &activation,
                    );
                }
            }
        }
        if let Some(row) = row_claimed.iter().position(|claimed| !*claimed) {
            return Err(format!(
                "expert-HQQ prefill test dispatch row {row} has no selected expert plan entry"
            ));
        }

        Ok(ExpertHqqPrefillTestDispatchOutput {
            sorted_row_count,
            routed_hidden_size: routed_hidden,
            values: output,
        })
    }

    pub fn execute_prefill_bf16_path_oracle(
        &self,
        plan: &ExpertHqqPrefillDispatchPlan,
        sorted_routed_inputs: &[f32],
        sorted_row_count: usize,
    ) -> Result<ExpertHqqPrefillBf16PathOracleOutput, String> {
        self.validate()?;
        validate_prefill_reference_plan(self, plan, sorted_row_count).map_err(|err| {
            err.replace(
                "expert-HQQ prefill reference execution",
                "expert-HQQ BF16-path oracle",
            )
            .replace(
                "expert-HQQ prefill reference",
                "expert-HQQ BF16-path oracle",
            )
        })?;
        let routed_hidden = self.header.routed_hidden_size;
        let intermediate = self.header.moe_intermediate_size;
        let w13_rows = if plan.experts_gated {
            checked_mul(2, intermediate, "BF16-path oracle gated W13 rows")?
        } else {
            intermediate
        };
        let expected_input_len = sorted_row_count
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ BF16-path oracle input byte count overflow".to_string())?;
        if sorted_routed_inputs.len() != expected_input_len {
            return Err(format!(
                "expert-HQQ BF16-path oracle input length {} != sorted_row_count*routed_hidden_size {}",
                sorted_routed_inputs.len(),
                expected_input_len
            ));
        }

        let input_bf16: Vec<f32> = sorted_routed_inputs
            .iter()
            .map(|&value| round_to_bf16_path_f32(value))
            .collect();
        let mut row_claimed = vec![false; sorted_row_count];
        let mut output = vec![0.0f32; expected_input_len];
        let mut w13_preactivation = vec![
            0.0f32;
            sorted_row_count.checked_mul(w13_rows).ok_or_else(
                || "expert-HQQ BF16-path oracle W13 output length overflow".to_string()
            )?
        ];
        let mut activation_out = vec![
            0.0f32;
            sorted_row_count.checked_mul(intermediate).ok_or_else(
                || "expert-HQQ BF16-path oracle activation output length overflow".to_string()
            )?
        ];
        for entry in &plan.entries {
            let row_end = entry
                .row_offset
                .checked_add(entry.row_count)
                .ok_or_else(|| {
                    format!(
                        "expert-HQQ BF16-path oracle row range overflow for expert {}",
                        entry.expert_idx
                    )
                })?;
            if row_end > sorted_row_count {
                return Err(format!(
                    "expert-HQQ BF16-path oracle row range {}..{} exceeds sorted_row_count {} for expert {}",
                    entry.row_offset, row_end, sorted_row_count, entry.expert_idx
                ));
            }
            for row in entry.row_offset..row_end {
                if row_claimed[row] {
                    return Err(format!(
                        "expert-HQQ BF16-path oracle row {row} is claimed by more than one selected expert"
                    ));
                }
                row_claimed[row] = true;
            }

            let w13 = self.require_tensor_record(entry.w13_key)?;
            let w2 = self.require_tensor_record(entry.w2_key)?;
            validate_prefill_reference_entry(self, plan, entry, w13, w2).map_err(|err| {
                err.replace(
                    "expert-HQQ prefill reference execution",
                    "expert-HQQ BF16-path oracle",
                )
                .replace(
                    "expert-HQQ prefill reference",
                    "expert-HQQ BF16-path oracle",
                )
            })?;
            let w13_dequant = dequantize_expert_hqq_record_to_f32(w13)?;
            let w2_dequant = dequantize_expert_hqq_record_to_f32(w2)?;
            let mut activation = vec![0.0f32; intermediate];
            for sorted_row in entry.row_offset..row_end {
                let input_start = sorted_row * routed_hidden;
                let input = &input_bf16[input_start..input_start + routed_hidden];
                let w13_start = sorted_row * w13_rows;
                let activation_start = sorted_row * intermediate;
                if plan.experts_gated {
                    for n in 0..intermediate {
                        let gate = round_to_bf16_path_f32(dot_f32_gpu_lane_order(
                            &w13_dequant[n * routed_hidden..(n + 1) * routed_hidden],
                            input,
                        ));
                        let up_row = intermediate + n;
                        let up = round_to_bf16_path_f32(dot_f32_gpu_lane_order(
                            &w13_dequant[up_row * routed_hidden..(up_row + 1) * routed_hidden],
                            input,
                        ));
                        let act = round_to_bf16_path_f32(silu(gate) * up);
                        w13_preactivation[w13_start + n] = gate;
                        w13_preactivation[w13_start + up_row] = up;
                        activation[n] = act;
                        activation_out[activation_start + n] = act;
                    }
                } else {
                    for n in 0..intermediate {
                        let preact = round_to_bf16_path_f32(dot_f32_gpu_lane_order(
                            &w13_dequant[n * routed_hidden..(n + 1) * routed_hidden],
                            input,
                        ));
                        let relu = preact.max(0.0);
                        let act = round_to_bf16_path_f32(relu * relu);
                        w13_preactivation[w13_start + n] = preact;
                        activation[n] = act;
                        activation_out[activation_start + n] = act;
                    }
                }
                for out_row in 0..routed_hidden {
                    output[input_start + out_row] = round_to_bf16_path_f32(dot_f32_gpu_lane_order(
                        &w2_dequant[out_row * intermediate..(out_row + 1) * intermediate],
                        &activation,
                    ));
                }
            }
        }
        if let Some(row) = row_claimed.iter().position(|claimed| !*claimed) {
            return Err(format!(
                "expert-HQQ BF16-path oracle row {row} has no selected expert plan entry"
            ));
        }

        Ok(ExpertHqqPrefillBf16PathOracleOutput {
            sorted_row_count,
            routed_hidden_size: routed_hidden,
            w13_rows,
            moe_intermediate_size: intermediate,
            input_bf16,
            w13_preactivation,
            activation: activation_out,
            values: output,
        })
    }

    #[cfg(all(test, has_prefill_kernels))]
    pub fn execute_prefill_test_gpu_prototype(
        &self,
        plan: &ExpertHqqPrefillDispatchPlan,
        sorted_routed_inputs: &[f32],
        sorted_row_count: usize,
    ) -> Result<ExpertHqqPrefillGpuPrototypeOutput, String> {
        self.validate()?;
        validate_prefill_test_dispatch_plan(self, plan, sorted_row_count)?;
        let routed_hidden = self.header.routed_hidden_size;
        let intermediate = self.header.moe_intermediate_size;
        let w13_rows = if plan.experts_gated {
            checked_mul(2, intermediate, "GPU prototype gated W13 rows")?
        } else {
            intermediate
        };
        let expected_input_len = sorted_row_count
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ GPU prototype input byte count overflow".to_string())?;
        if sorted_routed_inputs.len() != expected_input_len {
            return Err(format!(
                "expert-HQQ prefill GPU prototype input length {} != sorted_row_count*routed_hidden_size {}",
                sorted_routed_inputs.len(),
                expected_input_len
            ));
        }

        let mut row_claimed = vec![false; sorted_row_count];
        for entry in &plan.entries {
            let row_end = entry
                .row_offset
                .checked_add(entry.row_count)
                .ok_or_else(|| {
                    format!(
                        "expert-HQQ prefill GPU prototype row range overflow for expert {}",
                        entry.expert_idx
                    )
                })?;
            if row_end > sorted_row_count {
                return Err(format!(
                    "expert-HQQ prefill GPU prototype row range {}..{} exceeds sorted_row_count {} for expert {}",
                    entry.row_offset, row_end, sorted_row_count, entry.expert_idx
                ));
            }
            for row in entry.row_offset..row_end {
                if row_claimed[row] {
                    return Err(format!(
                        "expert-HQQ prefill GPU prototype row {row} is claimed by more than one selected expert"
                    ));
                }
                row_claimed[row] = true;
            }
            let w13 = self.require_tensor_record(entry.w13_key)?;
            let w2 = self.require_tensor_record(entry.w2_key)?;
            validate_prefill_test_dispatch_entry(self, plan, entry, w13, w2)?;
        }
        if let Some(row) = row_claimed.iter().position(|claimed| !*claimed) {
            return Err(format!(
                "expert-HQQ prefill GPU prototype row {row} has no selected expert plan entry"
            ));
        }

        let kernels = ExpertHqqGpuPrototypeKernels::new()?;
        let mut output = vec![0.0f32; expected_input_len];
        let mut w13_preactivation = vec![
            0.0f32;
            sorted_row_count.checked_mul(w13_rows).ok_or_else(
                || "expert-HQQ GPU prototype W13 output length overflow".to_string()
            )?
        ];
        let mut activation = vec![
            0.0f32;
            sorted_row_count.checked_mul(intermediate).ok_or_else(|| {
                "expert-HQQ GPU prototype activation output length overflow".to_string()
            })?
        ];
        for entry in &plan.entries {
            let w13 = self.require_tensor_record(entry.w13_key)?;
            let w2 = self.require_tensor_record(entry.w2_key)?;
            kernels.execute_entry(
                self,
                plan,
                entry,
                w13,
                w2,
                sorted_routed_inputs,
                &mut output,
                &mut w13_preactivation,
                &mut activation,
            )?;
        }

        Ok(ExpertHqqPrefillGpuPrototypeOutput {
            sorted_row_count,
            routed_hidden_size: routed_hidden,
            w13_rows,
            moe_intermediate_size: intermediate,
            w13_preactivation,
            activation,
            values: output,
        })
    }

    #[cfg(all(test, has_prefill_kernels))]
    pub fn execute_prefill_runtime_shaped_gpu_prototype(
        &self,
        layer_idx: usize,
        experts_gated: bool,
        runtime_blocks: &[ExpertHqqRuntimePrefillBlock],
        shape: ExpertHqqRuntimePrefillBufferShape,
        runtime_routed_inputs: &[f32],
    ) -> Result<ExpertHqqRuntimeShapedPrefillGpuOutput, String> {
        self.validate()?;
        if runtime_blocks.is_empty() {
            return Err(
                "expert-HQQ runtime-shaped GPU prototype requires at least one block".to_string(),
            );
        }
        if shape.total_sorted_rows == 0 {
            return Err(
                "expert-HQQ runtime-shaped GPU prototype total_sorted_rows must be nonzero"
                    .to_string(),
            );
        }
        let routed_hidden = self.header.routed_hidden_size;
        let intermediate = self.header.moe_intermediate_size;
        let w13_rows = if experts_gated {
            checked_mul(
                2,
                intermediate,
                "runtime-shaped GPU prototype gated W13 rows",
            )?
        } else {
            intermediate
        };
        if shape.input_row_stride < routed_hidden {
            return Err(format!(
                "expert-HQQ runtime-shaped input_row_stride {} < routed_hidden_size {}",
                shape.input_row_stride, routed_hidden
            ));
        }
        if shape.w13_row_stride < w13_rows {
            return Err(format!(
                "expert-HQQ runtime-shaped w13_row_stride {} < w13_rows {}",
                shape.w13_row_stride, w13_rows
            ));
        }
        if shape.activation_row_stride < intermediate {
            return Err(format!(
                "expert-HQQ runtime-shaped activation_row_stride {} < moe_intermediate_size {}",
                shape.activation_row_stride, intermediate
            ));
        }
        if shape.output_row_stride < routed_hidden {
            return Err(format!(
                "expert-HQQ runtime-shaped output_row_stride {} < routed_hidden_size {}",
                shape.output_row_stride, routed_hidden
            ));
        }
        let required_input_len = runtime_buffer_len(
            shape.total_sorted_rows,
            shape.input_row_stride,
            routed_hidden,
        )?;
        if runtime_routed_inputs.len() != required_input_len {
            return Err(format!(
                "expert-HQQ runtime-shaped input length {} != required {} for total_sorted_rows={} input_row_stride={} width={}",
                runtime_routed_inputs.len(),
                required_input_len,
                shape.total_sorted_rows,
                shape.input_row_stride,
                routed_hidden
            ));
        }

        let absolute_works: Vec<ExpertHqqPrefillWork> = runtime_blocks
            .iter()
            .map(|block| {
                ExpertHqqPrefillWork::new(
                    block.expert_idx,
                    block.absolute_row_offset,
                    block.row_count,
                )
            })
            .collect();
        let absolute_plan =
            self.prefill_dispatch_plan(layer_idx, experts_gated, &absolute_works)?;
        let mut claimed_rows = vec![false; shape.total_sorted_rows];
        let mut compact_works = Vec::with_capacity(runtime_blocks.len());
        let mut compact_row_count = 0usize;
        for (idx, block) in runtime_blocks.iter().enumerate() {
            if block.row_count == 0 {
                return Err(format!(
                    "expert-HQQ runtime-shaped block {idx} expert {} has zero rows",
                    block.expert_idx
                ));
            }
            if idx > 0
                && block.absolute_row_offset
                    < runtime_blocks[idx - 1]
                        .absolute_row_offset
                        .saturating_add(runtime_blocks[idx - 1].row_count)
            {
                return Err(format!(
                    "expert-HQQ runtime-shaped blocks must be sorted and non-overlapping: block {idx} starts at {} before previous end {}",
                    block.absolute_row_offset,
                    runtime_blocks[idx - 1].absolute_row_offset + runtime_blocks[idx - 1].row_count
                ));
            }
            let row_end = block
                .absolute_row_offset
                .checked_add(block.row_count)
                .ok_or_else(|| {
                    format!(
                        "expert-HQQ runtime-shaped row range overflow for expert {}",
                        block.expert_idx
                    )
                })?;
            if row_end > shape.total_sorted_rows {
                return Err(format!(
                    "expert-HQQ runtime-shaped row range {}..{} exceeds total_sorted_rows {} for expert {}",
                    block.absolute_row_offset, row_end, shape.total_sorted_rows, block.expert_idx
                ));
            }
            let abs_entry = &absolute_plan.entries[idx];
            if abs_entry.expert_idx != block.expert_idx
                || abs_entry.row_offset != block.absolute_row_offset
                || abs_entry.row_count != block.row_count
            {
                return Err(format!(
                    "expert-HQQ runtime-shaped descriptor plan mismatch for block {idx}: expert/offset/count expected {}/{}/{} got {}/{}/{}",
                    block.expert_idx,
                    block.absolute_row_offset,
                    block.row_count,
                    abs_entry.expert_idx,
                    abs_entry.row_offset,
                    abs_entry.row_count
                ));
            }
            for row in block.absolute_row_offset..row_end {
                if claimed_rows[row] {
                    return Err(format!(
                        "expert-HQQ runtime-shaped row {row} is claimed by more than one block"
                    ));
                }
                claimed_rows[row] = true;
            }
            compact_works.push(ExpertHqqPrefillWork::new(
                block.expert_idx,
                compact_row_count,
                block.row_count,
            ));
            compact_row_count = compact_row_count
                .checked_add(block.row_count)
                .ok_or_else(|| "expert-HQQ runtime-shaped compact row overflow".to_string())?;
        }
        let compact_plan = self.prefill_dispatch_plan(layer_idx, experts_gated, &compact_works)?;
        let compact_input_len = compact_row_count
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ runtime-shaped compact input length overflow".to_string())?;
        let mut compact_inputs = vec![0.0f32; compact_input_len];
        let mut compact_cursor = 0usize;
        for block in runtime_blocks {
            for local_row in 0..block.row_count {
                let absolute_row = block.absolute_row_offset + local_row;
                let runtime_start = absolute_row
                    .checked_mul(shape.input_row_stride)
                    .ok_or_else(|| {
                        "expert-HQQ runtime-shaped input row offset overflow".to_string()
                    })?;
                let compact_start = compact_cursor.checked_mul(routed_hidden).ok_or_else(|| {
                    "expert-HQQ runtime-shaped compact input row offset overflow".to_string()
                })?;
                compact_inputs[compact_start..compact_start + routed_hidden].copy_from_slice(
                    &runtime_routed_inputs[runtime_start..runtime_start + routed_hidden],
                );
                compact_cursor += 1;
            }
        }
        let compact_gpu = self.execute_prefill_test_gpu_prototype(
            &compact_plan,
            &compact_inputs,
            compact_row_count,
        )?;
        let w13_len = runtime_buffer_len(shape.total_sorted_rows, shape.w13_row_stride, w13_rows)?;
        let activation_len = runtime_buffer_len(
            shape.total_sorted_rows,
            shape.activation_row_stride,
            intermediate,
        )?;
        let output_len = runtime_buffer_len(
            shape.total_sorted_rows,
            shape.output_row_stride,
            routed_hidden,
        )?;
        let mut w13_preactivation = vec![f32::NAN; w13_len];
        let mut activation = vec![f32::NAN; activation_len];
        let mut values = vec![f32::NAN; output_len];
        compact_cursor = 0;
        for block in runtime_blocks {
            for local_row in 0..block.row_count {
                let absolute_row = block.absolute_row_offset + local_row;
                scatter_runtime_row(
                    &compact_gpu.w13_preactivation,
                    compact_cursor,
                    w13_rows,
                    &mut w13_preactivation,
                    absolute_row,
                    shape.w13_row_stride,
                )?;
                scatter_runtime_row(
                    &compact_gpu.activation,
                    compact_cursor,
                    intermediate,
                    &mut activation,
                    absolute_row,
                    shape.activation_row_stride,
                )?;
                scatter_runtime_row(
                    &compact_gpu.values,
                    compact_cursor,
                    routed_hidden,
                    &mut values,
                    absolute_row,
                    shape.output_row_stride,
                )?;
                compact_cursor += 1;
            }
        }

        Ok(ExpertHqqRuntimeShapedPrefillGpuOutput {
            total_sorted_rows: shape.total_sorted_rows,
            compact_row_count,
            routed_hidden_size: routed_hidden,
            w13_rows,
            moe_intermediate_size: intermediate,
            input_row_stride: shape.input_row_stride,
            w13_row_stride: shape.w13_row_stride,
            activation_row_stride: shape.activation_row_stride,
            output_row_stride: shape.output_row_stride,
            claimed_rows,
            w13_preactivation,
            activation,
            values,
        })
    }

    pub fn write_to_path(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let mut file = File::create(path)
            .map_err(|e| format!("failed to create expert-HQQ cache {}: {e}", path.display()))?;
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, &mut file);
        self.write_to(&mut writer)?;
        writer
            .flush()
            .map_err(|e| format!("failed to flush expert-HQQ cache {}: {e}", path.display()))
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), String> {
        self.validate()?;
        write_header(writer, &self.header)?;
        for record in &self.tensors {
            write_descriptor(writer, &record.descriptor)?;
        }
        for record in &self.tensors {
            writer
                .write_all(&record.packed)
                .map_err(|e| format!("failed to write expert-HQQ packed payload: {e}"))?;
            writer
                .write_all(&record.scales)
                .map_err(|e| format!("failed to write expert-HQQ scales payload: {e}"))?;
            writer
                .write_all(&record.zeros)
                .map_err(|e| format!("failed to write expert-HQQ zeros payload: {e}"))?;
        }
        Ok(())
    }

    pub fn read_from_path_with_expected(
        path: &Path,
        expected: &ExpertHqqCacheExpectation,
    ) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("failed to read expert-HQQ cache {}: {e}", path.display()))?;
        let mut cursor = Cursor::new(bytes.as_slice());
        let cache = Self::read_from(&mut cursor)?;
        if cursor.position() as usize != bytes.len() {
            return Err(format!(
                "expert-HQQ cache {} has {} trailing bytes",
                path.display(),
                bytes.len() - cursor.position() as usize
            ));
        }
        cache.validate_against(expected)?;
        Ok(cache)
    }

    pub fn read_from_path(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("failed to read expert-HQQ cache {}: {e}", path.display()))?;
        let mut cursor = Cursor::new(bytes.as_slice());
        let cache = Self::read_from(&mut cursor)?;
        if cursor.position() as usize != bytes.len() {
            return Err(format!(
                "expert-HQQ cache {} has {} trailing bytes",
                path.display(),
                bytes.len() - cursor.position() as usize
            ));
        }
        Ok(cache)
    }

    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, String> {
        let header = read_header(reader)?;
        header.validate()?;
        let mut descriptors = Vec::with_capacity(header.tensor_count);
        for _ in 0..header.tensor_count {
            descriptors.push(read_descriptor(reader)?);
        }
        let mut tensors = Vec::with_capacity(header.tensor_count);
        for descriptor in descriptors {
            descriptor.validate(&header)?;
            let mut packed = vec![0u8; descriptor.packed_bytes];
            let mut scales = vec![0u8; descriptor.scales_bytes];
            let mut zeros = vec![0u8; descriptor.zeros_bytes];
            reader
                .read_exact(&mut packed)
                .map_err(|e| format!("failed to read expert-HQQ packed payload: {e}"))?;
            reader
                .read_exact(&mut scales)
                .map_err(|e| format!("failed to read expert-HQQ scales payload: {e}"))?;
            reader
                .read_exact(&mut zeros)
                .map_err(|e| format!("failed to read expert-HQQ zeros payload: {e}"))?;
            tensors.push(ExpertHqqTensorRecord::new(
                descriptor, packed, scales, zeros,
            )?);
        }
        Self::new(header, tensors)
    }
}

pub fn write_expert_hqq_cache_from_inputs(
    path: &Path,
    header: ExpertHqqCacheHeader,
    tensors: Vec<ExpertHqqTensorInput>,
) -> Result<ExpertHqqCache, String> {
    let cache = ExpertHqqCache::from_inputs(header, tensors)?;
    cache.write_to_path(path)?;
    let loaded = ExpertHqqCache::read_from_path_with_expected(path, &cache.header.expectation())?;
    if loaded != cache {
        return Err(format!(
            "expert-HQQ cache readback mismatch after writing {}",
            path.display()
        ));
    }
    Ok(cache)
}

pub fn load_expert_hqq_cache(
    path: &Path,
    expected: &ExpertHqqCacheExpectation,
) -> Result<ExpertHqqCache, String> {
    ExpertHqqCache::read_from_path_with_expected(path, expected)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertHqqTraceStageComparison {
    pub count: usize,
    pub sum_abs: f64,
    pub max_abs: f64,
    pub l2: f64,
    pub mismatch_count: usize,
    pub sum_tolerance: f64,
    pub max_tolerance: f64,
    pub passes: bool,
}

#[derive(Debug, Clone)]
pub struct ExpertHqqTraceMismatchDetail {
    pub case_index: usize,
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub absolute_row_offset: usize,
    pub row_count: usize,
    pub stage: &'static str,
    pub linear_index: usize,
    pub local_row: usize,
    pub absolute_row: usize,
    pub column: usize,
    pub row_width: usize,
    pub actual_bits: u16,
    pub expected_bits: u16,
    pub actual_value: f32,
    pub expected_bf16_value: f32,
    pub expected_raw_value: f32,
    pub delta_abs: f64,
    pub actual_class: &'static str,
    pub expected_bf16_class: &'static str,
    pub expected_raw_class: &'static str,
    pub flush_to_zero_or_subnormal_rounding: bool,
    pub diagnostic: &'static str,
}

#[derive(Debug, Clone)]
struct ExpertHqqStageMismatchDetail {
    linear_index: usize,
    local_row: usize,
    column: usize,
    actual_bits: u16,
    expected_bits: u16,
    actual_value: f32,
    expected_bf16_value: f32,
    expected_raw_value: f32,
    delta_abs: f64,
    actual_class: &'static str,
    expected_bf16_class: &'static str,
    expected_raw_class: &'static str,
    flush_to_zero_or_subnormal_rounding: bool,
    diagnostic: &'static str,
}

#[derive(Debug, Clone)]
pub struct ExpertHqqTraceComparisonCaseReport {
    pub case_index: usize,
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub absolute_row_offset: usize,
    pub row_count: usize,
    pub input_row_width: usize,
    pub w13_row_width: usize,
    pub activation_row_width: usize,
    pub output_row_width: usize,
    pub nbits: u8,
    pub group_size: usize,
    pub hqq_layout: String,
    pub experts_gated: bool,
    pub input: ExpertHqqTraceStageComparison,
    pub w13: ExpertHqqTraceStageComparison,
    pub activation: ExpertHqqTraceStageComparison,
    pub output: ExpertHqqTraceStageComparison,
    pub mismatch_details: Vec<ExpertHqqTraceMismatchDetail>,
    pub passes_contract: bool,
}

#[derive(Debug, Clone)]
pub struct ExpertHqqTraceComparisonReport {
    pub trace_path: PathBuf,
    pub spec_path: PathBuf,
    pub cache_path: PathBuf,
    pub case_reports: Vec<ExpertHqqTraceComparisonCaseReport>,
    pub passes_contract: bool,
}

impl ExpertHqqTraceComparisonReport {
    pub fn case_count(&self) -> usize {
        let mut cases = BTreeSet::new();
        for case in &self.case_reports {
            cases.insert(case.case_index);
        }
        cases.len()
    }

    pub fn block_count(&self) -> usize {
        self.case_reports.len()
    }

    pub fn layer_count(&self) -> usize {
        let mut layers = BTreeSet::new();
        for case in &self.case_reports {
            layers.insert(case.layer_idx);
        }
        layers.len()
    }

    pub fn stage_totals(
        &self,
    ) -> (
        ExpertHqqTraceStageComparison,
        ExpertHqqTraceStageComparison,
        ExpertHqqTraceStageComparison,
        ExpertHqqTraceStageComparison,
    ) {
        let mut input = ExpertHqqTraceStageComparison::default();
        let mut w13 = ExpertHqqTraceStageComparison::default();
        let mut activation = ExpertHqqTraceStageComparison::default();
        let mut output = ExpertHqqTraceStageComparison::default();
        for case in &self.case_reports {
            merge_stage_comparison(&mut input, case.input);
            merge_stage_comparison(&mut w13, case.w13);
            merge_stage_comparison(&mut activation, case.activation);
            merge_stage_comparison(&mut output, case.output);
        }
        input.passes = self.case_reports.iter().all(|case| case.input.passes);
        w13.passes = self.case_reports.iter().all(|case| case.w13.passes);
        activation.passes = self.case_reports.iter().all(|case| case.activation.passes);
        output.passes = self.case_reports.iter().all(|case| case.output.passes);
        (input, w13, activation, output)
    }

    pub fn layer_stage_totals(
        &self,
    ) -> BTreeMap<
        usize,
        (
            ExpertHqqTraceStageComparison,
            ExpertHqqTraceStageComparison,
            ExpertHqqTraceStageComparison,
            ExpertHqqTraceStageComparison,
        ),
    > {
        let mut by_layer: BTreeMap<usize, _> = BTreeMap::new();
        for case in &self.case_reports {
            let entry = by_layer.entry(case.layer_idx).or_insert_with(|| {
                (
                    ExpertHqqTraceStageComparison::default(),
                    ExpertHqqTraceStageComparison::default(),
                    ExpertHqqTraceStageComparison::default(),
                    ExpertHqqTraceStageComparison::default(),
                )
            });
            merge_stage_comparison(&mut entry.0, case.input);
            merge_stage_comparison(&mut entry.1, case.w13);
            merge_stage_comparison(&mut entry.2, case.activation);
            merge_stage_comparison(&mut entry.3, case.output);
        }
        for (layer_idx, (input, w13, activation, output)) in by_layer.iter_mut() {
            input.passes = self
                .case_reports
                .iter()
                .filter(|case| case.layer_idx == *layer_idx)
                .all(|case| case.input.passes);
            w13.passes = self
                .case_reports
                .iter()
                .filter(|case| case.layer_idx == *layer_idx)
                .all(|case| case.w13.passes);
            activation.passes = self
                .case_reports
                .iter()
                .filter(|case| case.layer_idx == *layer_idx)
                .all(|case| case.activation.passes);
            output.passes = self
                .case_reports
                .iter()
                .filter(|case| case.layer_idx == *layer_idx)
                .all(|case| case.output.passes);
        }
        by_layer
    }
}

fn merge_stage_comparison(
    dst: &mut ExpertHqqTraceStageComparison,
    src: ExpertHqqTraceStageComparison,
) {
    dst.count += src.count;
    dst.sum_abs += src.sum_abs;
    dst.max_abs = dst.max_abs.max(src.max_abs);
    let l2_sq = dst.l2 * dst.l2 + src.l2 * src.l2;
    dst.l2 = l2_sq.sqrt();
    dst.mismatch_count += src.mismatch_count;
    dst.sum_tolerance += src.sum_tolerance;
    dst.max_tolerance = dst.max_tolerance.max(src.max_tolerance);
}

#[derive(Debug, Clone)]
struct ExpertHqqTraceFullBufferStage {
    stage_name: String,
    layer_idx: usize,
    expert_idx: usize,
    absolute_row_offset: usize,
    row_count: usize,
    row_width: usize,
    nbits: u8,
    group_size: usize,
    hqq_layout: String,
    experts_gated: bool,
    bits: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertHqqTraceBlockKey {
    layer_idx: usize,
    expert_idx: usize,
    absolute_row_offset: usize,
    row_count: usize,
    nbits: u8,
    group_size: usize,
    hqq_layout: String,
    experts_gated: bool,
}

impl ExpertHqqTraceBlockKey {
    fn describe(&self) -> String {
        format!(
            "layer={} expert={} offset={} rows={} nbits={} group={} layout={} gated={}",
            self.layer_idx,
            self.expert_idx,
            self.absolute_row_offset,
            self.row_count,
            self.nbits,
            self.group_size,
            self.hqq_layout,
            self.experts_gated,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertHqqTraceFilterKey {
    case_index: usize,
    layer_idx: usize,
    expert_idx: usize,
    absolute_row_offset: usize,
    row_count: usize,
}

impl ExpertHqqTraceFilterKey {
    fn describe(&self) -> String {
        format!(
            "case={} layer={} expert={} offset={} rows={}",
            self.case_index,
            self.layer_idx,
            self.expert_idx,
            self.absolute_row_offset,
            self.row_count
        )
    }
}

fn filter_key_for_block(
    case_index: usize,
    key: &ExpertHqqTraceBlockKey,
) -> ExpertHqqTraceFilterKey {
    ExpertHqqTraceFilterKey {
        case_index,
        layer_idx: key.layer_idx,
        expert_idx: key.expert_idx,
        absolute_row_offset: key.absolute_row_offset,
        row_count: key.row_count,
    }
}

fn read_expert_hqq_trace_failure_filter(
    path: &Path,
) -> Result<BTreeSet<ExpertHqqTraceFilterKey>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "failed to read expert-HQQ trace failure filter {}: {e}",
            path.display()
        )
    })?;
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| "expert-HQQ trace failure filter is empty".to_string())?;
    let columns = header.split('\t').collect::<Vec<_>>();
    let col = |name: &str| -> Result<usize, String> {
        columns.iter().position(|&col| col == name).ok_or_else(|| {
            format!(
                "expert-HQQ trace failure filter {} missing column {name}",
                path.display()
            )
        })
    };
    let case_col = col("case_index")?;
    let layer_col = col("layer")?;
    let expert_col = col("expert")?;
    let offset_col = col("absolute_row_offset")?;
    let rows_col = col("row_count")?;
    let stage_col = col("stage")?;
    let passes_col = col("passes")?;

    let mut filter = BTreeSet::new();
    for (line_idx, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != columns.len() {
            return Err(format!(
                "expert-HQQ trace failure filter {} line {} column count {} != header {}",
                path.display(),
                line_idx + 2,
                fields.len(),
                columns.len()
            ));
        }
        if fields[passes_col] != "false" || fields[stage_col] != "activation" {
            continue;
        }
        let Ok(case_index) = fields[case_col].parse::<usize>() else {
            continue;
        };
        let key = ExpertHqqTraceFilterKey {
            case_index,
            layer_idx: fields[layer_col].parse::<usize>().map_err(|e| {
                format!(
                    "expert-HQQ trace failure filter {} line {} invalid layer: {e}",
                    path.display(),
                    line_idx + 2
                )
            })?,
            expert_idx: fields[expert_col].parse::<usize>().map_err(|e| {
                format!(
                    "expert-HQQ trace failure filter {} line {} invalid expert: {e}",
                    path.display(),
                    line_idx + 2
                )
            })?,
            absolute_row_offset: fields[offset_col].parse::<usize>().map_err(|e| {
                format!(
                    "expert-HQQ trace failure filter {} line {} invalid absolute_row_offset: {e}",
                    path.display(),
                    line_idx + 2
                )
            })?,
            row_count: fields[rows_col].parse::<usize>().map_err(|e| {
                format!(
                    "expert-HQQ trace failure filter {} line {} invalid row_count: {e}",
                    path.display(),
                    line_idx + 2
                )
            })?,
        };
        if !filter.insert(key) {
            return Err(format!(
                "expert-HQQ trace failure filter {} line {} duplicates {}",
                path.display(),
                line_idx + 2,
                key.describe()
            ));
        }
    }
    if filter.is_empty() {
        return Err(format!(
            "expert-HQQ trace failure filter {} selected no failing activation blocks",
            path.display()
        ));
    }
    Ok(filter)
}

impl ExpertHqqTraceFullBufferStage {
    fn block_key(&self) -> ExpertHqqTraceBlockKey {
        ExpertHqqTraceBlockKey {
            layer_idx: self.layer_idx,
            expert_idx: self.expert_idx,
            absolute_row_offset: self.absolute_row_offset,
            row_count: self.row_count,
            nbits: self.nbits,
            group_size: self.group_size,
            hqq_layout: self.hqq_layout.clone(),
            experts_gated: self.experts_gated,
        }
    }
}

pub fn compare_expert_hqq_runtime_prefill_trace_paths(
    trace_path: &Path,
    spec_path: &Path,
    output_tsv_path: Option<&Path>,
) -> Result<ExpertHqqTraceComparisonReport, String> {
    compare_expert_hqq_runtime_prefill_trace_paths_impl(
        trace_path,
        spec_path,
        output_tsv_path,
        None,
        None,
    )
}

pub fn compare_expert_hqq_runtime_prefill_trace_paths_with_mismatch_details(
    trace_path: &Path,
    spec_path: &Path,
    output_tsv_path: Option<&Path>,
    mismatch_detail_tsv_path: Option<&Path>,
) -> Result<ExpertHqqTraceComparisonReport, String> {
    compare_expert_hqq_runtime_prefill_trace_paths_impl(
        trace_path,
        spec_path,
        output_tsv_path,
        mismatch_detail_tsv_path,
        None,
    )
}

pub fn compare_expert_hqq_runtime_prefill_trace_paths_filtered_by_failure_rows(
    trace_path: &Path,
    spec_path: &Path,
    failure_rows_tsv_path: &Path,
    output_tsv_path: Option<&Path>,
    mismatch_detail_tsv_path: Option<&Path>,
) -> Result<ExpertHqqTraceComparisonReport, String> {
    let filter = read_expert_hqq_trace_failure_filter(failure_rows_tsv_path)?;
    compare_expert_hqq_runtime_prefill_trace_paths_impl(
        trace_path,
        spec_path,
        output_tsv_path,
        mismatch_detail_tsv_path,
        Some(&filter),
    )
}

#[derive(Debug, Clone)]
pub struct ExpertHqqExactRowAttributionReport {
    pub response_path: PathBuf,
    pub spec_path: PathBuf,
    pub cache_path: PathBuf,
    pub layer_idx: usize,
    pub requested_expert_idx: usize,
    pub requested_sorted_row: usize,
    pub requested_col: usize,
    pub captured_rows: usize,
    pub selected_contributors: usize,
    pub max_hqq_gpu_vs_krhq_output_abs: f64,
    pub max_bf16_vs_krhq_output_abs: f64,
    pub selected_bf16_value: f64,
    pub selected_hqq_gpu_value: f64,
    pub selected_krhq_value: f64,
    pub selected_bf16_vs_hqq_gpu_abs: f64,
    pub selected_bf16_vs_krhq_abs: f64,
    pub selected_hqq_gpu_vs_krhq_abs: f64,
    pub attribution: String,
}

#[derive(Debug, Clone, Copy)]
struct ExpertHqqExactRowStageDelta {
    count: usize,
    sum_abs: f64,
    max_abs: f64,
    mismatch_count: usize,
}

#[derive(Debug, Clone)]
struct ExpertHqqParsedExactRow {
    absolute_row: usize,
    expert_idx: usize,
    is_requested_worst_global_row: bool,
    is_selected_contributor: bool,
    gather_src: Option<usize>,
    gather_weight: Option<f32>,
    bf16_input_bits: Vec<u16>,
    bf16_w13_bits: Vec<u16>,
    bf16_activation_bits: Vec<u16>,
    bf16_output_bits: Vec<u16>,
    hqq_input_bits: Vec<u16>,
    hqq_w13_bits: Vec<u16>,
    hqq_activation_bits: Vec<u16>,
    hqq_output_bits: Vec<u16>,
}

pub fn attribute_expert_hqq_exact_row_trace_paths(
    response_path: &Path,
    spec_path: &Path,
    output_tsv_path: &Path,
    details_json_path: Option<&Path>,
) -> Result<ExpertHqqExactRowAttributionReport, String> {
    let response_path = std::fs::canonicalize(response_path).map_err(|e| {
        format!(
            "failed to resolve expert-HQQ exact-row response {}: {e}",
            response_path.display()
        )
    })?;
    let spec = load_expert_hqq_diagnostic_cache_spec(spec_path)?;
    let cache = ExpertHqqCache::read_from_path(&spec.cache_path)?;
    spec.validate_cache_descriptors(&cache)?;
    cache.validate_required_tensors(&spec.required_tensors)?;

    let response_file = File::open(&response_path).map_err(|e| {
        format!(
            "failed to open expert-HQQ exact-row response {}: {e}",
            response_path.display()
        )
    })?;
    let response_reader = BufReader::with_capacity(1024 * 1024, response_file);
    let parsed: serde_json::Value = serde_json::from_reader(response_reader).map_err(|e| {
        format!(
            "malformed expert-HQQ exact-row response {}: {e}",
            response_path.display()
        )
    })?;
    let captures = find_expert_hqq_exact_row_capture_metadata(&parsed)?;
    if captures.len() != 1 {
        return Err(format!(
            "expert-HQQ exact-row attribution expected exactly one capture stage, found {}",
            captures.len()
        ));
    }
    let metadata = captures[0];
    if metadata.get("available").and_then(|v| v.as_bool()) != Some(true) {
        return Err(format!(
            "expert-HQQ exact-row capture is unavailable: {}",
            metadata
        ));
    }
    let layer_idx = required_json_usize(metadata, "layer_idx")?;
    let requested_expert_idx = required_json_usize(metadata, "expert")?;
    let requested_sorted_row = required_json_usize(metadata, "requested_sorted_row")?;
    let requested_col = required_json_usize(metadata, "requested_col")?;
    let hidden_size = required_json_usize(metadata, "hidden_size")?;
    let w13_rows = required_json_usize(metadata, "w13_rows")?;
    let intermediate_size = required_json_usize(metadata, "intermediate_size")?;
    let experts_gated = metadata
        .get("experts_gated")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| "expert-HQQ exact-row capture missing experts_gated".to_string())?;
    let rows_json = metadata
        .get("rows")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "expert-HQQ exact-row capture missing rows array".to_string())?;
    if rows_json.is_empty() {
        return Err("expert-HQQ exact-row capture rows array is empty".to_string());
    }
    let mut rows = Vec::with_capacity(rows_json.len());
    for row_json in rows_json {
        rows.push(parse_expert_hqq_exact_row(row_json)?);
    }
    if !rows
        .iter()
        .any(|row| row.absolute_row == requested_sorted_row)
    {
        return Err(format!(
            "expert-HQQ exact-row capture missing requested sorted row {}",
            requested_sorted_row
        ));
    }
    if !rows.iter().any(|row| row.is_selected_contributor) {
        return Err("expert-HQQ exact-row capture has no selected-row contributors".to_string());
    }

    let mut tsv = String::new();
    tsv.push_str("kind\tabsolute_row\texpert\tis_requested_worst_global_row\tis_selected_contributor\tgather_src\tgather_weight\tcol\tstage\tcount\tsum_abs\tmax_abs\tmismatch_count\tbf16_value_at_col\thqq_gpu_value_at_col\tkrhq_value_at_col\tbf16_vs_hqq_gpu_abs_at_col\tbf16_vs_krhq_abs_at_col\thqq_gpu_vs_krhq_abs_at_col\n");
    let mut details_rows = Vec::new();
    let mut max_hqq_gpu_vs_krhq_output_abs = 0.0f64;
    let mut max_bf16_vs_krhq_output_abs = 0.0f64;
    let mut selected_bf16_value = 0.0f64;
    let mut selected_hqq_gpu_value = 0.0f64;
    let mut selected_krhq_value = 0.0f64;
    let mut selected_weighted = 0usize;

    for row in &rows {
        validate_exact_row_widths(row, hidden_size, w13_rows, intermediate_size)?;
        let work = [ExpertHqqPrefillWork::new(row.expert_idx, 0, 1)];
        let plan = cache.prefill_dispatch_plan(layer_idx, experts_gated, &work)?;
        let input_values: Vec<f32> = row
            .bf16_input_bits
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect();
        let oracle = cache.execute_prefill_bf16_path_oracle(&plan, &input_values, 1)?;
        let krhq_input_bits: Vec<u16> = oracle.input_bf16.iter().map(|&v| f32_to_bf16(v)).collect();
        let krhq_w13_bits: Vec<u16> = oracle
            .w13_preactivation
            .iter()
            .map(|&v| f32_to_bf16(v))
            .collect();
        let krhq_activation_bits: Vec<u16> =
            oracle.activation.iter().map(|&v| f32_to_bf16(v)).collect();
        let krhq_output_bits: Vec<u16> = oracle.values.iter().map(|&v| f32_to_bf16(v)).collect();

        let input_delta = exact_row_stage_delta(&row.hqq_input_bits, &krhq_input_bits)?;
        let w13_delta = exact_row_stage_delta(&row.hqq_w13_bits, &krhq_w13_bits)?;
        let activation_delta =
            exact_row_stage_delta(&row.hqq_activation_bits, &krhq_activation_bits)?;
        let output_delta = exact_row_stage_delta(&row.hqq_output_bits, &krhq_output_bits)?;
        max_hqq_gpu_vs_krhq_output_abs = max_hqq_gpu_vs_krhq_output_abs.max(output_delta.max_abs);
        let bf16_vs_krhq_output = exact_row_stage_delta(&row.bf16_output_bits, &krhq_output_bits)?;
        max_bf16_vs_krhq_output_abs = max_bf16_vs_krhq_output_abs.max(bf16_vs_krhq_output.max_abs);

        let bf16_col = bf16_to_f32(*row.bf16_output_bits.get(requested_col).ok_or_else(|| {
            format!(
                "expert-HQQ exact-row BF16 output missing requested col {} for row {}",
                requested_col, row.absolute_row
            )
        })?) as f64;
        let hqq_col = bf16_to_f32(*row.hqq_output_bits.get(requested_col).ok_or_else(|| {
            format!(
                "expert-HQQ exact-row HQQ output missing requested col {} for row {}",
                requested_col, row.absolute_row
            )
        })?) as f64;
        let krhq_col = bf16_to_f32(*krhq_output_bits.get(requested_col).ok_or_else(|| {
            format!(
                "expert-HQQ exact-row KRHQ output missing requested col {} for row {}",
                requested_col, row.absolute_row
            )
        })?) as f64;
        let bf16_vs_hqq_col = (bf16_col - hqq_col).abs();
        let bf16_vs_krhq_col = (bf16_col - krhq_col).abs();
        let hqq_vs_krhq_col = (hqq_col - krhq_col).abs();

        for (stage, delta) in [
            ("input", input_delta),
            ("w13", w13_delta),
            ("activation", activation_delta),
            ("output", output_delta),
        ] {
            tsv.push_str(&format!(
                "row\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.18e}\t{:.18e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                row.absolute_row,
                row.expert_idx,
                row.is_requested_worst_global_row,
                row.is_selected_contributor,
                row.gather_src
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "".to_string()),
                row.gather_weight
                    .map(|v| format!("{v:.18e}"))
                    .unwrap_or_else(|| "".to_string()),
                requested_col,
                stage,
                delta.count,
                delta.sum_abs,
                delta.max_abs,
                delta.mismatch_count,
                if stage == "output" {
                    format!("{bf16_col:.18e}")
                } else {
                    "".to_string()
                },
                if stage == "output" {
                    format!("{hqq_col:.18e}")
                } else {
                    "".to_string()
                },
                if stage == "output" {
                    format!("{krhq_col:.18e}")
                } else {
                    "".to_string()
                },
                if stage == "output" {
                    format!("{bf16_vs_hqq_col:.18e}")
                } else {
                    "".to_string()
                },
                if stage == "output" {
                    format!("{bf16_vs_krhq_col:.18e}")
                } else {
                    "".to_string()
                },
                if stage == "output" {
                    format!("{hqq_vs_krhq_col:.18e}")
                } else {
                    "".to_string()
                },
            ));
        }

        if row.is_selected_contributor {
            let weight = row.gather_weight.ok_or_else(|| {
                format!(
                    "expert-HQQ exact-row selected contributor row {} missing gather_weight",
                    row.absolute_row
                )
            })? as f64;
            selected_weighted += 1;
            selected_bf16_value += bf16_col * weight;
            selected_hqq_gpu_value += hqq_col * weight;
            selected_krhq_value += krhq_col * weight;
        }

        details_rows.push(exact_row_dequant_details_json(
            &cache,
            layer_idx,
            row,
            experts_gated,
            requested_col,
            &oracle,
        )?);
    }
    if selected_weighted == 0 {
        return Err(
            "expert-HQQ exact-row attribution had zero weighted selected contributors".to_string(),
        );
    }
    let selected_bf16_vs_hqq_gpu_abs = (selected_bf16_value - selected_hqq_gpu_value).abs();
    let selected_bf16_vs_krhq_abs = (selected_bf16_value - selected_krhq_value).abs();
    let selected_hqq_gpu_vs_krhq_abs = (selected_hqq_gpu_value - selected_krhq_value).abs();
    let attribution = if max_hqq_gpu_vs_krhq_output_abs == 0.0 {
        "hqq_gpu_matches_krhq_dequant_math_at_output; divergence is attributable to HQQ6 dequantized-weight math versus captured BF16 expert output".to_string()
    } else {
        "hqq_gpu_differs_from_krhq_dequant_math_at_output; investigate GPU/cache/layout before attributing to quantization".to_string()
    };
    tsv.push_str(&format!(
        "selected_routed\t\t\t\t\t\t\t{}\toutput\t{}\t\t\t\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\n",
        requested_col,
        selected_weighted,
        selected_bf16_value,
        selected_hqq_gpu_value,
        selected_krhq_value,
        selected_bf16_vs_hqq_gpu_abs,
        selected_bf16_vs_krhq_abs,
        selected_hqq_gpu_vs_krhq_abs,
    ));
    std::fs::write(output_tsv_path, tsv).map_err(|e| {
        format!(
            "failed to write expert-HQQ exact-row attribution TSV {}: {e}",
            output_tsv_path.display()
        )
    })?;
    if let Some(path) = details_json_path {
        let details = serde_json::json!({
            "response_path": response_path,
            "spec_path": spec.spec_path,
            "cache_path": spec.cache_path,
            "layer_idx": layer_idx,
            "requested_expert_idx": requested_expert_idx,
            "requested_sorted_row": requested_sorted_row,
            "requested_col": requested_col,
            "experts_gated": experts_gated,
            "rows": details_rows,
            "selected_routed": {
                "contributor_rows": selected_weighted,
                "bf16_value": selected_bf16_value,
                "hqq_gpu_value": selected_hqq_gpu_value,
                "krhq_dequant_value": selected_krhq_value,
                "bf16_vs_hqq_gpu_abs": selected_bf16_vs_hqq_gpu_abs,
                "bf16_vs_krhq_abs": selected_bf16_vs_krhq_abs,
                "hqq_gpu_vs_krhq_abs": selected_hqq_gpu_vs_krhq_abs,
            },
            "attribution": attribution,
        });
        let json = serde_json::to_string_pretty(&details)
            .map_err(|e| format!("failed to serialize expert-HQQ exact-row details: {e}"))?;
        std::fs::write(path, json).map_err(|e| {
            format!(
                "failed to write expert-HQQ exact-row details JSON {}: {e}",
                path.display()
            )
        })?;
    }

    Ok(ExpertHqqExactRowAttributionReport {
        response_path,
        spec_path: spec.spec_path,
        cache_path: spec.cache_path,
        layer_idx,
        requested_expert_idx,
        requested_sorted_row,
        requested_col,
        captured_rows: rows.len(),
        selected_contributors: selected_weighted,
        max_hqq_gpu_vs_krhq_output_abs,
        max_bf16_vs_krhq_output_abs,
        selected_bf16_value,
        selected_hqq_gpu_value,
        selected_krhq_value,
        selected_bf16_vs_hqq_gpu_abs,
        selected_bf16_vs_krhq_abs,
        selected_hqq_gpu_vs_krhq_abs,
        attribution,
    })
}

fn find_expert_hqq_exact_row_capture_metadata(
    parsed: &serde_json::Value,
) -> Result<Vec<&serde_json::Value>, String> {
    let mut captures = Vec::new();
    if let Some(snapshots) = parsed
        .get("debug_reference_trace")
        .and_then(|v| v.get("prefill_stage_trace"))
        .and_then(|v| v.get("prefill_stage_snapshots"))
        .and_then(|v| v.as_array())
    {
        collect_expert_hqq_exact_row_capture_metadata(snapshots, &mut captures)?;
    }
    if let Some(results) = parsed.get("results").and_then(|v| v.as_array()) {
        for (case_idx, result) in results.iter().enumerate() {
            let snapshots = result
                .get("response")
                .and_then(|v| v.get("debug_reference_trace"))
                .and_then(|v| v.get("prefill_stage_trace"))
                .and_then(|v| v.get("prefill_stage_snapshots"))
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    format!("expert-HQQ exact-row response result {case_idx} missing snapshots")
                })?;
            collect_expert_hqq_exact_row_capture_metadata(snapshots, &mut captures)?;
        }
    }
    Ok(captures)
}

fn collect_expert_hqq_exact_row_capture_metadata<'a>(
    snapshots: &'a [serde_json::Value],
    captures: &mut Vec<&'a serde_json::Value>,
) -> Result<(), String> {
    for snapshot in snapshots {
        let stage = snapshot.get("stage").and_then(|v| v.as_str()).unwrap_or("");
        if stage.ends_with("expert_hqq_exact_row_quantization_attribution") {
            let metadata = snapshot
                .get("metadata")
                .ok_or_else(|| format!("expert-HQQ exact-row snapshot {stage} missing metadata"))?;
            captures.push(metadata);
        }
    }
    Ok(())
}

fn required_json_usize(value: &serde_json::Value, field: &str) -> Result<usize, String> {
    value
        .get(field)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("expert-HQQ exact-row capture missing numeric field {field}"))
}

fn required_nested_usize(value: &serde_json::Value, path: &[&str]) -> Result<usize, String> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key).ok_or_else(|| {
            format!(
                "expert-HQQ exact-row capture missing nested field {}",
                path.join(".")
            )
        })?;
    }
    cursor.as_u64().map(|v| v as usize).ok_or_else(|| {
        format!(
            "expert-HQQ exact-row capture nested field {} is not numeric",
            path.join(".")
        )
    })
}

fn optional_nested_usize(
    value: &serde_json::Value,
    path: &[&str],
) -> Result<Option<usize>, String> {
    let mut cursor = value;
    for key in path {
        let Some(next) = cursor.get(*key) else {
            return Ok(None);
        };
        if next.is_null() {
            return Ok(None);
        }
        cursor = next;
    }
    cursor.as_u64().map(|v| Some(v as usize)).ok_or_else(|| {
        format!(
            "expert-HQQ exact-row capture nested field {} is not numeric/null",
            path.join(".")
        )
    })
}

fn optional_nested_f32(value: &serde_json::Value, path: &[&str]) -> Result<Option<f32>, String> {
    let mut cursor = value;
    for key in path {
        let Some(next) = cursor.get(*key) else {
            return Ok(None);
        };
        if next.is_null() {
            return Ok(None);
        }
        cursor = next;
    }
    cursor.as_f64().map(|v| Some(v as f32)).ok_or_else(|| {
        format!(
            "expert-HQQ exact-row capture nested field {} is not numeric/null",
            path.join(".")
        )
    })
}

fn required_bf16_bits_array(value: &serde_json::Value, path: &[&str]) -> Result<Vec<u16>, String> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key).ok_or_else(|| {
            format!(
                "expert-HQQ exact-row capture missing BF16 bits field {}",
                path.join(".")
            )
        })?;
    }
    let array = cursor.as_array().ok_or_else(|| {
        format!(
            "expert-HQQ exact-row capture BF16 bits field {} is not an array",
            path.join(".")
        )
    })?;
    let mut out = Vec::with_capacity(array.len());
    for (idx, item) in array.iter().enumerate() {
        let raw = item.as_u64().ok_or_else(|| {
            format!(
                "expert-HQQ exact-row capture BF16 bits field {}[{}] is not numeric",
                path.join("."),
                idx
            )
        })?;
        if raw > u16::MAX as u64 {
            return Err(format!(
                "expert-HQQ exact-row capture BF16 bits field {}[{}] value {} exceeds u16",
                path.join("."),
                idx,
                raw
            ));
        }
        out.push(raw as u16);
    }
    Ok(out)
}

fn parse_expert_hqq_exact_row(
    row_json: &serde_json::Value,
) -> Result<ExpertHqqParsedExactRow, String> {
    let absolute_row = required_nested_usize(row_json, &["row_index", "absolute_row"])?;
    let expert_idx = required_nested_usize(row_json, &["row_index", "expert"])?;
    let is_requested_worst_global_row = row_json
        .pointer("/row_index/is_requested_worst_global_row")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| {
            "expert-HQQ exact-row capture row missing is_requested_worst_global_row".to_string()
        })?;
    let is_selected_contributor = row_json
        .pointer("/row_index/is_selected_trace_row_contributor")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| {
            "expert-HQQ exact-row capture row missing is_selected_trace_row_contributor".to_string()
        })?;
    let hqq_gpu = row_json
        .get("hqq_gpu")
        .filter(|value| !value.is_null())
        .ok_or_else(|| {
            format!(
                "expert-HQQ exact-row capture row {} missing hqq_gpu payload",
                absolute_row
            )
        })?;
    Ok(ExpertHqqParsedExactRow {
        absolute_row,
        expert_idx,
        is_requested_worst_global_row,
        is_selected_contributor,
        gather_src: optional_nested_usize(row_json, &["row_index", "gather_src"])?,
        gather_weight: optional_nested_f32(row_json, &["row_index", "gather_weight"])?,
        bf16_input_bits: required_bf16_bits_array(row_json, &["bf16", "input", "bf16_bits_u16"])?,
        bf16_w13_bits: required_bf16_bits_array(row_json, &["bf16", "w13", "bf16_bits_u16"])?,
        bf16_activation_bits: required_bf16_bits_array(
            row_json,
            &["bf16", "activation", "bf16_bits_u16"],
        )?,
        bf16_output_bits: required_bf16_bits_array(row_json, &["bf16", "output", "bf16_bits_u16"])?,
        hqq_input_bits: required_bf16_bits_array(hqq_gpu, &["input", "bf16_bits_u16"])?,
        hqq_w13_bits: required_bf16_bits_array(hqq_gpu, &["w13", "bf16_bits_u16"])?,
        hqq_activation_bits: required_bf16_bits_array(hqq_gpu, &["activation", "bf16_bits_u16"])?,
        hqq_output_bits: required_bf16_bits_array(hqq_gpu, &["output", "bf16_bits_u16"])?,
    })
}

fn validate_exact_row_widths(
    row: &ExpertHqqParsedExactRow,
    hidden_size: usize,
    w13_rows: usize,
    intermediate_size: usize,
) -> Result<(), String> {
    let checks = [
        ("bf16_input", row.bf16_input_bits.len(), hidden_size),
        ("bf16_w13", row.bf16_w13_bits.len(), w13_rows),
        (
            "bf16_activation",
            row.bf16_activation_bits.len(),
            intermediate_size,
        ),
        ("bf16_output", row.bf16_output_bits.len(), hidden_size),
        ("hqq_input", row.hqq_input_bits.len(), hidden_size),
        ("hqq_w13", row.hqq_w13_bits.len(), w13_rows),
        (
            "hqq_activation",
            row.hqq_activation_bits.len(),
            intermediate_size,
        ),
        ("hqq_output", row.hqq_output_bits.len(), hidden_size),
    ];
    for (label, actual, expected) in checks {
        if actual != expected {
            return Err(format!(
                "expert-HQQ exact-row {label} width {} != expected {} for row {}",
                actual, expected, row.absolute_row
            ));
        }
    }
    Ok(())
}

fn exact_row_stage_delta(
    actual_bits: &[u16],
    expected_bits: &[u16],
) -> Result<ExpertHqqExactRowStageDelta, String> {
    if actual_bits.len() != expected_bits.len() {
        return Err(format!(
            "expert-HQQ exact-row stage length mismatch actual={} expected={}",
            actual_bits.len(),
            expected_bits.len()
        ));
    }
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut mismatch_count = 0usize;
    for (&actual_bits, &expected_bits) in actual_bits.iter().zip(expected_bits) {
        if actual_bits != expected_bits {
            mismatch_count += 1;
        }
        let delta = (bf16_to_f32(actual_bits) - bf16_to_f32(expected_bits)).abs() as f64;
        sum_abs += delta;
        max_abs = max_abs.max(delta);
    }
    Ok(ExpertHqqExactRowStageDelta {
        count: actual_bits.len(),
        sum_abs,
        max_abs,
        mismatch_count,
    })
}

fn exact_row_dequant_details_json(
    cache: &ExpertHqqCache,
    layer_idx: usize,
    row: &ExpertHqqParsedExactRow,
    experts_gated: bool,
    requested_col: usize,
    oracle: &ExpertHqqPrefillBf16PathOracleOutput,
) -> Result<serde_json::Value, String> {
    let w2 = cache.require_tensor_record(ExpertHqqTensorKey::new(
        ExpertHqqTensorRole::W2,
        layer_idx,
        row.expert_idx,
    ))?;
    let w2_dequant = dequantize_expert_hqq_record_to_f32(w2)?;
    let intermediate = oracle.moe_intermediate_size;
    let hidden = oracle.routed_hidden_size;
    if requested_col >= hidden {
        return Err(format!(
            "expert-HQQ exact-row requested col {} out of hidden {}",
            requested_col, hidden
        ));
    }
    let w2_start = requested_col
        .checked_mul(intermediate)
        .ok_or_else(|| "expert-HQQ exact-row W2 row offset overflow".to_string())?;
    let mut w2_terms = Vec::with_capacity(intermediate);
    for idx in 0..intermediate {
        let activation = oracle.activation[idx];
        let weight = w2_dequant[w2_start + idx];
        w2_terms.push((idx, activation, weight, activation * weight));
    }
    w2_terms.sort_by(|a, b| b.3.abs().total_cmp(&a.3.abs()));
    let top_w2: Vec<_> = w2_terms
        .iter()
        .take(16)
        .map(|&(idx, activation, weight, contribution)| {
            serde_json::json!({
                "activation_index": idx,
                "activation_value": activation as f64,
                "activation_bf16_bits": format!("0x{:04x}", f32_to_bf16(activation)),
                "w2_dequant_value": weight as f64,
                "w2_dequant_f32_bits": format!("0x{:08x}", weight.to_bits()),
                "contribution": contribution as f64,
            })
        })
        .collect();
    let mut w13_top = serde_json::Value::Null;
    if !experts_gated {
        if let Some((activation_idx, _, _, _)) = w2_terms.first().copied() {
            let w13 = cache.require_tensor_record(ExpertHqqTensorKey::new(
                ExpertHqqTensorRole::W13,
                layer_idx,
                row.expert_idx,
            ))?;
            let w13_dequant = dequantize_expert_hqq_record_to_f32(w13)?;
            let w13_start = activation_idx
                .checked_mul(hidden)
                .ok_or_else(|| "expert-HQQ exact-row W13 row offset overflow".to_string())?;
            let mut w13_terms = Vec::with_capacity(hidden);
            for idx in 0..hidden {
                let input = bf16_to_f32(row.bf16_input_bits[idx]);
                let weight = w13_dequant[w13_start + idx];
                w13_terms.push((idx, input, weight, input * weight));
            }
            w13_terms.sort_by(|a, b| b.3.abs().total_cmp(&a.3.abs()));
            w13_top = serde_json::json!({
                "activation_index": activation_idx,
                "top_input_weight_contributors": w13_terms.iter().take(16).map(|&(idx, input, weight, contribution)| {
                    serde_json::json!({
                        "input_index": idx,
                        "input_value": input as f64,
                        "input_bf16_bits": format!("0x{:04x}", row.bf16_input_bits[idx]),
                        "w13_dequant_value": weight as f64,
                        "w13_dequant_f32_bits": format!("0x{:08x}", weight.to_bits()),
                        "contribution": contribution as f64,
                    })
                }).collect::<Vec<_>>(),
            });
        }
    }
    Ok(serde_json::json!({
        "absolute_row": row.absolute_row,
        "expert": row.expert_idx,
        "is_requested_worst_global_row": row.is_requested_worst_global_row,
        "is_selected_contributor": row.is_selected_contributor,
        "gather_src": row.gather_src,
        "gather_weight": row.gather_weight.map(|v| v as f64),
        "requested_col": requested_col,
        "bf16_output_col": bf16_to_f32(row.bf16_output_bits[requested_col]) as f64,
        "hqq_gpu_output_col": bf16_to_f32(row.hqq_output_bits[requested_col]) as f64,
        "krhq_dequant_output_col": oracle.values[requested_col] as f64,
        "krhq_dequant_output_col_bf16_bits": format!("0x{:04x}", f32_to_bf16(oracle.values[requested_col])),
        "w2_requested_col_top_dequant_contributors": top_w2,
        "w13_top_dequant_contributors_for_largest_w2_term": w13_top,
    }))
}

fn compare_expert_hqq_runtime_prefill_trace_paths_impl(
    trace_path: &Path,
    spec_path: &Path,
    output_tsv_path: Option<&Path>,
    mismatch_detail_tsv_path: Option<&Path>,
    block_filter: Option<&BTreeSet<ExpertHqqTraceFilterKey>>,
) -> Result<ExpertHqqTraceComparisonReport, String> {
    let total_started = Instant::now();
    let profile_enabled = std::env::var_os("KRASIS_EXPERT_HQQ_TRACE_COMPARE_PROFILE").is_some();
    let trace_path = std::fs::canonicalize(trace_path).map_err(|e| {
        format!(
            "failed to resolve expert-HQQ trace path {}: {e}",
            trace_path.display()
        )
    })?;
    let spec_started = Instant::now();
    let spec = load_expert_hqq_diagnostic_cache_spec(spec_path)?;
    if profile_enabled {
        eprintln!(
            "expert_hqq_trace_compare_profile phase=load_spec duration_ms={}",
            spec_started.elapsed().as_millis()
        );
    }
    let cache_started = Instant::now();
    let cache = ExpertHqqCache::read_from_path(&spec.cache_path)?;
    spec.validate_cache_descriptors(&cache)?;
    cache.validate_required_tensors(&spec.required_tensors)?;
    let cache_payload_bytes = cache
        .tensors
        .iter()
        .map(|record| {
            record.descriptor.packed_bytes
                + record.descriptor.scales_bytes
                + record.descriptor.zeros_bytes
        })
        .sum::<usize>();
    if profile_enabled {
        eprintln!(
            "expert_hqq_trace_compare_profile phase=load_cache duration_ms={} tensors={} cache_bytes={}",
            cache_started.elapsed().as_millis(),
            cache.tensors.len(),
            cache_payload_bytes
        );
    }

    let parse_started = Instant::now();
    let trace_file = File::open(&trace_path).map_err(|e| {
        format!(
            "failed to open expert-HQQ trace {}: {e}",
            trace_path.display()
        )
    })?;
    let trace_reader = BufReader::with_capacity(8 * 1024 * 1024, trace_file);
    let parsed: serde_json::Value = serde_json::from_reader(trace_reader).map_err(|e| {
        format!(
            "malformed expert-HQQ runtime trace {}: {e}",
            trace_path.display()
        )
    })?;
    if profile_enabled {
        eprintln!(
            "expert_hqq_trace_compare_profile phase=parse_trace duration_ms={}",
            parse_started.elapsed().as_millis()
        );
    }
    let results = parsed
        .get("results")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "expert-HQQ runtime trace missing results array".to_string())?;
    if results.is_empty() {
        return Err("expert-HQQ runtime trace results array is empty".to_string());
    }

    let mut case_reports = Vec::with_capacity(results.len());
    let mut matched_filter_keys = BTreeSet::new();
    let compare_started = Instant::now();
    for (case_index, result) in results.iter().enumerate() {
        let case_started = Instant::now();
        let snapshots = result
            .get("response")
            .and_then(|v| v.get("debug_reference_trace"))
            .and_then(|v| v.get("prefill_stage_trace"))
            .and_then(|v| v.get("prefill_stage_snapshots"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                format!("expert-HQQ runtime trace case {case_index} missing prefill snapshots")
            })?;
        let input_stages = extract_expert_hqq_trace_full_buffer_stages(
            snapshots,
            "input_full",
            "input",
            "row_major_selected_rows_by_routed_hidden",
        )?;
        let w13_stages = extract_expert_hqq_trace_full_buffer_stages(
            snapshots,
            "w13_full",
            "w13",
            "row_major_selected_rows_by_w13_rows",
        )?;
        let activation_stages = extract_expert_hqq_trace_full_buffer_stages(
            snapshots,
            "activation_full",
            "activation",
            "row_major_selected_rows_by_moe_intermediate",
        )?;
        let output_stages = extract_expert_hqq_trace_full_buffer_stages(
            snapshots,
            "output_full",
            "output",
            "row_major_selected_rows_by_routed_hidden",
        )?;
        let mut w13_by_key = map_expert_hqq_trace_full_buffer_stages("w13", w13_stages)?;
        let mut activation_by_key =
            map_expert_hqq_trace_full_buffer_stages("activation", activation_stages)?;
        let mut output_by_key = map_expert_hqq_trace_full_buffer_stages("output", output_stages)?;
        let mut seen_inputs = BTreeSet::new();
        let mut case_jobs = Vec::new();
        for input in input_stages {
            let key = input.block_key();
            let filter_key = filter_key_for_block(case_index, &key);
            if let Some(filter) = block_filter {
                if !filter.contains(&filter_key) {
                    continue;
                }
                matched_filter_keys.insert(filter_key);
            }
            if !seen_inputs.insert(key.clone()) {
                return Err(format!(
                    "expert-HQQ runtime trace case {case_index} has duplicate input full-buffer stage for {}",
                    key.describe()
                ));
            }
            let w13 = w13_by_key.remove(&key).ok_or_else(|| {
                format!(
                    "expert-HQQ runtime trace case {case_index} missing full-buffer stage w13 for {}",
                    key.describe()
                )
            })?;
            let activation = activation_by_key.remove(&key).ok_or_else(|| {
                format!(
                    "expert-HQQ runtime trace case {case_index} missing full-buffer stage activation for {}",
                    key.describe()
                )
            })?;
            let output = output_by_key.remove(&key).ok_or_else(|| {
                format!(
                    "expert-HQQ runtime trace case {case_index} missing full-buffer stage output for {}",
                    key.describe()
                )
            })?;
            case_jobs.push((input, w13, activation, output));
        }
        for (label, leftovers) in [
            ("w13", w13_by_key),
            ("activation", activation_by_key),
            ("output", output_by_key),
        ] {
            if let Some((key, _)) = leftovers.into_iter().next() {
                if let Some(filter) = block_filter {
                    let leftover_filter_key = filter_key_for_block(case_index, &key);
                    if !filter.contains(&leftover_filter_key) {
                        continue;
                    }
                }
                return Err(format!(
                    "expert-HQQ runtime trace case {case_index} has orphan {label} full-buffer stage for {}",
                    key.describe()
                ));
            }
        }
        let mut compared = case_jobs
            .into_par_iter()
            .map(|(input, w13, activation, output)| {
                compare_expert_hqq_trace_case(&cache, case_index, input, w13, activation, output)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let compared_count = compared.len();
        case_reports.append(&mut compared);
        if profile_enabled {
            eprintln!(
                "expert_hqq_trace_compare_profile phase=compare_case case={} blocks={} duration_ms={}",
                case_index,
                compared_count,
                case_started.elapsed().as_millis()
            );
        }
    }
    if profile_enabled {
        eprintln!(
            "expert_hqq_trace_compare_profile phase=compare_all_cases duration_ms={} blocks={}",
            compare_started.elapsed().as_millis(),
            case_reports.len()
        );
    }
    if let Some(filter) = block_filter {
        for missing in filter.difference(&matched_filter_keys) {
            return Err(format!(
                "expert-HQQ runtime trace failure filter block not found: {}",
                missing.describe()
            ));
        }
        if case_reports.is_empty() {
            return Err("expert-HQQ runtime trace failure filter selected no blocks".to_string());
        }
    }

    let passes_contract = case_reports.iter().all(|case| case.passes_contract);
    let report = ExpertHqqTraceComparisonReport {
        trace_path,
        spec_path: spec.spec_path,
        cache_path: spec.cache_path,
        case_reports,
        passes_contract,
    };
    if let Some(path) = output_tsv_path {
        let write_started = Instant::now();
        write_expert_hqq_trace_comparison_tsv(path, &report)?;
        if profile_enabled {
            eprintln!(
                "expert_hqq_trace_compare_profile phase=write_metrics duration_ms={} rows={}",
                write_started.elapsed().as_millis(),
                report.block_count() * 4 + report.layer_count() * 4 + 4
            );
        }
    }
    if let Some(path) = mismatch_detail_tsv_path {
        let write_started = Instant::now();
        write_expert_hqq_trace_mismatch_details_tsv(path, &report)?;
        if profile_enabled {
            eprintln!(
                "expert_hqq_trace_compare_profile phase=write_mismatch_details duration_ms={} rows={}",
                write_started.elapsed().as_millis(),
                report
                    .case_reports
                    .iter()
                    .map(|case| case.mismatch_details.len())
                    .sum::<usize>()
            );
        }
    }
    if profile_enabled {
        eprintln!(
            "expert_hqq_trace_compare_profile phase=total duration_ms={} blocks={} passes_contract={}",
            total_started.elapsed().as_millis(),
            report.block_count(),
            report.passes_contract
        );
    }
    Ok(report)
}

fn extract_expert_hqq_trace_full_buffer_stages(
    snapshots: &[serde_json::Value],
    suffix: &str,
    label: &str,
    expected_layout: &str,
) -> Result<Vec<ExpertHqqTraceFullBufferStage>, String> {
    let matches = snapshots
        .iter()
        .filter(|snap| {
            snap.get("stage")
                .and_then(|v| v.as_str())
                .map(|stage| stage.ends_with(suffix) && stage.contains("_sequential_moe_"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Err(format!(
            "expert-HQQ trace missing full-buffer stage suffix {suffix}"
        ));
    }
    let mut stages = Vec::with_capacity(matches.len());
    for snap in matches {
        stages.push(parse_expert_hqq_trace_full_buffer_stage(
            snap,
            label,
            expected_layout,
        )?);
    }
    Ok(stages)
}

fn map_expert_hqq_trace_full_buffer_stages(
    label: &str,
    stages: Vec<ExpertHqqTraceFullBufferStage>,
) -> Result<BTreeMap<ExpertHqqTraceBlockKey, ExpertHqqTraceFullBufferStage>, String> {
    let mut by_key = BTreeMap::new();
    for stage in stages {
        let key = stage.block_key();
        if by_key.insert(key.clone(), stage).is_some() {
            return Err(format!(
                "expert-HQQ runtime trace has duplicate {label} full-buffer stage for {}",
                key.describe()
            ));
        }
    }
    Ok(by_key)
}

fn parse_expert_hqq_trace_full_buffer_stage(
    snap: &serde_json::Value,
    label: &str,
    expected_layout: &str,
) -> Result<ExpertHqqTraceFullBufferStage, String> {
    let stage_name = snap
        .get("stage")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("expert-HQQ {label} stage missing stage name"))?
        .to_string();
    let metadata = snap
        .get("metadata")
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing metadata"))?;
    let layer_idx = read_json_usize(metadata, "layer_idx", label)?;
    if snap
        .get("layer")
        .and_then(|v| v.as_u64())
        .map(|layer| layer as usize != layer_idx)
        .unwrap_or(false)
    {
        return Err(format!(
            "expert-HQQ {label} stage {stage_name} layer metadata mismatch"
        ));
    }
    let expert_idx = read_json_usize(metadata, "expert", label)?;
    let absolute_row_offset = read_json_usize(metadata, "absolute_row_offset", label)?;
    let row_count = read_json_usize(metadata, "row_count", label)?;
    let row_width = read_json_usize(metadata, "row_width", label)?;
    let value_count = read_json_usize(metadata, "value_count", label)?;
    let nbits = read_json_u8(metadata, "nbits", label)?;
    let group_size = read_json_usize(metadata, "group_size", label)?;
    let hqq_layout = metadata
        .get("hqq_layout")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing hqq_layout"))?
        .to_string();
    let experts_gated = metadata
        .get("experts_gated")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing experts_gated"))?;
    let dtype = metadata
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing dtype"))?;
    if dtype != "bf16" {
        return Err(format!(
            "expert-HQQ {label} stage {stage_name} dtype mismatch: got {dtype:?} expected \"bf16\""
        ));
    }
    let layout = metadata
        .get("layout")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing layout"))?;
    if layout != expected_layout {
        return Err(format!(
            "expert-HQQ {label} stage {stage_name} layout mismatch: got {layout:?} expected {expected_layout:?}"
        ));
    }
    let trace = snap
        .get("trace")
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing trace"))?;
    let trace_dtype = trace
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} trace missing dtype"))?;
    if trace_dtype != "bf16" {
        return Err(format!(
            "expert-HQQ {label} stage {stage_name} trace dtype mismatch: got {trace_dtype:?} expected \"bf16\""
        ));
    }
    let bits_json = trace
        .get("bf16_bits_u16")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("expert-HQQ {label} stage {stage_name} missing bf16_bits_u16"))?;
    let expected_len = row_count.checked_mul(row_width).ok_or_else(|| {
        format!("expert-HQQ {label} stage {stage_name} row_count*row_width overflow")
    })?;
    if bits_json.len() != expected_len || value_count != expected_len {
        return Err(format!(
            "expert-HQQ {label} stage {stage_name} shape mismatch: bits={} value_count={} expected row_count*row_width={}",
            bits_json.len(),
            value_count,
            expected_len
        ));
    }
    let mut bits = Vec::with_capacity(bits_json.len());
    for (idx, value) in bits_json.iter().enumerate() {
        let raw = value.as_u64().ok_or_else(|| {
            format!("expert-HQQ {label} stage {stage_name} bf16_bits_u16[{idx}] is not an integer")
        })?;
        let bits_value: u16 = raw.try_into().map_err(|_| {
            format!("expert-HQQ {label} stage {stage_name} bf16_bits_u16[{idx}]={raw} exceeds u16")
        })?;
        bits.push(bits_value);
    }
    Ok(ExpertHqqTraceFullBufferStage {
        stage_name,
        layer_idx,
        expert_idx,
        absolute_row_offset,
        row_count,
        row_width,
        nbits,
        group_size,
        hqq_layout,
        experts_gated,
        bits,
    })
}

fn compare_expert_hqq_trace_case(
    cache: &ExpertHqqCache,
    case_index: usize,
    input: ExpertHqqTraceFullBufferStage,
    w13: ExpertHqqTraceFullBufferStage,
    activation: ExpertHqqTraceFullBufferStage,
    output: ExpertHqqTraceFullBufferStage,
) -> Result<ExpertHqqTraceComparisonCaseReport, String> {
    for stage in [&w13, &activation, &output] {
        if stage.layer_idx != input.layer_idx
            || stage.expert_idx != input.expert_idx
            || stage.absolute_row_offset != input.absolute_row_offset
            || stage.row_count != input.row_count
            || stage.nbits != input.nbits
            || stage.group_size != input.group_size
            || stage.hqq_layout != input.hqq_layout
            || stage.experts_gated != input.experts_gated
        {
            return Err(format!(
                "expert-HQQ trace case {case_index} stage {} metadata does not match input stage {}",
                stage.stage_name, input.stage_name
            ));
        }
    }
    if input.hqq_layout != expert_hqq_layout_for_nbits(input.nbits)? {
        return Err(format!(
            "expert-HQQ trace case {case_index} HQQ layout {:?} does not match nbits {}",
            input.hqq_layout, input.nbits
        ));
    }
    if input.row_width != cache.header.routed_hidden_size {
        return Err(format!(
            "expert-HQQ trace case {case_index} input row_width {} != routed_hidden_size {}",
            input.row_width, cache.header.routed_hidden_size
        ));
    }
    let expected_w13_rows = if input.experts_gated {
        checked_mul(
            2,
            cache.header.moe_intermediate_size,
            "trace comparator gated W13 rows",
        )?
    } else {
        cache.header.moe_intermediate_size
    };
    if w13.row_width != expected_w13_rows {
        return Err(format!(
            "expert-HQQ trace case {case_index} W13 row_width {} != expected {}",
            w13.row_width, expected_w13_rows
        ));
    }
    if activation.row_width != cache.header.moe_intermediate_size {
        return Err(format!(
            "expert-HQQ trace case {case_index} activation row_width {} != moe_intermediate_size {}",
            activation.row_width, cache.header.moe_intermediate_size
        ));
    }
    if output.row_width != cache.header.routed_hidden_size {
        return Err(format!(
            "expert-HQQ trace case {case_index} output row_width {} != routed_hidden_size {}",
            output.row_width, cache.header.routed_hidden_size
        ));
    }

    let plan = cache.prefill_dispatch_plan(
        input.layer_idx,
        input.experts_gated,
        &[ExpertHqqPrefillWork::new(
            input.expert_idx,
            0,
            input.row_count,
        )],
    )?;
    let input_values = input
        .bits
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .collect::<Vec<_>>();
    let oracle = cache.execute_prefill_bf16_path_oracle(&plan, &input_values, input.row_count)?;

    let (input_compare, input_mismatches) = compare_bf16_trace_stage_to_f32(
        "input",
        &input.bits,
        &oracle.input_bf16,
        input.row_width,
        0.0,
        0.0,
    )?;
    let (w13_compare, w13_mismatches) = compare_bf16_trace_stage_to_f32(
        "w13",
        &w13.bits,
        &oracle.w13_preactivation,
        w13.row_width,
        1.0e-30,
        1.0e-30,
    )?;
    let (mut activation_compare, activation_mismatches) = compare_bf16_trace_stage_to_f32(
        "activation",
        &activation.bits,
        &oracle.activation,
        activation.row_width,
        0.0,
        0.0,
    )?;
    activation_compare.passes =
        activation_trace_contract_passes(&activation_compare, &activation_mismatches);
    let (output_compare, output_mismatches) = compare_bf16_trace_stage_to_f32(
        "output",
        &output.bits,
        &oracle.values,
        output.row_width,
        0.0,
        0.0,
    )?;
    let passes_contract = input_compare.passes
        && w13_compare.passes
        && activation_compare.passes
        && output_compare.passes;
    let mut mismatch_details = Vec::new();
    append_trace_mismatch_details(
        &mut mismatch_details,
        case_index,
        input.layer_idx,
        input.expert_idx,
        input.absolute_row_offset,
        input.row_count,
        "input",
        input.row_width,
        input_mismatches,
    );
    append_trace_mismatch_details(
        &mut mismatch_details,
        case_index,
        input.layer_idx,
        input.expert_idx,
        input.absolute_row_offset,
        input.row_count,
        "w13",
        w13.row_width,
        w13_mismatches,
    );
    append_trace_mismatch_details(
        &mut mismatch_details,
        case_index,
        input.layer_idx,
        input.expert_idx,
        input.absolute_row_offset,
        input.row_count,
        "activation",
        activation.row_width,
        activation_mismatches,
    );
    append_trace_mismatch_details(
        &mut mismatch_details,
        case_index,
        input.layer_idx,
        input.expert_idx,
        input.absolute_row_offset,
        input.row_count,
        "output",
        output.row_width,
        output_mismatches,
    );

    Ok(ExpertHqqTraceComparisonCaseReport {
        case_index,
        layer_idx: input.layer_idx,
        expert_idx: input.expert_idx,
        absolute_row_offset: input.absolute_row_offset,
        row_count: input.row_count,
        input_row_width: input.row_width,
        w13_row_width: w13.row_width,
        activation_row_width: activation.row_width,
        output_row_width: output.row_width,
        nbits: input.nbits,
        group_size: input.group_size,
        hqq_layout: input.hqq_layout,
        experts_gated: input.experts_gated,
        input: input_compare,
        w13: w13_compare,
        activation: activation_compare,
        output: output_compare,
        mismatch_details,
        passes_contract,
    })
}

fn compare_bf16_trace_stage_to_f32(
    label: &str,
    actual_bits: &[u16],
    expected_values: &[f32],
    row_width: usize,
    sum_tolerance: f64,
    max_tolerance: f64,
) -> Result<
    (
        ExpertHqqTraceStageComparison,
        Vec<ExpertHqqStageMismatchDetail>,
    ),
    String,
> {
    if actual_bits.len() != expected_values.len() {
        return Err(format!(
            "expert-HQQ trace comparator {label} length mismatch: actual={} expected={}",
            actual_bits.len(),
            expected_values.len()
        ));
    }
    if row_width == 0 {
        return Err(format!(
            "expert-HQQ trace comparator {label} row_width must be > 0"
        ));
    }
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut l2_sq = 0.0f64;
    let mut mismatch_count = 0usize;
    let mut mismatch_details = Vec::new();
    let collect_details = sum_tolerance == 0.0 && max_tolerance == 0.0;
    for (idx, (&actual, &expected)) in actual_bits.iter().zip(expected_values.iter()).enumerate() {
        let expected_bits = f32_to_bf16(expected);
        if actual != expected_bits {
            mismatch_count += 1;
            if collect_details {
                let actual_value = bf16_to_f32(actual);
                let expected_bf16_value = bf16_to_f32(expected_bits);
                let actual_class = bf16_class(actual);
                let expected_bf16_class = bf16_class(expected_bits);
                let expected_raw_class = f32_class(expected);
                let flush_or_rounding = is_zero_subnormal_pair(actual, expected_bits)
                    || (is_bf16_zero_or_subnormal(actual)
                        && is_bf16_zero_or_subnormal(expected_bits)
                        && matches!(expected.classify(), std::num::FpCategory::Subnormal));
                mismatch_details.push(ExpertHqqStageMismatchDetail {
                    linear_index: idx,
                    local_row: idx / row_width,
                    column: idx % row_width,
                    actual_bits: actual,
                    expected_bits,
                    actual_value,
                    expected_bf16_value,
                    expected_raw_value: expected,
                    delta_abs: (actual_value - expected_bf16_value).abs() as f64,
                    actual_class,
                    expected_bf16_class,
                    expected_raw_class,
                    flush_to_zero_or_subnormal_rounding: flush_or_rounding,
                    diagnostic: mismatch_diagnostic(actual, expected_bits, expected),
                });
            }
        }
        let delta = (bf16_to_f32(actual) - bf16_to_f32(expected_bits)).abs() as f64;
        if !delta.is_finite() {
            return Err(format!(
                "expert-HQQ trace comparator {label} produced non-finite delta at index {idx}"
            ));
        }
        sum_abs += delta;
        max_abs = max_abs.max(delta);
        l2_sq += delta * delta;
    }
    Ok((
        ExpertHqqTraceStageComparison {
            count: actual_bits.len(),
            sum_abs,
            max_abs,
            l2: l2_sq.sqrt(),
            mismatch_count,
            sum_tolerance,
            max_tolerance,
            passes: sum_abs <= sum_tolerance && max_abs <= max_tolerance,
        },
        mismatch_details,
    ))
}

#[allow(clippy::too_many_arguments)]
fn append_trace_mismatch_details(
    dst: &mut Vec<ExpertHqqTraceMismatchDetail>,
    case_index: usize,
    layer_idx: usize,
    expert_idx: usize,
    absolute_row_offset: usize,
    row_count: usize,
    stage: &'static str,
    row_width: usize,
    details: Vec<ExpertHqqStageMismatchDetail>,
) {
    for detail in details {
        dst.push(ExpertHqqTraceMismatchDetail {
            case_index,
            layer_idx,
            expert_idx,
            absolute_row_offset,
            row_count,
            stage,
            linear_index: detail.linear_index,
            local_row: detail.local_row,
            absolute_row: absolute_row_offset + detail.local_row,
            column: detail.column,
            row_width,
            actual_bits: detail.actual_bits,
            expected_bits: detail.expected_bits,
            actual_value: detail.actual_value,
            expected_bf16_value: detail.expected_bf16_value,
            expected_raw_value: detail.expected_raw_value,
            delta_abs: detail.delta_abs,
            actual_class: detail.actual_class,
            expected_bf16_class: detail.expected_bf16_class,
            expected_raw_class: detail.expected_raw_class,
            flush_to_zero_or_subnormal_rounding: detail.flush_to_zero_or_subnormal_rounding,
            diagnostic: detail.diagnostic,
        });
    }
}

fn bf16_class(bits: u16) -> &'static str {
    let exp = (bits >> 7) & 0xff;
    let frac = bits & 0x7f;
    match (exp, frac) {
        (0, 0) => "zero",
        (0, _) => "subnormal",
        (0xff, 0) => "infinite",
        (0xff, _) => "nan",
        _ => "normal",
    }
}

fn f32_class(value: f32) -> &'static str {
    match value.classify() {
        std::num::FpCategory::Zero => "zero",
        std::num::FpCategory::Subnormal => "subnormal",
        std::num::FpCategory::Normal => "normal",
        std::num::FpCategory::Infinite => "infinite",
        std::num::FpCategory::Nan => "nan",
    }
}

fn is_bf16_zero_or_subnormal(bits: u16) -> bool {
    ((bits >> 7) & 0xff) == 0
}

fn is_zero_subnormal_pair(left: u16, right: u16) -> bool {
    matches!(
        (bf16_class(left), bf16_class(right)),
        ("zero", "subnormal") | ("subnormal", "zero")
    )
}

fn is_positive_bf16_subnormal(bits: u16) -> bool {
    bits > 0 && bits < 0x0080
}

fn is_activation_zero_vs_positive_subnormal_mismatch(
    detail: &ExpertHqqStageMismatchDetail,
) -> bool {
    detail.actual_bits == 0x0000
        && is_positive_bf16_subnormal(detail.expected_bits)
        && detail.expected_raw_value > 0.0
        && matches!(
            detail.expected_raw_value.classify(),
            std::num::FpCategory::Subnormal
        )
        && detail.diagnostic == "bf16_zero_vs_subnormal_flush_to_zero_candidate"
}

fn activation_trace_contract_passes(
    comparison: &ExpertHqqTraceStageComparison,
    mismatches: &[ExpertHqqStageMismatchDetail],
) -> bool {
    if comparison.passes {
        return true;
    }
    comparison.sum_tolerance == 0.0
        && comparison.max_tolerance == 0.0
        && comparison.mismatch_count > 0
        && comparison.mismatch_count == mismatches.len()
        && mismatches
            .iter()
            .all(is_activation_zero_vs_positive_subnormal_mismatch)
}

fn mismatch_diagnostic(actual_bits: u16, expected_bits: u16, expected_raw: f32) -> &'static str {
    if is_zero_subnormal_pair(actual_bits, expected_bits) {
        "bf16_zero_vs_subnormal_flush_to_zero_candidate"
    } else if is_bf16_zero_or_subnormal(actual_bits)
        && is_bf16_zero_or_subnormal(expected_bits)
        && matches!(expected_raw.classify(), std::num::FpCategory::Subnormal)
    {
        "bf16_subnormal_rounding_candidate"
    } else {
        "non_subnormal_mismatch"
    }
}

fn write_expert_hqq_trace_comparison_tsv(
    path: &Path,
    report: &ExpertHqqTraceComparisonReport,
) -> Result<(), String> {
    let mut lines = vec![
        "case_index\tlayer\texpert\tabsolute_row_offset\trow_count\tnbits\tgroup_size\texperts_gated\tstage\tcount\tsum_abs\tmax_abs\tl2\tmismatch_count\tsum_tolerance\tmax_tolerance\tpasses".to_string()
    ];
    for case in &report.case_reports {
        for (stage, metrics) in [
            ("input", case.input),
            ("w13", case.w13),
            ("activation", case.activation),
            ("output", case.output),
        ] {
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.18e}\t{:.18e}\t{:.18e}\t{}\t{:.18e}\t{:.18e}\t{}",
                case.case_index,
                case.layer_idx,
                case.expert_idx,
                case.absolute_row_offset,
                case.row_count,
                case.nbits,
                case.group_size,
                case.experts_gated,
                stage,
                metrics.count,
                metrics.sum_abs,
                metrics.max_abs,
                metrics.l2,
                metrics.mismatch_count,
                metrics.sum_tolerance,
                metrics.max_tolerance,
                metrics.passes,
            ));
        }
    }
    for (layer_idx, (input, w13, activation, output)) in report.layer_stage_totals() {
        for (stage, metrics) in [
            ("LAYER_input", input),
            ("LAYER_w13", w13),
            ("LAYER_activation", activation),
            ("LAYER_output", output),
        ] {
            lines.push(format!(
                "LAYER\t{}\tALL\tALL\tALL\tALL\tALL\tALL\t{}\t{}\t{:.18e}\t{:.18e}\t{:.18e}\t{}\t{:.18e}\t{:.18e}\t{}",
                layer_idx,
                stage,
                metrics.count,
                metrics.sum_abs,
                metrics.max_abs,
                metrics.l2,
                metrics.mismatch_count,
                metrics.sum_tolerance,
                metrics.max_tolerance,
                metrics.passes,
            ));
        }
    }
    let (input, w13, activation, output) = report.stage_totals();
    for (stage, metrics) in [
        ("TOTAL_input", input),
        ("TOTAL_w13", w13),
        ("TOTAL_activation", activation),
        ("TOTAL_output", output),
    ] {
        lines.push(format!(
            "ALL\tALL\tALL\tALL\tALL\tALL\tALL\tALL\t{}\t{}\t{:.18e}\t{:.18e}\t{:.18e}\t{}\t{:.18e}\t{:.18e}\t{}",
            stage,
            metrics.count,
            metrics.sum_abs,
            metrics.max_abs,
            metrics.l2,
            metrics.mismatch_count,
            metrics.sum_tolerance,
            metrics.max_tolerance,
            metrics.passes,
        ));
    }
    std::fs::write(path, format!("{}\n", lines.join("\n"))).map_err(|e| {
        format!(
            "failed to write expert-HQQ trace comparison TSV {}: {e}",
            path.display()
        )
    })
}

fn write_expert_hqq_trace_mismatch_details_tsv(
    path: &Path,
    report: &ExpertHqqTraceComparisonReport,
) -> Result<(), String> {
    let mut lines = vec![
        "case_index\tlayer\texpert\tabsolute_row_offset\trow_count\tstage\tlinear_index\tlocal_row\tabsolute_row\tcolumn\trow_width\tactual_bits_hex\texpected_bits_hex\tactual_bf16_value\texpected_bf16_value\texpected_raw_f32_value\tdelta_abs\tactual_class\texpected_bf16_class\texpected_raw_f32_class\tflush_to_zero_or_subnormal_rounding\tdiagnostic".to_string()
    ];
    for case in &report.case_reports {
        for detail in &case.mismatch_details {
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0x{:04x}\t0x{:04x}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{}\t{}\t{}\t{}\t{}",
                detail.case_index,
                detail.layer_idx,
                detail.expert_idx,
                detail.absolute_row_offset,
                detail.row_count,
                detail.stage,
                detail.linear_index,
                detail.local_row,
                detail.absolute_row,
                detail.column,
                detail.row_width,
                detail.actual_bits,
                detail.expected_bits,
                detail.actual_value,
                detail.expected_bf16_value,
                detail.expected_raw_value,
                detail.delta_abs,
                detail.actual_class,
                detail.expected_bf16_class,
                detail.expected_raw_class,
                detail.flush_to_zero_or_subnormal_rounding,
                detail.diagnostic,
            ));
        }
    }
    std::fs::write(path, format!("{}\n", lines.join("\n"))).map_err(|e| {
        format!(
            "failed to write expert-HQQ trace mismatch details TSV {}: {e}",
            path.display()
        )
    })
}

fn read_json_usize(value: &serde_json::Value, key: &str, label: &str) -> Result<usize, String> {
    let raw = value
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("expert-HQQ {label} metadata missing integer {key}"))?;
    raw.try_into()
        .map_err(|_| format!("expert-HQQ {label} metadata {key}={raw} exceeds usize"))
}

fn read_json_u8(value: &serde_json::Value, key: &str, label: &str) -> Result<u8, String> {
    let raw = value
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("expert-HQQ {label} metadata missing integer {key}"))?;
    raw.try_into()
        .map_err(|_| format!("expert-HQQ {label} metadata {key}={raw} exceeds u8"))
}

pub fn prefill_dispatch_plan_from_registered_cache(
    cache: Option<&ExpertHqqCache>,
    layer_idx: usize,
    experts_gated: bool,
    works: &[ExpertHqqPrefillWork],
) -> Result<ExpertHqqPrefillDispatchPlan, String> {
    let cache = cache
        .ok_or_else(|| "expert-HQQ cache is not registered for prefill dispatch".to_string())?;
    cache.prefill_dispatch_plan(layer_idx, experts_gated, works)
}

pub fn validate_expert_hqq_runtime_diagnostic_availability(
    cache: Option<&ExpertHqqCache>,
    model: ExpertHqqRuntimeDiagnosticModelShape,
    requirements: &[ExpertHqqRuntimeDiagnosticRequirement],
) -> Result<ExpertHqqRuntimeDiagnosticReport, String> {
    let cache =
        cache.ok_or_else(|| "expert-HQQ runtime diagnostic cache is not registered".to_string())?;
    if requirements.is_empty() {
        return Err(
            "expert-HQQ runtime diagnostic requires explicit layer/expert requirements".to_string(),
        );
    }
    if model.hidden_size == 0
        || model.routed_hidden_size == 0
        || model.moe_intermediate_size == 0
        || model.n_routed_experts == 0
        || model.num_hidden_layers == 0
    {
        return Err(format!(
            "expert-HQQ runtime diagnostic model shape is incomplete: hidden={} routed_hidden={} intermediate={} experts={} layers={}",
            model.hidden_size,
            model.routed_hidden_size,
            model.moe_intermediate_size,
            model.n_routed_experts,
            model.num_hidden_layers,
        ));
    }

    cache.validate()?;
    if cache.header.hidden_size != model.hidden_size
        || cache.header.routed_hidden_size != model.routed_hidden_size
        || cache.header.moe_intermediate_size != model.moe_intermediate_size
        || cache.header.n_routed_experts != model.n_routed_experts
        || cache.header.num_moe_layers != model.num_hidden_layers
    {
        return Err(format!(
            "expert-HQQ runtime diagnostic model mismatch: cache hidden/routed/inter/experts/layers={}/{}/{}/{}/{} runtime={}/{}/{}/{}/{}",
            cache.header.hidden_size,
            cache.header.routed_hidden_size,
            cache.header.moe_intermediate_size,
            cache.header.n_routed_experts,
            cache.header.num_moe_layers,
            model.hidden_size,
            model.routed_hidden_size,
            model.moe_intermediate_size,
            model.n_routed_experts,
            model.num_hidden_layers,
        ));
    }

    let expected_w13_rows = if model.experts_gated {
        checked_mul(
            2,
            model.moe_intermediate_size,
            "runtime diagnostic gated W13 rows",
        )?
    } else {
        model.moe_intermediate_size
    };
    let mut seen = HashSet::new();
    let mut tensors = Vec::with_capacity(requirements.len().saturating_mul(2));
    let mut total_payload_bytes = 0usize;
    for req in requirements {
        if req.layer_idx >= model.num_hidden_layers {
            return Err(format!(
                "expert-HQQ runtime diagnostic layer_idx {} out of range {}",
                req.layer_idx, model.num_hidden_layers
            ));
        }
        if req.expert_idx >= model.n_routed_experts {
            return Err(format!(
                "expert-HQQ runtime diagnostic expert_idx {} out of range {}",
                req.expert_idx, model.n_routed_experts
            ));
        }
        if !seen.insert((req.layer_idx, req.expert_idx)) {
            return Err(format!(
                "duplicate expert-HQQ runtime diagnostic requirement for layer={} expert={}",
                req.layer_idx, req.expert_idx
            ));
        }
        let w13_key =
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, req.layer_idx, req.expert_idx);
        let w2_key =
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, req.layer_idx, req.expert_idx);
        let w13 = cache.require_tensor_record(w13_key)?;
        let w2 = cache.require_tensor_record(w2_key)?;
        validate_runtime_diagnostic_tensor(
            w13,
            req,
            ExpertHqqTensorRole::W13,
            expected_w13_rows,
            model.routed_hidden_size,
        )?;
        validate_runtime_diagnostic_tensor(
            w2,
            req,
            ExpertHqqTensorRole::W2,
            model.routed_hidden_size,
            model.moe_intermediate_size,
        )?;
        for record in [w13, w2] {
            let packed = record.packed.len();
            let scales = record.scales.len();
            let zeros = record.zeros.len();
            total_payload_bytes = total_payload_bytes
                .checked_add(packed)
                .and_then(|v| v.checked_add(scales))
                .and_then(|v| v.checked_add(zeros))
                .ok_or_else(|| "expert-HQQ runtime diagnostic payload byte overflow".to_string())?;
            tensors.push(ExpertHqqRuntimeDiagnosticTensorSummary {
                role: record.descriptor.role,
                layer_idx: record.descriptor.layer_idx,
                expert_idx: record.descriptor.expert_idx,
                nbits: record.descriptor.nbits,
                group_size: record.descriptor.group_size,
                axis: record.descriptor.axis,
                layout: record.descriptor.layout.clone(),
                rows: record.descriptor.rows,
                cols: record.descriptor.cols,
                packed_bytes: packed,
                scales_bytes: scales,
                zeros_bytes: zeros,
            });
        }
    }
    Ok(ExpertHqqRuntimeDiagnosticReport {
        checked_experts: requirements.len(),
        tensor_records: tensors.len(),
        total_payload_bytes,
        tensors,
    })
}

pub fn validate_expert_hqq_runtime_prefill_diagnostic_contract(
    cache: Option<&ExpertHqqCache>,
    model: ExpertHqqRuntimeDiagnosticModelShape,
    layer_idx: usize,
    nbits: u8,
    group_size: usize,
    blocks: &[ExpertHqqRuntimePrefillBlock],
    shape: ExpertHqqRuntimePrefillBufferShape,
    buffer_lengths: ExpertHqqRuntimePrefillBufferLengths,
) -> Result<ExpertHqqRuntimePrefillDiagnosticReport, String> {
    let cache = cache.ok_or_else(|| {
        "expert-HQQ runtime prefill diagnostic cache is not registered".to_string()
    })?;
    if blocks.is_empty() {
        return Err(
            "expert-HQQ runtime prefill diagnostic requires at least one selected block"
                .to_string(),
        );
    }
    if shape.total_sorted_rows == 0 {
        return Err(
            "expert-HQQ runtime prefill diagnostic total_sorted_rows must be nonzero".to_string(),
        );
    }
    if layer_idx >= model.num_hidden_layers {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic layer_idx {} out of range {}",
            layer_idx, model.num_hidden_layers
        ));
    }
    if shape.input_row_stride < model.routed_hidden_size {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic input_row_stride {} < routed_hidden_size {}",
            shape.input_row_stride, model.routed_hidden_size
        ));
    }
    let w13_rows = expected_runtime_w13_rows(model.experts_gated, model.moe_intermediate_size)?;
    if shape.w13_row_stride < w13_rows {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic w13_row_stride {} < w13_rows {}",
            shape.w13_row_stride, w13_rows
        ));
    }
    if shape.activation_row_stride < model.moe_intermediate_size {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic activation_row_stride {} < moe_intermediate_size {}",
            shape.activation_row_stride, model.moe_intermediate_size
        ));
    }
    if shape.output_row_stride < model.routed_hidden_size {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic output_row_stride {} < routed_hidden_size {}",
            shape.output_row_stride, model.routed_hidden_size
        ));
    }
    let required_lengths = ExpertHqqRuntimePrefillBufferLengths::required(model, shape)?;
    if buffer_lengths != required_lengths {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic buffer length mismatch: got input/w13/activation/output={}/{}/{}/{} required={}/{}/{}/{}",
            buffer_lengths.input_values,
            buffer_lengths.w13_values,
            buffer_lengths.activation_values,
            buffer_lengths.output_values,
            required_lengths.input_values,
            required_lengths.w13_values,
            required_lengths.activation_values,
            required_lengths.output_values,
        ));
    }

    let mut requirements = Vec::with_capacity(blocks.len());
    let mut works = Vec::with_capacity(blocks.len());
    let mut claimed = vec![false; shape.total_sorted_rows];
    let mut claimed_rows = 0usize;
    let mut seen_experts = HashSet::new();
    for (idx, block) in blocks.iter().enumerate() {
        if block.row_count == 0 {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic block {idx} expert {} has zero rows",
                block.expert_idx
            ));
        }
        if block.expert_idx >= model.n_routed_experts {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic expert_idx {} out of range {}",
                block.expert_idx, model.n_routed_experts
            ));
        }
        if !seen_experts.insert(block.expert_idx) {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic duplicate selected expert {}",
                block.expert_idx
            ));
        }
        if idx > 0 {
            let prev = blocks[idx - 1];
            let prev_end = prev
                .absolute_row_offset
                .checked_add(prev.row_count)
                .ok_or_else(|| {
                    "expert-HQQ runtime prefill diagnostic previous row range overflow".to_string()
                })?;
            if block.absolute_row_offset < prev_end {
                return Err(format!(
                    "expert-HQQ runtime prefill diagnostic blocks must be sorted and non-overlapping: block {idx} starts at {} before previous end {}",
                    block.absolute_row_offset, prev_end
                ));
            }
        }
        let row_end = block
            .absolute_row_offset
            .checked_add(block.row_count)
            .ok_or_else(|| {
                format!(
                    "expert-HQQ runtime prefill diagnostic row range overflow for expert {}",
                    block.expert_idx
                )
            })?;
        if row_end > shape.total_sorted_rows {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic row range {}..{} exceeds total_sorted_rows {} for expert {}",
                block.absolute_row_offset, row_end, shape.total_sorted_rows, block.expert_idx
            ));
        }
        for row in block.absolute_row_offset..row_end {
            if claimed[row] {
                return Err(format!(
                    "expert-HQQ runtime prefill diagnostic row {row} is claimed by more than one block"
                ));
            }
            claimed[row] = true;
            claimed_rows = claimed_rows.checked_add(1).ok_or_else(|| {
                "expert-HQQ runtime prefill diagnostic claimed row overflow".to_string()
            })?;
        }
        requirements.push(ExpertHqqRuntimeDiagnosticRequirement::new(
            layer_idx,
            block.expert_idx,
            nbits,
            group_size,
        ));
        works.push(ExpertHqqPrefillWork::new(
            block.expert_idx,
            block.absolute_row_offset,
            block.row_count,
        ));
    }

    let plan = cache.prefill_dispatch_plan(layer_idx, model.experts_gated, &works)?;
    if plan.entries.len() != blocks.len() {
        return Err(format!(
            "expert-HQQ runtime prefill diagnostic plan entry count {} != block count {}",
            plan.entries.len(),
            blocks.len()
        ));
    }
    for (idx, (entry, block)) in plan.entries.iter().zip(blocks.iter()).enumerate() {
        if entry.expert_idx != block.expert_idx
            || entry.row_offset != block.absolute_row_offset
            || entry.row_count != block.row_count
        {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic plan/block mismatch at block {idx}: block expert/offset/count={}/{}/{} plan={}/{}/{}",
                block.expert_idx,
                block.absolute_row_offset,
                block.row_count,
                entry.expert_idx,
                entry.row_offset,
                entry.row_count
            ));
        }
        if entry.w13_nbits != nbits || entry.w2_nbits != nbits {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic nbits mismatch for expert {}: W13/W2={}/{} expected {}",
                entry.expert_idx, entry.w13_nbits, entry.w2_nbits, nbits
            ));
        }
        if entry.w13_group_size != group_size || entry.w2_group_size != group_size {
            return Err(format!(
                "expert-HQQ runtime prefill diagnostic group_size mismatch for expert {}: W13/W2={}/{} expected {}",
                entry.expert_idx, entry.w13_group_size, entry.w2_group_size, group_size
            ));
        }
    }

    let availability =
        validate_expert_hqq_runtime_diagnostic_availability(Some(cache), model, &requirements)?;
    let padding_rows = shape
        .total_sorted_rows
        .checked_sub(claimed_rows)
        .ok_or_else(|| "expert-HQQ runtime prefill diagnostic padding row underflow".to_string())?;
    let oracle = bf16_path_oracle_metadata(
        model,
        claimed_rows,
        plan.input_layout,
        plan.w13_output_layout,
        plan.activation_output_layout,
        plan.w2_output_layout,
    )?;

    Ok(ExpertHqqRuntimePrefillDiagnosticReport {
        layer_idx,
        experts_gated: model.experts_gated,
        nbits,
        group_size,
        total_sorted_rows: shape.total_sorted_rows,
        claimed_rows,
        padding_rows,
        plan_entries: plan.entries.len(),
        input_row_stride: shape.input_row_stride,
        w13_row_stride: shape.w13_row_stride,
        activation_row_stride: shape.activation_row_stride,
        output_row_stride: shape.output_row_stride,
        buffer_lengths: required_lengths,
        oracle,
        availability,
    })
}

fn validate_runtime_diagnostic_tensor(
    record: &ExpertHqqTensorRecord,
    req: &ExpertHqqRuntimeDiagnosticRequirement,
    role: ExpertHqqTensorRole,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    record.validate_payload_lengths()?;
    let desc = &record.descriptor;
    if desc.role != role {
        return Err(format!(
            "expert-HQQ runtime diagnostic role mismatch for layer={} expert={}: expected {} got {}",
            req.layer_idx,
            req.expert_idx,
            role.as_str(),
            desc.role.as_str(),
        ));
    }
    if desc.layer_idx != req.layer_idx || desc.expert_idx != req.expert_idx {
        return Err(format!(
            "expert-HQQ runtime diagnostic descriptor pairing mismatch: requested layer={} expert={} got layer={} expert={} role={}",
            req.layer_idx,
            req.expert_idx,
            desc.layer_idx,
            desc.expert_idx,
            role.as_str(),
        ));
    }
    if desc.nbits != req.nbits {
        return Err(format!(
            "expert-HQQ runtime diagnostic nbits mismatch for layer={} expert={} role={}: got {} expected {}",
            req.layer_idx,
            req.expert_idx,
            role.as_str(),
            desc.nbits,
            req.nbits,
        ));
    }
    if desc.group_size != req.group_size {
        return Err(format!(
            "expert-HQQ runtime diagnostic group_size mismatch for layer={} expert={} role={}: got {} expected {}",
            req.layer_idx,
            req.expert_idx,
            role.as_str(),
            desc.group_size,
            req.group_size,
        ));
    }
    if desc.axis != EXPERT_HQQ_AXIS {
        return Err(format!(
            "expert-HQQ runtime diagnostic axis mismatch for layer={} expert={} role={}: got {} expected {}",
            req.layer_idx,
            req.expert_idx,
            role.as_str(),
            desc.axis,
            EXPERT_HQQ_AXIS,
        ));
    }
    let expected_layout = expert_hqq_layout_for_nbits(req.nbits)?;
    if desc.layout != expected_layout {
        return Err(format!(
            "expert-HQQ runtime diagnostic layout mismatch for layer={} expert={} role={}: got '{}' expected '{}'",
            req.layer_idx,
            req.expert_idx,
            role.as_str(),
            desc.layout,
            expected_layout,
        ));
    }
    if desc.rows != rows || desc.cols != cols {
        return Err(format!(
            "expert-HQQ runtime diagnostic shape mismatch for layer={} expert={} role={}: got {}x{} expected {}x{}",
            req.layer_idx,
            req.expert_idx,
            role.as_str(),
            desc.rows,
            desc.cols,
            rows,
            cols,
        ));
    }
    Ok(())
}

pub fn execute_prefill_reference_from_registered_cache(
    cache: Option<&ExpertHqqCache>,
    plan: &ExpertHqqPrefillDispatchPlan,
    sorted_routed_inputs: &[f32],
    sorted_row_count: usize,
) -> Result<ExpertHqqPrefillReferenceOutput, String> {
    let cache = cache.ok_or_else(|| {
        "expert-HQQ cache is not registered for prefill reference execution".to_string()
    })?;
    cache.execute_prefill_reference(plan, sorted_routed_inputs, sorted_row_count)
}

#[cfg(test)]
pub fn execute_prefill_test_dispatch_from_registered_cache(
    cache: Option<&ExpertHqqCache>,
    plan: &ExpertHqqPrefillDispatchPlan,
    sorted_routed_inputs: &[f32],
    sorted_row_count: usize,
) -> Result<ExpertHqqPrefillTestDispatchOutput, String> {
    let cache = cache.ok_or_else(|| {
        "expert-HQQ cache is not registered for prefill test dispatch".to_string()
    })?;
    cache.execute_prefill_test_dispatch(plan, sorted_routed_inputs, sorted_row_count)
}

#[cfg(all(test, has_prefill_kernels))]
pub fn execute_prefill_test_gpu_prototype_from_registered_cache(
    cache: Option<&ExpertHqqCache>,
    plan: &ExpertHqqPrefillDispatchPlan,
    sorted_routed_inputs: &[f32],
    sorted_row_count: usize,
) -> Result<ExpertHqqPrefillGpuPrototypeOutput, String> {
    let cache = cache.ok_or_else(|| {
        "expert-HQQ cache is not registered for prefill GPU prototype".to_string()
    })?;
    cache.execute_prefill_test_gpu_prototype(plan, sorted_routed_inputs, sorted_row_count)
}

#[cfg(all(test, has_prefill_kernels))]
const EXPERT_HQQ_GPU_PREFILL_KERNELS_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/prefill_kernels.ptx"));

#[cfg(all(test, has_prefill_kernels))]
#[derive(Clone, Copy)]
struct ExpertHqqGpuRawCuFunc(cuda_sys::CUfunction);

#[cfg(all(test, has_prefill_kernels))]
fn extract_expert_hqq_gpu_cu_func(func: &CudaFunction) -> ExpertHqqGpuRawCuFunc {
    unsafe {
        let struct_ptr = func as *const _ as *const u8;
        let word0: cuda_sys::CUfunction = std::ptr::read(struct_ptr as *const _);
        let mut dummy = 0i32;
        let w0_valid = cuda_sys::lib().cuFuncGetAttribute(
            &mut dummy,
            cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
            word0,
        ) == cuda_sys::CUresult::CUDA_SUCCESS;
        ExpertHqqGpuRawCuFunc(if w0_valid {
            word0
        } else {
            std::ptr::read(struct_ptr.add(8) as *const _)
        })
    }
}

#[cfg(all(test, has_prefill_kernels))]
unsafe fn launch_expert_hqq_gpu_test_kernel(
    func: ExpertHqqGpuRawCuFunc,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &mut [*mut std::ffi::c_void],
) -> Result<(), String> {
    let err = cuda_sys::lib().cuLaunchKernel(
        func.0,
        grid.0,
        grid.1,
        grid.2,
        block.0,
        block.1,
        block.2,
        0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if err == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "expert-HQQ GPU prototype kernel launch failed: {:?} (grid={:?}, block={:?})",
            err, grid, block
        ))
    }
}

#[cfg(all(test, has_prefill_kernels))]
struct ExpertHqqGpuTensorBuffers {
    packed: CudaSlice<u8>,
    scales: CudaSlice<f32>,
    zeros: CudaSlice<f32>,
}

#[cfg(all(test, has_prefill_kernels))]
struct ExpertHqqGpuPrototypeKernels {
    device: Arc<CudaDevice>,
    hqq4_prefill_gemm_bf16: ExpertHqqGpuRawCuFunc,
    hqq6_prefill_gemm_bf16: ExpertHqqGpuRawCuFunc,
    hqq8_prefill_gemm_bf16: ExpertHqqGpuRawCuFunc,
    relu2_batched: ExpertHqqGpuRawCuFunc,
    silu_mul_batched: ExpertHqqGpuRawCuFunc,
}

#[cfg(all(test, has_prefill_kernels))]
impl ExpertHqqGpuPrototypeKernels {
    fn new() -> Result<Self, String> {
        let device = CudaDevice::new(0)
            .map_err(|e| format!("failed to create CUDA device for expert-HQQ test: {e}"))?;
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(EXPERT_HQQ_GPU_PREFILL_KERNELS_PTX),
                "expert_hqq_prefill_gpu_prototype_kernels",
                &[
                    "hqq4_prefill_gemm_bf16_kernel",
                    "hqq6_prefill_gemm_bf16_kernel",
                    "hqq8_prefill_gemm_bf16_kernel",
                    "relu2_batched_kernel",
                    "silu_mul_batched_kernel",
                ],
            )
            .map_err(|e| format!("failed to load expert-HQQ prefill test kernels PTX: {e}"))?;

        let hqq4 = extract_expert_hqq_gpu_cu_func(
            &device
                .get_func(
                    "expert_hqq_prefill_gpu_prototype_kernels",
                    "hqq4_prefill_gemm_bf16_kernel",
                )
                .ok_or_else(|| "missing hqq4_prefill_gemm_bf16_kernel".to_string())?,
        );
        let hqq6 = extract_expert_hqq_gpu_cu_func(
            &device
                .get_func(
                    "expert_hqq_prefill_gpu_prototype_kernels",
                    "hqq6_prefill_gemm_bf16_kernel",
                )
                .ok_or_else(|| "missing hqq6_prefill_gemm_bf16_kernel".to_string())?,
        );
        let hqq8 = extract_expert_hqq_gpu_cu_func(
            &device
                .get_func(
                    "expert_hqq_prefill_gpu_prototype_kernels",
                    "hqq8_prefill_gemm_bf16_kernel",
                )
                .ok_or_else(|| "missing hqq8_prefill_gemm_bf16_kernel".to_string())?,
        );
        let relu2 = extract_expert_hqq_gpu_cu_func(
            &device
                .get_func(
                    "expert_hqq_prefill_gpu_prototype_kernels",
                    "relu2_batched_kernel",
                )
                .ok_or_else(|| "missing relu2_batched_kernel".to_string())?,
        );
        let silu_mul = extract_expert_hqq_gpu_cu_func(
            &device
                .get_func(
                    "expert_hqq_prefill_gpu_prototype_kernels",
                    "silu_mul_batched_kernel",
                )
                .ok_or_else(|| "missing silu_mul_batched_kernel".to_string())?,
        );

        Ok(Self {
            device,
            hqq4_prefill_gemm_bf16: hqq4,
            hqq6_prefill_gemm_bf16: hqq6,
            hqq8_prefill_gemm_bf16: hqq8,
            relu2_batched: relu2,
            silu_mul_batched: silu_mul,
        })
    }

    fn execute_entry(
        &self,
        cache: &ExpertHqqCache,
        plan: &ExpertHqqPrefillDispatchPlan,
        entry: &ExpertHqqPrefillDispatchEntry,
        w13: &ExpertHqqTensorRecord,
        w2: &ExpertHqqTensorRecord,
        sorted_routed_inputs: &[f32],
        output: &mut [f32],
        w13_preactivation: &mut [f32],
        activation_out: &mut [f32],
    ) -> Result<(), String> {
        validate_prefill_test_dispatch_entry(cache, plan, entry, w13, w2)?;
        let routed_hidden = cache.header.routed_hidden_size;
        let intermediate = cache.header.moe_intermediate_size;
        let w13_rows = if plan.experts_gated {
            checked_mul(2, intermediate, "GPU prototype gated W13 rows")?
        } else {
            intermediate
        };
        let row_end = entry
            .row_offset
            .checked_add(entry.row_count)
            .ok_or_else(|| "expert-HQQ GPU prototype row range overflow".to_string())?;

        let input_start = entry
            .row_offset
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ GPU prototype input offset overflow".to_string())?;
        let input_end = row_end
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ GPU prototype input end overflow".to_string())?;
        let input_bits: Vec<u16> = sorted_routed_inputs[input_start..input_end]
            .iter()
            .map(|&value| f32_to_bf16(value))
            .collect();
        let d_input = self
            .device
            .htod_copy(input_bits)
            .map_err(|e| format!("failed to upload expert-HQQ GPU prototype input: {e}"))?;
        let d_w13_out = self.alloc_bf16(entry.row_count, w13_rows, "W13 output")?;
        let d_activation = self.alloc_bf16(entry.row_count, intermediate, "activation output")?;
        let d_w2_out = self.alloc_bf16(entry.row_count, routed_hidden, "W2 output")?;

        let w13_buf = self.upload_record(w13)?;
        let w2_buf = self.upload_record(w2)?;
        self.launch_hqq_prefill_gemm(entry.row_count, &d_w13_out, &d_input, w13, &w13_buf)?;
        if plan.experts_gated {
            self.launch_silu_mul(entry.row_count, intermediate, &d_activation, &d_w13_out)?;
        } else {
            self.launch_relu2(entry.row_count, intermediate, &d_activation, &d_w13_out)?;
        }
        self.launch_hqq_prefill_gemm(entry.row_count, &d_w2_out, &d_activation, w2, &w2_buf)?;
        self.device
            .synchronize()
            .map_err(|e| format!("expert-HQQ GPU prototype synchronize failed: {e}"))?;

        let w13_bits = self
            .device
            .dtoh_sync_copy(&d_w13_out)
            .map_err(|e| format!("failed to download expert-HQQ GPU prototype W13 output: {e}"))?;
        let activation_bits = self.device.dtoh_sync_copy(&d_activation).map_err(|e| {
            format!("failed to download expert-HQQ GPU prototype activation output: {e}")
        })?;
        let out_bits = self
            .device
            .dtoh_sync_copy(&d_w2_out)
            .map_err(|e| format!("failed to download expert-HQQ GPU prototype output: {e}"))?;
        let expected_w13 = entry
            .row_count
            .checked_mul(w13_rows)
            .ok_or_else(|| "expert-HQQ GPU prototype W13 output length overflow".to_string())?;
        if w13_bits.len() != expected_w13 {
            return Err(format!(
                "expert-HQQ GPU prototype W13 output length {} != expected {}",
                w13_bits.len(),
                expected_w13
            ));
        }
        let expected_activation = entry.row_count.checked_mul(intermediate).ok_or_else(|| {
            "expert-HQQ GPU prototype activation output length overflow".to_string()
        })?;
        if activation_bits.len() != expected_activation {
            return Err(format!(
                "expert-HQQ GPU prototype activation output length {} != expected {}",
                activation_bits.len(),
                expected_activation
            ));
        }
        let expected_out = entry
            .row_count
            .checked_mul(routed_hidden)
            .ok_or_else(|| "expert-HQQ GPU prototype output length overflow".to_string())?;
        if out_bits.len() != expected_out {
            return Err(format!(
                "expert-HQQ GPU prototype output length {} != expected {}",
                out_bits.len(),
                expected_out
            ));
        }
        for local_row in 0..entry.row_count {
            let global_w13_start = (entry.row_offset + local_row)
                .checked_mul(w13_rows)
                .ok_or_else(|| "expert-HQQ GPU prototype W13 output offset overflow".to_string())?;
            let local_w13_start = local_row * w13_rows;
            for col in 0..w13_rows {
                w13_preactivation[global_w13_start + col] =
                    bf16_to_f32(w13_bits[local_w13_start + col]);
            }
            let global_activation_start = (entry.row_offset + local_row)
                .checked_mul(intermediate)
                .ok_or_else(|| {
                    "expert-HQQ GPU prototype activation output offset overflow".to_string()
                })?;
            let local_activation_start = local_row * intermediate;
            for col in 0..intermediate {
                activation_out[global_activation_start + col] =
                    bf16_to_f32(activation_bits[local_activation_start + col]);
            }
            let global_start = (entry.row_offset + local_row)
                .checked_mul(routed_hidden)
                .ok_or_else(|| "expert-HQQ GPU prototype output offset overflow".to_string())?;
            let local_start = local_row * routed_hidden;
            for col in 0..routed_hidden {
                output[global_start + col] = bf16_to_f32(out_bits[local_start + col]);
            }
        }
        Ok(())
    }

    fn upload_record(
        &self,
        record: &ExpertHqqTensorRecord,
    ) -> Result<ExpertHqqGpuTensorBuffers, String> {
        let desc = &record.descriptor;
        let groups = group_count(desc.cols, desc.group_size)?;
        let scales =
            f32_le_bytes_to_vec("GPU prototype scales", &record.scales, desc.rows * groups)?;
        let zeros = f32_le_bytes_to_vec("GPU prototype zeros", &record.zeros, desc.rows * groups)?;
        Ok(ExpertHqqGpuTensorBuffers {
            packed: self.device.htod_copy(record.packed.clone()).map_err(|e| {
                format!("failed to upload expert-HQQ GPU prototype packed payload: {e}")
            })?,
            scales: self.device.htod_copy(scales).map_err(|e| {
                format!("failed to upload expert-HQQ GPU prototype scales payload: {e}")
            })?,
            zeros: self.device.htod_copy(zeros).map_err(|e| {
                format!("failed to upload expert-HQQ GPU prototype zeros payload: {e}")
            })?,
        })
    }

    fn alloc_bf16(&self, rows: usize, cols: usize, label: &str) -> Result<CudaSlice<u16>, String> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| format!("expert-HQQ GPU prototype {label} allocation overflow"))?;
        self.device
            .alloc_zeros::<u16>(len)
            .map_err(|e| format!("failed to allocate expert-HQQ GPU prototype {label}: {e}"))
    }

    fn launch_hqq_prefill_gemm(
        &self,
        m: usize,
        output: &CudaSlice<u16>,
        input: &CudaSlice<u16>,
        record: &ExpertHqqTensorRecord,
        buffers: &ExpertHqqGpuTensorBuffers,
    ) -> Result<(), String> {
        let desc = &record.descriptor;
        let groups = group_count(desc.cols, desc.group_size)?;
        let padded = padded_cols(desc.cols, desc.group_size)?;
        let packed_row_stride_bytes = match desc.nbits {
            4 => padded.div_ceil(2),
            6 => padded.div_ceil(4) * 3,
            8 => padded,
            other => {
                return Err(format!(
                    "expert-HQQ GPU prototype supports HQQ4/HQQ6/HQQ8 only, got nbits={other}"
                ))
            }
        };
        let scales_row_stride_bytes = groups
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "expert-HQQ GPU prototype scales row stride overflow".to_string())?;
        let zeros_row_stride_bytes = scales_row_stride_bytes;
        let kernel = match desc.nbits {
            4 => self.hqq4_prefill_gemm_bf16,
            6 => self.hqq6_prefill_gemm_bf16,
            8 => self.hqq8_prefill_gemm_bf16,
            other => {
                return Err(format!(
                    "expert-HQQ GPU prototype supports HQQ4/HQQ6/HQQ8 only, got nbits={other}"
                ))
            }
        };

        let grid_x = (desc.rows as u32).div_ceil(8);
        let grid_y = (m as u32).div_ceil(8);
        let mut a0 = *output.device_ptr();
        let mut a1 = *input.device_ptr();
        let mut a2 = *buffers.packed.device_ptr();
        let mut a3 = *buffers.scales.device_ptr();
        let mut a4 = *buffers.zeros.device_ptr();
        let mut a5 = m as i32;
        let mut a6 = desc.rows as i32;
        let mut a7 = desc.cols as i32;
        let mut a8 = desc.group_size as i32;
        let mut a9 = packed_row_stride_bytes as i32;
        let mut a10 = scales_row_stride_bytes as i32;
        let mut a11 = zeros_row_stride_bytes as i32;
        unsafe {
            launch_expert_hqq_gpu_test_kernel(
                kernel,
                (grid_x, grid_y, 1),
                (256, 1, 1),
                &mut [
                    &mut a0 as *mut _ as *mut std::ffi::c_void,
                    &mut a1 as *mut _ as *mut std::ffi::c_void,
                    &mut a2 as *mut _ as *mut std::ffi::c_void,
                    &mut a3 as *mut _ as *mut std::ffi::c_void,
                    &mut a4 as *mut _ as *mut std::ffi::c_void,
                    &mut a5 as *mut _ as *mut std::ffi::c_void,
                    &mut a6 as *mut _ as *mut std::ffi::c_void,
                    &mut a7 as *mut _ as *mut std::ffi::c_void,
                    &mut a8 as *mut _ as *mut std::ffi::c_void,
                    &mut a9 as *mut _ as *mut std::ffi::c_void,
                    &mut a10 as *mut _ as *mut std::ffi::c_void,
                    &mut a11 as *mut _ as *mut std::ffi::c_void,
                ],
            )
        }
    }

    fn launch_relu2(
        &self,
        m: usize,
        n: usize,
        output: &CudaSlice<u16>,
        input: &CudaSlice<u16>,
    ) -> Result<(), String> {
        let threads = activation_thread_count(n)?;
        let mut a0 = *output.device_ptr();
        let mut a1 = *input.device_ptr();
        let mut a2 = n as i32;
        unsafe {
            launch_expert_hqq_gpu_test_kernel(
                self.relu2_batched,
                (m as u32, 1, 1),
                (threads, 1, 1),
                &mut [
                    &mut a0 as *mut _ as *mut std::ffi::c_void,
                    &mut a1 as *mut _ as *mut std::ffi::c_void,
                    &mut a2 as *mut _ as *mut std::ffi::c_void,
                ],
            )
        }
    }

    fn launch_silu_mul(
        &self,
        m: usize,
        n: usize,
        output: &CudaSlice<u16>,
        input: &CudaSlice<u16>,
    ) -> Result<(), String> {
        let threads = activation_thread_count(n)?;
        let mut a0 = *output.device_ptr();
        let mut a1 = *input.device_ptr();
        let mut a2 = n as i32;
        unsafe {
            launch_expert_hqq_gpu_test_kernel(
                self.silu_mul_batched,
                (m as u32, 1, 1),
                (threads, 1, 1),
                &mut [
                    &mut a0 as *mut _ as *mut std::ffi::c_void,
                    &mut a1 as *mut _ as *mut std::ffi::c_void,
                    &mut a2 as *mut _ as *mut std::ffi::c_void,
                ],
            )
        }
    }
}

#[cfg(all(test, has_prefill_kernels))]
fn activation_thread_count(n: usize) -> Result<u32, String> {
    if n == 0 {
        return Err("expert-HQQ GPU prototype activation width must be > 0".to_string());
    }
    let threads = n.min(1024).div_ceil(32) * 32;
    Ok(threads as u32)
}

pub fn build_expert_hqq_cache_from_safetensors(
    header: ExpertHqqCacheHeader,
    specs: &[ExpertHqqSafetensorsTensorSpec],
) -> Result<ExpertHqqCache, String> {
    let mut inputs = Vec::with_capacity(specs.len());
    for spec in specs {
        inputs.push(build_expert_hqq_input_from_safetensors(spec)?);
    }
    ExpertHqqCache::from_inputs(header, inputs)
}

pub fn write_expert_hqq_cache_from_safetensors(
    path: &Path,
    header: ExpertHqqCacheHeader,
    specs: &[ExpertHqqSafetensorsTensorSpec],
) -> Result<ExpertHqqCache, String> {
    let mut inputs = Vec::with_capacity(specs.len());
    for spec in specs {
        inputs.push(build_expert_hqq_input_from_safetensors(spec)?);
    }
    write_expert_hqq_cache_from_inputs(path, header, inputs)
}

fn build_expert_hqq_input_from_safetensors(
    spec: &ExpertHqqSafetensorsTensorSpec,
) -> Result<ExpertHqqTensorInput, String> {
    let (weights, rows, cols) = load_safetensors_tensor_2d_f32(&spec.path, &spec.key)?;
    let (packed, scales, zeros) =
        quantize_expert_hqq_tensor_f32(&weights, rows, cols, spec.nbits, spec.group_size)?;
    ExpertHqqTensorInput::new(
        spec.role,
        spec.layer_idx,
        spec.expert_idx,
        rows,
        cols,
        spec.nbits,
        spec.group_size,
        packed,
        f32_vec_to_le_bytes(&scales),
        f32_vec_to_le_bytes(&zeros),
    )
}

fn load_safetensors_tensor_2d_f32(
    path: &Path,
    key: &str,
) -> Result<(Vec<f32>, usize, usize), String> {
    let safetensors = MmapSafetensors::open(path)
        .map_err(|e| format!("failed to open safetensors {}: {e}", path.display()))?;
    let info = safetensors
        .tensor_info(key)
        .ok_or_else(|| format!("Tensor not found: {key} in {}", path.display()))?;
    if info.shape.len() != 2 {
        return Err(format!(
            "expert-HQQ safetensors tensor {key} must be 2D, got shape {:?}",
            info.shape
        ));
    }
    let rows = info.shape[0];
    let cols = info.shape[1];
    if rows == 0 || cols == 0 {
        return Err(format!(
            "expert-HQQ safetensors tensor {key} must have positive shape, got {:?}",
            info.shape
        ));
    }
    let data = safetensors
        .tensor_data(key)
        .map_err(|e| format!("failed to read safetensors tensor {key}: {e}"))?;
    let expected_bytes = info
        .numel()
        .checked_mul(info.dtype.element_size())
        .ok_or_else(|| format!("expert-HQQ safetensors tensor {key} byte count overflow"))?;
    if data.len() != expected_bytes {
        return Err(format!(
            "expert-HQQ safetensors tensor {key} byte length {} != expected {}",
            data.len(),
            expected_bytes
        ));
    }
    let values = match &info.dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        Dtype::Bf16 => data
            .chunks_exact(2)
            .map(|chunk| bf16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
            .collect(),
        other => {
            return Err(format!(
                "expert-HQQ safetensors tensor {key} dtype {:?} is unsupported; expected BF16, F16, or F32",
                other
            ))
        }
    };
    Ok((values, rows, cols))
}

fn quantize_expert_hqq_tensor_f32(
    input: &[f32],
    rows: usize,
    cols: usize,
    nbits: u8,
    group_size: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>), String> {
    match nbits {
        4 => crate::hqq::hqq4_quantize_tensor_to_components(input, rows, cols, group_size, None),
        6 => quantize_hqq_affine_tensor_to_components(input, rows, cols, group_size, 63.0),
        8 => quantize_hqq_affine_tensor_to_components(input, rows, cols, group_size, 255.0),
        other => Err(format!(
            "expert-HQQ safetensors builder supports HQQ4/HQQ6/HQQ8 only, got nbits={other}"
        )),
    }
}

fn dequantize_expert_hqq_record_to_f32(record: &ExpertHqqTensorRecord) -> Result<Vec<f32>, String> {
    record.validate_payload_lengths()?;
    let desc = &record.descriptor;
    if desc.nbits != 4 && desc.nbits != 6 && desc.nbits != 8 {
        return Err(format!(
            "expert-HQQ prefill reference execution supports HQQ4/HQQ6/HQQ8 only, got nbits={}",
            desc.nbits
        ));
    }
    if desc.axis != EXPERT_HQQ_AXIS {
        return Err(format!(
            "expert-HQQ prefill reference execution requires axis={}, got {}",
            EXPERT_HQQ_AXIS, desc.axis
        ));
    }
    if desc.layout != expert_hqq_layout_for_nbits(desc.nbits)? {
        return Err(format!(
            "expert-HQQ prefill reference execution layout '{}' does not match nbits {}",
            desc.layout, desc.nbits
        ));
    }
    let groups = group_count(desc.cols, desc.group_size)?;
    let scales = f32_le_bytes_to_vec("scales", &record.scales, desc.rows * groups)?;
    let zeros = f32_le_bytes_to_vec("zeros", &record.zeros, desc.rows * groups)?;
    let padded = padded_cols(desc.cols, desc.group_size)?;
    let mut out = vec![0.0f32; desc.rows * desc.cols];
    for row in 0..desc.rows {
        for col in 0..desc.cols {
            let group = col / desc.group_size;
            let q = decode_expert_hqq_qvalue(desc, &record.packed, row, col, padded)?;
            let meta = row * groups + group;
            out[row * desc.cols + col] = (q as f32 - zeros[meta]) * scales[meta];
        }
    }
    Ok(out)
}

fn decode_expert_hqq_qvalue(
    desc: &ExpertHqqTensorDescriptor,
    packed: &[u8],
    row: usize,
    col: usize,
    padded_cols: usize,
) -> Result<u8, String> {
    match desc.nbits {
        4 => {
            let packed_cols = padded_cols.div_ceil(2);
            let byte = packed
                .get(row * packed_cols + col / 2)
                .ok_or_else(|| "expert-HQQ uint4 packed index out of range".to_string())?;
            Ok(if col % 2 == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            })
        }
        6 => {
            let block = col / 4;
            let rem = col % 4;
            let packed_cols = padded_cols.div_ceil(4) * 3;
            let base = row
                .checked_mul(packed_cols)
                .and_then(|v| v.checked_add(block * 3))
                .ok_or_else(|| "expert-HQQ uint6 packed index overflow".to_string())?;
            let a = *packed
                .get(base)
                .ok_or_else(|| "expert-HQQ uint6 packed index out of range".to_string())?;
            let b = *packed
                .get(base + 1)
                .ok_or_else(|| "expert-HQQ uint6 packed index out of range".to_string())?;
            let c = *packed
                .get(base + 2)
                .ok_or_else(|| "expert-HQQ uint6 packed index out of range".to_string())?;
            Ok(match rem {
                0 => a & 0x3f,
                1 => ((a >> 6) | ((b & 0x0f) << 2)) & 0x3f,
                2 => ((b >> 4) | ((c & 0x03) << 4)) & 0x3f,
                _ => (c >> 2) & 0x3f,
            })
        }
        8 => {
            let packed_cols = padded_cols;
            packed
                .get(row * packed_cols + col)
                .copied()
                .ok_or_else(|| "expert-HQQ uint8 packed index out of range".to_string())
        }
        other => Err(format!(
            "expert-HQQ prefill reference execution supports HQQ4/HQQ6/HQQ8 only, got nbits={other}"
        )),
    }
}

fn validate_prefill_reference_plan(
    cache: &ExpertHqqCache,
    plan: &ExpertHqqPrefillDispatchPlan,
    sorted_row_count: usize,
) -> Result<(), String> {
    if plan.layer_idx >= cache.header.num_moe_layers {
        return Err(format!(
            "expert-HQQ prefill reference layer_idx {} out of range {}",
            plan.layer_idx, cache.header.num_moe_layers
        ));
    }
    if plan.entries.is_empty() {
        return Err(
            "expert-HQQ prefill reference execution requires at least one plan entry".to_string(),
        );
    }
    if sorted_row_count == 0 {
        return Err(
            "expert-HQQ prefill reference execution requires sorted_row_count > 0".to_string(),
        );
    }
    let expected = ExpertHqqPrefillDispatchPlan {
        layer_idx: plan.layer_idx,
        experts_gated: plan.experts_gated,
        input_layout: "row_major_selected_rows_by_routed_hidden",
        w13_dequant_layout: "row_major_axis1_grouped_rows_by_routed_hidden",
        w13_output_layout: "row_major_selected_rows_by_w13_rows",
        activation_output_layout: "row_major_selected_rows_by_moe_intermediate",
        w2_dequant_layout: "row_major_axis1_grouped_routed_hidden_by_moe_intermediate",
        w2_output_layout: "row_major_selected_rows_by_routed_hidden",
        entries: Vec::new(),
    };
    if plan.input_layout != expected.input_layout
        || plan.w13_dequant_layout != expected.w13_dequant_layout
        || plan.w13_output_layout != expected.w13_output_layout
        || plan.activation_output_layout != expected.activation_output_layout
        || plan.w2_dequant_layout != expected.w2_dequant_layout
        || plan.w2_output_layout != expected.w2_output_layout
    {
        return Err("expert-HQQ prefill reference execution plan layout mismatch".to_string());
    }
    Ok(())
}

#[cfg(test)]
fn validate_prefill_test_dispatch_plan(
    cache: &ExpertHqqCache,
    plan: &ExpertHqqPrefillDispatchPlan,
    sorted_row_count: usize,
) -> Result<(), String> {
    validate_prefill_reference_plan(cache, plan, sorted_row_count).map_err(|err| {
        err.replace(
            "expert-HQQ prefill reference execution",
            "expert-HQQ prefill test dispatch",
        )
        .replace(
            "expert-HQQ prefill reference",
            "expert-HQQ prefill test dispatch",
        )
    })
}

fn validate_prefill_reference_entry(
    cache: &ExpertHqqCache,
    plan: &ExpertHqqPrefillDispatchPlan,
    entry: &ExpertHqqPrefillDispatchEntry,
    w13: &ExpertHqqTensorRecord,
    w2: &ExpertHqqTensorRecord,
) -> Result<(), String> {
    if entry.w13_key.role != ExpertHqqTensorRole::W13
        || w13.descriptor.role != ExpertHqqTensorRole::W13
    {
        return Err(format!(
            "expert-HQQ prefill reference W13 role mismatch for expert {}",
            entry.expert_idx
        ));
    }
    if entry.w2_key.role != ExpertHqqTensorRole::W2 || w2.descriptor.role != ExpertHqqTensorRole::W2
    {
        return Err(format!(
            "expert-HQQ prefill reference W2 role mismatch for expert {}",
            entry.expert_idx
        ));
    }
    if entry.w13_key.layer_idx != plan.layer_idx
        || entry.w2_key.layer_idx != plan.layer_idx
        || w13.descriptor.layer_idx != plan.layer_idx
        || w2.descriptor.layer_idx != plan.layer_idx
    {
        return Err(format!(
            "expert-HQQ prefill reference layer mismatch for expert {}",
            entry.expert_idx
        ));
    }
    if entry.w13_key.expert_idx != entry.expert_idx
        || entry.w2_key.expert_idx != entry.expert_idx
        || w13.descriptor.expert_idx != entry.expert_idx
        || w2.descriptor.expert_idx != entry.expert_idx
    {
        return Err(format!(
            "expert-HQQ prefill reference expert mismatch for selected expert {}",
            entry.expert_idx
        ));
    }
    let expected_w13_rows = if plan.experts_gated {
        checked_mul(
            2,
            cache.header.moe_intermediate_size,
            "reference gated W13 rows",
        )?
    } else {
        cache.header.moe_intermediate_size
    };
    if w13.descriptor.rows != expected_w13_rows
        || w13.descriptor.cols != cache.header.routed_hidden_size
        || w2.descriptor.rows != cache.header.routed_hidden_size
        || w2.descriptor.cols != cache.header.moe_intermediate_size
    {
        return Err(format!(
            "expert-HQQ prefill reference descriptor shape mismatch for expert {}",
            entry.expert_idx
        ));
    }
    if entry.w13_rows != w13.descriptor.rows
        || entry.w13_cols != w13.descriptor.cols
        || entry.w2_rows != w2.descriptor.rows
        || entry.w2_cols != w2.descriptor.cols
        || entry.w13_nbits != w13.descriptor.nbits
        || entry.w2_nbits != w2.descriptor.nbits
        || entry.w13_group_size != w13.descriptor.group_size
        || entry.w2_group_size != w2.descriptor.group_size
    {
        return Err(format!(
            "expert-HQQ prefill reference plan/cache metadata mismatch for expert {}",
            entry.expert_idx
        ));
    }
    if w13.descriptor.nbits != 4 && w13.descriptor.nbits != 6 && w13.descriptor.nbits != 8 {
        return Err(format!(
            "expert-HQQ prefill reference execution supports HQQ4/HQQ6/HQQ8 W13 only, got nbits={}",
            w13.descriptor.nbits
        ));
    }
    if w2.descriptor.nbits != 4 && w2.descriptor.nbits != 6 && w2.descriptor.nbits != 8 {
        return Err(format!(
            "expert-HQQ prefill reference execution supports HQQ4/HQQ6/HQQ8 W2 only, got nbits={}",
            w2.descriptor.nbits
        ));
    }
    Ok(())
}

#[cfg(test)]
fn validate_prefill_test_dispatch_entry(
    cache: &ExpertHqqCache,
    plan: &ExpertHqqPrefillDispatchPlan,
    entry: &ExpertHqqPrefillDispatchEntry,
    w13: &ExpertHqqTensorRecord,
    w2: &ExpertHqqTensorRecord,
) -> Result<(), String> {
    validate_prefill_reference_entry(cache, plan, entry, w13, w2).map_err(|err| {
        err.replace(
            "expert-HQQ prefill reference execution",
            "expert-HQQ prefill test dispatch",
        )
        .replace(
            "expert-HQQ prefill reference",
            "expert-HQQ prefill test dispatch",
        )
    })
}

fn f32_le_bytes_to_vec(label: &str, bytes: &[u8], expected_len: usize) -> Result<Vec<f32>, String> {
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("expert-HQQ {label} byte count overflow"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "expert-HQQ {label} byte length {} != expected {}",
            bytes.len(),
            expected_bytes
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[inline]
fn dot_f32(weights: &[f32], values: &[f32]) -> f32 {
    weights
        .iter()
        .zip(values.iter())
        .map(|(&w, &v)| w * v)
        .sum()
}

#[inline]
fn round_to_bf16_path_f32(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

#[inline]
fn dot_f32_gpu_lane_order(weights: &[f32], values: &[f32]) -> f32 {
    let mut partial = [0.0f32; 4];
    for lane in 0..4 {
        let mut acc = 0.0f32;
        let mut col = lane;
        while col < weights.len() {
            acc = weights[col].mul_add(values[col], acc);
            col += 4;
        }
        partial[lane] = acc;
    }
    ((partial[0] + partial[1]) + partial[2]) + partial[3]
}

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn quantize_hqq_affine_tensor_to_components(
    input: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    qmax: f32,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>), String> {
    if rows == 0 || cols == 0 {
        return Err("expert-HQQ affine quantizer expects positive rows and cols".to_string());
    }
    if group_size == 0 {
        return Err("expert-HQQ affine quantizer expects positive group_size".to_string());
    }
    let expected_len = rows
        .checked_mul(cols)
        .ok_or_else(|| "expert-HQQ affine rows * cols overflowed".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "expert-HQQ affine input length {} != rows*cols {}",
            input.len(),
            expected_len
        ));
    }
    let groups = group_count(cols, group_size)?;
    let padded = padded_cols(cols, group_size)?;
    let mut quant = vec![0u8; rows * padded];
    let mut scales = vec![0.0f32; rows * groups];
    let mut zeros = vec![0.0f32; rows * groups];
    for row in 0..rows {
        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(cols);
            let mut minv = f32::INFINITY;
            let mut maxv = f32::NEG_INFINITY;
            for col in start..end {
                let value = input[row * cols + col];
                minv = minv.min(value);
                maxv = maxv.max(value);
            }
            let scale = ((maxv - minv) / qmax).max(1e-8);
            let zero = (-minv / scale).clamp(0.0, qmax);
            scales[row * groups + group] = scale;
            zeros[row * groups + group] = zero;
            for col in start..end {
                let q = ((input[row * cols + col] / scale) + zero).round_ties_even();
                quant[row * padded + col] = q.clamp(0.0, qmax) as u8;
            }
        }
    }
    let packed = if qmax >= 255.0 {
        quant
    } else {
        pack_uint6_rows(&quant, rows, padded)?
    };
    Ok((packed, scales, zeros))
}

fn pack_uint6_rows(quant: &[u8], rows: usize, padded_cols: usize) -> Result<Vec<u8>, String> {
    let row_len = rows
        .checked_mul(padded_cols)
        .ok_or_else(|| "expert-HQQ uint6 quant length overflow".to_string())?;
    if quant.len() != row_len {
        return Err(format!(
            "expert-HQQ uint6 quant length {} != rows*padded_cols {}",
            quant.len(),
            row_len
        ));
    }
    let blocks = padded_cols.div_ceil(4);
    let packed_cols = blocks
        .checked_mul(3)
        .ok_or_else(|| "expert-HQQ uint6 packed column overflow".to_string())?;
    let mut packed = vec![0u8; rows * packed_cols];
    for row in 0..rows {
        let in_base = row * padded_cols;
        let out_base = row * packed_cols;
        for block in 0..blocks {
            let src = block * 4;
            let a = quant[in_base + src] & 0x3f;
            let b = if src + 1 < padded_cols {
                quant[in_base + src + 1] & 0x3f
            } else {
                0
            };
            let c = if src + 2 < padded_cols {
                quant[in_base + src + 2] & 0x3f
            } else {
                0
            };
            let d = if src + 3 < padded_cols {
                quant[in_base + src + 3] & 0x3f
            } else {
                0
            };
            let dst = out_base + block * 3;
            packed[dst] = a | ((b & 0x03) << 6);
            packed[dst + 1] = (b >> 2) | ((c & 0x0f) << 4);
            packed[dst + 2] = (c >> 4) | (d << 2);
        }
    }
    Ok(packed)
}

fn f32_vec_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn expected_runtime_w13_rows(
    experts_gated: bool,
    moe_intermediate_size: usize,
) -> Result<usize, String> {
    if experts_gated {
        checked_mul(
            2,
            moe_intermediate_size,
            "expert-HQQ runtime prefill diagnostic gated W13 rows",
        )
    } else {
        Ok(moe_intermediate_size)
    }
}

fn bf16_path_oracle_metadata(
    model: ExpertHqqRuntimeDiagnosticModelShape,
    sorted_row_count: usize,
    input_layout: &'static str,
    w13_output_layout: &'static str,
    activation_output_layout: &'static str,
    w2_output_layout: &'static str,
) -> Result<ExpertHqqBf16PathOracleMetadata, String> {
    let w13_rows = expected_runtime_w13_rows(model.experts_gated, model.moe_intermediate_size)?;
    Ok(ExpertHqqBf16PathOracleMetadata {
        sorted_row_count,
        routed_hidden_size: model.routed_hidden_size,
        w13_rows,
        moe_intermediate_size: model.moe_intermediate_size,
        input_bf16_values: checked_mul(
            sorted_row_count,
            model.routed_hidden_size,
            "BF16-path oracle input values",
        )?,
        w13_preactivation_values: checked_mul(
            sorted_row_count,
            w13_rows,
            "BF16-path oracle W13 values",
        )?,
        activation_values: checked_mul(
            sorted_row_count,
            model.moe_intermediate_size,
            "BF16-path oracle activation values",
        )?,
        output_values: checked_mul(
            sorted_row_count,
            model.routed_hidden_size,
            "BF16-path oracle output values",
        )?,
        input_layout,
        w13_output_layout,
        activation_output_layout,
        w2_output_layout,
        correctness_oracle: "bf16_path_oracle",
    })
}

fn runtime_buffer_len(rows: usize, row_stride: usize, width: usize) -> Result<usize, String> {
    if rows == 0 {
        return Ok(0);
    }
    rows.checked_sub(1)
        .and_then(|last| last.checked_mul(row_stride))
        .and_then(|start| start.checked_add(width))
        .ok_or_else(|| "expert-HQQ runtime-shaped buffer length overflow".to_string())
}

#[cfg(all(test, has_prefill_kernels))]
fn scatter_runtime_row(
    src: &[f32],
    src_row: usize,
    width: usize,
    dst: &mut [f32],
    dst_row: usize,
    dst_row_stride: usize,
) -> Result<(), String> {
    let src_start = src_row
        .checked_mul(width)
        .ok_or_else(|| "expert-HQQ runtime-shaped compact scatter source overflow".to_string())?;
    let src_end = src_start.checked_add(width).ok_or_else(|| {
        "expert-HQQ runtime-shaped compact scatter source end overflow".to_string()
    })?;
    let dst_start = dst_row
        .checked_mul(dst_row_stride)
        .ok_or_else(|| "expert-HQQ runtime-shaped scatter destination overflow".to_string())?;
    let dst_end = dst_start
        .checked_add(width)
        .ok_or_else(|| "expert-HQQ runtime-shaped scatter destination end overflow".to_string())?;
    if src_end > src.len() {
        return Err(format!(
            "expert-HQQ runtime-shaped compact scatter source range {}..{} exceeds {}",
            src_start,
            src_end,
            src.len()
        ));
    }
    if dst_end > dst.len() {
        return Err(format!(
            "expert-HQQ runtime-shaped scatter destination range {}..{} exceeds {}",
            dst_start,
            dst_end,
            dst.len()
        ));
    }
    dst[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
    Ok(())
}

fn write_header<W: Write>(writer: &mut W, header: &ExpertHqqCacheHeader) -> Result<(), String> {
    writer
        .write_all(EXPERT_HQQ_CACHE_MAGIC)
        .map_err(|e| format!("failed to write expert-HQQ magic: {e}"))?;
    writer
        .write_all(&header.version.to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ version: {e}"))?;
    write_u64(writer, header.hidden_size, "hidden_size")?;
    write_u64(writer, header.routed_hidden_size, "routed_hidden_size")?;
    write_u64(
        writer,
        header.moe_intermediate_size,
        "moe_intermediate_size",
    )?;
    write_u64(writer, header.n_routed_experts, "n_routed_experts")?;
    write_u64(writer, header.num_moe_layers, "num_moe_layers")?;
    writer
        .write_all(&header.config_hash.to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ config_hash: {e}"))?;
    write_u64(writer, header.tensor_count, "tensor_count")
}

fn read_header<R: Read>(reader: &mut R) -> Result<ExpertHqqCacheHeader, String> {
    let mut header = [0u8; EXPERT_HQQ_HEADER_SIZE];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("failed to read expert-HQQ header: {e}"))?;
    if &header[0..4] != EXPERT_HQQ_CACHE_MAGIC {
        return Err("bad magic in expert-HQQ cache".to_string());
    }
    Ok(ExpertHqqCacheHeader {
        version: u32::from_le_bytes(header[4..8].try_into().unwrap()),
        hidden_size: u64_to_usize(
            u64::from_le_bytes(header[8..16].try_into().unwrap()),
            "hidden_size",
        )?,
        routed_hidden_size: u64_to_usize(
            u64::from_le_bytes(header[16..24].try_into().unwrap()),
            "routed_hidden_size",
        )?,
        moe_intermediate_size: u64_to_usize(
            u64::from_le_bytes(header[24..32].try_into().unwrap()),
            "moe_intermediate_size",
        )?,
        n_routed_experts: u64_to_usize(
            u64::from_le_bytes(header[32..40].try_into().unwrap()),
            "n_routed_experts",
        )?,
        num_moe_layers: u64_to_usize(
            u64::from_le_bytes(header[40..48].try_into().unwrap()),
            "num_moe_layers",
        )?,
        config_hash: u64::from_le_bytes(header[48..56].try_into().unwrap()),
        tensor_count: u64_to_usize(
            u64::from_le_bytes(header[56..64].try_into().unwrap()),
            "tensor_count",
        )?,
    })
}

fn write_descriptor<W: Write>(
    writer: &mut W,
    desc: &ExpertHqqTensorDescriptor,
) -> Result<(), String> {
    writer
        .write_all(&desc.role.tag().to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ role: {e}"))?;
    writer
        .write_all(&(desc.nbits as u32).to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ nbits: {e}"))?;
    write_u64(writer, desc.layer_idx, "layer_idx")?;
    write_u64(writer, desc.expert_idx, "expert_idx")?;
    write_u64(writer, desc.rows, "rows")?;
    write_u64(writer, desc.cols, "cols")?;
    write_u64(writer, desc.group_size, "group_size")?;
    write_u64(writer, desc.axis, "axis")?;
    writer
        .write_all(&layout_code_for_nbits(desc.nbits)?.to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ layout code: {e}"))?;
    writer
        .write_all(&0u32.to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ reserved field: {e}"))?;
    write_u64(writer, desc.packed_bytes, "packed_bytes")?;
    write_u64(writer, desc.scales_bytes, "scales_bytes")?;
    write_u64(writer, desc.zeros_bytes, "zeros_bytes")
}

fn read_descriptor<R: Read>(reader: &mut R) -> Result<ExpertHqqTensorDescriptor, String> {
    let mut bytes = [0u8; EXPERT_HQQ_TENSOR_DESCRIPTOR_SIZE];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("failed to read expert-HQQ tensor descriptor: {e}"))?;
    let role = ExpertHqqTensorRole::from_tag(u32::from_le_bytes(bytes[0..4].try_into().unwrap()))?;
    let nbits_u32 = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let nbits =
        u8::try_from(nbits_u32).map_err(|_| format!("expert-HQQ nbits {nbits_u32} exceeds u8"))?;
    let layout_nbits =
        nbits_from_layout_code(u32::from_le_bytes(bytes[56..60].try_into().unwrap()))?;
    if layout_nbits != nbits {
        return Err(format!(
            "expert-HQQ descriptor nbits {} does not match layout code {}",
            nbits, layout_nbits
        ));
    }
    let reserved = u32::from_le_bytes(bytes[60..64].try_into().unwrap());
    if reserved != 0 {
        return Err(format!(
            "expert-HQQ descriptor reserved field must be zero, got {reserved}"
        ));
    }
    Ok(ExpertHqqTensorDescriptor {
        role,
        nbits,
        layer_idx: u64_to_usize(
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            "layer_idx",
        )?,
        expert_idx: u64_to_usize(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            "expert_idx",
        )?,
        rows: u64_to_usize(
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            "rows",
        )?,
        cols: u64_to_usize(
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            "cols",
        )?,
        group_size: u64_to_usize(
            u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            "group_size",
        )?,
        axis: u64_to_usize(
            u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            "axis",
        )?,
        layout: expert_hqq_layout_for_nbits(nbits)?.to_string(),
        packed_dtype: EXPERT_HQQ_PACKED_DTYPE.to_string(),
        scales_dtype: EXPERT_HQQ_SCALES_DTYPE.to_string(),
        zeros_dtype: EXPERT_HQQ_ZEROS_DTYPE.to_string(),
        packed_bytes: u64_to_usize(
            u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            "packed_bytes",
        )?,
        scales_bytes: u64_to_usize(
            u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
            "scales_bytes",
        )?,
        zeros_bytes: u64_to_usize(
            u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
            "zeros_bytes",
        )?,
    })
}

fn write_u64<W: Write>(writer: &mut W, value: usize, label: &str) -> Result<(), String> {
    let encoded = u64::try_from(value).map_err(|_| format!("expert-HQQ {label} exceeds u64"))?;
    writer
        .write_all(&encoded.to_le_bytes())
        .map_err(|e| format!("failed to write expert-HQQ {label}: {e}"))
}

fn u64_to_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("expert-HQQ {label} exceeds usize"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use serde::Deserialize;
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::env;
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom};
    use std::path::Path;

    fn sample_header(tensor_count: usize) -> ExpertHqqCacheHeader {
        ExpertHqqCacheHeader::new(8, 8, 4, 3, 2, 0x1234_5678_9abc_def0, tensor_count).unwrap()
    }

    fn sample_record(
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        nbits: u8,
    ) -> ExpertHqqTensorRecord {
        sample_input(role, layer_idx, expert_idx, nbits)
            .into_record()
            .unwrap()
    }

    fn sample_input(
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        nbits: u8,
    ) -> ExpertHqqTensorInput {
        let (rows, cols) = match role {
            ExpertHqqTensorRole::W13 => (4, 8),
            ExpertHqqTensorRole::W2 => (8, 4),
        };
        let desc =
            ExpertHqqTensorDescriptor::new(role, layer_idx, expert_idx, rows, cols, nbits, 4)
                .unwrap();
        let packed = (0..desc.packed_bytes).map(|v| (v & 0xff) as u8).collect();
        let scales = (0..desc.scales_bytes)
            .map(|v| (v.wrapping_mul(3) & 0xff) as u8)
            .collect();
        let zeros = (0..desc.zeros_bytes)
            .map(|v| (v.wrapping_mul(5) & 0xff) as u8)
            .collect();
        ExpertHqqTensorInput::new(
            role, layer_idx, expert_idx, rows, cols, nbits, 4, packed, scales, zeros,
        )
        .unwrap()
    }

    fn quantized_input_from_f32(
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        rows: usize,
        cols: usize,
        nbits: u8,
        group_size: usize,
        values: &[f32],
    ) -> ExpertHqqTensorInput {
        let (packed, scales, zeros) =
            quantize_expert_hqq_tensor_f32(values, rows, cols, nbits, group_size).unwrap();
        ExpertHqqTensorInput::new(
            role,
            layer_idx,
            expert_idx,
            rows,
            cols,
            nbits,
            group_size,
            packed,
            f32_vec_to_le_bytes(&scales),
            f32_vec_to_le_bytes(&zeros),
        )
        .unwrap()
    }

    fn sample_weight_store() -> crate::weights::WeightStore {
        let mut store = crate::weights::WeightStore::new();
        store.config.hidden_size = 8;
        store.config.moe_latent_size = 0;
        store.config.moe_intermediate_size = 4;
        store.config.n_routed_experts = 3;
        store.config.num_hidden_layers = 2;
        store.config.moe_layer_indices = vec![0, 1];
        store
    }

    fn sample_runtime_diagnostic_model(
        experts_gated: bool,
    ) -> ExpertHqqRuntimeDiagnosticModelShape {
        ExpertHqqRuntimeDiagnosticModelShape {
            hidden_size: 8,
            routed_hidden_size: 8,
            moe_intermediate_size: 4,
            n_routed_experts: 3,
            num_hidden_layers: 2,
            experts_gated,
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "krasis_{name}_{}_{}.bin",
            std::process::id(),
            unique_suffix()
        ))
    }

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    fn synthetic_values(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|idx| (((idx * 17 + seed * 11) % 41) as f32 - 20.0) / 37.0)
            .collect()
    }

    fn f32_test_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn write_diagnostic_cache_spec(
        spec_path: &Path,
        cache_path: &str,
        layer_idx: usize,
        experts: &[usize],
        roles: &[&str],
        nbits: u8,
        group_size: usize,
    ) {
        let experts_json = experts
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let roles_json = roles
            .iter()
            .map(|role| format!("\"{role}\""))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            spec_path,
            format!(
                "{{\"purpose\":\"runtime_prefill_diagnostic\",\"cache_path\":\"{cache_path}\",\"requirements\":[{{\"layer_idx\":{layer_idx},\"experts\":[{experts_json}],\"roles\":[{roles_json}],\"nbits\":{nbits},\"group_size\":{group_size}}}]}}"
            ),
        )
        .unwrap();
    }

    fn write_f32_safetensors(path: &Path, entries: &[(&str, Vec<usize>, Vec<f32>)]) {
        let owned: Vec<(String, Vec<usize>, Vec<u8>)> = entries
            .iter()
            .map(|(name, shape, values)| {
                ((*name).to_string(), shape.clone(), f32_test_bytes(values))
            })
            .collect();
        let mut views = Vec::with_capacity(owned.len());
        for (name, shape, bytes) in &owned {
            views.push((
                name.as_str(),
                TensorView::new(SafeDtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            ));
        }
        serialize_to_file(views, &None, path).unwrap();
    }

    fn write_owned_f32_safetensors(path: &Path, entries: &[(String, Vec<usize>, Vec<f32>)]) {
        let owned: Vec<(String, Vec<usize>, Vec<u8>)> = entries
            .iter()
            .map(|(name, shape, values)| (name.clone(), shape.clone(), f32_test_bytes(values)))
            .collect();
        let mut views = Vec::with_capacity(owned.len());
        for (name, shape, bytes) in &owned {
            views.push((
                name.as_str(),
                TensorView::new(SafeDtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
            ));
        }
        serialize_to_file(views, &None, path).unwrap();
    }

    fn write_generation_model_config(model_dir: &Path, n_experts: usize) -> u64 {
        std::fs::create_dir_all(model_dir).unwrap();
        let raw = format!(
            "{{\"hidden_size\":8,\"moe_intermediate_size\":4,\"n_routed_experts\":{n_experts},\"num_experts_per_tok\":2,\"num_hidden_layers\":2,\"mlp_hidden_act\":\"relu2\",\"hybrid_override_pattern\":\"EE\"}}\n"
        );
        std::fs::write(model_dir.join("config.json"), raw.as_bytes()).unwrap();
        expert_hqq_config_hash(raw.as_bytes())
    }

    fn write_generation_source_safetensors(path: &Path, n_experts: usize) {
        let mut entries = Vec::new();
        for expert in 0..n_experts {
            entries.push((
                format!("layer1.expert{expert}.w13"),
                vec![4, 8],
                synthetic_values(4, 8, expert + 10),
            ));
            entries.push((
                format!("layer1.expert{expert}.w2"),
                vec![8, 4],
                synthetic_values(8, 4, expert + 100),
            ));
        }
        write_owned_f32_safetensors(path, &entries);
    }

    fn write_generation_manifest<F>(
        manifest_path: &Path,
        model_dir: &Path,
        config_hash: u64,
        shard_path: &Path,
        cache_path: &Path,
        spec_path: &Path,
        n_experts: usize,
        mutate: F,
    ) where
        F: FnOnce(&mut serde_json::Value),
    {
        let mut tensors = Vec::new();
        for expert in 0..n_experts {
            tensors.push(serde_json::json!({
                "layer_idx": 1,
                "expert_idx": expert,
                "role": "w13",
                "tensor_key": format!("layer1.expert{expert}.w13"),
                "shard_path": shard_path,
                "expected_rows": 4,
                "expected_cols": 8
            }));
            tensors.push(serde_json::json!({
                "layer_idx": 1,
                "expert_idx": expert,
                "role": "w2",
                "tensor_key": format!("layer1.expert{expert}.w2"),
                "shard_path": shard_path,
                "expected_rows": 8,
                "expected_cols": 4
            }));
        }
        let mut manifest = serde_json::json!({
            "purpose": "expert_hqq_cache_generation",
            "model_dir": model_dir,
            "config_hash": format!("0x{config_hash:016x}"),
            "layer_idx": 1,
            "experts": (0..n_experts).collect::<Vec<_>>(),
            "roles": ["w13", "w2"],
            "nbits": 6,
            "group_size": 64,
            "axis": 1,
            "layout": "row_major_axis1_grouped_uint6_packed",
            "output_cache_path": cache_path,
            "diagnostic_spec_path": spec_path,
            "tensors": tensors
        });
        mutate(&mut manifest);
        std::fs::write(
            manifest_path,
            format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap()),
        )
        .unwrap();
    }

    #[test]
    fn expert_hqq_writer_loader_round_trips_hqq4_and_hqq6_w13_w2() {
        let header = sample_header(4);
        let inputs = vec![
            sample_input(ExpertHqqTensorRole::W13, 0, 0, 4),
            sample_input(ExpertHqqTensorRole::W2, 0, 0, 4),
            sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
            sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
        ];
        let path = temp_path("expert_hqq_round_trip");
        let cache = write_expert_hqq_cache_from_inputs(&path, header, inputs).unwrap();
        let loaded = load_expert_hqq_cache(&path, &cache.header.expectation()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded, cache);
        assert_eq!(loaded.tensors[0].descriptor.nbits, 4);
        assert_eq!(loaded.tensors[2].descriptor.nbits, 6);
    }

    #[test]
    fn expert_hqq_safetensors_builder_round_trips_hqq4_and_hqq6_w13_w2() {
        let st_path = temp_path("expert_hqq_builder_source_safetensors");
        write_f32_safetensors(
            &st_path,
            &[
                ("layer0.expert0.w13", vec![4, 8], synthetic_values(4, 8, 1)),
                ("layer0.expert0.w2", vec![8, 4], synthetic_values(8, 4, 2)),
                ("layer1.expert2.w13", vec![4, 8], synthetic_values(4, 8, 3)),
                ("layer1.expert2.w2", vec![8, 4], synthetic_values(8, 4, 4)),
            ],
        );
        let specs = vec![
            ExpertHqqSafetensorsTensorSpec::new(
                &st_path,
                "layer0.expert0.w13",
                ExpertHqqTensorRole::W13,
                0,
                0,
                4,
                4,
            )
            .unwrap(),
            ExpertHqqSafetensorsTensorSpec::new(
                &st_path,
                "layer0.expert0.w2",
                ExpertHqqTensorRole::W2,
                0,
                0,
                4,
                4,
            )
            .unwrap(),
            ExpertHqqSafetensorsTensorSpec::new(
                &st_path,
                "layer1.expert2.w13",
                ExpertHqqTensorRole::W13,
                1,
                2,
                6,
                4,
            )
            .unwrap(),
            ExpertHqqSafetensorsTensorSpec::new(
                &st_path,
                "layer1.expert2.w2",
                ExpertHqqTensorRole::W2,
                1,
                2,
                6,
                4,
            )
            .unwrap(),
        ];
        let cache_path = temp_path("expert_hqq_builder_cache");
        let cache = write_expert_hqq_cache_from_safetensors(
            &cache_path,
            sample_header(specs.len()),
            &specs,
        )
        .unwrap();
        let loaded = load_expert_hqq_cache(&cache_path, &cache.header.expectation()).unwrap();
        std::fs::remove_file(&st_path).unwrap();
        std::fs::remove_file(&cache_path).unwrap();

        assert_eq!(loaded, cache);
        assert_eq!(loaded.tensors.len(), 4);
        let hqq4_w13 = &loaded.tensors[0];
        assert_eq!(hqq4_w13.descriptor.role, ExpertHqqTensorRole::W13);
        assert_eq!(hqq4_w13.descriptor.layer_idx, 0);
        assert_eq!(hqq4_w13.descriptor.expert_idx, 0);
        assert_eq!(hqq4_w13.descriptor.nbits, 4);
        assert_eq!(hqq4_w13.descriptor.packed_bytes, 16);
        assert_eq!(hqq4_w13.descriptor.scales_bytes, 32);
        assert_eq!(hqq4_w13.descriptor.zeros_bytes, 32);

        let hqq6_w2 = &loaded.tensors[3];
        assert_eq!(hqq6_w2.descriptor.role, ExpertHqqTensorRole::W2);
        assert_eq!(hqq6_w2.descriptor.layer_idx, 1);
        assert_eq!(hqq6_w2.descriptor.expert_idx, 2);
        assert_eq!(hqq6_w2.descriptor.nbits, 6);
        assert_eq!(hqq6_w2.descriptor.packed_bytes, 24);
        assert_eq!(hqq6_w2.descriptor.scales_bytes, 32);
        assert_eq!(hqq6_w2.descriptor.zeros_bytes, 32);
        for record in &loaded.tensors {
            assert_eq!(record.packed.len(), record.descriptor.packed_bytes);
            assert_eq!(record.scales.len(), record.descriptor.scales_bytes);
            assert_eq!(record.zeros.len(), record.descriptor.zeros_bytes);
        }
    }

    #[test]
    fn expert_hqq_safetensors_builder_fails_closed_on_key_and_role_shape_mismatch() {
        let st_path = temp_path("expert_hqq_builder_bad_source");
        write_f32_safetensors(
            &st_path,
            &[("layer0.expert0.w13", vec![4, 8], synthetic_values(4, 8, 5))],
        );
        let missing = ExpertHqqSafetensorsTensorSpec::new(
            &st_path,
            "missing.w13",
            ExpertHqqTensorRole::W13,
            0,
            0,
            6,
            4,
        )
        .unwrap();
        let err = build_expert_hqq_cache_from_safetensors(sample_header(1), &[missing])
            .expect_err("missing safetensors key must fail closed");
        assert!(err.contains("Tensor not found"), "{err}");

        let wrong_role = ExpertHqqSafetensorsTensorSpec::new(
            &st_path,
            "layer0.expert0.w13",
            ExpertHqqTensorRole::W2,
            0,
            0,
            6,
            4,
        )
        .unwrap();
        let err = build_expert_hqq_cache_from_safetensors(sample_header(1), &[wrong_role])
            .expect_err("W2 role with W13-shaped tensor must fail closed");
        std::fs::remove_file(&st_path).unwrap();
        assert!(err.contains("W2 cols") || err.contains("W2 rows"), "{err}");
    }

    #[test]
    fn expert_hqq_safetensors_builder_readback_rejects_metadata_mismatch() {
        let st_path = temp_path("expert_hqq_builder_metadata_source");
        write_f32_safetensors(
            &st_path,
            &[("layer0.expert0.w2", vec![8, 4], synthetic_values(8, 4, 6))],
        );
        let spec = ExpertHqqSafetensorsTensorSpec::new(
            &st_path,
            "layer0.expert0.w2",
            ExpertHqqTensorRole::W2,
            0,
            0,
            6,
            4,
        )
        .unwrap();
        let cache_path = temp_path("expert_hqq_builder_metadata_cache");
        let cache = write_expert_hqq_cache_from_safetensors(&cache_path, sample_header(1), &[spec])
            .unwrap();
        let mut expected = cache.header.expectation();
        expected.config_hash ^= 1;
        let err = load_expert_hqq_cache(&cache_path, &expected)
            .expect_err("expected-header mismatch must fail closed");
        std::fs::remove_file(&st_path).unwrap();
        std::fs::remove_file(&cache_path).unwrap();
        assert!(err.contains("config_hash"), "{err}");
    }

    #[test]
    fn expert_hqq_generation_manifest_plans_and_writes_valid_cache() {
        let model_dir = temp_path("expert_hqq_generation_model");
        let config_hash = write_generation_model_config(&model_dir, 3);
        let shard_path = temp_path("expert_hqq_generation_source");
        write_generation_source_safetensors(&shard_path, 3);
        let manifest_path = temp_path("expert_hqq_generation_manifest");
        let cache_path = temp_path("expert_hqq_generation_cache");
        let spec_path = temp_path("expert_hqq_generation_diag_spec");
        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            3,
            |_| {},
        );

        let plan = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(plan.layer_idx, 1);
        assert_eq!(plan.experts, vec![0, 1, 2]);
        assert_eq!(plan.nbits, 6);
        assert_eq!(plan.group_size, 64);
        assert_eq!(plan.specs.len(), 6);
        assert_eq!(plan.required_tensors.len(), 6);

        let report = generate_expert_hqq_cache_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(report.expert_count, 3);
        assert_eq!(report.tensor_records, 6);
        assert!(report.total_payload_bytes > 0);
        assert!(report.cache_file_bytes > 0);
        let cache = load_expert_hqq_cache(&cache_path, &plan.header.expectation()).unwrap();
        assert_eq!(cache.tensors.len(), 6);
        for record in &cache.tensors {
            assert_eq!(record.descriptor.nbits, 6);
            assert_eq!(record.descriptor.group_size, 64);
        }
        let spec = load_expert_hqq_diagnostic_cache_spec(&spec_path).unwrap();
        assert_eq!(spec.required_tensors.len(), 6);
        assert!(spec
            .requirements
            .iter()
            .all(|req| req.nbits == 6 && req.group_size == 64));
    }

    #[test]
    fn expert_hqq_generation_manifest_accepts_hqq6_g32_cache_and_spec() {
        let model_dir = temp_path("expert_hqq_generation_g32_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_g32_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_g32_manifest");
        let cache_path = temp_path("expert_hqq_generation_g32_cache");
        let spec_path = temp_path("expert_hqq_generation_g32_diag_spec");
        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["group_size"] = serde_json::json!(32);
            },
        );

        let plan = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(plan.nbits, 6);
        assert_eq!(plan.group_size, 32);
        assert_eq!(plan.layout, "row_major_axis1_grouped_uint6_packed");
        assert!(plan.specs.iter().all(|spec| spec.group_size == 32));

        let report = generate_expert_hqq_cache_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(report.expert_count, 2);
        assert_eq!(report.tensor_records, 4);
        assert!(report.total_payload_bytes > 0);
        assert!(report.cache_file_bytes > 0);

        let cache = load_expert_hqq_cache(&cache_path, &plan.header.expectation()).unwrap();
        assert_eq!(cache.tensors.len(), 4);
        for record in &cache.tensors {
            assert_eq!(record.descriptor.nbits, 6);
            assert_eq!(record.descriptor.group_size, 32);
            let (packed, scales, zeros) = expert_hqq_component_sizes(
                record.descriptor.rows,
                record.descriptor.cols,
                record.descriptor.nbits,
                record.descriptor.group_size,
            )
            .unwrap();
            assert_eq!(record.descriptor.packed_bytes, packed);
            assert_eq!(record.descriptor.scales_bytes, scales);
            assert_eq!(record.descriptor.zeros_bytes, zeros);
            assert_eq!(record.packed.len(), packed);
            assert_eq!(record.scales.len(), scales);
            assert_eq!(record.zeros.len(), zeros);
        }

        let spec = load_expert_hqq_diagnostic_cache_spec(&spec_path).unwrap();
        assert_eq!(spec.required_tensors.len(), 4);
        assert!(spec
            .requirements
            .iter()
            .all(|req| req.nbits == 6 && req.group_size == 32));
    }

    #[test]
    fn expert_hqq_generation_manifest_accepts_hqq6_g16_cache_and_spec() {
        let model_dir = temp_path("expert_hqq_generation_g16_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_g16_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_g16_manifest");
        let cache_path = temp_path("expert_hqq_generation_g16_cache");
        let spec_path = temp_path("expert_hqq_generation_g16_diag_spec");
        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["group_size"] = serde_json::json!(16);
            },
        );

        let plan = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(plan.nbits, 6);
        assert_eq!(plan.group_size, 16);
        assert_eq!(plan.layout, "row_major_axis1_grouped_uint6_packed");
        assert!(plan.specs.iter().all(|spec| spec.group_size == 16));

        let report = generate_expert_hqq_cache_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(report.expert_count, 2);
        assert_eq!(report.tensor_records, 4);
        assert!(report.total_payload_bytes > 0);
        assert!(report.cache_file_bytes > 0);

        let cache = load_expert_hqq_cache(&cache_path, &plan.header.expectation()).unwrap();
        assert_eq!(cache.tensors.len(), 4);
        for record in &cache.tensors {
            assert_eq!(record.descriptor.nbits, 6);
            assert_eq!(record.descriptor.group_size, 16);
            let (packed, scales, zeros) = expert_hqq_component_sizes(
                record.descriptor.rows,
                record.descriptor.cols,
                record.descriptor.nbits,
                record.descriptor.group_size,
            )
            .unwrap();
            assert_eq!(record.descriptor.packed_bytes, packed);
            assert_eq!(record.descriptor.scales_bytes, scales);
            assert_eq!(record.descriptor.zeros_bytes, zeros);
            assert_eq!(record.packed.len(), packed);
            assert_eq!(record.scales.len(), scales);
            assert_eq!(record.zeros.len(), zeros);
        }

        let spec = load_expert_hqq_diagnostic_cache_spec(&spec_path).unwrap();
        assert_eq!(spec.required_tensors.len(), 4);
        assert!(spec
            .requirements
            .iter()
            .all(|req| req.nbits == 6 && req.group_size == 16));
    }

    #[test]
    fn expert_hqq_generation_manifest_accepts_hqq8_g64_cache_and_spec() {
        let model_dir = temp_path("expert_hqq_generation_hqq8_g64_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_hqq8_g64_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_hqq8_g64_manifest");
        let cache_path = temp_path("expert_hqq_generation_hqq8_g64_cache");
        let spec_path = temp_path("expert_hqq_generation_hqq8_g64_diag_spec");
        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["nbits"] = serde_json::json!(8);
                manifest["group_size"] = serde_json::json!(64);
                manifest["layout"] = serde_json::json!("row_major_axis1_grouped_uint8");
            },
        );

        let plan = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(plan.nbits, 8);
        assert_eq!(plan.group_size, 64);
        assert_eq!(plan.layout, "row_major_axis1_grouped_uint8");
        assert!(plan
            .specs
            .iter()
            .all(|spec| spec.nbits == 8 && spec.group_size == 64));

        let report = generate_expert_hqq_cache_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(report.expert_count, 2);
        assert_eq!(report.tensor_records, 4);
        assert!(report.total_payload_bytes > 0);
        assert!(report.cache_file_bytes > 0);

        let cache = load_expert_hqq_cache(&cache_path, &plan.header.expectation()).unwrap();
        assert_eq!(cache.tensors.len(), 4);
        for record in &cache.tensors {
            assert_eq!(record.descriptor.nbits, 8);
            assert_eq!(record.descriptor.group_size, 64);
            assert_eq!(record.descriptor.layout, "row_major_axis1_grouped_uint8");
            let (packed, scales, zeros) = expert_hqq_component_sizes(
                record.descriptor.rows,
                record.descriptor.cols,
                record.descriptor.nbits,
                record.descriptor.group_size,
            )
            .unwrap();
            assert_eq!(record.descriptor.packed_bytes, packed);
            assert_eq!(record.descriptor.scales_bytes, scales);
            assert_eq!(record.descriptor.zeros_bytes, zeros);
            assert_eq!(record.packed.len(), packed);
            assert_eq!(record.scales.len(), scales);
            assert_eq!(record.zeros.len(), zeros);
        }

        let spec = load_expert_hqq_diagnostic_cache_spec(&spec_path).unwrap();
        assert_eq!(spec.required_tensors.len(), 4);
        assert!(spec
            .requirements
            .iter()
            .all(|req| req.nbits == 8 && req.group_size == 64));
    }

    #[test]
    fn expert_hqq_generation_manifest_rejects_hqq8_non_g64() {
        let model_dir = temp_path("expert_hqq_generation_hqq8_bad_group_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_hqq8_bad_group_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_hqq8_bad_group_manifest");
        let cache_path = temp_path("expert_hqq_generation_hqq8_bad_group_cache");
        let spec_path = temp_path("expert_hqq_generation_hqq8_bad_group_diag_spec");
        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["nbits"] = serde_json::json!(8);
                manifest["group_size"] = serde_json::json!(32);
                manifest["layout"] = serde_json::json!("row_major_axis1_grouped_uint8");
            },
        );

        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("HQQ8 non-g64 generation must fail closed");
        assert!(
            err.contains("requires group_size in [64] for HQQ8"),
            "{err}"
        );
    }

    #[test]
    fn expert_hqq_generation_manifest_plans_and_writes_all_model_moe_layers() {
        let model_dir = temp_path("expert_hqq_generation_all_layers_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_all_layers_source");
        let mut entries = Vec::new();
        for layer_idx in [0usize, 1usize] {
            for expert in 0..2 {
                entries.push((
                    format!("layer{layer_idx}.expert{expert}.w13"),
                    vec![4, 8],
                    synthetic_values(4, 8, layer_idx * 100 + expert + 10),
                ));
                entries.push((
                    format!("layer{layer_idx}.expert{expert}.w2"),
                    vec![8, 4],
                    synthetic_values(8, 4, layer_idx * 100 + expert + 50),
                ));
            }
        }
        write_owned_f32_safetensors(&shard_path, &entries);
        let manifest_path = temp_path("expert_hqq_generation_all_layers_manifest");
        let cache_path = temp_path("expert_hqq_generation_all_layers_cache");
        let spec_path = temp_path("expert_hqq_generation_all_layers_spec");
        let model_dir_json = model_dir.display().to_string();
        let shard_path_json = shard_path.display().to_string();
        let cache_path_json = cache_path.display().to_string();
        let spec_path_json = spec_path.display().to_string();
        let tensors = [0usize, 1usize]
            .into_iter()
            .flat_map(|layer_idx| {
                let shard_path_json = shard_path_json.clone();
                (0..2).flat_map(move |expert| {
                    [
                        serde_json::json!({
                            "layer_idx": layer_idx,
                            "expert_idx": expert,
                            "role": "w13",
                            "tensor_key": format!("layer{layer_idx}.expert{expert}.w13"),
                            "shard_path": shard_path_json.clone(),
                            "expected_rows": 4,
                            "expected_cols": 8
                        }),
                        serde_json::json!({
                            "layer_idx": layer_idx,
                            "expert_idx": expert,
                            "role": "w2",
                            "tensor_key": format!("layer{layer_idx}.expert{expert}.w2"),
                            "shard_path": shard_path_json.clone(),
                            "expected_rows": 8,
                            "expected_cols": 4
                        }),
                    ]
                })
            })
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "purpose": "expert_hqq_cache_generation",
            "model_dir": model_dir_json,
            "config_hash": format!("0x{config_hash:016x}"),
            "layers": [0, 1],
            "experts": [0, 1],
            "roles": ["w13", "w2"],
            "nbits": 6,
            "group_size": 64,
            "axis": 1,
            "layout": "row_major_axis1_grouped_uint6_packed",
            "output_cache_path": cache_path_json,
            "diagnostic_spec_path": spec_path_json,
            "tensors": tensors
        });
        std::fs::write(
            &manifest_path,
            format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap()),
        )
        .unwrap();

        let plan = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(plan.layers, vec![0, 1]);
        assert_eq!(plan.layer_idx, 0);
        assert_eq!(plan.experts, vec![0, 1]);
        assert_eq!(plan.nbits, 6);
        assert_eq!(plan.group_size, 64);
        assert_eq!(plan.specs.len(), 8);
        assert_eq!(plan.required_tensors.len(), 8);

        let report = generate_expert_hqq_cache_from_manifest_path(&manifest_path).unwrap();
        assert_eq!(report.layers, vec![0, 1]);
        assert_eq!(report.expert_count, 2);
        assert_eq!(report.tensor_records, 8);
        let spec = load_expert_hqq_diagnostic_cache_spec(&spec_path).unwrap();
        assert_eq!(spec.requirements.len(), 4);
        assert_eq!(spec.required_tensors.len(), 8);
        assert!(spec
            .requirements
            .iter()
            .all(|req| req.nbits == 6 && req.group_size == 64));
    }

    #[test]
    fn expert_hqq_generation_manifest_rejects_malformed_missing_source_and_hash() {
        let malformed = temp_path("expert_hqq_generation_malformed_manifest");
        std::fs::write(&malformed, "{").unwrap();
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&malformed)
            .expect_err("malformed manifest must fail closed");
        assert!(
            err.contains("malformed expert-HQQ cache generation manifest"),
            "{err}"
        );

        let model_dir = temp_path("expert_hqq_generation_bad_hash_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_bad_hash_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_bad_hash_manifest");
        let cache_path = temp_path("expert_hqq_generation_bad_hash_cache");
        let spec_path = temp_path("expert_hqq_generation_bad_hash_spec");
        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash ^ 1,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |_| {},
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("wrong config hash must fail closed");
        assert!(err.contains("config_hash mismatch"), "{err}");

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["tensors"][0]["shard_path"] =
                    serde_json::json!("/definitely/missing/expert_hqq_source.safetensors");
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("missing shard must fail closed");
        assert!(
            err.contains("failed to resolve expert-HQQ safetensors shard_path"),
            "{err}"
        );

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["tensors"][0]["tensor_key"] = serde_json::json!("layer1.expert0.missing");
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("missing tensor key must fail closed");
        assert!(err.contains("Tensor not found"), "{err}");
    }

    #[test]
    fn expert_hqq_generation_manifest_rejects_partial_pairs_and_metadata_mismatch() {
        let model_dir = temp_path("expert_hqq_generation_metadata_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_metadata_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_metadata_manifest");
        let cache_path = temp_path("expert_hqq_generation_metadata_cache");
        let spec_path = temp_path("expert_hqq_generation_metadata_spec");

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["tensors"].as_array_mut().unwrap().pop();
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("partial W13/W2 pairs must fail closed");
        assert!(err.contains("complete W13/W2 pairs"), "{err}");

        for (field, value, expected) in [
            ("nbits", serde_json::json!(4), "requires nbits in [6, 8]"),
            ("nbits", serde_json::json!(5), "requires nbits in [6, 8]"),
            (
                "group_size",
                serde_json::json!(0),
                "requires group_size in [16, 32, 64]",
            ),
            (
                "group_size",
                serde_json::json!(8),
                "requires group_size in [16, 32, 64]",
            ),
            ("axis", serde_json::json!(0), "requires axis=1"),
            (
                "layout",
                serde_json::json!("row_major_axis1_grouped_uint4_packed"),
                "layout",
            ),
        ] {
            write_generation_manifest(
                &manifest_path,
                &model_dir,
                config_hash,
                &shard_path,
                &cache_path,
                &spec_path,
                2,
                |manifest| {
                    manifest[field] = value.clone();
                },
            );
            let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
                .expect_err("metadata mismatch must fail closed");
            assert!(err.contains(expected), "{field}: {err}");
        }

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["tensors"][0]["expected_rows"] = serde_json::json!(5);
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("shape metadata mismatch must fail closed");
        assert!(err.contains("shape metadata mismatch"), "{err}");
    }

    #[derive(Debug, Deserialize)]
    struct RealCalibFile {
        samples: Vec<RealCalibSample>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct RealCalibSample {
        layer_idx: usize,
        expert_idx: usize,
        proj_name: String,
        row_idx: usize,
        active_cols: Vec<usize>,
        active_vals: Vec<f32>,
    }

    fn read_tsv(path: &Path) -> Vec<HashMap<String, String>> {
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read TSV {}: {e}", path.display()));
        let mut lines = raw.lines();
        let header: Vec<String> = lines
            .next()
            .unwrap_or_else(|| panic!("empty TSV {}", path.display()))
            .split('\t')
            .map(|s| s.to_string())
            .collect();
        lines
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let mut row = HashMap::new();
                for (key, value) in header.iter().zip(line.split('\t')) {
                    row.insert(key.clone(), value.to_string());
                }
                row
            })
            .collect()
    }

    fn tsv_get<'a>(row: &'a HashMap<String, String>, key: &str) -> &'a str {
        row.get(key)
            .unwrap_or_else(|| panic!("missing TSV column {key}"))
            .as_str()
    }

    fn parse_usize(row: &HashMap<String, String>, key: &str) -> usize {
        tsv_get(row, key)
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("failed to parse {key}: {e}"))
    }

    fn parse_f64(row: &HashMap<String, String>, key: &str) -> f64 {
        tsv_get(row, key)
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("failed to parse {key}: {e}"))
    }

    fn config_usize(config: &serde_json::Value, key: &str) -> usize {
        config
            .get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("missing integer config key {key}")) as usize
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        hash
    }

    fn shard_path_for_key(model_dir: &Path, index: &serde_json::Value, key: &str) -> PathBuf {
        let shard = index
            .get("weight_map")
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("missing safetensors shard for key {key}"));
        model_dir.join(shard)
    }

    fn f32_bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn dequant_record_to_f32(record: &ExpertHqqTensorRecord) -> Vec<f32> {
        let desc = &record.descriptor;
        let groups = group_count(desc.cols, desc.group_size).unwrap();
        let scales = f32_bytes_to_vec(&record.scales);
        let zeros = f32_bytes_to_vec(&record.zeros);
        assert_eq!(scales.len(), desc.rows * groups);
        assert_eq!(zeros.len(), desc.rows * groups);
        let padded = padded_cols(desc.cols, desc.group_size).unwrap();
        let mut out = vec![0.0f32; desc.rows * desc.cols];
        for row in 0..desc.rows {
            for col in 0..desc.cols {
                let group = col / desc.group_size;
                let q = match desc.nbits {
                    4 => {
                        let packed_cols = padded.div_ceil(2);
                        let byte = record.packed[row * packed_cols + col / 2];
                        if col % 2 == 0 {
                            byte & 0x0f
                        } else {
                            (byte >> 4) & 0x0f
                        }
                    }
                    6 => {
                        let block = col / 4;
                        let rem = col % 4;
                        let packed_cols = padded.div_ceil(4) * 3;
                        let base = row * packed_cols + block * 3;
                        let a = record.packed[base];
                        let b = record.packed[base + 1];
                        let c = record.packed[base + 2];
                        match rem {
                            0 => a & 0x3f,
                            1 => ((a >> 6) | ((b & 0x0f) << 2)) & 0x3f,
                            2 => ((b >> 4) | ((c & 0x03) << 4)) & 0x3f,
                            _ => (c >> 2) & 0x3f,
                        }
                    }
                    other => panic!("unsupported proof nbits {other}"),
                };
                let meta = row * groups + group;
                out[row * desc.cols + col] = (q as f32 - zeros[meta]) * scales[meta];
            }
        }
        out
    }

    fn bf16_bits_to_f32(bits: u64) -> f32 {
        bf16_to_f32(bits as u16)
    }

    fn case_ids() -> [&'static str; 3] {
        [
            "moby_dick_ch1",
            "war_and_peace_opening",
            "les_miserables_bishop",
        ]
    }

    fn w13_samples_by_case(
        samples_path: &Path,
        row_table: &[HashMap<String, String>],
    ) -> HashMap<(String, usize, usize), RealCalibSample> {
        let raw = std::fs::read(samples_path)
            .unwrap_or_else(|e| panic!("failed to read samples {}: {e}", samples_path.display()));
        let parsed: RealCalibFile = serde_json::from_slice(&raw)
            .unwrap_or_else(|e| panic!("failed to parse samples {}: {e}", samples_path.display()));
        let mut cases_by_expert_row: BTreeMap<(usize, usize), Vec<String>> = BTreeMap::new();
        for row in row_table {
            let key = (parse_usize(row, "expert"), parse_usize(row, "row"));
            let case_id = tsv_get(row, "case_id").to_string();
            let cases = cases_by_expert_row.entry(key).or_default();
            if cases.last() != Some(&case_id) {
                cases.push(case_id);
            }
        }
        let needed: BTreeSet<(usize, usize)> = cases_by_expert_row.keys().copied().collect();
        let mut grouped: BTreeMap<(usize, usize), Vec<RealCalibSample>> = BTreeMap::new();
        for sample in parsed.samples {
            if sample.layer_idx == 1
                && sample.proj_name == "up_proj"
                && needed.contains(&(sample.expert_idx, sample.row_idx))
            {
                grouped
                    .entry((sample.expert_idx, sample.row_idx))
                    .or_default()
                    .push(sample);
            }
        }
        let mut by_case = HashMap::new();
        for ((expert, row), entries) in grouped {
            let mut unique: Vec<RealCalibSample> = Vec::new();
            for sample in entries {
                if unique.iter().any(|seen| {
                    seen.active_cols == sample.active_cols && seen.active_vals == sample.active_vals
                }) {
                    continue;
                }
                unique.push(sample);
            }
            let cases = cases_by_expert_row
                .get(&(expert, row))
                .unwrap_or_else(|| panic!("missing case map for expert {expert} row {row}"));
            assert_eq!(
                unique.len(),
                cases.len(),
                "expected {} prompt cases for expert {expert} row {row}",
                cases.len()
            );
            for (case_id, sample) in cases.iter().zip(unique.into_iter()) {
                by_case.insert((case_id.clone(), expert, row), sample);
            }
        }
        by_case
    }

    fn extract_w2_input_vectors(trace_path: &Path) -> HashMap<(String, usize, usize), Vec<f32>> {
        let raw = std::fs::read_to_string(trace_path)
            .unwrap_or_else(|e| panic!("failed to read W2 trace {}: {e}", trace_path.display()));
        let parsed: serde_json::Value = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse W2 trace {}: {e}", trace_path.display()));
        let results = parsed
            .get("results")
            .and_then(|v| v.as_array())
            .expect("W2 trace missing results array");
        let mut out = HashMap::new();
        for (case_idx, result) in results.iter().enumerate() {
            let case_id = case_ids()[case_idx].to_string();
            let snapshots = result["response"]["debug_reference_trace"]["prefill_stage_trace"]
                ["prefill_stage_snapshots"]
                .as_array()
                .expect("W2 trace missing snapshots");
            for snap in snapshots {
                let stage = snap.get("stage").and_then(|v| v.as_str()).unwrap_or("");
                let Some(suffix) = stage
                    .strip_prefix("layer1_sequential_moe_bf16_activation_w2_input_full_expert")
                else {
                    continue;
                };
                let Some(expert_str) = suffix.strip_suffix("_rows") else {
                    continue;
                };
                let expert = expert_str
                    .parse::<usize>()
                    .unwrap_or_else(|e| panic!("bad expert in stage {stage}: {e}"));
                let metadata = snap
                    .get("metadata")
                    .expect("W2 full trace missing metadata");
                let row_start = metadata["absolute_row_start"]
                    .as_u64()
                    .expect("W2 full trace missing absolute_row_start")
                    as usize;
                let row_width = metadata["row_width"]
                    .as_u64()
                    .expect("W2 full trace missing row_width")
                    as usize;
                let row_count = metadata["row_count"]
                    .as_u64()
                    .expect("W2 full trace missing row_count")
                    as usize;
                let bits = snap["trace"]["bf16_bits_u16"]
                    .as_array()
                    .expect("W2 full trace missing bf16_bits_u16");
                assert_eq!(bits.len(), row_count * row_width);
                for local_row in 0..row_count {
                    let sorted_row = row_start + local_row;
                    let mut vector = Vec::with_capacity(row_width);
                    for col in 0..row_width {
                        let raw_bits = bits[local_row * row_width + col]
                            .as_u64()
                            .expect("W2 bf16 bit entry must be integer");
                        vector.push(bf16_bits_to_f32(raw_bits));
                    }
                    out.insert((case_id.clone(), expert, sorted_row), vector);
                }
            }
        }
        out
    }

    fn extract_bf16_full_vectors(
        trace_path: &Path,
        stage_prefix: &str,
        label: &str,
    ) -> HashMap<(String, usize, usize), Vec<f32>> {
        let raw = std::fs::read_to_string(trace_path).unwrap_or_else(|e| {
            panic!("failed to read {label} trace {}: {e}", trace_path.display())
        });
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
            panic!(
                "failed to parse {label} trace {}: {e}",
                trace_path.display()
            )
        });
        let results = parsed
            .get("results")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("{label} trace missing results array"));
        let mut out = HashMap::new();
        for (case_idx, result) in results.iter().enumerate() {
            let case_id = case_ids()[case_idx].to_string();
            let snapshots = result["response"]["debug_reference_trace"]["prefill_stage_trace"]
                ["prefill_stage_snapshots"]
                .as_array()
                .unwrap_or_else(|| panic!("{label} trace missing snapshots"));
            for snap in snapshots {
                let stage = snap.get("stage").and_then(|v| v.as_str()).unwrap_or("");
                let Some(suffix) = stage.strip_prefix(stage_prefix) else {
                    continue;
                };
                let Some(expert_str) = suffix.strip_suffix("_rows") else {
                    continue;
                };
                let expert = expert_str
                    .parse::<usize>()
                    .unwrap_or_else(|e| panic!("bad expert in stage {stage}: {e}"));
                let metadata = snap
                    .get("metadata")
                    .unwrap_or_else(|| panic!("{label} trace {stage} missing metadata"));
                let row_start = metadata["absolute_row_start"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{label} trace {stage} missing absolute_row_start"))
                    as usize;
                let row_width = metadata["row_width"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{label} trace {stage} missing row_width"))
                    as usize;
                let row_count = metadata["row_count"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{label} trace {stage} missing row_count"))
                    as usize;
                let bits = snap["trace"]["bf16_bits_u16"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{label} trace {stage} missing bf16_bits_u16"));
                assert_eq!(bits.len(), row_count * row_width);
                for local_row in 0..row_count {
                    let sorted_row = row_start + local_row;
                    let mut vector = Vec::with_capacity(row_width);
                    for col in 0..row_width {
                        let raw_bits =
                            bits[local_row * row_width + col]
                                .as_u64()
                                .unwrap_or_else(|| {
                                    panic!("{label} trace {stage} bf16 bit entry must be integer")
                                });
                        vector.push(bf16_bits_to_f32(raw_bits));
                    }
                    let previous = out.insert((case_id.clone(), expert, sorted_row), vector);
                    assert!(
                        previous.is_none(),
                        "duplicate {label} vector for case={case_id} expert={expert} sorted_row={sorted_row}"
                    );
                }
            }
        }
        out
    }

    fn round_to_bf16_f32(value: f32) -> f32 {
        half::bf16::from_f32(value).to_f32()
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct StageDelta {
        sum_abs: f64,
        max_abs: f64,
        l2_sq: f64,
        count: usize,
    }

    impl StageDelta {
        fn add(&mut self, got: f32, expected: f32) {
            let delta = (got - expected).abs() as f64;
            self.sum_abs += delta;
            self.max_abs = self.max_abs.max(delta);
            self.l2_sq += delta * delta;
            self.count += 1;
        }

        fn add_slices(&mut self, got: &[f32], expected: &[f32]) {
            assert_eq!(got.len(), expected.len());
            for (&g, &e) in got.iter().zip(expected.iter()) {
                self.add(g, e);
            }
        }

        fn l2(self) -> f64 {
            self.l2_sq.sqrt()
        }
    }

    #[test]
    fn expert_hqq_generation_manifest_rejects_ambiguous_or_incomplete_layer_set() {
        let model_dir = temp_path("expert_hqq_generation_bad_layers_model");
        let config_hash = write_generation_model_config(&model_dir, 2);
        let shard_path = temp_path("expert_hqq_generation_bad_layers_source");
        write_generation_source_safetensors(&shard_path, 2);
        let manifest_path = temp_path("expert_hqq_generation_bad_layers_manifest");
        let cache_path = temp_path("expert_hqq_generation_bad_layers_cache");
        let spec_path = temp_path("expert_hqq_generation_bad_layers_spec");

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["layers"] = serde_json::json!([0, 1]);
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("manifest cannot specify both layer_idx and layers");
        assert!(err.contains("exactly one of layer_idx or layers"), "{err}");

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest.as_object_mut().unwrap().remove("layer_idx");
                manifest["layers"] = serde_json::json!([1]);
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("all-layer manifest must cover every model MoE layer");
        assert!(err.contains("exactly match model MoE layers"), "{err}");

        write_generation_manifest(
            &manifest_path,
            &model_dir,
            config_hash,
            &shard_path,
            &cache_path,
            &spec_path,
            2,
            |manifest| {
                manifest["layer_idx"] = serde_json::json!(99);
            },
        );
        let err = plan_expert_hqq_cache_generation_from_manifest_path(&manifest_path)
            .expect_err("out-of-range layer must fail closed");
        assert!(err.contains("out of range"), "{err}");
    }

    fn first_nonzero_stage(stages: &[(&str, f64)]) -> String {
        stages
            .iter()
            .find(|(_, max_abs)| *max_abs > 0.0)
            .map(|(stage, _)| (*stage).to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    fn first_stage_over_tolerance(stages: &[(&str, f64, f64)]) -> String {
        stages
            .iter()
            .find(|(_, value, tolerance)| *value > *tolerance)
            .map(|(stage, _, _)| (*stage).to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    fn find_record<'a>(
        cache: &'a ExpertHqqCache,
        role: ExpertHqqTensorRole,
        expert: usize,
    ) -> &'a ExpertHqqTensorRecord {
        cache
            .tensors
            .iter()
            .find(|record| {
                record.descriptor.role == role
                    && record.descriptor.layer_idx == 1
                    && record.descriptor.expert_idx == expert
            })
            .unwrap_or_else(|| panic!("missing {:?} expert {expert} record", role))
    }

    fn dot_sparse(row_weights: &[f32], cols: &[usize], vals: &[f32]) -> f64 {
        cols.iter()
            .zip(vals.iter())
            .map(|(&col, &value)| row_weights[col] as f64 * value as f64)
            .sum()
    }

    fn dot_dense(row_weights: &[f32], values: &[f32]) -> f32 {
        row_weights
            .iter()
            .zip(values.iter())
            .map(|(&w, &v)| w * v)
            .sum()
    }

    fn write_lines(path: &Path, lines: &[String]) {
        let mut file = File::create(path)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
        for line in lines {
            writeln!(file, "{line}")
                .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
        }
    }

    #[test]
    fn real_nemotron_selected_expert_hqq6_g64_cache_readback_replays_prior_fidelity() {
        if env::var("KRASIS_REAL_NANO_KRHQ_PROOF").ok().as_deref() != Some("1") {
            eprintln!("skipping real Nano KRHQ proof; set KRASIS_REAL_NANO_KRHQ_PROOF=1");
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real KRHQ proof"),
        );
        let artifact_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR")
                .expect("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real KRHQ proof"),
        );
        std::fs::create_dir_all(&artifact_dir).unwrap();
        let label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let cache_path = artifact_dir.join(format!("{label}_hqq6_g64.krhq"));

        let w13_rows = read_tsv(&artifact_dir.join(
            "20260626_0629_nemotron_nano_offline_expert_hqq_w13_fidelity_proof_w13_offline_hqq_row_compare.tsv",
        ));
        let w2_rows = read_tsv(&artifact_dir.join(
            "20260626_0641_nemotron_nano_offline_expert_hqq_w2_fidelity_proof_w2_offline_hqq_slot_compare.tsv",
        ));
        let w2_hqq6_rows: Vec<_> = w2_rows
            .iter()
            .filter(|row| tsv_get(row, "variant") == "hqq6_g64")
            .collect();
        let w13_samples = w13_samples_by_case(
            &artifact_dir.join(
                "20260626_0535_nemotron_nano_int4_w13_quantization_fidelity_mitigation_expert_int4_calib_samples.json",
            ),
            &w13_rows,
        );
        let w2_inputs = extract_w2_input_vectors(&artifact_dir.join(
            "20260626_0641_nemotron_nano_offline_expert_hqq_w2_fidelity_proof_bf16_w2_input_trace_outputs.json",
        ));

        let mut experts = BTreeSet::new();
        for row in &w13_rows {
            experts.insert(parse_usize(row, "expert"));
        }
        for row in &w2_hqq6_rows {
            experts.insert(parse_usize(row, "expert"));
        }

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let config_hash = fnv1a64(&config_bytes);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();

        let mut specs = Vec::new();
        for &expert in &experts {
            let w13_key = format!("backbone.layers.1.mixer.experts.{expert}.up_proj.weight");
            let w2_key = format!("backbone.layers.1.mixer.experts.{expert}.down_proj.weight");
            specs.push(
                ExpertHqqSafetensorsTensorSpec::new(
                    shard_path_for_key(&model_dir, &index, &w13_key),
                    &w13_key,
                    ExpertHqqTensorRole::W13,
                    1,
                    expert,
                    6,
                    64,
                )
                .unwrap(),
            );
            specs.push(
                ExpertHqqSafetensorsTensorSpec::new(
                    shard_path_for_key(&model_dir, &index, &w2_key),
                    &w2_key,
                    ExpertHqqTensorRole::W2,
                    1,
                    expert,
                    6,
                    64,
                )
                .unwrap(),
            );
        }
        let header = ExpertHqqCacheHeader::new(
            hidden_size,
            hidden_size,
            intermediate_size,
            n_routed_experts,
            num_layers,
            config_hash,
            specs.len(),
        )
        .unwrap();
        let cache = write_expert_hqq_cache_from_safetensors(&cache_path, header, &specs).unwrap();
        let loaded = load_expert_hqq_cache(&cache_path, &cache.header.expectation()).unwrap();
        assert_eq!(loaded, cache);
        assert_eq!(loaded.tensors.len(), experts.len() * 2);

        let mut contract = vec![
            "role\tlayer\texpert\tnbits\tgroup_size\trows\tcols\tpacked_bytes\tscales_bytes\tzeros_bytes\tpayload_lengths_match".to_string(),
        ];
        for record in &loaded.tensors {
            let desc = &record.descriptor;
            let (packed, scales, zeros) =
                expert_hqq_component_sizes(desc.rows, desc.cols, desc.nbits, desc.group_size)
                    .unwrap();
            assert_eq!(packed, desc.packed_bytes);
            assert_eq!(scales, desc.scales_bytes);
            assert_eq!(zeros, desc.zeros_bytes);
            assert_eq!(record.packed.len(), desc.packed_bytes);
            assert_eq!(record.scales.len(), desc.scales_bytes);
            assert_eq!(record.zeros.len(), desc.zeros_bytes);
            assert_eq!(desc.nbits, 6);
            assert_eq!(desc.group_size, 64);
            match desc.role {
                ExpertHqqTensorRole::W13 => {
                    assert_eq!(desc.rows, intermediate_size);
                    assert_eq!(desc.cols, hidden_size);
                }
                ExpertHqqTensorRole::W2 => {
                    assert_eq!(desc.rows, hidden_size);
                    assert_eq!(desc.cols, intermediate_size);
                }
            }
            contract.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\ttrue",
                desc.role.as_str(),
                desc.layer_idx,
                desc.expert_idx,
                desc.nbits,
                desc.group_size,
                desc.rows,
                desc.cols,
                desc.packed_bytes,
                desc.scales_bytes,
                desc.zeros_bytes
            ));
        }
        write_lines(
            &artifact_dir.join(format!("{label}_real_cache_descriptor_readback.tsv")),
            &contract,
        );

        let mut dequant: HashMap<(ExpertHqqTensorRole, usize), Vec<f32>> = HashMap::new();
        for &expert in &experts {
            dequant.insert(
                (ExpertHqqTensorRole::W13, expert),
                dequant_record_to_f32(find_record(&loaded, ExpertHqqTensorRole::W13, expert)),
            );
            dequant.insert(
                (ExpertHqqTensorRole::W2, expert),
                dequant_record_to_f32(find_record(&loaded, ExpertHqqTensorRole::W2, expert)),
            );
        }

        let mut bf16_w13_cache: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut w13_lines = vec![
            "case_id\texpert\trow\tprior_hqq6_g64_abs_err\treadback_hqq6_g64_abs_err\tabs_err_delta\tprior_hqq6_g64_dot\treadback_hqq6_g64_dot".to_string(),
        ];
        let mut w13_prior_abs = 0.0f64;
        let mut w13_readback_abs = 0.0f64;
        let mut w13_amax_abs = 0.0f64;
        for row in &w13_rows {
            let case_id = tsv_get(row, "case_id").to_string();
            let expert = parse_usize(row, "expert");
            let out_row = parse_usize(row, "row");
            let sample = w13_samples
                .get(&(case_id.clone(), expert, out_row))
                .unwrap_or_else(|| {
                    panic!("missing W13 sample for case={case_id} expert={expert} row={out_row}")
                });
            let bf16 = bf16_w13_cache.entry(expert).or_insert_with(|| {
                let key = format!("backbone.layers.1.mixer.experts.{expert}.up_proj.weight");
                let path = shard_path_for_key(&model_dir, &index, &key);
                let (weights, rows, cols) = load_safetensors_tensor_2d_f32(&path, &key).unwrap();
                assert_eq!(rows, intermediate_size);
                assert_eq!(cols, hidden_size);
                weights
            });
            let hqq = dequant.get(&(ExpertHqqTensorRole::W13, expert)).unwrap();
            let bf16_dot = dot_sparse(
                &bf16[out_row * hidden_size..(out_row + 1) * hidden_size],
                &sample.active_cols,
                &sample.active_vals,
            );
            let readback_dot = dot_sparse(
                &hqq[out_row * hidden_size..(out_row + 1) * hidden_size],
                &sample.active_cols,
                &sample.active_vals,
            );
            let readback_abs = (readback_dot - bf16_dot).abs();
            let prior_abs = parse_f64(row, "hqq6_g64_abs_err");
            let prior_dot = parse_f64(row, "hqq6_g64_dot_top_inputs");
            w13_prior_abs += prior_abs;
            w13_readback_abs += readback_abs;
            w13_amax_abs += parse_f64(row, "amax_abs_err");
            w13_lines.push(format!(
                "{case_id}\t{expert}\t{out_row}\t{prior_abs:.12}\t{readback_abs:.12}\t{:.12}\t{prior_dot:.12}\t{readback_dot:.12}",
                (readback_abs - prior_abs).abs()
            ));
        }
        write_lines(
            &artifact_dir.join(format!("{label}_w13_readback_replay.tsv")),
            &w13_lines,
        );

        let mut bf16_w2_cache: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut w2_lines = vec![
            "case_id\texpert\tsorted_row\ttopk_weight\tprior_hqq6_g64_sum_abs\treadback_hqq6_g64_sum_abs\tabs_err_delta".to_string(),
        ];
        let mut w2_prior_abs = 0.0f64;
        let mut w2_readback_abs = 0.0f64;
        let mut w2_amax_abs = 0.0f64;
        for row in &w2_hqq6_rows {
            let case_id = tsv_get(row, "case_id").to_string();
            let expert = parse_usize(row, "expert");
            let sorted_row = parse_usize(row, "sorted_row");
            let topk_weight = parse_f64(row, "topk_weight");
            let input = w2_inputs
                .get(&(case_id.clone(), expert, sorted_row))
                .unwrap_or_else(|| {
                    panic!(
                        "missing W2 input for case={case_id} expert={expert} sorted_row={sorted_row}"
                    )
                });
            assert_eq!(input.len(), intermediate_size);
            let bf16 = bf16_w2_cache.entry(expert).or_insert_with(|| {
                let key = format!("backbone.layers.1.mixer.experts.{expert}.down_proj.weight");
                let path = shard_path_for_key(&model_dir, &index, &key);
                let (weights, rows, cols) = load_safetensors_tensor_2d_f32(&path, &key).unwrap();
                assert_eq!(rows, hidden_size);
                assert_eq!(cols, intermediate_size);
                weights
            });
            let hqq = dequant.get(&(ExpertHqqTensorRole::W2, expert)).unwrap();
            let mut readback_abs = 0.0f64;
            for out_row in 0..hidden_size {
                let start = out_row * intermediate_size;
                let end = start + intermediate_size;
                let bf16_dot = dot_dense(&bf16[start..end], input);
                let hqq_dot = dot_dense(&hqq[start..end], input);
                readback_abs += (hqq_dot - bf16_dot).abs() as f64;
            }
            let prior_abs = parse_f64(row, "sum_abs");
            w2_prior_abs += prior_abs;
            w2_readback_abs += readback_abs;
            w2_lines.push(format!(
                "{case_id}\t{expert}\t{sorted_row}\t{topk_weight:.12}\t{prior_abs:.12}\t{readback_abs:.12}\t{:.12}",
                (readback_abs - prior_abs).abs()
            ));
        }
        for row in w2_rows
            .iter()
            .filter(|row| tsv_get(row, "variant") == "amax_g64")
        {
            w2_amax_abs += parse_f64(row, "sum_abs");
        }
        write_lines(
            &artifact_dir.join(format!("{label}_w2_readback_replay.tsv")),
            &w2_lines,
        );

        let w13_delta = (w13_readback_abs - w13_prior_abs).abs();
        let w2_delta = (w2_readback_abs - w2_prior_abs).abs();
        assert!(
            w13_delta < 1e-4,
            "W13 readback replay diverged from prior proof: prior={w13_prior_abs} readback={w13_readback_abs} delta={w13_delta}"
        );
        assert!(
            w2_delta < 1e-3,
            "W2 readback replay diverged from prior proof: prior={w2_prior_abs} readback={w2_readback_abs} delta={w2_delta}"
        );
        let metadata = std::fs::metadata(&cache_path).unwrap();
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("cache_bytes\t{}", metadata.len()),
            format!("selected_experts\t{:?}", experts),
            format!("tensor_records\t{}", loaded.tensors.len()),
            format!("nbits\t6"),
            format!("group_size\t64"),
            format!("w13_rows\t{}", w13_rows.len()),
            format!("w13_prior_hqq6_g64_abs_err_sum\t{w13_prior_abs:.12}"),
            format!("w13_readback_hqq6_g64_abs_err_sum\t{w13_readback_abs:.12}"),
            format!("w13_readback_delta_vs_prior\t{w13_delta:.12}"),
            format!(
                "w13_readback_over_amax\t{:.12}",
                w13_readback_abs / w13_amax_abs
            ),
            format!("w2_slots\t{}", w2_hqq6_rows.len()),
            format!("w2_prior_hqq6_g64_sum_abs\t{w2_prior_abs:.12}"),
            format!("w2_readback_hqq6_g64_sum_abs\t{w2_readback_abs:.12}"),
            format!("w2_readback_delta_vs_prior\t{w2_delta:.12}"),
            format!(
                "w2_readback_over_amax\t{:.12}",
                w2_readback_abs / w2_amax_abs
            ),
            "runtime_config_added\tfalse".to_string(),
            "model_load_integration_added\tfalse".to_string(),
            "prefill_dispatch_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "speed_work\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_readback_replay_summary.tsv")),
            &summary_lines,
        );
    }

    #[test]
    fn real_nemotron_selected_expert_hqq6_g64_reference_executor_matches_readback_proof() {
        if env::var("KRASIS_REAL_NANO_KRHQ_REFERENCE_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano KRHQ reference proof; set KRASIS_REAL_NANO_KRHQ_REFERENCE_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real KRHQ reference proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real KRHQ reference proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();
        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let label = "20260626_1004_nemotron_nano_real_expert_hqq_reference_execution_validation";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        let selected_rows = read_tsv(&artifact_dir.join(
            "20260626_0353_nemotron_nano_int4_prefill_layer1_branch_moe_output_producer_selected_expert_slot_compare.tsv",
        ));
        let w13_rows = read_tsv(&artifact_dir.join(
            "20260626_0629_nemotron_nano_offline_expert_hqq_w13_fidelity_proof_w13_offline_hqq_row_compare.tsv",
        ));
        let w2_rows = read_tsv(&artifact_dir.join(
            "20260626_0641_nemotron_nano_offline_expert_hqq_w2_fidelity_proof_w2_offline_hqq_slot_compare.tsv",
        ));
        let w2_hqq6_rows: Vec<_> = w2_rows
            .iter()
            .filter(|row| tsv_get(row, "variant") == "hqq6_g64")
            .collect();
        let w13_samples = w13_samples_by_case(
            &artifact_dir.join(
                "20260626_0535_nemotron_nano_int4_w13_quantization_fidelity_mitigation_expert_int4_calib_samples.json",
            ),
            &w13_rows,
        );
        let w2_inputs = extract_w2_input_vectors(&artifact_dir.join(
            "20260626_0641_nemotron_nano_offline_expert_hqq_w2_fidelity_proof_bf16_w2_input_trace_outputs.json",
        ));

        let mut descriptor_lines = vec![
            "case_id\tslot\ttopk_pos\texpert\tcompact_sorted_row\tplan_row_offset\tplan_row_count\tw13_nbits\tw2_nbits\tw13_group_size\tw2_group_size\tw13_rows\tw13_cols\tw2_rows\tw2_cols".to_string(),
        ];
        let mut w13_lines = vec![
            "case_id\texpert\tslot\trow\tbf16_sparse_dot\tprior_hqq6_g64_dot\texecutor_hqq6_g64_dot\tdot_delta\tprior_hqq6_g64_abs_err\texecutor_hqq6_g64_abs_err\tabs_err_delta\tactivation\texpected_relu2\tactivation_delta".to_string(),
        ];
        let mut w2_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tprior_hqq6_g64_sum_abs\treadback_hqq6_g64_sum_abs\tsum_abs_delta\texecutor_projection_self_max_abs_delta".to_string(),
        ];

        let mut w2_dequant: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut bf16_w2_cache: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut total_plan_entries = 0usize;
        let mut total_w13_rows = 0usize;
        let mut w13_prior_abs = 0.0f64;
        let mut w13_executor_abs = 0.0f64;
        let mut w13_prior_dot_delta_max = 0.0f64;
        let mut activation_delta_max = 0.0f64;
        let mut w2_prior_abs = 0.0f64;
        let mut w2_readback_abs = 0.0f64;
        let mut w2_delta_max = 0.0f64;
        let mut executor_projection_self_delta_max = 0.0f64;

        for case_id in case_ids() {
            let mut case_selected: Vec<_> = selected_rows
                .iter()
                .filter(|row| {
                    tsv_get(row, "case_id") == case_id
                        && tsv_get(row, "component") == "routed_input"
                })
                .collect();
            case_selected.sort_by_key(|row| parse_usize(row, "topk_pos"));
            assert_eq!(
                case_selected.len(),
                6,
                "expected six selected experts for {case_id}"
            );

            let mut works = Vec::with_capacity(case_selected.len());
            let mut inputs = vec![0.0f32; case_selected.len() * hidden_size];
            let mut slot_by_expert = HashMap::new();
            let mut sorted_row_by_expert = HashMap::new();
            for (slot, row) in case_selected.iter().enumerate() {
                let topk_pos = parse_usize(row, "topk_pos");
                assert_eq!(slot, topk_pos);
                let expert = parse_usize(row, "expert");
                let sorted_row = parse_usize(row, "sorted_row");
                works.push(ExpertHqqPrefillWork::new(expert, slot, 1));
                slot_by_expert.insert(expert, slot);
                sorted_row_by_expert.insert(expert, sorted_row);

                let sample = w13_samples
                    .get(&(case_id.to_string(), expert, 0))
                    .unwrap_or_else(|| {
                        panic!("missing W13 sparse input sample for case={case_id} expert={expert}")
                    });
                for (&col, &value) in sample.active_cols.iter().zip(sample.active_vals.iter()) {
                    inputs[slot * hidden_size + col] = value;
                }
            }

            let plan = cache.prefill_dispatch_plan(1, false, &works).unwrap();
            assert_eq!(plan.entries.len(), case_selected.len());
            let output = cache
                .execute_prefill_reference(&plan, &inputs, case_selected.len())
                .unwrap();
            assert_eq!(output.sorted_row_count, case_selected.len());
            assert_eq!(output.routed_hidden_size, hidden_size);
            assert_eq!(output.w13_rows, intermediate_size);
            assert_eq!(output.moe_intermediate_size, intermediate_size);
            assert_eq!(output.values.len(), case_selected.len() * hidden_size);
            assert_eq!(
                output.w13_preactivation.len(),
                case_selected.len() * intermediate_size
            );
            assert_eq!(
                output.activation.len(),
                case_selected.len() * intermediate_size
            );

            for entry in &plan.entries {
                let sorted_row = sorted_row_by_expert
                    .get(&entry.expert_idx)
                    .copied()
                    .unwrap();
                descriptor_lines.push(format!(
                    "{case_id}\t{}\t{}\t{}\t{sorted_row}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    entry.row_offset,
                    entry.row_offset,
                    entry.expert_idx,
                    entry.row_offset,
                    entry.row_count,
                    entry.w13_nbits,
                    entry.w2_nbits,
                    entry.w13_group_size,
                    entry.w2_group_size,
                    entry.w13_rows,
                    entry.w13_cols,
                    entry.w2_rows,
                    entry.w2_cols
                ));
                assert_eq!(entry.w13_nbits, 6);
                assert_eq!(entry.w2_nbits, 6);
                assert_eq!(entry.w13_group_size, 64);
                assert_eq!(entry.w2_group_size, 64);
            }
            total_plan_entries += plan.entries.len();

            for row in w13_rows
                .iter()
                .filter(|row| tsv_get(row, "case_id") == case_id)
            {
                let expert = parse_usize(row, "expert");
                let out_row = parse_usize(row, "row");
                let slot = *slot_by_expert.get(&expert).unwrap_or_else(|| {
                    panic!("missing selected slot for case={case_id} expert={expert}")
                });
                let got = output.w13_preactivation[slot * output.w13_rows + out_row] as f64;
                let prior_dot = parse_f64(row, "hqq6_g64_dot_top_inputs");
                let bf16_dot = parse_f64(row, "bf16_dot_top_inputs");
                let prior_abs = parse_f64(row, "hqq6_g64_abs_err");
                let got_abs = (got - bf16_dot).abs();
                let dot_delta = (got - prior_dot).abs();
                let expected_activation = (got as f32).max(0.0).powi(2);
                let activation = output.activation[slot * intermediate_size + out_row];
                let activation_delta = (activation - expected_activation).abs() as f64;
                w13_prior_dot_delta_max = w13_prior_dot_delta_max.max(dot_delta);
                activation_delta_max = activation_delta_max.max(activation_delta);
                w13_prior_abs += prior_abs;
                w13_executor_abs += got_abs;
                total_w13_rows += 1;
                w13_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{out_row}\t{bf16_dot:.12}\t{prior_dot:.12}\t{got:.12}\t{dot_delta:.12}\t{prior_abs:.12}\t{got_abs:.12}\t{:.12}\t{activation:.12}\t{expected_activation:.12}\t{activation_delta:.12}",
                    (got_abs - prior_abs).abs()
                ));
            }

            for row in w2_hqq6_rows
                .iter()
                .filter(|row| tsv_get(row, "case_id") == case_id)
            {
                let expert = parse_usize(row, "expert");
                let sorted_row = parse_usize(row, "sorted_row");
                let slot = *slot_by_expert.get(&expert).unwrap_or_else(|| {
                    panic!("missing selected slot for case={case_id} expert={expert}")
                });
                assert_eq!(
                    sorted_row,
                    *sorted_row_by_expert.get(&expert).unwrap(),
                    "W2 proof sorted row should match selected-slot artifact"
                );
                let input = w2_inputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!(
                            "missing W2 input for case={case_id} expert={expert} sorted_row={sorted_row}"
                        )
                    });
                assert_eq!(input.len(), intermediate_size);
                let hqq = w2_dequant.entry(expert).or_insert_with(|| {
                    dequantize_expert_hqq_record_to_f32(find_record(
                        &cache,
                        ExpertHqqTensorRole::W2,
                        expert,
                    ))
                    .unwrap()
                });
                let bf16 = bf16_w2_cache.entry(expert).or_insert_with(|| {
                    let key = format!("backbone.layers.1.mixer.experts.{expert}.down_proj.weight");
                    let path = shard_path_for_key(&model_dir, &index, &key);
                    let (weights, rows, cols) =
                        load_safetensors_tensor_2d_f32(&path, &key).unwrap();
                    assert_eq!(rows, hidden_size);
                    assert_eq!(cols, intermediate_size);
                    weights
                });
                let mut readback_abs = 0.0f64;
                let mut self_delta = 0.0f64;
                let executor_activation =
                    &output.activation[slot * intermediate_size..(slot + 1) * intermediate_size];
                let executor_values = &output.values[slot * hidden_size..(slot + 1) * hidden_size];
                for out_row in 0..hidden_size {
                    let start = out_row * intermediate_size;
                    let end = start + intermediate_size;
                    let bf16_dot = dot_dense(&bf16[start..end], input);
                    let hqq_dot = dot_dense(&hqq[start..end], input);
                    readback_abs += (hqq_dot - bf16_dot).abs() as f64;
                    let manual_executor_dot = dot_dense(&hqq[start..end], executor_activation);
                    self_delta = self_delta
                        .max((manual_executor_dot - executor_values[out_row]).abs() as f64);
                }
                let prior_abs = parse_f64(row, "sum_abs");
                let delta = (readback_abs - prior_abs).abs();
                w2_prior_abs += prior_abs;
                w2_readback_abs += readback_abs;
                w2_delta_max = w2_delta_max.max(delta);
                executor_projection_self_delta_max =
                    executor_projection_self_delta_max.max(self_delta);
                w2_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{prior_abs:.12}\t{readback_abs:.12}\t{delta:.12}\t{self_delta:.12}"
                ));
            }
        }

        let w13_abs_delta = (w13_executor_abs - w13_prior_abs).abs();
        let w2_abs_delta = (w2_readback_abs - w2_prior_abs).abs();
        assert_eq!(total_plan_entries, 18);
        assert_eq!(total_w13_rows, 108);
        assert!(
            w13_prior_dot_delta_max < 1e-5,
            "W13 dot delta max {w13_prior_dot_delta_max}"
        );
        assert!(w13_abs_delta < 1e-4, "W13 abs sum delta {w13_abs_delta}");
        assert!(
            activation_delta_max < 1e-6,
            "activation relu2 delta max {activation_delta_max}"
        );
        assert!(w2_abs_delta < 1e-3, "W2 abs sum delta {w2_abs_delta}");
        assert!(
            executor_projection_self_delta_max < 1e-6,
            "executor W2 projection self delta max {executor_projection_self_delta_max}"
        );

        write_lines(
            &artifact_dir.join(format!("{label}_descriptor_plan_validation.tsv")),
            &descriptor_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_w13_reference_executor_replay.tsv")),
            &w13_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_w2_reference_executor_replay.tsv")),
            &w2_lines,
        );
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            "cache_source_gate\t0836".to_string(),
            "executor_source_gate\t0938".to_string(),
            "selected_cases\t3".to_string(),
            format!("plan_entries\t{total_plan_entries}"),
            format!("tensor_records\t{}", cache.tensors.len()),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            format!("w13_rows\t{total_w13_rows}"),
            format!("w13_prior_hqq6_g64_abs_err_sum\t{w13_prior_abs:.12}"),
            format!("w13_executor_hqq6_g64_abs_err_sum\t{w13_executor_abs:.12}"),
            format!("w13_executor_delta_vs_prior\t{w13_abs_delta:.12}"),
            format!("w13_prior_dot_delta_max\t{w13_prior_dot_delta_max:.12}"),
            format!("activation_relu2_delta_max\t{activation_delta_max:.12}"),
            format!("w2_slots\t{}", w2_hqq6_rows.len()),
            format!("w2_prior_hqq6_g64_sum_abs\t{w2_prior_abs:.12}"),
            format!("w2_readback_hqq6_g64_sum_abs\t{w2_readback_abs:.12}"),
            format!("w2_readback_delta_vs_prior\t{w2_abs_delta:.12}"),
            format!("w2_slot_delta_max\t{w2_delta_max:.12}"),
            format!("executor_projection_self_delta_max\t{executor_projection_self_delta_max:.12}"),
            "runtime_config_added\tfalse".to_string(),
            "gpu_prefill_dispatch_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_work\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_reference_execution_summary.tsv")),
            &summary_lines,
        );
    }

    #[test]
    fn real_nemotron_full_routed_input_hqq_branch_replay_matches_bf16_trace() {
        if env::var("KRASIS_REAL_NANO_KRHQ_BRANCH_REPLAY_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano full KRHQ branch replay proof; set KRASIS_REAL_NANO_KRHQ_BRANCH_REPLAY_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real KRHQ branch replay proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real KRHQ branch replay proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();
        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path =
            artifact_dir.join(format!("{label}_bf16_branch_replay_trace_outputs.json"));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();

        let selected_rows = read_tsv(&artifact_dir.join(
            "20260626_0353_nemotron_nano_int4_prefill_layer1_branch_moe_output_producer_selected_expert_slot_compare.tsv",
        ));
        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "branch output",
        );
        let mut slot_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tinput_hash\tbf16_branch_sum_abs_cpu_f32\tbf16_branch_sum_abs_cpu_bf16rounded\tbf16_branch_max_abs_cpu_bf16rounded\thqq_branch_sum_abs_vs_bf16\thqq_branch_max_abs_vs_bf16\thqq_branch_l2_vs_bf16\tbf16_closer_than_hqq".to_string(),
        ];
        let mut summary_lines = vec!["metric\tvalue".to_string()];
        let mut bf16_w13_cache: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut bf16_w2_cache: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut total_slots = 0usize;
        let mut total_values = 0usize;
        let mut bf16_cpu_sum_abs = 0.0f64;
        let mut bf16_cpu_rounded_sum_abs = 0.0f64;
        let mut bf16_cpu_rounded_max_abs = 0.0f64;
        let mut hqq_sum_abs = 0.0f64;
        let mut hqq_max_abs = 0.0f64;
        let mut hqq_l2_sq = 0.0f64;
        let mut captured_routed_slot_vectors = 0usize;
        let mut captured_branch_slot_vectors = 0usize;
        let mut slots_where_bf16_not_closer = 0usize;
        let mut first_bf16_not_closer = String::new();

        for case_id in case_ids() {
            let mut case_selected: Vec<_> = selected_rows
                .iter()
                .filter(|row| {
                    tsv_get(row, "case_id") == case_id
                        && tsv_get(row, "component") == "routed_input"
                })
                .collect();
            case_selected.sort_by_key(|row| parse_usize(row, "topk_pos"));
            assert_eq!(
                case_selected.len(),
                6,
                "expected six selected experts for {case_id}"
            );

            let mut works = Vec::with_capacity(case_selected.len());
            let mut inputs = vec![0.0f32; case_selected.len() * hidden_size];
            let mut slot_by_expert = HashMap::new();
            let mut sorted_row_by_expert = HashMap::new();
            for (slot, row) in case_selected.iter().enumerate() {
                let topk_pos = parse_usize(row, "topk_pos");
                assert_eq!(slot, topk_pos);
                let expert = parse_usize(row, "expert");
                let sorted_row = parse_usize(row, "sorted_row");
                works.push(ExpertHqqPrefillWork::new(expert, slot, 1));
                slot_by_expert.insert(expert, slot);
                sorted_row_by_expert.insert(expert, sorted_row);
                let input = routed_inputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing routed input for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_routed_slot_vectors += 1;
                assert_eq!(input.len(), hidden_size);
                inputs[slot * hidden_size..(slot + 1) * hidden_size].copy_from_slice(input);
            }

            let plan = cache.prefill_dispatch_plan(1, false, &works).unwrap();
            let output = cache
                .execute_prefill_reference(&plan, &inputs, case_selected.len())
                .unwrap();
            assert_eq!(output.values.len(), case_selected.len() * hidden_size);
            assert_eq!(
                output.activation.len(),
                case_selected.len() * intermediate_size
            );

            for entry in &plan.entries {
                let expert = entry.expert_idx;
                let slot = *slot_by_expert.get(&expert).unwrap();
                let sorted_row = *sorted_row_by_expert.get(&expert).unwrap();
                let input = &inputs[slot * hidden_size..(slot + 1) * hidden_size];
                let captured_branch = branch_outputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing branch output for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_branch_slot_vectors += 1;
                assert_eq!(captured_branch.len(), hidden_size);

                let w13 = bf16_w13_cache.entry(expert).or_insert_with(|| {
                    let key = format!("backbone.layers.1.mixer.experts.{expert}.up_proj.weight");
                    let path = shard_path_for_key(&model_dir, &index, &key);
                    let (weights, rows, cols) =
                        load_safetensors_tensor_2d_f32(&path, &key).unwrap();
                    assert_eq!(rows, intermediate_size);
                    assert_eq!(cols, hidden_size);
                    weights
                });
                let w2 = bf16_w2_cache.entry(expert).or_insert_with(|| {
                    let key = format!("backbone.layers.1.mixer.experts.{expert}.down_proj.weight");
                    let path = shard_path_for_key(&model_dir, &index, &key);
                    let (weights, rows, cols) =
                        load_safetensors_tensor_2d_f32(&path, &key).unwrap();
                    assert_eq!(rows, hidden_size);
                    assert_eq!(cols, intermediate_size);
                    weights
                });

                let mut bf16_activation = vec![0.0f32; intermediate_size];
                for row in 0..intermediate_size {
                    let preact = round_to_bf16_f32(dot_dense(
                        &w13[row * hidden_size..(row + 1) * hidden_size],
                        input,
                    ));
                    let relu = preact.max(0.0);
                    bf16_activation[row] = round_to_bf16_f32(relu * relu);
                }

                let hqq_branch = &output.values[slot * hidden_size..(slot + 1) * hidden_size];
                let mut slot_bf16_sum_abs = 0.0f64;
                let mut slot_bf16_rounded_sum_abs = 0.0f64;
                let mut slot_bf16_rounded_max_abs = 0.0f64;
                let mut slot_hqq_sum_abs = 0.0f64;
                let mut slot_hqq_max_abs = 0.0f64;
                let mut slot_hqq_l2_sq = 0.0f64;
                for out_row in 0..hidden_size {
                    let start = out_row * intermediate_size;
                    let end = start + intermediate_size;
                    let bf16_cpu = dot_dense(&w2[start..end], &bf16_activation);
                    let bf16_cpu_rounded = round_to_bf16_f32(bf16_cpu);
                    let captured = captured_branch[out_row];
                    let bf16_delta = (bf16_cpu - captured).abs() as f64;
                    let rounded_delta = (bf16_cpu_rounded - captured).abs() as f64;
                    let hqq_delta = (hqq_branch[out_row] - captured).abs() as f64;
                    slot_bf16_sum_abs += bf16_delta;
                    slot_bf16_rounded_sum_abs += rounded_delta;
                    slot_bf16_rounded_max_abs = slot_bf16_rounded_max_abs.max(rounded_delta);
                    slot_hqq_sum_abs += hqq_delta;
                    slot_hqq_max_abs = slot_hqq_max_abs.max(hqq_delta);
                    slot_hqq_l2_sq += hqq_delta * hqq_delta;
                }

                let bf16_closer_than_hqq = slot_bf16_rounded_sum_abs < slot_hqq_sum_abs;
                if !bf16_closer_than_hqq {
                    slots_where_bf16_not_closer += 1;
                    if first_bf16_not_closer.is_empty() {
                        first_bf16_not_closer = format!(
                            "case={case_id} expert={expert} bf16_sum={slot_bf16_rounded_sum_abs:.12} hqq_sum={slot_hqq_sum_abs:.12}"
                        );
                    }
                }
                total_slots += 1;
                total_values += hidden_size;
                bf16_cpu_sum_abs += slot_bf16_sum_abs;
                bf16_cpu_rounded_sum_abs += slot_bf16_rounded_sum_abs;
                bf16_cpu_rounded_max_abs = bf16_cpu_rounded_max_abs.max(slot_bf16_rounded_max_abs);
                hqq_sum_abs += slot_hqq_sum_abs;
                hqq_max_abs = hqq_max_abs.max(slot_hqq_max_abs);
                hqq_l2_sq += slot_hqq_l2_sq;
                let input_hash = {
                    let mut bytes = Vec::with_capacity(input.len() * 4);
                    for &value in input {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    format!("0x{:016x}", fnv1a64(&bytes))
                };
                slot_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{input_hash}\t{slot_bf16_sum_abs:.12}\t{slot_bf16_rounded_sum_abs:.12}\t{slot_bf16_rounded_max_abs:.12}\t{slot_hqq_sum_abs:.12}\t{slot_hqq_max_abs:.12}\t{:.12}\t{bf16_closer_than_hqq}",
                    slot_hqq_l2_sq.sqrt()
                ));
            }
        }

        assert_eq!(total_slots, 18);
        assert_eq!(captured_routed_slot_vectors, 18);
        assert_eq!(captured_branch_slot_vectors, 18);
        assert_eq!(total_values, 18 * hidden_size);
        summary_lines.extend([
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "selected_cases\t3".to_string(),
            format!("slots\t{total_slots}"),
            format!("values\t{total_values}"),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            format!("captured_routed_slot_vectors\t{captured_routed_slot_vectors}"),
            format!("captured_branch_slot_vectors\t{captured_branch_slot_vectors}"),
            format!("bf16_cpu_f32_vs_captured_sum_abs\t{bf16_cpu_sum_abs:.12}"),
            format!("bf16_cpu_bf16rounded_vs_captured_sum_abs\t{bf16_cpu_rounded_sum_abs:.12}"),
            format!("bf16_cpu_bf16rounded_vs_captured_max_abs\t{bf16_cpu_rounded_max_abs:.12}"),
            format!("hqq6_g64_branch_vs_bf16_captured_sum_abs\t{hqq_sum_abs:.12}"),
            format!("hqq6_g64_branch_vs_bf16_captured_max_abs\t{hqq_max_abs:.12}"),
            format!(
                "hqq6_g64_branch_vs_bf16_captured_l2\t{:.12}",
                hqq_l2_sq.sqrt()
            ),
            format!(
                "bf16_cpu_bf16rounded_closer_than_hqq_slots\t{}",
                total_slots - slots_where_bf16_not_closer
            ),
            format!(
                "bf16_cpu_bf16rounded_not_closer_than_hqq_slots\t{slots_where_bf16_not_closer}"
            ),
            format!(
                "bf16_cpu_bf16rounded_total_closer_than_hqq\t{}",
                bf16_cpu_rounded_sum_abs < hqq_sum_abs
            ),
            format!("first_bf16_not_closer\t{first_bf16_not_closer}"),
            "runtime_config_added\tfalse".to_string(),
            "gpu_prefill_dispatch_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_work\tfalse".to_string(),
        ]);
        write_lines(
            &artifact_dir.join(format!("{label}_full_branch_slot_compare.tsv")),
            &slot_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_full_branch_replay_summary.tsv")),
            &summary_lines,
        );
        assert_eq!(
            slots_where_bf16_not_closer, 0,
            "BF16 replay should be closer than HQQ for every slot; first failure: {first_bf16_not_closer}"
        );
        assert!(
            bf16_cpu_rounded_sum_abs < hqq_sum_abs,
            "BF16 replay should be closer to captured branch than HQQ: bf16_sum={bf16_cpu_rounded_sum_abs} hqq_sum={hqq_sum_abs}"
        );
    }

    #[test]
    fn real_nemotron_test_dispatch_replays_full_branch_capture() {
        if env::var("KRASIS_REAL_NANO_KRHQ_TEST_DISPATCH_REPLAY_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano KRHQ test-dispatch replay proof; set KRASIS_REAL_NANO_KRHQ_TEST_DISPATCH_REPLAY_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real KRHQ test-dispatch proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real KRHQ test-dispatch proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();

        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let trace_label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let label = "20260626_1217_nemotron_nano_real_krhq_test_only_dispatch_replay_validation";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path = artifact_dir.join(format!(
            "{trace_label}_bf16_branch_replay_trace_outputs.json"
        ));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let selected_rows = read_tsv(&artifact_dir.join(
            "20260626_0353_nemotron_nano_int4_prefill_layer1_branch_moe_output_producer_selected_expert_slot_compare.tsv",
        ));
        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "branch output",
        );
        let prior_summary =
            read_tsv(&artifact_dir.join(format!("{trace_label}_full_branch_replay_summary.tsv")));
        let prior_metric = |name: &str| -> f64 {
            prior_summary
                .iter()
                .find(|row| tsv_get(row, "metric") == name)
                .unwrap_or_else(|| panic!("missing prior summary metric {name}"))
                .get("value")
                .unwrap()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("failed to parse prior metric {name}: {e}"))
        };

        let mut descriptor_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tplan_row_offset\tplan_row_count\tw13_role\tw2_role\tw13_nbits\tw2_nbits\tw13_group_size\tw2_group_size\tw13_layout\tw2_layout\tw13_rows\tw13_cols\tw2_rows\tw2_cols".to_string(),
        ];
        let mut slot_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tinput_hash\tdispatch_reference_sum_abs\tdispatch_reference_max_abs\tdispatch_branch_sum_abs_vs_bf16\tdispatch_branch_max_abs_vs_bf16\tdispatch_branch_l2_vs_bf16".to_string(),
        ];

        let mut total_slots = 0usize;
        let mut total_values = 0usize;
        let mut captured_routed_slot_vectors = 0usize;
        let mut captured_branch_slot_vectors = 0usize;
        let mut dispatch_reference_sum_abs = 0.0f64;
        let mut dispatch_reference_max_abs = 0.0f64;
        let mut dispatch_branch_sum_abs = 0.0f64;
        let mut dispatch_branch_max_abs = 0.0f64;
        let mut dispatch_branch_l2_sq = 0.0f64;

        for case_id in case_ids() {
            let mut case_selected: Vec<_> = selected_rows
                .iter()
                .filter(|row| {
                    tsv_get(row, "case_id") == case_id
                        && tsv_get(row, "component") == "routed_input"
                })
                .collect();
            case_selected.sort_by_key(|row| parse_usize(row, "topk_pos"));
            assert_eq!(
                case_selected.len(),
                6,
                "expected six selected experts for {case_id}"
            );

            let mut works = Vec::with_capacity(case_selected.len());
            let mut inputs = vec![0.0f32; case_selected.len() * hidden_size];
            let mut slot_by_expert = HashMap::new();
            let mut sorted_row_by_expert = HashMap::new();
            for (slot, row) in case_selected.iter().enumerate() {
                let topk_pos = parse_usize(row, "topk_pos");
                assert_eq!(slot, topk_pos);
                let expert = parse_usize(row, "expert");
                let sorted_row = parse_usize(row, "sorted_row");
                works.push(ExpertHqqPrefillWork::new(expert, slot, 1));
                slot_by_expert.insert(expert, slot);
                sorted_row_by_expert.insert(expert, sorted_row);
                let input = routed_inputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing routed input for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_routed_slot_vectors += 1;
                assert_eq!(input.len(), hidden_size);
                inputs[slot * hidden_size..(slot + 1) * hidden_size].copy_from_slice(input);
            }

            let plan = cache.prefill_dispatch_plan(1, false, &works).unwrap();
            assert_eq!(plan.entries.len(), case_selected.len());
            assert_eq!(plan.layer_idx, 1);
            assert!(!plan.experts_gated);
            assert_eq!(
                plan.input_layout,
                "row_major_selected_rows_by_routed_hidden"
            );
            let reference = cache
                .execute_prefill_reference(&plan, &inputs, case_selected.len())
                .unwrap();
            let dispatch = cache
                .execute_prefill_test_dispatch(&plan, &inputs, case_selected.len())
                .unwrap();
            assert_eq!(dispatch.sorted_row_count, reference.sorted_row_count);
            assert_eq!(dispatch.routed_hidden_size, reference.routed_hidden_size);
            assert_eq!(dispatch.values.len(), reference.values.len());

            for (got, expected) in dispatch.values.iter().zip(reference.values.iter()) {
                let delta = (*got - *expected).abs() as f64;
                dispatch_reference_sum_abs += delta;
                dispatch_reference_max_abs = dispatch_reference_max_abs.max(delta);
            }

            for entry in &plan.entries {
                let expert = entry.expert_idx;
                let slot = *slot_by_expert.get(&expert).unwrap();
                let sorted_row = *sorted_row_by_expert.get(&expert).unwrap();
                assert_eq!(entry.row_offset, slot);
                assert_eq!(entry.row_count, 1);
                assert_eq!(entry.w13_key.role, ExpertHqqTensorRole::W13);
                assert_eq!(entry.w2_key.role, ExpertHqqTensorRole::W2);
                assert_eq!(entry.w13_key.layer_idx, 1);
                assert_eq!(entry.w2_key.layer_idx, 1);
                assert_eq!(entry.w13_key.expert_idx, expert);
                assert_eq!(entry.w2_key.expert_idx, expert);
                assert_eq!(entry.w13_nbits, 6);
                assert_eq!(entry.w2_nbits, 6);
                assert_eq!(entry.w13_group_size, 64);
                assert_eq!(entry.w2_group_size, 64);
                assert_eq!(entry.w13_rows, intermediate_size);
                assert_eq!(entry.w13_cols, hidden_size);
                assert_eq!(entry.w2_rows, hidden_size);
                assert_eq!(entry.w2_cols, intermediate_size);
                let w13 = cache.require_tensor_record(entry.w13_key).unwrap();
                let w2 = cache.require_tensor_record(entry.w2_key).unwrap();
                assert_eq!(
                    w13.descriptor.layout,
                    "row_major_axis1_grouped_uint6_packed"
                );
                assert_eq!(w2.descriptor.layout, "row_major_axis1_grouped_uint6_packed");
                descriptor_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    entry.row_offset,
                    entry.row_count,
                    entry.w13_key.role.as_str(),
                    entry.w2_key.role.as_str(),
                    entry.w13_nbits,
                    entry.w2_nbits,
                    entry.w13_group_size,
                    entry.w2_group_size,
                    w13.descriptor.layout,
                    w2.descriptor.layout,
                    entry.w13_rows,
                    entry.w13_cols,
                    entry.w2_rows,
                    entry.w2_cols
                ));

                let captured_branch = branch_outputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing branch output for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_branch_slot_vectors += 1;
                assert_eq!(captured_branch.len(), hidden_size);

                let dispatch_branch =
                    &dispatch.values[slot * hidden_size..(slot + 1) * hidden_size];
                let reference_branch =
                    &reference.values[slot * hidden_size..(slot + 1) * hidden_size];
                let mut slot_reference_sum_abs = 0.0f64;
                let mut slot_reference_max_abs = 0.0f64;
                let mut slot_branch_sum_abs = 0.0f64;
                let mut slot_branch_max_abs = 0.0f64;
                let mut slot_branch_l2_sq = 0.0f64;
                for out_row in 0..hidden_size {
                    let reference_delta =
                        (dispatch_branch[out_row] - reference_branch[out_row]).abs() as f64;
                    let branch_delta =
                        (dispatch_branch[out_row] - captured_branch[out_row]).abs() as f64;
                    slot_reference_sum_abs += reference_delta;
                    slot_reference_max_abs = slot_reference_max_abs.max(reference_delta);
                    slot_branch_sum_abs += branch_delta;
                    slot_branch_max_abs = slot_branch_max_abs.max(branch_delta);
                    slot_branch_l2_sq += branch_delta * branch_delta;
                }
                dispatch_branch_sum_abs += slot_branch_sum_abs;
                dispatch_branch_max_abs = dispatch_branch_max_abs.max(slot_branch_max_abs);
                dispatch_branch_l2_sq += slot_branch_l2_sq;
                total_slots += 1;
                total_values += hidden_size;
                let input = &inputs[slot * hidden_size..(slot + 1) * hidden_size];
                let input_hash = {
                    let mut bytes = Vec::with_capacity(input.len() * 4);
                    for &value in input {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    format!("0x{:016x}", fnv1a64(&bytes))
                };
                slot_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{input_hash}\t{slot_reference_sum_abs:.12}\t{slot_reference_max_abs:.12}\t{slot_branch_sum_abs:.12}\t{slot_branch_max_abs:.12}\t{:.12}",
                    slot_branch_l2_sq.sqrt()
                ));
            }
        }

        assert_eq!(total_slots, 18);
        assert_eq!(total_values, 18 * hidden_size);
        assert_eq!(captured_routed_slot_vectors, 18);
        assert_eq!(captured_branch_slot_vectors, 18);
        assert!(
            dispatch_reference_max_abs < 1e-9,
            "test dispatch diverged from offline reference executor: max_delta={dispatch_reference_max_abs}"
        );
        assert!(
            dispatch_reference_sum_abs < 1e-6,
            "test dispatch diverged from offline reference executor: sum_delta={dispatch_reference_sum_abs}"
        );
        let prior_sum = prior_metric("hqq6_g64_branch_vs_bf16_captured_sum_abs");
        let prior_max = prior_metric("hqq6_g64_branch_vs_bf16_captured_max_abs");
        let prior_l2 = prior_metric("hqq6_g64_branch_vs_bf16_captured_l2");
        let dispatch_l2 = dispatch_branch_l2_sq.sqrt();
        assert!(
            (dispatch_branch_sum_abs - prior_sum).abs() < 1e-6,
            "dispatch branch sum {dispatch_branch_sum_abs} diverged from prior HQQ branch sum {prior_sum}"
        );
        assert!(
            (dispatch_branch_max_abs - prior_max).abs() < 1e-9,
            "dispatch branch max {dispatch_branch_max_abs} diverged from prior HQQ branch max {prior_max}"
        );
        assert!(
            (dispatch_l2 - prior_l2).abs() < 1e-9,
            "dispatch branch l2 {dispatch_l2} diverged from prior HQQ branch l2 {prior_l2}"
        );

        write_lines(
            &artifact_dir.join(format!("{label}_descriptor_plan_validation.tsv")),
            &descriptor_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_dispatch_slot_compare.tsv")),
            &slot_lines,
        );
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "cache_source_gate\t0836".to_string(),
            "capture_source_gate\t1023".to_string(),
            "dispatch_consumer_source_gate\t1153".to_string(),
            "selected_cases\t3".to_string(),
            format!("plan_entries\t{total_slots}"),
            format!("tensor_records\t{}", cache.tensors.len()),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            format!("captured_routed_slot_vectors\t{captured_routed_slot_vectors}"),
            format!("captured_branch_slot_vectors\t{captured_branch_slot_vectors}"),
            format!("values\t{total_values}"),
            format!("dispatch_reference_sum_abs\t{dispatch_reference_sum_abs:.12}"),
            format!("dispatch_reference_max_abs\t{dispatch_reference_max_abs:.12}"),
            format!("dispatch_branch_vs_bf16_captured_sum_abs\t{dispatch_branch_sum_abs:.12}"),
            format!("dispatch_branch_vs_bf16_captured_max_abs\t{dispatch_branch_max_abs:.12}"),
            format!("dispatch_branch_vs_bf16_captured_l2\t{dispatch_l2:.12}"),
            format!("prior_1023_hqq6_g64_branch_sum_abs\t{prior_sum:.12}"),
            format!("prior_1023_hqq6_g64_branch_max_abs\t{prior_max:.12}"),
            format!("prior_1023_hqq6_g64_branch_l2\t{prior_l2:.12}"),
            format!(
                "dispatch_sum_delta_vs_1023\t{:.12}",
                (dispatch_branch_sum_abs - prior_sum).abs()
            ),
            format!(
                "dispatch_max_delta_vs_1023\t{:.12}",
                (dispatch_branch_max_abs - prior_max).abs()
            ),
            format!(
                "dispatch_l2_delta_vs_1023\t{:.12}",
                (dispatch_l2 - prior_l2).abs()
            ),
            "runtime_config_added\tfalse".to_string(),
            "auto_selection_added\tfalse".to_string(),
            "gpu_prefill_runtime_consumer_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_work\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_dispatch_replay_summary.tsv")),
            &summary_lines,
        );
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn real_nemotron_gpu_prototype_replays_full_branch_capture() {
        if env::var("KRASIS_REAL_NANO_KRHQ_GPU_PROTOTYPE_REPLAY_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano KRHQ GPU prototype replay proof; set KRASIS_REAL_NANO_KRHQ_GPU_PROTOTYPE_REPLAY_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real KRHQ GPU prototype proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real KRHQ GPU prototype proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();

        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let trace_label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let cpu_label =
            "20260626_1217_nemotron_nano_real_krhq_test_only_dispatch_replay_validation";
        let label = "20260626_1259_nemotron_nano_real_gpu_prototype_replay_validation";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path = artifact_dir.join(format!(
            "{trace_label}_bf16_branch_replay_trace_outputs.json"
        ));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let selected_rows = read_tsv(&artifact_dir.join(
            "20260626_0353_nemotron_nano_int4_prefill_layer1_branch_moe_output_producer_selected_expert_slot_compare.tsv",
        ));
        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "branch output",
        );
        let prior_summary =
            read_tsv(&artifact_dir.join(format!("{trace_label}_full_branch_replay_summary.tsv")));
        let cpu_summary =
            read_tsv(&artifact_dir.join(format!("{cpu_label}_dispatch_replay_summary.tsv")));
        let metric = |rows: &[HashMap<String, String>], name: &str| -> f64 {
            rows.iter()
                .find(|row| tsv_get(row, "metric") == name)
                .unwrap_or_else(|| panic!("missing summary metric {name}"))
                .get("value")
                .unwrap()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("failed to parse summary metric {name}: {e}"))
        };

        let mut descriptor_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tplan_row_offset\tplan_row_count\tw13_role\tw2_role\tw13_nbits\tw2_nbits\tw13_group_size\tw2_group_size\tw13_layout\tw2_layout\tw13_rows\tw13_cols\tw2_rows\tw2_cols".to_string(),
        ];
        let mut slot_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tinput_hash\tgpu_cpu_dispatch_sum_abs\tgpu_cpu_dispatch_max_abs\tgpu_reference_sum_abs\tgpu_reference_max_abs\tgpu_cpu_final_bf16rounded_sum_abs\tgpu_cpu_final_bf16rounded_max_abs\tgpu_branch_sum_abs_vs_bf16\tgpu_branch_max_abs_vs_bf16\tgpu_branch_l2_vs_bf16".to_string(),
        ];

        let mut total_slots = 0usize;
        let mut total_values = 0usize;
        let mut captured_routed_slot_vectors = 0usize;
        let mut captured_branch_slot_vectors = 0usize;
        let mut gpu_cpu_sum_abs = 0.0f64;
        let mut gpu_cpu_max_abs = 0.0f64;
        let mut gpu_reference_sum_abs = 0.0f64;
        let mut gpu_reference_max_abs = 0.0f64;
        let mut gpu_cpu_final_bf16rounded_sum_abs = 0.0f64;
        let mut gpu_cpu_final_bf16rounded_max_abs = 0.0f64;
        let mut gpu_branch_sum_abs = 0.0f64;
        let mut gpu_branch_max_abs = 0.0f64;
        let mut gpu_branch_l2_sq = 0.0f64;

        for case_id in case_ids() {
            let mut case_selected: Vec<_> = selected_rows
                .iter()
                .filter(|row| {
                    tsv_get(row, "case_id") == case_id
                        && tsv_get(row, "component") == "routed_input"
                })
                .collect();
            case_selected.sort_by_key(|row| parse_usize(row, "topk_pos"));
            assert_eq!(
                case_selected.len(),
                6,
                "expected six selected experts for {case_id}"
            );

            let mut works = Vec::with_capacity(case_selected.len());
            let mut inputs = vec![0.0f32; case_selected.len() * hidden_size];
            let mut slot_by_expert = HashMap::new();
            let mut sorted_row_by_expert = HashMap::new();
            for (slot, row) in case_selected.iter().enumerate() {
                let topk_pos = parse_usize(row, "topk_pos");
                assert_eq!(slot, topk_pos);
                let expert = parse_usize(row, "expert");
                let sorted_row = parse_usize(row, "sorted_row");
                works.push(ExpertHqqPrefillWork::new(expert, slot, 1));
                slot_by_expert.insert(expert, slot);
                sorted_row_by_expert.insert(expert, sorted_row);
                let input = routed_inputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing routed input for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_routed_slot_vectors += 1;
                assert_eq!(input.len(), hidden_size);
                inputs[slot * hidden_size..(slot + 1) * hidden_size].copy_from_slice(input);
            }

            let plan = cache.prefill_dispatch_plan(1, false, &works).unwrap();
            assert_eq!(plan.entries.len(), case_selected.len());
            assert_eq!(plan.layer_idx, 1);
            assert!(!plan.experts_gated);
            assert_eq!(
                plan.input_layout,
                "row_major_selected_rows_by_routed_hidden"
            );
            let reference = cache
                .execute_prefill_reference(&plan, &inputs, case_selected.len())
                .unwrap();
            let cpu_dispatch = cache
                .execute_prefill_test_dispatch(&plan, &inputs, case_selected.len())
                .unwrap();
            let gpu_dispatch = cache
                .execute_prefill_test_gpu_prototype(&plan, &inputs, case_selected.len())
                .unwrap();
            assert_eq!(cpu_dispatch.sorted_row_count, reference.sorted_row_count);
            assert_eq!(gpu_dispatch.sorted_row_count, reference.sorted_row_count);
            assert_eq!(
                cpu_dispatch.routed_hidden_size,
                reference.routed_hidden_size
            );
            assert_eq!(
                gpu_dispatch.routed_hidden_size,
                reference.routed_hidden_size
            );
            assert_eq!(cpu_dispatch.values.len(), reference.values.len());
            assert_eq!(gpu_dispatch.values.len(), reference.values.len());

            for entry in &plan.entries {
                let expert = entry.expert_idx;
                let slot = *slot_by_expert.get(&expert).unwrap();
                let sorted_row = *sorted_row_by_expert.get(&expert).unwrap();
                assert_eq!(entry.row_offset, slot);
                assert_eq!(entry.row_count, 1);
                assert_eq!(entry.w13_key.role, ExpertHqqTensorRole::W13);
                assert_eq!(entry.w2_key.role, ExpertHqqTensorRole::W2);
                assert_eq!(entry.w13_key.layer_idx, 1);
                assert_eq!(entry.w2_key.layer_idx, 1);
                assert_eq!(entry.w13_key.expert_idx, expert);
                assert_eq!(entry.w2_key.expert_idx, expert);
                assert_eq!(entry.w13_nbits, 6);
                assert_eq!(entry.w2_nbits, 6);
                assert_eq!(entry.w13_group_size, 64);
                assert_eq!(entry.w2_group_size, 64);
                assert_eq!(entry.w13_rows, intermediate_size);
                assert_eq!(entry.w13_cols, hidden_size);
                assert_eq!(entry.w2_rows, hidden_size);
                assert_eq!(entry.w2_cols, intermediate_size);
                let w13 = cache.require_tensor_record(entry.w13_key).unwrap();
                let w2 = cache.require_tensor_record(entry.w2_key).unwrap();
                assert_eq!(
                    w13.descriptor.layout,
                    "row_major_axis1_grouped_uint6_packed"
                );
                assert_eq!(w2.descriptor.layout, "row_major_axis1_grouped_uint6_packed");
                descriptor_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    entry.row_offset,
                    entry.row_count,
                    entry.w13_key.role.as_str(),
                    entry.w2_key.role.as_str(),
                    entry.w13_nbits,
                    entry.w2_nbits,
                    entry.w13_group_size,
                    entry.w2_group_size,
                    w13.descriptor.layout,
                    w2.descriptor.layout,
                    entry.w13_rows,
                    entry.w13_cols,
                    entry.w2_rows,
                    entry.w2_cols
                ));

                let captured_branch = branch_outputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing branch output for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_branch_slot_vectors += 1;
                assert_eq!(captured_branch.len(), hidden_size);

                let gpu_branch = &gpu_dispatch.values[slot * hidden_size..(slot + 1) * hidden_size];
                let cpu_branch = &cpu_dispatch.values[slot * hidden_size..(slot + 1) * hidden_size];
                let reference_branch =
                    &reference.values[slot * hidden_size..(slot + 1) * hidden_size];
                let mut slot_gpu_cpu_sum_abs = 0.0f64;
                let mut slot_gpu_cpu_max_abs = 0.0f64;
                let mut slot_gpu_reference_sum_abs = 0.0f64;
                let mut slot_gpu_reference_max_abs = 0.0f64;
                let mut slot_gpu_cpu_final_bf16rounded_sum_abs = 0.0f64;
                let mut slot_gpu_cpu_final_bf16rounded_max_abs = 0.0f64;
                let mut slot_branch_sum_abs = 0.0f64;
                let mut slot_branch_max_abs = 0.0f64;
                let mut slot_branch_l2_sq = 0.0f64;
                for out_row in 0..hidden_size {
                    assert!(
                        gpu_branch[out_row].is_finite(),
                        "GPU prototype output is not finite for case={case_id} expert={expert} slot={slot} out_row={out_row}"
                    );
                    let cpu_delta = (gpu_branch[out_row] - cpu_branch[out_row]).abs() as f64;
                    let reference_delta =
                        (gpu_branch[out_row] - reference_branch[out_row]).abs() as f64;
                    let cpu_rounded = bf16_to_f32(f32_to_bf16(cpu_branch[out_row]));
                    let cpu_rounded_delta = (gpu_branch[out_row] - cpu_rounded).abs() as f64;
                    let branch_delta =
                        (gpu_branch[out_row] - captured_branch[out_row]).abs() as f64;
                    slot_gpu_cpu_sum_abs += cpu_delta;
                    slot_gpu_cpu_max_abs = slot_gpu_cpu_max_abs.max(cpu_delta);
                    slot_gpu_reference_sum_abs += reference_delta;
                    slot_gpu_reference_max_abs = slot_gpu_reference_max_abs.max(reference_delta);
                    slot_gpu_cpu_final_bf16rounded_sum_abs += cpu_rounded_delta;
                    slot_gpu_cpu_final_bf16rounded_max_abs =
                        slot_gpu_cpu_final_bf16rounded_max_abs.max(cpu_rounded_delta);
                    slot_branch_sum_abs += branch_delta;
                    slot_branch_max_abs = slot_branch_max_abs.max(branch_delta);
                    slot_branch_l2_sq += branch_delta * branch_delta;
                }
                gpu_cpu_sum_abs += slot_gpu_cpu_sum_abs;
                gpu_cpu_max_abs = gpu_cpu_max_abs.max(slot_gpu_cpu_max_abs);
                gpu_reference_sum_abs += slot_gpu_reference_sum_abs;
                gpu_reference_max_abs = gpu_reference_max_abs.max(slot_gpu_reference_max_abs);
                gpu_cpu_final_bf16rounded_sum_abs += slot_gpu_cpu_final_bf16rounded_sum_abs;
                gpu_cpu_final_bf16rounded_max_abs =
                    gpu_cpu_final_bf16rounded_max_abs.max(slot_gpu_cpu_final_bf16rounded_max_abs);
                gpu_branch_sum_abs += slot_branch_sum_abs;
                gpu_branch_max_abs = gpu_branch_max_abs.max(slot_branch_max_abs);
                gpu_branch_l2_sq += slot_branch_l2_sq;
                total_slots += 1;
                total_values += hidden_size;
                let input = &inputs[slot * hidden_size..(slot + 1) * hidden_size];
                let input_hash = {
                    let mut bytes = Vec::with_capacity(input.len() * 4);
                    for &value in input {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    format!("0x{:016x}", fnv1a64(&bytes))
                };
                slot_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{input_hash}\t{slot_gpu_cpu_sum_abs:.12}\t{slot_gpu_cpu_max_abs:.12}\t{slot_gpu_reference_sum_abs:.12}\t{slot_gpu_reference_max_abs:.12}\t{slot_gpu_cpu_final_bf16rounded_sum_abs:.12}\t{slot_gpu_cpu_final_bf16rounded_max_abs:.12}\t{slot_branch_sum_abs:.12}\t{slot_branch_max_abs:.12}\t{:.12}",
                    slot_branch_l2_sq.sqrt()
                ));
            }
        }

        assert_eq!(total_slots, 18);
        assert_eq!(total_values, 18 * hidden_size);
        assert_eq!(captured_routed_slot_vectors, 18);
        assert_eq!(captured_branch_slot_vectors, 18);
        let prior_sum = metric(&prior_summary, "hqq6_g64_branch_vs_bf16_captured_sum_abs");
        let prior_max = metric(&prior_summary, "hqq6_g64_branch_vs_bf16_captured_max_abs");
        let prior_l2 = metric(&prior_summary, "hqq6_g64_branch_vs_bf16_captured_l2");
        let cpu_sum = metric(&cpu_summary, "dispatch_branch_vs_bf16_captured_sum_abs");
        let cpu_max = metric(&cpu_summary, "dispatch_branch_vs_bf16_captured_max_abs");
        let cpu_l2 = metric(&cpu_summary, "dispatch_branch_vs_bf16_captured_l2");
        let gpu_l2 = gpu_branch_l2_sq.sqrt();

        write_lines(
            &artifact_dir.join(format!("{label}_descriptor_plan_validation.tsv")),
            &descriptor_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_gpu_slot_compare.tsv")),
            &slot_lines,
        );
        let gpu_cpu_exact_match = gpu_cpu_max_abs == 0.0 && gpu_cpu_sum_abs == 0.0;
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "cache_source_gate\t0836".to_string(),
            "capture_source_gate\t1023".to_string(),
            "cpu_dispatch_source_gate\t1217".to_string(),
            "gpu_prototype_source_gate\t1236".to_string(),
            "selected_cases\t3".to_string(),
            format!("plan_entries\t{total_slots}"),
            format!("tensor_records\t{}", cache.tensors.len()),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            format!("captured_routed_slot_vectors\t{captured_routed_slot_vectors}"),
            format!("captured_branch_slot_vectors\t{captured_branch_slot_vectors}"),
            format!("values\t{total_values}"),
            format!("gpu_cpu_dispatch_sum_abs\t{gpu_cpu_sum_abs:.12}"),
            format!("gpu_cpu_dispatch_max_abs\t{gpu_cpu_max_abs:.12}"),
            format!("gpu_reference_sum_abs\t{gpu_reference_sum_abs:.12}"),
            format!("gpu_reference_max_abs\t{gpu_reference_max_abs:.12}"),
            format!("gpu_cpu_final_bf16rounded_sum_abs\t{gpu_cpu_final_bf16rounded_sum_abs:.12}"),
            format!("gpu_cpu_final_bf16rounded_max_abs\t{gpu_cpu_final_bf16rounded_max_abs:.12}"),
            format!("gpu_cpu_exact_match\t{gpu_cpu_exact_match}"),
            format!("gpu_branch_vs_bf16_captured_sum_abs\t{gpu_branch_sum_abs:.12}"),
            format!("gpu_branch_vs_bf16_captured_max_abs\t{gpu_branch_max_abs:.12}"),
            format!("gpu_branch_vs_bf16_captured_l2\t{gpu_l2:.12}"),
            format!("prior_1023_hqq6_g64_branch_sum_abs\t{prior_sum:.12}"),
            format!("prior_1023_hqq6_g64_branch_max_abs\t{prior_max:.12}"),
            format!("prior_1023_hqq6_g64_branch_l2\t{prior_l2:.12}"),
            format!("prior_1217_cpu_dispatch_branch_sum_abs\t{cpu_sum:.12}"),
            format!("prior_1217_cpu_dispatch_branch_max_abs\t{cpu_max:.12}"),
            format!("prior_1217_cpu_dispatch_branch_l2\t{cpu_l2:.12}"),
            format!(
                "gpu_sum_delta_vs_1023\t{:.12}",
                (gpu_branch_sum_abs - prior_sum).abs()
            ),
            format!(
                "gpu_max_delta_vs_1023\t{:.12}",
                (gpu_branch_max_abs - prior_max).abs()
            ),
            format!("gpu_l2_delta_vs_1023\t{:.12}", (gpu_l2 - prior_l2).abs()),
            format!(
                "gpu_sum_delta_vs_1217_cpu\t{:.12}",
                (gpu_branch_sum_abs - cpu_sum).abs()
            ),
            format!(
                "gpu_max_delta_vs_1217_cpu\t{:.12}",
                (gpu_branch_max_abs - cpu_max).abs()
            ),
            format!("gpu_l2_delta_vs_1217_cpu\t{:.12}", (gpu_l2 - cpu_l2).abs()),
            "runtime_prefill_consumer_added\tfalse".to_string(),
            "config_knob_added\tfalse".to_string(),
            "auto_selection_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_benchmark\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_gpu_replay_summary.tsv")),
            &summary_lines,
        );
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn real_nemotron_gpu_bf16_path_alignment_diagnoses_intermediates() {
        let alignment_proof = env::var("KRASIS_REAL_NANO_GPU_BF16_ALIGNMENT_PROOF")
            .ok()
            .as_deref()
            == Some("1");
        let oracle_contract_proof = env::var("KRASIS_REAL_NANO_GPU_BF16_ORACLE_CONTRACT_PROOF")
            .ok()
            .as_deref()
            == Some("1");
        if !alignment_proof && !oracle_contract_proof {
            eprintln!(
                "skipping real Nano GPU BF16-path alignment proof; set KRASIS_REAL_NANO_GPU_BF16_ALIGNMENT_PROOF=1 or KRASIS_REAL_NANO_GPU_BF16_ORACLE_CONTRACT_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real GPU BF16 alignment proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real GPU BF16 alignment proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();

        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let trace_label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let prior_gpu_label = "20260626_1259_nemotron_nano_real_gpu_prototype_replay_validation";
        let label = if oracle_contract_proof {
            "20260626_1347_nemotron_nano_gpu_bf16_oracle_correctness_contract"
        } else {
            "20260626_1319_nemotron_nano_gpu_bf16_path_numeric_alignment_diagnostics"
        };
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path = artifact_dir.join(format!(
            "{trace_label}_bf16_branch_replay_trace_outputs.json"
        ));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let selected_rows = read_tsv(&artifact_dir.join(
            "20260626_0353_nemotron_nano_int4_prefill_layer1_branch_moe_output_producer_selected_expert_slot_compare.tsv",
        ));
        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "branch output",
        );
        let prior_gpu_summary =
            read_tsv(&artifact_dir.join(format!("{prior_gpu_label}_gpu_replay_summary.tsv")));
        let prior_metric = |name: &str| -> f64 {
            prior_gpu_summary
                .iter()
                .find(|row| tsv_get(row, "metric") == name)
                .unwrap_or_else(|| panic!("missing prior GPU summary metric {name}"))
                .get("value")
                .unwrap()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("failed to parse prior GPU metric {name}: {e}"))
        };

        let mut slot_lines = vec![
            "case_id\texpert\tslot\tsorted_row\tinput_hash\tinput_f32_bf16_sum_abs\tinput_f32_bf16_max_abs\tgpu_oracle_w13_sum_abs\tgpu_oracle_w13_max_abs\tgpu_oracle_activation_sum_abs\tgpu_oracle_activation_max_abs\tgpu_oracle_w2_sum_abs\tgpu_oracle_w2_max_abs\treference_oracle_w13_sum_abs\treference_oracle_w13_max_abs\treference_oracle_activation_sum_abs\treference_oracle_activation_max_abs\treference_oracle_w2_sum_abs\treference_oracle_w2_max_abs\toracle_branch_sum_abs_vs_bf16\toracle_branch_max_abs_vs_bf16\toracle_branch_l2_vs_bf16\tgpu_branch_sum_abs_vs_bf16\tgpu_branch_max_abs_vs_bf16\tgpu_branch_l2_vs_bf16".to_string(),
        ];

        let mut total_slots = 0usize;
        let mut total_values = 0usize;
        let mut captured_routed_slot_vectors = 0usize;
        let mut captured_branch_slot_vectors = 0usize;
        let mut input_f32_bf16 = StageDelta::default();
        let mut gpu_oracle_w13 = StageDelta::default();
        let mut gpu_oracle_activation = StageDelta::default();
        let mut gpu_oracle_w2 = StageDelta::default();
        let mut reference_oracle_w13 = StageDelta::default();
        let mut reference_oracle_activation = StageDelta::default();
        let mut reference_oracle_w2 = StageDelta::default();
        let mut oracle_branch = StageDelta::default();
        let mut gpu_branch = StageDelta::default();

        for case_id in case_ids() {
            let mut case_selected: Vec<_> = selected_rows
                .iter()
                .filter(|row| {
                    tsv_get(row, "case_id") == case_id
                        && tsv_get(row, "component") == "routed_input"
                })
                .collect();
            case_selected.sort_by_key(|row| parse_usize(row, "topk_pos"));
            assert_eq!(
                case_selected.len(),
                6,
                "expected six selected experts for {case_id}"
            );

            let mut works = Vec::with_capacity(case_selected.len());
            let mut inputs = vec![0.0f32; case_selected.len() * hidden_size];
            let mut slot_by_expert = HashMap::new();
            let mut sorted_row_by_expert = HashMap::new();
            for (slot, row) in case_selected.iter().enumerate() {
                let topk_pos = parse_usize(row, "topk_pos");
                assert_eq!(slot, topk_pos);
                let expert = parse_usize(row, "expert");
                let sorted_row = parse_usize(row, "sorted_row");
                works.push(ExpertHqqPrefillWork::new(expert, slot, 1));
                slot_by_expert.insert(expert, slot);
                sorted_row_by_expert.insert(expert, sorted_row);
                let input = routed_inputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing routed input for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_routed_slot_vectors += 1;
                assert_eq!(input.len(), hidden_size);
                inputs[slot * hidden_size..(slot + 1) * hidden_size].copy_from_slice(input);
            }

            let plan = cache.prefill_dispatch_plan(1, false, &works).unwrap();
            assert_eq!(plan.entries.len(), case_selected.len());
            assert_eq!(plan.layer_idx, 1);
            assert!(!plan.experts_gated);
            let reference = cache
                .execute_prefill_reference(&plan, &inputs, case_selected.len())
                .unwrap();
            let oracle = cache
                .execute_prefill_bf16_path_oracle(&plan, &inputs, case_selected.len())
                .unwrap();
            let gpu = cache
                .execute_prefill_test_gpu_prototype(&plan, &inputs, case_selected.len())
                .unwrap();
            assert_eq!(reference.sorted_row_count, case_selected.len());
            assert_eq!(oracle.sorted_row_count, reference.sorted_row_count);
            assert_eq!(gpu.sorted_row_count, reference.sorted_row_count);
            assert_eq!(reference.routed_hidden_size, hidden_size);
            assert_eq!(oracle.routed_hidden_size, hidden_size);
            assert_eq!(gpu.routed_hidden_size, hidden_size);
            assert_eq!(reference.w13_rows, intermediate_size);
            assert_eq!(oracle.w13_rows, reference.w13_rows);
            assert_eq!(gpu.w13_rows, reference.w13_rows);
            assert_eq!(reference.moe_intermediate_size, intermediate_size);
            assert_eq!(oracle.moe_intermediate_size, intermediate_size);
            assert_eq!(gpu.moe_intermediate_size, intermediate_size);
            assert_eq!(
                reference.w13_preactivation.len(),
                oracle.w13_preactivation.len()
            );
            assert_eq!(gpu.w13_preactivation.len(), oracle.w13_preactivation.len());
            assert_eq!(reference.activation.len(), oracle.activation.len());
            assert_eq!(gpu.activation.len(), oracle.activation.len());
            assert_eq!(reference.values.len(), oracle.values.len());
            assert_eq!(gpu.values.len(), oracle.values.len());

            for entry in &plan.entries {
                let expert = entry.expert_idx;
                let slot = *slot_by_expert.get(&expert).unwrap();
                let sorted_row = *sorted_row_by_expert.get(&expert).unwrap();
                assert_eq!(entry.row_offset, slot);
                assert_eq!(entry.row_count, 1);
                assert_eq!(entry.w13_key.role, ExpertHqqTensorRole::W13);
                assert_eq!(entry.w2_key.role, ExpertHqqTensorRole::W2);
                assert_eq!(entry.w13_nbits, 6);
                assert_eq!(entry.w2_nbits, 6);
                assert_eq!(entry.w13_group_size, 64);
                assert_eq!(entry.w2_group_size, 64);
                let w13 = cache.require_tensor_record(entry.w13_key).unwrap();
                let w2 = cache.require_tensor_record(entry.w2_key).unwrap();
                assert_eq!(
                    w13.descriptor.layout,
                    "row_major_axis1_grouped_uint6_packed"
                );
                assert_eq!(w2.descriptor.layout, "row_major_axis1_grouped_uint6_packed");

                let captured_branch = branch_outputs
                    .get(&(case_id.to_string(), expert, sorted_row))
                    .unwrap_or_else(|| {
                        panic!("missing branch output for case={case_id} expert={expert} sorted_row={sorted_row}")
                    });
                captured_branch_slot_vectors += 1;
                assert_eq!(captured_branch.len(), hidden_size);

                let input = &inputs[slot * hidden_size..(slot + 1) * hidden_size];
                let input_bf16 = &oracle.input_bf16[slot * hidden_size..(slot + 1) * hidden_size];
                let w13_range = slot * intermediate_size..(slot + 1) * intermediate_size;
                let activation_range = slot * intermediate_size..(slot + 1) * intermediate_size;
                let output_range = slot * hidden_size..(slot + 1) * hidden_size;

                let mut slot_input = StageDelta::default();
                let mut slot_gpu_w13 = StageDelta::default();
                let mut slot_gpu_activation = StageDelta::default();
                let mut slot_gpu_w2 = StageDelta::default();
                let mut slot_reference_w13 = StageDelta::default();
                let mut slot_reference_activation = StageDelta::default();
                let mut slot_reference_w2 = StageDelta::default();
                let mut slot_oracle_branch = StageDelta::default();
                let mut slot_gpu_branch = StageDelta::default();
                slot_input.add_slices(input, input_bf16);
                slot_gpu_w13.add_slices(
                    &gpu.w13_preactivation[w13_range.clone()],
                    &oracle.w13_preactivation[w13_range.clone()],
                );
                slot_gpu_activation.add_slices(
                    &gpu.activation[activation_range.clone()],
                    &oracle.activation[activation_range.clone()],
                );
                slot_gpu_w2.add_slices(
                    &gpu.values[output_range.clone()],
                    &oracle.values[output_range.clone()],
                );
                slot_reference_w13.add_slices(
                    &reference.w13_preactivation[w13_range.clone()],
                    &oracle.w13_preactivation[w13_range.clone()],
                );
                slot_reference_activation.add_slices(
                    &reference.activation[activation_range.clone()],
                    &oracle.activation[activation_range.clone()],
                );
                slot_reference_w2.add_slices(
                    &reference.values[output_range.clone()],
                    &oracle.values[output_range.clone()],
                );
                slot_oracle_branch
                    .add_slices(&oracle.values[output_range.clone()], captured_branch);
                slot_gpu_branch.add_slices(&gpu.values[output_range.clone()], captured_branch);
                input_f32_bf16.add_slices(input, input_bf16);
                gpu_oracle_w13.add_slices(
                    &gpu.w13_preactivation[w13_range.clone()],
                    &oracle.w13_preactivation[w13_range.clone()],
                );
                gpu_oracle_activation.add_slices(
                    &gpu.activation[activation_range.clone()],
                    &oracle.activation[activation_range.clone()],
                );
                gpu_oracle_w2.add_slices(
                    &gpu.values[output_range.clone()],
                    &oracle.values[output_range.clone()],
                );
                reference_oracle_w13.add_slices(
                    &reference.w13_preactivation[w13_range.clone()],
                    &oracle.w13_preactivation[w13_range.clone()],
                );
                reference_oracle_activation.add_slices(
                    &reference.activation[activation_range.clone()],
                    &oracle.activation[activation_range.clone()],
                );
                reference_oracle_w2.add_slices(
                    &reference.values[output_range.clone()],
                    &oracle.values[output_range.clone()],
                );
                oracle_branch.add_slices(&oracle.values[output_range.clone()], captured_branch);
                gpu_branch.add_slices(&gpu.values[output_range.clone()], captured_branch);

                for value in gpu.w13_preactivation[w13_range.clone()]
                    .iter()
                    .chain(gpu.activation[activation_range.clone()].iter())
                    .chain(gpu.values[output_range.clone()].iter())
                {
                    assert!(
                        value.is_finite(),
                        "GPU BF16-path diagnostic value is not finite for case={case_id} expert={expert}"
                    );
                }

                total_slots += 1;
                total_values += hidden_size;
                let input_hash = {
                    let mut bytes = Vec::with_capacity(input.len() * 4);
                    for &value in input {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    format!("0x{:016x}", fnv1a64(&bytes))
                };
                slot_lines.push(format!(
                    "{case_id}\t{expert}\t{slot}\t{sorted_row}\t{input_hash}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}",
                    slot_input.sum_abs,
                    slot_input.max_abs,
                    slot_gpu_w13.sum_abs,
                    slot_gpu_w13.max_abs,
                    slot_gpu_activation.sum_abs,
                    slot_gpu_activation.max_abs,
                    slot_gpu_w2.sum_abs,
                    slot_gpu_w2.max_abs,
                    slot_reference_w13.sum_abs,
                    slot_reference_w13.max_abs,
                    slot_reference_activation.sum_abs,
                    slot_reference_activation.max_abs,
                    slot_reference_w2.sum_abs,
                    slot_reference_w2.max_abs,
                    slot_oracle_branch.sum_abs,
                    slot_oracle_branch.max_abs,
                    slot_oracle_branch.l2(),
                    slot_gpu_branch.sum_abs,
                    slot_gpu_branch.max_abs,
                    slot_gpu_branch.l2()
                ));
            }
        }

        assert_eq!(total_slots, 18);
        assert_eq!(total_values, 18 * hidden_size);
        assert_eq!(captured_routed_slot_vectors, 18);
        assert_eq!(captured_branch_slot_vectors, 18);
        assert_eq!(gpu_oracle_w13.count, 18 * intermediate_size);
        assert_eq!(gpu_oracle_activation.count, 18 * intermediate_size);
        assert_eq!(gpu_oracle_w2.count, 18 * hidden_size);
        let first_gpu_oracle_stage = first_nonzero_stage(&[
            (
                "HQQ W13 GEMM accumulation/output cast",
                gpu_oracle_w13.max_abs,
            ),
            ("activation", gpu_oracle_activation.max_abs),
            (
                "W2 GEMM/output cast or output indexing",
                gpu_oracle_w2.max_abs,
            ),
        ]);
        let first_cpu_bf16_stage = first_nonzero_stage(&[
            ("input BF16 conversion", input_f32_bf16.max_abs),
            (
                "HQQ W13 GEMM accumulation/output cast",
                reference_oracle_w13.max_abs,
            ),
            ("activation", reference_oracle_activation.max_abs),
            ("W2 GEMM/output cast", reference_oracle_w2.max_abs),
        ]);
        let prior_gpu_cpu_sum = prior_metric("gpu_cpu_dispatch_sum_abs");
        let prior_gpu_cpu_max = prior_metric("gpu_cpu_dispatch_max_abs");
        let prior_gpu_branch_sum = prior_metric("gpu_branch_vs_bf16_captured_sum_abs");
        let prior_gpu_branch_max = prior_metric("gpu_branch_vs_bf16_captured_max_abs");
        let prior_gpu_branch_l2 = prior_metric("gpu_branch_vs_bf16_captured_l2");
        let gpu_oracle_w13_sum_tolerance = 1.0e-30f64;
        let gpu_oracle_w13_max_tolerance = 1.0e-30f64;
        let gpu_oracle_exact_tolerance = 0.0f64;
        let branch_metric_delta_tolerance = 1.0e-9f64;
        let first_gpu_oracle_contract_violation_stage = first_stage_over_tolerance(&[
            (
                "HQQ W13 GEMM accumulation/output cast sum",
                gpu_oracle_w13.sum_abs,
                gpu_oracle_w13_sum_tolerance,
            ),
            (
                "HQQ W13 GEMM accumulation/output cast max",
                gpu_oracle_w13.max_abs,
                gpu_oracle_w13_max_tolerance,
            ),
            (
                "activation sum",
                gpu_oracle_activation.sum_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "activation max",
                gpu_oracle_activation.max_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "W2 GEMM/output cast or output indexing sum",
                gpu_oracle_w2.sum_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "W2 GEMM/output cast or output indexing max",
                gpu_oracle_w2.max_abs,
                gpu_oracle_exact_tolerance,
            ),
        ]);
        let gpu_matches_bf16_path_oracle_contract = gpu_oracle_w13.sum_abs
            <= gpu_oracle_w13_sum_tolerance
            && gpu_oracle_w13.max_abs <= gpu_oracle_w13_max_tolerance
            && gpu_oracle_activation.sum_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_activation.max_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_w2.sum_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_w2.max_abs <= gpu_oracle_exact_tolerance;
        assert!(
            gpu_matches_bf16_path_oracle_contract,
            "GPU prototype violates BF16-path oracle contract: W13 sum/max {:.18e}/{:.18e}, activation sum/max {:.18e}/{:.18e}, W2 sum/max {:.18e}/{:.18e}",
            gpu_oracle_w13.sum_abs,
            gpu_oracle_w13.max_abs,
            gpu_oracle_activation.sum_abs,
            gpu_oracle_activation.max_abs,
            gpu_oracle_w2.sum_abs,
            gpu_oracle_w2.max_abs
        );
        assert!(
            (gpu_branch.sum_abs - prior_gpu_branch_sum).abs() <= branch_metric_delta_tolerance,
            "GPU branch sum delta vs 1259 exceeded tolerance: got {:.18e}, tolerance {:.18e}",
            (gpu_branch.sum_abs - prior_gpu_branch_sum).abs(),
            branch_metric_delta_tolerance
        );
        assert!(
            (gpu_branch.max_abs - prior_gpu_branch_max).abs() <= branch_metric_delta_tolerance,
            "GPU branch max delta vs 1259 exceeded tolerance: got {:.18e}, tolerance {:.18e}",
            (gpu_branch.max_abs - prior_gpu_branch_max).abs(),
            branch_metric_delta_tolerance
        );
        assert!(
            (gpu_branch.l2() - prior_gpu_branch_l2).abs() <= branch_metric_delta_tolerance,
            "GPU branch L2 delta vs 1259 exceeded tolerance: got {:.18e}, tolerance {:.18e}",
            (gpu_branch.l2() - prior_gpu_branch_l2).abs(),
            branch_metric_delta_tolerance
        );

        write_lines(
            &artifact_dir.join(format!("{label}_stage_slot_compare.tsv")),
            &slot_lines,
        );
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "cache_source_gate\t0836".to_string(),
            "capture_source_gate\t1023".to_string(),
            "gpu_prototype_source_gate\t1236".to_string(),
            "prior_gpu_replay_source_gate\t1259".to_string(),
            "selected_cases\t3".to_string(),
            format!("selected_slots\t{total_slots}"),
            format!("output_values\t{total_values}"),
            format!("w13_values\t{}", gpu_oracle_w13.count),
            format!("activation_values\t{}", gpu_oracle_activation.count),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            format!("captured_routed_slot_vectors\t{captured_routed_slot_vectors}"),
            format!("captured_branch_slot_vectors\t{captured_branch_slot_vectors}"),
            format!("input_f32_bf16_sum_abs\t{:.18e}", input_f32_bf16.sum_abs),
            format!("input_f32_bf16_max_abs\t{:.18e}", input_f32_bf16.max_abs),
            format!("gpu_oracle_w13_sum_abs\t{:.18e}", gpu_oracle_w13.sum_abs),
            format!("gpu_oracle_w13_max_abs\t{:.18e}", gpu_oracle_w13.max_abs),
            format!(
                "gpu_oracle_activation_sum_abs\t{:.18e}",
                gpu_oracle_activation.sum_abs
            ),
            format!(
                "gpu_oracle_activation_max_abs\t{:.18e}",
                gpu_oracle_activation.max_abs
            ),
            format!("gpu_oracle_w2_sum_abs\t{:.18e}", gpu_oracle_w2.sum_abs),
            format!("gpu_oracle_w2_max_abs\t{:.18e}", gpu_oracle_w2.max_abs),
            format!(
                "reference_oracle_w13_sum_abs\t{:.18e}",
                reference_oracle_w13.sum_abs
            ),
            format!(
                "reference_oracle_w13_max_abs\t{:.18e}",
                reference_oracle_w13.max_abs
            ),
            format!(
                "reference_oracle_activation_sum_abs\t{:.18e}",
                reference_oracle_activation.sum_abs
            ),
            format!(
                "reference_oracle_activation_max_abs\t{:.18e}",
                reference_oracle_activation.max_abs
            ),
            format!(
                "reference_oracle_w2_sum_abs\t{:.18e}",
                reference_oracle_w2.sum_abs
            ),
            format!(
                "reference_oracle_w2_max_abs\t{:.18e}",
                reference_oracle_w2.max_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                oracle_branch.sum_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_max_abs\t{:.18e}",
                oracle_branch.max_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_l2\t{:.18e}",
                oracle_branch.l2()
            ),
            format!(
                "gpu_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                gpu_branch.sum_abs
            ),
            format!(
                "gpu_branch_vs_bf16_captured_max_abs\t{:.18e}",
                gpu_branch.max_abs
            ),
            format!("gpu_branch_vs_bf16_captured_l2\t{:.18e}", gpu_branch.l2()),
            format!("prior_1259_gpu_cpu_sum_abs\t{prior_gpu_cpu_sum:.18e}"),
            format!("prior_1259_gpu_cpu_max_abs\t{prior_gpu_cpu_max:.18e}"),
            format!("prior_1259_gpu_branch_sum_abs\t{prior_gpu_branch_sum:.18e}"),
            format!("prior_1259_gpu_branch_max_abs\t{prior_gpu_branch_max:.18e}"),
            format!("prior_1259_gpu_branch_l2\t{prior_gpu_branch_l2:.18e}"),
            "correctness_oracle\tbf16_path_oracle".to_string(),
            "cpu_f32_reference_role\tdiagnostic_context".to_string(),
            "captured_bf16_branch_metric_role\tpreserved_metric_context".to_string(),
            format!(
                "gpu_oracle_w13_sum_tolerance\t{gpu_oracle_w13_sum_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_w13_max_tolerance\t{gpu_oracle_w13_max_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_activation_sum_tolerance\t{gpu_oracle_exact_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_activation_max_tolerance\t{gpu_oracle_exact_tolerance:.18e}"
            ),
            format!("gpu_oracle_w2_sum_tolerance\t{gpu_oracle_exact_tolerance:.18e}"),
            format!("gpu_oracle_w2_max_tolerance\t{gpu_oracle_exact_tolerance:.18e}"),
            format!("branch_metric_delta_tolerance\t{branch_metric_delta_tolerance:.18e}"),
            format!(
                "reference_oracle_final_sum_delta_vs_1259_cpu\t{:.18e}",
                (reference_oracle_w2.sum_abs - prior_gpu_cpu_sum).abs()
            ),
            format!("gpu_oracle_final_sum_abs\t{:.18e}", gpu_oracle_w2.sum_abs),
            format!(
                "gpu_branch_sum_delta_vs_1259\t{:.18e}",
                (gpu_branch.sum_abs - prior_gpu_branch_sum).abs()
            ),
            format!(
                "gpu_branch_max_delta_vs_1259\t{:.18e}",
                (gpu_branch.max_abs - prior_gpu_branch_max).abs()
            ),
            format!(
                "gpu_branch_l2_delta_vs_1259\t{:.18e}",
                (gpu_branch.l2() - prior_gpu_branch_l2).abs()
            ),
            format!("first_gpu_oracle_divergent_stage\t{first_gpu_oracle_stage}"),
            format!(
                "first_gpu_oracle_contract_violation_stage\t{first_gpu_oracle_contract_violation_stage}"
            ),
            format!("first_cpu_reference_bf16_path_divergent_stage\t{first_cpu_bf16_stage}"),
            format!(
                "gpu_matches_bf16_path_oracle_contract\t{gpu_matches_bf16_path_oracle_contract}"
            ),
            format!(
                "gpu_matches_bf16_path_oracle\t{}",
                gpu_matches_bf16_path_oracle_contract
            ),
            "runtime_prefill_consumer_added\tfalse".to_string(),
            "config_knob_added\tfalse".to_string(),
            "auto_selection_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_benchmark\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_stage_alignment_summary.tsv")),
            &summary_lines,
        );
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn real_nemotron_full_block_gpu_prototype_matches_bf16_path_oracle() {
        if env::var("KRASIS_REAL_NANO_FULL_BLOCK_GPU_PROTOTYPE_REPLAY_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano full-block GPU prototype proof; set KRASIS_REAL_NANO_FULL_BLOCK_GPU_PROTOTYPE_REPLAY_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real full-block GPU proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real full-block GPU proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();

        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let trace_label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let oracle_label = "20260626_1347_nemotron_nano_gpu_bf16_oracle_correctness_contract";
        let label = "20260626_1412_nemotron_nano_full_block_gpu_prototype_replay_validation";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path = artifact_dir.join(format!(
            "{trace_label}_bf16_branch_replay_trace_outputs.json"
        ));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "branch output",
        );
        assert_eq!(routed_inputs.len(), 385);
        assert_eq!(branch_outputs.len(), 385);
        for key in routed_inputs.keys() {
            assert!(
                branch_outputs.contains_key(key),
                "missing captured branch output for full-block routed key {:?}",
                key
            );
        }

        let oracle_summary =
            read_tsv(&artifact_dir.join(format!("{oracle_label}_stage_alignment_summary.tsv")));
        let oracle_metric = |name: &str| -> f64 {
            oracle_summary
                .iter()
                .find(|row| tsv_get(row, "metric") == name)
                .unwrap_or_else(|| panic!("missing oracle summary metric {name}"))
                .get("value")
                .unwrap()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("failed to parse oracle metric {name}: {e}"))
        };
        let prior_selected_gpu_w13_sum = oracle_metric("gpu_oracle_w13_sum_abs");
        let prior_selected_gpu_w13_max = oracle_metric("gpu_oracle_w13_max_abs");
        let prior_selected_reference_w2_sum = oracle_metric("reference_oracle_w2_sum_abs");
        let gpu_oracle_w13_sum_tolerance = oracle_metric("gpu_oracle_w13_sum_tolerance");
        let gpu_oracle_w13_max_tolerance = oracle_metric("gpu_oracle_w13_max_tolerance");
        let gpu_oracle_exact_tolerance = 0.0f64;

        let mut descriptor_lines = vec![
            "case_id\texpert\tplan_row_offset\tplan_row_count\tsorted_row_start\tsorted_row_end\trows_contiguous\tw13_role\tw2_role\tw13_nbits\tw2_nbits\tw13_group_size\tw2_group_size\tw13_layout\tw2_layout\tw13_rows\tw13_cols\tw2_rows\tw2_cols".to_string(),
        ];
        let mut row_lines = vec![
            "case_id\texpert\tlocal_row\tsorted_row\tinput_hash\tinput_f32_bf16_sum_abs\tinput_f32_bf16_max_abs\tgpu_oracle_w13_sum_abs\tgpu_oracle_w13_max_abs\tgpu_oracle_activation_sum_abs\tgpu_oracle_activation_max_abs\tgpu_oracle_w2_sum_abs\tgpu_oracle_w2_max_abs\treference_oracle_w13_sum_abs\treference_oracle_w13_max_abs\treference_oracle_activation_sum_abs\treference_oracle_activation_max_abs\treference_oracle_w2_sum_abs\treference_oracle_w2_max_abs\treference_branch_sum_abs_vs_bf16\treference_branch_max_abs_vs_bf16\treference_branch_l2_vs_bf16\toracle_branch_sum_abs_vs_bf16\toracle_branch_max_abs_vs_bf16\toracle_branch_l2_vs_bf16\tgpu_branch_sum_abs_vs_bf16\tgpu_branch_max_abs_vs_bf16\tgpu_branch_l2_vs_bf16".to_string(),
        ];

        let mut total_cases = 0usize;
        let mut total_plan_entries = 0usize;
        let mut total_rows = 0usize;
        let mut total_values = 0usize;
        let mut captured_routed_block_vectors = 0usize;
        let mut captured_branch_block_vectors = 0usize;
        let mut input_f32_bf16 = StageDelta::default();
        let mut gpu_oracle_w13 = StageDelta::default();
        let mut gpu_oracle_activation = StageDelta::default();
        let mut gpu_oracle_w2 = StageDelta::default();
        let mut reference_oracle_w13 = StageDelta::default();
        let mut reference_oracle_activation = StageDelta::default();
        let mut reference_oracle_w2 = StageDelta::default();
        let mut reference_branch = StageDelta::default();
        let mut oracle_branch = StageDelta::default();
        let mut gpu_branch = StageDelta::default();

        for case_id in case_ids() {
            let mut by_expert: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for ((row_case, expert, sorted_row), _) in &routed_inputs {
                if row_case == case_id {
                    by_expert.entry(*expert).or_default().push(*sorted_row);
                }
            }
            assert!(
                !by_expert.is_empty(),
                "missing full-block routed inputs for case {case_id}"
            );
            let mut blocks: Vec<(usize, Vec<usize>)> = by_expert.into_iter().collect();
            for (_, rows) in &mut blocks {
                rows.sort_unstable();
                rows.dedup();
                assert!(!rows.is_empty());
                let contiguous = rows.len() == rows[rows.len() - 1] - rows[0] + 1;
                assert!(
                    contiguous,
                    "full-block rows for case={case_id} are not contiguous"
                );
            }
            blocks.sort_by_key(|(_, rows)| rows[0]);

            let case_row_count: usize = blocks.iter().map(|(_, rows)| rows.len()).sum();
            let mut works = Vec::with_capacity(blocks.len());
            let mut inputs = vec![0.0f32; case_row_count * hidden_size];
            let mut local_row_by_key: HashMap<(usize, usize), usize> = HashMap::new();
            let mut cursor = 0usize;
            for (expert, rows) in &blocks {
                works.push(ExpertHqqPrefillWork::new(*expert, cursor, rows.len()));
                for (local_in_block, sorted_row) in rows.iter().enumerate() {
                    let local_row = cursor + local_in_block;
                    local_row_by_key.insert((*expert, *sorted_row), local_row);
                    let input = routed_inputs
                        .get(&(case_id.to_string(), *expert, *sorted_row))
                        .unwrap_or_else(|| {
                            panic!(
                                "missing full-block routed input for case={case_id} expert={expert} sorted_row={sorted_row}"
                            )
                        });
                    assert_eq!(input.len(), hidden_size);
                    inputs[local_row * hidden_size..(local_row + 1) * hidden_size]
                        .copy_from_slice(input);
                    captured_routed_block_vectors += 1;
                }
                cursor += rows.len();
            }
            assert_eq!(cursor, case_row_count);

            let plan = cache.prefill_dispatch_plan(1, false, &works).unwrap();
            assert_eq!(plan.entries.len(), blocks.len());
            assert_eq!(plan.layer_idx, 1);
            assert!(!plan.experts_gated);
            assert_eq!(
                plan.input_layout,
                "row_major_selected_rows_by_routed_hidden"
            );
            let reference = cache
                .execute_prefill_reference(&plan, &inputs, case_row_count)
                .unwrap();
            let oracle = cache
                .execute_prefill_bf16_path_oracle(&plan, &inputs, case_row_count)
                .unwrap();
            let gpu = cache
                .execute_prefill_test_gpu_prototype(&plan, &inputs, case_row_count)
                .unwrap();
            assert_eq!(reference.sorted_row_count, case_row_count);
            assert_eq!(oracle.sorted_row_count, reference.sorted_row_count);
            assert_eq!(gpu.sorted_row_count, reference.sorted_row_count);
            assert_eq!(reference.routed_hidden_size, hidden_size);
            assert_eq!(oracle.routed_hidden_size, hidden_size);
            assert_eq!(gpu.routed_hidden_size, hidden_size);
            assert_eq!(reference.w13_rows, intermediate_size);
            assert_eq!(oracle.w13_rows, reference.w13_rows);
            assert_eq!(gpu.w13_rows, reference.w13_rows);
            assert_eq!(reference.moe_intermediate_size, intermediate_size);
            assert_eq!(oracle.moe_intermediate_size, intermediate_size);
            assert_eq!(gpu.moe_intermediate_size, intermediate_size);
            assert_eq!(
                reference.w13_preactivation.len(),
                case_row_count * intermediate_size
            );
            assert_eq!(
                oracle.w13_preactivation.len(),
                reference.w13_preactivation.len()
            );
            assert_eq!(
                gpu.w13_preactivation.len(),
                reference.w13_preactivation.len()
            );
            assert_eq!(
                reference.activation.len(),
                case_row_count * intermediate_size
            );
            assert_eq!(oracle.activation.len(), reference.activation.len());
            assert_eq!(gpu.activation.len(), reference.activation.len());
            assert_eq!(reference.values.len(), case_row_count * hidden_size);
            assert_eq!(oracle.values.len(), reference.values.len());
            assert_eq!(gpu.values.len(), reference.values.len());

            for entry in &plan.entries {
                let rows = blocks
                    .iter()
                    .find(|(expert, _)| *expert == entry.expert_idx)
                    .map(|(_, rows)| rows)
                    .unwrap_or_else(|| {
                        panic!(
                            "missing full-block rows for planned expert {}",
                            entry.expert_idx
                        )
                    });
                assert_eq!(
                    entry.row_count,
                    rows.len(),
                    "planned row count mismatch for case={case_id} expert={}",
                    entry.expert_idx
                );
                assert_eq!(
                    local_row_by_key[&(entry.expert_idx, rows[0])],
                    entry.row_offset
                );
                assert_eq!(entry.w13_key.role, ExpertHqqTensorRole::W13);
                assert_eq!(entry.w2_key.role, ExpertHqqTensorRole::W2);
                assert_eq!(entry.w13_key.layer_idx, 1);
                assert_eq!(entry.w2_key.layer_idx, 1);
                assert_eq!(entry.w13_key.expert_idx, entry.expert_idx);
                assert_eq!(entry.w2_key.expert_idx, entry.expert_idx);
                assert_eq!(entry.w13_nbits, 6);
                assert_eq!(entry.w2_nbits, 6);
                assert_eq!(entry.w13_group_size, 64);
                assert_eq!(entry.w2_group_size, 64);
                assert_eq!(entry.w13_rows, intermediate_size);
                assert_eq!(entry.w13_cols, hidden_size);
                assert_eq!(entry.w2_rows, hidden_size);
                assert_eq!(entry.w2_cols, intermediate_size);
                let w13 = cache.require_tensor_record(entry.w13_key).unwrap();
                let w2 = cache.require_tensor_record(entry.w2_key).unwrap();
                assert_eq!(
                    w13.descriptor.layout,
                    "row_major_axis1_grouped_uint6_packed"
                );
                assert_eq!(w2.descriptor.layout, "row_major_axis1_grouped_uint6_packed");
                descriptor_lines.push(format!(
                    "{case_id}\t{}\t{}\t{}\t{}\t{}\ttrue\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    entry.expert_idx,
                    entry.row_offset,
                    entry.row_count,
                    rows[0],
                    rows[rows.len() - 1],
                    entry.w13_key.role.as_str(),
                    entry.w2_key.role.as_str(),
                    entry.w13_nbits,
                    entry.w2_nbits,
                    entry.w13_group_size,
                    entry.w2_group_size,
                    w13.descriptor.layout,
                    w2.descriptor.layout,
                    entry.w13_rows,
                    entry.w13_cols,
                    entry.w2_rows,
                    entry.w2_cols
                ));
            }

            for (expert, rows) in &blocks {
                for sorted_row in rows {
                    let local_row = local_row_by_key[&(*expert, *sorted_row)];
                    let captured_branch = branch_outputs
                        .get(&(case_id.to_string(), *expert, *sorted_row))
                        .unwrap_or_else(|| {
                            panic!(
                                "missing full-block branch output for case={case_id} expert={expert} sorted_row={sorted_row}"
                            )
                        });
                    captured_branch_block_vectors += 1;
                    assert_eq!(captured_branch.len(), hidden_size);

                    let input_range = local_row * hidden_size..(local_row + 1) * hidden_size;
                    let w13_range =
                        local_row * intermediate_size..(local_row + 1) * intermediate_size;
                    let activation_range = w13_range.clone();
                    let output_range = input_range.clone();
                    let input = &inputs[input_range.clone()];
                    let input_bf16 = &oracle.input_bf16[input_range.clone()];

                    let mut row_input = StageDelta::default();
                    let mut row_gpu_w13 = StageDelta::default();
                    let mut row_gpu_activation = StageDelta::default();
                    let mut row_gpu_w2 = StageDelta::default();
                    let mut row_reference_w13 = StageDelta::default();
                    let mut row_reference_activation = StageDelta::default();
                    let mut row_reference_w2 = StageDelta::default();
                    let mut row_reference_branch = StageDelta::default();
                    let mut row_oracle_branch = StageDelta::default();
                    let mut row_gpu_branch = StageDelta::default();

                    row_input.add_slices(input, input_bf16);
                    row_gpu_w13.add_slices(
                        &gpu.w13_preactivation[w13_range.clone()],
                        &oracle.w13_preactivation[w13_range.clone()],
                    );
                    row_gpu_activation.add_slices(
                        &gpu.activation[activation_range.clone()],
                        &oracle.activation[activation_range.clone()],
                    );
                    row_gpu_w2.add_slices(
                        &gpu.values[output_range.clone()],
                        &oracle.values[output_range.clone()],
                    );
                    row_reference_w13.add_slices(
                        &reference.w13_preactivation[w13_range.clone()],
                        &oracle.w13_preactivation[w13_range.clone()],
                    );
                    row_reference_activation.add_slices(
                        &reference.activation[activation_range.clone()],
                        &oracle.activation[activation_range.clone()],
                    );
                    row_reference_w2.add_slices(
                        &reference.values[output_range.clone()],
                        &oracle.values[output_range.clone()],
                    );
                    row_reference_branch
                        .add_slices(&reference.values[output_range.clone()], captured_branch);
                    row_oracle_branch
                        .add_slices(&oracle.values[output_range.clone()], captured_branch);
                    row_gpu_branch.add_slices(&gpu.values[output_range.clone()], captured_branch);

                    input_f32_bf16.add_slices(input, input_bf16);
                    gpu_oracle_w13.add_slices(
                        &gpu.w13_preactivation[w13_range.clone()],
                        &oracle.w13_preactivation[w13_range.clone()],
                    );
                    gpu_oracle_activation.add_slices(
                        &gpu.activation[activation_range.clone()],
                        &oracle.activation[activation_range.clone()],
                    );
                    gpu_oracle_w2.add_slices(
                        &gpu.values[output_range.clone()],
                        &oracle.values[output_range.clone()],
                    );
                    reference_oracle_w13.add_slices(
                        &reference.w13_preactivation[w13_range.clone()],
                        &oracle.w13_preactivation[w13_range.clone()],
                    );
                    reference_oracle_activation.add_slices(
                        &reference.activation[activation_range.clone()],
                        &oracle.activation[activation_range.clone()],
                    );
                    reference_oracle_w2.add_slices(
                        &reference.values[output_range.clone()],
                        &oracle.values[output_range.clone()],
                    );
                    reference_branch
                        .add_slices(&reference.values[output_range.clone()], captured_branch);
                    oracle_branch.add_slices(&oracle.values[output_range.clone()], captured_branch);
                    gpu_branch.add_slices(&gpu.values[output_range.clone()], captured_branch);

                    for value in gpu.w13_preactivation[w13_range.clone()]
                        .iter()
                        .chain(gpu.activation[activation_range.clone()].iter())
                        .chain(gpu.values[output_range.clone()].iter())
                    {
                        assert!(
                            value.is_finite(),
                            "GPU full-block value is not finite for case={case_id} expert={expert} sorted_row={sorted_row}"
                        );
                    }

                    total_rows += 1;
                    total_values += hidden_size;
                    let input_hash = {
                        let mut bytes = Vec::with_capacity(input.len() * 4);
                        for &value in input {
                            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                        }
                        format!("0x{:016x}", fnv1a64(&bytes))
                    };
                    row_lines.push(format!(
                        "{case_id}\t{expert}\t{local_row}\t{sorted_row}\t{input_hash}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}",
                        row_input.sum_abs,
                        row_input.max_abs,
                        row_gpu_w13.sum_abs,
                        row_gpu_w13.max_abs,
                        row_gpu_activation.sum_abs,
                        row_gpu_activation.max_abs,
                        row_gpu_w2.sum_abs,
                        row_gpu_w2.max_abs,
                        row_reference_w13.sum_abs,
                        row_reference_w13.max_abs,
                        row_reference_activation.sum_abs,
                        row_reference_activation.max_abs,
                        row_reference_w2.sum_abs,
                        row_reference_w2.max_abs,
                        row_reference_branch.sum_abs,
                        row_reference_branch.max_abs,
                        row_reference_branch.l2(),
                        row_oracle_branch.sum_abs,
                        row_oracle_branch.max_abs,
                        row_oracle_branch.l2(),
                        row_gpu_branch.sum_abs,
                        row_gpu_branch.max_abs,
                        row_gpu_branch.l2()
                    ));
                }
            }

            total_cases += 1;
            total_plan_entries += plan.entries.len();
        }

        assert_eq!(total_cases, 3);
        assert_eq!(total_plan_entries, 18);
        assert_eq!(total_rows, 385);
        assert_eq!(total_values, 385 * hidden_size);
        assert_eq!(captured_routed_block_vectors, 385);
        assert_eq!(captured_branch_block_vectors, 385);
        assert_eq!(gpu_oracle_w13.count, 385 * intermediate_size);
        assert_eq!(gpu_oracle_activation.count, 385 * intermediate_size);
        assert_eq!(gpu_oracle_w2.count, 385 * hidden_size);

        let first_gpu_oracle_stage = first_nonzero_stage(&[
            (
                "HQQ W13 GEMM accumulation/output cast",
                gpu_oracle_w13.max_abs,
            ),
            ("activation", gpu_oracle_activation.max_abs),
            (
                "W2 GEMM/output cast or output indexing",
                gpu_oracle_w2.max_abs,
            ),
        ]);
        let first_cpu_bf16_stage = first_nonzero_stage(&[
            ("input BF16 conversion", input_f32_bf16.max_abs),
            (
                "HQQ W13 GEMM accumulation/output cast",
                reference_oracle_w13.max_abs,
            ),
            ("activation", reference_oracle_activation.max_abs),
            ("W2 GEMM/output cast", reference_oracle_w2.max_abs),
        ]);
        let first_gpu_oracle_contract_violation_stage = first_stage_over_tolerance(&[
            (
                "HQQ W13 GEMM accumulation/output cast sum",
                gpu_oracle_w13.sum_abs,
                gpu_oracle_w13_sum_tolerance,
            ),
            (
                "HQQ W13 GEMM accumulation/output cast max",
                gpu_oracle_w13.max_abs,
                gpu_oracle_w13_max_tolerance,
            ),
            (
                "activation sum",
                gpu_oracle_activation.sum_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "activation max",
                gpu_oracle_activation.max_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "W2 GEMM/output cast or output indexing sum",
                gpu_oracle_w2.sum_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "W2 GEMM/output cast or output indexing max",
                gpu_oracle_w2.max_abs,
                gpu_oracle_exact_tolerance,
            ),
        ]);
        let gpu_matches_bf16_path_oracle_contract = gpu_oracle_w13.sum_abs
            <= gpu_oracle_w13_sum_tolerance
            && gpu_oracle_w13.max_abs <= gpu_oracle_w13_max_tolerance
            && gpu_oracle_activation.sum_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_activation.max_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_w2.sum_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_w2.max_abs <= gpu_oracle_exact_tolerance;

        write_lines(
            &artifact_dir.join(format!("{label}_descriptor_plan_validation.tsv")),
            &descriptor_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_full_block_row_compare.tsv")),
            &row_lines,
        );
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "cache_source_gate\t0836".to_string(),
            "capture_source_gate\t1023".to_string(),
            "gpu_prototype_source_gate\t1236".to_string(),
            "bf16_oracle_contract_source_gate\t1347".to_string(),
            format!("cases\t{total_cases}"),
            format!("plan_entries\t{total_plan_entries}"),
            format!("full_block_rows\t{total_rows}"),
            format!("output_values\t{total_values}"),
            format!("w13_values\t{}", gpu_oracle_w13.count),
            format!("activation_values\t{}", gpu_oracle_activation.count),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            format!("captured_routed_block_vectors\t{captured_routed_block_vectors}"),
            format!("captured_branch_block_vectors\t{captured_branch_block_vectors}"),
            format!("input_f32_bf16_sum_abs\t{:.18e}", input_f32_bf16.sum_abs),
            format!("input_f32_bf16_max_abs\t{:.18e}", input_f32_bf16.max_abs),
            format!("gpu_oracle_w13_sum_abs\t{:.18e}", gpu_oracle_w13.sum_abs),
            format!("gpu_oracle_w13_max_abs\t{:.18e}", gpu_oracle_w13.max_abs),
            format!(
                "gpu_oracle_activation_sum_abs\t{:.18e}",
                gpu_oracle_activation.sum_abs
            ),
            format!(
                "gpu_oracle_activation_max_abs\t{:.18e}",
                gpu_oracle_activation.max_abs
            ),
            format!("gpu_oracle_w2_sum_abs\t{:.18e}", gpu_oracle_w2.sum_abs),
            format!("gpu_oracle_w2_max_abs\t{:.18e}", gpu_oracle_w2.max_abs),
            format!(
                "reference_oracle_w13_sum_abs\t{:.18e}",
                reference_oracle_w13.sum_abs
            ),
            format!(
                "reference_oracle_w13_max_abs\t{:.18e}",
                reference_oracle_w13.max_abs
            ),
            format!(
                "reference_oracle_activation_sum_abs\t{:.18e}",
                reference_oracle_activation.sum_abs
            ),
            format!(
                "reference_oracle_activation_max_abs\t{:.18e}",
                reference_oracle_activation.max_abs
            ),
            format!(
                "reference_oracle_w2_sum_abs\t{:.18e}",
                reference_oracle_w2.sum_abs
            ),
            format!(
                "reference_oracle_w2_max_abs\t{:.18e}",
                reference_oracle_w2.max_abs
            ),
            format!(
                "reference_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                reference_branch.sum_abs
            ),
            format!(
                "reference_branch_vs_bf16_captured_max_abs\t{:.18e}",
                reference_branch.max_abs
            ),
            format!(
                "reference_branch_vs_bf16_captured_l2\t{:.18e}",
                reference_branch.l2()
            ),
            format!(
                "oracle_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                oracle_branch.sum_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_max_abs\t{:.18e}",
                oracle_branch.max_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_l2\t{:.18e}",
                oracle_branch.l2()
            ),
            format!(
                "gpu_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                gpu_branch.sum_abs
            ),
            format!(
                "gpu_branch_vs_bf16_captured_max_abs\t{:.18e}",
                gpu_branch.max_abs
            ),
            format!("gpu_branch_vs_bf16_captured_l2\t{:.18e}", gpu_branch.l2()),
            format!(
                "prior_1347_selected_gpu_oracle_w13_sum_abs\t{prior_selected_gpu_w13_sum:.18e}"
            ),
            format!(
                "prior_1347_selected_gpu_oracle_w13_max_abs\t{prior_selected_gpu_w13_max:.18e}"
            ),
            format!(
                "prior_1347_selected_reference_oracle_w2_sum_abs\t{prior_selected_reference_w2_sum:.18e}"
            ),
            "correctness_oracle\tbf16_path_oracle".to_string(),
            "cpu_f32_reference_role\tdiagnostic_context".to_string(),
            "captured_bf16_branch_metric_role\tfull_block_diagnostic_context".to_string(),
            format!(
                "gpu_oracle_w13_sum_tolerance\t{gpu_oracle_w13_sum_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_w13_max_tolerance\t{gpu_oracle_w13_max_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_activation_sum_tolerance\t{gpu_oracle_exact_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_activation_max_tolerance\t{gpu_oracle_exact_tolerance:.18e}"
            ),
            format!("gpu_oracle_w2_sum_tolerance\t{gpu_oracle_exact_tolerance:.18e}"),
            format!("gpu_oracle_w2_max_tolerance\t{gpu_oracle_exact_tolerance:.18e}"),
            format!("gpu_oracle_final_sum_abs\t{:.18e}", gpu_oracle_w2.sum_abs),
            format!("first_gpu_oracle_divergent_stage\t{first_gpu_oracle_stage}"),
            format!(
                "first_gpu_oracle_contract_violation_stage\t{first_gpu_oracle_contract_violation_stage}"
            ),
            format!("first_cpu_reference_bf16_path_divergent_stage\t{first_cpu_bf16_stage}"),
            format!(
                "gpu_matches_bf16_path_oracle_contract\t{gpu_matches_bf16_path_oracle_contract}"
            ),
            "runtime_prefill_consumer_added\tfalse".to_string(),
            "config_knob_added\tfalse".to_string(),
            "auto_selection_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_benchmark\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_full_block_replay_summary.tsv")),
            &summary_lines,
        );

        assert!(
            gpu_matches_bf16_path_oracle_contract,
            "full-block GPU prototype violates BF16-path oracle contract: W13 sum/max {:.18e}/{:.18e}, activation sum/max {:.18e}/{:.18e}, W2 sum/max {:.18e}/{:.18e}; first violation {first_gpu_oracle_contract_violation_stage}",
            gpu_oracle_w13.sum_abs,
            gpu_oracle_w13.max_abs,
            gpu_oracle_activation.sum_abs,
            gpu_oracle_activation.max_abs,
            gpu_oracle_w2.sum_abs,
            gpu_oracle_w2.max_abs
        );
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn real_nemotron_runtime_shaped_gpu_prefill_buffers_match_bf16_path_oracle() {
        if env::var("KRASIS_REAL_NANO_RUNTIME_SHAPED_GPU_PREFILL_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano runtime-shaped GPU prefill proof; set KRASIS_REAL_NANO_RUNTIME_SHAPED_GPU_PREFILL_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for real runtime-shaped GPU proof"),
        );
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for real runtime-shaped GPU proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();

        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let trace_label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let full_block_label =
            "20260626_1412_nemotron_nano_full_block_gpu_prototype_replay_validation";
        let label =
            "20260626_1432_nemotron_nano_runtime_shaped_gpu_prefill_buffer_contract_validation";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path = artifact_dir.join(format!(
            "{trace_label}_bf16_branch_replay_trace_outputs.json"
        ));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "runtime-shaped routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "runtime-shaped branch output",
        );
        assert_eq!(routed_inputs.len(), 385);
        assert_eq!(branch_outputs.len(), 385);
        for key in routed_inputs.keys() {
            assert!(
                branch_outputs.contains_key(key),
                "missing captured branch output for runtime-shaped key {:?}",
                key
            );
        }

        let full_block_summary = read_tsv(
            &artifact_dir.join(format!("{full_block_label}_full_block_replay_summary.tsv")),
        );
        let full_block_metric = |name: &str| -> f64 {
            full_block_summary
                .iter()
                .find(|row| tsv_get(row, "metric") == name)
                .unwrap_or_else(|| panic!("missing full-block summary metric {name}"))
                .get("value")
                .unwrap()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("failed to parse full-block metric {name}: {e}"))
        };
        let full_block_gpu_w13_sum = full_block_metric("gpu_oracle_w13_sum_abs");
        let full_block_gpu_w13_max = full_block_metric("gpu_oracle_w13_max_abs");
        let full_block_reference_w2_sum = full_block_metric("reference_oracle_w2_sum_abs");
        let gpu_oracle_w13_sum_tolerance = full_block_metric("gpu_oracle_w13_sum_tolerance");
        let gpu_oracle_w13_max_tolerance = full_block_metric("gpu_oracle_w13_max_tolerance");
        let gpu_oracle_exact_tolerance = 0.0f64;

        let mut descriptor_lines = vec![
            "case_id\texpert\tabsolute_row_offset\trow_count\trow_end\tcompact_row_offset\ttotal_sorted_rows\tclaimed_rows\tpadding_rows\tinput_stride\tw13_stride\tactivation_stride\toutput_stride\tw13_role\tw2_role\tw13_nbits\tw2_nbits\tw13_group_size\tw2_group_size\tw13_layout\tw2_layout".to_string(),
        ];
        let mut row_lines = vec![
            "case_id\texpert\tcompact_row\tabsolute_row\tinput_hash\tinput_f32_bf16_sum_abs\tinput_f32_bf16_max_abs\tgpu_oracle_w13_sum_abs\tgpu_oracle_w13_max_abs\tgpu_oracle_activation_sum_abs\tgpu_oracle_activation_max_abs\tgpu_oracle_w2_sum_abs\tgpu_oracle_w2_max_abs\treference_oracle_w13_sum_abs\treference_oracle_w13_max_abs\treference_oracle_activation_sum_abs\treference_oracle_activation_max_abs\treference_oracle_w2_sum_abs\treference_oracle_w2_max_abs\treference_branch_sum_abs_vs_bf16\treference_branch_max_abs_vs_bf16\treference_branch_l2_vs_bf16\toracle_branch_sum_abs_vs_bf16\toracle_branch_max_abs_vs_bf16\toracle_branch_l2_vs_bf16\tgpu_branch_sum_abs_vs_bf16\tgpu_branch_max_abs_vs_bf16\tgpu_branch_l2_vs_bf16".to_string(),
        ];

        let mut total_cases = 0usize;
        let mut total_plan_entries = 0usize;
        let mut total_claimed_rows = 0usize;
        let mut total_runtime_rows = 0usize;
        let mut total_padding_rows = 0usize;
        let mut total_output_values = 0usize;
        let mut total_output_buffer_values = 0usize;
        let mut captured_routed_block_vectors = 0usize;
        let mut captured_branch_block_vectors = 0usize;
        let mut input_f32_bf16 = StageDelta::default();
        let mut gpu_oracle_w13 = StageDelta::default();
        let mut gpu_oracle_activation = StageDelta::default();
        let mut gpu_oracle_w2 = StageDelta::default();
        let mut reference_oracle_w13 = StageDelta::default();
        let mut reference_oracle_activation = StageDelta::default();
        let mut reference_oracle_w2 = StageDelta::default();
        let mut reference_branch = StageDelta::default();
        let mut oracle_branch = StageDelta::default();
        let mut gpu_branch = StageDelta::default();
        let mut padding_w13_untouched = true;
        let mut padding_activation_untouched = true;
        let mut padding_output_untouched = true;

        for case_id in case_ids() {
            let mut by_expert: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for ((row_case, expert, sorted_row), _) in &routed_inputs {
                if row_case == case_id {
                    by_expert.entry(*expert).or_default().push(*sorted_row);
                }
            }
            assert!(
                !by_expert.is_empty(),
                "missing runtime-shaped routed inputs for case {case_id}"
            );
            let mut blocks_by_expert: Vec<(usize, Vec<usize>)> = by_expert.into_iter().collect();
            for (_, rows) in &mut blocks_by_expert {
                rows.sort_unstable();
                rows.dedup();
                assert!(!rows.is_empty());
                assert_eq!(
                    rows.len(),
                    rows[rows.len() - 1] - rows[0] + 1,
                    "runtime-shaped rows for case={case_id} must be contiguous inside each expert block"
                );
            }
            blocks_by_expert.sort_by_key(|(_, rows)| rows[0]);
            let case_row_count: usize = blocks_by_expert.iter().map(|(_, rows)| rows.len()).sum();
            let max_row = blocks_by_expert
                .iter()
                .flat_map(|(_, rows)| rows.iter().copied())
                .max()
                .expect("runtime-shaped block rows should be non-empty");
            let total_sorted_rows = max_row + 1;
            let shape = ExpertHqqRuntimePrefillBufferShape::contiguous_for_cache(
                &cache,
                false,
                total_sorted_rows,
            )
            .unwrap();
            let mut runtime_inputs = vec![
                777.0f32;
                runtime_buffer_len(
                    shape.total_sorted_rows,
                    shape.input_row_stride,
                    hidden_size
                )
                .unwrap()
            ];
            let mut runtime_blocks = Vec::with_capacity(blocks_by_expert.len());
            let mut compact_works = Vec::with_capacity(blocks_by_expert.len());
            let mut compact_inputs = vec![0.0f32; case_row_count * hidden_size];
            let mut local_row_by_key: HashMap<(usize, usize), usize> = HashMap::new();
            let mut compact_cursor = 0usize;
            for (expert, rows) in &blocks_by_expert {
                runtime_blocks.push(ExpertHqqRuntimePrefillBlock::new(
                    *expert,
                    rows[0],
                    rows.len(),
                ));
                compact_works.push(ExpertHqqPrefillWork::new(
                    *expert,
                    compact_cursor,
                    rows.len(),
                ));
                for (local_in_block, sorted_row) in rows.iter().enumerate() {
                    let compact_row = compact_cursor + local_in_block;
                    local_row_by_key.insert((*expert, *sorted_row), compact_row);
                    let input = routed_inputs
                        .get(&(case_id.to_string(), *expert, *sorted_row))
                        .unwrap_or_else(|| {
                            panic!(
                                "missing runtime-shaped routed input for case={case_id} expert={expert} sorted_row={sorted_row}"
                            )
                        });
                    assert_eq!(input.len(), hidden_size);
                    let runtime_start = sorted_row * shape.input_row_stride;
                    runtime_inputs[runtime_start..runtime_start + hidden_size]
                        .copy_from_slice(input);
                    compact_inputs[compact_row * hidden_size..(compact_row + 1) * hidden_size]
                        .copy_from_slice(input);
                    captured_routed_block_vectors += 1;
                }
                compact_cursor += rows.len();
            }
            assert_eq!(compact_cursor, case_row_count);

            let compact_plan = cache
                .prefill_dispatch_plan(1, false, &compact_works)
                .unwrap();
            let absolute_works: Vec<ExpertHqqPrefillWork> = runtime_blocks
                .iter()
                .map(|block| {
                    ExpertHqqPrefillWork::new(
                        block.expert_idx,
                        block.absolute_row_offset,
                        block.row_count,
                    )
                })
                .collect();
            let absolute_plan = cache
                .prefill_dispatch_plan(1, false, &absolute_works)
                .unwrap();
            let reference = cache
                .execute_prefill_reference(&compact_plan, &compact_inputs, case_row_count)
                .unwrap();
            let oracle = cache
                .execute_prefill_bf16_path_oracle(&compact_plan, &compact_inputs, case_row_count)
                .unwrap();
            let runtime = cache
                .execute_prefill_runtime_shaped_gpu_prototype(
                    1,
                    false,
                    &runtime_blocks,
                    shape,
                    &runtime_inputs,
                )
                .unwrap();
            assert_eq!(runtime.total_sorted_rows, total_sorted_rows);
            assert_eq!(runtime.compact_row_count, case_row_count);
            assert_eq!(
                runtime
                    .claimed_rows
                    .iter()
                    .filter(|&&claimed| claimed)
                    .count(),
                case_row_count
            );
            assert_eq!(runtime.input_row_stride, hidden_size);
            assert_eq!(runtime.w13_row_stride, intermediate_size);
            assert_eq!(runtime.activation_row_stride, intermediate_size);
            assert_eq!(runtime.output_row_stride, hidden_size);
            assert_eq!(runtime.routed_hidden_size, hidden_size);
            assert_eq!(runtime.w13_rows, intermediate_size);
            assert_eq!(runtime.moe_intermediate_size, intermediate_size);

            for (entry_idx, entry) in absolute_plan.entries.iter().enumerate() {
                let block = runtime_blocks[entry_idx];
                assert_eq!(entry.expert_idx, block.expert_idx);
                assert_eq!(entry.row_offset, block.absolute_row_offset);
                assert_eq!(entry.row_count, block.row_count);
                let w13 = cache.require_tensor_record(entry.w13_key).unwrap();
                let w2 = cache.require_tensor_record(entry.w2_key).unwrap();
                assert_eq!(entry.w13_key.role, ExpertHqqTensorRole::W13);
                assert_eq!(entry.w2_key.role, ExpertHqqTensorRole::W2);
                assert_eq!(entry.w13_nbits, 6);
                assert_eq!(entry.w2_nbits, 6);
                assert_eq!(entry.w13_group_size, 64);
                assert_eq!(entry.w2_group_size, 64);
                assert_eq!(
                    w13.descriptor.layout,
                    "row_major_axis1_grouped_uint6_packed"
                );
                assert_eq!(w2.descriptor.layout, "row_major_axis1_grouped_uint6_packed");
                descriptor_lines.push(format!(
                    "{case_id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    entry.expert_idx,
                    entry.row_offset,
                    entry.row_count,
                    entry.row_offset + entry.row_count,
                    compact_plan.entries[entry_idx].row_offset,
                    total_sorted_rows,
                    case_row_count,
                    total_sorted_rows - case_row_count,
                    runtime.input_row_stride,
                    runtime.w13_row_stride,
                    runtime.activation_row_stride,
                    runtime.output_row_stride,
                    entry.w13_key.role.as_str(),
                    entry.w2_key.role.as_str(),
                    entry.w13_nbits,
                    entry.w2_nbits,
                    entry.w13_group_size,
                    entry.w2_group_size,
                    w13.descriptor.layout,
                    w2.descriptor.layout
                ));
            }

            for row in 0..total_sorted_rows {
                if runtime.claimed_rows[row] {
                    continue;
                }
                let w13_start = row * runtime.w13_row_stride;
                let act_start = row * runtime.activation_row_stride;
                let out_start = row * runtime.output_row_stride;
                padding_w13_untouched &= runtime.w13_preactivation
                    [w13_start..w13_start + intermediate_size]
                    .iter()
                    .all(|value| value.is_nan());
                padding_activation_untouched &= runtime.activation
                    [act_start..act_start + intermediate_size]
                    .iter()
                    .all(|value| value.is_nan());
                padding_output_untouched &= runtime.values[out_start..out_start + hidden_size]
                    .iter()
                    .all(|value| value.is_nan());
            }

            for (expert, rows) in &blocks_by_expert {
                for sorted_row in rows {
                    assert!(runtime.claimed_rows[*sorted_row]);
                    let compact_row = local_row_by_key[&(*expert, *sorted_row)];
                    let captured_branch = branch_outputs
                        .get(&(case_id.to_string(), *expert, *sorted_row))
                        .unwrap_or_else(|| {
                            panic!(
                                "missing runtime-shaped branch output for case={case_id} expert={expert} sorted_row={sorted_row}"
                            )
                        });
                    captured_branch_block_vectors += 1;
                    assert_eq!(captured_branch.len(), hidden_size);

                    let compact_input_range =
                        compact_row * hidden_size..(compact_row + 1) * hidden_size;
                    let compact_w13_range =
                        compact_row * intermediate_size..(compact_row + 1) * intermediate_size;
                    let runtime_input_start = sorted_row * shape.input_row_stride;
                    let runtime_w13_start = sorted_row * runtime.w13_row_stride;
                    let runtime_activation_start = sorted_row * runtime.activation_row_stride;
                    let runtime_output_start = sorted_row * runtime.output_row_stride;
                    let runtime_input =
                        &runtime_inputs[runtime_input_start..runtime_input_start + hidden_size];
                    let input_bf16 = &oracle.input_bf16[compact_input_range.clone()];
                    let runtime_w13 = &runtime.w13_preactivation
                        [runtime_w13_start..runtime_w13_start + intermediate_size];
                    let runtime_activation = &runtime.activation
                        [runtime_activation_start..runtime_activation_start + intermediate_size];
                    let runtime_output =
                        &runtime.values[runtime_output_start..runtime_output_start + hidden_size];

                    let mut row_input = StageDelta::default();
                    let mut row_gpu_w13 = StageDelta::default();
                    let mut row_gpu_activation = StageDelta::default();
                    let mut row_gpu_w2 = StageDelta::default();
                    let mut row_reference_w13 = StageDelta::default();
                    let mut row_reference_activation = StageDelta::default();
                    let mut row_reference_w2 = StageDelta::default();
                    let mut row_reference_branch = StageDelta::default();
                    let mut row_oracle_branch = StageDelta::default();
                    let mut row_gpu_branch = StageDelta::default();

                    row_input.add_slices(runtime_input, input_bf16);
                    row_gpu_w13.add_slices(
                        runtime_w13,
                        &oracle.w13_preactivation[compact_w13_range.clone()],
                    );
                    row_gpu_activation.add_slices(
                        runtime_activation,
                        &oracle.activation[compact_w13_range.clone()],
                    );
                    row_gpu_w2
                        .add_slices(runtime_output, &oracle.values[compact_input_range.clone()]);
                    row_reference_w13.add_slices(
                        &reference.w13_preactivation[compact_w13_range.clone()],
                        &oracle.w13_preactivation[compact_w13_range.clone()],
                    );
                    row_reference_activation.add_slices(
                        &reference.activation[compact_w13_range.clone()],
                        &oracle.activation[compact_w13_range.clone()],
                    );
                    row_reference_w2.add_slices(
                        &reference.values[compact_input_range.clone()],
                        &oracle.values[compact_input_range.clone()],
                    );
                    row_reference_branch.add_slices(
                        &reference.values[compact_input_range.clone()],
                        captured_branch,
                    );
                    row_oracle_branch
                        .add_slices(&oracle.values[compact_input_range.clone()], captured_branch);
                    row_gpu_branch.add_slices(runtime_output, captured_branch);

                    input_f32_bf16.add_slices(runtime_input, input_bf16);
                    gpu_oracle_w13.add_slices(
                        runtime_w13,
                        &oracle.w13_preactivation[compact_w13_range.clone()],
                    );
                    gpu_oracle_activation.add_slices(
                        runtime_activation,
                        &oracle.activation[compact_w13_range.clone()],
                    );
                    gpu_oracle_w2
                        .add_slices(runtime_output, &oracle.values[compact_input_range.clone()]);
                    reference_oracle_w13.add_slices(
                        &reference.w13_preactivation[compact_w13_range.clone()],
                        &oracle.w13_preactivation[compact_w13_range.clone()],
                    );
                    reference_oracle_activation.add_slices(
                        &reference.activation[compact_w13_range.clone()],
                        &oracle.activation[compact_w13_range.clone()],
                    );
                    reference_oracle_w2.add_slices(
                        &reference.values[compact_input_range.clone()],
                        &oracle.values[compact_input_range.clone()],
                    );
                    reference_branch.add_slices(
                        &reference.values[compact_input_range.clone()],
                        captured_branch,
                    );
                    oracle_branch
                        .add_slices(&oracle.values[compact_input_range.clone()], captured_branch);
                    gpu_branch.add_slices(runtime_output, captured_branch);

                    for value in runtime_w13
                        .iter()
                        .chain(runtime_activation.iter())
                        .chain(runtime_output.iter())
                    {
                        assert!(
                            value.is_finite(),
                            "runtime-shaped GPU value is not finite for case={case_id} expert={expert} sorted_row={sorted_row}"
                        );
                    }

                    let input_hash = {
                        let mut bytes = Vec::with_capacity(runtime_input.len() * 4);
                        for &value in runtime_input {
                            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                        }
                        format!("0x{:016x}", fnv1a64(&bytes))
                    };
                    row_lines.push(format!(
                        "{case_id}\t{expert}\t{compact_row}\t{sorted_row}\t{input_hash}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}\t{:.18e}",
                        row_input.sum_abs,
                        row_input.max_abs,
                        row_gpu_w13.sum_abs,
                        row_gpu_w13.max_abs,
                        row_gpu_activation.sum_abs,
                        row_gpu_activation.max_abs,
                        row_gpu_w2.sum_abs,
                        row_gpu_w2.max_abs,
                        row_reference_w13.sum_abs,
                        row_reference_w13.max_abs,
                        row_reference_activation.sum_abs,
                        row_reference_activation.max_abs,
                        row_reference_w2.sum_abs,
                        row_reference_w2.max_abs,
                        row_reference_branch.sum_abs,
                        row_reference_branch.max_abs,
                        row_reference_branch.l2(),
                        row_oracle_branch.sum_abs,
                        row_oracle_branch.max_abs,
                        row_oracle_branch.l2(),
                        row_gpu_branch.sum_abs,
                        row_gpu_branch.max_abs,
                        row_gpu_branch.l2()
                    ));
                }
            }

            total_cases += 1;
            total_plan_entries += runtime_blocks.len();
            total_claimed_rows += case_row_count;
            total_runtime_rows += total_sorted_rows;
            total_padding_rows += total_sorted_rows - case_row_count;
            total_output_values += case_row_count * hidden_size;
            total_output_buffer_values += runtime.values.len();
        }

        assert_eq!(total_cases, 3);
        assert_eq!(total_plan_entries, 18);
        assert_eq!(total_claimed_rows, 385);
        assert_eq!(captured_routed_block_vectors, 385);
        assert_eq!(captured_branch_block_vectors, 385);
        assert!(padding_w13_untouched);
        assert!(padding_activation_untouched);
        assert!(padding_output_untouched);
        assert_eq!(gpu_oracle_w13.count, 385 * intermediate_size);
        assert_eq!(gpu_oracle_activation.count, 385 * intermediate_size);
        assert_eq!(gpu_oracle_w2.count, 385 * hidden_size);

        let first_gpu_oracle_stage = first_nonzero_stage(&[
            (
                "HQQ W13 GEMM accumulation/output cast",
                gpu_oracle_w13.max_abs,
            ),
            ("activation", gpu_oracle_activation.max_abs),
            (
                "W2 GEMM/output cast or runtime output indexing",
                gpu_oracle_w2.max_abs,
            ),
        ]);
        let first_cpu_bf16_stage = first_nonzero_stage(&[
            ("input BF16 conversion", input_f32_bf16.max_abs),
            (
                "HQQ W13 GEMM accumulation/output cast",
                reference_oracle_w13.max_abs,
            ),
            ("activation", reference_oracle_activation.max_abs),
            ("W2 GEMM/output cast", reference_oracle_w2.max_abs),
        ]);
        let first_gpu_oracle_contract_violation_stage = first_stage_over_tolerance(&[
            (
                "HQQ W13 GEMM accumulation/output cast sum",
                gpu_oracle_w13.sum_abs,
                gpu_oracle_w13_sum_tolerance,
            ),
            (
                "HQQ W13 GEMM accumulation/output cast max",
                gpu_oracle_w13.max_abs,
                gpu_oracle_w13_max_tolerance,
            ),
            (
                "activation sum",
                gpu_oracle_activation.sum_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "activation max",
                gpu_oracle_activation.max_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "W2 GEMM/output cast or runtime output indexing sum",
                gpu_oracle_w2.sum_abs,
                gpu_oracle_exact_tolerance,
            ),
            (
                "W2 GEMM/output cast or runtime output indexing max",
                gpu_oracle_w2.max_abs,
                gpu_oracle_exact_tolerance,
            ),
        ]);
        let gpu_matches_bf16_path_oracle_contract = gpu_oracle_w13.sum_abs
            <= gpu_oracle_w13_sum_tolerance
            && gpu_oracle_w13.max_abs <= gpu_oracle_w13_max_tolerance
            && gpu_oracle_activation.sum_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_activation.max_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_w2.sum_abs <= gpu_oracle_exact_tolerance
            && gpu_oracle_w2.max_abs <= gpu_oracle_exact_tolerance;

        write_lines(
            &artifact_dir.join(format!(
                "{label}_runtime_shaped_descriptor_plan_validation.tsv"
            )),
            &descriptor_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_runtime_shaped_row_compare.tsv")),
            &row_lines,
        );
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "cache_source_gate\t0836".to_string(),
            "capture_source_gate\t1023".to_string(),
            "gpu_prototype_source_gate\t1236".to_string(),
            "bf16_oracle_contract_source_gate\t1347".to_string(),
            "full_block_source_gate\t1412".to_string(),
            format!("cases\t{total_cases}"),
            format!("plan_entries\t{total_plan_entries}"),
            format!("claimed_rows\t{total_claimed_rows}"),
            format!("runtime_total_sorted_rows\t{total_runtime_rows}"),
            format!("runtime_padding_rows\t{total_padding_rows}"),
            format!("output_values\t{total_output_values}"),
            format!("runtime_output_buffer_values\t{total_output_buffer_values}"),
            format!("w13_values\t{}", gpu_oracle_w13.count),
            format!("activation_values\t{}", gpu_oracle_activation.count),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            "runtime_input_layout\tabsolute_sorted_rows_by_routed_hidden".to_string(),
            "runtime_w13_output_layout\tabsolute_sorted_rows_by_w13_rows".to_string(),
            "runtime_activation_output_layout\tabsolute_sorted_rows_by_intermediate".to_string(),
            "runtime_w2_output_layout\tabsolute_sorted_rows_by_routed_hidden".to_string(),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            format!("captured_routed_block_vectors\t{captured_routed_block_vectors}"),
            format!("captured_branch_block_vectors\t{captured_branch_block_vectors}"),
            format!("input_f32_bf16_sum_abs\t{:.18e}", input_f32_bf16.sum_abs),
            format!("input_f32_bf16_max_abs\t{:.18e}", input_f32_bf16.max_abs),
            format!("gpu_oracle_w13_sum_abs\t{:.18e}", gpu_oracle_w13.sum_abs),
            format!("gpu_oracle_w13_max_abs\t{:.18e}", gpu_oracle_w13.max_abs),
            format!(
                "gpu_oracle_activation_sum_abs\t{:.18e}",
                gpu_oracle_activation.sum_abs
            ),
            format!(
                "gpu_oracle_activation_max_abs\t{:.18e}",
                gpu_oracle_activation.max_abs
            ),
            format!("gpu_oracle_w2_sum_abs\t{:.18e}", gpu_oracle_w2.sum_abs),
            format!("gpu_oracle_w2_max_abs\t{:.18e}", gpu_oracle_w2.max_abs),
            format!(
                "reference_oracle_w13_sum_abs\t{:.18e}",
                reference_oracle_w13.sum_abs
            ),
            format!(
                "reference_oracle_w13_max_abs\t{:.18e}",
                reference_oracle_w13.max_abs
            ),
            format!(
                "reference_oracle_activation_sum_abs\t{:.18e}",
                reference_oracle_activation.sum_abs
            ),
            format!(
                "reference_oracle_activation_max_abs\t{:.18e}",
                reference_oracle_activation.max_abs
            ),
            format!(
                "reference_oracle_w2_sum_abs\t{:.18e}",
                reference_oracle_w2.sum_abs
            ),
            format!(
                "reference_oracle_w2_max_abs\t{:.18e}",
                reference_oracle_w2.max_abs
            ),
            format!(
                "reference_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                reference_branch.sum_abs
            ),
            format!(
                "reference_branch_vs_bf16_captured_max_abs\t{:.18e}",
                reference_branch.max_abs
            ),
            format!(
                "reference_branch_vs_bf16_captured_l2\t{:.18e}",
                reference_branch.l2()
            ),
            format!(
                "oracle_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                oracle_branch.sum_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_max_abs\t{:.18e}",
                oracle_branch.max_abs
            ),
            format!(
                "oracle_branch_vs_bf16_captured_l2\t{:.18e}",
                oracle_branch.l2()
            ),
            format!(
                "gpu_branch_vs_bf16_captured_sum_abs\t{:.18e}",
                gpu_branch.sum_abs
            ),
            format!(
                "gpu_branch_vs_bf16_captured_max_abs\t{:.18e}",
                gpu_branch.max_abs
            ),
            format!("gpu_branch_vs_bf16_captured_l2\t{:.18e}", gpu_branch.l2()),
            format!(
                "prior_1412_gpu_oracle_w13_sum_abs\t{full_block_gpu_w13_sum:.18e}"
            ),
            format!(
                "prior_1412_gpu_oracle_w13_max_abs\t{full_block_gpu_w13_max:.18e}"
            ),
            format!(
                "prior_1412_reference_oracle_w2_sum_abs\t{full_block_reference_w2_sum:.18e}"
            ),
            "correctness_oracle\tbf16_path_oracle".to_string(),
            "cpu_f32_reference_role\tdiagnostic_context".to_string(),
            "captured_bf16_branch_metric_role\truntime_shaped_diagnostic_context".to_string(),
            format!(
                "gpu_oracle_w13_sum_tolerance\t{gpu_oracle_w13_sum_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_w13_max_tolerance\t{gpu_oracle_w13_max_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_activation_sum_tolerance\t{gpu_oracle_exact_tolerance:.18e}"
            ),
            format!(
                "gpu_oracle_activation_max_tolerance\t{gpu_oracle_exact_tolerance:.18e}"
            ),
            format!("gpu_oracle_w2_sum_tolerance\t{gpu_oracle_exact_tolerance:.18e}"),
            format!("gpu_oracle_w2_max_tolerance\t{gpu_oracle_exact_tolerance:.18e}"),
            format!("gpu_oracle_final_sum_abs\t{:.18e}", gpu_oracle_w2.sum_abs),
            format!("first_gpu_oracle_divergent_stage\t{first_gpu_oracle_stage}"),
            format!(
                "first_gpu_oracle_contract_violation_stage\t{first_gpu_oracle_contract_violation_stage}"
            ),
            format!("first_cpu_reference_bf16_path_divergent_stage\t{first_cpu_bf16_stage}"),
            format!(
                "gpu_matches_bf16_path_oracle_contract\t{gpu_matches_bf16_path_oracle_contract}"
            ),
            format!("padding_w13_untouched\t{padding_w13_untouched}"),
            format!("padding_activation_untouched\t{padding_activation_untouched}"),
            format!("padding_output_untouched\t{padding_output_untouched}"),
            "runtime_prefill_consumer_added\tfalse".to_string(),
            "config_knob_added\tfalse".to_string(),
            "auto_selection_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_to_marlin_added\tfalse".to_string(),
            "speed_benchmark\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_runtime_shaped_replay_summary.tsv")),
            &summary_lines,
        );

        assert!(
            gpu_matches_bf16_path_oracle_contract,
            "runtime-shaped GPU prototype violates BF16-path oracle contract: W13 sum/max {:.18e}/{:.18e}, activation sum/max {:.18e}/{:.18e}, W2 sum/max {:.18e}/{:.18e}; first violation {first_gpu_oracle_contract_violation_stage}",
            gpu_oracle_w13.sum_abs,
            gpu_oracle_w13.max_abs,
            gpu_oracle_activation.sum_abs,
            gpu_oracle_activation.max_abs,
            gpu_oracle_w2.sum_abs,
            gpu_oracle_w2.max_abs
        );
        assert!(padding_w13_untouched);
        assert!(padding_activation_untouched);
        assert!(padding_output_untouched);
    }

    #[test]
    fn expert_hqq_payload_byte_counts_match_hqq4_hqq6_layouts() {
        assert_eq!(
            expert_hqq_component_sizes(3, 5, 4, 4).unwrap(),
            (12, 24, 24)
        );
        assert_eq!(
            expert_hqq_component_sizes(3, 5, 6, 4).unwrap(),
            (18, 24, 24)
        );
        assert_eq!(
            expert_hqq_component_sizes(3, 5, 8, 4).unwrap(),
            (24, 24, 24)
        );

        let err = ExpertHqqTensorInput::new(
            ExpertHqqTensorRole::W13,
            0,
            0,
            4,
            8,
            6,
            4,
            vec![0u8; 23],
            vec![0u8; 32],
            vec![0u8; 32],
        )
        .expect_err("bad packed payload length must fail closed");
        assert!(err.contains("packed length"), "{err}");
    }

    #[test]
    fn expert_hqq_writer_rejects_role_shape_mismatch() {
        let desc =
            ExpertHqqTensorDescriptor::new(ExpertHqqTensorRole::W2, 0, 0, 4, 8, 6, 4).unwrap();
        let input = ExpertHqqTensorInput::new(
            ExpertHqqTensorRole::W2,
            0,
            0,
            4,
            8,
            6,
            4,
            vec![0u8; desc.packed_bytes],
            vec![0u8; desc.scales_bytes],
            vec![0u8; desc.zeros_bytes],
        )
        .unwrap();
        let err = ExpertHqqCache::from_inputs(sample_header(1), vec![input])
            .expect_err("W2 with W13-shaped rows/cols must fail closed");
        assert!(err.contains("W2 cols") || err.contains("W2 rows"), "{err}");
    }

    #[test]
    fn expert_hqq_writer_rejects_nbits_group_axis_mismatch() {
        let mut axis_bad = sample_input(ExpertHqqTensorRole::W13, 0, 0, 6);
        axis_bad.descriptor.axis = 0;
        let err = ExpertHqqCache::from_inputs(sample_header(1), vec![axis_bad])
            .expect_err("axis mismatch must fail closed");
        assert!(err.contains("axis"), "{err}");

        let mut nbits_bad = sample_input(ExpertHqqTensorRole::W13, 0, 0, 6);
        nbits_bad.descriptor.nbits = 5;
        let err = ExpertHqqCache::from_inputs(sample_header(1), vec![nbits_bad])
            .expect_err("nbits mismatch must fail closed");
        assert!(err.contains("Unsupported expert-HQQ nbits"), "{err}");

        let mut group_bad = sample_input(ExpertHqqTensorRole::W13, 0, 0, 6);
        group_bad.descriptor.group_size = 8;
        let err = ExpertHqqCache::from_inputs(sample_header(1), vec![group_bad])
            .expect_err("group-size byte-count mismatch must fail closed");
        assert!(err.contains("component bytes mismatch"), "{err}");
    }

    #[test]
    fn expert_hqq_cache_rejects_header_mismatch() {
        let cache = ExpertHqqCache::new(
            sample_header(2),
            vec![
                sample_record(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_record(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let path = temp_path("expert_hqq_header_mismatch");
        cache.write_to_path(&path).unwrap();
        let mut expected = cache.header.expectation();
        expected.routed_hidden_size += 1;
        let err = ExpertHqqCache::read_from_path_with_expected(&path, &expected)
            .expect_err("mismatched header must fail closed");
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("routed_hidden_size"), "{err}");
    }

    #[test]
    fn expert_hqq_cache_rejects_corrupt_tensor_metadata() {
        let cache = ExpertHqqCache::new(
            sample_header(2),
            vec![
                sample_record(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_record(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let path = temp_path("expert_hqq_bad_tensor_metadata");
        cache.write_to_path(&path).unwrap();
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start((EXPERT_HQQ_HEADER_SIZE + 4) as u64))
            .unwrap();
        file.write_all(&5u32.to_le_bytes()).unwrap();
        drop(file);

        let err = ExpertHqqCache::read_from_path_with_expected(&path, &cache.header.expectation())
            .expect_err("unsupported nbits must fail closed");
        std::fs::remove_file(&path).unwrap();
        assert!(
            err.contains("layout code") || err.contains("Unsupported expert-HQQ nbits"),
            "{err}"
        );
    }

    #[test]
    fn expert_hqq_cache_rejects_duplicate_projection_descriptor() {
        let err = ExpertHqqCache::new(
            sample_header(2),
            vec![
                sample_record(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_record(ExpertHqqTensorRole::W13, 0, 1, 6),
            ],
        )
        .expect_err("duplicate layer/expert/role descriptors must fail closed");
        assert!(err.contains("duplicate expert-HQQ descriptor"), "{err}");
    }

    #[test]
    fn expert_hqq_weight_store_registers_cache_and_looks_up_projection() {
        let mut store = sample_weight_store();
        let path = temp_path("expert_hqq_weight_store_register");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let required = [
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1),
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1),
        ];
        store
            .register_expert_hqq_cache_from_path(&path, cache.header.config_hash, &required)
            .unwrap();
        std::fs::remove_file(&path).unwrap();

        let w13 = store
            .require_expert_hqq_tensor(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1))
            .unwrap();
        assert_eq!(w13.descriptor.role, ExpertHqqTensorRole::W13);
        assert_eq!(w13.descriptor.layer_idx, 0);
        assert_eq!(w13.descriptor.expert_idx, 1);
        assert_eq!(w13.descriptor.nbits, 6);

        let missing = store
            .require_expert_hqq_tensor(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 1, 1))
            .expect_err("unregistered projection lookup must fail closed");
        assert!(
            missing.contains("missing required expert-HQQ descriptor"),
            "{missing}"
        );
    }

    #[test]
    fn expert_hqq_weight_store_registration_rejects_missing_descriptor_without_marlin_fallback() {
        let mut store = sample_weight_store();
        let path = temp_path("expert_hqq_weight_store_missing_descriptor");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(1),
            vec![sample_input(ExpertHqqTensorRole::W13, 0, 1, 6)],
        )
        .unwrap();
        let required = [
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1),
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1),
        ];
        let err = store
            .register_expert_hqq_cache_from_path(&path, cache.header.config_hash, &required)
            .expect_err("missing W2 descriptor must fail closed");
        std::fs::remove_file(&path).unwrap();
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
        assert!(
            store.expert_hqq_cache.is_none(),
            "failed expert-HQQ registration must not attach metadata or fall back to Marlin"
        );
    }

    #[test]
    fn expert_hqq_weight_store_registration_rejects_model_shape_mismatch() {
        let mut store = sample_weight_store();
        store.config.hidden_size += 1;
        let path = temp_path("expert_hqq_weight_store_shape_mismatch");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let required = [ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1)];
        let err = store
            .register_expert_hqq_cache_from_path(&path, cache.header.config_hash, &required)
            .expect_err("model-shape mismatch must fail closed");
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("hidden_size"), "{err}");
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_weight_store_registration_rejects_duplicate_descriptor_cache() {
        let mut store = sample_weight_store();
        let path = temp_path("expert_hqq_weight_store_duplicate_descriptor");
        let header = sample_header(2);
        let record = sample_record(ExpertHqqTensorRole::W13, 0, 1, 6);
        {
            let mut file = std::fs::File::create(&path).unwrap();
            write_header(&mut file, &header).unwrap();
            write_descriptor(&mut file, &record.descriptor).unwrap();
            write_descriptor(&mut file, &record.descriptor).unwrap();
            file.write_all(&record.packed).unwrap();
            file.write_all(&record.scales).unwrap();
            file.write_all(&record.zeros).unwrap();
            file.write_all(&record.packed).unwrap();
            file.write_all(&record.scales).unwrap();
            file.write_all(&record.zeros).unwrap();
        }
        let required = [ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1)];
        let err = store
            .register_expert_hqq_cache_from_path(&path, header.config_hash, &required)
            .expect_err("duplicate descriptor cache must fail closed");
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("duplicate expert-HQQ descriptor"), "{err}");
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_weight_store_registration_rejects_implicit_empty_requirement_set() {
        let mut store = sample_weight_store();
        let path = temp_path("expert_hqq_weight_store_empty_requirement_set");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let err = store
            .register_expert_hqq_cache_from_path(&path, cache.header.config_hash, &[])
            .expect_err("implicit cache registration must fail closed");
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("explicit descriptor requirements"), "{err}");
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_diagnostic_spec_registration_registers_valid_cache() {
        let mut store = sample_weight_store();
        let cache_path = temp_path("expert_hqq_diag_spec_valid_cache");
        let spec_path = temp_path("expert_hqq_diag_spec_valid_spec");
        let cache = write_expert_hqq_cache_from_inputs(
            &cache_path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let cache_file_name = cache_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        write_diagnostic_cache_spec(&spec_path, &cache_file_name, 0, &[1], &["w13", "w2"], 6, 4);

        store
            .register_expert_hqq_diagnostic_cache_from_spec_path(
                &spec_path,
                cache.header.config_hash,
            )
            .unwrap();
        let w2 = store
            .require_expert_hqq_tensor(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1))
            .unwrap();
        assert_eq!(w2.descriptor.nbits, 6);
        assert_eq!(w2.descriptor.group_size, 4);

        std::fs::remove_file(&cache_path).unwrap();
        std::fs::remove_file(&spec_path).unwrap();
    }

    #[test]
    fn expert_hqq_diagnostic_spec_registration_rejects_malformed_or_missing_spec() {
        let mut store = sample_weight_store();
        let missing = temp_path("expert_hqq_diag_spec_missing");
        let err = store
            .register_expert_hqq_diagnostic_cache_from_spec_path(&missing, 0x1234_5678_9abc_def0)
            .expect_err("missing spec file must fail closed");
        assert!(err.contains("failed to resolve"), "{err}");
        assert!(store.expert_hqq_cache.is_none());

        let malformed = temp_path("expert_hqq_diag_spec_malformed");
        std::fs::write(&malformed, "{").unwrap();
        let err = store
            .register_expert_hqq_diagnostic_cache_from_spec_path(&malformed, 0x1234_5678_9abc_def0)
            .expect_err("malformed JSON spec must fail closed");
        std::fs::remove_file(&malformed).unwrap();
        assert!(
            err.contains("malformed expert-HQQ diagnostic cache spec"),
            "{err}"
        );
        assert!(store.expert_hqq_cache.is_none());

        let missing_cache_spec = temp_path("expert_hqq_diag_spec_missing_cache");
        let missing_cache = temp_path("expert_hqq_diag_spec_missing_cache_file");
        write_diagnostic_cache_spec(
            &missing_cache_spec,
            &missing_cache.to_string_lossy(),
            0,
            &[1],
            &["w13", "w2"],
            6,
            4,
        );
        let err = store
            .register_expert_hqq_diagnostic_cache_from_spec_path(
                &missing_cache_spec,
                0x1234_5678_9abc_def0,
            )
            .expect_err("missing cache file must fail closed");
        std::fs::remove_file(&missing_cache_spec).unwrap();
        assert!(
            err.contains("failed to resolve expert-HQQ diagnostic cache_path"),
            "{err}"
        );
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_diagnostic_spec_registration_rejects_wrong_config_hash() {
        let mut store = sample_weight_store();
        let cache_path = temp_path("expert_hqq_diag_spec_hash_cache");
        let spec_path = temp_path("expert_hqq_diag_spec_hash_spec");
        let cache = write_expert_hqq_cache_from_inputs(
            &cache_path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        write_diagnostic_cache_spec(
            &spec_path,
            &cache_path.to_string_lossy(),
            0,
            &[1],
            &["w13", "w2"],
            6,
            4,
        );

        let err = store
            .register_expert_hqq_diagnostic_cache_from_spec_path(
                &spec_path,
                cache.header.config_hash ^ 1,
            )
            .expect_err("wrong config hash must fail closed");
        std::fs::remove_file(&cache_path).unwrap();
        std::fs::remove_file(&spec_path).unwrap();
        assert!(err.contains("config_hash"), "{err}");
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_diagnostic_spec_registration_rejects_incomplete_w13_w2_pairs() {
        let mut store = sample_weight_store();
        let cache_path = temp_path("expert_hqq_diag_spec_missing_pair_cache");
        let spec_path = temp_path("expert_hqq_diag_spec_missing_pair_spec");
        let cache = write_expert_hqq_cache_from_inputs(
            &cache_path,
            sample_header(1),
            vec![sample_input(ExpertHqqTensorRole::W13, 0, 1, 6)],
        )
        .unwrap();
        write_diagnostic_cache_spec(
            &spec_path,
            &cache_path.to_string_lossy(),
            0,
            &[1],
            &["w13", "w2"],
            6,
            4,
        );

        let err = store
            .register_expert_hqq_diagnostic_cache_from_spec_path(
                &spec_path,
                cache.header.config_hash,
            )
            .expect_err("cache missing W2 must fail closed");
        std::fs::remove_file(&cache_path).unwrap();
        std::fs::remove_file(&spec_path).unwrap();
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_diagnostic_spec_registration_rejects_nbits_group_or_layout_mismatch() {
        for (name, nbits, group_size, expected) in [
            ("nbits", 4u8, 4usize, "nbits mismatch"),
            ("group", 6u8, 8usize, "group_size mismatch"),
        ] {
            let mut store = sample_weight_store();
            let cache_path = temp_path(&format!("expert_hqq_diag_spec_{name}_cache"));
            let spec_path = temp_path(&format!("expert_hqq_diag_spec_{name}_spec"));
            let cache = write_expert_hqq_cache_from_inputs(
                &cache_path,
                sample_header(2),
                vec![
                    sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                    sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
                ],
            )
            .unwrap();
            write_diagnostic_cache_spec(
                &spec_path,
                &cache_path.to_string_lossy(),
                0,
                &[1],
                &["w13", "w2"],
                nbits,
                group_size,
            );
            let err = store
                .register_expert_hqq_diagnostic_cache_from_spec_path(
                    &spec_path,
                    cache.header.config_hash,
                )
                .expect_err("spec/cache metadata mismatch must fail closed");
            std::fs::remove_file(&cache_path).unwrap();
            std::fs::remove_file(&spec_path).unwrap();
            assert!(err.contains(expected), "{err}");
            assert!(store.expert_hqq_cache.is_none());
        }

        let mut store = sample_weight_store();
        let cache_path = temp_path("expert_hqq_diag_spec_layout_cache");
        let spec_path = temp_path("expert_hqq_diag_spec_layout_spec");
        let cache = write_expert_hqq_cache_from_inputs(
            &cache_path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        {
            let mut file = OpenOptions::new().write(true).open(&cache_path).unwrap();
            file.seek(SeekFrom::Start((EXPERT_HQQ_HEADER_SIZE + 56) as u64))
                .unwrap();
            file.write_all(&4u32.to_le_bytes()).unwrap();
        }
        write_diagnostic_cache_spec(
            &spec_path,
            &cache_path.to_string_lossy(),
            0,
            &[1],
            &["w13", "w2"],
            6,
            4,
        );
        let err = store
            .register_expert_hqq_diagnostic_cache_from_spec_path(
                &spec_path,
                cache.header.config_hash,
            )
            .expect_err("layout mismatch must fail closed");
        std::fs::remove_file(&cache_path).unwrap();
        std::fs::remove_file(&spec_path).unwrap();
        assert!(
            err.contains("layout code") || err.contains("layout"),
            "{err}"
        );
        assert!(store.expert_hqq_cache.is_none());
    }

    #[test]
    fn expert_hqq_prefill_dispatch_plan_maps_registered_selected_experts() {
        let mut store = sample_weight_store();
        store.config.experts_gated = false;
        let path = temp_path("expert_hqq_prefill_dispatch_contract");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(4),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W13, 0, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 2, 6),
            ],
        )
        .unwrap();
        let required = [
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1),
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1),
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 2),
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 2),
        ];
        store
            .register_expert_hqq_cache_from_path(&path, cache.header.config_hash, &required)
            .unwrap();
        std::fs::remove_file(&path).unwrap();

        let works = [
            ExpertHqqPrefillWork::new(1, 0, 2),
            ExpertHqqPrefillWork::new(2, 64, 1),
        ];
        let plan = prefill_dispatch_plan_from_registered_cache(
            store.expert_hqq_cache.as_ref(),
            0,
            store.config.experts_gated,
            &works,
        )
        .unwrap();
        assert_eq!(plan.layer_idx, 0);
        assert!(!plan.experts_gated);
        assert_eq!(
            plan.input_layout,
            "row_major_selected_rows_by_routed_hidden"
        );
        assert_eq!(
            plan.w13_dequant_layout,
            "row_major_axis1_grouped_rows_by_routed_hidden"
        );
        assert_eq!(
            plan.w13_output_layout,
            "row_major_selected_rows_by_w13_rows"
        );
        assert_eq!(
            plan.activation_output_layout,
            "row_major_selected_rows_by_moe_intermediate"
        );
        assert_eq!(
            plan.w2_dequant_layout,
            "row_major_axis1_grouped_routed_hidden_by_moe_intermediate"
        );
        assert_eq!(
            plan.w2_output_layout,
            "row_major_selected_rows_by_routed_hidden"
        );
        assert_eq!(plan.entries.len(), 2);
        assert_eq!(plan.entries[0].expert_idx, 1);
        assert_eq!(plan.entries[0].row_offset, 0);
        assert_eq!(plan.entries[0].row_count, 2);
        assert_eq!(
            plan.entries[0].w13_key,
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1)
        );
        assert_eq!(
            plan.entries[0].w2_key,
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1)
        );
        assert_eq!((plan.entries[0].w13_rows, plan.entries[0].w13_cols), (4, 8));
        assert_eq!((plan.entries[0].w2_rows, plan.entries[0].w2_cols), (8, 4));
        assert_eq!(plan.entries[0].w13_nbits, 6);
        assert_eq!(plan.entries[0].w2_nbits, 6);
        assert_eq!(plan.entries[0].w13_group_size, 4);
        assert_eq!(plan.entries[0].w2_group_size, 4);
        assert_eq!(plan.entries[1].expert_idx, 2);
        assert_eq!(plan.entries[1].row_offset, 64);
        assert_eq!(plan.entries[1].row_count, 1);
    }

    #[test]
    fn expert_hqq_prefill_dispatch_fails_closed_without_registered_metadata() {
        let store = sample_weight_store();
        let err = prefill_dispatch_plan_from_registered_cache(
            store.expert_hqq_cache.as_ref(),
            0,
            false,
            &[ExpertHqqPrefillWork::new(1, 0, 1)],
        )
        .expect_err("missing registered KRHQ metadata must fail closed");
        assert!(err.contains("not registered"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_dispatch_fails_closed_on_incomplete_projection_pair() {
        let mut store = sample_weight_store();
        store.config.experts_gated = false;
        let path = temp_path("expert_hqq_prefill_dispatch_missing_w2");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(1),
            vec![sample_input(ExpertHqqTensorRole::W13, 0, 1, 6)],
        )
        .unwrap();
        store
            .register_expert_hqq_cache_from_path(
                &path,
                cache.header.config_hash,
                &[ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1)],
            )
            .unwrap();
        std::fs::remove_file(&path).unwrap();

        let err = prefill_dispatch_plan_from_registered_cache(
            store.expert_hqq_cache.as_ref(),
            0,
            store.config.experts_gated,
            &[ExpertHqqPrefillWork::new(1, 0, 1)],
        )
        .expect_err("registered W13 without W2 must fail closed before dispatch");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
    }

    #[test]
    fn expert_hqq_prefill_dispatch_fails_closed_on_gated_shape_mismatch() {
        let mut store = sample_weight_store();
        store.config.experts_gated = true;
        let path = temp_path("expert_hqq_prefill_dispatch_gated_mismatch");
        let cache = write_expert_hqq_cache_from_inputs(
            &path,
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
            ],
        )
        .unwrap();
        let required = [
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1),
            ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1),
        ];
        store
            .register_expert_hqq_cache_from_path(&path, cache.header.config_hash, &required)
            .unwrap();
        std::fs::remove_file(&path).unwrap();

        let err = prefill_dispatch_plan_from_registered_cache(
            store.expert_hqq_cache.as_ref(),
            0,
            store.config.experts_gated,
            &[ExpertHqqPrefillWork::new(1, 0, 1)],
        )
        .expect_err("ungated W13 metadata must not satisfy gated prefill dispatch");
        assert!(err.contains("W13 rows"), "{err}");
        assert!(err.contains("experts_gated=true"), "{err}");
    }

    #[test]
    fn expert_hqq_runtime_diagnostic_locates_registered_w13_w2_payloads() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let report = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 6, 4)],
        )
        .unwrap();
        assert_eq!(report.checked_experts, 1);
        assert_eq!(report.tensor_records, 2);
        assert_eq!(report.tensors.len(), 2);
        assert!(report.total_payload_bytes > 0);
        assert_eq!(report.tensors[0].role, ExpertHqqTensorRole::W13);
        assert_eq!(report.tensors[1].role, ExpertHqqTensorRole::W2);
        assert_eq!(report.tensors[0].layer_idx, 1);
        assert_eq!(report.tensors[0].expert_idx, 2);
        assert_eq!(report.tensors[0].nbits, 6);
        assert_eq!(report.tensors[0].group_size, 4);
    }

    #[test]
    fn expert_hqq_runtime_prefill_diagnostic_contract_validates_shape_and_oracle_metadata() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(4),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 1, 6),
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let model = sample_runtime_diagnostic_model(false);
        let blocks = [
            ExpertHqqRuntimePrefillBlock::new(1, 2, 2),
            ExpertHqqRuntimePrefillBlock::new(2, 5, 1),
        ];
        let shape = ExpertHqqRuntimePrefillBufferShape {
            total_sorted_rows: 8,
            input_row_stride: 10,
            w13_row_stride: 6,
            activation_row_stride: 5,
            output_row_stride: 11,
        };
        let lengths = ExpertHqqRuntimePrefillBufferLengths::required(model, shape).unwrap();
        let report = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&cache),
            model,
            1,
            6,
            4,
            &blocks,
            shape,
            lengths,
        )
        .unwrap();
        assert_eq!(report.plan_entries, 2);
        assert_eq!(report.claimed_rows, 3);
        assert_eq!(report.padding_rows, 5);
        assert_eq!(report.buffer_lengths.input_values, 78);
        assert_eq!(report.buffer_lengths.w13_values, 46);
        assert_eq!(report.buffer_lengths.activation_values, 39);
        assert_eq!(report.buffer_lengths.output_values, 85);
        assert_eq!(report.availability.checked_experts, 2);
        assert_eq!(report.availability.tensor_records, 4);
        assert_eq!(report.oracle.correctness_oracle, "bf16_path_oracle");
        assert_eq!(report.oracle.sorted_row_count, 3);
        assert_eq!(report.oracle.input_bf16_values, 24);
        assert_eq!(report.oracle.w13_preactivation_values, 12);
        assert_eq!(report.oracle.activation_values, 12);
        assert_eq!(report.oracle.output_values, 24);
    }

    #[test]
    fn expert_hqq_runtime_prefill_diagnostic_contract_accepts_hqq8_metadata() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 8),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 8),
            ],
        )
        .unwrap();
        let model = sample_runtime_diagnostic_model(false);
        let blocks = [ExpertHqqRuntimePrefillBlock::new(2, 1, 1)];
        let shape = ExpertHqqRuntimePrefillBufferShape::contiguous_for_cache(&cache, false, 4)
            .expect("runtime-shaped contiguous shape should build");
        let lengths = ExpertHqqRuntimePrefillBufferLengths::required(model, shape).unwrap();
        let report = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&cache),
            model,
            1,
            8,
            4,
            &blocks,
            shape,
            lengths,
        )
        .expect("HQQ8 runtime diagnostic metadata should validate");
        assert_eq!(report.availability.checked_experts, 1);
        assert_eq!(report.availability.tensor_records, 2);
        assert_eq!(report.plan_entries, 1);
        assert_eq!(report.oracle.correctness_oracle, "bf16_path_oracle");
    }

    #[test]
    fn expert_hqq_runtime_prefill_diagnostic_contract_fails_closed_on_shape_and_buffer_mismatch() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let model = sample_runtime_diagnostic_model(false);
        let blocks = [ExpertHqqRuntimePrefillBlock::new(2, 1, 1)];
        let shape = ExpertHqqRuntimePrefillBufferShape {
            total_sorted_rows: 4,
            input_row_stride: 8,
            w13_row_stride: 4,
            activation_row_stride: 4,
            output_row_stride: 8,
        };
        let lengths = ExpertHqqRuntimePrefillBufferLengths::required(model, shape).unwrap();

        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            None, model, 1, 6, 4, &blocks, shape, lengths,
        )
        .expect_err("absent runtime KRHQ cache must fail closed");
        assert!(err.contains("not registered"), "{err}");

        let mut bad_shape = shape;
        bad_shape.output_row_stride = 7;
        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&cache),
            model,
            1,
            6,
            4,
            &blocks,
            bad_shape,
            lengths,
        )
        .expect_err("output stride smaller than hidden must fail closed");
        assert!(err.contains("output_row_stride"), "{err}");

        let mut bad_lengths = lengths;
        bad_lengths.output_values += 1;
        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&cache),
            model,
            1,
            6,
            4,
            &blocks,
            shape,
            bad_lengths,
        )
        .expect_err("runtime buffer length mismatch must fail closed");
        assert!(err.contains("buffer length mismatch"), "{err}");

        let unsorted = [
            ExpertHqqRuntimePrefillBlock::new(2, 3, 1),
            ExpertHqqRuntimePrefillBlock::new(1, 2, 1),
        ];
        let two_expert_cache = ExpertHqqCache::from_inputs(
            sample_header(4),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 1, 6),
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&two_expert_cache),
            model,
            1,
            6,
            4,
            &unsorted,
            shape,
            lengths,
        )
        .expect_err("unsorted runtime blocks must fail closed");
        assert!(err.contains("sorted and non-overlapping"), "{err}");
    }

    #[test]
    fn expert_hqq_runtime_prefill_diagnostic_contract_fails_closed_on_metadata_mismatch() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let model = sample_runtime_diagnostic_model(false);
        let blocks = [ExpertHqqRuntimePrefillBlock::new(2, 1, 1)];
        let shape = ExpertHqqRuntimePrefillBufferShape::contiguous_for_cache(&cache, false, 4)
            .expect("runtime-shaped contiguous shape should build");
        let lengths = ExpertHqqRuntimePrefillBufferLengths::required(model, shape).unwrap();

        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&cache),
            model,
            1,
            4,
            4,
            &blocks,
            shape,
            lengths,
        )
        .expect_err("nbits mismatch must fail closed");
        assert!(err.contains("nbits mismatch"), "{err}");

        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&cache),
            model,
            1,
            6,
            8,
            &blocks,
            shape,
            lengths,
        )
        .expect_err("group-size mismatch must fail closed");
        assert!(err.contains("group_size mismatch"), "{err}");

        let w13_only = ExpertHqqCache::from_inputs(
            sample_header(1),
            vec![sample_input(ExpertHqqTensorRole::W13, 1, 2, 6)],
        )
        .unwrap();
        let err = validate_expert_hqq_runtime_prefill_diagnostic_contract(
            Some(&w13_only),
            model,
            1,
            6,
            4,
            &blocks,
            shape,
            lengths,
        )
        .expect_err("missing W2 role pair must fail closed");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
    }

    #[test]
    fn real_nano_krhq_runtime_diagnostic_locates_selected_layer1_payloads() {
        if env::var("KRASIS_REAL_NANO_KRHQ_RUNTIME_DIAGNOSTIC_AVAILABILITY_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano KRHQ runtime diagnostic proof; set KRASIS_REAL_NANO_KRHQ_RUNTIME_DIAGNOSTIC_AVAILABILITY_PROOF=1"
            );
            return;
        }
        let model_dir = PathBuf::from(
            env::var("KRASIS_REAL_NANO_MODEL_DIR")
                .expect("KRASIS_REAL_NANO_MODEL_DIR is required for runtime diagnostic proof"),
        );
        let artifact_dir =
            PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
                "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for runtime diagnostic proof",
            ));
        std::fs::create_dir_all(&artifact_dir).unwrap();
        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        let selected_experts = [26usize, 42, 47, 72, 88, 89, 112];
        let requirements: Vec<_> = selected_experts
            .iter()
            .copied()
            .map(|expert_idx| ExpertHqqRuntimeDiagnosticRequirement::new(1, expert_idx, 6, 64))
            .collect();
        let report = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            ExpertHqqRuntimeDiagnosticModelShape {
                hidden_size: expected.hidden_size,
                routed_hidden_size: expected.routed_hidden_size,
                moe_intermediate_size: expected.moe_intermediate_size,
                n_routed_experts: expected.n_routed_experts,
                num_hidden_layers: expected.num_moe_layers,
                experts_gated: false,
            },
            &requirements,
        )
        .unwrap();
        assert_eq!(report.checked_experts, selected_experts.len());
        assert_eq!(report.tensor_records, selected_experts.len() * 2);
        assert!(report.total_payload_bytes > 0);
        for &expert_idx in selected_experts.iter() {
            for role in [ExpertHqqTensorRole::W13, ExpertHqqTensorRole::W2] {
                assert!(
                    report.tensors.iter().any(|tensor| {
                        tensor.role == role
                            && tensor.layer_idx == 1
                            && tensor.expert_idx == expert_idx
                            && tensor.nbits == 6
                            && tensor.group_size == 64
                            && tensor.axis == 1
                    }),
                    "missing {role:?} layer=1 expert={expert_idx}"
                );
            }
        }
        let mut lines = vec![
            "field\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("selected_experts\t{:?}", selected_experts),
            format!("checked_experts\t{}", report.checked_experts),
            format!("tensor_records\t{}", report.tensor_records),
            format!("total_payload_bytes\t{}", report.total_payload_bytes),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "axis\t1".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            "gpu_kernels_launched\tfalse".to_string(),
            "output_comparison_added\tfalse".to_string(),
        ];
        for tensor in report.tensors.iter() {
            lines.push(format!(
                "tensor\trole={} layer={} expert={} rows={} cols={} packed={} scales={} zeros={}",
                tensor.role.as_str(),
                tensor.layer_idx,
                tensor.expert_idx,
                tensor.rows,
                tensor.cols,
                tensor.packed_bytes,
                tensor.scales_bytes,
                tensor.zeros_bytes
            ));
        }
        write_lines(
            &artifact_dir.join(
                "20260626_1508_nemotron_nano_runtime_krhq_metadata_cache_availability_real_nano_payload_availability.tsv",
            ),
            &lines,
        );
    }

    #[test]
    fn real_nano_krhq_runtime_prefill_contract_validates_full_blocks_without_gpu() {
        if env::var("KRASIS_REAL_NANO_KRHQ_RUNTIME_PREFILL_CONTRACT_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping real Nano KRHQ runtime prefill contract proof; set KRASIS_REAL_NANO_KRHQ_RUNTIME_PREFILL_CONTRACT_PROOF=1"
            );
            return;
        }
        let model_dir =
            PathBuf::from(env::var("KRASIS_REAL_NANO_MODEL_DIR").expect(
                "KRASIS_REAL_NANO_MODEL_DIR is required for runtime prefill contract proof",
            ));
        let artifact_dir = PathBuf::from(env::var("KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR").expect(
            "KRASIS_REAL_NANO_KRHQ_ARTIFACT_DIR is required for runtime prefill contract proof",
        ));
        std::fs::create_dir_all(&artifact_dir).unwrap();

        let cache_label = "20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation";
        let trace_label = "20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay";
        let label = "20260626_1537_nemotron_nano_bf16_oracle_runtime_diagnostic_api_surface";
        let cache_path = artifact_dir.join(format!("{cache_label}_hqq6_g64.krhq"));
        let trace_path = artifact_dir.join(format!(
            "{trace_label}_bf16_branch_replay_trace_outputs.json"
        ));

        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let hidden_size = config_usize(&config, "hidden_size");
        let intermediate_size = config_usize(&config, "moe_intermediate_size");
        let n_routed_experts = config_usize(&config, "n_routed_experts");
        let num_layers = config_usize(&config, "num_hidden_layers");
        let expected = ExpertHqqCacheExpectation {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_moe_layers: num_layers,
            config_hash: fnv1a64(&config_bytes),
        };
        let cache = load_expert_hqq_cache(&cache_path, &expected).unwrap();
        assert_eq!(cache.tensors.len(), 14);

        let routed_inputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_routed_input_full_expert",
            "runtime prefill contract routed input",
        );
        let branch_outputs = extract_bf16_full_vectors(
            &trace_path,
            "layer1_sequential_moe_bf16_branch_output_full_expert",
            "runtime prefill contract branch output",
        );
        assert_eq!(routed_inputs.len(), 385);
        assert_eq!(branch_outputs.len(), 385);
        for key in routed_inputs.keys() {
            assert!(
                branch_outputs.contains_key(key),
                "missing branch output coverage for runtime prefill contract key {:?}",
                key
            );
        }

        let model = ExpertHqqRuntimeDiagnosticModelShape {
            hidden_size,
            routed_hidden_size: hidden_size,
            moe_intermediate_size: intermediate_size,
            n_routed_experts,
            num_hidden_layers: num_layers,
            experts_gated: false,
        };
        let mut descriptor_lines = vec![
            "case_id\texpert\tabsolute_row_offset\trow_count\trow_end\ttotal_sorted_rows\tclaimed_rows\tpadding_rows\tinput_stride\tw13_stride\tactivation_stride\toutput_stride\tinput_values\tw13_values\tactivation_values\toutput_values\toracle_input_bf16_values\toracle_w13_values\toracle_activation_values\toracle_output_values".to_string(),
        ];
        let mut total_cases = 0usize;
        let mut total_plan_entries = 0usize;
        let mut total_claimed_rows = 0usize;
        let mut total_runtime_rows = 0usize;
        let mut total_padding_rows = 0usize;
        let mut total_runtime_input_values = 0usize;
        let mut total_runtime_w13_values = 0usize;
        let mut total_runtime_activation_values = 0usize;
        let mut total_runtime_output_values = 0usize;
        let mut total_oracle_input_values = 0usize;
        let mut total_oracle_w13_values = 0usize;
        let mut total_oracle_activation_values = 0usize;
        let mut total_oracle_output_values = 0usize;

        for case_id in case_ids() {
            let mut by_expert: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for ((row_case, expert, sorted_row), _) in &routed_inputs {
                if row_case == case_id {
                    by_expert.entry(*expert).or_default().push(*sorted_row);
                }
            }
            assert!(
                !by_expert.is_empty(),
                "missing runtime prefill contract routed inputs for case {case_id}"
            );
            let mut blocks_by_expert: Vec<(usize, Vec<usize>)> = by_expert.into_iter().collect();
            for (_, rows) in &mut blocks_by_expert {
                rows.sort_unstable();
                rows.dedup();
                assert!(!rows.is_empty());
                assert_eq!(
                    rows.len(),
                    rows[rows.len() - 1] - rows[0] + 1,
                    "runtime prefill contract rows for case={case_id} must be contiguous inside each expert block"
                );
            }
            blocks_by_expert.sort_by_key(|(_, rows)| rows[0]);
            let case_row_count: usize = blocks_by_expert.iter().map(|(_, rows)| rows.len()).sum();
            let max_row = blocks_by_expert
                .iter()
                .flat_map(|(_, rows)| rows.iter().copied())
                .max()
                .expect("runtime prefill contract rows should be non-empty");
            let total_sorted_rows = max_row + 1;
            let shape = ExpertHqqRuntimePrefillBufferShape::contiguous_for_cache(
                &cache,
                false,
                total_sorted_rows,
            )
            .unwrap();
            let lengths = ExpertHqqRuntimePrefillBufferLengths::required(model, shape).unwrap();
            let blocks: Vec<_> = blocks_by_expert
                .iter()
                .map(|(expert, rows)| {
                    ExpertHqqRuntimePrefillBlock::new(*expert, rows[0], rows.len())
                })
                .collect();
            let report = validate_expert_hqq_runtime_prefill_diagnostic_contract(
                Some(&cache),
                model,
                1,
                6,
                64,
                &blocks,
                shape,
                lengths,
            )
            .unwrap();
            assert_eq!(report.claimed_rows, case_row_count);
            assert_eq!(report.total_sorted_rows, total_sorted_rows);
            assert_eq!(report.plan_entries, blocks.len());
            assert_eq!(report.oracle.sorted_row_count, case_row_count);
            assert_eq!(report.oracle.correctness_oracle, "bf16_path_oracle");
            assert_eq!(
                report.oracle.input_bf16_values,
                case_row_count * hidden_size
            );
            assert_eq!(
                report.oracle.w13_preactivation_values,
                case_row_count * intermediate_size
            );
            assert_eq!(
                report.oracle.activation_values,
                case_row_count * intermediate_size
            );
            assert_eq!(report.oracle.output_values, case_row_count * hidden_size);
            assert_eq!(report.availability.checked_experts, blocks.len());
            assert_eq!(report.availability.tensor_records, blocks.len() * 2);
            total_cases += 1;
            total_plan_entries += report.plan_entries;
            total_claimed_rows += report.claimed_rows;
            total_runtime_rows += report.total_sorted_rows;
            total_padding_rows += report.padding_rows;
            total_runtime_input_values += report.buffer_lengths.input_values;
            total_runtime_w13_values += report.buffer_lengths.w13_values;
            total_runtime_activation_values += report.buffer_lengths.activation_values;
            total_runtime_output_values += report.buffer_lengths.output_values;
            total_oracle_input_values += report.oracle.input_bf16_values;
            total_oracle_w13_values += report.oracle.w13_preactivation_values;
            total_oracle_activation_values += report.oracle.activation_values;
            total_oracle_output_values += report.oracle.output_values;
            for (block_idx, block) in blocks.iter().enumerate() {
                descriptor_lines.push(format!(
                    "{case_id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    block.expert_idx,
                    block.absolute_row_offset,
                    block.row_count,
                    block.absolute_row_offset + block.row_count,
                    report.total_sorted_rows,
                    report.claimed_rows,
                    report.padding_rows,
                    report.input_row_stride,
                    report.w13_row_stride,
                    report.activation_row_stride,
                    report.output_row_stride,
                    report.buffer_lengths.input_values,
                    report.buffer_lengths.w13_values,
                    report.buffer_lengths.activation_values,
                    report.buffer_lengths.output_values,
                    report.oracle.input_bf16_values,
                    report.oracle.w13_preactivation_values,
                    report.oracle.activation_values,
                    report.oracle.output_values
                ));
                assert_eq!(
                    report.availability.tensors[block_idx * 2].expert_idx,
                    block.expert_idx
                );
            }
        }
        assert_eq!(total_cases, 3);
        assert_eq!(total_claimed_rows, 385);
        assert_eq!(total_plan_entries, 18);
        let summary_lines = vec![
            "metric\tvalue".to_string(),
            format!("cache_path\t{}", cache_path.display()),
            format!("trace_path\t{}", trace_path.display()),
            "cache_source_gate\t0836".to_string(),
            "capture_source_gate\t1023".to_string(),
            "availability_source_gate\t1508".to_string(),
            format!("cases\t{total_cases}"),
            format!("plan_entries\t{total_plan_entries}"),
            format!("claimed_rows\t{total_claimed_rows}"),
            format!("runtime_total_sorted_rows\t{total_runtime_rows}"),
            format!("runtime_padding_rows\t{total_padding_rows}"),
            format!("runtime_input_values\t{total_runtime_input_values}"),
            format!("runtime_w13_values\t{total_runtime_w13_values}"),
            format!("runtime_activation_values\t{total_runtime_activation_values}"),
            format!("runtime_output_values\t{total_runtime_output_values}"),
            format!("oracle_input_bf16_values\t{total_oracle_input_values}"),
            format!("oracle_w13_values\t{total_oracle_w13_values}"),
            format!("oracle_activation_values\t{total_oracle_activation_values}"),
            format!("oracle_output_values\t{total_oracle_output_values}"),
            format!("captured_routed_vectors_total\t{}", routed_inputs.len()),
            format!("captured_branch_vectors_total\t{}", branch_outputs.len()),
            "nbits\t6".to_string(),
            "group_size\t64".to_string(),
            "layout\trow_major_axis1_grouped_uint6_packed".to_string(),
            "correctness_oracle_metadata\tbf16_path_oracle".to_string(),
            "gpu_kernels_launched\tfalse".to_string(),
            "output_comparison_added\tfalse".to_string(),
            "runtime_prefill_hook_added\tfalse".to_string(),
            "config_knob_added\tfalse".to_string(),
            "decode_hcs_added\tfalse".to_string(),
            "fallback_added\tfalse".to_string(),
            "speed_work\tfalse".to_string(),
        ];
        write_lines(
            &artifact_dir.join(format!("{label}_real_nano_runtime_prefill_contract.tsv")),
            &summary_lines,
        );
        write_lines(
            &artifact_dir.join(format!("{label}_real_nano_runtime_prefill_blocks.tsv")),
            &descriptor_lines,
        );
    }

    #[test]
    fn expert_hqq_runtime_diagnostic_fails_closed_without_cache() {
        let err = validate_expert_hqq_runtime_diagnostic_availability(
            None,
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 6, 4)],
        )
        .expect_err("absent KRHQ cache must fail closed");
        assert!(err.contains("not registered"), "{err}");
    }

    #[test]
    fn expert_hqq_runtime_diagnostic_fails_closed_on_wrong_model_layer_or_expert() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let mut wrong_model = sample_runtime_diagnostic_model(false);
        wrong_model.hidden_size += 1;
        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            wrong_model,
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 6, 4)],
        )
        .expect_err("model-shape mismatch must fail closed");
        assert!(err.contains("model mismatch"), "{err}");

        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(2, 2, 6, 4)],
        )
        .expect_err("wrong layer must fail closed");
        assert!(err.contains("layer_idx"), "{err}");

        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 3, 6, 4)],
        )
        .expect_err("wrong expert must fail closed");
        assert!(err.contains("expert_idx"), "{err}");
    }

    #[test]
    fn expert_hqq_runtime_diagnostic_fails_closed_on_nbits_group_or_layout_mismatch() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 2, 6),
            ],
        )
        .unwrap();
        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 4, 4)],
        )
        .expect_err("nbits mismatch must fail closed");
        assert!(err.contains("nbits mismatch"), "{err}");

        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 6, 8)],
        )
        .expect_err("group-size mismatch must fail closed");
        assert!(err.contains("group_size mismatch"), "{err}");

        let mut bad_layout = cache.clone();
        bad_layout.tensors[0].descriptor.layout =
            "row_major_axis1_grouped_uint4_packed".to_string();
        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&bad_layout),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 6, 4)],
        )
        .expect_err("layout mismatch must fail closed");
        assert!(err.contains("layout"), "{err}");
    }

    #[test]
    fn expert_hqq_runtime_diagnostic_fails_closed_on_role_pairing_mismatch() {
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 1, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 1, 1, 6),
            ],
        )
        .unwrap();
        let err = validate_expert_hqq_runtime_diagnostic_availability(
            Some(&cache),
            sample_runtime_diagnostic_model(false),
            &[ExpertHqqRuntimeDiagnosticRequirement::new(1, 2, 6, 4)],
        )
        .expect_err("mismatched W13/W2 role pair must fail closed");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
    }

    fn ungated_reference_cache_for_nbits(nbits: u8) -> ExpertHqqCache {
        let w13 = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let w2 = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ];
        ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                quantized_input_from_f32(ExpertHqqTensorRole::W13, 0, 1, 4, 8, nbits, 4, &w13),
                quantized_input_from_f32(ExpertHqqTensorRole::W2, 0, 1, 8, 4, nbits, 4, &w2),
            ],
        )
        .unwrap()
    }

    fn ungated_reference_cache() -> ExpertHqqCache {
        ungated_reference_cache_for_nbits(6)
    }

    fn exact_hqq6_input_from_quantized(
        role: ExpertHqqTensorRole,
        layer_idx: usize,
        expert_idx: usize,
        rows: usize,
        cols: usize,
        group_size: usize,
        qvalues: &[u8],
    ) -> ExpertHqqTensorInput {
        assert_eq!(qvalues.len(), rows * cols);
        let padded = padded_cols(cols, group_size).unwrap();
        let groups = group_count(cols, group_size).unwrap();
        let mut quant = vec![0u8; rows * padded];
        for row in 0..rows {
            for col in 0..cols {
                quant[row * padded + col] = qvalues[row * cols + col] & 0x3f;
            }
        }
        ExpertHqqTensorInput::new(
            role,
            layer_idx,
            expert_idx,
            rows,
            cols,
            6,
            group_size,
            pack_uint6_rows(&quant, rows, padded).unwrap(),
            f32_vec_to_le_bytes(&vec![1.0; rows * groups]),
            f32_vec_to_le_bytes(&vec![0.0; rows * groups]),
        )
        .unwrap()
    }

    fn exact_ungated_hqq6_reference_cache_for_layers(layers: &[usize]) -> ExpertHqqCache {
        let w13 = vec![
            1, 0, 0, 0, 0, 0, 0, 0, //
            0, 1, 0, 0, 0, 0, 0, 0, //
            0, 0, 1, 0, 0, 0, 0, 0, //
            0, 0, 0, 1, 0, 0, 0, 0,
        ];
        let w2 = vec![
            1, 0, 0, 0, //
            0, 1, 0, 0, //
            0, 0, 1, 0, //
            0, 0, 0, 1, //
            1, 1, 0, 0, //
            0, 0, 1, 1, //
            1, 0, 1, 0, //
            0, 1, 0, 1,
        ];
        let mut inputs = Vec::with_capacity(layers.len() * 2);
        for &layer_idx in layers {
            inputs.push(exact_hqq6_input_from_quantized(
                ExpertHqqTensorRole::W13,
                layer_idx,
                1,
                4,
                8,
                4,
                &w13,
            ));
            inputs.push(exact_hqq6_input_from_quantized(
                ExpertHqqTensorRole::W2,
                layer_idx,
                1,
                8,
                4,
                4,
                &w2,
            ));
        }
        ExpertHqqCache::from_inputs(sample_header(inputs.len()), inputs).unwrap()
    }

    fn exact_ungated_hqq6_reference_cache() -> ExpertHqqCache {
        exact_ungated_hqq6_reference_cache_for_layers(&[0])
    }

    fn write_trace_comparator_fixture<F>(
        mutate: F,
    ) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)
    where
        F: FnOnce(&mut serde_json::Value),
    {
        write_trace_comparator_fixture_with_input(
            vec![
                2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            mutate,
        )
    }

    fn write_trace_comparator_fixture_with_input<F>(
        input_f32: Vec<f32>,
        mutate: F,
    ) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)
    where
        F: FnOnce(&mut serde_json::Value),
    {
        assert_eq!(input_f32.len(), 16);
        let cache = exact_ungated_hqq6_reference_cache();
        let cache_path = temp_path("expert_hqq_trace_compare_cache");
        let spec_path = temp_path("expert_hqq_trace_compare_spec");
        let trace_path = temp_path("expert_hqq_trace_compare_trace");
        let metrics_path = temp_path("expert_hqq_trace_compare_metrics");
        cache.write_to_path(&cache_path).unwrap();
        write_diagnostic_cache_spec(
            &spec_path,
            &cache_path.display().to_string(),
            0,
            &[1],
            &["w13", "w2"],
            6,
            4,
        );
        let input_bits: Vec<u16> = input_f32.iter().map(|&value| f32_to_bf16(value)).collect();
        let input_values: Vec<f32> = input_bits.iter().map(|&bits| bf16_to_f32(bits)).collect();
        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 0, 2)])
            .unwrap();
        let oracle = cache
            .execute_prefill_bf16_path_oracle(&plan, &input_values, 2)
            .unwrap();
        let stage = |suffix: &str, label: &str, row_width: usize, layout: &str, bits: Vec<u16>| {
            serde_json::json!({
                "stage": format!("layer0_sequential_moe_expert_hqq_runtime_prefill_single_block_{suffix}"),
                "layer": 0,
                "chunk": 0,
                "absolute_position": 1,
                "token_id": 0,
                "metadata": {
                    "available": true,
                    "scope": "request-scoped single-block expert-HQQ GPU prefill diagnostic full-buffer export",
                    "buffer_label": label,
                    "layer_idx": 0,
                    "expert": 1,
                    "absolute_row_offset": 0,
                    "row_count": 2,
                    "row_width": row_width,
                    "value_count": bits.len(),
                    "dtype": "bf16",
                    "layout": layout,
                    "experts_gated": false,
                    "activation": 1,
                    "nbits": 6,
                    "group_size": 4,
                    "hqq_layout": "row_major_axis1_grouped_uint6_packed",
                    "input_row_stride": 8,
                    "w13_row_stride": 4,
                    "activation_row_stride": 4,
                    "output_row_stride": 8
                },
                "trace": {
                    "available": true,
                    "source": suffix,
                    "dtype": "bf16",
                    "bf16_bits_u16": bits.iter().map(|&v| v as u64).collect::<Vec<_>>()
                }
            })
        };
        let mut trace = serde_json::json!({
            "results": [
                {
                    "response": {
                        "debug_reference_trace": {
                            "prefill_stage_trace": {
                                "prefill_stage_snapshots": [
                                    stage("input_full", "input", 8, "row_major_selected_rows_by_routed_hidden", input_bits),
                                    stage(
                                        "w13_full",
                                        "w13_preactivation",
                                        4,
                                        "row_major_selected_rows_by_w13_rows",
                                        oracle
                                            .w13_preactivation
                                            .iter()
                                            .map(|&value| f32_to_bf16(value))
                                            .collect()
                                    ),
                                    stage(
                                        "activation_full",
                                        "activation",
                                        4,
                                        "row_major_selected_rows_by_moe_intermediate",
                                        oracle.activation.iter().map(|&value| f32_to_bf16(value)).collect()
                                    ),
                                    stage(
                                        "output_full",
                                        "output",
                                        8,
                                        "row_major_selected_rows_by_routed_hidden",
                                        oracle.values.iter().map(|&value| f32_to_bf16(value)).collect()
                                    )
                                ]
                            }
                        }
                    }
                }
            ]
        });
        mutate(&mut trace);
        std::fs::write(
            &trace_path,
            format!("{}\n", serde_json::to_string_pretty(&trace).unwrap()),
        )
        .unwrap();
        (trace_path, spec_path, metrics_path)
    }

    fn bits_json(bits: &[u16]) -> serde_json::Value {
        serde_json::json!({
            "available": true,
            "dtype": "bf16",
            "bf16_bits_u16": bits.iter().map(|&v| v as u64).collect::<Vec<_>>(),
        })
    }

    fn write_exact_row_attribution_fixture<F>(
        mutate: F,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )
    where
        F: FnOnce(&mut serde_json::Value),
    {
        let cache = exact_ungated_hqq6_reference_cache();
        let cache_path = temp_path("expert_hqq_exact_row_attr_cache");
        let spec_path = temp_path("expert_hqq_exact_row_attr_spec");
        let response_path = temp_path("expert_hqq_exact_row_attr_response");
        let metrics_path = temp_path("expert_hqq_exact_row_attr_metrics");
        let details_path = temp_path("expert_hqq_exact_row_attr_details");
        cache.write_to_path(&cache_path).unwrap();
        write_diagnostic_cache_spec(
            &spec_path,
            &cache_path.display().to_string(),
            0,
            &[1],
            &["w13", "w2"],
            6,
            4,
        );
        let input_f32 = vec![2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let input_bits: Vec<u16> = input_f32.iter().map(|&value| f32_to_bf16(value)).collect();
        let input_values: Vec<f32> = input_bits.iter().map(|&bits| bf16_to_f32(bits)).collect();
        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 0, 1)])
            .unwrap();
        let oracle = cache
            .execute_prefill_bf16_path_oracle(&plan, &input_values, 1)
            .unwrap();
        let w13_bits: Vec<u16> = oracle
            .w13_preactivation
            .iter()
            .map(|&value| f32_to_bf16(value))
            .collect();
        let activation_bits: Vec<u16> = oracle
            .activation
            .iter()
            .map(|&value| f32_to_bf16(value))
            .collect();
        let output_bits: Vec<u16> = oracle
            .values
            .iter()
            .map(|&value| f32_to_bf16(value))
            .collect();
        let row_payload = serde_json::json!({
            "row_index": {
                "absolute_row": 0,
                "is_requested_worst_global_row": true,
                "is_selected_trace_row_contributor": true,
                "expert": 1,
                "block_start": 0,
                "block_end": 1,
                "is_cached": true,
                "gather_src": 0,
                "gather_weight": 1.0,
                "gather_weight_bits": "0x3f800000"
            },
            "bf16": {
                "input": bits_json(&input_bits),
                "w13": bits_json(&w13_bits),
                "activation": bits_json(&activation_bits),
                "output": bits_json(&output_bits)
            },
            "hqq_gpu": {
                "absolute_row": 0,
                "local_row": 0,
                "input": bits_json(&input_bits),
                "w13": bits_json(&w13_bits),
                "activation": bits_json(&activation_bits),
                "output": bits_json(&output_bits)
            },
            "target_col": {
                "col": 0
            }
        });
        let mut response = serde_json::json!({
            "debug_reference_trace": {
                "prefill_stage_trace": {
                    "prefill_stage_snapshots": [
                        {
                            "stage": "layer0_sequential_moe_expert_hqq_exact_row_quantization_attribution",
                            "layer": 0,
                            "chunk": 0,
                            "absolute_position": 0,
                            "token_id": 0,
                            "metadata": {
                                "available": true,
                                "layer_idx": 0,
                                "expert": 1,
                                "requested_sorted_row": 0,
                                "requested_col": 0,
                                "trace_row": 0,
                                "total_active": 1,
                                "hidden_size": 8,
                                "w13_rows": 4,
                                "intermediate_size": 4,
                                "experts_gated": false,
                                "activation": 1,
                                "nbits": 6,
                                "group_size": 4,
                                "rows": [row_payload]
                            }
                        }
                    ]
                }
            }
        });
        mutate(&mut response);
        std::fs::write(
            &response_path,
            format!("{}\n", serde_json::to_string_pretty(&response).unwrap()),
        )
        .unwrap();
        (response_path, spec_path, metrics_path, details_path)
    }

    #[test]
    fn expert_hqq_exact_row_attribution_valid_fixture_passes() {
        let (response_path, spec_path, metrics_path, details_path) =
            write_exact_row_attribution_fixture(|_| {});
        let report = attribute_expert_hqq_exact_row_trace_paths(
            &response_path,
            &spec_path,
            &metrics_path,
            Some(&details_path),
        )
        .expect("valid exact-row attribution fixture should pass");
        assert_eq!(report.captured_rows, 1);
        assert_eq!(report.selected_contributors, 1);
        assert_eq!(report.max_hqq_gpu_vs_krhq_output_abs, 0.0);
        assert!(report.attribution.contains("hqq_gpu_matches_krhq"));
    }

    #[test]
    fn expert_hqq_exact_row_attribution_fails_closed_on_missing_hqq_buffer() {
        let (response_path, spec_path, metrics_path, _details_path) =
            write_exact_row_attribution_fixture(|response| {
                response["debug_reference_trace"]["prefill_stage_trace"]
                    ["prefill_stage_snapshots"][0]["metadata"]["rows"][0]["hqq_gpu"]["output"] =
                    serde_json::Value::Null;
            });
        let err = attribute_expert_hqq_exact_row_trace_paths(
            &response_path,
            &spec_path,
            &metrics_path,
            None,
        )
        .expect_err("missing hqq output must fail closed");
        assert!(err.contains("missing BF16 bits field output"), "{err}");
    }

    #[test]
    fn expert_hqq_exact_row_attribution_fails_closed_on_shape_mismatch() {
        let (response_path, spec_path, metrics_path, _details_path) =
            write_exact_row_attribution_fixture(|response| {
                let bits = response["debug_reference_trace"]["prefill_stage_trace"]
                    ["prefill_stage_snapshots"][0]["metadata"]["rows"][0]["hqq_gpu"]["output"]
                    ["bf16_bits_u16"]
                    .as_array_mut()
                    .unwrap();
                bits.pop();
            });
        let err = attribute_expert_hqq_exact_row_trace_paths(
            &response_path,
            &spec_path,
            &metrics_path,
            None,
        )
        .expect_err("shape mismatch must fail closed");
        assert!(err.contains("hqq_output width"), "{err}");
    }

    fn subnormal_activation_fixture_input() -> Vec<f32> {
        vec![
            1.0e-20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    }

    fn first_stage_bits_mut<'a>(
        trace: &'a mut serde_json::Value,
        suffix: &str,
    ) -> &'a mut Vec<serde_json::Value> {
        let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
            ["prefill_stage_trace"]["prefill_stage_snapshots"]
            .as_array_mut()
            .unwrap();
        let stage = snapshots
            .iter_mut()
            .find(|snap| {
                snap["stage"]
                    .as_str()
                    .map(|stage| stage.ends_with(suffix))
                    .unwrap_or(false)
            })
            .unwrap();
        stage["trace"]["bf16_bits_u16"].as_array_mut().unwrap()
    }

    fn find_positive_bf16_subnormal(bits: &[serde_json::Value]) -> usize {
        bits.iter()
            .position(|value| {
                value
                    .as_u64()
                    .map(|bits| bits > 0 && bits < 0x80)
                    .unwrap_or(false)
            })
            .expect("fixture must contain a positive BF16 subnormal")
    }

    fn find_normal_bf16(bits: &[serde_json::Value]) -> usize {
        bits.iter()
            .position(|value| {
                value
                    .as_u64()
                    .map(|bits| bits >= 0x0080 && bits < 0x7f80)
                    .unwrap_or(false)
            })
            .expect("fixture must contain a normal BF16 value")
    }

    #[test]
    fn expert_hqq_trace_comparator_valid_small_fixture_passes() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|_| {});
        let report = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect("valid small trace fixture should compare");
        assert!(report.passes_contract);
        assert_eq!(report.case_count(), 1);
        let case = &report.case_reports[0];
        assert_eq!(case.layer_idx, 0);
        assert_eq!(case.expert_idx, 1);
        assert_eq!(case.row_count, 2);
        assert_eq!(case.input.sum_abs, 0.0);
        assert_eq!(case.w13.sum_abs, 0.0);
        assert_eq!(case.activation.sum_abs, 0.0);
        assert_eq!(case.output.sum_abs, 0.0);
        assert!(metrics_path.is_file());
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_accepts_activation_zero_vs_positive_subnormal_contract() {
        let details_path = temp_path("expert_hqq_trace_compare_activation_subnormal_details");
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture_with_input(
            subnormal_activation_fixture_input(),
            |trace| {
                let bits = first_stage_bits_mut(trace, "activation_full");
                let subnormal_idx = find_positive_bf16_subnormal(bits);
                bits[subnormal_idx] = serde_json::json!(0u16);
            },
        );
        let report = compare_expert_hqq_runtime_prefill_trace_paths_with_mismatch_details(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
            Some(&details_path),
        )
        .expect("zero-vs-positive-subnormal activation fixture should compare");
        assert!(report.passes_contract);
        let case = &report.case_reports[0];
        assert!(case.activation.passes);
        assert_eq!(case.activation.mismatch_count, 1);
        let (_input, _w13, activation, _output) = report.stage_totals();
        assert!(activation.passes);
        assert_eq!(activation.mismatch_count, 1);
        let details = std::fs::read_to_string(&details_path).unwrap();
        assert!(details.contains("activation"));
        assert!(details.contains("0x0000"));
        assert!(details.contains("bf16_zero_vs_subnormal_flush_to_zero_candidate"));
        let metrics = std::fs::read_to_string(&metrics_path).unwrap();
        assert!(metrics.contains("TOTAL_activation"));
        assert!(metrics
            .lines()
            .any(|line| line.contains("TOTAL_activation") && line.ends_with("\ttrue")));
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(details_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_fails_closed_on_normal_activation_mismatch() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let bits = first_stage_bits_mut(trace, "activation_full");
            let normal_idx = find_normal_bf16(bits);
            bits[normal_idx] = serde_json::json!(0u16);
        });
        let report = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect("normal activation mismatch fixture should compare and fail contract");
        assert!(!report.passes_contract);
        assert!(!report.case_reports[0].activation.passes);
        assert_eq!(report.case_reports[0].activation.mismatch_count, 1);
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_does_not_apply_subnormal_rule_to_output_stage() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture_with_input(
            subnormal_activation_fixture_input(),
            |trace| {
                let bits = first_stage_bits_mut(trace, "output_full");
                let subnormal_idx = find_positive_bf16_subnormal(bits);
                bits[subnormal_idx] = serde_json::json!(0u16);
            },
        );
        let report = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect("wrong-stage subnormal fixture should compare and fail contract");
        assert!(!report.passes_contract);
        assert!(report.case_reports[0].activation.passes);
        assert!(!report.case_reports[0].output.passes);
        assert_eq!(report.case_reports[0].output.mismatch_count, 1);
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_activation_subnormal_contract_rejects_negative_subnormal_reference() {
        let expected = bf16_to_f32(0x8001);
        let detail = ExpertHqqStageMismatchDetail {
            linear_index: 0,
            local_row: 0,
            column: 0,
            actual_bits: 0x0000,
            expected_bits: 0x8001,
            actual_value: 0.0,
            expected_bf16_value: expected,
            expected_raw_value: expected,
            delta_abs: expected.abs() as f64,
            actual_class: "zero",
            expected_bf16_class: "subnormal",
            expected_raw_class: "subnormal",
            flush_to_zero_or_subnormal_rounding: true,
            diagnostic: mismatch_diagnostic(0x0000, 0x8001, expected),
        };
        let comparison = ExpertHqqTraceStageComparison {
            count: 1,
            sum_abs: detail.delta_abs,
            max_abs: detail.delta_abs,
            l2: detail.delta_abs,
            mismatch_count: 1,
            sum_tolerance: 0.0,
            max_tolerance: 0.0,
            passes: false,
        };
        assert!(!activation_trace_contract_passes(&comparison, &[detail]));
    }

    #[test]
    fn expert_hqq_trace_comparator_writes_subnormal_mismatch_details() {
        let details_path = temp_path("expert_hqq_trace_compare_mismatch_details");
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            let activation = snapshots
                .iter_mut()
                .find(|snap| {
                    snap["stage"]
                        .as_str()
                        .map(|stage| stage.ends_with("activation_full"))
                        .unwrap_or(false)
                })
                .unwrap();
            let bits = activation["trace"]["bf16_bits_u16"].as_array_mut().unwrap();
            let zero_idx = bits
                .iter()
                .position(|value| value.as_u64() == Some(0))
                .expect("fixture activation must contain at least one zero BF16 value");
            bits[zero_idx] = serde_json::json!(1u16);
            let mut unselected_block = snapshots.clone();
            for snap in &mut unselected_block {
                snap["metadata"]["absolute_row_offset"] = serde_json::json!(2);
            }
            snapshots.extend(unselected_block);
        });
        let report = compare_expert_hqq_runtime_prefill_trace_paths_with_mismatch_details(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
            Some(&details_path),
        )
        .expect("subnormal mismatch fixture should compare and fail contract");
        assert!(!report.passes_contract);
        let details = std::fs::read_to_string(&details_path).unwrap();
        assert!(details.contains("activation"));
        assert!(details.contains("0x0001"));
        assert!(details.contains("0x0000"));
        assert!(details.contains("bf16_zero_vs_subnormal_flush_to_zero_candidate"));
        assert!(details.contains("true"));
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(details_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_filters_failure_rows_for_mismatch_details() {
        let details_path = temp_path("expert_hqq_trace_compare_filtered_mismatch_details");
        let filter_path = temp_path("expert_hqq_trace_compare_failure_filter");
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            let activation = snapshots
                .iter_mut()
                .find(|snap| {
                    snap["stage"]
                        .as_str()
                        .map(|stage| stage.ends_with("activation_full"))
                        .unwrap_or(false)
                })
                .unwrap();
            let bits = activation["trace"]["bf16_bits_u16"].as_array_mut().unwrap();
            let zero_idx = bits
                .iter()
                .position(|value| value.as_u64() == Some(0))
                .expect("fixture activation must contain at least one zero BF16 value");
            bits[zero_idx] = serde_json::json!(1u16);
        });
        std::fs::write(
            &filter_path,
            "case_index\tlayer\texpert\tabsolute_row_offset\trow_count\tstage\tcount\tsum_abs\tmax_abs\tl2\tmismatch_count\tsum_tolerance\tmax_tolerance\tpasses\n0\t0\t1\t0\t2\tactivation\t8\t0\t0\t0\t1\t0\t0\tfalse\nLAYER\t0\tALL\tALL\tALL\tLAYER_activation\t8\t0\t0\t0\t1\t0\t0\tfalse\n",
        )
        .unwrap();
        let report = compare_expert_hqq_runtime_prefill_trace_paths_filtered_by_failure_rows(
            &trace_path,
            &spec_path,
            &filter_path,
            Some(&metrics_path),
            Some(&details_path),
        )
        .expect("filtered subnormal mismatch fixture should compare and fail contract");
        assert!(!report.passes_contract);
        assert_eq!(report.block_count(), 1);
        let details = std::fs::read_to_string(&details_path).unwrap();
        assert!(details.contains("0x0001"));
        assert!(details.contains("bf16_zero_vs_subnormal_flush_to_zero_candidate"));
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(details_path);
        let _ = std::fs::remove_file(filter_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_valid_multi_block_fixture_passes() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            let mut second_block = snapshots.clone();
            for snap in &mut second_block {
                snap["stage"] = serde_json::json!(snap["stage"]
                    .as_str()
                    .unwrap()
                    .replace("single_block", "all_active_blocks"));
                snap["metadata"]["scope"] = serde_json::json!(
                    "request-scoped all-active-block expert-HQQ GPU prefill diagnostic full-buffer export"
                );
                snap["metadata"]["absolute_row_offset"] = serde_json::json!(2);
            }
            snapshots.extend(second_block);
        });
        let report = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect("valid multi-block trace fixture should compare");
        assert!(report.passes_contract);
        assert_eq!(report.case_count(), 1);
        assert_eq!(report.block_count(), 2);
        let (_input, w13, _activation, _output) = report.stage_totals();
        assert_eq!(w13.count, 16);
        assert_eq!(w13.sum_abs, 0.0);
        assert!(metrics_path.is_file());
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(report.cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_valid_multi_layer_fixture_passes() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            let mut layer1_block = snapshots.clone();
            for snap in &mut layer1_block {
                snap["stage"] = serde_json::json!(snap["stage"]
                    .as_str()
                    .unwrap()
                    .replace("layer0_", "layer1_")
                    .replace("single_block", "all_active_blocks"));
                snap["layer"] = serde_json::json!(1);
                snap["metadata"]["scope"] = serde_json::json!(
                    "request-scoped all-MoE-layer all-active-block expert-HQQ GPU prefill diagnostic full-buffer export"
                );
                snap["metadata"]["layer_idx"] = serde_json::json!(1);
            }
            snapshots.extend(layer1_block);
        });
        let cache = exact_ungated_hqq6_reference_cache_for_layers(&[0, 1]);
        let cache_path = temp_path("expert_hqq_trace_compare_multilayer_cache");
        cache.write_to_path(&cache_path).unwrap();
        std::fs::write(
            &spec_path,
            format!(
                "{{\"purpose\":\"runtime_prefill_diagnostic\",\"cache_path\":\"{}\",\"requirements\":[{{\"layer_idx\":0,\"experts\":[1],\"roles\":[\"w13\",\"w2\"],\"nbits\":6,\"group_size\":4}},{{\"layer_idx\":1,\"experts\":[1],\"roles\":[\"w13\",\"w2\"],\"nbits\":6,\"group_size\":4}}]}}",
                cache_path.display()
            ),
        )
        .unwrap();
        let report = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect("valid multi-layer trace fixture should compare");
        assert!(report.passes_contract);
        assert_eq!(report.case_count(), 1);
        assert_eq!(report.layer_count(), 2);
        assert_eq!(report.block_count(), 2);
        let by_layer = report.layer_stage_totals();
        assert_eq!(by_layer.len(), 2);
        for (layer_idx, (_input, w13, _activation, _output)) in by_layer {
            assert!(layer_idx <= 1);
            assert_eq!(w13.count, 8);
            assert_eq!(w13.sum_abs, 0.0);
        }
        let metrics = std::fs::read_to_string(&metrics_path).unwrap();
        assert!(metrics.contains("LAYER\t0\tALL"));
        assert!(metrics.contains("LAYER\t1\tALL"));
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
        let _ = std::fs::remove_file(report.cache_path);
        let _ = std::fs::remove_file(cache_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_fails_closed_on_duplicate_block_stage() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            snapshots.push(snapshots[0].clone());
        });
        let err = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect_err("duplicate input full-buffer stage must fail closed");
        assert!(err.contains("duplicate input full-buffer stage"), "{err}");
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_fails_closed_on_missing_buffer() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            snapshots.pop();
        });
        let err = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect_err("missing output full buffer must fail closed");
        assert!(err.contains("missing full-buffer stage"), "{err}");
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_fails_closed_on_wrong_layer_or_expert() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            let snapshots = trace["results"][0]["response"]["debug_reference_trace"]
                ["prefill_stage_trace"]["prefill_stage_snapshots"]
                .as_array_mut()
                .unwrap();
            for snap in snapshots {
                snap["metadata"]["expert"] = serde_json::json!(99);
            }
        });
        let err = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect_err("wrong expert must fail closed");
        assert!(
            err.contains("expert_idx 99 out of range")
                || err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_fails_closed_on_shape_mismatch() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            trace["results"][0]["response"]["debug_reference_trace"]["prefill_stage_trace"]
                ["prefill_stage_snapshots"][0]["metadata"]["row_count"] = serde_json::json!(3);
        });
        let err = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect_err("shape mismatch must fail closed");
        assert!(err.contains("shape mismatch"), "{err}");
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
    }

    #[test]
    fn expert_hqq_trace_comparator_fails_closed_on_dtype_or_layout_mismatch() {
        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            trace["results"][0]["response"]["debug_reference_trace"]["prefill_stage_trace"]
                ["prefill_stage_snapshots"][0]["trace"]["dtype"] = serde_json::json!("f32");
        });
        let err = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect_err("dtype mismatch must fail closed");
        assert!(err.contains("dtype mismatch"), "{err}");
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);

        let (trace_path, spec_path, metrics_path) = write_trace_comparator_fixture(|trace| {
            trace["results"][0]["response"]["debug_reference_trace"]["prefill_stage_trace"]
                ["prefill_stage_snapshots"][1]["metadata"]["layout"] =
                serde_json::json!("wrong_layout");
        });
        let err = compare_expert_hqq_runtime_prefill_trace_paths(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
        )
        .expect_err("layout mismatch must fail closed");
        assert!(err.contains("layout mismatch"), "{err}");
        let _ = std::fs::remove_file(trace_path);
        let _ = std::fs::remove_file(spec_path);
        let _ = std::fs::remove_file(metrics_path);
    }

    fn ungated_reference_plan(cache: &ExpertHqqCache) -> ExpertHqqPrefillDispatchPlan {
        cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 0, 2)])
            .unwrap()
    }

    #[test]
    fn expert_hqq_prefill_reference_executes_ungated_relu2_w13_w2() {
        let cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&cache);
        let inputs = vec![
            2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let output = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect("ungated reference execution should succeed");
        assert_eq!(output.sorted_row_count, 2);
        assert_eq!(output.routed_hidden_size, 8);
        let expected = vec![
            4.0, 9.0, 16.0, 25.0, 13.0, 41.0, 20.0, 34.0, //
            1.0, 0.0, 4.0, 0.0, 1.0, 4.0, 5.0, 0.0,
        ];
        for (idx, (&got, &want)) in output.values.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5, "idx={idx} got={got} want={want}");
        }
    }

    #[test]
    fn expert_hqq_prefill_reference_executes_hqq4_payloads() {
        let cache = ungated_reference_cache_for_nbits(4);
        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 0, 1)])
            .unwrap();
        let inputs = vec![2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let output = cache
            .execute_prefill_reference(&plan, &inputs, 1)
            .expect("HQQ4 reference execution should succeed");
        let expected = [4.0, 9.0, 16.0, 25.0, 13.0, 41.0, 20.0, 34.0];
        for (idx, (&got, &want)) in output.values.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-3, "idx={idx} got={got} want={want}");
        }
    }

    #[test]
    fn expert_hqq_prefill_reference_executes_gated_silu_w13_w2() {
        let gate_rows = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let up_rows = vec![
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut w13 = gate_rows;
        w13.extend(up_rows);
        let w2 = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ];
        let cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                quantized_input_from_f32(ExpertHqqTensorRole::W13, 0, 1, 8, 8, 6, 4, &w13),
                quantized_input_from_f32(ExpertHqqTensorRole::W2, 0, 1, 8, 4, 6, 4, &w2),
            ],
        )
        .unwrap();
        let plan = cache
            .prefill_dispatch_plan(0, true, &[ExpertHqqPrefillWork::new(1, 0, 1)])
            .unwrap();
        let inputs = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let output = cache
            .execute_prefill_reference(&plan, &inputs, 1)
            .expect("gated reference execution should succeed");
        let act = [
            super::silu(1.0) * 2.0,
            super::silu(2.0) * 4.0,
            super::silu(3.0) * 6.0,
            super::silu(4.0) * 8.0,
        ];
        let expected = [
            act[0],
            act[1],
            act[2],
            act[3],
            act[0] + act[1],
            act[2] + act[3],
            act[0] + act[2],
            act[1] + act[3],
        ];
        for (idx, (&got, &want)) in output.values.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4, "idx={idx} got={got} want={want}");
        }
    }

    #[test]
    fn expert_hqq_prefill_reference_fails_closed_on_missing_plan_entries() {
        let cache = ungated_reference_cache();
        let mut plan = ungated_reference_plan(&cache);
        plan.entries.clear();
        let inputs = vec![0.0; 8];
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 1)
            .expect_err("empty plan must fail closed");
        assert!(err.contains("at least one plan entry"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_reference_fails_closed_on_mismatched_row_ranges() {
        let cache = ungated_reference_cache();
        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 1, 2)])
            .unwrap();
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("out-of-range plan rows must fail closed");
        assert!(err.contains("row range"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_reference_fails_closed_on_missing_row_coverage() {
        let cache = ungated_reference_cache();
        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 1, 1)])
            .unwrap();
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("uncovered sorted rows must fail closed");
        assert!(err.contains("no selected expert plan entry"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_reference_fails_closed_on_wrong_role_pairing() {
        let cache = ungated_reference_cache();
        let mut plan = ungated_reference_plan(&cache);
        plan.entries[0].w13_key = ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1);
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("wrong W13/W2 role pairing must fail closed");
        assert!(err.contains("W13 role mismatch"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_reference_fails_closed_on_unsupported_metadata() {
        let cache = ungated_reference_cache();
        let mut plan = ungated_reference_plan(&cache);
        plan.entries[0].w13_group_size = 2;
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("plan/cache group metadata mismatch must fail closed");
        assert!(err.contains("metadata mismatch"), "{err}");

        let mut cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&cache);
        cache.tensors[0].descriptor.layout = "bad_layout".to_string();
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("bad layout metadata must fail closed");
        assert!(err.contains("layout"), "{err}");

        let mut cache = ungated_reference_cache();
        let mut plan = ungated_reference_plan(&cache);
        cache.tensors[0].descriptor.nbits = 5;
        plan.entries[0].w13_nbits = 5;
        let err = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("unsupported nbits must fail closed");
        assert!(err.contains("Unsupported expert-HQQ nbits 5"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_reference_fails_closed_without_marlin_fallback() {
        let full_cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&full_cache);
        let w13_only_cache = ExpertHqqCache::new(
            sample_header(1),
            vec![full_cache
                .require_tensor_record(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1))
                .unwrap()
                .clone()],
        )
        .unwrap();
        let inputs = vec![0.0; 16];
        let err = w13_only_cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect_err("missing W2 must fail instead of falling back to Marlin");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_matches_reference_ungated_hqq6() {
        let cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&cache);
        let inputs = vec![
            2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let reference = cache.execute_prefill_reference(&plan, &inputs, 2).unwrap();
        let dispatch = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect("test-only dispatch should execute from validated KRHQ metadata");
        assert_eq!(dispatch.sorted_row_count, reference.sorted_row_count);
        assert_eq!(dispatch.routed_hidden_size, reference.routed_hidden_size);
        assert_eq!(dispatch.values, reference.values);
    }

    #[test]
    fn expert_hqq_prefill_reference_and_test_dispatch_execute_ungated_hqq8() {
        let cache = ungated_reference_cache_for_nbits(8);
        let plan = ungated_reference_plan(&cache);
        let inputs = vec![
            2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let reference = cache
            .execute_prefill_reference(&plan, &inputs, 2)
            .expect("HQQ8 reference execution should succeed");
        let dispatch = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect("HQQ8 test-only dispatch should execute from validated KRHQ metadata");
        assert_eq!(dispatch.sorted_row_count, reference.sorted_row_count);
        assert_eq!(dispatch.routed_hidden_size, reference.routed_hidden_size);
        assert_eq!(dispatch.values, reference.values);
        let expected = vec![
            4.0, 9.0, 16.0, 25.0, 13.0, 41.0, 20.0, 34.0, //
            1.0, 0.0, 4.0, 0.0, 1.0, 4.0, 5.0, 0.0,
        ];
        for (idx, (&got, &want)) in reference.values.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-3, "idx={idx} got={got} want={want}");
        }
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn expert_hqq_prefill_gpu_prototype_matches_cpu_dispatch_ungated_hqq6() {
        if env::var("KRASIS_EXPERT_HQQ_GPU_PROTOTYPE_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping gated expert-HQQ GPU prototype proof; set KRASIS_EXPERT_HQQ_GPU_PROTOTYPE_PROOF=1"
            );
            return;
        }

        let cache = exact_ungated_hqq6_reference_cache();
        let plan = ungated_reference_plan(&cache);
        let inputs = vec![
            2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cpu_dispatch = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect("CPU test dispatch should execute");
        let gpu_dispatch = cache
            .execute_prefill_test_gpu_prototype(&plan, &inputs, 2)
            .expect("GPU prototype should execute from validated KRHQ metadata");
        assert_eq!(gpu_dispatch.sorted_row_count, cpu_dispatch.sorted_row_count);
        assert_eq!(
            gpu_dispatch.routed_hidden_size,
            cpu_dispatch.routed_hidden_size
        );
        assert_eq!(gpu_dispatch.values, cpu_dispatch.values);

        let registered = execute_prefill_test_gpu_prototype_from_registered_cache(
            Some(&cache),
            &plan,
            &inputs,
            2,
        )
        .expect("registered cache GPU prototype entrypoint should execute");
        assert_eq!(registered.values, cpu_dispatch.values);
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn expert_hqq_prefill_gpu_prototype_fails_closed_before_cuda_dispatch() {
        let cache = exact_ungated_hqq6_reference_cache();
        let plan = ungated_reference_plan(&cache);
        let inputs = vec![0.0; 16];

        let err = execute_prefill_test_gpu_prototype_from_registered_cache(None, &plan, &inputs, 2)
            .expect_err("missing registered metadata must fail closed");
        assert!(err.contains("not registered"), "{err}");

        let mut bad_plan = plan.clone();
        bad_plan.entries[0].w13_group_size = 2;
        let err = cache
            .execute_prefill_test_gpu_prototype(&bad_plan, &inputs, 2)
            .expect_err("metadata mismatch must fail closed");
        assert!(err.contains("metadata mismatch"), "{err}");

        let mut invalid_nbits_cache = cache.clone();
        let mut invalid_nbits_plan = plan.clone();
        invalid_nbits_cache.tensors[0].descriptor.nbits = 5;
        invalid_nbits_plan.entries[0].w13_nbits = 5;
        let err = invalid_nbits_cache
            .execute_prefill_test_gpu_prototype(&invalid_nbits_plan, &inputs, 2)
            .expect_err("unsupported nbits must fail closed");
        assert!(err.contains("Unsupported expert-HQQ nbits 5"), "{err}");

        let w13_only_cache = ExpertHqqCache::new(
            sample_header(1),
            vec![cache
                .require_tensor_record(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1))
                .unwrap()
                .clone()],
        )
        .unwrap();
        let err = w13_only_cache
            .execute_prefill_test_gpu_prototype(&plan, &inputs, 2)
            .expect_err("missing W2 must fail instead of falling back to Marlin");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn expert_hqq_runtime_shaped_gpu_prototype_matches_bf16_oracle_synthetic() {
        if env::var("KRASIS_EXPERT_HQQ_RUNTIME_SHAPED_GPU_PROTOTYPE_PROOF")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping runtime-shaped expert-HQQ GPU prototype proof; set KRASIS_EXPERT_HQQ_RUNTIME_SHAPED_GPU_PROTOTYPE_PROOF=1"
            );
            return;
        }

        let cache = exact_ungated_hqq6_reference_cache();
        let blocks = [ExpertHqqRuntimePrefillBlock::new(1, 2, 2)];
        let shape = ExpertHqqRuntimePrefillBufferShape {
            total_sorted_rows: 6,
            input_row_stride: 10,
            w13_row_stride: 5,
            activation_row_stride: 5,
            output_row_stride: 10,
        };
        let mut runtime_inputs =
            vec![
                123.0;
                runtime_buffer_len(shape.total_sorted_rows, shape.input_row_stride, 8).unwrap()
            ];
        let compact_inputs = vec![
            2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for row in 0..2 {
            let dst = (2 + row) * shape.input_row_stride;
            let src = row * 8;
            runtime_inputs[dst..dst + 8].copy_from_slice(&compact_inputs[src..src + 8]);
        }
        let compact_plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 0, 2)])
            .unwrap();
        let oracle = cache
            .execute_prefill_bf16_path_oracle(&compact_plan, &compact_inputs, 2)
            .unwrap();
        let runtime = cache
            .execute_prefill_runtime_shaped_gpu_prototype(0, false, &blocks, shape, &runtime_inputs)
            .unwrap();
        assert_eq!(runtime.total_sorted_rows, 6);
        assert_eq!(runtime.compact_row_count, 2);
        assert_eq!(runtime.routed_hidden_size, 8);
        assert_eq!(runtime.w13_rows, 4);
        assert_eq!(runtime.moe_intermediate_size, 4);
        assert_eq!(runtime.input_row_stride, 10);
        assert_eq!(runtime.output_row_stride, 10);
        assert_eq!(runtime.claimed_rows.iter().filter(|&&v| v).count(), 2);
        assert!(runtime.claimed_rows[2]);
        assert!(runtime.claimed_rows[3]);
        for compact_row in 0..2 {
            let absolute_row = 2 + compact_row;
            let w13_dst = absolute_row * shape.w13_row_stride;
            let act_dst = absolute_row * shape.activation_row_stride;
            let out_dst = absolute_row * shape.output_row_stride;
            let w13_src = compact_row * 4;
            let out_src = compact_row * 8;
            assert_eq!(
                &runtime.w13_preactivation[w13_dst..w13_dst + 4],
                &oracle.w13_preactivation[w13_src..w13_src + 4]
            );
            assert_eq!(
                &runtime.activation[act_dst..act_dst + 4],
                &oracle.activation[w13_src..w13_src + 4]
            );
            assert_eq!(
                &runtime.values[out_dst..out_dst + 8],
                &oracle.values[out_src..out_src + 8]
            );
            assert!(runtime.w13_preactivation[w13_dst + 4].is_nan());
            assert!(runtime.activation[act_dst + 4].is_nan());
            assert!(runtime.values[out_dst + 8].is_nan());
            assert!(runtime.values[out_dst + 9].is_nan());
        }
        for row in [0usize, 1, 4, 5] {
            assert!(!runtime.claimed_rows[row]);
            let out = row * shape.output_row_stride;
            assert!(runtime.values[out..out + 8].iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    #[cfg(has_prefill_kernels)]
    fn expert_hqq_runtime_shaped_gpu_prototype_fails_closed_before_cuda_dispatch() {
        let cache = exact_ungated_hqq6_reference_cache();
        let shape = ExpertHqqRuntimePrefillBufferShape::contiguous_for_cache(&cache, false, 4)
            .expect("runtime-shaped contiguous shape should build");
        let inputs = vec![0.0; runtime_buffer_len(4, shape.input_row_stride, 8).unwrap()];

        let err = cache
            .execute_prefill_runtime_shaped_gpu_prototype(0, false, &[], shape, &inputs)
            .expect_err("empty runtime-shaped blocks must fail closed");
        assert!(err.contains("at least one block"), "{err}");

        let mut bad_shape = shape;
        bad_shape.output_row_stride = 7;
        let err = cache
            .execute_prefill_runtime_shaped_gpu_prototype(
                0,
                false,
                &[ExpertHqqRuntimePrefillBlock::new(1, 0, 1)],
                bad_shape,
                &inputs,
            )
            .expect_err("output stride smaller than hidden must fail closed");
        assert!(err.contains("output_row_stride"), "{err}");

        let two_expert_cache = ExpertHqqCache::from_inputs(
            sample_header(4),
            vec![
                sample_input(ExpertHqqTensorRole::W13, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 1, 6),
                sample_input(ExpertHqqTensorRole::W13, 0, 2, 6),
                sample_input(ExpertHqqTensorRole::W2, 0, 2, 6),
            ],
        )
        .unwrap();
        let err = two_expert_cache
            .execute_prefill_runtime_shaped_gpu_prototype(
                0,
                false,
                &[
                    ExpertHqqRuntimePrefillBlock::new(1, 2, 1),
                    ExpertHqqRuntimePrefillBlock::new(2, 1, 1),
                ],
                shape,
                &inputs,
            )
            .expect_err("unsorted runtime blocks must fail closed");
        assert!(err.contains("sorted and non-overlapping"), "{err}");

        let err = cache
            .execute_prefill_runtime_shaped_gpu_prototype(
                0,
                false,
                &[ExpertHqqRuntimePrefillBlock::new(1, 3, 2)],
                shape,
                &inputs,
            )
            .expect_err("runtime block beyond total rows must fail closed");
        assert!(err.contains("exceeds total_sorted_rows"), "{err}");

        let w13_only_cache = ExpertHqqCache::new(
            sample_header(1),
            vec![cache
                .require_tensor_record(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1))
                .unwrap()
                .clone()],
        )
        .unwrap();
        let err = w13_only_cache
            .execute_prefill_runtime_shaped_gpu_prototype(
                0,
                false,
                &[ExpertHqqRuntimePrefillBlock::new(1, 0, 1)],
                shape,
                &inputs,
            )
            .expect_err("missing W2 must fail instead of falling back to Marlin");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_matches_reference_hqq4_and_gated_silu() {
        let hqq4_cache = ungated_reference_cache_for_nbits(4);
        let hqq4_plan = hqq4_cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 0, 1)])
            .unwrap();
        let hqq4_inputs = vec![2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let hqq4_reference = hqq4_cache
            .execute_prefill_reference(&hqq4_plan, &hqq4_inputs, 1)
            .unwrap();
        let hqq4_dispatch = hqq4_cache
            .execute_prefill_test_dispatch(&hqq4_plan, &hqq4_inputs, 1)
            .unwrap();
        assert_eq!(hqq4_dispatch.values, hqq4_reference.values);

        let gate_rows = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let up_rows = vec![
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut w13 = gate_rows;
        w13.extend(up_rows);
        let w2 = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ];
        let gated_cache = ExpertHqqCache::from_inputs(
            sample_header(2),
            vec![
                quantized_input_from_f32(ExpertHqqTensorRole::W13, 0, 1, 8, 8, 6, 4, &w13),
                quantized_input_from_f32(ExpertHqqTensorRole::W2, 0, 1, 8, 4, 6, 4, &w2),
            ],
        )
        .unwrap();
        let gated_plan = gated_cache
            .prefill_dispatch_plan(0, true, &[ExpertHqqPrefillWork::new(1, 0, 1)])
            .unwrap();
        let gated_inputs = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let gated_reference = gated_cache
            .execute_prefill_reference(&gated_plan, &gated_inputs, 1)
            .unwrap();
        let gated_dispatch = gated_cache
            .execute_prefill_test_dispatch(&gated_plan, &gated_inputs, 1)
            .unwrap();
        assert_eq!(gated_dispatch.values, gated_reference.values);
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_fails_closed_without_registered_metadata() {
        let cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&cache);
        let inputs = vec![0.0; 16];
        let err = execute_prefill_test_dispatch_from_registered_cache(None, &plan, &inputs, 2)
            .expect_err("missing registered KRHQ metadata must fail closed");
        assert!(err.contains("not registered"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_fails_closed_on_row_mismatch() {
        let cache = ungated_reference_cache();
        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 1, 2)])
            .unwrap();
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("out-of-range selected rows must fail closed");
        assert!(err.contains("row range"), "{err}");

        let plan = cache
            .prefill_dispatch_plan(0, false, &[ExpertHqqPrefillWork::new(1, 1, 1)])
            .unwrap();
        let err = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("missing sorted row coverage must fail closed");
        assert!(err.contains("no selected expert plan entry"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_fails_closed_on_shape_and_role_mismatch() {
        let cache = ungated_reference_cache();
        let mut plan = ungated_reference_plan(&cache);
        plan.entries[0].w2_cols += 1;
        let inputs = vec![0.0; 16];
        let err = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("plan/cache shape mismatch must fail closed");
        assert!(err.contains("metadata mismatch"), "{err}");

        let mut plan = ungated_reference_plan(&cache);
        plan.entries[0].w13_key = ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1);
        let err = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("wrong W13/W2 role pairing must fail closed");
        assert!(err.contains("W13 role mismatch"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_fails_closed_on_unsupported_metadata() {
        let mut invalid_nbits_cache = ungated_reference_cache();
        let mut invalid_nbits_plan = ungated_reference_plan(&invalid_nbits_cache);
        invalid_nbits_cache.tensors[0].descriptor.nbits = 5;
        invalid_nbits_plan.entries[0].w13_nbits = 5;
        let inputs = vec![0.0; 16];
        let err = invalid_nbits_cache
            .execute_prefill_test_dispatch(&invalid_nbits_plan, &inputs, 2)
            .expect_err("unsupported nbits must fail closed");
        assert!(err.contains("Unsupported expert-HQQ nbits 5"), "{err}");

        let cache = ungated_reference_cache();
        let mut plan = ungated_reference_plan(&cache);
        plan.entries[0].w13_group_size = 2;
        let err = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("group metadata mismatch must fail closed");
        assert!(err.contains("metadata mismatch"), "{err}");

        let mut cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&cache);
        cache.tensors[0].descriptor.layout = "bad_layout".to_string();
        let err = cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("layout mismatch must fail closed");
        assert!(err.contains("layout"), "{err}");
    }

    #[test]
    fn expert_hqq_prefill_test_dispatch_fails_closed_without_marlin_fallback() {
        let full_cache = ungated_reference_cache();
        let plan = ungated_reference_plan(&full_cache);
        let w2_only_cache = ExpertHqqCache::new(
            sample_header(1),
            vec![full_cache
                .require_tensor_record(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W2, 0, 1))
                .unwrap()
                .clone()],
        )
        .unwrap();
        let inputs = vec![0.0; 16];
        let err = w2_only_cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("missing W13 must fail instead of falling back to Marlin");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
        assert!(
            !err.to_ascii_lowercase().contains("fallback"),
            "failure must be explicit, not a fallback path: {err}"
        );

        let w13_only_cache = ExpertHqqCache::new(
            sample_header(1),
            vec![full_cache
                .require_tensor_record(ExpertHqqTensorKey::new(ExpertHqqTensorRole::W13, 0, 1))
                .unwrap()
                .clone()],
        )
        .unwrap();
        let err = w13_only_cache
            .execute_prefill_test_dispatch(&plan, &inputs, 2)
            .expect_err("missing W2 must fail instead of falling back to Marlin");
        assert!(
            err.contains("missing required expert-HQQ descriptor"),
            "{err}"
        );
        assert!(
            !err.to_ascii_lowercase().contains("fallback"),
            "failure must be explicit, not a fallback path: {err}"
        );
    }
}
