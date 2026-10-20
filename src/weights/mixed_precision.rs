//! Immutable mixed routed-expert INT4/INT8 precision-manifest contract.
//!
//! Version-2 treatments execute from one compact cache containing exactly one
//! selected INT4 or INT8 record per routed expert. The build is source- and
//! calibration-bound; normal runtime never needs homogeneous INT4 and INT8
//! banks. Immutable version-1 two-source manifests remain readable.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashSet;

pub const MIXED_MANIFEST_VERSION: u32 = 2;
pub const MIXED_COVERAGE_MANIFEST_VERSION: u32 = 3;
pub const MIXED_MANIFEST_FORMAT: &str = "Krasis routed mixed Marlin INT4/INT8";
pub const MIXED_RANKING_METHOD: &str =
    "paired_int4_int8_output_disagreement_under_routed_activation_moments_v1";
pub const MIXED_RANKING_EXPECTED_V2: &str = "full_expert_bf16_residual_expected_v2";
pub const MIXED_RANKING_TAIL_V2: &str = "full_expert_bf16_residual_tail_v2";
pub const MIXED_RANKING_BLENDED_V2: &str = "full_expert_bf16_residual_blended_v2";
pub const MIXED_RANKING_EXPECTED_EXPOSURE_V3: &str =
    "full_expert_bf16_residual_expected_exposure_v3";
pub const MIXED_RANKING_BLENDED_EXPOSURE_V3: &str = "full_expert_bf16_residual_blended_exposure_v3";
pub const MIXED_COMPACT_CACHE_MAGIC: &[u8; 4] = b"KRMX";
pub const MIXED_COMPACT_CACHE_VERSION: u32 = 1;
pub const MIXED_COMPACT_CACHE_HEADER_BYTES: usize = 128;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedCacheSourceIdentity {
    pub bits: u8,
    pub basename: String,
    pub bytes: u64,
    pub sha256: String,
    pub header_sha256: String,
    /// Exact FNV-1a cache-configuration identity stored in Marlin v7.
    pub header_config_fnv1a: u64,
    /// Exact calibration tag stored in Marlin v7 (`amax` or `search_rmse`).
    /// INT8 sources currently carry `amax`, whose tag is zero.
    pub expert_int4_calibration_mode: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedCompactCacheIdentity {
    pub basename: String,
    pub bytes: u64,
    pub sha256: String,
    pub header_sha256: String,
    pub payload_sha256: String,
    pub region_map_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedCalibrationIdentity {
    pub artifact_sha256: String,
    pub ranking_evidence_sha256: String,
    pub ranking_method: String,
    pub quality_gain_scale_exponent: i32,
    pub corpus_sha256: Vec<String>,
    pub capture_artifact_sha256: Vec<String>,
    pub excluded_input_set_sha256: String,
    /// Exact KIC1 activation-moment artifact used by search-RMSE. Legacy
    /// version-1 manifests predate this explicit edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_rmse_calibration_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage_policy: Option<MixedCoveragePolicy>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedCoveragePolicy {
    pub method: String,
    pub minimum_routes_per_expert: u64,
    pub required_corpora: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedRegionCoverage {
    pub route_count: u64,
    pub corpus_route_counts: Vec<u64>,
    pub selection_reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedBudget {
    /// Requested additional bytes divided by the full INT4 routed payload.
    /// 500/1000/1500/2000 represent +5/+10/+15/+20 percent.
    pub requested_basis_points: u32,
    pub baseline_int4_routed_bytes: u64,
    pub maximum_added_bytes: u64,
    pub achieved_added_bytes: u64,
    /// Achieved percentage in millionths of one percent. Exactly +5% is
    /// represented as 5_000_000.
    pub achieved_percent_millionths: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedRegion {
    pub model_layer: u32,
    pub moe_layer: u32,
    pub expert: u32,
    pub bits: u8,
    pub int4_bytes: u64,
    pub int8_bytes: u64,
    /// Fixed-point INT4-to-INT8 output disagreement under the bound calibration
    /// activation moments. Promotion removes this disagreement for the region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_gain_units: Option<i64>,
    /// Zero-based order in quality-gain-per-added-byte ranking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<MixedRegionCoverage>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedPrecisionManifest {
    pub format: String,
    pub schema_version: u32,
    pub checkpoint_sha256: String,
    pub cache_namespace: String,
    pub config_sha256: String,
    pub group_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub routed_layers: u32,
    pub routed_experts_per_layer: u32,
    pub experts_gated: bool,
    pub default_bits: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_int4: Option<MixedCacheSourceIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_int8: Option<MixedCacheSourceIdentity>,
    /// Version-2 manifests execute from one compact selected-record cache.
    /// Omitted only for immutable legacy version-1 manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_cache: Option<MixedCompactCacheIdentity>,
    pub calibration: MixedCalibrationIdentity,
    pub budget: MixedBudget,
    /// Complete layer-major/expert-major routed index. Unselected regions are
    /// explicit INT4 rows rather than inferred gaps.
    pub regions: Vec<MixedRegion>,
    /// Explicit experimental source-copy ablation, never a reserve policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scales_only: Option<Box<super::scales_ablation::ScalesOnlyProvenance>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedPrecisionGeometry {
    pub group_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub routed_layers: usize,
    pub routed_experts_per_layer: usize,
    pub experts_gated: bool,
    /// Absolute model-layer identity for each zero-based routed layer.
    pub model_layers: Vec<u32>,
    pub int4_expert_bytes: usize,
    pub int8_expert_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromotionCandidate {
    pub model_layer: u32,
    pub moe_layer: u32,
    pub expert: u32,
    pub int4_bytes: u64,
    pub int8_bytes: u64,
    pub quality_gain_units: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromotionSelection {
    pub ranked: Vec<PromotionCandidate>,
    pub selected: Vec<PromotionCandidate>,
    pub maximum_added_bytes: u64,
    pub achieved_added_bytes: u64,
}

fn sha256_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn basename_valid(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains('/')
        && !value.contains('\\')
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn region_map_sha256(regions: &[MixedRegion]) -> Result<String, String> {
    let bytes = serde_json::to_vec(regions)
        .map_err(|error| format!("serialize mixed routed-expert region map: {error}"))?;
    Ok(sha256_hex(&bytes))
}

/// Verify the immutable ranking artifact referenced by a precision manifest.
/// The manifest SHA binds treatment identity; this additional edge ensures the
/// measured evidence file shipped beside it has not been lost or corrupted.
pub fn validate_ranking_evidence_bytes(
    manifest: &MixedPrecisionManifest,
    raw: &[u8],
) -> Result<(), String> {
    if manifest.schema_version == super::scales_ablation::SCHEMA_VERSION {
        let source = &manifest.scales_only.as_ref().ok_or("missing scales-only provenance")?.source_reserve;
        return validate_ranking_evidence_bytes(source, raw);
    }
    if sha256_hex(raw) != manifest.calibration.ranking_evidence_sha256 {
        return Err("mixed routed-expert ranking evidence identity mismatch".to_string());
    }
    if manifest.schema_version == MIXED_COVERAGE_MANIFEST_VERSION {
        let evidence: serde_json::Value = serde_json::from_slice(raw)
            .map_err(|error| format!("parse mixed coverage ranking evidence: {error}"))?;
        let policy = manifest
            .calibration
            .coverage_policy
            .as_ref()
            .ok_or("missing mixed coverage policy")?;
        let tail_fraction = evidence["tail_fraction"]
            .as_f64()
            .ok_or("mixed ranking lacks tail fraction")?;
        let tail_observations = evidence["minimum_tail_observations"]
            .as_u64()
            .ok_or("mixed ranking lacks tail observations")?;
        if !tail_fraction.is_finite()
            || tail_fraction <= 0.0
            || tail_fraction > 0.1
            || tail_observations == 0
            || (tail_observations as f64 / tail_fraction).ceil() as u64
                != policy.minimum_routes_per_expert
        {
            return Err(
                "mixed coverage threshold differs from tail measurement requirements".to_string(),
            );
        }
        if evidence["coverage_policy"].as_str() != Some(policy.method.as_str())
            || evidence["minimum_routes_per_expert"].as_u64()
                != Some(policy.minimum_routes_per_expert)
            || evidence["search_rmse_calibration_sha256"].as_str()
                != manifest
                    .calibration
                    .search_rmse_calibration_sha256
                    .as_deref()
        {
            return Err("mixed coverage ranking policy/calibration mismatch".to_string());
        }
        let candidates = evidence["candidates"]
            .as_array()
            .ok_or("mixed ranking lacks candidates")?;
        if candidates.len() != manifest.regions.len() {
            return Err("mixed coverage ranking cardinality mismatch".to_string());
        }
        let ordering = match manifest.calibration.ranking_method.as_str() {
            MIXED_RANKING_EXPECTED_EXPOSURE_V3 => "expected",
            MIXED_RANKING_TAIL_V2 => "tail",
            MIXED_RANKING_BLENDED_EXPOSURE_V3 => "blended",
            _ => return Err("mixed coverage ranking method unsupported".to_string()),
        };
        for (region, candidate) in manifest.regions.iter().zip(candidates) {
            let coverage = region
                .coverage
                .as_ref()
                .ok_or("mixed region lacks coverage")?;
            let counts = serde_json::to_value(&coverage.corpus_route_counts)
                .map_err(|error| error.to_string())?;
            let required = coverage.route_count < policy.minimum_routes_per_expert
                || coverage.corpus_route_counts.contains(&0);
            if candidate["model_layer"].as_u64() != Some(region.model_layer as u64)
                || candidate["moe_layer"].as_u64() != Some(region.moe_layer as u64)
                || candidate["expert"].as_u64() != Some(region.expert as u64)
                || candidate["int4_bytes"].as_u64() != Some(region.int4_bytes)
                || candidate["int8_bytes"].as_u64() != Some(region.int8_bytes)
                || candidate["route_count"].as_u64() != Some(coverage.route_count)
                || candidate["corpus_route_counts"] != counts
                || candidate["coverage_required_int8"].as_bool() != Some(required)
                || candidate[format!("{ordering}_gain_units")].as_i64() != region.quality_gain_units
                || candidate[format!("{ordering}_rank")].as_u64()
                    != region.quality_rank.map(u64::from)
            {
                return Err(
                    "mixed coverage region differs from measured ranking evidence".to_string(),
                );
            }
            if required
                && [
                    "metrics",
                    "expected_gain_units",
                    "tail_gain_units",
                    "blended_gain_units",
                    "expected_rank",
                    "tail_rank",
                    "blended_rank",
                ]
                .iter()
                .any(|key| candidate.get(key).is_some())
            {
                return Err("coverage-required expert has invented measured scores".to_string());
            }
        }
    }
    Ok(())
}

/// Parse only the canonical representation used as cache identity. A
/// semantically equivalent but differently serialized file is rejected so its
/// SHA-256 is stable across builder, launcher, heatmap, and runtime.
pub fn parse_canonical_manifest(raw: &[u8]) -> Result<MixedPrecisionManifest, String> {
    let manifest: MixedPrecisionManifest = serde_json::from_slice(raw)
        .map_err(|error| format!("parse mixed routed-expert manifest: {error}"))?;
    let canonical = serde_json::to_vec(&manifest)
        .map_err(|error| format!("serialize mixed routed-expert manifest: {error}"))?;
    if canonical != raw {
        return Err("mixed routed-expert manifest is not canonical compact JSON".to_string());
    }
    Ok(manifest)
}

pub fn validate_manifest(
    manifest: &MixedPrecisionManifest,
    geometry: &MixedPrecisionGeometry,
    checkpoint_sha256: &str,
    cache_namespace: &str,
    config_sha256: &str,
) -> Result<(), String> {
    if manifest.schema_version == super::scales_ablation::SCHEMA_VERSION {
        return super::scales_ablation::validate(manifest, geometry, checkpoint_sha256, cache_namespace, config_sha256);
    }
    if manifest.scales_only.is_some() {
        return Err("experimental provenance is forbidden in reserve/legacy manifests".into());
    }
    if manifest.format != MIXED_MANIFEST_FORMAT
        || !matches!(
            manifest.schema_version,
            1 | MIXED_MANIFEST_VERSION | MIXED_COVERAGE_MANIFEST_VERSION
        )
        || manifest.default_bits != 4
    {
        return Err("unsupported mixed routed-expert manifest contract".to_string());
    }
    if (manifest.schema_version == 1
        && (manifest.compact_cache.is_some()
            || manifest.source_int4.is_none()
            || manifest.source_int8.is_none()))
        || (manifest.schema_version >= MIXED_MANIFEST_VERSION
            && (manifest.compact_cache.is_none()
                || manifest.source_int4.is_some()
                || manifest.source_int8.is_some()))
    {
        return Err("mixed routed-expert compact-cache/version contract mismatch".to_string());
    }
    if let (Some(source4), Some(source8)) = (&manifest.source_int4, &manifest.source_int8) {
        if source4.bits != 4
            || source8.bits != 8
            || !basename_valid(&source4.basename)
            || !basename_valid(&source8.basename)
            || source4.basename == source8.basename
            || source4.bytes == 0
            || source8.bytes == 0
            || !matches!(
                source4.expert_int4_calibration_mode.as_str(),
                "amax" | "search_rmse"
            )
            || source8.expert_int4_calibration_mode != "amax"
        {
            return Err("unsupported mixed routed-expert source-cache contract".to_string());
        }
    } else if manifest.source_int4.is_some() != manifest.source_int8.is_some() {
        return Err("mixed routed-expert source-cache identities must be paired".to_string());
    }
    if manifest.checkpoint_sha256 != checkpoint_sha256
        || manifest.cache_namespace != cache_namespace
        || manifest.config_sha256 != config_sha256
    {
        return Err("mixed routed-expert checkpoint/config identity mismatch".to_string());
    }
    if manifest.group_size as usize != geometry.group_size
        || manifest.hidden_size as usize != geometry.hidden_size
        || manifest.intermediate_size as usize != geometry.intermediate_size
        || manifest.routed_layers as usize != geometry.routed_layers
        || manifest.routed_experts_per_layer as usize != geometry.routed_experts_per_layer
        || manifest.experts_gated != geometry.experts_gated
        || geometry.model_layers.len() != geometry.routed_layers
    {
        return Err("mixed routed-expert geometry mismatch".to_string());
    }
    if geometry.int8_expert_bytes <= geometry.int4_expert_bytes {
        return Err("mixed INT8 expert geometry must exceed INT4 bytes".to_string());
    }
    let mut required_sha256 = vec![
        &manifest.checkpoint_sha256,
        &manifest.config_sha256,
        &manifest.calibration.artifact_sha256,
        &manifest.calibration.ranking_evidence_sha256,
        &manifest.calibration.excluded_input_set_sha256,
    ];
    if let Some(value) = manifest.calibration.search_rmse_calibration_sha256.as_ref() {
        required_sha256.push(value);
    }
    if let (Some(source4), Some(source8)) = (&manifest.source_int4, &manifest.source_int8) {
        required_sha256.extend([
            &source4.sha256,
            &source4.header_sha256,
            &source8.sha256,
            &source8.header_sha256,
        ]);
    }
    for value in required_sha256 {
        if !sha256_valid(value) {
            return Err("mixed routed-expert manifest contains an invalid SHA-256".to_string());
        }
    }
    if let Some(compact) = &manifest.compact_cache {
        if !basename_valid(&compact.basename)
            || compact.bytes == 0
            || [
                &compact.sha256,
                &compact.header_sha256,
                &compact.payload_sha256,
                &compact.region_map_sha256,
            ]
            .into_iter()
            .any(|value| !sha256_valid(value))
        {
            return Err("mixed routed-expert compact cache identity is invalid".to_string());
        }
    }
    let ranking_method_valid = if manifest.schema_version == 1 {
        manifest.calibration.ranking_method == MIXED_RANKING_METHOD
    } else {
        matches!(
            manifest.calibration.ranking_method.as_str(),
            MIXED_RANKING_EXPECTED_V2
                | MIXED_RANKING_TAIL_V2
                | MIXED_RANKING_BLENDED_V2
                | MIXED_RANKING_EXPECTED_EXPOSURE_V3
                | MIXED_RANKING_BLENDED_EXPOSURE_V3
        )
    };
    if !ranking_method_valid
        || (manifest.schema_version >= MIXED_MANIFEST_VERSION
            && manifest
                .calibration
                .search_rmse_calibration_sha256
                .is_none())
        || manifest.calibration.quality_gain_scale_exponent != -12
        || manifest.calibration.corpus_sha256.is_empty()
        || manifest.calibration.corpus_sha256.len()
            != manifest.calibration.capture_artifact_sha256.len()
        || manifest
            .calibration
            .corpus_sha256
            .iter()
            .chain(manifest.calibration.capture_artifact_sha256.iter())
            .any(|value| !sha256_valid(value))
    {
        return Err("mixed routed-expert calibration identity is invalid".to_string());
    }
    let unique_corpora: HashSet<&str> = manifest
        .calibration
        .corpus_sha256
        .iter()
        .map(String::as_str)
        .collect();
    let unique_captures: HashSet<&str> = manifest
        .calibration
        .capture_artifact_sha256
        .iter()
        .map(String::as_str)
        .collect();
    if unique_corpora.len() != manifest.calibration.corpus_sha256.len()
        || unique_captures.len() != manifest.calibration.capture_artifact_sha256.len()
    {
        return Err("mixed routed-expert calibration identities must be unique".to_string());
    }
    let capture_binding = serde_json::to_vec(&manifest.calibration.capture_artifact_sha256)
        .map_err(|error| format!("serialize mixed calibration capture binding: {error}"))?;
    if manifest.calibration.artifact_sha256 != sha256_hex(&capture_binding) {
        return Err(
            "mixed routed-expert aggregate calibration identity does not match its captures"
                .to_string(),
        );
    }
    if !matches!(
        manifest.budget.requested_basis_points,
        0 | 500 | 1000 | 1500 | 2000 | 10_000
    ) {
        return Err("mixed routed-expert budget must be 0/+5/+10/+15/+20/full INT8".to_string());
    }
    let region_count = geometry
        .routed_layers
        .checked_mul(geometry.routed_experts_per_layer)
        .ok_or_else(|| "mixed routed-expert region count overflow".to_string())?;
    let coverage_policy = manifest.calibration.coverage_policy.as_ref();
    if (manifest.schema_version == MIXED_COVERAGE_MANIFEST_VERSION) != coverage_policy.is_some() {
        return Err("mixed coverage policy/version mismatch".to_string());
    }
    if let Some(policy) = coverage_policy {
        if policy.method != "underobserved_int8_v1"
            || policy.minimum_routes_per_expert == 0
            || policy.required_corpora as usize != manifest.calibration.corpus_sha256.len()
        {
            return Err("mixed coverage policy is invalid".to_string());
        }
    }
    let baseline = region_count
        .checked_mul(geometry.int4_expert_bytes)
        .ok_or_else(|| "mixed routed-expert INT4 baseline overflow".to_string())?;
    let mut maximum_added = baseline
        .checked_mul(manifest.budget.requested_basis_points as usize)
        .ok_or_else(|| "mixed routed-expert budget multiplication overflow".to_string())?
        / 10_000;
    if manifest.budget.baseline_int4_routed_bytes != baseline as u64
        || manifest.regions.len() != region_count
    {
        return Err("mixed routed-expert budget/region cardinality mismatch".to_string());
    }

    let mut identities = HashSet::with_capacity(region_count);
    let mut quality_ranks = HashSet::with_capacity(region_count);
    let mut candidates = Vec::with_capacity(region_count);
    let mut achieved_added = 0usize;
    let mut required_added = 0u64;
    let mut required_identities = HashSet::new();
    for (index, region) in manifest.regions.iter().enumerate() {
        let moe_layer = index / geometry.routed_experts_per_layer;
        let expert = index % geometry.routed_experts_per_layer;
        if region.moe_layer as usize != moe_layer
            || geometry.model_layers.get(moe_layer).copied() != Some(region.model_layer)
            || region.expert as usize != expert
            || !identities.insert((region.moe_layer, region.expert))
            || !matches!(region.bits, 4 | 8)
            || region.int4_bytes != geometry.int4_expert_bytes as u64
            || region.int8_bytes != geometry.int8_expert_bytes as u64
        {
            return Err(format!(
                "mixed routed-expert region index/geometry mismatch at row {index}"
            ));
        }
        let required = if let Some(policy) = coverage_policy {
            let coverage = region
                .coverage
                .as_ref()
                .ok_or("mixed region lacks coverage evidence")?;
            let total = coverage
                .corpus_route_counts
                .iter()
                .try_fold(0u64, |sum, count| sum.checked_add(*count))
                .ok_or("mixed coverage route count overflow")?;
            if coverage.corpus_route_counts.len() != policy.required_corpora as usize
                || total != coverage.route_count
            {
                return Err("mixed per-corpus coverage counts mismatch".to_string());
            }
            let required = total < policy.minimum_routes_per_expert
                || coverage.corpus_route_counts.contains(&0);
            let reason = if required {
                "coverage_required_int8"
            } else if region.bits == 8 {
                "measured_promotion"
            } else {
                "measured_int4"
            };
            if coverage.selection_reason != reason
                || (required
                    && (region.bits != 8
                        || region.quality_gain_units.is_some()
                        || region.quality_rank.is_some()))
            {
                return Err("mixed coverage selection/score contract mismatch".to_string());
            }
            required
        } else {
            if region.coverage.is_some() {
                return Err("legacy mixed region has coverage policy fields".to_string());
            }
            false
        };
        if required {
            required_added = required_added
                .checked_add(region.int8_bytes - region.int4_bytes)
                .ok_or("mixed coverage reserve overflow")?;
            required_identities.insert((region.moe_layer, region.expert));
        } else {
            let gain = region
                .quality_gain_units
                .ok_or("measured mixed region lacks quality gain")?;
            let rank = region
                .quality_rank
                .ok_or("measured mixed region lacks quality rank")?;
            if gain < 0 || rank as usize >= region_count || !quality_ranks.insert(rank) {
                return Err("mixed measured quality gain/rank mismatch".to_string());
            }
            candidates.push(PromotionCandidate {
                model_layer: region.model_layer,
                moe_layer: region.moe_layer,
                expert: region.expert,
                int4_bytes: region.int4_bytes,
                int8_bytes: region.int8_bytes,
                quality_gain_units: gain,
            });
        }
        if region.bits == 8 {
            achieved_added = achieved_added
                .checked_add(
                    geometry
                        .int8_expert_bytes
                        .checked_sub(geometry.int4_expert_bytes)
                        .ok_or_else(|| "INT8 routed expert is smaller than INT4".to_string())?,
                )
                .ok_or_else(|| "mixed routed-expert achieved budget overflow".to_string())?;
        }
    }
    if coverage_policy.is_some() && manifest.budget.requested_basis_points == 0 {
        maximum_added =
            usize::try_from(required_added).map_err(|_| "coverage reserve size overflow")?;
    }
    if manifest.budget.maximum_added_bytes != maximum_added as u64
        || required_added > maximum_added as u64
    {
        return Err("mixed coverage reserve exceeds requested budget".to_string());
    }
    let achieved_millionths = if baseline == 0 {
        0
    } else {
        (achieved_added as u128 * 100_000_000u128 / baseline as u128) as u64
    };
    if achieved_added > maximum_added
        || manifest.budget.achieved_added_bytes != achieved_added as u64
        || manifest.budget.achieved_percent_millionths != achieved_millionths
    {
        return Err("mixed routed-expert achieved byte budget mismatch".to_string());
    }

    let expected = select_promotions_with_cap(
        candidates,
        maximum_added as u64 - required_added,
        manifest.budget.requested_basis_points == 10_000,
    )?;
    if expected.achieved_added_bytes + required_added != achieved_added as u64 {
        return Err(
            "mixed routed-expert regions do not implement the measured ranking budget".to_string(),
        );
    }
    for (rank, candidate) in expected.ranked.iter().enumerate() {
        let row = candidate.moe_layer as usize * geometry.routed_experts_per_layer
            + candidate.expert as usize;
        if manifest.regions[row].quality_rank != Some(rank as u32) {
            return Err(
                "mixed routed-expert quality ranks do not match ranking evidence".to_string(),
            );
        }
    }
    let mut selected: HashSet<(u32, u32)> = expected
        .selected
        .iter()
        .map(|candidate| (candidate.moe_layer, candidate.expert))
        .collect();
    selected.extend(required_identities);
    if manifest
        .regions
        .iter()
        .any(|region| (region.bits == 8) != selected.contains(&(region.moe_layer, region.expert)))
    {
        return Err(
            "mixed routed-expert precision rows do not match ranking selection".to_string(),
        );
    }
    Ok(())
}

/// Deterministically rank by exact fixed-point quality gain per added byte.
/// Cross multiplication avoids floating-point ordering and division drift.
pub fn select_promotions(
    candidates: Vec<PromotionCandidate>,
    baseline_int4_routed_bytes: u64,
    requested_basis_points: u32,
) -> Result<PromotionSelection, String> {
    if !matches!(
        requested_basis_points,
        0 | 500 | 1000 | 1500 | 2000 | 10_000
    ) {
        return Err("promotion selection requires 0/+5/+10/+15/+20/full INT8".to_string());
    }
    if baseline_int4_routed_bytes == 0 || candidates.is_empty() {
        return Err("promotion selection requires non-empty routed weights".to_string());
    }
    let cap = baseline_int4_routed_bytes
        .checked_mul(requested_basis_points as u64)
        .ok_or("promotion budget overflow")?
        / 10_000;
    select_promotions_with_cap(candidates, cap, requested_basis_points == 10_000)
}

fn select_promotions_with_cap(
    mut candidates: Vec<PromotionCandidate>,
    maximum_added_bytes: u64,
    full_int8: bool,
) -> Result<PromotionSelection, String> {
    let mut seen = HashSet::with_capacity(candidates.len());
    for candidate in &candidates {
        if !seen.insert((candidate.moe_layer, candidate.expert)) {
            return Err(format!(
                "duplicate promotion candidate L{}E{}",
                candidate.moe_layer, candidate.expert,
            ));
        }
        if candidate.int8_bytes <= candidate.int4_bytes || candidate.quality_gain_units < 0 {
            return Err(format!(
                "invalid promotion candidate L{}E{}: bytes={}/{} gain={}",
                candidate.moe_layer,
                candidate.expert,
                candidate.int4_bytes,
                candidate.int8_bytes,
                candidate.quality_gain_units,
            ));
        }
    }
    candidates.sort_by(|left, right| {
        let left_delta = left.int8_bytes - left.int4_bytes;
        let right_delta = right.int8_bytes - right.int4_bytes;
        let left_cross = left.quality_gain_units as i128 * right_delta as i128;
        let right_cross = right.quality_gain_units as i128 * left_delta as i128;
        right_cross
            .cmp(&left_cross)
            .then_with(|| left.moe_layer.cmp(&right.moe_layer))
            .then_with(|| left.expert.cmp(&right.expert))
            .then_with(|| left.model_layer.cmp(&right.model_layer))
    });
    let mut selected = Vec::new();
    let mut achieved_added_bytes = 0u64;
    for candidate in &candidates {
        // Full routed INT8 is a positive control, not a quality-ranked budget.
        // It must contain every expert even when the measured local gain is
        // exactly zero. Other budgets never spend bytes on a zero-gain row.
        if !full_int8 && candidate.quality_gain_units <= 0 {
            continue;
        }
        let delta = candidate.int8_bytes - candidate.int4_bytes;
        let next = achieved_added_bytes
            .checked_add(delta)
            .ok_or_else(|| "promotion achieved-byte counter overflow".to_string())?;
        if next <= maximum_added_bytes {
            selected.push(*candidate);
            achieved_added_bytes = next;
        }
    }
    Ok(PromotionSelection {
        ranked: candidates,
        selected,
        maximum_added_bytes,
        achieved_added_bytes,
    })
}

pub fn compare_quality_ratio(
    left_gain: i64,
    left_added_bytes: u64,
    right_gain: i64,
    right_added_bytes: u64,
) -> Ordering {
    (left_gain as i128 * right_added_bytes as i128)
        .cmp(&(right_gain as i128 * left_added_bytes as i128))
}

/// Return the layer-major region rows owned by one pipeline partition. This is
/// shared by every GPU ordinal; ownership is expressed only as a routed-layer
/// range, never as a device-specific branch.
pub fn regions_for_moe_partition<'a>(
    manifest: &'a MixedPrecisionManifest,
    start_moe_layer: usize,
    num_moe_layers: usize,
) -> Result<&'a [MixedRegion], String> {
    let total_layers = manifest.routed_layers as usize;
    let experts = manifest.routed_experts_per_layer as usize;
    let end_layer = start_moe_layer
        .checked_add(num_moe_layers)
        .ok_or_else(|| "mixed routed-expert partition layer range overflow".to_string())?;
    if end_layer > total_layers {
        return Err(format!(
            "mixed routed-expert layer range [{start_moe_layer}, {end_layer}) exceeds {total_layers} routed layers"
        ));
    }
    let start = start_moe_layer
        .checked_mul(experts)
        .ok_or_else(|| "mixed routed-expert partition row offset overflow".to_string())?;
    let end = end_layer
        .checked_mul(experts)
        .ok_or_else(|| "mixed routed-expert partition row end overflow".to_string())?;
    manifest
        .regions
        .get(start..end)
        .ok_or_else(|| "mixed routed-expert partition rows are unavailable".to_string())
}

/// Build disjoint router-slot masks for heterogeneous Marlin decode. The
/// original weights remain the only source for the final router-slot reduction;
/// each precision launch receives a non-zero weight for exactly the slots it
/// owns so both launches may safely target the same per-slot output arrays.
pub fn partition_router_weights(
    expert_ids: &[i32],
    weights: &[f32],
    expert_bits: &[u8],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if expert_ids.len() != weights.len() {
        return Err("mixed routed-expert IDs/weights length mismatch".to_string());
    }
    let mut int4 = Vec::with_capacity(weights.len());
    let mut int8 = Vec::with_capacity(weights.len());
    for (slot, (&expert_id, &weight)) in expert_ids.iter().zip(weights).enumerate() {
        if expert_id < 0 {
            if weight != 0.0 {
                return Err(format!(
                    "mixed routed-expert inactive slot {slot} has nonzero weight"
                ));
            }
            int4.push(0.0);
            int8.push(0.0);
            continue;
        }
        let expert = usize::try_from(expert_id)
            .map_err(|_| format!("mixed routed-expert slot {slot} ID overflow"))?;
        match expert_bits.get(expert).copied() {
            Some(4) => {
                int4.push(weight);
                int8.push(0.0);
            }
            Some(8) => {
                int4.push(0.0);
                int8.push(weight);
            }
            Some(bits) => {
                return Err(format!(
                    "mixed routed-expert slot {slot} selects unsupported INT{bits} expert {expert}"
                ))
            }
            None => {
                return Err(format!(
                    "mixed routed-expert slot {slot} selects missing expert {expert}"
                ))
            }
        }
    }
    Ok((int4, int8))
}

/// Partition the fused-prefill expert-block index without changing its layout.
/// `-1` is the Marlin MoE skip marker. Every active source block appears in
/// exactly one output list at the identical index.
pub fn partition_prefill_expert_blocks(
    expert_blocks: &[i32],
    expert_bits: &[u8],
) -> Result<(Vec<i32>, Vec<i32>), String> {
    let mut int4 = Vec::with_capacity(expert_blocks.len());
    let mut int8 = Vec::with_capacity(expert_blocks.len());
    for (block, &expert_id) in expert_blocks.iter().enumerate() {
        if expert_id == -1 {
            int4.push(-1);
            int8.push(-1);
            continue;
        }
        if expert_id < -1 {
            return Err(format!(
                "mixed routed-expert prefill block {block} has invalid ID {expert_id}"
            ));
        }
        let expert = usize::try_from(expert_id)
            .map_err(|_| format!("mixed routed-expert prefill block {block} ID overflow"))?;
        match expert_bits.get(expert).copied() {
            Some(4) => {
                int4.push(expert_id);
                int8.push(-1);
            }
            Some(8) => {
                int4.push(-1);
                int8.push(expert_id);
            }
            Some(bits) => {
                return Err(format!(
                    "mixed routed-expert prefill block {block} selects unsupported INT{bits} expert {expert}"
                ))
            }
            None => {
                return Err(format!(
                    "mixed routed-expert prefill block {block} selects missing expert {expert}"
                ))
            }
        }
    }
    Ok((int4, int8))
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    fn hex(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn candidate(layer: u32, expert: u32, gain: i64) -> PromotionCandidate {
        PromotionCandidate {
            model_layer: layer + 3,
            moe_layer: layer,
            expert,
            int4_bytes: 100,
            int8_bytes: 200,
            quality_gain_units: gain,
        }
    }

    fn fixture_manifest() -> (Vec<u8>, MixedPrecisionGeometry) {
        let geometry = MixedPrecisionGeometry {
            group_size: 128,
            hidden_size: 256,
            intermediate_size: 512,
            routed_layers: 1,
            routed_experts_per_layer: 4,
            experts_gated: true,
            model_layers: vec![3],
            int4_expert_bytes: 16,
            int8_expert_bytes: 24,
        };
        let gains = [20, 40, 10, 30];
        let bits = [4, 8, 4, 4];
        let ranks = [2, 0, 3, 1];
        let capture_artifact_sha256 = vec![hex(10)];
        let artifact_sha256 = sha256_hex(&serde_json::to_vec(&capture_artifact_sha256).unwrap());
        let manifest = MixedPrecisionManifest {
            scales_only: None,
            format: MIXED_MANIFEST_FORMAT.to_string(),
            schema_version: 1,
            checkpoint_sha256: hex(1),
            cache_namespace: "fixture-namespace".to_string(),
            config_sha256: hex(2),
            group_size: 128,
            hidden_size: 256,
            intermediate_size: 512,
            routed_layers: 1,
            routed_experts_per_layer: 4,
            experts_gated: true,
            default_bits: 4,
            source_int4: Some(MixedCacheSourceIdentity {
                bits: 4,
                basename: "source-int4.bin".to_string(),
                bytes: 64,
                sha256: hex(3),
                header_sha256: hex(4),
                header_config_fnv1a: 41,
                expert_int4_calibration_mode: "search_rmse".to_string(),
            }),
            source_int8: Some(MixedCacheSourceIdentity {
                bits: 8,
                basename: "source-int8.bin".to_string(),
                bytes: 96,
                sha256: hex(5),
                header_sha256: hex(6),
                header_config_fnv1a: 81,
                expert_int4_calibration_mode: "amax".to_string(),
            }),
            compact_cache: None,
            calibration: MixedCalibrationIdentity {
                artifact_sha256,
                ranking_evidence_sha256: sha256_hex(b"fixture-ranking-evidence"),
                ranking_method: MIXED_RANKING_METHOD.to_string(),
                quality_gain_scale_exponent: -12,
                corpus_sha256: vec![hex(9)],
                capture_artifact_sha256,
                excluded_input_set_sha256: hex(11),
                search_rmse_calibration_sha256: None,
                coverage_policy: None,
            },
            budget: MixedBudget {
                requested_basis_points: 2000,
                baseline_int4_routed_bytes: 64,
                maximum_added_bytes: 12,
                achieved_added_bytes: 8,
                achieved_percent_millionths: 12_500_000,
            },
            regions: (0..4)
                .map(|expert| MixedRegion {
                    model_layer: 3,
                    moe_layer: 0,
                    expert: expert as u32,
                    bits: bits[expert],
                    int4_bytes: 16,
                    int8_bytes: 24,
                    quality_gain_units: Some(gains[expert]),
                    quality_rank: Some(ranks[expert]),
                    coverage: None,
                })
                .collect(),
        };
        (serde_json::to_vec(&manifest).unwrap(), geometry)
    }

    #[test]
    fn exact_budget_selection_is_ratio_ranked_and_stable() {
        let selection = select_promotions(
            vec![candidate(1, 0, 9), candidate(0, 1, 10), candidate(0, 0, 10)],
            2_000,
            1000,
        )
        .unwrap();
        assert_eq!(selection.maximum_added_bytes, 200);
        assert_eq!(selection.achieved_added_bytes, 200);
        assert_eq!(selection.selected.len(), 2);
        assert_eq!(
            (
                selection.selected[0].moe_layer,
                selection.selected[0].expert
            ),
            (0, 0)
        );
        assert_eq!(
            (
                selection.selected[1].moe_layer,
                selection.selected[1].expert
            ),
            (0, 1)
        );
    }

    pub(crate) fn coverage_fixture() -> (MixedPrecisionManifest, MixedPrecisionGeometry) {
        let (raw, geometry) = fixture_manifest();
        let mut manifest = parse_canonical_manifest(&raw).unwrap();
        manifest.schema_version = MIXED_COVERAGE_MANIFEST_VERSION;
        manifest.source_int4 = None;
        manifest.source_int8 = None;
        manifest.compact_cache = Some(MixedCompactCacheIdentity {
            basename: "mixed.krmx".into(),
            bytes: 1,
            sha256: hex(1),
            header_sha256: hex(2),
            payload_sha256: hex(3),
            region_map_sha256: hex(4),
        });
        manifest.calibration.ranking_method = MIXED_RANKING_EXPECTED_EXPOSURE_V3.into();
        manifest.calibration.search_rmse_calibration_sha256 = Some(hex(12));
        manifest.calibration.coverage_policy = Some(MixedCoveragePolicy {
            method: "underobserved_int8_v1".into(),
            minimum_routes_per_expert: 500,
            required_corpora: 1,
        });
        for region in &mut manifest.regions {
            let required = region.expert == 1;
            region.coverage = Some(MixedRegionCoverage {
                route_count: if required { 1 } else { 500 },
                corpus_route_counts: vec![if required { 1 } else { 500 }],
                selection_reason: if required {
                    "coverage_required_int8"
                } else {
                    "measured_int4"
                }
                .into(),
            });
            region.quality_rank = match region.expert {
                0 => Some(1),
                1 => None,
                2 => Some(2),
                _ => Some(0),
            };
            if required {
                region.quality_gain_units = None;
            }
        }
        (manifest, geometry)
    }

    #[test]
    fn coverage_reserve_is_explicit_and_charged_to_the_cap() {
        let (mut manifest, geometry) = coverage_fixture();
        validate_manifest(&manifest, &geometry, &hex(1), "fixture-namespace", &hex(2)).unwrap();
        manifest.budget.requested_basis_points = 0;
        manifest.budget.maximum_added_bytes = 8;
        validate_manifest(&manifest, &geometry, &hex(1), "fixture-namespace", &hex(2)).unwrap();
        assert!(manifest.regions[1].quality_gain_units.is_none());
        let canonical = serde_json::to_vec(&manifest).unwrap();
        assert_eq!(parse_canonical_manifest(&canonical).unwrap(), manifest);
        manifest.budget.maximum_added_bytes = 7;
        assert!(
            validate_manifest(&manifest, &geometry, &hex(1), "fixture-namespace", &hex(2)).is_err()
        );
    }

    #[test]
    fn coverage_policy_rejects_fake_scores_counts_and_unprotected_experts() {
        let (manifest, geometry) = coverage_fixture();
        for change in 0..5 {
            let mut bad = manifest.clone();
            match change {
                0 => bad.regions[1].quality_gain_units = Some(0),
                1 => bad.regions[1].bits = 4,
                2 => bad.regions[1].coverage.as_mut().unwrap().route_count = 2,
                3 => bad.calibration.coverage_policy = None,
                _ => bad.regions[0].quality_rank = None,
            }
            assert!(
                validate_manifest(&bad, &geometry, &hex(1), "fixture-namespace", &hex(2)).is_err()
            );
        }
    }

    #[test]
    fn budget_never_uses_a_partial_expert_or_exceeds_the_cap() {
        let selection = select_promotions(
            vec![candidate(0, 0, 3), candidate(0, 1, 2), candidate(0, 2, 1)],
            1_001,
            500,
        )
        .unwrap();
        assert_eq!(selection.maximum_added_bytes, 50);
        assert_eq!(selection.achieved_added_bytes, 0);
        assert!(selection.selected.is_empty());
    }

    #[test]
    fn zero_budget_selects_none_and_full_control_selects_every_expert() {
        let candidates = vec![candidate(0, 0, 5), candidate(0, 1, 0), candidate(0, 2, 1)];
        let zero = select_promotions(candidates.clone(), 300, 0).unwrap();
        assert!(zero.selected.is_empty());
        assert_eq!(zero.achieved_added_bytes, 0);
        let full = select_promotions(candidates, 300, 10_000).unwrap();
        assert_eq!(full.selected.len(), 3);
        assert_eq!(full.achieved_added_bytes, 300);
    }

    #[test]
    fn unsupported_duplicate_and_negative_candidates_fail_closed() {
        assert!(select_promotions(vec![candidate(0, 0, 1)], 100, 750).is_err());
        assert!(
            select_promotions(vec![candidate(0, 0, 1), candidate(0, 0, 2)], 2_000, 500,).is_err()
        );
        assert!(select_promotions(vec![candidate(0, 0, -1)], 2_000, 500).is_err());
        let selection = select_promotions(vec![candidate(0, 0, 0)], 2_000, 500).unwrap();
        assert!(selection.selected.is_empty());
    }

    #[test]
    fn ratio_compare_uses_exact_cross_multiplication() {
        assert_eq!(compare_quality_ratio(2, 4, 3, 9), Ordering::Greater);
        assert_eq!(compare_quality_ratio(2, 4, 1, 2), Ordering::Equal);
    }

    #[test]
    fn compact_manifest_accepts_exposure_methods_and_preserves_old_methods() {
        let (raw, geometry) = fixture_manifest();
        let legacy: MixedPrecisionManifest = serde_json::from_slice(&raw).unwrap();
        validate_manifest(&legacy, &geometry, &hex(1), "fixture-namespace", &hex(2)).unwrap();
        let mut compact = legacy.clone();
        compact.source_int4 = None;
        compact.source_int8 = None;
        compact.schema_version = MIXED_MANIFEST_VERSION;
        compact.calibration.search_rmse_calibration_sha256 = Some(hex(12));
        compact.compact_cache = Some(MixedCompactCacheIdentity {
            basename: "compact.krmx".to_string(),
            bytes: 88,
            sha256: hex(13),
            header_sha256: hex(14),
            payload_sha256: hex(15),
            region_map_sha256: region_map_sha256(&compact.regions).unwrap(),
        });
        for method in [
            MIXED_RANKING_EXPECTED_V2,
            MIXED_RANKING_TAIL_V2,
            MIXED_RANKING_BLENDED_V2,
            MIXED_RANKING_EXPECTED_EXPOSURE_V3,
            MIXED_RANKING_BLENDED_EXPOSURE_V3,
        ] {
            compact.calibration.ranking_method = method.to_string();
            validate_manifest(&compact, &geometry, &hex(1), "fixture-namespace", &hex(2)).unwrap();
            let mut wrong_legacy = legacy.clone();
            wrong_legacy.calibration.ranking_method = method.to_string();
            assert!(validate_manifest(
                &wrong_legacy,
                &geometry,
                &hex(1),
                "fixture-namespace",
                &hex(2)
            )
            .is_err());
        }
        compact.calibration.ranking_method = "unknown_method".to_string();
        assert!(
            validate_manifest(&compact, &geometry, &hex(1), "fixture-namespace", &hex(2)).is_err()
        );
    }

    #[test]
    fn version_two_rejects_parallel_homogeneous_source_banks() {
        let (raw, geometry) = fixture_manifest();
        let mut manifest: MixedPrecisionManifest = serde_json::from_slice(&raw).unwrap();
        let legacy_source = manifest.source_int4.take();
        manifest.source_int8 = None;
        manifest.schema_version = MIXED_MANIFEST_VERSION;
        manifest.calibration.ranking_method = MIXED_RANKING_EXPECTED_V2.to_string();
        manifest.calibration.search_rmse_calibration_sha256 = Some(hex(12));
        manifest.compact_cache = Some(MixedCompactCacheIdentity {
            basename: "compact.krmx".to_string(),
            bytes: 88,
            sha256: hex(13),
            header_sha256: hex(14),
            payload_sha256: hex(15),
            region_map_sha256: region_map_sha256(&manifest.regions).unwrap(),
        });
        validate_manifest(&manifest, &geometry, &hex(1), "fixture-namespace", &hex(2)).unwrap();
        manifest.source_int4 = legacy_source;
        assert!(
            validate_manifest(&manifest, &geometry, &hex(1), "fixture-namespace", &hex(2),)
                .unwrap_err()
                .contains("compact-cache/version")
        );
    }

    #[test]
    fn decode_router_masks_are_disjoint_and_preserve_original_weights() {
        let ids = [2, 0, 1, -1];
        let weights = [0.4, 0.3, 0.2, 0.0];
        let (int4, int8) = partition_router_weights(&ids, &weights, &[8, 4, 8]).unwrap();
        assert_eq!(int4, vec![0.0, 0.0, 0.2, 0.0]);
        assert_eq!(int8, vec![0.4, 0.3, 0.0, 0.0]);
        for slot in 0..weights.len() {
            assert_eq!(int4[slot] + int8[slot], weights[slot]);
            assert!(!(int4[slot] != 0.0 && int8[slot] != 0.0));
        }
    }

    #[test]
    fn prefill_block_masks_preserve_indices_and_fail_closed() {
        let blocks = [0, 0, 1, 2, 2, -1];
        let (int4, int8) = partition_prefill_expert_blocks(&blocks, &[4, 8, 4]).unwrap();
        assert_eq!(int4, vec![0, 0, -1, 2, 2, -1]);
        assert_eq!(int8, vec![-1, -1, 1, -1, -1, -1]);
        assert!(partition_prefill_expert_blocks(&[3], &[4, 8, 4]).is_err());
        assert!(partition_router_weights(&[0], &[1.0, 2.0], &[4]).is_err());
    }

    #[test]
    fn pipeline_partitions_use_layer_ranges_without_gpu_index_assumptions() {
        let (raw, _) = fixture_manifest();
        let mut manifest = parse_canonical_manifest(&raw).unwrap();
        let template = manifest.regions.clone();
        manifest.routed_layers = 3;
        manifest.regions = (0..3)
            .flat_map(|layer| {
                template.iter().cloned().map(move |mut region| {
                    region.moe_layer = layer;
                    region.model_layer = layer + 3;
                    region
                })
            })
            .collect();
        let middle = regions_for_moe_partition(&manifest, 1, 1).unwrap();
        assert_eq!(middle.len(), 4);
        assert!(middle.iter().all(|region| region.moe_layer == 1));
        let tail = regions_for_moe_partition(&manifest, 1, 2).unwrap();
        assert_eq!(tail.len(), 8);
        assert_eq!(tail.first().unwrap().moe_layer, 1);
        assert_eq!(tail.last().unwrap().moe_layer, 2);
        assert!(regions_for_moe_partition(&manifest, 2, 2)
            .unwrap_err()
            .contains("exceeds 3 routed layers"));
    }

    #[test]
    fn canonical_manifest_round_trip_and_identity_validate() {
        let (raw, geometry) = fixture_manifest();
        let manifest = parse_canonical_manifest(&raw).unwrap();
        validate_manifest(&manifest, &geometry, &hex(1), "fixture-namespace", &hex(2)).unwrap();
        assert_eq!(manifest.budget.maximum_added_bytes, 12);
        assert_eq!(manifest.budget.achieved_added_bytes, 8);
        assert_eq!(
            manifest
                .regions
                .iter()
                .filter(|region| region.bits == 8)
                .count(),
            1
        );
        let mut noncanonical = raw;
        noncanonical.push(b'\n');
        assert!(parse_canonical_manifest(&noncanonical).is_err());
    }

    #[test]
    fn ranking_evidence_corruption_fails_closed() {
        let (raw, _) = fixture_manifest();
        let manifest = parse_canonical_manifest(&raw).unwrap();
        validate_ranking_evidence_bytes(&manifest, b"fixture-ranking-evidence").unwrap();
        assert!(
            validate_ranking_evidence_bytes(&manifest, b"corrupt-ranking-evidence")
                .unwrap_err()
                .contains("identity mismatch")
        );
    }

    #[test]
    fn checkpoint_layer_and_ranking_mismatch_fail_closed() {
        let (raw, geometry) = fixture_manifest();
        let manifest = parse_canonical_manifest(&raw).unwrap();
        assert!(
            validate_manifest(&manifest, &geometry, &hex(12), "fixture-namespace", &hex(2),)
                .is_err()
        );
        let mut wrong_layers = geometry.clone();
        wrong_layers.model_layers[0] = 4;
        assert!(validate_manifest(
            &manifest,
            &wrong_layers,
            &hex(1),
            "fixture-namespace",
            &hex(2),
        )
        .is_err());
        let mut wrong_ranking = manifest;
        wrong_ranking.regions[0].quality_rank = Some(0);
        assert!(validate_manifest(
            &wrong_ranking,
            &geometry,
            &hex(1),
            "fixture-namespace",
            &hex(2),
        )
        .is_err());

        let (raw, _) = fixture_manifest();
        let mut wrong_gain = parse_canonical_manifest(&raw).unwrap();
        wrong_gain.regions[0].quality_gain_units = Some(-1);
        assert!(validate_manifest(
            &wrong_gain,
            &geometry,
            &hex(1),
            "fixture-namespace",
            &hex(2),
        )
        .is_err());

        let (raw, _) = fixture_manifest();
        let mut wrong_capture_binding = parse_canonical_manifest(&raw).unwrap();
        wrong_capture_binding.calibration.artifact_sha256 = hex(7);
        assert!(validate_manifest(
            &wrong_capture_binding,
            &geometry,
            &hex(1),
            "fixture-namespace",
            &hex(2),
        )
        .unwrap_err()
        .contains("aggregate calibration identity"));

        let (raw, _) = fixture_manifest();
        let mut duplicate_calibration = parse_canonical_manifest(&raw).unwrap();
        duplicate_calibration.calibration.corpus_sha256.push(hex(9));
        duplicate_calibration
            .calibration
            .capture_artifact_sha256
            .push(hex(10));
        assert!(validate_manifest(
            &duplicate_calibration,
            &geometry,
            &hex(1),
            "fixture-namespace",
            &hex(2),
        )
        .is_err());
    }
}
