//! Source-copy experiment contract. No quantization or reserve-policy fallback.
use super::mixed_precision::{self as mp, *};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 4;
pub const FORMAT: &str = "Krasis experimental scales-only Marlin INT4";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceRecord {
    pub index: usize,
    pub source: String,
    pub source_offset: u64,
    pub output_offset: u64,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScalesOnlyProvenance {
    pub method: String,
    pub protocol_sha256: String,
    pub coverage_sha256: String,
    pub source_reserve_sha256: String,
    pub source_reserve: MixedPrecisionManifest,
    pub original_amax: MixedCacheSourceIdentity,
    pub replacement_indices: Vec<usize>,
    pub records: Vec<SourceRecord>,
}

fn hash_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn replacement_indices(source: &MixedPrecisionManifest) -> Result<Vec<usize>, String> {
    if source.schema_version != MIXED_COVERAGE_MANIFEST_VERSION
        || source.scales_only.is_some()
        || source.budget.requested_basis_points != 0
    {
        return Err("scales-only source must be the minimum-reserve v3 treatment".into());
    }
    let policy = source
        .calibration
        .coverage_policy
        .as_ref()
        .ok_or("missing source coverage policy")?;
    source
        .regions
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let coverage = match r.coverage.as_ref() {
                Some(c) => c,
                None => return Some(Err("source region lacks coverage".into())),
            };
            let required = coverage.route_count < policy.minimum_routes_per_expert
                || coverage.corpus_route_counts.contains(&0);
            if (required && r.bits != 8) || (!required && r.bits != 4) {
                return Some(Err(
                    "source contains non-reserve promotions or missing reserve".into(),
                ));
            }
            required.then_some(Ok(i))
        })
        .collect()
}

pub fn project_regions(source: &MixedPrecisionManifest) -> Result<Vec<MixedRegion>, String> {
    let indices = replacement_indices(source)?;
    let mut regions = source.regions.clone();
    for i in indices {
        regions[i].bits = 4;
        regions[i].coverage.as_mut().unwrap().selection_reason =
            "experimental_original_amax_int4".into();
    }
    Ok(regions)
}

pub fn geometry(source: &MixedPrecisionManifest) -> Result<MixedPrecisionGeometry, String> {
    let first = source.regions.first().ok_or("empty source region map")?;
    Ok(MixedPrecisionGeometry {
        group_size: source.group_size as usize,
        hidden_size: source.hidden_size as usize,
        intermediate_size: source.intermediate_size as usize,
        routed_layers: source.routed_layers as usize,
        routed_experts_per_layer: source.routed_experts_per_layer as usize,
        experts_gated: source.experts_gated,
        model_layers: source
            .regions
            .chunks(source.routed_experts_per_layer.max(1) as usize)
            .map(|rows| rows[0].model_layer)
            .collect(),
        int4_expert_bytes: usize::try_from(first.int4_bytes).map_err(|e| e.to_string())?,
        int8_expert_bytes: usize::try_from(first.int8_bytes).map_err(|e| e.to_string())?,
    })
}

pub fn validate(
    m: &MixedPrecisionManifest,
    g: &MixedPrecisionGeometry,
    checkpoint: &str,
    namespace: &str,
    config: &str,
) -> Result<(), String> {
    let p = m
        .scales_only
        .as_ref()
        .ok_or("scales-only manifest lacks provenance")?;
    let s = &p.source_reserve;
    // Reject recursive/nested experimental sources before invoking v3 validation.
    let replacements = replacement_indices(s)?;
    mp::validate_manifest(s, g, checkpoint, namespace, config)?;
    if p.method != "copy_searched_int4_replace_coverage_reserve_with_original_amax_v1"
        || [
            &p.protocol_sha256,
            &p.coverage_sha256,
            &p.source_reserve_sha256,
            &p.original_amax.sha256,
            &p.original_amax.header_sha256,
        ]
        .iter()
        .any(|h| !hash_valid(h))
        || p.source_reserve_sha256
            != mp::sha256_hex(&serde_json::to_vec(s).map_err(|e| e.to_string())?)
        || p.replacement_indices != replacements
        || p.original_amax.bits != 4
        || p.original_amax.expert_int4_calibration_mode != "amax"
        || p.original_amax.basename.is_empty()
        || p.original_amax.basename.contains(['/', '\\'])
        || p.original_amax.bytes < 64 + s.budget.baseline_int4_routed_bytes
    {
        return Err("scales-only source/protocol/replacement identity mismatch".into());
    }
    let c = m
        .compact_cache
        .as_ref()
        .ok_or("missing experimental compact cache")?;
    if c.basename.is_empty()
        || c.basename.contains(['/', '\\'])
        || [
            &c.sha256,
            &c.header_sha256,
            &c.payload_sha256,
            &c.region_map_sha256,
        ]
        .iter()
        .any(|h| !hash_valid(h))
        || c.bytes != 128 + s.budget.baseline_int4_routed_bytes
        || c.region_map_sha256 != mp::region_map_sha256(&m.regions)?
    {
        return Err("scales-only compact cache identity mismatch".into());
    }
    let mut expected = s.clone();
    expected.format = FORMAT.into();
    expected.schema_version = SCHEMA_VERSION;
    expected.regions = project_regions(s)?;
    expected.budget.maximum_added_bytes = 0;
    expected.budget.achieved_added_bytes = 0;
    expected.budget.achieved_percent_millionths = 0;
    expected.compact_cache = m.compact_cache.clone();
    expected.scales_only = m.scales_only.clone();
    if *m != expected || p.records.len() != m.regions.len() {
        return Err("scales-only manifest changed non-ablation fields".into());
    }
    let mut source_offset = 128u64;
    let mut output_offset = 128u64;
    for (i, (r, record)) in s.regions.iter().zip(&p.records).enumerate() {
        let rare = replacements.binary_search(&i).is_ok();
        let offset = if rare {
            64 + i as u64 * r.int4_bytes
        } else {
            source_offset
        };
        if record.index != i
            || record.source
                != if rare {
                    "original_amax"
                } else {
                    "source_reserve"
                }
            || record.source_offset != offset
            || record.output_offset != output_offset
            || record.bytes != r.int4_bytes
            || !hash_valid(&record.sha256)
        {
            return Err(format!("scales-only record provenance mismatch at {i}"));
        }
        source_offset = source_offset
            .checked_add(if r.bits == 4 {
                r.int4_bytes
            } else {
                r.int8_bytes
            })
            .ok_or("source offset overflow")?;
        output_offset = output_offset
            .checked_add(r.int4_bytes)
            .ok_or("output offset overflow")?;
    }
    Ok(())
}

/// Independently bind all route counts to the original capture coverage report.
pub fn validate_coverage(source: &MixedPrecisionManifest, raw: &[u8]) -> Result<(), String> {
    let v: serde_json::Value = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    let policy = source
        .calibration
        .coverage_policy
        .as_ref()
        .ok_or("missing source coverage")?;
    let layers = v["layers"].as_array().ok_or("coverage lacks layers")?;
    if v["minimum_routes_per_expert"].as_u64() != Some(policy.minimum_routes_per_expert)
        || layers.len() != source.routed_layers as usize
    {
        return Err("coverage threshold/layers mismatch".into());
    }
    for (l, rows) in layers.iter().zip(
        source
            .regions
            .chunks(source.routed_experts_per_layer.max(1) as usize),
    ) {
        let totals = l["route_counts"]
            .as_array()
            .ok_or("missing coverage totals")?;
        let corpora = l["corpora"].as_array().ok_or("missing coverage corpora")?;
        if totals.len() != rows.len()
            || corpora.len() != policy.required_corpora as usize
            || corpora
                .iter()
                .any(|c| c["route_counts"].as_array().map(Vec::len) != Some(rows.len()))
        {
            return Err("coverage cardinality mismatch".into());
        }
        for r in rows {
            let cov = r.coverage.as_ref().ok_or("missing region coverage")?;
            if l["model_layer"].as_u64() != Some(r.model_layer as u64)
                || totals[r.expert as usize].as_u64() != Some(cov.route_count)
                || corpora
                    .iter()
                    .zip(&cov.corpus_route_counts)
                    .any(|(c, n)| c["route_counts"][r.expert as usize].as_u64() != Some(*n))
            {
                return Err("capture coverage differs from source reserve".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn fixture() -> MixedPrecisionManifest {
        let (mut source, g) = mp::tests::coverage_fixture();
        source.budget.requested_basis_points = 0;
        source.budget.maximum_added_bytes = 8;
        mp::validate_manifest(
            &source,
            &g,
            &source.checkpoint_sha256,
            &source.cache_namespace,
            &source.config_sha256,
        )
        .unwrap();
        let original = MixedCacheSourceIdentity {
            bits: 4,
            basename: "original.bin".into(),
            bytes: 128,
            sha256: "ab".repeat(32),
            header_sha256: "cd".repeat(32),
            header_config_fnv1a: 0,
            expert_int4_calibration_mode: "amax".into(),
        };
        let mut m = source.clone();
        m.format = FORMAT.into();
        m.schema_version = SCHEMA_VERSION;
        m.regions = project_regions(&source).unwrap();
        m.budget.maximum_added_bytes = 0;
        m.budget.achieved_added_bytes = 0;
        m.budget.achieved_percent_millionths = 0;
        m.compact_cache.as_mut().unwrap().bytes = 128 + source.budget.baseline_int4_routed_bytes;
        m.compact_cache.as_mut().unwrap().region_map_sha256 =
            mp::region_map_sha256(&m.regions).unwrap();
        let mut source_offset = 128;
        let mut output_offset = 128;
        let records = source
            .regions
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let rare = r.bits == 8;
                let record = SourceRecord {
                    index: i,
                    source: if rare {
                        "original_amax"
                    } else {
                        "source_reserve"
                    }
                    .into(),
                    source_offset: if rare {
                        64 + i as u64 * r.int4_bytes
                    } else {
                        source_offset
                    },
                    output_offset,
                    bytes: r.int4_bytes,
                    sha256: "ef".repeat(32),
                };
                source_offset += if rare { r.int8_bytes } else { r.int4_bytes };
                output_offset += r.int4_bytes;
                record
            })
            .collect();
        m.scales_only = Some(Box::new(ScalesOnlyProvenance {
            method: "copy_searched_int4_replace_coverage_reserve_with_original_amax_v1".into(),
            protocol_sha256: "12".repeat(32),
            coverage_sha256: "34".repeat(32),
            source_reserve_sha256: mp::sha256_hex(&serde_json::to_vec(&source).unwrap()),
            replacement_indices: vec![1],
            source_reserve: source,
            original_amax: original,
            records,
        }));
        m
    }
    #[test]
    fn experimental_projection_roundtrips_without_relaxing_reserve() {
        let m = fixture();
        let g = geometry(&m).unwrap();
        mp::validate_manifest(
            &m,
            &g,
            &m.checkpoint_sha256,
            &m.cache_namespace,
            &m.config_sha256,
        )
        .unwrap();
        let raw = serde_json::to_vec(&m).unwrap();
        assert_eq!(mp::parse_canonical_manifest(&raw).unwrap(), m);
        let mut bad = m.scales_only.as_ref().unwrap().source_reserve.clone();
        bad.regions[1].bits = 4;
        assert!(mp::validate_manifest(
            &bad,
            &g,
            &bad.checkpoint_sha256,
            &bad.cache_namespace,
            &bad.config_sha256
        )
        .is_err());
    }
    #[test]
    fn rejects_source_replacement_metadata_and_geometry_tampering() {
        for case in 0..13 {
            let mut m = fixture();
            let g = geometry(&m).unwrap();
            match case {
                0 => m.regions[0].quality_gain_units = Some(0),
                1 => m.regions[1].quality_gain_units = Some(0),
                2 => m.regions[1].bits = 8,
                3 => m.scales_only.as_mut().unwrap().replacement_indices = vec![0],
                4 => m.scales_only.as_mut().unwrap().records[0].source_offset += 1,
                5 => m.scales_only.as_mut().unwrap().records[1].sha256 = "bad".into(),
                6 => {
                    m.scales_only.as_mut().unwrap().source_reserve.regions[1]
                        .coverage
                        .as_mut()
                        .unwrap()
                        .route_count += 1
                }
                7 => m.scales_only.as_mut().unwrap().source_reserve_sha256 = "aa".repeat(32),
                8 => {
                    m.scales_only
                        .as_mut()
                        .unwrap()
                        .original_amax
                        .expert_int4_calibration_mode = "search_rmse".into()
                }
                9 => m.budget.maximum_added_bytes = 1,
                10 => m.schema_version = 3,
                11 => {
                    m.scales_only
                        .as_mut()
                        .unwrap()
                        .source_reserve
                        .schema_version = 4
                }
                _ => m.hidden_size += 1,
            }
            assert!(
                mp::validate_manifest(
                    &m,
                    &g,
                    &m.checkpoint_sha256,
                    &m.cache_namespace,
                    &m.config_sha256
                )
                .is_err(),
                "case {case}"
            );
        }
    }
    #[test]
    fn coverage_counts_must_match_every_expert_and_corpus() {
        let m = fixture();
        let s = &m.scales_only.as_ref().unwrap().source_reserve;
        let counts: Vec<_> = s
            .regions
            .iter()
            .map(|r| r.coverage.as_ref().unwrap().route_count)
            .collect();
        let mut value = serde_json::json!({"minimum_routes_per_expert":500,"layers":[{"model_layer":3,"route_counts":counts,"corpora":[{"route_counts":counts}]}]});
        validate_coverage(s, &serde_json::to_vec(&value).unwrap()).unwrap();
        value["layers"][0]["corpora"][0]["route_counts"][1] = serde_json::json!(2);
        assert!(validate_coverage(s, &serde_json::to_vec(&value).unwrap()).is_err());
    }
}
