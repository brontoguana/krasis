use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const TRACE_SCHEMA: &str = "krasis_expert_prefetch_trace_v1";
pub const DATASET_SCHEMA_V1: &str = "krasis_expert_prefetch_dataset_v1";
pub const DATASET_SCHEMA_V2: &str = "krasis_expert_prefetch_dataset_v2";
const MAX_PREDICTOR_CANDIDATES_PER_LAYER: usize = 64;
const MAX_TRANSITION_CANDIDATES_PER_SOURCE: usize = 32;
const DEFAULT_DATASET_HISTORY_TOKENS: usize = 4;
const DEFAULT_DATASET_PRIOR_LAYERS: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum ExpertPrefetchTraceRecord {
    #[serde(rename = "meta")]
    Meta(ExpertPrefetchTraceMeta),
    #[serde(rename = "predecode")]
    Predecode(ExpertPredecodeTraceEvent),
    #[serde(rename = "route")]
    Route(ExpertRouteTraceEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPrefetchTraceMeta {
    pub schema: String,
    pub created_unix_ms: u64,
    pub producer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRouteTraceEvent {
    pub schema: String,
    pub request_seq: u64,
    pub request_label: String,
    pub step: usize,
    pub layer: usize,
    pub num_experts: usize,
    pub topk: usize,
    pub expert_ids: Vec<i32>,
    pub weights: Vec<f32>,
    pub hcs_hits: Vec<bool>,
    pub cold_experts: usize,
    pub cold_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeTraceEvent {
    pub schema: String,
    #[serde(default)]
    pub predecode_version: String,
    pub request_seq: u64,
    pub request_label: String,
    pub prompt_tokens: usize,
    pub token_window: usize,
    pub first_token_ids: Vec<u32>,
    pub last_token_ids: Vec<u32>,
    pub count_layers: usize,
    pub count_experts_per_layer: usize,
    pub prompt_expert_counts: Vec<ExpertPredecodeCount>,
    #[serde(default)]
    pub prompt_expert_weight_sums: Vec<ExpertPredecodeWeightSum>,
    #[serde(default)]
    pub recency_windows: Vec<usize>,
    #[serde(default)]
    pub prompt_expert_recency_counts: Vec<ExpertPredecodeRecencyCount>,
    #[serde(default)]
    pub prompt_expert_recency_weight_sums: Vec<ExpertPredecodeRecencyWeightSum>,
    #[serde(default)]
    pub final_token_routes: Vec<ExpertPredecodeFinalRoute>,
    #[serde(default)]
    pub prompt_route_entropy: Vec<ExpertPredecodeLayerEntropy>,
    #[serde(default)]
    pub prompt_route_confidence: Vec<ExpertPredecodeLayerConfidence>,
    #[serde(default)]
    pub prompt_route_recency_confidence: Vec<ExpertPredecodeWindowConfidence>,
    #[serde(default)]
    pub position_buckets: Vec<ExpertPredecodePositionBucket>,
    #[serde(default)]
    pub prompt_expert_bucket_counts: Vec<ExpertPredecodeBucketCount>,
    #[serde(default)]
    pub prompt_expert_bucket_weight_sums: Vec<ExpertPredecodeBucketWeightSum>,
    #[serde(default)]
    pub prompt_route_head: Vec<ExpertPredecodeTokenRoute>,
    #[serde(default)]
    pub prompt_route_tail: Vec<ExpertPredecodeTokenRoute>,
    pub hcs_loaded: usize,
    pub hcs_total: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeCount {
    pub layer: usize,
    pub expert: usize,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeWeightSum {
    pub layer: usize,
    pub expert: usize,
    pub weight_sum: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeRecencyCount {
    pub window: usize,
    pub layer: usize,
    pub expert: usize,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeRecencyWeightSum {
    pub window: usize,
    pub layer: usize,
    pub expert: usize,
    pub weight_sum: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeFinalRoute {
    pub layer: usize,
    pub expert_ids: Vec<i32>,
    pub weights: Vec<f32>,
    pub entropy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeLayerEntropy {
    pub layer: usize,
    pub mean_topk_entropy: f32,
    pub final_topk_entropy: f32,
    pub samples: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeLayerConfidence {
    pub layer: usize,
    pub mean_top1_weight: f32,
    pub mean_top1_top2_margin: f32,
    pub mean_topk_weight_sum: f32,
    pub final_top1_weight: f32,
    pub final_top2_weight: f32,
    pub final_top1_top2_margin: f32,
    pub final_topk_weight_sum: f32,
    pub samples: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeWindowConfidence {
    pub window: usize,
    pub layer: usize,
    pub mean_topk_entropy: f32,
    pub mean_top1_weight: f32,
    pub mean_top1_top2_margin: f32,
    pub samples: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodePositionBucket {
    pub bucket: usize,
    pub start_token: usize,
    pub end_token: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeBucketCount {
    pub bucket: usize,
    pub layer: usize,
    pub expert: usize,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeBucketWeightSum {
    pub bucket: usize,
    pub layer: usize,
    pub expert: usize,
    pub weight_sum: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPredecodeTokenRoute {
    pub token_index: usize,
    pub layer: usize,
    pub expert_ids: Vec<i32>,
    pub weights: Vec<f32>,
    pub entropy: f32,
    pub top1_weight: f32,
    pub top2_weight: f32,
    pub top1_top2_margin: f32,
    pub topk_weight_sum: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ExpertPredecodeRichFeatures {
    pub version: String,
    pub recency_windows: Vec<usize>,
    pub weighted_counts: Vec<f32>,
    pub recency_counts: Vec<Vec<u64>>,
    pub recency_weight_sums: Vec<Vec<f32>>,
    pub final_routes: Vec<ExpertPredecodeFinalRoute>,
    pub layer_entropy: Vec<ExpertPredecodeLayerEntropy>,
    pub layer_confidence: Vec<ExpertPredecodeLayerConfidence>,
    pub recency_confidence: Vec<ExpertPredecodeWindowConfidence>,
    pub position_buckets: Vec<ExpertPredecodePositionBucket>,
    pub bucket_counts: Vec<Vec<u64>>,
    pub bucket_weight_sums: Vec<Vec<f32>>,
    pub head_routes: Vec<ExpertPredecodeTokenRoute>,
    pub tail_routes: Vec<ExpertPredecodeTokenRoute>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TraceFilter {
    pub request_label: Option<String>,
    pub request_label_prefix: Option<String>,
}

impl TraceFilter {
    fn matches_label(&self, request_label: &str) -> bool {
        let request_label = normalize_request_label(request_label);
        if let Some(label) = self.request_label.as_ref() {
            if request_label != normalize_request_label(label) {
                return false;
            }
        }
        if let Some(prefix) = self.request_label_prefix.as_ref() {
            if !request_label.starts_with(prefix) {
                return false;
            }
        }
        true
    }

    fn matches(&self, route: &ExpertRouteTraceEvent) -> bool {
        self.matches_label(&route.request_label)
    }

    fn describe(&self) -> String {
        match (&self.request_label, &self.request_label_prefix) {
            (Some(label), Some(prefix)) => {
                format!("request_label={} request_label_prefix={}", label, prefix)
            }
            (Some(label), None) => format!("request_label={}", label),
            (None, Some(prefix)) => format!("request_label_prefix={}", prefix),
            (None, None) => "none".to_string(),
        }
    }
}

fn normalize_request_label(label: &str) -> &str {
    label
        .strip_suffix("_nosse")
        .or_else(|| label.strip_suffix("_sse"))
        .unwrap_or(label)
}

fn request_key(request_seq: u64, request_label: &str) -> RequestKey {
    RequestKey {
        request_seq,
        request_label: normalize_request_label(request_label).to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RequestKey {
    request_seq: u64,
    request_label: String,
}

pub fn filter_routes(
    routes: &[ExpertRouteTraceEvent],
    filter: &TraceFilter,
) -> Vec<ExpertRouteTraceEvent> {
    routes
        .iter()
        .filter(|route| filter.matches(route))
        .cloned()
        .collect()
}

pub struct ExpertPrefetchTraceWriter {
    writer: BufWriter<File>,
    record_count: u64,
}

impl ExpertPrefetchTraceWriter {
    pub fn from_env() -> Result<Option<Self>, String> {
        let path = match std::env::var("KRASIS_EXPERT_PREFETCH_TRACE") {
            Ok(raw) if !raw.trim().is_empty() => raw,
            _ => return Ok(None),
        };
        Self::create(path.trim()).map(Some)
    }

    pub fn create(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create expert prefetch trace dir {:?}: {}", parent, e))?;
            }
        }
        let file = File::create(path)
            .map_err(|e| format!("create expert prefetch trace {:?}: {}", path, e))?;
        let mut writer = Self {
            writer: BufWriter::new(file),
            record_count: 0,
        };
        let meta = ExpertPrefetchTraceRecord::Meta(ExpertPrefetchTraceMeta {
            schema: TRACE_SCHEMA.to_string(),
            created_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
                .min(u64::MAX as u128) as u64,
            producer: "krasis_gpu_decode".to_string(),
        });
        writer.write_record(&meta)?;
        Ok(writer)
    }

    fn write_record(&mut self, record: &ExpertPrefetchTraceRecord) -> Result<(), String> {
        serde_json::to_writer(&mut self.writer, record)
            .map_err(|e| format!("serialize expert prefetch trace: {}", e))?;
        self.writer
            .write_all(b"\n")
            .map_err(|e| format!("write expert prefetch trace: {}", e))?;
        self.record_count = self.record_count.saturating_add(1);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_route(
        &mut self,
        request_seq: u64,
        request_label: &str,
        step: usize,
        layer: usize,
        num_experts: usize,
        topk: usize,
        expert_ids: &[i32],
        weights: &[f32],
        hcs_hits: &[bool],
        cold_bytes: u64,
    ) -> Result<(), String> {
        let topk = topk
            .min(expert_ids.len())
            .min(weights.len())
            .min(hcs_hits.len());
        let cold_experts = hcs_hits.iter().take(topk).filter(|&&hit| !hit).count();
        let event = ExpertPrefetchTraceRecord::Route(ExpertRouteTraceEvent {
            schema: TRACE_SCHEMA.to_string(),
            request_seq,
            request_label: request_label.to_string(),
            step,
            layer,
            num_experts,
            topk,
            expert_ids: expert_ids.iter().take(topk).copied().collect(),
            weights: weights.iter().take(topk).copied().collect(),
            hcs_hits: hcs_hits.iter().take(topk).copied().collect(),
            cold_experts,
            cold_bytes,
        });
        self.write_record(&event)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_predecode(
        &mut self,
        request_seq: u64,
        request_label: &str,
        token_ids: &[u32],
        prompt_counts: &[u64],
        rich_features: Option<&ExpertPredecodeRichFeatures>,
        count_layers: usize,
        count_experts_per_layer: usize,
        hcs_loaded: usize,
        hcs_total: usize,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<(), String> {
        let token_window = std::env::var("KRASIS_EXPERT_PREFETCH_TRACE_TOKEN_WINDOW")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(128);
        let first_token_ids = token_ids.iter().take(token_window).copied().collect();
        let mut last_token_ids: Vec<u32> =
            token_ids.iter().rev().take(token_window).copied().collect();
        last_token_ids.reverse();

        let mut counts = Vec::new();
        let expected = count_layers.saturating_mul(count_experts_per_layer);
        let limit = expected.min(prompt_counts.len());
        for idx in 0..limit {
            let count = prompt_counts[idx];
            if count == 0 {
                continue;
            }
            counts.push(ExpertPredecodeCount {
                layer: idx / count_experts_per_layer,
                expert: idx % count_experts_per_layer,
                count,
            });
        }

        let mut weight_sums = Vec::new();
        let mut recency_windows = Vec::new();
        let mut recency_counts = Vec::new();
        let mut recency_weight_sums = Vec::new();
        let mut final_routes = Vec::new();
        let mut layer_entropy = Vec::new();
        let mut layer_confidence = Vec::new();
        let mut recency_confidence = Vec::new();
        let mut position_buckets = Vec::new();
        let mut bucket_counts = Vec::new();
        let mut bucket_weight_sums = Vec::new();
        let mut head_routes = Vec::new();
        let mut tail_routes = Vec::new();
        let mut predecode_version = "predecode_v1".to_string();
        if let Some(rich) = rich_features {
            predecode_version = rich.version.clone();
            recency_windows = rich.recency_windows.clone();
            let limit = expected.min(rich.weighted_counts.len());
            for idx in 0..limit {
                let weight_sum = rich.weighted_counts[idx];
                if weight_sum > 0.0 {
                    weight_sums.push(ExpertPredecodeWeightSum {
                        layer: idx / count_experts_per_layer,
                        expert: idx % count_experts_per_layer,
                        weight_sum,
                    });
                }
            }
            for (window_idx, window) in recency_windows.iter().copied().enumerate() {
                if let Some(window_counts) = rich.recency_counts.get(window_idx) {
                    let limit = expected.min(window_counts.len());
                    for idx in 0..limit {
                        let count = window_counts[idx];
                        if count > 0 {
                            recency_counts.push(ExpertPredecodeRecencyCount {
                                window,
                                layer: idx / count_experts_per_layer,
                                expert: idx % count_experts_per_layer,
                                count,
                            });
                        }
                    }
                }
                if let Some(window_weights) = rich.recency_weight_sums.get(window_idx) {
                    let limit = expected.min(window_weights.len());
                    for idx in 0..limit {
                        let weight_sum = window_weights[idx];
                        if weight_sum > 0.0 {
                            recency_weight_sums.push(ExpertPredecodeRecencyWeightSum {
                                window,
                                layer: idx / count_experts_per_layer,
                                expert: idx % count_experts_per_layer,
                                weight_sum,
                            });
                        }
                    }
                }
            }
            final_routes = rich.final_routes.clone();
            layer_entropy = rich.layer_entropy.clone();
            layer_confidence = rich.layer_confidence.clone();
            recency_confidence = rich.recency_confidence.clone();
            position_buckets = rich.position_buckets.clone();
            for (bucket_idx, bucket_counts_raw) in rich.bucket_counts.iter().enumerate() {
                let limit = expected.min(bucket_counts_raw.len());
                for idx in 0..limit {
                    let count = bucket_counts_raw[idx];
                    if count > 0 {
                        bucket_counts.push(ExpertPredecodeBucketCount {
                            bucket: bucket_idx,
                            layer: idx / count_experts_per_layer,
                            expert: idx % count_experts_per_layer,
                            count,
                        });
                    }
                }
            }
            for (bucket_idx, bucket_weights_raw) in rich.bucket_weight_sums.iter().enumerate() {
                let limit = expected.min(bucket_weights_raw.len());
                for idx in 0..limit {
                    let weight_sum = bucket_weights_raw[idx];
                    if weight_sum > 0.0 {
                        bucket_weight_sums.push(ExpertPredecodeBucketWeightSum {
                            bucket: bucket_idx,
                            layer: idx / count_experts_per_layer,
                            expert: idx % count_experts_per_layer,
                            weight_sum,
                        });
                    }
                }
            }
            head_routes = rich.head_routes.clone();
            tail_routes = rich.tail_routes.clone();
        }

        let event = ExpertPrefetchTraceRecord::Predecode(ExpertPredecodeTraceEvent {
            schema: TRACE_SCHEMA.to_string(),
            predecode_version,
            request_seq,
            request_label: request_label.to_string(),
            prompt_tokens: token_ids.len(),
            token_window,
            first_token_ids,
            last_token_ids,
            count_layers,
            count_experts_per_layer,
            prompt_expert_counts: counts,
            prompt_expert_weight_sums: weight_sums,
            recency_windows,
            prompt_expert_recency_counts: recency_counts,
            prompt_expert_recency_weight_sums: recency_weight_sums,
            final_token_routes: final_routes,
            prompt_route_entropy: layer_entropy,
            prompt_route_confidence: layer_confidence,
            prompt_route_recency_confidence: recency_confidence,
            position_buckets,
            prompt_expert_bucket_counts: bucket_counts,
            prompt_expert_bucket_weight_sums: bucket_weight_sums,
            prompt_route_head: head_routes,
            prompt_route_tail: tail_routes,
            hcs_loaded,
            hcs_total,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
        });
        self.write_record(&event)
    }

    pub fn flush(&mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|e| format!("flush expert prefetch trace: {}", e))
    }
}

impl Drop for ExpertPrefetchTraceWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

pub fn read_route_trace(path: impl AsRef<Path>) -> Result<Vec<ExpertRouteTraceEvent>, String> {
    let file =
        File::open(path.as_ref()).map_err(|e| format!("open trace {:?}: {}", path.as_ref(), e))?;
    let reader = BufReader::new(file);
    let mut routes = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read trace line {}: {}", line_no + 1, e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: ExpertPrefetchTraceRecord = serde_json::from_str(trimmed)
            .map_err(|e| format!("parse trace line {}: {}", line_no + 1, e))?;
        if let ExpertPrefetchTraceRecord::Route(route) = record {
            if route.schema != TRACE_SCHEMA {
                return Err(format!(
                    "trace line {} has unsupported schema {:?}",
                    line_no + 1,
                    route.schema
                ));
            }
            routes.push(route);
        }
    }
    Ok(routes)
}

#[derive(Debug, Clone)]
pub struct ExpertPrefetchTraceData {
    pub routes: Vec<ExpertRouteTraceEvent>,
    pub predecodes: Vec<ExpertPredecodeTraceEvent>,
}

pub fn read_prefetch_trace(path: impl AsRef<Path>) -> Result<ExpertPrefetchTraceData, String> {
    let file =
        File::open(path.as_ref()).map_err(|e| format!("open trace {:?}: {}", path.as_ref(), e))?;
    let reader = BufReader::new(file);
    let mut routes = Vec::new();
    let mut predecodes = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read trace line {}: {}", line_no + 1, e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: ExpertPrefetchTraceRecord = serde_json::from_str(trimmed)
            .map_err(|e| format!("parse trace line {}: {}", line_no + 1, e))?;
        match record {
            ExpertPrefetchTraceRecord::Route(route) => {
                if route.schema != TRACE_SCHEMA {
                    return Err(format!(
                        "trace line {} has unsupported route schema {:?}",
                        line_no + 1,
                        route.schema
                    ));
                }
                routes.push(route);
            }
            ExpertPrefetchTraceRecord::Predecode(predecode) => {
                if predecode.schema != TRACE_SCHEMA {
                    return Err(format!(
                        "trace line {} has unsupported predecode schema {:?}",
                        line_no + 1,
                        predecode.schema
                    ));
                }
                predecodes.push(predecode);
            }
            ExpertPrefetchTraceRecord::Meta(meta) => {
                if meta.schema != TRACE_SCHEMA {
                    return Err(format!(
                        "trace line {} has unsupported meta schema {:?}",
                        line_no + 1,
                        meta.schema
                    ));
                }
            }
        }
    }
    Ok(ExpertPrefetchTraceData { routes, predecodes })
}

#[derive(Debug, Clone, Serialize)]
pub struct OracleSummary {
    pub schema: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub lookahead_routes: usize,
    pub budget_experts: usize,
    pub label_experts: u64,
    pub current_cold_experts: u64,
    pub oracle_coverable_experts: u64,
    pub oracle_coverable_cold_experts: u64,
    pub current_cold_rate_pct: f64,
    pub oracle_coverable_recall_pct: f64,
    pub oracle_coverable_cold_reduction_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValueOracleSummary {
    pub schema: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub lookahead_routes: usize,
    pub budget_experts: usize,
    pub windows: u64,
    pub label_experts: u64,
    pub current_cold_experts: u64,
    pub current_cold_bytes: u64,
    pub value_coverable_cold_experts: u64,
    pub value_coverable_cold_bytes: u64,
    pub current_cold_rate_pct: f64,
    pub value_coverable_cold_reduction_pct: f64,
    pub value_coverable_cold_bytes_reduction_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HcsOracleSummary {
    pub schema: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filtered_predecodes: usize,
    pub filter: String,
    pub budget_source: String,
    pub budget_override: Option<usize>,
    pub requests: u64,
    pub requests_with_predecode_budget: u64,
    pub total_decode_experts: u64,
    pub current_hcs_hits: u64,
    pub current_cold_experts: u64,
    pub current_cold_bytes: u64,
    pub oracle_hit_hits: u64,
    pub oracle_hit_cold_experts: u64,
    pub oracle_hit_cold_bytes: u64,
    pub oracle_cold_hits: u64,
    pub oracle_cold_cold_experts: u64,
    pub oracle_cold_cold_bytes: u64,
    pub current_hcs_hit_rate_pct: f64,
    pub oracle_hit_hit_rate_pct: f64,
    pub oracle_cold_hit_rate_pct: f64,
    pub oracle_hit_cold_expert_reduction_pct: f64,
    pub oracle_cold_cold_expert_reduction_pct: f64,
    pub oracle_hit_cold_bytes_reduction_pct: f64,
    pub oracle_cold_cold_bytes_reduction_pct: f64,
    pub oracle_hit_dma_bound_speedup_x: f64,
    pub oracle_cold_dma_bound_speedup_x: f64,
    pub per_request: Vec<HcsOracleRequestSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HcsOracleRequestSummary {
    pub request_seq: u64,
    pub request_label: String,
    pub routes: usize,
    pub steps: usize,
    pub hcs_budget: usize,
    pub hcs_total: Option<usize>,
    pub expert_bytes: u64,
    pub total_decode_experts: u64,
    pub current_hcs_hits: u64,
    pub current_cold_experts: u64,
    pub current_cold_bytes: u64,
    pub oracle_hit_hits: u64,
    pub oracle_hit_cold_experts: u64,
    pub oracle_hit_cold_bytes: u64,
    pub oracle_cold_hits: u64,
    pub oracle_cold_cold_experts: u64,
    pub oracle_cold_cold_bytes: u64,
    pub current_hcs_hit_rate_pct: f64,
    pub oracle_hit_hit_rate_pct: f64,
    pub oracle_cold_hit_rate_pct: f64,
    pub oracle_hit_cold_bytes_reduction_pct: f64,
    pub oracle_cold_cold_bytes_reduction_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineSummary {
    pub schema: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub budget_experts: usize,
    pub label_experts: u64,
    pub baselines: Vec<BaselineMetric>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineMetric {
    pub name: String,
    pub hits: u64,
    pub recall_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictorSummary {
    pub schema: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub lookahead_routes: usize,
    pub budget_experts: usize,
    pub windows: u64,
    pub label_experts: u64,
    pub current_cold_experts: u64,
    pub current_cold_bytes: u64,
    pub predictors: Vec<PredictorMetric>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictorMetric {
    pub name: String,
    pub predicted_experts: u64,
    pub label_hits: u64,
    pub cold_hits: u64,
    pub cold_bytes_hit: u64,
    pub label_recall_pct: f64,
    pub cold_recall_pct: f64,
    pub cold_bytes_reduction_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportSummary {
    pub schema: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub lookahead_routes: usize,
    pub budget_experts: usize,
    pub value_oracle: ValueOracleSummary,
    pub predictors: PredictorSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetMetaRecord {
    pub event: String,
    pub schema: String,
    pub feature_schema: String,
    pub source_trace: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub lookahead_routes: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub topk: usize,
    pub history_tokens: usize,
    pub prior_layers: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetSampleRecord {
    pub event: String,
    pub schema: String,
    pub feature_schema: String,
    pub sample_id: u64,
    pub request_seq: u64,
    pub request_label: String,
    pub step: usize,
    pub layer: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub topk: usize,
    pub current_experts: Vec<i32>,
    pub current_cold_experts: Vec<i32>,
    pub current_weights: Vec<f32>,
    pub current_weighted_experts: Vec<DatasetWeightedExpert>,
    pub current_weight_stats: DatasetWeightStats,
    pub previous_same_layer: Option<DatasetRouteSnapshot>,
    pub recent_same_layer_counts: Vec<DatasetExpertCount>,
    pub previous_layers_current_step: Vec<DatasetRouteSnapshot>,
    pub future_cold: Vec<DatasetColdLabel>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetWeightedExpert {
    pub expert: i32,
    pub rank: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetWeightStats {
    pub valid_experts: usize,
    pub weight_sum: f32,
    pub weight_max: f32,
    pub weight_min: f32,
    pub top1_top2_margin: f32,
    pub entropy: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetRouteSnapshot {
    pub request_seq: u64,
    pub request_label: String,
    pub step: usize,
    pub layer: usize,
    pub weighted_experts: Vec<DatasetWeightedExpert>,
    pub weight_stats: DatasetWeightStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetExpertCount {
    pub expert: i32,
    pub count: u32,
    pub weight_sum: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetColdLabel {
    pub delta: usize,
    pub layer: usize,
    pub expert: i32,
    pub bytes: u64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetSummary {
    pub schema: String,
    pub dataset_schema: String,
    pub feature_schema: String,
    pub source_trace: String,
    pub output: String,
    pub trace_routes: usize,
    pub filtered_routes: usize,
    pub filter: String,
    pub lookahead_routes: usize,
    pub samples: u64,
    pub skipped_no_future: u64,
    pub skipped_no_cold_labels: u64,
    pub num_layers: usize,
    pub num_experts: usize,
    pub topk: usize,
    pub history_tokens: usize,
    pub prior_layers: usize,
    pub cold_labels: u64,
    pub cold_bytes: u64,
}

pub fn oracle_summary(
    routes: &[ExpertRouteTraceEvent],
    trace_routes: usize,
    filter: &TraceFilter,
    lookahead_routes: usize,
    budget_experts: usize,
) -> OracleSummary {
    let mut label_experts = 0u64;
    let mut current_cold = 0u64;
    let mut coverable = 0u64;
    let mut coverable_cold = 0u64;

    for (idx, route) in routes.iter().enumerate() {
        let future = future_window(routes, idx, lookahead_routes);
        if future.is_empty() {
            continue;
        }
        let mut labels = Vec::new();
        let mut cold_labels = HashSet::new();
        for ev in future {
            for (pos, &eid) in ev.expert_ids.iter().enumerate().take(ev.topk) {
                if eid < 0 {
                    continue;
                }
                labels.push((ev.layer, eid));
                if !ev.hcs_hits.get(pos).copied().unwrap_or(false) {
                    cold_labels.insert((ev.layer, eid));
                }
            }
        }
        if labels.is_empty() {
            continue;
        }
        label_experts += labels.len() as u64;
        current_cold += cold_labels.len() as u64;

        let mut oracle_budget = HashSet::new();
        for key in labels.iter().copied() {
            if oracle_budget.len() >= budget_experts {
                break;
            }
            oracle_budget.insert(key);
        }
        for key in labels.iter() {
            if oracle_budget.contains(key) {
                coverable += 1;
            }
        }
        for key in cold_labels.iter() {
            if oracle_budget.contains(key) {
                coverable_cold += 1;
            }
        }
        let _ = route;
    }

    OracleSummary {
        schema: "krasis_expert_prefetch_oracle_v1".to_string(),
        trace_routes,
        filtered_routes: routes.len(),
        filter: filter.describe(),
        lookahead_routes,
        budget_experts,
        label_experts,
        current_cold_experts: current_cold,
        oracle_coverable_experts: coverable,
        oracle_coverable_cold_experts: coverable_cold,
        current_cold_rate_pct: pct(current_cold, label_experts),
        oracle_coverable_recall_pct: pct(coverable, label_experts),
        oracle_coverable_cold_reduction_pct: pct(coverable_cold, current_cold),
    }
}

pub fn value_oracle_summary(
    routes: &[ExpertRouteTraceEvent],
    trace_routes: usize,
    filter: &TraceFilter,
    lookahead_routes: usize,
    budget_experts: usize,
) -> ValueOracleSummary {
    let mut windows = 0u64;
    let mut label_experts = 0u64;
    let mut current_cold = 0u64;
    let mut current_cold_bytes = 0u64;
    let mut coverable_cold = 0u64;
    let mut coverable_cold_bytes = 0u64;

    for idx in 0..routes.len() {
        let future = future_window(routes, idx, lookahead_routes);
        if future.is_empty() {
            continue;
        }
        let mut labels = HashSet::new();
        let mut cold = HashMap::new();
        for (distance, ev) in future.iter().enumerate() {
            for (pos, &eid) in ev.expert_ids.iter().enumerate().take(ev.topk) {
                if eid < 0 {
                    continue;
                }
                let key = ExpertKey {
                    layer: ev.layer,
                    expert: eid,
                };
                labels.insert(key);
                if !ev.hcs_hits.get(pos).copied().unwrap_or(false) {
                    let bytes = cold_bytes_for_pos(ev);
                    let weight = ev.weights.get(pos).copied().unwrap_or(0.0) as f64;
                    cold.entry(key)
                        .and_modify(|candidate: &mut Candidate| {
                            candidate.bytes = candidate.bytes.saturating_add(bytes);
                            candidate.weight += weight;
                            candidate.distance = candidate.distance.min(distance + 1);
                        })
                        .or_insert(Candidate {
                            key,
                            bytes,
                            weight,
                            distance: distance + 1,
                        });
                }
            }
        }
        if labels.is_empty() {
            continue;
        }
        windows += 1;
        label_experts += labels.len() as u64;
        current_cold += cold.len() as u64;
        current_cold_bytes = current_cold_bytes
            .saturating_add(cold.values().map(|candidate| candidate.bytes).sum::<u64>());

        let selected = select_candidates(cold.values().cloned(), budget_experts);
        coverable_cold += selected.len() as u64;
        coverable_cold_bytes = coverable_cold_bytes.saturating_add(
            selected
                .iter()
                .map(|candidate| candidate.bytes)
                .sum::<u64>(),
        );
    }

    ValueOracleSummary {
        schema: "krasis_expert_prefetch_value_oracle_v1".to_string(),
        trace_routes,
        filtered_routes: routes.len(),
        filter: filter.describe(),
        lookahead_routes,
        budget_experts,
        windows,
        label_experts,
        current_cold_experts: current_cold,
        current_cold_bytes,
        value_coverable_cold_experts: coverable_cold,
        value_coverable_cold_bytes: coverable_cold_bytes,
        current_cold_rate_pct: pct(current_cold, label_experts),
        value_coverable_cold_reduction_pct: pct(coverable_cold, current_cold),
        value_coverable_cold_bytes_reduction_pct: pct(coverable_cold_bytes, current_cold_bytes),
    }
}

pub fn hcs_oracle_summary(
    routes: &[ExpertRouteTraceEvent],
    trace_routes: usize,
    predecodes: &[ExpertPredecodeTraceEvent],
    filter: &TraceFilter,
    budget_override: Option<usize>,
) -> HcsOracleSummary {
    let filtered_predecodes: Vec<&ExpertPredecodeTraceEvent> = predecodes
        .iter()
        .filter(|predecode| filter.matches_label(&predecode.request_label))
        .collect();
    let mut predecode_by_request: HashMap<RequestKey, &ExpertPredecodeTraceEvent> = HashMap::new();
    for predecode in filtered_predecodes.iter().copied() {
        predecode_by_request.insert(
            request_key(predecode.request_seq, &predecode.request_label),
            predecode,
        );
    }

    let mut routes_by_request: BTreeMap<RequestKey, Vec<&ExpertRouteTraceEvent>> = BTreeMap::new();
    for route in routes.iter().filter(|route| filter.matches(route)) {
        routes_by_request
            .entry(request_key(route.request_seq, &route.request_label))
            .or_default()
            .push(route);
    }

    let mut request_summaries = Vec::new();
    let requests_with_predecode_budget = 0u64;
    let mut total_decode_experts = 0u64;
    let mut current_hcs_hits = 0u64;
    let mut current_cold_experts = 0u64;
    let mut current_cold_bytes = 0u64;
    let mut oracle_hit_hits = 0u64;
    let mut oracle_hit_cold_experts = 0u64;
    let mut oracle_hit_cold_bytes = 0u64;
    let mut oracle_cold_hits = 0u64;
    let mut oracle_cold_cold_experts = 0u64;
    let mut oracle_cold_cold_bytes = 0u64;

    for (key, req_routes) in routes_by_request {
        if req_routes.is_empty() {
            continue;
        }
        let predecode = predecode_by_request.get(&key).copied();
        let budget = budget_override.unwrap_or(0);
        if budget == 0 {
            continue;
        }
        let summary = hcs_oracle_request_summary(
            &key,
            &req_routes,
            budget,
            predecode.map(|predecode| predecode.hcs_total),
        );
        total_decode_experts += summary.total_decode_experts;
        current_hcs_hits += summary.current_hcs_hits;
        current_cold_experts += summary.current_cold_experts;
        current_cold_bytes += summary.current_cold_bytes;
        oracle_hit_hits += summary.oracle_hit_hits;
        oracle_hit_cold_experts += summary.oracle_hit_cold_experts;
        oracle_hit_cold_bytes += summary.oracle_hit_cold_bytes;
        oracle_cold_hits += summary.oracle_cold_hits;
        oracle_cold_cold_experts += summary.oracle_cold_cold_experts;
        oracle_cold_cold_bytes += summary.oracle_cold_cold_bytes;
        request_summaries.push(summary);
    }

    HcsOracleSummary {
        schema: "krasis_hcs_request_heatmap_oracle_v1".to_string(),
        trace_routes,
        filtered_routes: routes.iter().filter(|route| filter.matches(route)).count(),
        filtered_predecodes: filtered_predecodes.len(),
        filter: filter.describe(),
        budget_source: if budget_override.is_some() {
            "override".to_string()
        } else {
            "required_override_missing".to_string()
        },
        budget_override,
        requests: request_summaries.len() as u64,
        requests_with_predecode_budget,
        total_decode_experts,
        current_hcs_hits,
        current_cold_experts,
        current_cold_bytes,
        oracle_hit_hits,
        oracle_hit_cold_experts,
        oracle_hit_cold_bytes,
        oracle_cold_hits,
        oracle_cold_cold_experts,
        oracle_cold_cold_bytes,
        current_hcs_hit_rate_pct: pct(current_hcs_hits, total_decode_experts),
        oracle_hit_hit_rate_pct: pct(oracle_hit_hits, total_decode_experts),
        oracle_cold_hit_rate_pct: pct(oracle_cold_hits, total_decode_experts),
        oracle_hit_cold_expert_reduction_pct: pct(
            current_cold_experts.saturating_sub(oracle_hit_cold_experts),
            current_cold_experts,
        ),
        oracle_cold_cold_expert_reduction_pct: pct(
            current_cold_experts.saturating_sub(oracle_cold_cold_experts),
            current_cold_experts,
        ),
        oracle_hit_cold_bytes_reduction_pct: pct(
            current_cold_bytes.saturating_sub(oracle_hit_cold_bytes),
            current_cold_bytes,
        ),
        oracle_cold_cold_bytes_reduction_pct: pct(
            current_cold_bytes.saturating_sub(oracle_cold_cold_bytes),
            current_cold_bytes,
        ),
        oracle_hit_dma_bound_speedup_x: speedup_ratio(current_cold_bytes, oracle_hit_cold_bytes),
        oracle_cold_dma_bound_speedup_x: speedup_ratio(current_cold_bytes, oracle_cold_cold_bytes),
        per_request: request_summaries,
    }
}

fn hcs_oracle_request_summary(
    key: &RequestKey,
    routes: &[&ExpertRouteTraceEvent],
    budget: usize,
    hcs_total: Option<usize>,
) -> HcsOracleRequestSummary {
    let mut use_scores: HashMap<ExpertKey, (u64, f64)> = HashMap::new();
    let mut cold_scores: HashMap<ExpertKey, (u64, f64)> = HashMap::new();
    let mut total_decode_experts = 0u64;
    let mut current_hcs_hits = 0u64;
    let mut current_cold_experts = 0u64;
    let mut current_cold_bytes = 0u64;
    let mut expert_bytes_samples = Vec::new();
    let mut steps = HashSet::new();

    for route in routes {
        steps.insert(route.step);
        if route.cold_experts > 0 && route.cold_bytes > 0 {
            expert_bytes_samples.push(route.cold_bytes / route.cold_experts as u64);
        }
        let expert_bytes = cold_bytes_for_pos(route);
        for (pos, &eid) in route.expert_ids.iter().enumerate().take(route.topk) {
            if eid < 0 {
                continue;
            }
            total_decode_experts += 1;
            let key = ExpertKey {
                layer: route.layer,
                expert: eid,
            };
            let weight = route.weights.get(pos).copied().unwrap_or(0.0) as f64;
            use_scores
                .entry(key)
                .and_modify(|score| {
                    score.0 = score.0.saturating_add(1);
                    score.1 += weight;
                })
                .or_insert((1, weight));
            if route.hcs_hits.get(pos).copied().unwrap_or(false) {
                current_hcs_hits += 1;
            } else {
                current_cold_experts += 1;
                current_cold_bytes = current_cold_bytes.saturating_add(expert_bytes);
                cold_scores
                    .entry(key)
                    .and_modify(|score| {
                        score.0 = score.0.saturating_add(expert_bytes);
                        score.1 += weight;
                    })
                    .or_insert((expert_bytes, weight));
            }
        }
    }

    let expert_bytes = expert_bytes_samples.into_iter().max().unwrap_or(0);
    let oracle_hit_set = select_score_keys(use_scores, budget, false);
    let oracle_cold_set = select_score_keys(cold_scores, budget, true);
    let (oracle_hit_hits, oracle_hit_cold_experts, oracle_hit_cold_bytes) =
        score_hcs_selection(routes, &oracle_hit_set, expert_bytes);
    let (oracle_cold_hits, oracle_cold_cold_experts, oracle_cold_cold_bytes) =
        score_hcs_selection(routes, &oracle_cold_set, expert_bytes);

    HcsOracleRequestSummary {
        request_seq: key.request_seq,
        request_label: key.request_label.clone(),
        routes: routes.len(),
        steps: steps.len(),
        hcs_budget: budget,
        hcs_total,
        expert_bytes,
        total_decode_experts,
        current_hcs_hits,
        current_cold_experts,
        current_cold_bytes,
        oracle_hit_hits,
        oracle_hit_cold_experts,
        oracle_hit_cold_bytes,
        oracle_cold_hits,
        oracle_cold_cold_experts,
        oracle_cold_cold_bytes,
        current_hcs_hit_rate_pct: pct(current_hcs_hits, total_decode_experts),
        oracle_hit_hit_rate_pct: pct(oracle_hit_hits, total_decode_experts),
        oracle_cold_hit_rate_pct: pct(oracle_cold_hits, total_decode_experts),
        oracle_hit_cold_bytes_reduction_pct: pct(
            current_cold_bytes.saturating_sub(oracle_hit_cold_bytes),
            current_cold_bytes,
        ),
        oracle_cold_cold_bytes_reduction_pct: pct(
            current_cold_bytes.saturating_sub(oracle_cold_cold_bytes),
            current_cold_bytes,
        ),
    }
}

fn select_score_keys(
    scores: HashMap<ExpertKey, (u64, f64)>,
    budget: usize,
    prefer_value: bool,
) -> HashSet<ExpertKey> {
    let mut ranked: Vec<(ExpertKey, u64, f64)> = scores
        .into_iter()
        .map(|(key, (count_or_bytes, weight))| (key, count_or_bytes, weight))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal))
            .then_with(|| a.0.layer.cmp(&b.0.layer))
            .then_with(|| a.0.expert.cmp(&b.0.expert))
    });
    if prefer_value {
        ranked.retain(|(_, value, _)| *value > 0);
    }
    ranked.truncate(budget);
    ranked.into_iter().map(|(key, _, _)| key).collect()
}

fn score_hcs_selection(
    routes: &[&ExpertRouteTraceEvent],
    selected: &HashSet<ExpertKey>,
    expert_bytes: u64,
) -> (u64, u64, u64) {
    let mut hits = 0u64;
    let mut cold_experts = 0u64;
    let mut cold_bytes = 0u64;
    for route in routes {
        let route_expert_bytes = if route.cold_experts > 0 {
            cold_bytes_for_pos(route)
        } else {
            expert_bytes
        };
        for &eid in route.expert_ids.iter().take(route.topk) {
            if eid < 0 {
                continue;
            }
            let key = ExpertKey {
                layer: route.layer,
                expert: eid,
            };
            if selected.contains(&key) {
                hits += 1;
            } else {
                cold_experts += 1;
                cold_bytes = cold_bytes.saturating_add(route_expert_bytes);
            }
        }
    }
    (hits, cold_experts, cold_bytes)
}

fn speedup_ratio(current_bytes: u64, oracle_bytes: u64) -> f64 {
    if current_bytes == 0 {
        0.0
    } else if oracle_bytes == 0 {
        f64::INFINITY
    } else {
        current_bytes as f64 / oracle_bytes as f64
    }
}

fn future_window(
    routes: &[ExpertRouteTraceEvent],
    idx: usize,
    lookahead_routes: usize,
) -> &[ExpertRouteTraceEvent] {
    if lookahead_routes == 0 || idx + 1 >= routes.len() {
        return &[];
    }
    let current = &routes[idx];
    let mut end = idx + 1;
    let max_end = (idx + 1 + lookahead_routes).min(routes.len());
    while end < max_end
        && routes[end].request_seq == current.request_seq
        && routes[end].request_label == current.request_label
        && routes[end].step == current.step
    {
        end += 1;
    }
    &routes[idx + 1..end]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExpertKey {
    layer: usize,
    expert: i32,
}

#[derive(Debug, Clone)]
struct Candidate {
    key: ExpertKey,
    bytes: u64,
    weight: f64,
    distance: usize,
}

fn cold_bytes_for_pos(route: &ExpertRouteTraceEvent) -> u64 {
    if route.cold_experts == 0 {
        0
    } else {
        route.cold_bytes / route.cold_experts as u64
    }
}

fn select_candidates(
    candidates: impl IntoIterator<Item = Candidate>,
    budget_experts: usize,
) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = candidates.into_iter().collect();
    candidates.sort_by(|a, b| {
        b.bytes
            .cmp(&a.bytes)
            .then_with(|| a.distance.cmp(&b.distance))
            .then_with(|| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal))
            .then_with(|| a.key.layer.cmp(&b.key.layer))
            .then_with(|| a.key.expert.cmp(&b.key.expert))
    });
    candidates.truncate(budget_experts);
    candidates
}

fn future_labels(
    routes: &[ExpertRouteTraceEvent],
    idx: usize,
    lookahead_routes: usize,
) -> (HashSet<ExpertKey>, HashMap<ExpertKey, u64>) {
    let mut labels = HashSet::new();
    let mut cold = HashMap::new();
    for ev in future_window(routes, idx, lookahead_routes) {
        for (pos, &eid) in ev.expert_ids.iter().enumerate().take(ev.topk) {
            if eid < 0 {
                continue;
            }
            let key = ExpertKey {
                layer: ev.layer,
                expert: eid,
            };
            labels.insert(key);
            if !ev.hcs_hits.get(pos).copied().unwrap_or(false) {
                cold.entry(key)
                    .and_modify(|bytes: &mut u64| {
                        *bytes = bytes.saturating_add(cold_bytes_for_pos(ev));
                    })
                    .or_insert_with(|| cold_bytes_for_pos(ev));
            }
        }
    }
    (labels, cold)
}

pub fn baseline_summary(
    routes: &[ExpertRouteTraceEvent],
    trace_routes: usize,
    filter: &TraceFilter,
    budget_experts: usize,
) -> BaselineSummary {
    let mut previous_same_layer: HashMap<usize, Vec<i32>> = HashMap::new();
    let mut lru_by_layer: HashMap<usize, VecDeque<i32>> = HashMap::new();
    let mut previous_route: Option<Vec<i32>> = None;

    let mut label_experts = 0u64;
    let mut last_same_layer_hits = 0u64;
    let mut per_layer_lru_hits = 0u64;
    let mut previous_route_hits = 0u64;

    for route in routes {
        let labels: Vec<i32> = route
            .expert_ids
            .iter()
            .take(route.topk)
            .copied()
            .filter(|&eid| eid >= 0)
            .collect();
        if labels.is_empty() {
            continue;
        }
        label_experts += labels.len() as u64;

        if let Some(predicted) = previous_same_layer.get(&route.layer) {
            last_same_layer_hits += count_hits(&labels, predicted, budget_experts);
        }
        if let Some(cache) = lru_by_layer.get(&route.layer) {
            let predicted: Vec<i32> = cache.iter().take(budget_experts).copied().collect();
            per_layer_lru_hits += count_hits(&labels, &predicted, budget_experts);
        }
        if let Some(predicted) = previous_route.as_ref() {
            previous_route_hits += count_hits(&labels, predicted, budget_experts);
        }

        previous_same_layer.insert(route.layer, labels.clone());
        let cache = lru_by_layer.entry(route.layer).or_default();
        for &eid in labels.iter().rev() {
            if let Some(pos) = cache.iter().position(|&v| v == eid) {
                cache.remove(pos);
            }
            cache.push_front(eid);
        }
        while cache.len() > budget_experts {
            cache.pop_back();
        }
        previous_route = Some(labels);
    }

    BaselineSummary {
        schema: "krasis_expert_prefetch_baseline_v1".to_string(),
        trace_routes,
        filtered_routes: routes.len(),
        filter: filter.describe(),
        budget_experts,
        label_experts,
        baselines: vec![
            metric("previous_same_layer", last_same_layer_hits, label_experts),
            metric("per_layer_lru", per_layer_lru_hits, label_experts),
            metric("previous_route", previous_route_hits, label_experts),
        ],
    }
}

pub fn predictor_summary(
    routes: &[ExpertRouteTraceEvent],
    trace_routes: usize,
    filter: &TraceFilter,
    lookahead_routes: usize,
    budget_experts: usize,
) -> PredictorSummary {
    let mut last_by_layer: HashMap<usize, Vec<i32>> = HashMap::new();
    let mut last_cold_by_layer: HashMap<usize, Vec<i32>> = HashMap::new();
    let mut lru_by_layer: HashMap<usize, VecDeque<i32>> = HashMap::new();
    let mut cold_lru_by_layer: HashMap<usize, VecDeque<i32>> = HashMap::new();
    let mut freq_by_layer: HashMap<usize, HashMap<i32, u64>> = HashMap::new();
    let mut transition_counts: HashMap<(usize, i32, usize), HashMap<i32, u64>> = HashMap::new();
    let mut cold_transition_counts: HashMap<(usize, i32, usize), HashMap<i32, u64>> =
        HashMap::new();

    let mut windows = 0u64;
    let mut label_experts = 0u64;
    let mut current_cold = 0u64;
    let mut current_cold_bytes = 0u64;
    let mut metrics = vec![
        PredictorAccum::new("future_previous_same_layer"),
        PredictorAccum::new("future_previous_cold_same_layer"),
        PredictorAccum::new("future_per_layer_lru"),
        PredictorAccum::new("future_per_layer_cold_lru"),
        PredictorAccum::new("future_per_layer_frequency"),
        PredictorAccum::new("route_transition_online"),
        PredictorAccum::new("cold_route_transition_online"),
        PredictorAccum::new("route_transition_plus_lru"),
    ];

    for idx in 0..routes.len() {
        let route = &routes[idx];
        let future = future_window(routes, idx, lookahead_routes);
        if !future.is_empty() {
            let (labels, cold) = future_labels(routes, idx, lookahead_routes);
            if !labels.is_empty() {
                windows += 1;
                label_experts += labels.len() as u64;
                current_cold += cold.len() as u64;
                current_cold_bytes =
                    current_cold_bytes.saturating_add(cold.values().copied().sum::<u64>());

                let previous =
                    previous_same_layer_predictions(future, &last_by_layer, budget_experts);
                metrics[0].score(&previous, &labels, &cold);

                let previous_cold =
                    previous_same_layer_predictions(future, &last_cold_by_layer, budget_experts);
                metrics[1].score(&previous_cold, &labels, &cold);

                let lru = per_layer_lru_predictions(future, &lru_by_layer, budget_experts);
                metrics[2].score(&lru, &labels, &cold);

                let cold_lru =
                    per_layer_lru_predictions(future, &cold_lru_by_layer, budget_experts);
                metrics[3].score(&cold_lru, &labels, &cold);

                let freq = per_layer_frequency_predictions(future, &freq_by_layer, budget_experts);
                metrics[4].score(&freq, &labels, &cold);

                let transition = route_transition_predictions(
                    route,
                    &transition_counts,
                    lookahead_routes,
                    budget_experts,
                );
                metrics[5].score(&transition, &labels, &cold);

                let cold_transition = route_transition_predictions(
                    route,
                    &cold_transition_counts,
                    lookahead_routes,
                    budget_experts,
                );
                metrics[6].score(&cold_transition, &labels, &cold);

                let hybrid = merge_scored_predictions(
                    route_transition_scored(route, &transition_counts, lookahead_routes, 1.0),
                    per_layer_lru_scored(future, &cold_lru_by_layer, 0.15),
                    budget_experts,
                );
                metrics[7].score(&hybrid, &labels, &cold);
            }
        }

        update_predictor_state(
            route,
            future,
            &mut last_by_layer,
            &mut last_cold_by_layer,
            &mut lru_by_layer,
            &mut cold_lru_by_layer,
            &mut freq_by_layer,
            &mut transition_counts,
            &mut cold_transition_counts,
        );
    }

    PredictorSummary {
        schema: "krasis_expert_prefetch_predictors_v1".to_string(),
        trace_routes,
        filtered_routes: routes.len(),
        filter: filter.describe(),
        lookahead_routes,
        budget_experts,
        windows,
        label_experts,
        current_cold_experts: current_cold,
        current_cold_bytes,
        predictors: metrics
            .into_iter()
            .map(|metric| metric.finish(label_experts, current_cold, current_cold_bytes))
            .collect(),
    }
}

pub fn write_dataset(
    routes: &[ExpertRouteTraceEvent],
    trace_routes: usize,
    filter: &TraceFilter,
    source_trace: &str,
    output_path: impl AsRef<Path>,
    lookahead_routes: usize,
    max_samples: Option<u64>,
    history_tokens: usize,
    prior_layers: usize,
) -> Result<DatasetSummary, String> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create dataset dir {:?}: {}", parent, e))?;
        }
    }
    let file = File::create(output_path)
        .map_err(|e| format!("create expert prefetch dataset {:?}: {}", output_path, e))?;
    let mut writer = BufWriter::new(file);

    let num_layers = routes
        .iter()
        .map(|route| route.layer)
        .max()
        .map(|layer| layer + 1)
        .unwrap_or(0);
    let num_experts = routes
        .iter()
        .map(|route| route.num_experts)
        .max()
        .unwrap_or(0);
    let topk = routes.iter().map(|route| route.topk).max().unwrap_or(0);
    let meta = DatasetMetaRecord {
        event: "meta".to_string(),
        schema: DATASET_SCHEMA_V2.to_string(),
        feature_schema: "route_history_v1".to_string(),
        source_trace: source_trace.to_string(),
        trace_routes,
        filtered_routes: routes.len(),
        filter: filter.describe(),
        lookahead_routes,
        num_layers,
        num_experts,
        topk,
        history_tokens,
        prior_layers,
    };
    serde_json::to_writer(&mut writer, &meta).map_err(|e| format!("write dataset meta: {}", e))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write dataset meta newline: {}", e))?;

    let mut samples = 0u64;
    let mut skipped_no_future = 0u64;
    let mut skipped_no_cold_labels = 0u64;
    let mut cold_labels = 0u64;
    let mut cold_bytes = 0u64;
    let mut previous_same_layer: HashMap<usize, DatasetRouteSnapshot> = HashMap::new();
    let mut recent_same_layer: HashMap<usize, VecDeque<DatasetRouteSnapshot>> = HashMap::new();
    let mut previous_layers_current_step: VecDeque<DatasetRouteSnapshot> = VecDeque::new();
    let mut current_step_key: Option<(u64, String, usize)> = None;

    for (idx, route) in routes.iter().enumerate() {
        if max_samples.map(|max| samples >= max).unwrap_or(false) {
            break;
        }
        let step_key = (route.request_seq, route.request_label.clone(), route.step);
        if current_step_key.as_ref() != Some(&step_key) {
            previous_layers_current_step.clear();
            current_step_key = Some(step_key);
        }
        let snapshot = route_snapshot(route);
        let previous_same = previous_same_layer.get(&route.layer).cloned();
        let recent_counts =
            same_layer_history_counts(recent_same_layer.get(&route.layer), history_tokens);
        let prior_layer_snapshots = previous_layers_current_step
            .iter()
            .rev()
            .take(prior_layers)
            .cloned()
            .collect::<Vec<_>>();
        let future = future_window(routes, idx, lookahead_routes);
        if future.is_empty() {
            skipped_no_future += 1;
            update_dataset_history(
                snapshot,
                &mut previous_same_layer,
                &mut recent_same_layer,
                &mut previous_layers_current_step,
                history_tokens,
                prior_layers,
            );
            continue;
        }
        let labels = dataset_future_cold_labels(future);
        if labels.is_empty() {
            skipped_no_cold_labels += 1;
            update_dataset_history(
                snapshot,
                &mut previous_same_layer,
                &mut recent_same_layer,
                &mut previous_layers_current_step,
                history_tokens,
                prior_layers,
            );
            continue;
        }
        let current_experts: Vec<i32> = route
            .expert_ids
            .iter()
            .take(route.topk)
            .copied()
            .filter(|&expert| expert >= 0)
            .collect();
        if current_experts.is_empty() {
            update_dataset_history(
                snapshot,
                &mut previous_same_layer,
                &mut recent_same_layer,
                &mut previous_layers_current_step,
                history_tokens,
                prior_layers,
            );
            continue;
        }
        let current_cold_experts: Vec<i32> = route
            .expert_ids
            .iter()
            .enumerate()
            .take(route.topk)
            .filter_map(|(pos, &expert)| {
                if expert >= 0 && !route.hcs_hits.get(pos).copied().unwrap_or(false) {
                    Some(expert)
                } else {
                    None
                }
            })
            .collect();
        cold_labels += labels.len() as u64;
        cold_bytes = cold_bytes.saturating_add(labels.iter().map(|label| label.bytes).sum::<u64>());
        let sample = DatasetSampleRecord {
            event: "sample".to_string(),
            schema: DATASET_SCHEMA_V2.to_string(),
            feature_schema: "route_history_v1".to_string(),
            sample_id: samples,
            request_seq: route.request_seq,
            request_label: route.request_label.clone(),
            step: route.step,
            layer: route.layer,
            num_layers,
            num_experts: route.num_experts,
            topk: route.topk,
            current_experts,
            current_cold_experts,
            current_weights: route.weights.iter().take(route.topk).copied().collect(),
            current_weighted_experts: snapshot.weighted_experts.clone(),
            current_weight_stats: snapshot.weight_stats.clone(),
            previous_same_layer: previous_same,
            recent_same_layer_counts: recent_counts,
            previous_layers_current_step: prior_layer_snapshots,
            future_cold: labels,
        };
        serde_json::to_writer(&mut writer, &sample)
            .map_err(|e| format!("write dataset sample {}: {}", samples, e))?;
        writer
            .write_all(b"\n")
            .map_err(|e| format!("write dataset sample newline: {}", e))?;
        samples += 1;
        update_dataset_history(
            snapshot,
            &mut previous_same_layer,
            &mut recent_same_layer,
            &mut previous_layers_current_step,
            history_tokens,
            prior_layers,
        );
    }
    writer
        .flush()
        .map_err(|e| format!("flush dataset {:?}: {}", output_path, e))?;

    Ok(DatasetSummary {
        schema: "krasis_expert_prefetch_dataset_summary_v1".to_string(),
        dataset_schema: DATASET_SCHEMA_V2.to_string(),
        feature_schema: "route_history_v1".to_string(),
        source_trace: source_trace.to_string(),
        output: output_path.display().to_string(),
        trace_routes,
        filtered_routes: routes.len(),
        filter: filter.describe(),
        lookahead_routes,
        samples,
        skipped_no_future,
        skipped_no_cold_labels,
        num_layers,
        num_experts,
        topk,
        history_tokens,
        prior_layers,
        cold_labels,
        cold_bytes,
    })
}

fn route_snapshot(route: &ExpertRouteTraceEvent) -> DatasetRouteSnapshot {
    let weighted_experts = weighted_experts(route);
    DatasetRouteSnapshot {
        request_seq: route.request_seq,
        request_label: route.request_label.clone(),
        step: route.step,
        layer: route.layer,
        weight_stats: weight_stats(&weighted_experts),
        weighted_experts,
    }
}

fn weighted_experts(route: &ExpertRouteTraceEvent) -> Vec<DatasetWeightedExpert> {
    route
        .expert_ids
        .iter()
        .enumerate()
        .take(route.topk)
        .filter_map(|(rank, &expert)| {
            if expert < 0 {
                return None;
            }
            Some(DatasetWeightedExpert {
                expert,
                rank,
                weight: route.weights.get(rank).copied().unwrap_or(0.0),
            })
        })
        .collect()
}

fn weight_stats(weighted: &[DatasetWeightedExpert]) -> DatasetWeightStats {
    let valid_experts = weighted.len();
    let weight_sum: f32 = weighted.iter().map(|item| item.weight).sum();
    let weight_max = weighted
        .iter()
        .map(|item| item.weight)
        .fold(0.0f32, f32::max);
    let weight_min = if weighted.is_empty() {
        0.0
    } else {
        weighted
            .iter()
            .map(|item| item.weight)
            .fold(f32::INFINITY, f32::min)
    };
    let top1 = weighted.first().map(|item| item.weight).unwrap_or(0.0);
    let top2 = weighted.get(1).map(|item| item.weight).unwrap_or(0.0);
    let entropy = if weight_sum > 0.0 {
        weighted
            .iter()
            .filter_map(|item| {
                let p = item.weight / weight_sum;
                if p > 0.0 {
                    Some(-p * p.ln())
                } else {
                    None
                }
            })
            .sum()
    } else {
        0.0
    };
    DatasetWeightStats {
        valid_experts,
        weight_sum,
        weight_max,
        weight_min,
        top1_top2_margin: top1 - top2,
        entropy,
    }
}

fn same_layer_history_counts(
    history: Option<&VecDeque<DatasetRouteSnapshot>>,
    history_tokens: usize,
) -> Vec<DatasetExpertCount> {
    let mut counts: HashMap<i32, DatasetExpertCount> = HashMap::new();
    if let Some(history) = history {
        for snapshot in history.iter().rev().take(history_tokens) {
            for expert in snapshot.weighted_experts.iter() {
                counts
                    .entry(expert.expert)
                    .and_modify(|count| {
                        count.count = count.count.saturating_add(1);
                        count.weight_sum += expert.weight;
                    })
                    .or_insert(DatasetExpertCount {
                        expert: expert.expert,
                        count: 1,
                        weight_sum: expert.weight,
                    });
            }
        }
    }
    let mut values: Vec<DatasetExpertCount> = counts.into_values().collect();
    values.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| {
                b.weight_sum
                    .partial_cmp(&a.weight_sum)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| a.expert.cmp(&b.expert))
    });
    values
}

fn update_dataset_history(
    snapshot: DatasetRouteSnapshot,
    previous_same_layer: &mut HashMap<usize, DatasetRouteSnapshot>,
    recent_same_layer: &mut HashMap<usize, VecDeque<DatasetRouteSnapshot>>,
    previous_layers_current_step: &mut VecDeque<DatasetRouteSnapshot>,
    history_tokens: usize,
    prior_layers: usize,
) {
    previous_same_layer.insert(snapshot.layer, snapshot.clone());
    let history = recent_same_layer.entry(snapshot.layer).or_default();
    history.push_back(snapshot.clone());
    while history.len() > history_tokens {
        history.pop_front();
    }
    previous_layers_current_step.push_back(snapshot);
    while previous_layers_current_step.len() > prior_layers {
        previous_layers_current_step.pop_front();
    }
}

fn dataset_future_cold_labels(future: &[ExpertRouteTraceEvent]) -> Vec<DatasetColdLabel> {
    let mut by_key: HashMap<ExpertKey, DatasetColdLabel> = HashMap::new();
    for (distance, ev) in future.iter().enumerate() {
        let delta = distance + 1;
        for (pos, &expert) in ev.expert_ids.iter().enumerate().take(ev.topk) {
            if expert < 0 || ev.hcs_hits.get(pos).copied().unwrap_or(false) {
                continue;
            }
            let key = ExpertKey {
                layer: ev.layer,
                expert,
            };
            let bytes = cold_bytes_for_pos(ev);
            let weight = ev.weights.get(pos).copied().unwrap_or(0.0) as f64;
            by_key
                .entry(key)
                .and_modify(|label| {
                    label.bytes = label.bytes.saturating_add(bytes);
                    label.weight += weight;
                    label.delta = label.delta.min(delta);
                })
                .or_insert(DatasetColdLabel {
                    delta,
                    layer: ev.layer,
                    expert,
                    bytes,
                    weight,
                });
        }
    }
    let mut labels: Vec<DatasetColdLabel> = by_key.into_values().collect();
    labels.sort_by(|a, b| {
        b.bytes
            .cmp(&a.bytes)
            .then_with(|| a.delta.cmp(&b.delta))
            .then_with(|| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal))
            .then_with(|| a.layer.cmp(&b.layer))
            .then_with(|| a.expert.cmp(&b.expert))
    });
    labels
}

fn previous_same_layer_predictions(
    future: &[ExpertRouteTraceEvent],
    last_by_layer: &HashMap<usize, Vec<i32>>,
    budget_experts: usize,
) -> Vec<ExpertKey> {
    let mut scored = Vec::new();
    for (distance, ev) in future.iter().enumerate() {
        if let Some(experts) = last_by_layer.get(&ev.layer) {
            for (rank, &expert) in experts.iter().enumerate() {
                scored.push(ScoredCandidate {
                    key: ExpertKey {
                        layer: ev.layer,
                        expert,
                    },
                    score: 1.0 / (rank + 1) as f64,
                    distance: distance + 1,
                });
            }
        }
    }
    select_scored(scored, budget_experts)
}

fn per_layer_lru_predictions(
    future: &[ExpertRouteTraceEvent],
    lru_by_layer: &HashMap<usize, VecDeque<i32>>,
    budget_experts: usize,
) -> Vec<ExpertKey> {
    select_scored(
        per_layer_lru_scored(future, lru_by_layer, 1.0),
        budget_experts,
    )
}

fn per_layer_lru_scored(
    future: &[ExpertRouteTraceEvent],
    lru_by_layer: &HashMap<usize, VecDeque<i32>>,
    scale: f64,
) -> Vec<ScoredCandidate> {
    let mut scored = Vec::new();
    for (distance, ev) in future.iter().enumerate() {
        if let Some(cache) = lru_by_layer.get(&ev.layer) {
            for (rank, &expert) in cache
                .iter()
                .take(MAX_PREDICTOR_CANDIDATES_PER_LAYER)
                .enumerate()
            {
                scored.push(ScoredCandidate {
                    key: ExpertKey {
                        layer: ev.layer,
                        expert,
                    },
                    score: scale / (rank + 1) as f64,
                    distance: distance + 1,
                });
            }
        }
    }
    scored
}

fn per_layer_frequency_predictions(
    future: &[ExpertRouteTraceEvent],
    freq_by_layer: &HashMap<usize, HashMap<i32, u64>>,
    budget_experts: usize,
) -> Vec<ExpertKey> {
    let mut scored = Vec::new();
    for (distance, ev) in future.iter().enumerate() {
        if let Some(counts) = freq_by_layer.get(&ev.layer) {
            let mut top: Vec<(i32, u64)> = counts
                .iter()
                .map(|(&expert, &count)| (expert, count))
                .collect();
            top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (expert, count) in top.into_iter().take(MAX_PREDICTOR_CANDIDATES_PER_LAYER) {
                scored.push(ScoredCandidate {
                    key: ExpertKey {
                        layer: ev.layer,
                        expert,
                    },
                    score: count as f64,
                    distance: distance + 1,
                });
            }
        }
    }
    select_scored(scored, budget_experts)
}

fn route_transition_predictions(
    route: &ExpertRouteTraceEvent,
    transition_counts: &HashMap<(usize, i32, usize), HashMap<i32, u64>>,
    lookahead_routes: usize,
    budget_experts: usize,
) -> Vec<ExpertKey> {
    select_scored(
        route_transition_scored(route, transition_counts, lookahead_routes, 1.0),
        budget_experts,
    )
}

fn route_transition_scored(
    route: &ExpertRouteTraceEvent,
    transition_counts: &HashMap<(usize, i32, usize), HashMap<i32, u64>>,
    lookahead_routes: usize,
    scale: f64,
) -> Vec<ScoredCandidate> {
    let current: Vec<i32> = route
        .expert_ids
        .iter()
        .take(route.topk)
        .copied()
        .filter(|&expert| expert >= 0)
        .collect();
    let mut scores: HashMap<ExpertKey, (f64, usize)> = HashMap::new();
    for delta in 1..=lookahead_routes {
        let dst_layer = route.layer + delta;
        for &src_expert in current.iter() {
            if let Some(counts) = transition_counts.get(&(route.layer, src_expert, delta)) {
                let mut top: Vec<(i32, u64)> = counts
                    .iter()
                    .map(|(&expert, &count)| (expert, count))
                    .collect();
                top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                for (dst_expert, count) in
                    top.into_iter().take(MAX_TRANSITION_CANDIDATES_PER_SOURCE)
                {
                    let key = ExpertKey {
                        layer: dst_layer,
                        expert: dst_expert,
                    };
                    scores
                        .entry(key)
                        .and_modify(|(score, distance)| {
                            *score += scale * count as f64;
                            *distance = (*distance).min(delta);
                        })
                        .or_insert((scale * count as f64, delta));
                }
            }
        }
    }
    scores
        .into_iter()
        .map(|(key, (score, distance))| ScoredCandidate {
            key,
            score,
            distance,
        })
        .collect()
}

fn merge_scored_predictions(
    left: Vec<ScoredCandidate>,
    right: Vec<ScoredCandidate>,
    budget_experts: usize,
) -> Vec<ExpertKey> {
    let mut scores: HashMap<ExpertKey, (f64, usize)> = HashMap::new();
    for candidate in left.into_iter().chain(right.into_iter()) {
        scores
            .entry(candidate.key)
            .and_modify(|(score, distance)| {
                *score += candidate.score;
                *distance = (*distance).min(candidate.distance);
            })
            .or_insert((candidate.score, candidate.distance));
    }
    select_scored(
        scores
            .into_iter()
            .map(|(key, (score, distance))| ScoredCandidate {
                key,
                score,
                distance,
            }),
        budget_experts,
    )
}

fn update_predictor_state(
    route: &ExpertRouteTraceEvent,
    future: &[ExpertRouteTraceEvent],
    last_by_layer: &mut HashMap<usize, Vec<i32>>,
    last_cold_by_layer: &mut HashMap<usize, Vec<i32>>,
    lru_by_layer: &mut HashMap<usize, VecDeque<i32>>,
    cold_lru_by_layer: &mut HashMap<usize, VecDeque<i32>>,
    freq_by_layer: &mut HashMap<usize, HashMap<i32, u64>>,
    transition_counts: &mut HashMap<(usize, i32, usize), HashMap<i32, u64>>,
    cold_transition_counts: &mut HashMap<(usize, i32, usize), HashMap<i32, u64>>,
) {
    let labels: Vec<i32> = route
        .expert_ids
        .iter()
        .take(route.topk)
        .copied()
        .filter(|&expert| expert >= 0)
        .collect();
    let cold_labels: Vec<i32> = route
        .expert_ids
        .iter()
        .enumerate()
        .take(route.topk)
        .filter_map(|(pos, &expert)| {
            if expert >= 0 && !route.hcs_hits.get(pos).copied().unwrap_or(false) {
                Some(expert)
            } else {
                None
            }
        })
        .collect();
    if labels.is_empty() {
        return;
    }

    for &src_expert in labels.iter() {
        for (distance, ev) in future.iter().enumerate() {
            let delta = distance + 1;
            let counts = transition_counts
                .entry((route.layer, src_expert, delta))
                .or_default();
            let cold_counts = cold_transition_counts
                .entry((route.layer, src_expert, delta))
                .or_default();
            for &dst_expert in ev.expert_ids.iter().take(ev.topk) {
                if dst_expert >= 0 {
                    *counts.entry(dst_expert).or_default() += 1;
                }
            }
            for (pos, &dst_expert) in ev.expert_ids.iter().enumerate().take(ev.topk) {
                if dst_expert >= 0 && !ev.hcs_hits.get(pos).copied().unwrap_or(false) {
                    *cold_counts.entry(dst_expert).or_default() += 1;
                }
            }
        }
    }

    last_by_layer.insert(route.layer, labels.clone());
    if !cold_labels.is_empty() {
        last_cold_by_layer.insert(route.layer, cold_labels.clone());
    }
    let cache = lru_by_layer.entry(route.layer).or_default();
    for &expert in labels.iter().rev() {
        if let Some(pos) = cache.iter().position(|&value| value == expert) {
            cache.remove(pos);
        }
        cache.push_front(expert);
    }
    let max_lru = route.num_experts.max(route.topk).max(1);
    while cache.len() > max_lru {
        cache.pop_back();
    }

    let cold_cache = cold_lru_by_layer.entry(route.layer).or_default();
    for &expert in cold_labels.iter().rev() {
        if let Some(pos) = cold_cache.iter().position(|&value| value == expert) {
            cold_cache.remove(pos);
        }
        cold_cache.push_front(expert);
    }
    while cold_cache.len() > max_lru {
        cold_cache.pop_back();
    }

    let counts = freq_by_layer.entry(route.layer).or_default();
    for expert in labels {
        *counts.entry(expert).or_default() += 1;
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredCandidate {
    key: ExpertKey,
    score: f64,
    distance: usize,
}

fn select_scored(
    candidates: impl IntoIterator<Item = ScoredCandidate>,
    budget_experts: usize,
) -> Vec<ExpertKey> {
    let mut best: HashMap<ExpertKey, (f64, usize)> = HashMap::new();
    for candidate in candidates {
        best.entry(candidate.key)
            .and_modify(|(score, distance)| {
                *score += candidate.score;
                *distance = (*distance).min(candidate.distance);
            })
            .or_insert((candidate.score, candidate.distance));
    }
    let mut candidates: Vec<ScoredCandidate> = best
        .into_iter()
        .map(|(key, (score, distance))| ScoredCandidate {
            key,
            score,
            distance,
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.distance.cmp(&b.distance))
            .then_with(|| a.key.layer.cmp(&b.key.layer))
            .then_with(|| a.key.expert.cmp(&b.key.expert))
    });
    candidates
        .into_iter()
        .take(budget_experts)
        .map(|candidate| candidate.key)
        .collect()
}

struct PredictorAccum {
    name: String,
    predicted_experts: u64,
    label_hits: u64,
    cold_hits: u64,
    cold_bytes_hit: u64,
}

impl PredictorAccum {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            predicted_experts: 0,
            label_hits: 0,
            cold_hits: 0,
            cold_bytes_hit: 0,
        }
    }

    fn score(
        &mut self,
        predicted: &[ExpertKey],
        labels: &HashSet<ExpertKey>,
        cold: &HashMap<ExpertKey, u64>,
    ) {
        self.predicted_experts += predicted.len() as u64;
        let predicted: HashSet<ExpertKey> = predicted.iter().copied().collect();
        self.label_hits += labels.iter().filter(|key| predicted.contains(key)).count() as u64;
        for (key, &bytes) in cold {
            if predicted.contains(key) {
                self.cold_hits += 1;
                self.cold_bytes_hit = self.cold_bytes_hit.saturating_add(bytes);
            }
        }
    }

    fn finish(
        self,
        label_experts: u64,
        current_cold: u64,
        current_cold_bytes: u64,
    ) -> PredictorMetric {
        PredictorMetric {
            name: self.name,
            predicted_experts: self.predicted_experts,
            label_hits: self.label_hits,
            cold_hits: self.cold_hits,
            cold_bytes_hit: self.cold_bytes_hit,
            label_recall_pct: pct(self.label_hits, label_experts),
            cold_recall_pct: pct(self.cold_hits, current_cold),
            cold_bytes_reduction_pct: pct(self.cold_bytes_hit, current_cold_bytes),
        }
    }
}

fn count_hits(labels: &[i32], predicted: &[i32], budget_experts: usize) -> u64 {
    let predicted: HashSet<i32> = predicted.iter().take(budget_experts).copied().collect();
    labels
        .iter()
        .filter(|&&eid| predicted.contains(&eid))
        .count() as u64
}

fn metric(name: &str, hits: u64, total: u64) -> BaselineMetric {
    BaselineMetric {
        name: name.to_string(),
        hits,
        recall_pct: pct(hits, total),
    }
}

fn pct(num: u64, denom: u64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        num as f64 / denom as f64 * 100.0
    }
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("Usage: expert_prefetch <oracle|value-oracle|hcs-oracle|baseline|predictors|report|dataset> <trace.jsonl> [--lookahead N] [--budget N] [--request-label LABEL] [--request-label-prefix PREFIX] [--out PATH] [--max-samples N] [--history-tokens N] [--prior-layers N] (hcs-oracle requires --budget)".to_string());
    }
    let command = args[0].as_str();
    let trace_path = &args[1];
    let mut lookahead = 5usize;
    let mut budget = 64usize;
    let mut budget_set = false;
    let mut filter = TraceFilter::default();
    let mut output_path: Option<String> = None;
    let mut max_samples: Option<u64> = None;
    let mut history_tokens = DEFAULT_DATASET_HISTORY_TOKENS;
    let mut prior_layers = DEFAULT_DATASET_PRIOR_LAYERS;
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--lookahead" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--lookahead requires a value")?;
                lookahead = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --lookahead {:?}: {}", raw, e))?;
            }
            "--budget" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--budget requires a value")?;
                budget = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --budget {:?}: {}", raw, e))?;
                budget_set = true;
            }
            "--request-label" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--request-label requires a value")?;
                filter.request_label = Some(raw.clone());
            }
            "--request-label-prefix" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or("--request-label-prefix requires a value")?;
                filter.request_label_prefix = Some(raw.clone());
            }
            "--out" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--out requires a value")?;
                output_path = Some(raw.clone());
            }
            "--max-samples" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--max-samples requires a value")?;
                max_samples = Some(
                    raw.parse::<u64>()
                        .map_err(|e| format!("invalid --max-samples {:?}: {}", raw, e))?,
                );
            }
            "--history-tokens" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--history-tokens requires a value")?;
                history_tokens = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --history-tokens {:?}: {}", raw, e))?;
            }
            "--prior-layers" => {
                idx += 1;
                let raw = args.get(idx).ok_or("--prior-layers requires a value")?;
                prior_layers = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --prior-layers {:?}: {}", raw, e))?;
            }
            other => return Err(format!("unknown expert_prefetch argument {:?}", other)),
        }
        idx += 1;
    }

    let hcs_oracle_command = matches!(
        command,
        "hcs-oracle" | "hcs_oracle" | "request-oracle" | "request_oracle"
    );
    if hcs_oracle_command && !budget_set {
        return Err(
            "hcs-oracle requires --budget <resident_experts>; trace predecode hcs_loaded is not HCS capacity"
                .to_string(),
        );
    }

    let trace_data = read_prefetch_trace(trace_path)?;
    let raw_routes = trace_data.routes;
    let trace_routes = raw_routes.len();
    let routes = filter_routes(&raw_routes, &filter);
    let value = match command {
        "oracle" => serde_json::to_value(oracle_summary(
            &routes,
            trace_routes,
            &filter,
            lookahead,
            budget,
        ))
        .map_err(|e| e.to_string())?,
        "value-oracle" | "value_oracle" => serde_json::to_value(value_oracle_summary(
            &routes,
            trace_routes,
            &filter,
            lookahead,
            budget,
        ))
        .map_err(|e| e.to_string())?,
        "hcs-oracle" | "hcs_oracle" | "request-oracle" | "request_oracle" => {
            serde_json::to_value(hcs_oracle_summary(
                &raw_routes,
                trace_routes,
                &trace_data.predecodes,
                &filter,
                budget_set.then_some(budget),
            ))
            .map_err(|e| e.to_string())?
        }
        "baseline" | "baselines" => {
            serde_json::to_value(baseline_summary(&routes, trace_routes, &filter, budget))
                .map_err(|e| e.to_string())?
        }
        "predictors" | "predictor" => serde_json::to_value(predictor_summary(
            &routes,
            trace_routes,
            &filter,
            lookahead,
            budget,
        ))
        .map_err(|e| e.to_string())?,
        "report" => {
            let value_oracle =
                value_oracle_summary(&routes, trace_routes, &filter, lookahead, budget);
            let predictors = predictor_summary(&routes, trace_routes, &filter, lookahead, budget);
            serde_json::to_value(ReportSummary {
                schema: "krasis_expert_prefetch_report_v1".to_string(),
                trace_routes,
                filtered_routes: routes.len(),
                filter: filter.describe(),
                lookahead_routes: lookahead,
                budget_experts: budget,
                value_oracle,
                predictors,
            })
            .map_err(|e| e.to_string())?
        }
        "dataset" => {
            let out = output_path
                .as_ref()
                .ok_or("dataset command requires --out PATH")?;
            serde_json::to_value(write_dataset(
                &routes,
                trace_routes,
                &filter,
                trace_path,
                out,
                lookahead,
                max_samples,
                history_tokens,
                prior_layers,
            )?)
            .map_err(|e| e.to_string())?
        }
        _ => return Err(format!("unknown expert_prefetch command {:?}", command)),
    };
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, &value)
        .map_err(|e| format!("write expert_prefetch result: {}", e))?;
    handle
        .write_all(b"\n")
        .map_err(|e| format!("write expert_prefetch newline: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(
        step: usize,
        layer: usize,
        expert_ids: &[i32],
        hcs_hits: &[bool],
    ) -> ExpertRouteTraceEvent {
        ExpertRouteTraceEvent {
            schema: TRACE_SCHEMA.to_string(),
            request_seq: 1,
            request_label: "test".to_string(),
            step,
            layer,
            num_experts: 128,
            topk: expert_ids.len(),
            expert_ids: expert_ids.to_vec(),
            weights: vec![1.0; expert_ids.len()],
            hcs_hits: hcs_hits.to_vec(),
            cold_experts: hcs_hits.iter().filter(|&&hit| !hit).count(),
            cold_bytes: 0,
        }
    }

    #[test]
    fn oracle_uses_same_step_future_routes() {
        let routes = vec![
            route(0, 0, &[1, 2], &[true, false]),
            route(0, 1, &[3, 4], &[false, false]),
            route(0, 2, &[5, 6], &[true, false]),
            route(1, 0, &[7, 8], &[false, false]),
        ];
        let summary = oracle_summary(&routes, routes.len(), &TraceFilter::default(), 2, 3);
        assert_eq!(summary.label_experts, 6);
        assert_eq!(summary.current_cold_experts, 4);
        assert_eq!(summary.oracle_coverable_cold_experts, 3);
    }

    #[test]
    fn baselines_score_previous_same_layer() {
        let routes = vec![
            route(0, 0, &[1, 2], &[true, true]),
            route(0, 1, &[9, 10], &[true, true]),
            route(1, 0, &[1, 3], &[true, true]),
        ];
        let summary = baseline_summary(&routes, routes.len(), &TraceFilter::default(), 2);
        let prev = summary
            .baselines
            .iter()
            .find(|m| m.name == "previous_same_layer")
            .unwrap();
        assert_eq!(prev.hits, 1);
    }

    #[test]
    fn value_oracle_prioritizes_cold_labels() {
        let routes = vec![
            route(0, 0, &[1, 2], &[true, true]),
            route(0, 1, &[3, 4], &[true, false]),
            route(0, 2, &[5, 6], &[false, false]),
        ];
        let summary = value_oracle_summary(&routes, routes.len(), &TraceFilter::default(), 2, 2);
        assert_eq!(summary.current_cold_experts, 3);
        assert_eq!(summary.value_coverable_cold_experts, 2);
    }

    #[test]
    fn predictors_score_future_layer_lru() {
        let routes = vec![
            route(0, 0, &[1, 2], &[true, true]),
            route(0, 1, &[9, 10], &[true, true]),
            route(1, 0, &[3, 4], &[true, true]),
            route(1, 1, &[9, 11], &[true, false]),
        ];
        let summary = predictor_summary(&routes, routes.len(), &TraceFilter::default(), 1, 2);
        let lru = summary
            .predictors
            .iter()
            .find(|m| m.name == "future_per_layer_lru")
            .unwrap();
        assert!(lru.label_hits >= 1);
    }
}
