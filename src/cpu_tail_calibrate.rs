//! Standalone CPU-tail startup calibration and optimizer.
//!
//! This module deliberately has no dependency on a loaded model or decode
//! store. It derives routed-expert geometry from an existing Krasis config,
//! synthesizes exact-sized INT4 buffers, measures CPU expert execution while a
//! real pinned-host H2D stream is active, and scores the measured operating
//! points against an externally supplied route-queue histogram.

use crate::cpu_tail::{expert_forward_transposed_persistent, PersistentTransposedTeam};
use crate::moe::{
    expert_forward_marlin_int4_cpu_tail, expert_forward_transposed_int4_cpu_tail, ExpertScratch,
};
use cudarc::driver::{sys as cuda_sys, CudaDevice};
use half::bf16;
use rayon::ThreadPool;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Cli {
    config_path: PathBuf,
    group_size_override: Option<usize>,
    gpu_ordinal: Option<usize>,
    gpu_uuid: Option<String>,
    histogram_logs: Vec<PathBuf>,
    output: PathBuf,
    budget_seconds: f64,
    baseline_ms_per_token: Option<f64>,
    isolated_reference_ms: Option<f64>,
    integrated_reference_ms: Option<f64>,
    two_team: bool,
    layer_gap_us: Option<u64>,
    smoke: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpertGeometry {
    config_path: String,
    model_config_path: String,
    model_name: String,
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
    experts_per_token: usize,
    expert_bytes: usize,
    w13_packed_words: usize,
    w13_scale_values: usize,
    w2_packed_words: usize,
    w2_scale_values: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CacheDomain {
    id: String,
    physical_cpus: Vec<usize>,
    size_bytes: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct CpuTopology {
    allowed_logical_cpus: Vec<usize>,
    physical_cpus: Vec<usize>,
    cache_domains: Vec<CacheDomain>,
    affinity_supported: bool,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct QueueHistogram {
    sources: Vec<String>,
    generated_tokens: u64,
    depth_counts: BTreeMap<String, u64>,
    queues_per_token: f64,
    cold_experts_per_token: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum WeightFormat {
    Transposed,
    MarlinDirect,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PlacementPolicy {
    Compact,
    Spread,
}

impl PlacementPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Spread => "spread",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct LatencySummary {
    samples: usize,
    mean_ms: f64,
    min_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    experts_per_second: f64,
    effective_weight_gib_s: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Prediction {
    handoff_margin_ms: f64,
    copy_ms_per_expert: f64,
    win_probability_by_depth: BTreeMap<String, f64>,
    absorbed_experts_per_token: f64,
    predicted_win_rate: f64,
    nominal_saved_ms_per_token: f64,
    dma_degradation_ms_per_token: f64,
    net_saved_ms_per_token: f64,
    predicted_baseline_tok_s: Option<f64>,
    predicted_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SweepMeasurement {
    phase: String,
    format: WeightFormat,
    threads: usize,
    placement: PlacementPolicy,
    cpu_ids: Vec<usize>,
    latency: LatencySummary,
    dma_gib_s: Option<f64>,
    dma_degradation_pct: Option<f64>,
    prediction: Option<Prediction>,
}

#[derive(Debug, Clone, Serialize)]
struct GapAttribution {
    isolated_reference_ms: Option<f64>,
    integrated_reference_ms: Option<f64>,
    observed_gap_ms: Option<f64>,
    best_measured_isolated_ms: f64,
    best_measured_concurrent_ms: f64,
    concurrency_explained_ms: f64,
    concurrency_explained_pct_of_observed_gap: Option<f64>,
    remaining_scheduler_handoff_ms: f64,
    remaining_pct_of_observed_gap: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OptimizerDecision {
    recommendation: String,
    reason: String,
    best_threads: usize,
    best_placement: PlacementPolicy,
    best_cpu_ids: Vec<usize>,
    raw_prediction: Prediction,
    conservative_prediction: Prediction,
    conservative_margin_ms_per_token: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TwoTeamPrediction {
    copy_ms_per_expert: f64,
    worker_0_win_probability_by_depth: BTreeMap<String, f64>,
    worker_1_win_probability_by_depth: BTreeMap<String, f64>,
    worker_0_absorbed_experts_per_token: f64,
    worker_1_absorbed_experts_per_token: f64,
    total_absorbed_experts_per_token: f64,
    nominal_saved_ms_per_token: f64,
    dma_degradation_ms_per_token: f64,
    net_saved_ms_per_token: f64,
    predicted_baseline_tok_s: Option<f64>,
    predicted_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct TwoTeamMeasurement {
    placement: PlacementPolicy,
    team_threads: [usize; 2],
    team_cpu_ids: [Vec<usize>; 2],
    team_latency: [LatencySummary; 2],
    dma_gib_s: f64,
    dma_degradation_pct: f64,
    prediction: TwoTeamPrediction,
}

#[derive(Debug, Clone, Serialize)]
struct PersistentSingleMeasurement {
    placement: PlacementPolicy,
    threads: usize,
    cpu_ids: Vec<usize>,
    latency: LatencySummary,
    dma_gib_s: f64,
    dma_degradation_pct: f64,
    prediction: Prediction,
}

#[derive(Debug, Clone, Serialize)]
struct TwoTeamOptimizerDecision {
    recommendation: String,
    reason: String,
    layer_gap_us: u64,
    best_placement: PlacementPolicy,
    best_team_threads: [usize; 2],
    best_team_cpu_ids: [Vec<usize>; 2],
    prediction: TwoTeamPrediction,
    matched_single_net_saved_ms_per_token: f64,
    incremental_net_saved_ms_per_token_vs_single: f64,
}

#[derive(Debug, Serialize)]
struct CalibrationReport {
    schema_version: u32,
    generated_unix_seconds: u64,
    smoke: bool,
    requested_budget_seconds: f64,
    elapsed_seconds: f64,
    geometry: ExpertGeometry,
    gpu_ordinal: usize,
    gpu_uuid: String,
    gpu_name: String,
    topology: CpuTopology,
    queue_histogram: QueueHistogram,
    replica_count: usize,
    synthetic_working_set_bytes: usize,
    dma_baseline_windows_gib_s: Vec<f64>,
    dma_baseline_gib_s: f64,
    cpu_only_sweep: Vec<SweepMeasurement>,
    concurrent_sweep: Vec<SweepMeasurement>,
    best_format_comparison: Vec<SweepMeasurement>,
    optimizer: OptimizerDecision,
    persistent_single_sweep: Vec<PersistentSingleMeasurement>,
    two_team_sweep: Vec<TwoTeamMeasurement>,
    two_team_optimizer: Option<TwoTeamOptimizerDecision>,
    gap_attribution: GapAttribution,
    skipped_or_truncated: Vec<String>,
}

struct SyntheticExpert {
    w13_packed: Vec<u32>,
    w13_scales: Vec<u16>,
    w2_packed: Vec<u32>,
    w2_scales: Vec<u16>,
}

struct HostPinned {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for HostPinned {}
unsafe impl Sync for HostPinned {}

impl HostPinned {
    fn new(bytes: usize) -> Result<Self, String> {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { cuda_sys::lib().cuMemHostAlloc(&mut ptr, bytes, 0) };
        check_cuda(rc, format!("cuMemHostAlloc({bytes})"))?;
        unsafe {
            std::ptr::write_bytes(ptr as *mut u8, 0x5a, bytes);
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            bytes,
        })
    }
}

impl Drop for HostPinned {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cuda_sys::lib().cuMemFreeHost(self.ptr.cast());
            }
        }
    }
}

struct DmaResources {
    device: Arc<CudaDevice>,
    host: HostPinned,
    device_ptr: cuda_sys::CUdeviceptr,
    stream: cuda_sys::CUstream,
}

unsafe impl Send for DmaResources {}
unsafe impl Sync for DmaResources {}

impl DmaResources {
    fn new(device: Arc<CudaDevice>, bytes: usize) -> Result<Self, String> {
        device
            .bind_to_thread()
            .map_err(|e| format!("bind CUDA context: {e}"))?;
        let host = HostPinned::new(bytes)?;
        let mut device_ptr = 0;
        check_cuda(
            unsafe { cuda_sys::lib().cuMemAlloc_v2(&mut device_ptr, bytes) },
            format!("cuMemAlloc_v2({bytes})"),
        )?;
        let mut stream = std::ptr::null_mut();
        if let Err(error) = check_cuda(
            unsafe {
                cuda_sys::lib().cuStreamCreate(
                    &mut stream,
                    cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "cuStreamCreate".to_string(),
        ) {
            unsafe {
                let _ = cuda_sys::lib().cuMemFree_v2(device_ptr);
            }
            return Err(error);
        }
        Ok(Self {
            device,
            host,
            device_ptr,
            stream,
        })
    }

    fn copy_once(&self) -> Result<(), String> {
        check_cuda(
            unsafe {
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    self.device_ptr,
                    self.host.ptr.cast(),
                    self.host.bytes,
                    self.stream,
                )
            },
            "cuMemcpyHtoDAsync_v2".to_string(),
        )?;
        check_cuda(
            unsafe { cuda_sys::lib().cuStreamSynchronize(self.stream) },
            "cuStreamSynchronize".to_string(),
        )
    }
}

impl Drop for DmaResources {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        unsafe {
            if !self.stream.is_null() {
                let _ = cuda_sys::lib().cuStreamSynchronize(self.stream);
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.stream);
            }
            if self.device_ptr != 0 {
                let _ = cuda_sys::lib().cuMemFree_v2(self.device_ptr);
            }
        }
    }
}

fn check_cuda(rc: cuda_sys::CUresult, operation: String) -> Result<(), String> {
    if rc == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed: {rc:?}"))
    }
}

fn usage() -> &'static str {
    "Usage: cpu-tail-calibrate <config> --output <report.json> --budget-seconds <seconds> \\
     [--gpu-ordinal N | --gpu-uuid GPU-...] [--group-size N] \\
     --histogram-log <cpu-tail.log> [--histogram-log <cpu-tail.log> ...] \\
     [--baseline-ms-per-token MS] [--isolated-reference-ms MS] \\
     [--integrated-reference-ms MS] [--two-team --layer-gap-us N] [--smoke]"
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        return Err(usage().to_string());
    }
    let config_path = PathBuf::from(&args[0]);
    let mut cli = Cli {
        config_path,
        group_size_override: None,
        gpu_ordinal: None,
        gpu_uuid: None,
        histogram_logs: Vec::new(),
        output: PathBuf::new(),
        budget_seconds: 0.0,
        baseline_ms_per_token: None,
        isolated_reference_ms: None,
        integrated_reference_ms: None,
        two_team: false,
        layer_gap_us: None,
        smoke: false,
    };
    let mut index = 1;
    while index < args.len() {
        let flag = &args[index];
        let next = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            args.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--group-size" => {
                cli.group_size_override = Some(
                    next(&mut index)?
                        .parse()
                        .map_err(|_| "invalid --group-size".to_string())?,
                )
            }
            "--gpu-ordinal" => {
                cli.gpu_ordinal = Some(
                    next(&mut index)?
                        .parse()
                        .map_err(|_| "invalid --gpu-ordinal".to_string())?,
                )
            }
            "--gpu-uuid" => cli.gpu_uuid = Some(next(&mut index)?.to_string()),
            "--histogram-log" => cli.histogram_logs.push(PathBuf::from(next(&mut index)?)),
            "--output" => cli.output = PathBuf::from(next(&mut index)?),
            "--budget-seconds" => {
                cli.budget_seconds = next(&mut index)?
                    .parse()
                    .map_err(|_| "invalid --budget-seconds".to_string())?
            }
            "--baseline-ms-per-token" => {
                cli.baseline_ms_per_token = Some(
                    next(&mut index)?
                        .parse()
                        .map_err(|_| "invalid --baseline-ms-per-token".to_string())?,
                )
            }
            "--isolated-reference-ms" => {
                cli.isolated_reference_ms = Some(
                    next(&mut index)?
                        .parse()
                        .map_err(|_| "invalid --isolated-reference-ms".to_string())?,
                )
            }
            "--integrated-reference-ms" => {
                cli.integrated_reference_ms = Some(
                    next(&mut index)?
                        .parse()
                        .map_err(|_| "invalid --integrated-reference-ms".to_string())?,
                )
            }
            "--two-team" => cli.two_team = true,
            "--layer-gap-us" => {
                cli.layer_gap_us = Some(
                    next(&mut index)?
                        .parse()
                        .map_err(|_| "invalid --layer-gap-us".to_string())?,
                )
            }
            "--smoke" => cli.smoke = true,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
        index += 1;
    }
    if !cli.config_path.is_file() {
        return Err(format!("config not found: {}", cli.config_path.display()));
    }
    if cli.output.as_os_str().is_empty() {
        return Err("--output is required".to_string());
    }
    if !cli.budget_seconds.is_finite() || cli.budget_seconds <= 0.0 {
        return Err("--budget-seconds must be positive and is required so the timed iteration count is derived from an explicit run budget".to_string());
    }
    if cli.histogram_logs.is_empty() {
        return Err("at least one --histogram-log is required; queue depths must come from recorded runtime evidence".to_string());
    }
    if cli.gpu_ordinal.is_some() && cli.gpu_uuid.is_some() {
        return Err("choose either --gpu-ordinal or --gpu-uuid, not both".to_string());
    }
    if cli.two_team && cli.layer_gap_us.is_none() {
        return Err(
            "--two-team requires --layer-gap-us from measured layer-cadence evidence".to_string(),
        );
    }
    if !cli.two_team && cli.layer_gap_us.is_some() {
        return Err("--layer-gap-us is only valid with --two-team".to_string());
    }
    Ok(cli)
}

fn parse_conf(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut values = BTreeMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        let unquoted = if value.len() >= 2
            && ((value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\'')))
        {
            &value[1..value.len() - 1]
        } else {
            value
        };
        values.insert(key.trim().to_string(), unquoted.to_string());
    }
    Ok(values)
}

fn json_usize(value: &Value, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(result) = value.get(*key).and_then(Value::as_u64) {
            return Some(result as usize);
        }
        if let Some(result) = value
            .get("text_config")
            .and_then(|nested| nested.get(*key))
            .and_then(Value::as_u64)
        {
            return Some(result as usize);
        }
    }
    None
}

fn checked_expert_geometry(cli: &Cli) -> Result<ExpertGeometry, String> {
    let conf = parse_conf(&cli.config_path)?;
    let model_path = conf
        .get("MODEL_PATH")
        .ok_or_else(|| format!("{} has no MODEL_PATH", cli.config_path.display()))?;
    let model_config_path = Path::new(model_path).join("config.json");
    let config: Value = serde_json::from_str(
        &fs::read_to_string(&model_config_path)
            .map_err(|e| format!("read {}: {e}", model_config_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", model_config_path.display()))?;
    let hidden_size = json_usize(&config, &["hidden_size"])
        .ok_or_else(|| "model config has no hidden_size".to_string())?;
    let intermediate_size = json_usize(
        &config,
        &[
            "moe_intermediate_size",
            "expert_intermediate_size",
            "intermediate_size",
        ],
    )
    .ok_or_else(|| "model config has no routed-expert intermediate size".to_string())?;
    let experts_per_token = json_usize(
        &config,
        &["num_experts_per_tok", "num_selected_experts", "moe_topk"],
    )
    .ok_or_else(|| "model config has no experts-per-token value".to_string())?;
    let group_size = cli
        .group_size_override
        .or_else(|| conf.get("CFG_EXPERT_GROUP_SIZE").and_then(|v| v.parse().ok()))
        .ok_or_else(|| {
            "expert group size is not explicit in this config; pass --group-size with the actual runtime value rather than assuming a model-specific default".to_string()
        })?;
    if hidden_size == 0
        || intermediate_size == 0
        || group_size == 0
        || hidden_size % 64 != 0
        || intermediate_size % 64 != 0
        || hidden_size % group_size != 0
        || intermediate_size % group_size != 0
        || group_size >= hidden_size
        || group_size >= intermediate_size
    {
        return Err(format!(
            "unsupported CPU-tail geometry hidden={hidden_size} intermediate={intermediate_size} group={group_size}"
        ));
    }
    let w13_packed_words = (hidden_size / 8)
        .checked_mul(2 * intermediate_size)
        .ok_or_else(|| "w13 packed length overflow".to_string())?;
    let w13_scale_values = (hidden_size / group_size)
        .checked_mul(2 * intermediate_size)
        .ok_or_else(|| "w13 scale length overflow".to_string())?;
    let w2_packed_words = (intermediate_size / 8)
        .checked_mul(hidden_size)
        .ok_or_else(|| "w2 packed length overflow".to_string())?;
    let w2_scale_values = (intermediate_size / group_size)
        .checked_mul(hidden_size)
        .ok_or_else(|| "w2 scale length overflow".to_string())?;
    let expert_bytes = w13_packed_words
        .checked_mul(4)
        .and_then(|v| v.checked_add(w13_scale_values * 2))
        .and_then(|v| v.checked_add(w2_packed_words * 4))
        .and_then(|v| v.checked_add(w2_scale_values * 2))
        .ok_or_else(|| "expert byte length overflow".to_string())?;
    Ok(ExpertGeometry {
        config_path: cli.config_path.display().to_string(),
        model_config_path: model_config_path.display().to_string(),
        model_name: Path::new(model_path)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or(model_path)
            .to_string(),
        hidden_size,
        intermediate_size,
        group_size,
        experts_per_token,
        expert_bytes,
        w13_packed_words,
        w13_scale_values,
        w2_packed_words,
        w2_scale_values,
    })
}

fn parse_cpu_list(value: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in value.trim().split(',') {
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

#[cfg(target_os = "linux")]
fn process_allowed_cpus() -> Vec<usize> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let rc =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if rc != 0 {
        return Vec::new();
    }
    (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn process_allowed_cpus() -> Vec<usize> {
    Vec::new()
}

fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn parse_size_bytes(value: &str) -> Option<usize> {
    let value = value.trim();
    let split = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    let number = value[..split].parse::<usize>().ok()?;
    match value[split..].trim().to_ascii_uppercase().as_str() {
        "K" | "KB" => number.checked_mul(1024),
        "M" | "MB" => number.checked_mul(1024 * 1024),
        "G" | "GB" => number.checked_mul(1024 * 1024 * 1024),
        "" => Some(number),
        _ => None,
    }
}

fn detect_topology() -> CpuTopology {
    let mut notes = Vec::new();
    let mut allowed = process_allowed_cpus();
    if allowed.is_empty() {
        allowed = read_trimmed("/sys/devices/system/cpu/online")
            .map(|v| parse_cpu_list(&v))
            .unwrap_or_default();
    }
    if allowed.is_empty() {
        let count = thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        allowed = (0..count).collect();
        notes.push("sysfs/affinity unavailable; using available_parallelism fallback".to_string());
    }
    let allowed_set: BTreeSet<usize> = allowed.iter().copied().collect();
    let mut core_first = BTreeMap::<(String, String), usize>::new();
    for &cpu in &allowed {
        let package = read_trimmed(format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
        ))
        .unwrap_or_else(|| "0".to_string());
        let core = read_trimmed(format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id"))
            .unwrap_or_else(|| cpu.to_string());
        core_first
            .entry((package, core))
            .and_modify(|existing| *existing = (*existing).min(cpu))
            .or_insert(cpu);
    }
    let mut physical_cpus: Vec<usize> = core_first.into_values().collect();
    physical_cpus.sort_unstable();

    let mut domains = BTreeMap::<String, CacheDomain>::new();
    for &cpu in &physical_cpus {
        let cache_root = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/cache"));
        let mut found = false;
        if let Ok(entries) = fs::read_dir(cache_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if read_trimmed(path.join("level")).as_deref() != Some("3") {
                    continue;
                }
                let id = read_trimmed(path.join("id"))
                    .or_else(|| read_trimmed(path.join("shared_cpu_list")))
                    .unwrap_or_else(|| "fallback".to_string());
                let size_bytes =
                    read_trimmed(path.join("size")).and_then(|value| parse_size_bytes(&value));
                let domain = domains.entry(id.clone()).or_insert(CacheDomain {
                    id,
                    physical_cpus: Vec::new(),
                    size_bytes,
                });
                let shared = read_trimmed(path.join("shared_cpu_list"))
                    .map(|value| parse_cpu_list(&value))
                    .unwrap_or_else(|| vec![cpu]);
                for shared_cpu in shared {
                    if allowed_set.contains(&shared_cpu)
                        && physical_cpus.binary_search(&shared_cpu).is_ok()
                    {
                        domain.physical_cpus.push(shared_cpu);
                    }
                }
                found = true;
                break;
            }
        }
        if !found {
            domains
                .entry("fallback".to_string())
                .or_insert(CacheDomain {
                    id: "fallback".to_string(),
                    physical_cpus: Vec::new(),
                    size_bytes: None,
                })
                .physical_cpus
                .push(cpu);
        }
    }
    let mut cache_domains: Vec<CacheDomain> = domains.into_values().collect();
    for domain in &mut cache_domains {
        domain.physical_cpus.sort_unstable();
        domain.physical_cpus.dedup();
    }
    cache_domains.sort_by(|a, b| a.id.cmp(&b.id));
    if cache_domains.is_empty() {
        cache_domains.push(CacheDomain {
            id: "fallback".to_string(),
            physical_cpus: physical_cpus.clone(),
            size_bytes: None,
        });
    }
    let affinity_supported = cfg!(target_os = "linux") && !process_allowed_cpus().is_empty();
    CpuTopology {
        allowed_logical_cpus: allowed,
        physical_cpus,
        cache_domains,
        affinity_supported,
        notes,
    }
}

fn placement_order(topology: &CpuTopology, policy: PlacementPolicy) -> Vec<usize> {
    let mut result = Vec::new();
    match policy {
        PlacementPolicy::Compact => {
            for domain in &topology.cache_domains {
                result.extend(domain.physical_cpus.iter().copied());
            }
        }
        PlacementPolicy::Spread => {
            let max_len = topology
                .cache_domains
                .iter()
                .map(|domain| domain.physical_cpus.len())
                .max()
                .unwrap_or(0);
            for index in 0..max_len {
                for domain in &topology.cache_domains {
                    if let Some(cpu) = domain.physical_cpus.get(index) {
                        result.push(*cpu);
                    }
                }
            }
        }
    }
    result.dedup();
    if result.is_empty() {
        topology.physical_cpus.clone()
    } else {
        result
    }
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) -> bool {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) -> bool {
    false
}

fn build_pool(cpu_ids: &[usize], affinity_supported: bool) -> Result<ThreadPool, String> {
    let ids = cpu_ids.to_vec();
    rayon::ThreadPoolBuilder::new()
        .num_threads(ids.len())
        .start_handler(move |index| {
            if affinity_supported {
                let _ = pin_current_thread(ids[index % ids.len()]);
            }
        })
        .build()
        .map_err(|e| format!("build rayon pool: {e}"))
}

fn thread_ladder(physical_count: usize, smoke: bool) -> Vec<usize> {
    if smoke {
        return vec![physical_count.max(1).min(2)];
    }
    let physical_count = physical_count.max(1);
    let mut values = Vec::new();
    let mut current = 1usize;
    while current < physical_count {
        values.push(current);
        match current.checked_mul(2) {
            Some(next) => current = next,
            None => break,
        }
    }
    values.push(physical_count);
    values.sort_unstable();
    values.dedup();
    values
}

#[derive(Debug, Clone)]
struct TwoTeamCandidate {
    placement: PlacementPolicy,
    cpu_ids: [Vec<usize>; 2],
}

fn two_team_candidates(topology: &CpuTopology, smoke: bool) -> Vec<TwoTeamCandidate> {
    let ladder = thread_ladder(topology.physical_cpus.len(), smoke);
    let policies: &[PlacementPolicy] = if smoke {
        &[PlacementPolicy::Spread]
    } else {
        &[PlacementPolicy::Compact, PlacementPolicy::Spread]
    };
    let mut candidates = Vec::new();
    for &placement in policies {
        let order = placement_order(topology, placement);
        for &worker_0_threads in &ladder {
            for &worker_1_threads in &ladder {
                // Worker 0 owns the deepest claim and should not receive fewer
                // resources than worker 1. Both teams must fit in disjoint
                // runtime-discovered physical CPU sets.
                if worker_1_threads > worker_0_threads
                    || worker_0_threads + worker_1_threads > order.len()
                {
                    continue;
                }
                candidates.push(TwoTeamCandidate {
                    placement,
                    cpu_ids: [
                        order[..worker_0_threads].to_vec(),
                        order[worker_0_threads..worker_0_threads + worker_1_threads].to_vec(),
                    ],
                });
            }
        }
    }
    candidates
}

fn persistent_single_candidates(
    candidates: &[TwoTeamCandidate],
) -> Vec<(PlacementPolicy, Vec<usize>)> {
    let mut result = Vec::new();
    for candidate in candidates {
        if !result.iter().any(|(placement, cpu_ids)| {
            *placement == candidate.placement && *cpu_ids == candidate.cpu_ids[0]
        }) {
            result.push((candidate.placement, candidate.cpu_ids[0].clone()));
        }
    }
    result
}

fn parse_summary_fields(line: &str) -> BTreeMap<&str, &str> {
    line.split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect()
}

fn load_histogram(paths: &[PathBuf]) -> Result<QueueHistogram, String> {
    let mut generated_tokens = 0u64;
    let mut counts = [0u64; 9];
    let mut sources = Vec::new();
    for path in paths {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read histogram {}: {e}", path.display()))?;
        let mut best: Option<(u64, [u64; 9])> = None;
        for line in text
            .lines()
            .filter(|line| line.contains("CPU TAIL SUMMARY"))
        {
            let fields = parse_summary_fields(line);
            let Some(generated) = fields.get("generated").and_then(|v| v.parse::<u64>().ok())
            else {
                continue;
            };
            let Some(depths) = fields.get("attempts_by_depth_2_to_9plus") else {
                continue;
            };
            let parsed: Vec<u64> = depths
                .split(',')
                .filter_map(|v| v.parse::<u64>().ok())
                .collect();
            if parsed.len() != 8 {
                continue;
            }
            let mut row = [0u64; 9];
            row[0] = fields
                .get("short_queue_skips")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            row[1..].copy_from_slice(&parsed);
            if best
                .as_ref()
                .map(|(tokens, _)| generated >= *tokens)
                .unwrap_or(true)
            {
                best = Some((generated, row));
            }
        }
        let (generated, row) = best.ok_or_else(|| {
            format!(
                "{} contains no parseable CPU TAIL SUMMARY queue histogram",
                path.display()
            )
        })?;
        generated_tokens = generated_tokens.saturating_add(generated);
        for (total, value) in counts.iter_mut().zip(row) {
            *total = total.saturating_add(value);
        }
        sources.push(path.display().to_string());
    }
    if generated_tokens == 0 {
        return Err("queue histogram has zero generated tokens".to_string());
    }
    let mut depth_counts = BTreeMap::new();
    for (index, count) in counts.iter().enumerate() {
        let key = if index == 8 {
            "9+".to_string()
        } else {
            (index + 1).to_string()
        };
        depth_counts.insert(key, *count);
    }
    let queues: u64 = counts.iter().sum();
    let cold_experts: u64 = counts
        .iter()
        .enumerate()
        .map(|(index, count)| (index as u64 + 1) * count)
        .sum();
    Ok(QueueHistogram {
        sources,
        generated_tokens,
        depth_counts,
        queues_per_token: queues as f64 / generated_tokens as f64,
        cold_experts_per_token: cold_experts as f64 / generated_tokens as f64,
    })
}

fn synthetic_expert(geometry: &ExpertGeometry, seed: u64) -> SyntheticExpert {
    fn words(len: usize, seed: u64) -> Vec<u32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u32
            })
            .collect()
    }
    let scale = bf16::from_f32(1.0 / i16::MAX as f32).to_bits();
    SyntheticExpert {
        w13_packed: words(geometry.w13_packed_words, seed),
        w13_scales: vec![scale; geometry.w13_scale_values],
        w2_packed: words(geometry.w2_packed_words, seed ^ 0x9e3779b97f4a7c15),
        w2_scales: vec![scale; geometry.w2_scale_values],
    }
}

fn latency_summary(samples_ms: &[f64], expert_bytes: usize) -> LatencySummary {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean_ms = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let percentile = |fraction: f64| {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index]
    };
    LatencySummary {
        samples: sorted.len(),
        mean_ms,
        min_ms: sorted[0],
        p50_ms: percentile(0.50),
        p90_ms: percentile(0.90),
        p95_ms: percentile(0.95),
        max_ms: *sorted.last().unwrap(),
        experts_per_second: 1000.0 / mean_ms,
        effective_weight_gib_s: expert_bytes as f64
            / (mean_ms / 1000.0)
            / (1024.0 * 1024.0 * 1024.0),
    }
}

fn run_cpu_iterations(
    pool: &ThreadPool,
    format: WeightFormat,
    geometry: &ExpertGeometry,
    replicas: &[SyntheticExpert],
    activation: &[u16],
    seconds: f64,
) -> Result<(LatencySummary, Vec<f64>), String> {
    let cancel = AtomicU64::new(0);
    let mut scratch = ExpertScratch::new(
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.group_size,
    );
    let mut output = vec![0u16; geometry.hidden_size];
    let run_one =
        |expert: &SyntheticExpert, scratch: &mut ExpertScratch, output: &mut [u16]| -> bool {
            match format {
                WeightFormat::Transposed => expert_forward_transposed_int4_cpu_tail(
                    &expert.w13_packed,
                    &expert.w13_scales,
                    &expert.w2_packed,
                    &expert.w2_scales,
                    activation,
                    output,
                    geometry.hidden_size,
                    geometry.intermediate_size,
                    geometry.group_size,
                    0.0,
                    1.0,
                    scratch,
                    &cancel,
                    1,
                ),
                WeightFormat::MarlinDirect => expert_forward_marlin_int4_cpu_tail(
                    &expert.w13_packed,
                    &expert.w13_scales,
                    &expert.w2_packed,
                    &expert.w2_scales,
                    activation,
                    output,
                    geometry.hidden_size,
                    geometry.intermediate_size,
                    geometry.group_size,
                    0.0,
                    1.0,
                    scratch,
                    &cancel,
                    1,
                ),
            }
        };
    let pilot_started = Instant::now();
    let pilot_ok = pool.install(|| run_one(&replicas[0], &mut scratch, &mut output));
    if !pilot_ok {
        return Err("CPU calibration pilot was unexpectedly cancelled".to_string());
    }
    let pilot_s = pilot_started.elapsed().as_secs_f64().max(f64::EPSILON);
    let iterations = (seconds / pilot_s).ceil().max(1.0) as usize;
    let mut samples = Vec::with_capacity(iterations);
    pool.install(|| {
        for index in 0..iterations {
            let started = Instant::now();
            let ok = run_one(&replicas[index % replicas.len()], &mut scratch, &mut output);
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            if !ok {
                break;
            }
        }
    });
    if samples.len() != iterations {
        return Err(format!(
            "CPU calibration stopped after {}/{} iterations",
            samples.len(),
            iterations
        ));
    }
    Ok((latency_summary(&samples, geometry.expert_bytes), samples))
}

fn run_dma_for(resources: Arc<DmaResources>, seconds: f64) -> Result<f64, String> {
    thread::Builder::new()
        .name("krasis-calibrate-dma".to_string())
        .spawn(move || -> Result<f64, String> {
            resources
                .device
                .bind_to_thread()
                .map_err(|e| format!("bind CUDA context for DMA: {e}"))?;
            resources.copy_once()?;
            let started = Instant::now();
            let mut bytes = 0u64;
            while started.elapsed().as_secs_f64() < seconds {
                resources.copy_once()?;
                bytes = bytes.saturating_add(resources.host.bytes as u64);
            }
            let elapsed = started.elapsed().as_secs_f64();
            Ok(bytes as f64 / elapsed / (1024.0 * 1024.0 * 1024.0))
        })
        .map_err(|e| format!("spawn baseline DMA worker: {e}"))?
        .join()
        .map_err(|_| "baseline DMA worker panicked".to_string())?
}

fn run_cpu_with_dma(
    resources: Arc<DmaResources>,
    pool: &ThreadPool,
    format: WeightFormat,
    geometry: &ExpertGeometry,
    replicas: &[SyntheticExpert],
    activation: &[u16],
    seconds: f64,
) -> Result<(LatencySummary, Vec<f64>, f64), String> {
    let stop = Arc::new(AtomicBool::new(false));
    let bytes = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(2));
    let worker_resources = Arc::clone(&resources);
    let worker_stop = Arc::clone(&stop);
    let worker_bytes = Arc::clone(&bytes);
    let worker_barrier = Arc::clone(&barrier);
    let worker = thread::Builder::new()
        .name("krasis-calibrate-dma".to_string())
        .spawn(move || -> Result<(f64, f64), String> {
            worker_resources
                .device
                .bind_to_thread()
                .map_err(|e| format!("bind CUDA context in DMA worker: {e}"))?;
            worker_resources.copy_once()?;
            worker_barrier.wait();
            let started = Instant::now();
            while !worker_stop.load(Ordering::Acquire) {
                worker_resources.copy_once()?;
                worker_bytes.fetch_add(worker_resources.host.bytes as u64, Ordering::Relaxed);
            }
            Ok((
                started.elapsed().as_secs_f64(),
                worker_bytes.load(Ordering::Relaxed) as f64,
            ))
        })
        .map_err(|e| format!("spawn DMA worker: {e}"))?;
    barrier.wait();
    let cpu_result = run_cpu_iterations(pool, format, geometry, replicas, activation, seconds);
    stop.store(true, Ordering::Release);
    let (dma_elapsed, dma_bytes) = worker
        .join()
        .map_err(|_| "DMA worker panicked".to_string())??;
    let (latency, samples) = cpu_result?;
    let dma_gib_s = dma_bytes / dma_elapsed / (1024.0 * 1024.0 * 1024.0);
    Ok((latency, samples, dma_gib_s))
}

fn run_persistent_single_with_dma(
    resources: Arc<DmaResources>,
    geometry: &ExpertGeometry,
    replicas: &[SyntheticExpert],
    activation: &[u16],
    cpu_ids: &[usize],
    seconds: f64,
    layer_gap_us: u64,
) -> Result<(LatencySummary, Vec<f64>, f64), String> {
    if replicas.is_empty() {
        return Err("persistent single-team calibration has no synthetic experts".to_string());
    }
    let team = PersistentTransposedTeam::new(cpu_ids)?;
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(2));
    let dma_resources = Arc::clone(&resources);
    let dma_stop = Arc::clone(&stop);
    let dma_barrier = Arc::clone(&barrier);
    let dma_worker = thread::Builder::new()
        .name("krasis-calibrate-persistent-single-dma".to_string())
        .spawn(move || -> Result<(f64, f64), String> {
            dma_resources
                .device
                .bind_to_thread()
                .map_err(|e| format!("bind CUDA context in persistent single DMA worker: {e}"))?;
            dma_resources.copy_once()?;
            dma_barrier.wait();
            let started = Instant::now();
            let mut bytes = 0u64;
            while !dma_stop.load(Ordering::Acquire) {
                dma_resources.copy_once()?;
                bytes = bytes.saturating_add(dma_resources.host.bytes as u64);
            }
            Ok((started.elapsed().as_secs_f64(), bytes as f64))
        })
        .map_err(|e| format!("spawn persistent single DMA worker: {e}"))?;

    let cancel = AtomicU64::new(0);
    let mut scratch = ExpertScratch::new(
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.group_size,
    );
    let mut output = vec![0u16; geometry.hidden_size];
    let mut samples = Vec::new();
    barrier.wait();
    let started = Instant::now();
    let mut iteration = 0usize;
    while started.elapsed().as_secs_f64() < seconds {
        if iteration != 0 && layer_gap_us != 0 {
            thread::sleep(Duration::from_micros(layer_gap_us));
        }
        let expert = &replicas[iteration % replicas.len()];
        let expert_started = Instant::now();
        let outcome = expert_forward_transposed_persistent(
            &team,
            &expert.w13_packed,
            &expert.w13_scales,
            &expert.w2_packed,
            &expert.w2_scales,
            activation,
            &mut output,
            geometry.hidden_size,
            geometry.intermediate_size,
            geometry.group_size,
            0.0,
            1.0,
            &mut scratch,
            &cancel,
            1,
        );
        samples.push(expert_started.elapsed().as_secs_f64() * 1000.0);
        let completed = match outcome {
            Ok(completed) => completed,
            Err(error) => {
                stop.store(true, Ordering::Release);
                let _ = dma_worker.join();
                return Err(error);
            }
        };
        if !completed {
            stop.store(true, Ordering::Release);
            let _ = dma_worker.join();
            return Err(
                "persistent single-team calibration was unexpectedly cancelled".to_string(),
            );
        }
        iteration += 1;
    }
    stop.store(true, Ordering::Release);
    let (dma_elapsed, dma_bytes) = dma_worker
        .join()
        .map_err(|_| "persistent single-team DMA worker panicked".to_string())??;
    if samples.is_empty() {
        return Err("persistent single-team calibration produced no samples".to_string());
    }
    let latency = latency_summary(&samples, geometry.expert_bytes);
    let dma_gib_s = dma_bytes / dma_elapsed / (1024.0 * 1024.0 * 1024.0);
    Ok((latency, samples, dma_gib_s))
}

fn run_two_persistent_with_dma(
    resources: Arc<DmaResources>,
    geometry: &ExpertGeometry,
    replicas: &[SyntheticExpert],
    activation: &[u16],
    cpu_ids: &[Vec<usize>; 2],
    seconds: f64,
    layer_gap_us: u64,
) -> Result<([LatencySummary; 2], [Vec<f64>; 2], f64), String> {
    if replicas.len() < 2 {
        return Err(
            "two-team calibration requires at least two distinct synthetic experts".to_string(),
        );
    }
    let replicas_per_team = replicas.len() / 2;
    if replicas_per_team == 0 {
        return Err("two-team calibration resolved zero replicas per team".to_string());
    }

    let teams = [
        PersistentTransposedTeam::new(&cpu_ids[0])?,
        PersistentTransposedTeam::new(&cpu_ids[1])?,
    ];
    let stop = AtomicBool::new(false);
    let ready = Barrier::new(3);
    let cycle_start = Barrier::new(2);
    let cycle_done = Barrier::new(2);
    let cycle_decision = Barrier::new(2);

    let (sample_sets, dma_elapsed, dma_bytes) = thread::scope(
        |scope| -> Result<([Vec<f64>; 2], f64, f64), String> {
            let dma_resources = &resources;
            let dma_ready = &ready;
            let dma_stop = &stop;
            let dma_handle = scope.spawn(move || -> Result<(f64, f64), String> {
                dma_resources
                    .device
                    .bind_to_thread()
                    .map_err(|e| format!("bind CUDA context in two-team DMA worker: {e}"))?;
                dma_resources.copy_once()?;
                dma_ready.wait();
                let started = Instant::now();
                let mut bytes = 0u64;
                while !dma_stop.load(Ordering::Acquire) {
                    dma_resources.copy_once()?;
                    bytes = bytes.saturating_add(dma_resources.host.bytes as u64);
                }
                Ok((started.elapsed().as_secs_f64(), bytes as f64))
            });

            let mut team_handles = Vec::with_capacity(2);
            for team_index in 0..2 {
                let team = &teams[team_index];
                let worker_ready = &ready;
                let worker_stop = &stop;
                let worker_cycle_start = &cycle_start;
                let worker_cycle_done = &cycle_done;
                let worker_cycle_decision = &cycle_decision;
                team_handles.push(scope.spawn(move || -> Result<Vec<f64>, String> {
                    let cancel = AtomicU64::new(0);
                    let mut scratch = ExpertScratch::new(
                        geometry.hidden_size,
                        geometry.intermediate_size,
                        geometry.group_size,
                    );
                    let mut output = vec![0u16; geometry.hidden_size];
                    let mut samples = Vec::new();
                    let mut iteration = 0usize;
                    let mut failure = None;
                    worker_ready.wait();
                    let measurement_started = Instant::now();
                    loop {
                        if team_index == 0 && iteration != 0 && layer_gap_us != 0 {
                            thread::sleep(Duration::from_micros(layer_gap_us));
                        }
                        worker_cycle_start.wait();
                        if worker_stop.load(Ordering::Acquire) {
                            break;
                        }
                        let replica_index =
                            team_index * replicas_per_team + iteration % replicas_per_team;
                        let expert = &replicas[replica_index];
                        let started = Instant::now();
                        let outcome = expert_forward_transposed_persistent(
                            team,
                            &expert.w13_packed,
                            &expert.w13_scales,
                            &expert.w2_packed,
                            &expert.w2_scales,
                            activation,
                            &mut output,
                            geometry.hidden_size,
                            geometry.intermediate_size,
                            geometry.group_size,
                            0.0,
                            1.0,
                            &mut scratch,
                            &cancel,
                            1,
                        );
                        samples.push(started.elapsed().as_secs_f64() * 1000.0);
                        match outcome {
                            Ok(true) => {}
                            Ok(false) => {
                                failure = Some(format!(
                                    "two-team calibration worker {team_index} was unexpectedly cancelled"
                                ));
                                worker_stop.store(true, Ordering::Release);
                            }
                            Err(error) => {
                                failure = Some(format!(
                                    "two-team calibration worker {team_index}: {error}"
                                ));
                                worker_stop.store(true, Ordering::Release);
                            }
                        }
                        worker_cycle_done.wait();
                        if team_index == 0 && measurement_started.elapsed().as_secs_f64() >= seconds
                        {
                            worker_stop.store(true, Ordering::Release);
                        }
                        worker_cycle_decision.wait();
                        iteration += 1;
                    }
                    if let Some(error) = failure {
                        return Err(error);
                    }
                    if samples.is_empty() {
                        return Err(format!(
                            "two-team calibration worker {team_index} produced no samples"
                        ));
                    }
                    Ok(samples)
                }));
            }

            let worker_0 = team_handles
                .remove(0)
                .join()
                .map_err(|_| "two-team calibration worker 0 panicked".to_string())??;
            let worker_1 = team_handles
                .remove(0)
                .join()
                .map_err(|_| "two-team calibration worker 1 panicked".to_string())??;
            stop.store(true, Ordering::Release);
            let (dma_elapsed, dma_bytes) = dma_handle
                .join()
                .map_err(|_| "two-team DMA worker panicked".to_string())??;
            Ok(([worker_0, worker_1], dma_elapsed, dma_bytes))
        },
    )?;

    let latencies = [
        latency_summary(&sample_sets[0], geometry.expert_bytes),
        latency_summary(&sample_sets[1], geometry.expert_bytes),
    ];
    let dma_gib_s = dma_bytes / dma_elapsed / (1024.0 * 1024.0 * 1024.0);
    Ok((latencies, sample_sets, dma_gib_s))
}

fn prediction(
    histogram: &QueueHistogram,
    samples_ms: &[f64],
    handoff_margin_ms: f64,
    dma_gib_s: f64,
    dma_baseline_gib_s: f64,
    expert_bytes: usize,
    baseline_ms_per_token: Option<f64>,
) -> Prediction {
    let copy_ms = expert_bytes as f64 / (dma_gib_s * 1024.0 * 1024.0 * 1024.0) * 1000.0;
    let baseline_copy_ms =
        expert_bytes as f64 / (dma_baseline_gib_s * 1024.0 * 1024.0 * 1024.0) * 1000.0;
    let mut probabilities = BTreeMap::new();
    let mut weighted_wins = 0.0;
    let mut total_queues = 0u64;
    for (depth_label, count) in &histogram.depth_counts {
        let depth = depth_label
            .trim_end_matches('+')
            .parse::<usize>()
            .unwrap_or(1);
        let deadline_ms = depth.saturating_sub(1) as f64 * copy_ms;
        let wins = samples_ms
            .iter()
            .filter(|sample| **sample + handoff_margin_ms <= deadline_ms)
            .count();
        let probability = wins as f64 / samples_ms.len() as f64;
        probabilities.insert(depth_label.clone(), probability);
        weighted_wins += probability * *count as f64;
        total_queues = total_queues.saturating_add(*count);
    }
    let absorbed_experts_per_token = weighted_wins / histogram.generated_tokens as f64;
    let predicted_win_rate = if total_queues == 0 {
        0.0
    } else {
        weighted_wins / total_queues as f64
    };
    let nominal_saved_ms = absorbed_experts_per_token * copy_ms;
    let dma_degradation_ms =
        histogram.cold_experts_per_token * (copy_ms - baseline_copy_ms).max(0.0);
    let net_saved_ms = nominal_saved_ms - dma_degradation_ms;
    let predicted_baseline_tok_s = baseline_ms_per_token.map(|ms| 1000.0 / ms);
    let predicted_tok_s =
        baseline_ms_per_token.map(|ms| 1000.0 / (ms - net_saved_ms).max(f64::EPSILON));
    Prediction {
        handoff_margin_ms,
        copy_ms_per_expert: copy_ms,
        win_probability_by_depth: probabilities,
        absorbed_experts_per_token,
        predicted_win_rate,
        nominal_saved_ms_per_token: nominal_saved_ms,
        dma_degradation_ms_per_token: dma_degradation_ms,
        net_saved_ms_per_token: net_saved_ms,
        predicted_baseline_tok_s,
        predicted_tok_s,
    }
}

fn two_team_prediction(
    histogram: &QueueHistogram,
    samples_ms: &[Vec<f64>; 2],
    dma_gib_s: f64,
    dma_baseline_gib_s: f64,
    expert_bytes: usize,
    baseline_ms_per_token: Option<f64>,
) -> TwoTeamPrediction {
    let copy_ms = expert_bytes as f64 / (dma_gib_s * 1024.0 * 1024.0 * 1024.0) * 1000.0;
    let baseline_copy_ms =
        expert_bytes as f64 / (dma_baseline_gib_s * 1024.0 * 1024.0 * 1024.0) * 1000.0;
    let mut worker_probabilities = [BTreeMap::new(), BTreeMap::new()];
    let mut weighted_wins = [0.0f64; 2];

    for (depth_label, count) in &histogram.depth_counts {
        let depth = depth_label
            .trim_end_matches('+')
            .parse::<usize>()
            .unwrap_or(1);
        let deadlines = [
            depth.saturating_sub(1) as f64 * copy_ms,
            depth.saturating_sub(2) as f64 * copy_ms,
        ];
        for worker in 0..2 {
            let eligible = if worker == 0 { depth >= 2 } else { depth >= 3 };
            let probability = if eligible {
                samples_ms[worker]
                    .iter()
                    .filter(|sample| **sample <= deadlines[worker])
                    .count() as f64
                    / samples_ms[worker].len() as f64
            } else {
                0.0
            };
            worker_probabilities[worker].insert(depth_label.clone(), probability);
            weighted_wins[worker] += probability * *count as f64;
        }
    }

    let absorbed = [
        weighted_wins[0] / histogram.generated_tokens as f64,
        weighted_wins[1] / histogram.generated_tokens as f64,
    ];
    let total_absorbed = absorbed[0] + absorbed[1];
    let nominal_saved_ms = total_absorbed * copy_ms;
    let dma_degradation_ms =
        histogram.cold_experts_per_token * (copy_ms - baseline_copy_ms).max(0.0);
    let net_saved_ms = nominal_saved_ms - dma_degradation_ms;
    TwoTeamPrediction {
        copy_ms_per_expert: copy_ms,
        worker_0_win_probability_by_depth: worker_probabilities[0].clone(),
        worker_1_win_probability_by_depth: worker_probabilities[1].clone(),
        worker_0_absorbed_experts_per_token: absorbed[0],
        worker_1_absorbed_experts_per_token: absorbed[1],
        total_absorbed_experts_per_token: total_absorbed,
        nominal_saved_ms_per_token: nominal_saved_ms,
        dma_degradation_ms_per_token: dma_degradation_ms,
        net_saved_ms_per_token: net_saved_ms,
        predicted_baseline_tok_s: baseline_ms_per_token.map(|ms| 1000.0 / ms),
        predicted_tok_s: baseline_ms_per_token
            .map(|ms| 1000.0 / (ms - net_saved_ms).max(f64::EPSILON)),
    }
}

fn cuda_uuid_string(uuid: &cuda_sys::CUuuid) -> String {
    let bytes: Vec<u8> = uuid.bytes.iter().map(|value| *value as u8).collect();
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

fn resolve_gpu(cli: &Cli) -> Result<(usize, String), String> {
    let count = CudaDevice::count().map_err(|e| format!("CUDA device count: {e}"))? as usize;
    let wanted = cli
        .gpu_uuid
        .as_ref()
        .map(|value| value.to_ascii_lowercase());
    for ordinal in 0..count {
        if let Some(requested) = cli.gpu_ordinal {
            if ordinal != requested {
                continue;
            }
        }
        let mut uuid = cuda_sys::CUuuid::default();
        check_cuda(
            unsafe { cuda_sys::lib().cuDeviceGetUuid_v2(&mut uuid, ordinal as i32) },
            format!("cuDeviceGetUuid_v2({ordinal})"),
        )?;
        let uuid_string = cuda_uuid_string(&uuid);
        if wanted
            .as_ref()
            .map(|value| value == &uuid_string.to_ascii_lowercase())
            .unwrap_or(true)
        {
            return Ok((ordinal, uuid_string));
        }
    }
    Err(format!(
        "requested GPU not found (ordinal={:?}, uuid={:?})",
        cli.gpu_ordinal, cli.gpu_uuid
    ))
}

fn format_measurement(
    phase: &str,
    format: WeightFormat,
    threads: usize,
    placement: PlacementPolicy,
    cpu_ids: Vec<usize>,
    latency: LatencySummary,
    dma_gib_s: Option<f64>,
    dma_baseline_gib_s: f64,
    prediction: Option<Prediction>,
) -> SweepMeasurement {
    SweepMeasurement {
        phase: phase.to_string(),
        format,
        threads,
        placement,
        cpu_ids,
        latency,
        dma_gib_s,
        dma_degradation_pct: dma_gib_s
            .map(|value| (dma_baseline_gib_s - value) / dma_baseline_gib_s * 100.0),
        prediction,
    }
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return Err("CPU-tail calibration requires AVX2 and FMA".to_string());
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    return Err("CPU-tail calibration currently requires x86 AVX2 and FMA".to_string());

    let cli = parse_cli(args)?;
    let overall_started = Instant::now();
    let geometry = checked_expert_geometry(&cli)?;
    let histogram = load_histogram(&cli.histogram_logs)?;
    let topology = detect_topology();
    if topology.physical_cpus.is_empty() {
        return Err("no physical CPUs available to calibration".to_string());
    }
    let full_thread_counts = thread_ladder(topology.physical_cpus.len(), false);
    let thread_counts = thread_ladder(topology.physical_cpus.len(), cli.smoke);
    let two_team_candidates = if cli.two_team {
        let candidates = two_team_candidates(&topology, cli.smoke);
        if candidates.is_empty() {
            return Err(
                "runtime topology exposes no valid non-overlapping two-team CPU split".to_string(),
            );
        }
        candidates
    } else {
        Vec::new()
    };
    let persistent_single_candidates = persistent_single_candidates(&two_team_candidates);
    let policies: Vec<PlacementPolicy> = if cli.smoke {
        vec![PlacementPolicy::Spread]
    } else {
        vec![PlacementPolicy::Compact, PlacementPolicy::Spread]
    };
    let dma_baseline_window_count =
        full_thread_counts.len() * [PlacementPolicy::Compact, PlacementPolicy::Spread].len();
    let configurations = thread_counts.len() * policies.len();
    let measured_phases = 2 * configurations
        + 4
        + dma_baseline_window_count
        + persistent_single_candidates.len()
        + two_team_candidates.len();
    let phase_seconds = cli.budget_seconds / measured_phases as f64;
    if !phase_seconds.is_finite() || phase_seconds <= 0.0 {
        return Err("measurement phase budget resolved to zero".to_string());
    }

    let llc_bytes: usize = topology
        .cache_domains
        .iter()
        .filter_map(|domain| domain.size_bytes)
        .sum();
    let replicas_per_team = if llc_bytes == 0 {
        1
    } else {
        llc_bytes / geometry.expert_bytes + 1
    };
    let replica_count = replicas_per_team
        .checked_mul(if cli.two_team { 2 } else { 1 })
        .ok_or_else(|| "synthetic replica-count overflow".to_string())?;
    let working_set_bytes = replica_count
        .checked_mul(geometry.expert_bytes)
        .ok_or_else(|| "synthetic working-set byte overflow".to_string())?;
    eprintln!(
        "CPU TAIL CALIBRATION model={} hidden={} intermediate={} group={} expert_bytes={} replicas={} replicas_per_team={} working_set_mib={:.3} physical_cores={} cache_domains={} phase_seconds={:.3} persistent_single_candidates={} two_team_candidates={}",
        geometry.model_name,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.group_size,
        geometry.expert_bytes,
        replica_count,
        replicas_per_team,
        working_set_bytes as f64 / (1024.0 * 1024.0),
        topology.physical_cpus.len(),
        topology.cache_domains.len(),
        phase_seconds,
        persistent_single_candidates.len(),
        two_team_candidates.len(),
    );
    let replicas: Vec<SyntheticExpert> = (0..replica_count)
        .map(|index| synthetic_expert(&geometry, index as u64 + 1))
        .collect();
    let activation: Vec<u16> = (0..geometry.hidden_size)
        .map(|index| bf16::from_f32(((index % 257) as f32 / 257.0 - 0.5) * 0.125).to_bits())
        .collect();

    let (gpu_ordinal, gpu_uuid) = resolve_gpu(&cli)?;
    let device = CudaDevice::new(gpu_ordinal)
        .map_err(|e| format!("create CUDA device {gpu_ordinal}: {e}"))?;
    let gpu_name = device
        .name()
        .map_err(|e| format!("query CUDA device name: {e}"))?;
    let dma = Arc::new(DmaResources::new(
        Arc::clone(&device),
        geometry.expert_bytes,
    )?);
    let mut dma_baseline_windows_gib_s = Vec::with_capacity(dma_baseline_window_count);
    for window in 0..dma_baseline_window_count {
        let gib_s = run_dma_for(Arc::clone(&dma), phase_seconds)?;
        eprintln!(
            "CPU TAIL CALIBRATION DMA baseline_window={}/{} gib_s={gib_s:.6} \
             duration_s={phase_seconds:.3}",
            window + 1,
            dma_baseline_window_count,
        );
        dma_baseline_windows_gib_s.push(gib_s);
    }
    let dma_baseline_gib_s = dma_baseline_windows_gib_s
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| "DMA baseline produced no measurement windows".to_string())?;
    eprintln!(
        "CPU TAIL CALIBRATION DMA baseline_gib_s={dma_baseline_gib_s:.6} copy_ms_per_expert={:.6}",
        geometry.expert_bytes as f64 / (dma_baseline_gib_s * 1024.0 * 1024.0 * 1024.0) * 1000.0
    );

    let mut cpu_only = Vec::new();
    let mut concurrent = Vec::new();
    let mut concurrent_samples = Vec::<Vec<f64>>::new();
    let mut skipped_or_truncated = Vec::new();
    for &threads in &thread_counts {
        for &placement in &policies {
            let order = placement_order(&topology, placement);
            if order.len() < threads {
                skipped_or_truncated.push(format!(
                    "{} threads={} skipped: placement exposes only {} physical CPUs",
                    placement.label(),
                    threads,
                    order.len()
                ));
                continue;
            }
            let cpu_ids = order[..threads].to_vec();
            let pool = build_pool(&cpu_ids, topology.affinity_supported)?;
            let (isolated_latency, _) = run_cpu_iterations(
                &pool,
                WeightFormat::Transposed,
                &geometry,
                &replicas,
                &activation,
                phase_seconds,
            )?;
            eprintln!(
                "CPU TAIL CALIBRATION cpu_only threads={} placement={} mean_ms={:.6} p95_ms={:.6} gib_s={:.3}",
                threads,
                placement.label(),
                isolated_latency.mean_ms,
                isolated_latency.p95_ms,
                isolated_latency.effective_weight_gib_s,
            );
            cpu_only.push(format_measurement(
                "cpu_only",
                WeightFormat::Transposed,
                threads,
                placement,
                cpu_ids.clone(),
                isolated_latency,
                None,
                dma_baseline_gib_s,
                None,
            ));
            let (concurrent_latency, samples, dma_gib_s) = run_cpu_with_dma(
                Arc::clone(&dma),
                &pool,
                WeightFormat::Transposed,
                &geometry,
                &replicas,
                &activation,
                phase_seconds,
            )?;
            let raw_prediction = prediction(
                &histogram,
                &samples,
                0.0,
                dma_gib_s,
                dma_baseline_gib_s,
                geometry.expert_bytes,
                cli.baseline_ms_per_token,
            );
            eprintln!(
                "CPU TAIL CALIBRATION concurrent threads={} placement={} mean_ms={:.6} p95_ms={:.6} dma_gib_s={:.6} dma_degradation_pct={:.3} predicted_win_pct={:.3} net_saved_ms_per_token={:.6}",
                threads,
                placement.label(),
                concurrent_latency.mean_ms,
                concurrent_latency.p95_ms,
                dma_gib_s,
                (dma_baseline_gib_s - dma_gib_s) / dma_baseline_gib_s * 100.0,
                raw_prediction.predicted_win_rate * 100.0,
                raw_prediction.net_saved_ms_per_token,
            );
            concurrent.push(format_measurement(
                "concurrent",
                WeightFormat::Transposed,
                threads,
                placement,
                cpu_ids,
                concurrent_latency,
                Some(dma_gib_s),
                dma_baseline_gib_s,
                Some(raw_prediction),
            ));
            concurrent_samples.push(samples);
        }
    }
    if concurrent.is_empty() {
        return Err("all concurrent configurations were skipped".to_string());
    }
    let best_index = concurrent
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.prediction
                .as_ref()
                .unwrap()
                .net_saved_ms_per_token
                .total_cmp(&right.prediction.as_ref().unwrap().net_saved_ms_per_token)
        })
        .map(|(index, _)| index)
        .unwrap();
    let best = &concurrent[best_index];
    let best_samples = &concurrent_samples[best_index];
    let best_pool = build_pool(&best.cpu_ids, topology.affinity_supported)?;

    let mut format_comparison = Vec::new();
    let mut best_isolated_by_format = BTreeMap::new();
    let mut best_concurrent_by_format = BTreeMap::new();
    for format in [WeightFormat::Transposed, WeightFormat::MarlinDirect] {
        let (isolated_latency, _) = run_cpu_iterations(
            &best_pool,
            format,
            &geometry,
            &replicas,
            &activation,
            phase_seconds,
        )?;
        best_isolated_by_format.insert(format as u8, isolated_latency.mean_ms);
        format_comparison.push(format_measurement(
            "best_format_cpu_only",
            format,
            best.threads,
            best.placement,
            best.cpu_ids.clone(),
            isolated_latency,
            None,
            dma_baseline_gib_s,
            None,
        ));
        let (concurrent_latency, samples, dma_gib_s) = run_cpu_with_dma(
            Arc::clone(&dma),
            &best_pool,
            format,
            &geometry,
            &replicas,
            &activation,
            phase_seconds,
        )?;
        best_concurrent_by_format.insert(format as u8, concurrent_latency.mean_ms);
        let raw = prediction(
            &histogram,
            &samples,
            0.0,
            dma_gib_s,
            dma_baseline_gib_s,
            geometry.expert_bytes,
            cli.baseline_ms_per_token,
        );
        format_comparison.push(format_measurement(
            "best_format_concurrent",
            format,
            best.threads,
            best.placement,
            best.cpu_ids.clone(),
            concurrent_latency,
            Some(dma_gib_s),
            dma_baseline_gib_s,
            Some(raw),
        ));
    }

    let best_isolated_ms = cpu_only
        .iter()
        .find(|measurement| {
            measurement.threads == best.threads && measurement.placement == best.placement
        })
        .map(|measurement| measurement.latency.mean_ms)
        .unwrap_or_else(|| best_isolated_by_format[&(WeightFormat::Transposed as u8)]);
    let best_concurrent_ms = best.latency.mean_ms;
    let concurrency_explained_ms = (best_concurrent_ms - best_isolated_ms).max(0.0);
    let observed_gap_ms = cli
        .isolated_reference_ms
        .zip(cli.integrated_reference_ms)
        .map(|(isolated, integrated)| (integrated - isolated).max(0.0));
    let remaining_scheduler_handoff_ms = observed_gap_ms
        .map(|gap| (gap - concurrency_explained_ms).max(0.0))
        .unwrap_or(0.0);
    let gap_attribution = GapAttribution {
        isolated_reference_ms: cli.isolated_reference_ms,
        integrated_reference_ms: cli.integrated_reference_ms,
        observed_gap_ms,
        best_measured_isolated_ms: best_isolated_ms,
        best_measured_concurrent_ms: best_concurrent_ms,
        concurrency_explained_ms,
        concurrency_explained_pct_of_observed_gap: observed_gap_ms
            .filter(|gap| *gap > 0.0)
            .map(|gap| concurrency_explained_ms / gap * 100.0),
        remaining_scheduler_handoff_ms,
        remaining_pct_of_observed_gap: observed_gap_ms
            .filter(|gap| *gap > 0.0)
            .map(|gap| remaining_scheduler_handoff_ms / gap * 100.0),
    };
    let best_dma = best.dma_gib_s.unwrap();
    let raw_prediction = prediction(
        &histogram,
        best_samples,
        0.0,
        best_dma,
        dma_baseline_gib_s,
        geometry.expert_bytes,
        cli.baseline_ms_per_token,
    );
    let conservative_prediction = prediction(
        &histogram,
        best_samples,
        remaining_scheduler_handoff_ms,
        best_dma,
        dma_baseline_gib_s,
        geometry.expert_bytes,
        cli.baseline_ms_per_token,
    );
    let recommendation = if conservative_prediction.net_saved_ms_per_token > 0.0 {
        "enable_candidate"
    } else {
        "decline"
    };
    let optimizer = OptimizerDecision {
        recommendation: recommendation.to_string(),
        reason: if recommendation == "enable_candidate" {
            "Measured concurrent CPU completion still predicts positive net token-time saving after charging DMA degradation and the unexplained scheduler/handoff margin; one end-to-end validation run is justified.".to_string()
        } else {
            "Measured concurrent CPU completion does not clear break-even after charging DMA degradation and the unexplained scheduler/handoff margin.".to_string()
        },
        best_threads: best.threads,
        best_placement: best.placement,
        best_cpu_ids: best.cpu_ids.clone(),
        raw_prediction,
        conservative_margin_ms_per_token: conservative_prediction.net_saved_ms_per_token,
        conservative_prediction,
    };

    let mut persistent_single_sweep = Vec::new();
    for (placement, cpu_ids) in &persistent_single_candidates {
        let (latency, samples, dma_gib_s) = run_persistent_single_with_dma(
            Arc::clone(&dma),
            &geometry,
            &replicas[..replicas_per_team],
            &activation,
            cpu_ids,
            phase_seconds,
            cli.layer_gap_us.unwrap_or(0),
        )?;
        let prediction = prediction(
            &histogram,
            &samples,
            0.0,
            dma_gib_s,
            dma_baseline_gib_s,
            geometry.expert_bytes,
            cli.baseline_ms_per_token,
        );
        eprintln!(
            "CPU TAIL CALIBRATION persistent_single placement={} threads={} mean_ms={:.6} dma_gib_s={:.6} dma_degradation_pct={:.3} absorbed_per_token={:.6} net_saved_ms_per_token={:.6}",
            placement.label(),
            cpu_ids.len(),
            latency.mean_ms,
            dma_gib_s,
            (dma_baseline_gib_s - dma_gib_s) / dma_baseline_gib_s * 100.0,
            prediction.absorbed_experts_per_token,
            prediction.net_saved_ms_per_token,
        );
        persistent_single_sweep.push(PersistentSingleMeasurement {
            placement: *placement,
            threads: cpu_ids.len(),
            cpu_ids: cpu_ids.clone(),
            latency,
            dma_gib_s,
            dma_degradation_pct: (dma_baseline_gib_s - dma_gib_s) / dma_baseline_gib_s * 100.0,
            prediction,
        });
    }

    let mut two_team_sweep = Vec::new();
    for candidate in &two_team_candidates {
        let mut seen = BTreeSet::new();
        for cpu in candidate.cpu_ids.iter().flatten() {
            if !seen.insert(*cpu) {
                return Err(format!(
                    "two-team candidate has overlapping CPU {}: {:?}",
                    cpu, candidate.cpu_ids
                ));
            }
        }
        let (latencies, samples, dma_gib_s) = run_two_persistent_with_dma(
            Arc::clone(&dma),
            &geometry,
            &replicas,
            &activation,
            &candidate.cpu_ids,
            phase_seconds,
            cli.layer_gap_us.unwrap_or(0),
        )?;
        let prediction = two_team_prediction(
            &histogram,
            &samples,
            dma_gib_s,
            dma_baseline_gib_s,
            geometry.expert_bytes,
            cli.baseline_ms_per_token,
        );
        eprintln!(
            "CPU TAIL CALIBRATION two_team placement={} team_threads={},{} team0_mean_ms={:.6} team1_mean_ms={:.6} dma_gib_s={:.6} dma_degradation_pct={:.3} worker0_absorbed_per_token={:.6} worker1_absorbed_per_token={:.6} net_saved_ms_per_token={:.6}",
            candidate.placement.label(),
            candidate.cpu_ids[0].len(),
            candidate.cpu_ids[1].len(),
            latencies[0].mean_ms,
            latencies[1].mean_ms,
            dma_gib_s,
            (dma_baseline_gib_s - dma_gib_s) / dma_baseline_gib_s * 100.0,
            prediction.worker_0_absorbed_experts_per_token,
            prediction.worker_1_absorbed_experts_per_token,
            prediction.net_saved_ms_per_token,
        );
        two_team_sweep.push(TwoTeamMeasurement {
            placement: candidate.placement,
            team_threads: [candidate.cpu_ids[0].len(), candidate.cpu_ids[1].len()],
            team_cpu_ids: candidate.cpu_ids.clone(),
            team_latency: latencies,
            dma_gib_s,
            dma_degradation_pct: (dma_baseline_gib_s - dma_gib_s) / dma_baseline_gib_s * 100.0,
            prediction,
        });
    }
    let mut best_incremental: Option<(&TwoTeamMeasurement, &PersistentSingleMeasurement, f64)> =
        None;
    for row in &two_team_sweep {
        if row.prediction.worker_1_absorbed_experts_per_token <= 0.0 {
            continue;
        }
        let matched_single = persistent_single_sweep
            .iter()
            .find(|single| {
                single.placement == row.placement && single.cpu_ids == row.team_cpu_ids[0]
            })
            .ok_or_else(|| {
                format!(
                    "two-team row has no matched persistent single baseline: placement={} team0={:?}",
                    row.placement.label(),
                    row.team_cpu_ids[0]
                )
            })?;
        let incremental = row.prediction.net_saved_ms_per_token
            - matched_single.prediction.net_saved_ms_per_token;
        if best_incremental
            .as_ref()
            .map(|(_, _, best)| incremental > *best)
            .unwrap_or(true)
        {
            best_incremental = Some((row, matched_single, incremental));
        }
    }
    let two_team_optimizer =
        best_incremental.map(|(best_two, matched_single, incremental)| {
            TwoTeamOptimizerDecision {
                recommendation: if incremental > 0.0 {
                    "enable_candidate".to_string()
                } else {
                    "decline".to_string()
                },
                reason: if incremental > 0.0 {
                    "The best measured non-overlapping two-team split predicts more net token-time saving than the best measured single-team point after charging live DMA degradation.".to_string()
                } else {
                    "No measured two-team split improves predicted net token time over the best measured single-team point after charging live DMA degradation.".to_string()
                },
                layer_gap_us: cli.layer_gap_us.unwrap_or(0),
                best_placement: best_two.placement,
                best_team_threads: best_two.team_threads,
                best_team_cpu_ids: best_two.team_cpu_ids.clone(),
                prediction: best_two.prediction.clone(),
                matched_single_net_saved_ms_per_token: matched_single
                    .prediction
                    .net_saved_ms_per_token,
                incremental_net_saved_ms_per_token_vs_single: incremental,
            }
        });

    drop(dma);
    drop(device);
    let report = CalibrationReport {
        schema_version: 2,
        generated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs(),
        smoke: cli.smoke,
        requested_budget_seconds: cli.budget_seconds,
        elapsed_seconds: overall_started.elapsed().as_secs_f64(),
        geometry,
        gpu_ordinal,
        gpu_uuid,
        gpu_name,
        topology,
        queue_histogram: histogram,
        replica_count,
        synthetic_working_set_bytes: working_set_bytes,
        dma_baseline_windows_gib_s,
        dma_baseline_gib_s,
        cpu_only_sweep: cpu_only,
        concurrent_sweep: concurrent,
        best_format_comparison: format_comparison,
        optimizer,
        persistent_single_sweep,
        two_team_sweep,
        two_team_optimizer,
        gap_attribution,
        skipped_or_truncated,
    };
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create output directory {}: {e}", parent.display()))?;
    }
    fs::write(
        &cli.output,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| format!("serialize calibration report: {e}"))?,
    )
    .map_err(|e| format!("write {}: {e}", cli.output.display()))?;
    eprintln!(
        "CPU TAIL CALIBRATION RESULT recommendation={} best_threads={} best_placement={} concurrent_mean_ms={:.6} dma_gib_s={:.6} conservative_win_pct={:.3} absorbed_experts_per_token={:.6} net_saved_ms_per_token={:.6} predicted_tok_s={:?} output={}",
        report.optimizer.recommendation,
        report.optimizer.best_threads,
        report.optimizer.best_placement.label(),
        best_concurrent_ms,
        best_dma,
        report.optimizer.conservative_prediction.predicted_win_rate * 100.0,
        report.optimizer.conservative_prediction.absorbed_experts_per_token,
        report.optimizer.conservative_prediction.net_saved_ms_per_token,
        report.optimizer.conservative_prediction.predicted_tok_s,
        cli.output.display(),
    );
    if let Some(two_team) = report.two_team_optimizer.as_ref() {
        eprintln!(
            "CPU TAIL CALIBRATION TWO TEAM RESULT recommendation={} placement={} team_threads={},{} team0_mean_ms={:.6} team1_mean_ms={:.6} dma_gib_s={:.6} net_saved_ms_per_token={:.6} matched_single_net_saved_ms_per_token={:.6} incremental_vs_single_ms_per_token={:.6}",
            two_team.recommendation,
            two_team.best_placement.label(),
            two_team.best_team_threads[0],
            two_team.best_team_threads[1],
            report
                .two_team_sweep
                .iter()
                .find(|row| row.team_cpu_ids == two_team.best_team_cpu_ids)
                .map(|row| row.team_latency[0].mean_ms)
                .unwrap_or(0.0),
            report
                .two_team_sweep
                .iter()
                .find(|row| row.team_cpu_ids == two_team.best_team_cpu_ids)
                .map(|row| row.team_latency[1].mean_ms)
                .unwrap_or(0.0),
            report
                .two_team_sweep
                .iter()
                .find(|row| row.team_cpu_ids == two_team.best_team_cpu_ids)
                .map(|row| row.dma_gib_s)
                .unwrap_or(0.0),
            two_team.prediction.net_saved_ms_per_token,
            two_team.matched_single_net_saved_ms_per_token,
            two_team.incremental_net_saved_ms_per_token_vs_single,
        );
    }
    Ok(())
}
