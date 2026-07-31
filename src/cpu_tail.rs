//! Opportunistic CPU execution of one demand-cold tail expert.
//!
//! The GPU remains the deadline path. A persistent Rust worker consumes the
//! existing Marlin host allocation while CUDA copies surviving cold experts
//! from the front of the route queue. The caller accepts a CPU result only if
//! it wins that race; otherwise cancellation is cooperative and normal H2D
//! proceeds.

use crate::kernel::avx2::{
    expert_matmul_int4_transposed_integer, quantize_activation_int16,
    quantize_activation_int16_f32, silu_quantize_int16_avx2,
};
use crate::moe::{
    expert_forward_marlin_int4_cpu_tail, expert_forward_transposed_int4_cpu_tail, fast_sigmoid,
    ExpertScratch,
};
use crate::weights::marlin::f32_to_bf16;
use crate::weights::UnifiedExpertWeights;
use cudarc::driver::sys as cuda_sys;
use rayon::prelude::*;
use std::any::Any;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuTailWeightFormat {
    Marlin,
    Transposed,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuTailExpertRef {
    pub format: CpuTailWeightFormat,
    pub w13_packed_ptr: usize,
    pub w13_packed_len: usize,
    pub w13_scales_ptr: usize,
    pub w13_scales_len: usize,
    pub w2_packed_ptr: usize,
    pub w2_packed_len: usize,
    pub w2_scales_ptr: usize,
    pub w2_scales_len: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub group_size: usize,
    pub swiglu_limit: f32,
    pub activation_alpha: f32,
}

unsafe impl Send for CpuTailExpertRef {}

#[derive(Debug, Clone, Copy)]
pub struct CpuTailTransposedSource {
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub w13_packed_ptr: usize,
    pub w13_packed_len: usize,
    pub w13_scales_ptr: usize,
    pub w13_scales_len: usize,
    pub w2_packed_ptr: usize,
    pub w2_packed_len: usize,
    pub w2_scales_ptr: usize,
    pub w2_scales_len: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub group_size: usize,
}

pub struct CpuTailTransposedTier {
    experts: Vec<Vec<Option<UnifiedExpertWeights>>>,
    expert_count: usize,
    selected_resident_count: usize,
    total_bytes: usize,
    conversion_s: f64,
}

impl CpuTailTransposedTier {
    fn mem_available_bytes() -> Result<usize, String> {
        let contents = fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("CPU-tail transposed tier cannot read /proc/meminfo: {e}"))?;
        let available_kib = contents
            .lines()
            .find_map(|line| {
                let mut fields = line.split_whitespace();
                match (fields.next(), fields.next(), fields.next()) {
                    (Some("MemAvailable:"), Some(value), Some("kB")) => value.parse::<usize>().ok(),
                    _ => None,
                }
            })
            .ok_or_else(|| {
                "CPU-tail transposed tier could not parse MemAvailable from /proc/meminfo"
                    .to_string()
            })?;
        available_kib
            .checked_mul(1024)
            .ok_or_else(|| "CPU-tail transposed MemAvailable overflow".to_string())
    }

    pub fn build(
        layer_expert_counts: &[usize],
        selected_resident_count: usize,
        sources: Vec<CpuTailTransposedSource>,
    ) -> Result<Self, String> {
        let total_bytes = sources.iter().try_fold(0usize, |total, source| {
            let expert_bytes = source
                .w13_packed_len
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| {
                    source
                        .w13_scales_len
                        .checked_mul(std::mem::size_of::<u16>())
                        .and_then(|bytes| value.checked_add(bytes))
                })
                .and_then(|value| {
                    source
                        .w2_packed_len
                        .checked_mul(std::mem::size_of::<u32>())
                        .and_then(|bytes| value.checked_add(bytes))
                })
                .and_then(|value| {
                    source
                        .w2_scales_len
                        .checked_mul(std::mem::size_of::<u16>())
                        .and_then(|bytes| value.checked_add(bytes))
                })
                .ok_or_else(|| {
                    format!(
                        "CPU-tail transposed tier byte-size overflow at layer {} expert {}",
                        source.layer_idx, source.expert_idx
                    )
                })?;
            total
                .checked_add(expert_bytes)
                .ok_or_else(|| "CPU-tail transposed tier aggregate byte-size overflow".to_string())
        })?;

        let available_bytes = Self::mem_available_bytes()?;
        // The experiment must leave headroom proportional to its own footprint
        // for conversion workers, allocator metadata, and normal server growth.
        let proportional_headroom = total_bytes / 10;
        let required_with_headroom = total_bytes
            .checked_add(proportional_headroom)
            .ok_or_else(|| "CPU-tail transposed tier RAM requirement overflow".to_string())?;
        if required_with_headroom > available_bytes {
            return Err(format!(
                "KRASIS_CPU_TAIL_TRANSPOSED=1 requires {:.3} GiB duplicate RAM plus {:.3} GiB proportional headroom, but MemAvailable is {:.3} GiB; refusing to degrade or partially populate the tier",
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                proportional_headroom as f64 / (1024.0 * 1024.0 * 1024.0),
                available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            ));
        }

        eprintln!(
            "CPU TAIL TRANSPOSED BUILD start experts={} selected_residents={} bytes={} gib={:.3} mem_available_gib={:.3} proportional_headroom_gib={:.3}",
            sources.len(),
            selected_resident_count,
            total_bytes,
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            proportional_headroom as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        let started = Instant::now();
        let completed = AtomicU64::new(0);
        let progress_interval = (sources.len() / 20).max(1) as u64;
        let converted: Vec<Result<(usize, usize, UnifiedExpertWeights), String>> = sources
            .par_iter()
            .map(|source| {
                let w13_packed = unsafe {
                    std::slice::from_raw_parts(
                        source.w13_packed_ptr as *const u32,
                        source.w13_packed_len,
                    )
                };
                let w13_scales = unsafe {
                    std::slice::from_raw_parts(
                        source.w13_scales_ptr as *const u16,
                        source.w13_scales_len,
                    )
                };
                let w2_packed = unsafe {
                    std::slice::from_raw_parts(
                        source.w2_packed_ptr as *const u32,
                        source.w2_packed_len,
                    )
                };
                let w2_scales = unsafe {
                    std::slice::from_raw_parts(
                        source.w2_scales_ptr as *const u16,
                        source.w2_scales_len,
                    )
                };
                let expert = UnifiedExpertWeights::from_marlin_int4_transposed(
                    w13_packed,
                    w13_scales,
                    w2_packed,
                    w2_scales,
                    source.hidden_size,
                    source.intermediate_size,
                    source.group_size,
                )
                .map_err(|error| {
                    format!(
                        "CPU-tail transposed conversion failed at layer {} expert {}: {}",
                        source.layer_idx, source.expert_idx, error
                    )
                })?;
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % progress_interval == 0 || done == sources.len() as u64 {
                    eprintln!(
                        "CPU TAIL TRANSPOSED BUILD progress={}/{} elapsed_s={:.3}",
                        done,
                        sources.len(),
                        started.elapsed().as_secs_f64(),
                    );
                }
                Ok((source.layer_idx, source.expert_idx, expert))
            })
            .collect();

        let mut experts: Vec<Vec<Option<UnifiedExpertWeights>>> = layer_expert_counts
            .iter()
            .map(|&count| std::iter::repeat_with(|| None).take(count).collect())
            .collect();
        for converted_expert in converted {
            let (layer_idx, expert_idx, expert) = converted_expert?;
            let slot = experts
                .get_mut(layer_idx)
                .and_then(|layer| layer.get_mut(expert_idx))
                .ok_or_else(|| {
                    format!(
                        "CPU-tail transposed tier index out of range at layer {} expert {}",
                        layer_idx, expert_idx
                    )
                })?;
            *slot = Some(expert);
        }
        let conversion_s = started.elapsed().as_secs_f64();
        eprintln!(
            "CPU TAIL TRANSPOSED BUILD complete experts={} selected_residents={} bytes={} gib={:.3} conversion_s={:.3}",
            sources.len(),
            selected_resident_count,
            total_bytes,
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            conversion_s,
        );

        Ok(Self {
            experts,
            expert_count: sources.len(),
            selected_resident_count,
            total_bytes,
            conversion_s,
        })
    }

    pub fn expert(&self, layer_idx: usize, expert_idx: usize) -> Option<&UnifiedExpertWeights> {
        self.experts
            .get(layer_idx)
            .and_then(|layer| layer.get(expert_idx))
            .and_then(|expert| expert.as_ref())
    }

    pub fn expert_count(&self) -> usize {
        self.expert_count
    }

    pub fn selected_resident_count(&self) -> usize {
        self.selected_resident_count
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn conversion_s(&self) -> f64 {
        self.conversion_s
    }
}

impl Drop for CpuTailTransposedTier {
    fn drop(&mut self) {
        eprintln!(
            "CPU TAIL TRANSPOSED DROP experts={} bytes={} gib={:.3}",
            self.expert_count,
            self.total_bytes,
            self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
}

#[derive(Debug)]
struct CpuTailJob {
    sequence: u64,
    expert: CpuTailExpertRef,
    depth_bucket: usize,
    claimed: Option<Instant>,
    submitted: Option<Instant>,
}

enum CpuTailCommand {
    Run(CpuTailJob),
    Shutdown,
}

#[derive(Debug)]
pub struct CpuTailResult {
    pub sequence: u64,
    pub format: CpuTailWeightFormat,
    pub depth_bucket: usize,
    pub completed: bool,
    pub elapsed_s: f64,
    /// Submit-to-worker-pickup latency (timing mode only; 0.0 otherwise).
    pub dispatch_lag_s: f64,
    /// Worker pickup to the beginning of expert execution.
    pub pickup_to_worker_start_s: f64,
    /// Main-thread claim selection to the beginning of expert execution.
    pub claim_to_worker_start_s: f64,
    /// Expert execution itself, excluding channel wake and result visibility.
    pub kernel_compute_s: f64,
    /// Expert completion to the caller observing the channel result.
    pub compute_to_result_visible_s: f64,
    completed_at: Option<Instant>,
    pub error: Option<String>,
}

struct PinnedHostBuffer {
    ptr: *mut u8,
    size: usize,
}

impl PinnedHostBuffer {
    fn new(size: usize) -> Result<Self, String> {
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let err = unsafe {
            cuda_sys::lib().cuMemHostAlloc(
                &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
                size,
                0,
            )
        };
        if err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!("CPU tail cuMemHostAlloc({size}): {err:?}"));
        }
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        Ok(Self { ptr, size })
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cuda_sys::lib().cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

struct CalibratedCpuTailConfig {
    artifact: String,
    placement: String,
    cpu_ids: Vec<usize>,
    worker_index: usize,
    worker_count: usize,
}

#[cfg(target_os = "linux")]
pub(crate) fn process_allowed_cpus() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
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

pub fn configured_worker_count() -> Result<usize, String> {
    let Some(value) = std::env::var_os("KRASIS_CPU_TAIL_WORKERS") else {
        return Ok(1);
    };
    match value.to_string_lossy().as_ref() {
        "1" => Ok(1),
        "2" => {
            if std::env::var_os("KRASIS_CPU_TAIL_CALIBRATION_JSON").is_none() {
                return Err(
                    "KRASIS_CPU_TAIL_WORKERS=2 requires KRASIS_CPU_TAIL_CALIBRATION_JSON with a measured two_team_optimizer"
                        .to_string(),
                );
            }
            Ok(2)
        }
        other => Err(format!(
            "KRASIS_CPU_TAIL_WORKERS must be 1 or 2, got {other}"
        )),
    }
}

fn calibrated_cpu_tail_config(
    worker_index: usize,
    worker_count: usize,
) -> Result<Option<CalibratedCpuTailConfig>, String> {
    if worker_index >= worker_count || !(1..=2).contains(&worker_count) {
        return Err(format!(
            "invalid CPU-tail worker selection index={worker_index} count={worker_count}"
        ));
    }
    let Some(artifact_os) = std::env::var_os("KRASIS_CPU_TAIL_CALIBRATION_JSON") else {
        if worker_count != 1 {
            return Err("two-worker CPU tail requires a measured calibration artifact".to_string());
        }
        return Ok(None);
    };
    #[cfg(not(target_os = "linux"))]
    return Err(format!(
        "KRASIS_CPU_TAIL_CALIBRATION_JSON={} requests calibrated CPU affinity, but this build does not implement native thread affinity on the current OS",
        artifact_os.to_string_lossy(),
    ));

    #[cfg(target_os = "linux")]
    {
        let artifact = artifact_os.to_string_lossy().into_owned();
        let contents = fs::read_to_string(&artifact)
            .map_err(|e| format!("read CPU-tail calibration artifact {artifact}: {e}"))?;
        let report: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| format!("parse CPU-tail calibration artifact {artifact}: {e}"))?;
        let (placement, threads, cpu_ids, all_team_cpu_ids) = if worker_count == 1 {
            let optimizer = report.get("optimizer").ok_or_else(|| {
                format!("CPU-tail calibration artifact {artifact} has no optimizer object")
            })?;
            let threads = optimizer
                .get("best_threads")
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} has no positive optimizer.best_threads"
                    )
                })?;
            let placement = optimizer
                .get("best_placement")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} has no optimizer.best_placement"
                    )
                })?
                .to_string();
            let cpu_ids = parse_cpu_id_array(
                optimizer.get("best_cpu_ids"),
                &artifact,
                "optimizer.best_cpu_ids",
            )?;
            (placement, threads, cpu_ids.clone(), vec![cpu_ids])
        } else {
            let optimizer = report.get("two_team_optimizer").ok_or_else(|| {
                format!("CPU-tail calibration artifact {artifact} has no two_team_optimizer object")
            })?;
            if optimizer
                .get("recommendation")
                .and_then(|value| value.as_str())
                != Some("enable_candidate")
            {
                return Err(format!(
                    "CPU-tail calibration artifact {artifact} does not recommend enabling worker 2"
                ));
            }
            let placement = optimizer
                .get("best_placement")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} has no two_team_optimizer.best_placement"
                    )
                })?
                .to_string();
            let thread_values = optimizer
                .get("best_team_threads")
                .and_then(|value| value.as_array())
                .filter(|values| values.len() == 2)
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} has no two-element two_team_optimizer.best_team_threads"
                    )
                })?;
            let threads = thread_values[worker_index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} has invalid thread count for worker {worker_index}"
                    )
                })?;
            let team_values = optimizer
                .get("best_team_cpu_ids")
                .and_then(|value| value.as_array())
                .filter(|values| values.len() == 2)
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} has no two-element two_team_optimizer.best_team_cpu_ids"
                    )
                })?;
            let all_team_cpu_ids = team_values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    parse_cpu_id_array(
                        Some(value),
                        &artifact,
                        &format!("two_team_optimizer.best_team_cpu_ids[{index}]"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let cpu_ids = all_team_cpu_ids[worker_index].clone();
            (placement, threads, cpu_ids, all_team_cpu_ids)
        };
        if cpu_ids.len() != threads {
            return Err(format!(
                "CPU-tail calibration artifact {artifact} is inconsistent: best_threads={threads}, best_cpu_ids={}",
                cpu_ids.len(),
            ));
        }
        let mut unique = cpu_ids.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != cpu_ids.len() {
            return Err(format!(
                "CPU-tail calibration artifact {artifact} contains duplicate CPU ids: {cpu_ids:?}"
            ));
        }
        if worker_count == 2 {
            let mut combined: Vec<usize> = all_team_cpu_ids.iter().flatten().copied().collect();
            let combined_len = combined.len();
            combined.sort_unstable();
            combined.dedup();
            if combined.len() != combined_len {
                return Err(format!(
                    "CPU-tail calibration artifact {artifact} selects overlapping CPU sets for its two workers: {all_team_cpu_ids:?}"
                ));
            }
        }
        let allowed = process_allowed_cpus();
        if allowed.is_empty() {
            return Err(
                "cannot read process CPU affinity for calibrated CPU-tail pool".to_string(),
            );
        }
        if let Some(cpu) = cpu_ids.iter().find(|cpu| !allowed.contains(cpu)) {
            return Err(format!(
                "CPU-tail calibration artifact {artifact} selects CPU {cpu}, outside this process's allowed affinity set"
            ));
        }

        Ok(Some(CalibratedCpuTailConfig {
            artifact,
            placement,
            cpu_ids,
            worker_index,
            worker_count,
        }))
    }
}

fn parse_cpu_id_array(
    value: Option<&serde_json::Value>,
    artifact: &str,
    field: &str,
) -> Result<Vec<usize>, String> {
    value
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            format!("CPU-tail calibration artifact {artifact} has no {field} array")
        })?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    format!(
                        "CPU-tail calibration artifact {artifact} field {field} contains a non-integer CPU id"
                    )
                })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct PersistentMatmulJob {
    packed_addr: usize,
    scales_addr: usize,
    act_addr: usize,
    act_scales_addr: usize,
    output_addr: usize,
    k: usize,
    n: usize,
    group_size: usize,
    spin_after: bool,
}

struct PersistentTeamState {
    job: Option<PersistentMatmulJob>,
    completed_workers: usize,
    error: Option<String>,
}

struct PersistentTeamShared {
    state: Mutex<PersistentTeamState>,
    work_ready: Condvar,
    work_done: Condvar,
    generation: AtomicU64,
    shutdown: AtomicBool,
}

pub(crate) struct PersistentTransposedTeam {
    shared: Arc<PersistentTeamShared>,
    workers: Vec<JoinHandle<()>>,
}

impl PersistentTransposedTeam {
    pub(crate) fn new(cpu_ids: &[usize]) -> Result<Self, String> {
        if cpu_ids.is_empty() {
            return Err("persistent CPU-tail team requires at least one CPU".to_string());
        }
        let shared = Arc::new(PersistentTeamShared {
            state: Mutex::new(PersistentTeamState {
                job: None,
                completed_workers: 0,
                error: None,
            }),
            work_ready: Condvar::new(),
            work_done: Condvar::new(),
            generation: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        });
        let (started_tx, started_rx) = mpsc::sync_channel(cpu_ids.len());
        let mut workers: Vec<JoinHandle<()>> = Vec::with_capacity(cpu_ids.len());
        let worker_count = cpu_ids.len();
        for (worker_index, &cpu_id) in cpu_ids.iter().enumerate() {
            let worker_shared = Arc::clone(&shared);
            let worker_started = started_tx.clone();
            let handle = match std::thread::Builder::new()
                .name(format!("krasis-cpu-tail-matmul-{worker_index}"))
                .spawn(move || {
                    let pinned = pin_current_thread(cpu_id);
                    let _ = worker_started.send((worker_index, cpu_id, pinned));
                    if !pinned {
                        return;
                    }
                    let mut observed_generation = 0u64;
                    loop {
                        let generation = worker_shared.generation.load(Ordering::Acquire);
                        if generation == observed_generation {
                            let mut state = worker_shared.state.lock().unwrap();
                            while worker_shared.generation.load(Ordering::Acquire)
                                == observed_generation
                                && !worker_shared.shutdown.load(Ordering::Acquire)
                            {
                                state = worker_shared.work_ready.wait(state).unwrap();
                            }
                            drop(state);
                        }
                        if worker_shared.shutdown.load(Ordering::Acquire) {
                            break;
                        }
                        let generation = worker_shared.generation.load(Ordering::Acquire);
                        let job = {
                            let state = worker_shared.state.lock().unwrap();
                            state.job.expect("persistent CPU-tail team job missing")
                        };
                        observed_generation = generation;
                        let outcome = std::panic::catch_unwind(|| unsafe {
                            let chunk_n = 256usize;
                            let chunks = job.n.div_ceil(chunk_n);
                            for chunk_index in (worker_index..chunks).step_by(worker_count) {
                                let n_start = chunk_index * chunk_n;
                                let n_count = (job.n - n_start).min(chunk_n);
                                expert_matmul_int4_transposed_integer(
                                    job.packed_addr as *const u32,
                                    job.scales_addr as *const u16,
                                    job.act_addr as *const i16,
                                    job.act_scales_addr as *const f32,
                                    (job.output_addr as *mut f32).add(n_start),
                                    job.k,
                                    job.n,
                                    n_start,
                                    n_count,
                                    job.group_size,
                                );
                            }
                        });
                        let mut state = worker_shared.state.lock().unwrap();
                        if let Err(payload) = outcome {
                            state.error.get_or_insert_with(|| {
                                format!(
                                    "persistent CPU-tail matmul worker panic: {}",
                                    panic_message(payload)
                                )
                            });
                        }
                        state.completed_workers += 1;
                        if state.completed_workers == worker_count {
                            worker_shared.work_done.notify_one();
                        }
                        drop(state);

                        // W13 is always followed immediately by activation math
                        // and W2. Stay awake across that known phase boundary,
                        // but park after W2 so an idle server consumes no cores.
                        if job.spin_after {
                            while worker_shared.generation.load(Ordering::Acquire)
                                == observed_generation
                                && !worker_shared.shutdown.load(Ordering::Acquire)
                            {
                                std::hint::spin_loop();
                            }
                        }
                    }
                }) {
                Ok(handle) => handle,
                Err(error) => {
                    shared.shutdown.store(true, Ordering::Release);
                    shared.generation.fetch_add(1, Ordering::Release);
                    shared.work_ready.notify_all();
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(format!(
                        "spawn persistent CPU-tail matmul worker {worker_index}: {error}"
                    ));
                }
            };
            workers.push(handle);
        }
        drop(started_tx);
        let mut failures = Vec::new();
        for _ in 0..cpu_ids.len() {
            match started_rx.recv() {
                Ok((worker_index, cpu_id, true)) => {
                    let _ = (worker_index, cpu_id);
                }
                Ok((worker_index, cpu_id, false)) => {
                    failures.push(format!("worker {worker_index} CPU {cpu_id}"));
                }
                Err(error) => failures.push(format!("startup channel: {error}")),
            }
        }
        if !failures.is_empty() {
            shared.shutdown.store(true, Ordering::Release);
            shared.generation.fetch_add(1, Ordering::Release);
            shared.work_ready.notify_all();
            for worker in workers {
                let _ = worker.join();
            }
            return Err(format!(
                "persistent CPU-tail team affinity failed: {}",
                failures.join(", ")
            ));
        }
        Ok(Self { shared, workers })
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        packed: &[u32],
        scales: &[u16],
        act_int16: &[i16],
        act_scales: &[f32],
        output: &mut [f32],
        k: usize,
        n: usize,
        group_size: usize,
        spin_after: bool,
    ) -> Result<(), String> {
        let mut state = self.shared.state.lock().unwrap();
        state.job = Some(PersistentMatmulJob {
            packed_addr: packed.as_ptr() as usize,
            scales_addr: scales.as_ptr() as usize,
            act_addr: act_int16.as_ptr() as usize,
            act_scales_addr: act_scales.as_ptr() as usize,
            output_addr: output.as_mut_ptr() as usize,
            k,
            n,
            group_size,
            spin_after,
        });
        state.completed_workers = 0;
        state.error = None;
        self.shared.generation.fetch_add(1, Ordering::Release);
        self.shared.work_ready.notify_all();
        while state.completed_workers != self.workers.len() {
            state = self.shared.work_done.wait(state).unwrap();
        }
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        Ok(())
    }

    fn park_after_w13(&self) -> Result<(), String> {
        let mut state = self.shared.state.lock().unwrap();
        state.job = Some(PersistentMatmulJob {
            packed_addr: 0,
            scales_addr: 0,
            act_addr: 0,
            act_scales_addr: 0,
            output_addr: 0,
            k: 0,
            n: 0,
            group_size: 1,
            spin_after: false,
        });
        state.completed_workers = 0;
        state.error = None;
        self.shared.generation.fetch_add(1, Ordering::Release);
        self.shared.work_ready.notify_all();
        while state.completed_workers != self.workers.len() {
            state = self.shared.work_done.wait(state).unwrap();
        }
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        Ok(())
    }
}

impl Drop for PersistentTransposedTeam {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        self.shared.work_ready.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_forward_transposed_persistent(
    team: &PersistentTransposedTeam,
    w13_packed: &[u32],
    w13_scales: &[u16],
    w2_packed: &[u32],
    w2_scales: &[u16],
    activation_bf16: &[u16],
    output_bf16: &mut [u16],
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
    swiglu_limit: f32,
    activation_alpha: f32,
    scratch: &mut ExpertScratch,
    cancel_sequence: &AtomicU64,
    sequence: u64,
) -> Result<bool, String> {
    quantize_activation_int16(
        activation_bf16,
        group_size,
        &mut scratch.input_act_int16,
        &mut scratch.input_act_scales,
    );
    if cancel_sequence.load(Ordering::Acquire) == sequence {
        return Ok(false);
    }
    team.matmul(
        w13_packed,
        w13_scales,
        &scratch.input_act_int16,
        &scratch.input_act_scales,
        &mut scratch.w13_out,
        hidden_size,
        2 * intermediate_size,
        group_size,
        true,
    )?;
    if cancel_sequence.load(Ordering::Acquire) == sequence {
        team.park_after_w13()?;
        return Ok(false);
    }

    if swiglu_limit > 0.0 {
        for i in 0..intermediate_size {
            let mut gate = scratch.w13_out[i];
            let mut up = scratch.w13_out[intermediate_size + i];
            if gate > swiglu_limit {
                gate = swiglu_limit;
            }
            if up > swiglu_limit {
                up = swiglu_limit;
            }
            if up < -swiglu_limit {
                up = -swiglu_limit;
            }
            scratch.w13_out[i] = (up + 1.0) * gate * fast_sigmoid(gate * activation_alpha);
        }
        quantize_activation_int16_f32(
            &scratch.w13_out[..intermediate_size],
            group_size,
            &mut scratch.hidden_int16,
            &mut scratch.hidden_scales,
        );
    } else {
        let w13_ptr = scratch.w13_out.as_mut_ptr();
        unsafe {
            silu_quantize_int16_avx2(
                w13_ptr,
                w13_ptr.add(intermediate_size),
                scratch.hidden_int16.as_mut_ptr(),
                scratch.hidden_scales.as_mut_ptr(),
                intermediate_size,
                group_size,
            );
        }
    }
    if cancel_sequence.load(Ordering::Acquire) == sequence {
        team.park_after_w13()?;
        return Ok(false);
    }
    team.matmul(
        w2_packed,
        w2_scales,
        &scratch.hidden_int16,
        &scratch.hidden_scales,
        &mut scratch.expert_out,
        intermediate_size,
        hidden_size,
        group_size,
        false,
    )?;
    if cancel_sequence.load(Ordering::Acquire) == sequence {
        return Ok(false);
    }
    for (dst, &src) in output_bf16.iter_mut().zip(&scratch.expert_out) {
        *dst = f32_to_bf16(src);
    }
    Ok(true)
}

pub struct CpuTailRuntime {
    hidden_size: usize,
    timing_enabled: bool,
    input: PinnedHostBuffer,
    output: PinnedHostBuffer,
    command_tx: SyncSender<CpuTailCommand>,
    result_rx: Mutex<Receiver<CpuTailResult>>,
    cancel_sequence: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
    next_sequence: u64,
    busy_sequence: Option<u64>,
}

impl CpuTailRuntime {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        group_size: usize,
        timing_enabled: bool,
    ) -> Result<Self, String> {
        Self::new_for_worker(
            hidden_size,
            intermediate_size,
            group_size,
            timing_enabled,
            0,
            1,
        )
    }

    pub fn new_for_worker(
        hidden_size: usize,
        intermediate_size: usize,
        group_size: usize,
        timing_enabled: bool,
        worker_index: usize,
        worker_count: usize,
    ) -> Result<Self, String> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return Err("CPU tail requires an x86 CPU with AVX2 and FMA support".to_string());
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        return Err("CPU tail currently requires an x86 CPU with AVX2 and FMA support".to_string());

        if hidden_size == 0 || intermediate_size == 0 || group_size == 0 {
            return Err(format!(
                "CPU tail dimensions must be non-zero: hidden={hidden_size} intermediate={intermediate_size} group={group_size}"
            ));
        }
        if hidden_size % 64 != 0
            || intermediate_size % 32 != 0
            || group_size % 16 != 0
            || group_size >= hidden_size
            || group_size >= intermediate_size
        {
            return Err(format!(
                "CPU tail unsupported Marlin dimensions: hidden={hidden_size} (must be divisible by 64), intermediate={intermediate_size} (must be divisible by 32), group={group_size} (must be divisible by 16 and smaller than both matrix K dimensions)"
            ));
        }
        let bytes = hidden_size
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "CPU tail pinned buffer size overflow".to_string())?;
        let input = PinnedHostBuffer::new(bytes)?;
        let output = PinnedHostBuffer::new(bytes)?;
        let input_addr = input.ptr as usize;
        let output_addr = output.ptr as usize;
        let (command_tx, command_rx) = mpsc::sync_channel::<CpuTailCommand>(1);
        let (result_tx, result_rx) = mpsc::sync_channel::<CpuTailResult>(1);
        let cancel_sequence = Arc::new(AtomicU64::new(0));
        let worker_cancel = Arc::clone(&cancel_sequence);
        let calibrated_config = calibrated_cpu_tail_config(worker_index, worker_count)?;
        let persistent_team = calibrated_config
            .as_ref()
            .map(|config| PersistentTransposedTeam::new(&config.cpu_ids))
            .transpose()?;
        if let Some(config) = calibrated_config.as_ref() {
            eprintln!(
                "CPU TAIL EXECUTOR worker={}/{} source=calibration artifact={} placement={} threads={} cpu_ids={} strategy=persistent_two_phase",
                config.worker_index,
                config.worker_count,
                config.artifact,
                config.placement,
                config.cpu_ids.len(),
                config
                    .cpu_ids
                    .iter()
                    .map(|cpu| cpu.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        } else {
            eprintln!(
                "CPU TAIL EXECUTOR worker={}/{} source=global rayon_threads={} affinity=runtime_default strategy=rayon",
                worker_index,
                worker_count,
                rayon::current_num_threads(),
            );
        }

        let worker = std::thread::Builder::new()
            .name("krasis-cpu-tail".to_string())
            .spawn(move || {
                let mut scratch = ExpertScratch::new(hidden_size, intermediate_size, group_size);
                while let Ok(command) = command_rx.recv() {
                    let CpuTailCommand::Run(job) = command else {
                        break;
                    };
                    let picked_up = timing_enabled.then(Instant::now);
                    let dispatch_lag_s = match (picked_up.as_ref(), job.submitted.as_ref()) {
                        (Some(picked_up), Some(submitted)) => {
                            picked_up.duration_since(*submitted).as_secs_f64()
                        }
                        _ => 0.0,
                    };
                    let worker_started = timing_enabled.then(Instant::now);
                    let pickup_to_worker_start_s =
                        match (worker_started.as_ref(), picked_up.as_ref()) {
                            (Some(worker_started), Some(picked_up)) => {
                                worker_started.duration_since(*picked_up).as_secs_f64()
                            }
                            _ => 0.0,
                        };
                    let claim_to_worker_start_s =
                        match (worker_started.as_ref(), job.claimed.as_ref()) {
                            (Some(worker_started), Some(claimed)) => {
                                worker_started.duration_since(*claimed).as_secs_f64()
                            }
                            _ => 0.0,
                        };
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let expert = job.expert;
                        if expert.hidden_size != hidden_size
                            || expert.intermediate_size != intermediate_size
                            || expert.group_size != group_size
                        {
                            panic!(
                                "CPU tail job shape changed: job hidden={} intermediate={} group={} runtime hidden={} intermediate={} group={}",
                                expert.hidden_size,
                                expert.intermediate_size,
                                expert.group_size,
                                hidden_size,
                                intermediate_size,
                                group_size,
                            );
                        }
                        let w13_packed = unsafe {
                            std::slice::from_raw_parts(
                                expert.w13_packed_ptr as *const u32,
                                expert.w13_packed_len,
                            )
                        };
                        let w13_scales = unsafe {
                            std::slice::from_raw_parts(
                                expert.w13_scales_ptr as *const u16,
                                expert.w13_scales_len,
                            )
                        };
                        let w2_packed = unsafe {
                            std::slice::from_raw_parts(
                                expert.w2_packed_ptr as *const u32,
                                expert.w2_packed_len,
                            )
                        };
                        let w2_scales = unsafe {
                            std::slice::from_raw_parts(
                                expert.w2_scales_ptr as *const u16,
                                expert.w2_scales_len,
                            )
                        };
                        let activation = unsafe {
                            std::slice::from_raw_parts(
                                input_addr as *const u16,
                                hidden_size,
                            )
                        };
                        let output = unsafe {
                            std::slice::from_raw_parts_mut(
                                output_addr as *mut u16,
                                hidden_size,
                            )
                        };
                        let mut execute = || -> Result<bool, String> {
                            match expert.format {
                            CpuTailWeightFormat::Marlin => {
                                Ok(expert_forward_marlin_int4_cpu_tail(
                                    w13_packed,
                                    w13_scales,
                                    w2_packed,
                                    w2_scales,
                                    activation,
                                    output,
                                    hidden_size,
                                    intermediate_size,
                                    group_size,
                                    expert.swiglu_limit,
                                    expert.activation_alpha,
                                    &mut scratch,
                                    &worker_cancel,
                                    job.sequence,
                                ))
                            }
                            CpuTailWeightFormat::Transposed => {
                                if let Some(team) = persistent_team.as_ref() {
                                    expert_forward_transposed_persistent(
                                        team,
                                        w13_packed,
                                        w13_scales,
                                        w2_packed,
                                        w2_scales,
                                        activation,
                                        output,
                                        hidden_size,
                                        intermediate_size,
                                        group_size,
                                        expert.swiglu_limit,
                                        expert.activation_alpha,
                                        &mut scratch,
                                        &worker_cancel,
                                        job.sequence,
                                    )
                                } else {
                                    Ok(expert_forward_transposed_int4_cpu_tail(
                                        w13_packed,
                                        w13_scales,
                                        w2_packed,
                                        w2_scales,
                                        activation,
                                        output,
                                        hidden_size,
                                        intermediate_size,
                                        group_size,
                                        expert.swiglu_limit,
                                        expert.activation_alpha,
                                        &mut scratch,
                                        &worker_cancel,
                                        job.sequence,
                                    ))
                                }
                            }
                        }
                        };
                        execute()
                    }));
                    let completed_at = timing_enabled.then(Instant::now);
                    let (completed, error) = match outcome {
                        Ok(Ok(completed)) => (completed, None),
                        Ok(Err(error)) => (false, Some(error)),
                        Err(payload) => (
                            false,
                            Some(format!("CPU tail worker panic: {}", panic_message(payload))),
                        ),
                    };
                    if result_tx
                        .send(CpuTailResult {
                            sequence: job.sequence,
                            format: job.expert.format,
                            depth_bucket: job.depth_bucket,
                            completed,
                            elapsed_s: match (completed_at.as_ref(), picked_up.as_ref()) {
                                (Some(completed_at), Some(picked_up)) => {
                                    completed_at.duration_since(*picked_up).as_secs_f64()
                                }
                                _ => 0.0,
                            },
                            dispatch_lag_s,
                            pickup_to_worker_start_s,
                            claim_to_worker_start_s,
                            kernel_compute_s: match (
                                completed_at.as_ref(),
                                worker_started.as_ref(),
                            ) {
                                (Some(completed_at), Some(worker_started)) => completed_at
                                    .duration_since(*worker_started)
                                    .as_secs_f64(),
                                _ => 0.0,
                            },
                            compute_to_result_visible_s: 0.0,
                            completed_at,
                            error,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|e| format!("failed to spawn CPU tail worker: {e}"))?;

        Ok(Self {
            hidden_size,
            timing_enabled,
            input,
            output,
            command_tx,
            result_rx: Mutex::new(result_rx),
            cancel_sequence,
            worker: Some(worker),
            next_sequence: 1,
            busy_sequence: None,
        })
    }

    pub fn input_ptr(&self) -> *mut u8 {
        self.input.ptr
    }

    pub fn output_ptr(&self) -> *const u8 {
        self.output.ptr
    }

    pub fn output_bytes(&self) -> usize {
        self.output.size
    }

    pub fn is_idle(&mut self) -> Result<bool, String> {
        if self.busy_sequence.is_none() {
            return Ok(true);
        }
        match self.try_result()? {
            Some(result) => {
                if let Some(error) = result.error {
                    return Err(error);
                }
                self.busy_sequence = None;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn submit(
        &mut self,
        expert: CpuTailExpertRef,
        depth_bucket: usize,
        claimed: Option<Instant>,
    ) -> Result<u64, String> {
        if !self.is_idle()? {
            return Err("CPU tail worker is still busy".to_string());
        }
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        self.cancel_sequence.store(0, Ordering::Release);
        self.command_tx
            .send(CpuTailCommand::Run(CpuTailJob {
                sequence,
                expert,
                depth_bucket,
                claimed,
                submitted: self.timing_enabled.then(Instant::now),
            }))
            .map_err(|_| "CPU tail worker command channel disconnected".to_string())?;
        self.busy_sequence = Some(sequence);
        Ok(sequence)
    }

    pub fn try_result(&mut self) -> Result<Option<CpuTailResult>, String> {
        let result_rx = self
            .result_rx
            .lock()
            .map_err(|_| "CPU tail result channel mutex was poisoned".to_string())?;
        match result_rx.try_recv() {
            Ok(mut result) => {
                if self.busy_sequence != Some(result.sequence) {
                    return Err(format!(
                        "CPU tail result sequence mismatch: busy={:?} result={}",
                        self.busy_sequence, result.sequence
                    ));
                }
                if let Some(completed_at) = result.completed_at.take() {
                    result.compute_to_result_visible_s = completed_at.elapsed().as_secs_f64();
                }
                Ok(Some(result))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err("CPU tail worker result channel disconnected".to_string())
            }
        }
    }

    pub fn cancel(&self, sequence: u64) {
        self.cancel_sequence.store(sequence, Ordering::Release);
    }

    pub fn finish(&mut self, sequence: u64) -> Result<(), String> {
        if self.busy_sequence != Some(sequence) {
            return Err(format!(
                "CPU tail finish sequence mismatch: busy={:?} finish={sequence}",
                self.busy_sequence
            ));
        }
        self.busy_sequence = None;
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn busy_sequence(&self) -> Option<u64> {
        self.busy_sequence
    }
}

impl Drop for CpuTailRuntime {
    fn drop(&mut self) {
        if let Some(sequence) = self.busy_sequence {
            self.cancel(sequence);
        }
        let _ = self.command_tx.send(CpuTailCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn two_persistent_cpu_tail_teams_cancel_independently_and_together() {
        let allowed = process_allowed_cpus();
        assert!(
            allowed.len() >= 2,
            "two-team CPU-tail test requires at least two allowed CPUs"
        );
        let split = (allowed.len() / 2).max(1);
        let team0_ids = &allowed[..split];
        let team1_ids = &allowed[split..];
        assert!(!team1_ids.is_empty());

        let hidden = 1024usize;
        let intermediate = 512usize;
        let group_size = 128usize;
        let w13_packed = vec![0x1234_5678u32; (hidden / 8) * (2 * intermediate)];
        let w2_packed = vec![0x89ab_cdefu32; (intermediate / 8) * hidden];
        let scale = f32_to_bf16(1.0 / i16::MAX as f32);
        let w13_scales = vec![scale; (hidden / group_size) * (2 * intermediate)];
        let w2_scales = vec![scale; (intermediate / group_size) * hidden];
        let activation = vec![f32_to_bf16(0.01); hidden];

        let run_pair = |cancel0: u64, cancel1: u64, sequence0: u64, sequence1: u64| {
            let team0 = PersistentTransposedTeam::new(team0_ids).expect("team 0");
            let team1 = PersistentTransposedTeam::new(team1_ids).expect("team 1");
            let cancel0_atomic = AtomicU64::new(cancel0);
            let cancel1_atomic = AtomicU64::new(cancel1);
            std::thread::scope(|scope| {
                let first = scope.spawn(|| {
                    let mut output = vec![0u16; hidden];
                    let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);
                    expert_forward_transposed_persistent(
                        &team0,
                        &w13_packed,
                        &w13_scales,
                        &w2_packed,
                        &w2_scales,
                        &activation,
                        &mut output,
                        hidden,
                        intermediate,
                        group_size,
                        0.0,
                        1.0,
                        &mut scratch,
                        &cancel0_atomic,
                        sequence0,
                    )
                    .expect("team 0 forward")
                });
                let second = scope.spawn(|| {
                    let mut output = vec![0u16; hidden];
                    let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);
                    expert_forward_transposed_persistent(
                        &team1,
                        &w13_packed,
                        &w13_scales,
                        &w2_packed,
                        &w2_scales,
                        &activation,
                        &mut output,
                        hidden,
                        intermediate,
                        group_size,
                        0.0,
                        1.0,
                        &mut scratch,
                        &cancel1_atomic,
                        sequence1,
                    )
                    .expect("team 1 forward")
                });
                (first.join().unwrap(), second.join().unwrap())
            })
        };

        assert_eq!(run_pair(0, 11, 10, 11), (true, false));
        assert_eq!(run_pair(20, 21, 20, 21), (false, false));
    }

    #[test]
    fn persistent_transposed_team_matches_rayon_glm_shape_and_tears_down() {
        let hidden = 6144usize;
        let intermediate = 2048usize;
        let group_size = 128usize;
        let w13_words = (hidden / 8) * (2 * intermediate);
        let w2_words = (intermediate / 8) * hidden;
        let w13_packed: Vec<u32> = (0..w13_words)
            .map(|index| (index as u32).wrapping_mul(0x9e37_79b9))
            .collect();
        let w2_packed: Vec<u32> = (0..w2_words)
            .map(|index| (index as u32).wrapping_mul(0x85eb_ca6b))
            .collect();
        let scale = f32_to_bf16(1.0 / i16::MAX as f32);
        let w13_scales = vec![scale; (hidden / group_size) * (2 * intermediate)];
        let w2_scales = vec![scale; (intermediate / group_size) * hidden];
        let activation: Vec<u16> = (0..hidden)
            .map(|index| f32_to_bf16(((index % group_size) as f32 / group_size as f32 - 0.5) * 0.1))
            .collect();
        let cancel = AtomicU64::new(0);
        let mut rayon_scratch = ExpertScratch::new(hidden, intermediate, group_size);
        let mut rayon_output = vec![0u16; hidden];
        assert!(expert_forward_transposed_int4_cpu_tail(
            &w13_packed,
            &w13_scales,
            &w2_packed,
            &w2_scales,
            &activation,
            &mut rayon_output,
            hidden,
            intermediate,
            group_size,
            0.0,
            1.0,
            &mut rayon_scratch,
            &cancel,
            1,
        ));

        let persistent_bench =
            std::env::var("KRASIS_CPU_TAIL_PERSISTENT_BENCH").as_deref() == Ok("1");
        let mut cpu_ids = process_allowed_cpus();
        if !persistent_bench {
            cpu_ids.truncate(cpu_ids.len().min(4));
        }
        let team = PersistentTransposedTeam::new(&cpu_ids).expect("persistent team");
        let mut persistent_scratch = ExpertScratch::new(hidden, intermediate, group_size);
        let mut persistent_output = vec![0u16; hidden];
        assert!(expert_forward_transposed_persistent(
            &team,
            &w13_packed,
            &w13_scales,
            &w2_packed,
            &w2_scales,
            &activation,
            &mut persistent_output,
            hidden,
            intermediate,
            group_size,
            0.0,
            1.0,
            &mut persistent_scratch,
            &cancel,
            1,
        )
        .expect("persistent forward"));
        assert_eq!(persistent_output, rayon_output);

        if persistent_bench {
            let gap_us = std::env::var("KRASIS_CPU_TAIL_BENCH_GAP_US")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let iterations = std::env::var("KRASIS_CPU_TAIL_BENCH_ITERS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(128)
                .max(1);
            let mut compute_s = 0.0f64;
            for _ in 0..iterations {
                if gap_us != 0 {
                    std::thread::sleep(std::time::Duration::from_micros(gap_us));
                }
                let started = Instant::now();
                assert!(expert_forward_transposed_persistent(
                    &team,
                    &w13_packed,
                    &w13_scales,
                    &w2_packed,
                    &w2_scales,
                    &activation,
                    &mut persistent_output,
                    hidden,
                    intermediate,
                    group_size,
                    0.0,
                    1.0,
                    &mut persistent_scratch,
                    &cancel,
                    1,
                )
                .expect("persistent bench forward"));
                compute_s += started.elapsed().as_secs_f64();
            }
            eprintln!(
                "CPU_TAIL_PERSISTENT_BENCH threads={} gap_us={} compute_ms_per_expert={:.6}",
                cpu_ids.len(),
                gap_us,
                compute_s / iterations as f64 * 1000.0,
            );
        }

        cancel.store(2, Ordering::Release);
        assert!(!expert_forward_transposed_persistent(
            &team,
            &w13_packed,
            &w13_scales,
            &w2_packed,
            &w2_scales,
            &activation,
            &mut persistent_output,
            hidden,
            intermediate,
            group_size,
            0.0,
            1.0,
            &mut persistent_scratch,
            &cancel,
            2,
        )
        .expect("persistent cancellation"));
        drop(team);
    }
}
