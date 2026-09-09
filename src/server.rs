//! Rust HTTP server for Krasis — replaces Python FastAPI/uvicorn.
//!
//! Handles tokenization, HTTP parsing, and SSE streaming entirely in Rust.
//! Prefill and decode run in Rust on the production request path.
//! Python remains for startup/orchestration and model ownership.
//!
//! Single-request at a time (matches our hardware constraint).

use crate::gpu_decode::GpuDecodeStore;
use sha2::{Digest, Sha256};

/// Streaming detokenizer that buffers incomplete UTF-8 sequences.
///
/// Some characters (emojis, CJK, etc.) span multiple tokens in byte-level BPE.
/// Decoding each token individually produces incomplete UTF-8 bytes → U+FFFD.
/// This struct buffers tokens until the decoded text contains no trailing FFFD,
/// then emits the complete text.
pub struct StreamDetokenizer<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    pending: Vec<u32>,
    hidden_token_ids: std::collections::HashSet<u32>,
    preserved_special_tokens: std::collections::HashMap<u32, &'static str>,
}

impl<'a> StreamDetokenizer<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::new(),
            hidden_token_ids: std::collections::HashSet::new(),
            preserved_special_tokens: std::collections::HashMap::new(),
        }
    }

    /// Preserve native grammar delimiters that a model registers as special
    /// tokens, while still suppressing the request's measured stop IDs.
    pub fn for_tool_calls(
        tokenizer: &'a tokenizers::Tokenizer,
        stop_ids: &[usize],
        grammar_tokens: &'static [&'static str],
    ) -> Self {
        let hidden_token_ids: std::collections::HashSet<u32> = stop_ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();
        let preserved_special_tokens = grammar_tokens
            .iter()
            .filter_map(|&token| tokenizer.token_to_id(token).map(|id| (id, token)))
            .filter(|(id, _)| !hidden_token_ids.contains(id))
            .collect();
        Self {
            tokenizer,
            pending: Vec::new(),
            hidden_token_ids,
            preserved_special_tokens,
        }
    }

    /// Add a token. Returns the decoded text if the sequence is complete UTF-8,
    /// or an empty string if we're still buffering incomplete bytes.
    pub fn add(&mut self, token_id: u32) -> String {
        if self.hidden_token_ids.contains(&token_id) {
            return self.flush();
        }
        if let Some(&token) = self.preserved_special_tokens.get(&token_id) {
            let mut decoded = self.flush();
            decoded.push_str(token);
            return decoded;
        }
        self.pending.push(token_id);
        let decoded = self
            .tokenizer
            .decode(&self.pending, true)
            .unwrap_or_default();
        if decoded.is_empty() {
            return String::new();
        }
        // If the decoded text ends with U+FFFD, we likely have incomplete bytes.
        // Buffer up to 8 tokens before giving up and emitting anyway.
        if decoded.ends_with('\u{FFFD}') && self.pending.len() < 8 {
            return String::new();
        }
        self.pending.clear();
        decoded
    }

    /// Flush any remaining buffered tokens (end of stream).
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let decoded = self
            .tokenizer
            .decode(&self.pending, true)
            .unwrap_or_default();
        self.pending.clear();
        decoded
    }
}

fn decode_token_preserving_tool_specials(
    tokenizer: &tokenizers::Tokenizer,
    token_id: u32,
    grammar_tokens: &'static [&'static str],
) -> String {
    if let Some(token) = grammar_tokens
        .iter()
        .copied()
        .find(|token| tokenizer.token_to_id(token) == Some(token_id))
    {
        token.to_string()
    } else {
        tokenizer.decode(&[token_id], true).unwrap_or_default()
    }
}
use pyo3::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

const LATENCY_BUCKETS: [f64; 14] = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 60.0,
];

#[derive(Clone, Default)]
struct PromHistogram {
    buckets: [u64; LATENCY_BUCKETS.len()],
    count: u64,
    sum: f64,
}

impl PromHistogram {
    fn observe(&mut self, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        self.count = self.count.saturating_add(1);
        self.sum += seconds;
        for (index, upper) in LATENCY_BUCKETS.iter().enumerate() {
            if seconds <= *upper {
                self.buckets[index] = self.buckets[index].saturating_add(1);
            }
        }
    }

    fn render(&self, output: &mut String, name: &str) {
        for (index, upper) in LATENCY_BUCKETS.iter().enumerate() {
            output.push_str(&format!(
                "{name}_bucket{{le=\"{upper}\"}} {}\n",
                self.buckets[index]
            ));
        }
        output.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {}\n", self.count));
        output.push_str(&format!(
            "{name}_sum {}\n{name}_count {}\n",
            self.sum, self.count
        ));
    }
}

#[derive(Clone, Default)]
struct ServingMetricsInner {
    running: u64,
    prompt_tokens: u64,
    generation_tokens: u64,
    prefix_cache_query_tokens: u64,
    prefix_cache_hit_tokens: u64,
    outcomes_stop: u64,
    outcomes_length: u64,
    outcomes_tool_calls: u64,
    outcomes_error: u64,
    outcomes_abort: u64,
    active_kv_tokens: usize,
    ttft: PromHistogram,
    tpot: PromHistogram,
    e2e: PromHistogram,
    session_enabled: bool,
    session_budget_bytes: usize,
    session_resident_bytes: usize,
    session_resident_snapshots: usize,
    session_active_gpu_tokens: usize,
    session_hits: u64,
    session_misses: u64,
    session_evictions: u64,
    session_save_count: u64,
    session_save_bytes: u64,
    session_save_seconds: f64,
    session_restore_count: u64,
    session_restore_bytes: u64,
    session_restore_seconds: f64,
}

struct ServingMetrics {
    max_context_tokens: usize,
    inner: Mutex<ServingMetricsInner>,
}

impl ServingMetrics {
    fn new(max_context_tokens: usize) -> Self {
        Self {
            max_context_tokens,
            inner: Mutex::new(ServingMetricsInner::default()),
        }
    }

    fn with_inner(&self, update: impl FnOnce(&mut ServingMetricsInner)) {
        match self.inner.lock() {
            Ok(mut inner) => update(&mut inner),
            Err(error) => log::error!("Serving metrics lock poisoned: {error}"),
        }
    }

    fn request_started(&self) {
        self.with_inner(|inner| inner.running = inner.running.saturating_add(1));
    }

    fn set_active_kv_tokens(&self, tokens: usize) {
        self.with_inner(|inner| inner.active_kv_tokens = tokens);
    }

    fn observe_ttft(&self, seconds: f64) {
        self.with_inner(|inner| inner.ttft.observe(seconds));
    }

    fn token_generated(&self, interval_seconds: f64, active_kv_tokens: usize) {
        self.with_inner(|inner| {
            inner.tpot.observe(interval_seconds);
            inner.active_kv_tokens = active_kv_tokens;
        });
    }

    fn request_finished(
        &self,
        elapsed_seconds: f64,
        prompt_tokens: usize,
        generated_tokens: usize,
        reused_tokens: usize,
        outcome: &str,
    ) {
        self.with_inner(|inner| {
            inner.running = inner.running.saturating_sub(1);
            inner.active_kv_tokens = 0;
            inner.prompt_tokens = inner.prompt_tokens.saturating_add(prompt_tokens as u64);
            inner.generation_tokens = inner
                .generation_tokens
                .saturating_add(generated_tokens as u64);
            inner.prefix_cache_query_tokens = inner
                .prefix_cache_query_tokens
                .saturating_add(prompt_tokens as u64);
            inner.prefix_cache_hit_tokens = inner
                .prefix_cache_hit_tokens
                .saturating_add(reused_tokens.min(prompt_tokens) as u64);
            inner.e2e.observe(elapsed_seconds);
            let counter = match outcome {
                "stop" => &mut inner.outcomes_stop,
                "length" => &mut inner.outcomes_length,
                "tool_calls" => &mut inner.outcomes_tool_calls,
                "abort" => &mut inner.outcomes_abort,
                _ => &mut inner.outcomes_error,
            };
            *counter = counter.saturating_add(1);
        });
    }

    fn render(&self, waiting: usize) -> String {
        let inner = match self.inner.lock() {
            Ok(inner) => inner.clone(),
            Err(error) => return format!("# serving metrics unavailable: {error}\n"),
        };
        let kv_fraction = if self.max_context_tokens > 0 {
            inner.active_kv_tokens as f64 / self.max_context_tokens as f64
        } else {
            0.0
        };
        let mut output = String::new();
        output.push_str("# HELP vllm:num_requests_running Chat requests executing on the model worker.\n# TYPE vllm:num_requests_running gauge\n");
        output.push_str(&format!("vllm:num_requests_running {}\n", inner.running));
        output.push_str("# HELP vllm:num_requests_waiting Requests queued for the model worker.\n# TYPE vllm:num_requests_waiting gauge\n");
        output.push_str(&format!("vllm:num_requests_waiting {waiting}\n"));
        output.push_str("# HELP vllm:prompt_tokens_total Prompt tokens observed for terminal chat requests.\n# TYPE vllm:prompt_tokens_total counter\n");
        output.push_str(&format!(
            "vllm:prompt_tokens_total {}\n",
            inner.prompt_tokens
        ));
        output.push_str("# HELP vllm:generation_tokens_total Generated tokens observed for terminal chat requests.\n# TYPE vllm:generation_tokens_total counter\n");
        output.push_str(&format!(
            "vllm:generation_tokens_total {}\n",
            inner.generation_tokens
        ));
        output.push_str("# HELP vllm:prefix_cache_queries_total Prompt tokens checked for exact prefix reuse.\n# TYPE vllm:prefix_cache_queries_total counter\n");
        output.push_str(&format!(
            "vllm:prefix_cache_queries_total {}\n",
            inner.prefix_cache_query_tokens
        ));
        output.push_str("# HELP vllm:prefix_cache_hits_total Prompt tokens reused from an exact prefix.\n# TYPE vllm:prefix_cache_hits_total counter\n");
        output.push_str(&format!(
            "vllm:prefix_cache_hits_total {}\n",
            inner.prefix_cache_hit_tokens
        ));
        output.push_str("# HELP vllm:kv_cache_usage_perc Fraction of the configured logical context occupied by the active sequence.\n# TYPE vllm:kv_cache_usage_perc gauge\n");
        output.push_str(&format!("vllm:kv_cache_usage_perc {kv_fraction}\n"));
        output.push_str(&format!(
            "krasis_active_kv_tokens {}\nkrasis_max_context_tokens {}\n",
            inner.active_kv_tokens, self.max_context_tokens
        ));
        output.push_str("# TYPE vllm:time_to_first_token_seconds histogram\n");
        inner
            .ttft
            .render(&mut output, "vllm:time_to_first_token_seconds");
        output.push_str("# TYPE vllm:inter_token_latency_seconds histogram\n");
        inner
            .tpot
            .render(&mut output, "vllm:inter_token_latency_seconds");
        output.push_str("# TYPE vllm:e2e_request_latency_seconds histogram\n");
        inner
            .e2e
            .render(&mut output, "vllm:e2e_request_latency_seconds");
        output.push_str("# HELP vllm:request_success_total Completed chat requests by terminal outcome.\n# TYPE vllm:request_success_total counter\n");
        for (reason, value) in [
            ("stop", inner.outcomes_stop),
            ("length", inner.outcomes_length),
            ("tool_calls", inner.outcomes_tool_calls),
            ("error", inner.outcomes_error),
            ("abort", inner.outcomes_abort),
        ] {
            output.push_str(&format!(
                "vllm:request_success_total{{finished_reason=\"{reason}\"}} {value}\n"
            ));
        }
        output.push_str(&format!(
            "krasis_session_cache_enabled {}\nkrasis_session_cache_budget_bytes {}\nkrasis_session_cache_resident_bytes {}\nkrasis_session_cache_resident_snapshots {}\nkrasis_session_cache_active_gpu_tokens {}\nkrasis_session_cache_hits_total {}\nkrasis_session_cache_misses_total {}\nkrasis_session_cache_evictions_total {}\nkrasis_session_cache_save_count_total {}\nkrasis_session_cache_save_bytes_total {}\nkrasis_session_cache_save_seconds_total {}\nkrasis_session_cache_restore_count_total {}\nkrasis_session_cache_restore_bytes_total {}\nkrasis_session_cache_restore_seconds_total {}\n",
            usize::from(inner.session_enabled),
            inner.session_budget_bytes,
            inner.session_resident_bytes,
            inner.session_resident_snapshots,
            inner.session_active_gpu_tokens,
            inner.session_hits,
            inner.session_misses,
            inner.session_evictions,
            inner.session_save_count,
            inner.session_save_bytes,
            inner.session_save_seconds,
            inner.session_restore_count,
            inner.session_restore_bytes,
            inner.session_restore_seconds,
        ));
        output
    }
}

struct ServingRequestGuard {
    metrics: Arc<ServingMetrics>,
    started_at: Instant,
    finished: bool,
}

impl ServingRequestGuard {
    fn new(metrics: Arc<ServingMetrics>, started_at: Instant) -> Self {
        metrics.request_started();
        Self {
            metrics,
            started_at,
            finished: false,
        }
    }

    fn finish(
        &mut self,
        prompt_tokens: usize,
        generated_tokens: usize,
        reused_tokens: usize,
        outcome: &str,
    ) {
        self.metrics.request_finished(
            self.started_at.elapsed().as_secs_f64(),
            prompt_tokens,
            generated_tokens,
            reused_tokens,
            outcome,
        );
        self.finished = true;
    }
}

impl Drop for ServingRequestGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.request_finished(
                self.started_at.elapsed().as_secs_f64(),
                0,
                0,
                0,
                "error",
            );
        }
    }
}

fn abort_if_cuda_context_poisoned(context: &str, err: &str) {
    if err.contains("CUDA_ERROR_ILLEGAL_ADDRESS")
        || err.to_ascii_lowercase().contains("illegal address")
    {
        crate::vram_monitor::fatal_cuda_context_error(context, err);
    }
}

fn context_window_fits(
    prompt_tokens: usize,
    max_output_tokens: usize,
    max_context_tokens: usize,
) -> bool {
    prompt_tokens < max_context_tokens && max_output_tokens <= max_context_tokens - prompt_tokens
}

/// Global pointer to the server's `running` flag so the raw signal handler
/// can set it to `false` without going through Python's signal mechanism.
/// This is only written once (before the accept loop) and read from the
/// signal handler, so the raw pointer is safe in practice.
#[cfg(unix)]
static SIGINT_RUNNING: AtomicBool = AtomicBool::new(false);
#[cfg(unix)]
static SIGNAL_FLAG_PTR: std::sync::atomic::AtomicPtr<AtomicBool> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Raw signal handler for SIGINT and SIGTERM.  Sets the server's `running`
/// flag to false so the accept loop exits cleanly, even when the GIL is
/// released (Python signal handlers can't run during allow_threads).
#[cfg(unix)]
extern "C" fn shutdown_signal_handler(_sig: libc::c_int) {
    let ptr = SIGNAL_FLAG_PTR.load(Ordering::Acquire);
    if !ptr.is_null() {
        // Safety: ptr points to the Arc<AtomicBool>'s inner value,
        // which outlives this handler (server.run() is still on the stack).
        unsafe { &*ptr }.store(false, Ordering::Release);
    }
    // Also set our own flag so we can detect it was us
    SIGINT_RUNNING.store(true, Ordering::Release);
}

/// Server state shared across request handling.
struct ServerState {
    py_model: Py<PyAny>,
    model_name: String,
    tokenizer: tokenizers::Tokenizer,
    chat_template: crate::chat_template::ChatTemplateEngine,
    max_context_tokens: usize,
    default_enable_thinking: bool,
    /// Token ID for `</think>` — when set, thinking tokens are exempt from max_tokens.
    thinking_end_token: Option<usize>,
    /// Raw pointer to a GpuDecodeStore instance (set from Python during server init).
    /// Safety: the fair model worker serializes all requests that can touch
    /// the shared GPU workspace, so no concurrent raw-pointer access occurs.
    gpu_store_addr: usize,
    /// When set, write full request JSON to this directory for debugging IDE clients.
    /// Enabled by KRASIS_LOG_REQUESTS=1 (writes to logs/requests/).
    log_requests_dir: Option<String>,
    /// Multi-GPU: auxiliary store addresses (empty = single GPU mode).
    aux_gpu_store_addrs: Vec<usize>,
    /// Multi-GPU: layer indices where each segment boundary falls.
    multi_gpu_split_layers: Vec<usize>,
    /// Multi-GPU: number of GQA layers before each split point (for KV cache indexing).
    multi_gpu_gqa_offsets: Vec<usize>,
    /// Shared Rust prefill engine — Arc+Mutex shared with benchmark path.
    /// When engine is available inside the Mutex, prefill runs entirely in Rust.
    rust_prefill: Arc<std::sync::Mutex<Option<crate::gpu_prefill::PrefillEngine>>>,
    /// Model's EOS token IDs (from generation_config.json).
    /// These are always included in stop_ids for decode, matching the main branch behavior.
    eos_stop_ids: Vec<usize>,
    /// Monotonic order for /v1/internal/reference_test requests.
    reference_test_request_order: u64,
    /// Rust-owned active GPU sequence boundary. Inactive RAM snapshots are
    /// added by the next phase; this entry is the zero-transfer fast path.
    session_cache: SessionCacheRuntime,
    serving_metrics: Arc<ServingMetrics>,
}

#[derive(Default)]
struct SessionCacheRuntime {
    enabled: bool,
    ram_fraction: f64,
    active: Option<ActiveSequenceState>,
    compatibility: Option<crate::session_cache::SessionCompatibilitySignature>,
    ram_store: Option<crate::session_cache::RamSessionStore>,
    prefill_samples: Vec<(usize, f64)>,
    restore_samples: Vec<(usize, f64)>,
    session_locks: Arc<SessionLockTable>,
    metrics: SessionCacheMetrics,
}

#[derive(Clone, Copy, Debug)]
enum SessionCacheMissReason {
    NoMatch,
    SignatureMismatch,
    Evicted,
    RestoreNotWorthIt,
    Divergence,
    RequestDisabled,
    ImageInput,
    MultiGpuPending,
    SpeculativeDecode,
    NoSuffix,
    RestoreFailed,
}

#[derive(Default)]
struct SessionCacheMetrics {
    active_hits: u64,
    ram_hits: u64,
    no_match_misses: u64,
    signature_mismatch_misses: u64,
    evicted_misses: u64,
    restore_not_worth_it_misses: u64,
    divergence_misses: u64,
    request_disabled_misses: u64,
    image_input_misses: u64,
    multi_gpu_pending_misses: u64,
    speculative_decode_misses: u64,
    no_suffix_misses: u64,
    restore_failed_misses: u64,
    save_count: u64,
    save_bytes: u64,
    save_total_ms: f64,
    save_last_ms: f64,
    restore_count: u64,
    restore_bytes: u64,
    restore_total_ms: f64,
    restore_last_ms: f64,
    mid_prefill_boundary_skipped: u64,
}

fn increment_metric(counter: &mut u64, name: &str) {
    if let Some(next) = counter.checked_add(1) {
        *counter = next;
    } else {
        log::error!("Session-cache metric {} exhausted u64", name);
    }
}

impl SessionCacheMetrics {
    fn record_hit(&mut self, ram: bool) {
        if ram {
            increment_metric(&mut self.ram_hits, "ram_hits");
        } else {
            increment_metric(&mut self.active_hits, "active_hits");
        }
    }

    fn record_miss(&mut self, reason: SessionCacheMissReason) {
        let (counter, name) = match reason {
            SessionCacheMissReason::NoMatch => (&mut self.no_match_misses, "no_match_misses"),
            SessionCacheMissReason::SignatureMismatch => (
                &mut self.signature_mismatch_misses,
                "signature_mismatch_misses",
            ),
            SessionCacheMissReason::Evicted => (&mut self.evicted_misses, "evicted_misses"),
            SessionCacheMissReason::RestoreNotWorthIt => (
                &mut self.restore_not_worth_it_misses,
                "restore_not_worth_it_misses",
            ),
            SessionCacheMissReason::Divergence => {
                (&mut self.divergence_misses, "divergence_misses")
            }
            SessionCacheMissReason::RequestDisabled => {
                (&mut self.request_disabled_misses, "request_disabled_misses")
            }
            SessionCacheMissReason::ImageInput => {
                (&mut self.image_input_misses, "image_input_misses")
            }
            SessionCacheMissReason::MultiGpuPending => (
                &mut self.multi_gpu_pending_misses,
                "multi_gpu_pending_misses",
            ),
            SessionCacheMissReason::SpeculativeDecode => (
                &mut self.speculative_decode_misses,
                "speculative_decode_misses",
            ),
            SessionCacheMissReason::NoSuffix => (&mut self.no_suffix_misses, "no_suffix_misses"),
            SessionCacheMissReason::RestoreFailed => {
                (&mut self.restore_failed_misses, "restore_failed_misses")
            }
        };
        increment_metric(counter, name);
    }

    fn record_save(&mut self, bytes: usize, elapsed_ms: f64) {
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            log::error!(
                "Refusing non-finite or negative session-cache save timing: {} ms",
                elapsed_ms
            );
            return;
        }
        increment_metric(&mut self.save_count, "save_count");
        match u64::try_from(bytes)
            .ok()
            .and_then(|bytes| self.save_bytes.checked_add(bytes))
        {
            Some(total) => self.save_bytes = total,
            None => log::error!("Session-cache save byte metric overflow"),
        }
        self.save_total_ms += elapsed_ms;
        self.save_last_ms = elapsed_ms;
    }

    fn record_restore(&mut self, bytes: usize, elapsed_ms: f64) {
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            log::error!(
                "Refusing non-finite or negative session-cache restore timing: {} ms",
                elapsed_ms
            );
            return;
        }
        increment_metric(&mut self.restore_count, "restore_count");
        match u64::try_from(bytes)
            .ok()
            .and_then(|bytes| self.restore_bytes.checked_add(bytes))
        {
            Some(total) => self.restore_bytes = total,
            None => log::error!("Session-cache restore byte metric overflow"),
        }
        self.restore_total_ms += elapsed_ms;
        self.restore_last_ms = elapsed_ms;
    }
}

struct ActiveSequenceState {
    consumed_token_ids: Vec<u32>,
    snapshot_id: crate::session_cache::SnapshotId,
    requires_device_checkpoint: bool,
}

struct PendingBoundarySnapshot {
    reservation: crate::session_cache::RamReservationId,
    consumed_token_ids: Vec<u32>,
    capture: crate::gpu_prefill::PrefillSequenceBoundaryCapture,
}

fn invalidate_active_sequence(state: &mut ServerState, reason: &str) {
    if state.session_cache.active.take().is_some() {
        log::info!("Prefix cache active GPU state invalidated: {}", reason);
    }
}

fn validate_prefix_cache_ram_fraction(fraction: f64) -> Result<f64, String> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return Err(format!(
            "prefix_cache_ram_fraction must be finite and in (0, 1], got {fraction}"
        ));
    }
    Ok(fraction)
}

fn session_cache_multi_gpu_pending(
    prefix_cache_enabled: bool,
    aux_gpu_store_addrs: &[usize],
) -> bool {
    prefix_cache_enabled && !aux_gpu_store_addrs.is_empty()
}

fn session_cache_runtime_materialization_enabled(
    prefix_cache_enabled: bool,
    aux_gpu_store_addrs: &[usize],
) -> bool {
    prefix_cache_enabled
        && !session_cache_multi_gpu_pending(prefix_cache_enabled, aux_gpu_store_addrs)
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn build_session_compatibility_signature(
    model_name: &str,
    tokenizer_path: &str,
    chat_template: &crate::chat_template::ChatTemplateEngine,
    gpu_store_addr: usize,
    aux_gpu_store_addrs: &[usize],
) -> Result<crate::session_cache::SessionCompatibilitySignature, String> {
    if gpu_store_addr == 0 {
        return Err("session cache requires a configured GPU decode store".to_string());
    }
    let tokenizer_bytes = std::fs::read(tokenizer_path)
        .map_err(|error| format!("read tokenizer for session signature: {error}"))?;
    let model_dir = std::path::Path::new(tokenizer_path)
        .parent()
        .ok_or_else(|| "tokenizer path has no model directory".to_string())?;
    let config_path = model_dir.join("config.json");
    let config_bytes = std::fs::read(&config_path).map_err(|error| {
        format!(
            "read model config {} for session signature: {error}",
            config_path.display()
        )
    })?;

    let mut materials = Vec::with_capacity(aux_gpu_store_addrs.len() + 1);
    let primary = unsafe { &*(gpu_store_addr as *const GpuDecodeStore) };
    materials.push(primary.session_compatibility_material_rust()?);
    for &address in aux_gpu_store_addrs {
        if address == 0 {
            return Err(
                "session cache topology contains a zero auxiliary store address".to_string(),
            );
        }
        let store = unsafe { &*(address as *const GpuDecodeStore) };
        materials.push(store.session_compatibility_material_rust()?);
    }
    let first = materials
        .first()
        .ok_or_else(|| "session cache has no compatibility material".to_string())?;
    for material in materials.iter().skip(1) {
        if material.model_num_layers != first.model_num_layers
            || material.expert_quantization != first.expert_quantization
            || material.attention_quantization != first.attention_quantization
            || material.kv_format != first.kv_format
            || material.kv_key_bits != first.kv_key_bits
            || material.kv_value_bits != first.kv_value_bits
        {
            return Err(format!(
                "session cache runtime configuration disagrees between GPU {} and GPU {}",
                first.ownership.device_ordinal, material.ownership.device_ordinal
            ));
        }
    }
    let topology = materials
        .iter()
        .map(|material| material.ownership.clone())
        .collect();
    let state_layout = serde_json::to_vec(
        &materials
            .iter()
            .map(|material| &material.state_layout)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| format!("serialize session state layout: {error}"))?;
    let model_config_sha256 = sha256_bytes(&config_bytes);
    let mut signature = crate::session_cache::SessionCompatibilitySignature {
        snapshot_format_version: crate::session_cache::SESSION_SNAPSHOT_FORMAT_VERSION,
        runtime_version: env!("CARGO_PKG_VERSION").to_string(),
        model_identity: format!("{}:{:02x?}", model_name, model_config_sha256),
        model_revision: None,
        tokenizer_sha256: sha256_bytes(&tokenizer_bytes),
        chat_template_sha256: sha256_bytes(chat_template.compatibility_source().as_bytes()),
        expert_quantization: first.expert_quantization.clone(),
        attention_quantization: first.attention_quantization.clone(),
        kv_format: first.kv_format.clone(),
        kv_key_bits: first.kv_key_bits,
        kv_value_bits: first.kv_value_bits,
        model_num_layers: first.model_num_layers,
        topology,
        state_layout_sha256: sha256_bytes(&state_layout),
    };
    signature.topology.sort_by_key(|owner| owner.layer_start);
    signature.validate()?;
    Ok(signature)
}

fn session_snapshot_reservation_bytes(
    state: &ServerState,
    consumed_token_count: usize,
) -> Result<usize, String> {
    let compatibility = state
        .session_cache
        .compatibility
        .as_ref()
        .ok_or_else(|| "session cache compatibility signature is unavailable".to_string())?;
    let mut bytes = std::mem::size_of::<crate::session_cache::SessionSnapshot>()
        .checked_add(compatibility.heap_bytes())
        .and_then(|value| {
            value.checked_add(consumed_token_count.saturating_mul(std::mem::size_of::<u32>()))
        })
        .and_then(|value| {
            value.checked_add(
                compatibility
                    .topology
                    .len()
                    .saturating_mul(std::mem::size_of::<
                        crate::session_cache::DeviceSequencePosition,
                    >()),
            )
        })
        .ok_or_else(|| "session snapshot reservation estimate overflow".to_string())?;
    for address in
        std::iter::once(state.gpu_store_addr).chain(state.aux_gpu_store_addrs.iter().copied())
    {
        let store = unsafe { &*(address as *const GpuDecodeStore) };
        bytes = bytes
            .checked_add(store.sequence_state_snapshot_cost_estimate_rust(consumed_token_count)?)
            .ok_or_else(|| "session snapshot reservation estimate overflow".to_string())?;
    }
    Ok(bytes)
}

fn cancel_boundary_reservation(
    session_cache: &mut SessionCacheRuntime,
    reservation: &mut Option<crate::session_cache::RamReservationId>,
) {
    let Some(reservation_id) = reservation.take() else {
        return;
    };
    if let Some(store) = session_cache.ram_store.as_mut() {
        if let Err(error) = store.cancel_reservation(reservation_id) {
            log::error!(
                "Failed to cancel prefix-cache boundary reservation {:?}: {}",
                reservation_id,
                error,
            );
        }
    }
}

fn rollback_pending_boundary_snapshot(
    state: &mut ServerState,
    pending: &mut Option<PendingBoundarySnapshot>,
    reservation: &mut Option<crate::session_cache::RamReservationId>,
) {
    if let Some(pending) = pending.take() {
        *reservation = Some(pending.reservation);
    }
    cancel_boundary_reservation(&mut state.session_cache, reservation);
}

fn commit_pending_boundary_snapshot(
    state: &mut ServerState,
    pending: PendingBoundarySnapshot,
) -> Result<(crate::session_cache::SnapshotId, usize, f64, f64), String> {
    let reservation = pending.reservation;
    let save_ms = pending.capture.save_ms;
    let restore_ms = pending.capture.restore_ms;
    let result = (|| {
        let snapshot = crate::session_cache::SessionSnapshot {
            compatibility: state
                .session_cache
                .compatibility
                .as_ref()
                .ok_or_else(|| "session cache compatibility signature is unavailable".to_string())?
                .clone(),
            consumed_token_ids: pending.consumed_token_ids,
            positions: pending.capture.positions,
            state_blobs: pending.capture.state_blobs,
        };
        snapshot.validate()?;
        let actual_bytes = snapshot.memory_cost_bytes();
        let snapshot_id = state
            .session_cache
            .ram_store
            .as_mut()
            .ok_or_else(|| "RAM session store is unavailable".to_string())?
            .commit(reservation, snapshot)?;
        state
            .session_cache
            .restore_samples
            .push((actual_bytes, restore_ms));
        Ok((snapshot_id, actual_bytes, save_ms, restore_ms))
    })();
    if result.is_err() {
        if let Some(store) = state.session_cache.ram_store.as_mut() {
            let _ = store.cancel_reservation(reservation);
        }
    }
    result
}

fn active_boundary_tokens_for_publication(
    stage_required: bool,
    pending_tokens: Option<&[u32]>,
    stable_tokens: Option<&[u32]>,
    sequence_start: usize,
    has_base_snapshot: bool,
) -> Option<Vec<u32>> {
    if !stage_required {
        return None;
    }
    if let Some(tokens) = pending_tokens {
        return Some(tokens.to_vec());
    }
    stable_tokens
        .filter(|tokens| has_base_snapshot && tokens.len() == sequence_start)
        .map(<[u32]>::to_vec)
}

fn active_plan_requires_device_checkpoint(
    plan: crate::session_cache::ActivePrefixPlan,
    published_requires_device_checkpoint: bool,
) -> bool {
    published_requires_device_checkpoint
        && matches!(plan, crate::session_cache::ActivePrefixPlan::Append { .. })
}

fn active_plan_requires_stage_restore(
    plan: crate::session_cache::ActivePrefixPlan,
    published_requires_device_checkpoint: bool,
) -> bool {
    published_requires_device_checkpoint
        && matches!(
            plan,
            crate::session_cache::ActivePrefixPlan::Append { .. }
                | crate::session_cache::ActivePrefixPlan::TruncateKvAndAppend { .. }
        )
}

fn internal_capture_boundary(
    snapshot_boundary: Option<usize>,
    sequence_start: usize,
    request_tokens: usize,
) -> Option<usize> {
    snapshot_boundary.filter(|&boundary| boundary > sequence_start && boundary < request_tokens)
}

fn restore_snapshot_to_gpu(
    gpu_store_addr: usize,
    aux_gpu_store_addrs: &[usize],
    snapshot: &crate::session_cache::SessionSnapshot,
) -> Result<f64, String> {
    let mut total_ms = 0.0f64;
    for address in std::iter::once(gpu_store_addr).chain(aux_gpu_store_addrs.iter().copied()) {
        let store = unsafe { &mut *(address as *mut GpuDecodeStore) };
        let device_ordinal = usize::try_from(store.device_ordinal()).map_err(|_| {
            format!(
                "CUDA device ordinal is negative: {}",
                store.device_ordinal()
            )
        })?;
        let position = snapshot
            .positions
            .iter()
            .find(|position| position.device_ordinal == device_ordinal)
            .ok_or_else(|| {
                format!(
                    "snapshot has no position for CUDA device {}",
                    store.device_ordinal()
                )
            })?;
        let blobs: Vec<_> = snapshot
            .state_blobs
            .iter()
            .filter(|blob| {
                blob.device_ordinal == device_ordinal
                    && !crate::session_cache::is_prefill_stage_kind(&blob.kind)
            })
            .collect();
        total_ms += store.restore_sequence_state_rust(
            snapshot.consumed_token_ids.len(),
            &blobs,
            position.rope_position_delta,
        )?;
    }
    Ok(total_ms)
}

fn snapshot_current_sequence_to_ram(
    state: &mut ServerState,
    consumed_token_ids: &[u32],
    previous_snapshot_id: Option<crate::session_cache::SnapshotId>,
) -> Result<(crate::session_cache::SnapshotId, usize, f64, f64), String> {
    let previous_token_count = if let Some(id) = previous_snapshot_id {
        let previous = state
            .session_cache
            .ram_store
            .as_mut()
            .ok_or_else(|| "RAM session store is unavailable".to_string())?
            .get(id)?
            .ok_or_else(|| "incremental snapshot base was evicted".to_string())?;
        if previous.consumed_token_ids.len() > consumed_token_ids.len()
            || previous.consumed_token_ids.as_slice()
                != &consumed_token_ids[..previous.consumed_token_ids.len()]
        {
            return Err("incremental snapshot base is not an exact token prefix".to_string());
        }
        Some(previous.consumed_token_ids.len())
    } else {
        None
    };
    let required_bytes = session_snapshot_reservation_bytes(state, consumed_token_ids.len())?;
    let reservation = state
        .session_cache
        .ram_store
        .as_mut()
        .ok_or_else(|| "RAM session store is unavailable".to_string())?
        .reserve_protecting(
            required_bytes,
            previous_snapshot_id
                .as_ref()
                .map_or(&[], std::slice::from_ref),
        )?;
    let result = (|| {
        let addresses: Vec<_> = std::iter::once(state.gpu_store_addr)
            .chain(state.aux_gpu_store_addrs.iter().copied())
            .collect();
        let allocation_count = addresses.iter().try_fold(0usize, |count, &address| {
            let store = unsafe { &*(address as *const GpuDecodeStore) };
            count
                .checked_add(store.sequence_state_allocation_count_rust())
                .ok_or_else(|| "session snapshot allocation count overflow".to_string())
        })?;
        let mut positions = Vec::with_capacity(addresses.len());
        let mut state_blobs = Vec::with_capacity(allocation_count);
        let mut save_ms = 0.0f64;
        for address in addresses {
            let store = unsafe { &mut *(address as *mut GpuDecodeStore) };
            let device_ordinal = usize::try_from(store.device_ordinal()).map_err(|_| {
                format!(
                    "CUDA device ordinal is negative: {}",
                    store.device_ordinal()
                )
            })?;
            store.set_kv_position_rust(consumed_token_ids.len());
            positions.push(store.sequence_position_rust()?);
            let (mut blobs, device_save_ms) = if let (Some(id), Some(previous_tokens)) =
                (previous_snapshot_id, previous_token_count)
            {
                let previous = state
                    .session_cache
                    .ram_store
                    .as_mut()
                    .ok_or_else(|| "RAM session store is unavailable".to_string())?
                    .get(id)?
                    .ok_or_else(|| "incremental snapshot base was evicted".to_string())?;
                let previous_blobs: Vec<_> = previous
                    .state_blobs
                    .iter()
                    .filter(|blob| blob.device_ordinal == device_ordinal)
                    .collect();
                store.snapshot_sequence_state_incremental_rust(
                    previous_tokens,
                    consumed_token_ids.len(),
                    &previous_blobs,
                )?
            } else {
                store.snapshot_sequence_state_rust(consumed_token_ids.len())?
            };
            save_ms += device_save_ms;
            state_blobs.append(&mut blobs);
        }
        let snapshot = crate::session_cache::SessionSnapshot {
            compatibility: state
                .session_cache
                .compatibility
                .as_ref()
                .ok_or_else(|| "session cache compatibility signature is unavailable".to_string())?
                .clone(),
            consumed_token_ids: consumed_token_ids.to_vec(),
            positions,
            state_blobs,
        };
        snapshot.validate()?;
        let actual_bytes = snapshot.memory_cost_bytes();
        let id = state
            .session_cache
            .ram_store
            .as_mut()
            .ok_or_else(|| "RAM session store is unavailable".to_string())?
            .commit(reservation, snapshot)?;
        Ok((id, actual_bytes, save_ms))
    })();
    let (id, actual_bytes, save_ms) = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = state
                .session_cache
                .ram_store
                .as_mut()
                .ok_or_else(|| "RAM session store is unavailable".to_string())?
                .cancel_reservation(reservation);
            return Err(error);
        }
    };

    // The first restore of each real layout is also the runtime H2D
    // calibration point. It restores identical bytes to the just-snapshotted
    // state and therefore cannot change model outputs.
    let restore_ms = {
        let snapshot = state
            .session_cache
            .ram_store
            .as_mut()
            .ok_or_else(|| "RAM session store is unavailable".to_string())?
            .get(id)?
            .ok_or_else(|| "newly committed session snapshot disappeared".to_string())?;
        restore_snapshot_to_gpu(
            state.gpu_store_addr,
            &state.aux_gpu_store_addrs,
            snapshot.as_ref(),
        )?
    };
    state
        .session_cache
        .restore_samples
        .push((actual_bytes, restore_ms));
    Ok((id, actual_bytes, save_ms, restore_ms))
}

fn restore_snapshot_by_id(
    state: &mut ServerState,
    id: crate::session_cache::SnapshotId,
) -> Result<(Vec<u32>, usize, f64), String> {
    let (tokens, bytes, restore_ms) = {
        let snapshot = state
            .session_cache
            .ram_store
            .as_mut()
            .ok_or_else(|| "RAM session store is unavailable".to_string())?
            .get(id)?
            .ok_or_else(|| "selected RAM session snapshot was evicted".to_string())?;
        let tokens = snapshot.consumed_token_ids.clone();
        let bytes = snapshot.memory_cost_bytes();
        let restore_ms = restore_snapshot_to_gpu(
            state.gpu_store_addr,
            &state.aux_gpu_store_addrs,
            snapshot.as_ref(),
        )?;
        (tokens, bytes, restore_ms)
    };
    Ok((tokens, bytes, restore_ms))
}

fn predicted_restore_ms(samples: &[(usize, f64)], bytes: usize) -> Option<f64> {
    samples
        .iter()
        .filter(|(sample_bytes, sample_ms)| *sample_bytes > 0 && *sample_ms > 0.0)
        .min_by_key(|(sample_bytes, _)| sample_bytes.abs_diff(bytes))
        .map(|(sample_bytes, sample_ms)| *sample_ms * bytes as f64 / *sample_bytes as f64)
}

fn predicted_avoided_prefill_ms(
    samples: &[(usize, f64)],
    prompt_tokens: usize,
    reused_tokens: usize,
) -> Option<f64> {
    if prompt_tokens == 0 || reused_tokens == 0 {
        return None;
    }
    samples
        .iter()
        .filter(|(sample_tokens, sample_ms)| *sample_tokens > 0 && *sample_ms > 0.0)
        .min_by_key(|(sample_tokens, _)| sample_tokens.abs_diff(prompt_tokens))
        .map(|(sample_tokens, sample_ms)| {
            let full_ms = *sample_ms * prompt_tokens as f64 / *sample_tokens as f64;
            full_ms * reused_tokens as f64 / prompt_tokens as f64
        })
}

#[derive(Clone)]
struct ServerInfo {
    model_name: String,
    max_context_tokens: usize,
    supports_vision: bool,
    serving_metrics: Arc<ServingMetrics>,
}

fn drain_vram_pressure_for_state(
    state: &mut ServerState,
    reason: &str,
    force_measure: bool,
) -> usize {
    let mut total_evicted = 0usize;
    if state.gpu_store_addr != 0 {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        let (evicted, freed_mb, final_free_mb) =
            store.hcs_drain_vram_pressure(reason, force_measure);
        if evicted > 0 {
            log::warn!(
                "VRAM pressure drain {} primary: evicted {} soft experts, freed {:.1} MB, final_free={} MB",
                reason,
                evicted,
                freed_mb,
                final_free_mb,
            );
            total_evicted += evicted;
        }
    }

    for (idx, &addr) in state.aux_gpu_store_addrs.iter().enumerate() {
        if addr == 0 {
            continue;
        }
        let aux_reason = format!("{}_aux{}", reason, idx + 1);
        let store = unsafe { &mut *(addr as *mut GpuDecodeStore) };
        let (evicted, freed_mb, final_free_mb) =
            store.hcs_drain_vram_pressure(&aux_reason, force_measure);
        if evicted > 0 {
            log::warn!(
                "VRAM pressure drain {}: evicted {} soft experts, freed {:.1} MB, final_free={} MB",
                aux_reason,
                evicted,
                freed_mb,
                final_free_mb,
            );
            total_evicted += evicted;
        }
    }

    total_evicted
}

enum ModelRequest {
    Chat {
        stream: TcpStream,
        body: String,
        received_at: Instant,
    },
    SessionCacheStats {
        stream: TcpStream,
    },
    PrefillLogits {
        stream: TcpStream,
        body: String,
    },
    TeacherForcedDecodeLogits {
        stream: TcpStream,
        body: String,
    },
    ReferenceTest {
        stream: TcpStream,
        body: String,
    },
    SequenceStateInventory {
        stream: TcpStream,
        body: String,
    },
    SequenceStateTransferMeasurement {
        stream: TcpStream,
        body: String,
    },
}

struct QueuedModelRequest {
    ticket: u64,
    enqueued_at: Instant,
    request: ModelRequest,
}

/// Admission is serialized only long enough to assign and send a ticket, so
/// the unbounded model-worker channel receives requests in exact ticket order
/// even when connection threads race. GPU execution remains single-workspace.
struct FairModelScheduler {
    sender: mpsc::Sender<QueuedModelRequest>,
    next_ticket: Mutex<u64>,
    queued: AtomicUsize,
}

impl FairModelScheduler {
    fn new(sender: mpsc::Sender<QueuedModelRequest>) -> Self {
        Self {
            sender,
            next_ticket: Mutex::new(0),
            queued: AtomicUsize::new(0),
        }
    }

    fn enqueue(&self, request: ModelRequest) -> Result<u64, String> {
        let mut next = self
            .next_ticket
            .lock()
            .map_err(|error| format!("model scheduler lock poisoned: {error}"))?;
        let ticket = *next;
        *next = next
            .checked_add(1)
            .ok_or_else(|| "model scheduler ticket counter exhausted".to_string())?;
        self.queued.fetch_add(1, Ordering::AcqRel);
        if self
            .sender
            .send(QueuedModelRequest {
                ticket,
                enqueued_at: Instant::now(),
                request,
            })
            .is_err()
        {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            return Err("model worker is unavailable".to_string());
        }
        Ok(ticket)
    }

    fn mark_dequeued(&self) -> usize {
        let previous = self.queued.fetch_sub(1, Ordering::AcqRel);
        if previous == 0 {
            self.queued.store(0, Ordering::Release);
            log::error!("model scheduler queue accounting underflow");
            0
        } else {
            previous - 1
        }
    }

    fn queued(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SessionLockKey {
    Snapshot(crate::session_cache::SnapshotId),
    ExactBoundary(Arc<[u32]>),
}

#[derive(Default)]
struct SessionLockTable {
    held: Mutex<std::collections::HashSet<SessionLockKey>>,
    changed: Condvar,
}

impl SessionLockTable {
    fn acquire(self: &Arc<Self>, key: SessionLockKey) -> Result<SessionRequestLease, String> {
        let mut held = self
            .held
            .lock()
            .map_err(|error| format!("session lock table poisoned: {error}"))?;
        while held.contains(&key) {
            held = self
                .changed
                .wait(held)
                .map_err(|error| format!("session lock wait poisoned: {error}"))?;
        }
        if !held.insert(key.clone()) {
            return Err("session lock insertion failed".to_string());
        }
        Ok(SessionRequestLease {
            table: Arc::clone(self),
            key: Some(key),
        })
    }
}

struct SessionRequestLease {
    table: Arc<SessionLockTable>,
    key: Option<SessionLockKey>,
}

impl Drop for SessionRequestLease {
    fn drop(&mut self) {
        let Some(key) = self.key.take() else {
            return;
        };
        match self.table.held.lock() {
            Ok(mut held) => {
                if !held.remove(&key) {
                    log::error!("session request lease disappeared before release");
                }
                self.table.changed.notify_all();
            }
            Err(error) => log::error!("session lock release poisoned: {error}"),
        }
    }
}

fn handle_model_request(request: ModelRequest, state: &mut ServerState) {
    match request {
        ModelRequest::Chat {
            mut stream,
            body,
            received_at,
        } => handle_chat_completion(&mut stream, &body, state, received_at),
        ModelRequest::SessionCacheStats { mut stream } => {
            handle_session_cache_stats(&mut stream, state)
        }
        ModelRequest::PrefillLogits { mut stream, body } => {
            handle_prefill_logits(&mut stream, &body, state)
        }
        ModelRequest::TeacherForcedDecodeLogits { mut stream, body } => {
            handle_teacher_forced_decode_logits(&mut stream, &body, state)
        }
        ModelRequest::ReferenceTest { mut stream, body } => {
            handle_reference_test(&mut stream, &body, state)
        }
        ModelRequest::SequenceStateInventory { mut stream, body } => {
            handle_sequence_state_inventory(&mut stream, &body, state)
        }
        ModelRequest::SequenceStateTransferMeasurement { mut stream, body } => {
            handle_sequence_state_transfer_measurement(&mut stream, &body, state)
        }
    }
}

fn reject_model_request(request: ModelRequest, message: &str) {
    let mut stream = match request {
        ModelRequest::Chat { stream, .. }
        | ModelRequest::SessionCacheStats { stream }
        | ModelRequest::PrefillLogits { stream, .. }
        | ModelRequest::TeacherForcedDecodeLogits { stream, .. }
        | ModelRequest::ReferenceTest { stream, .. }
        | ModelRequest::SequenceStateInventory { stream, .. }
        | ModelRequest::SequenceStateTransferMeasurement { stream, .. } => stream,
    };
    let _ = send_json(
        &mut stream,
        500,
        &format!(r#"{{"error":"{}"}}"#, json_escape(message)),
    );
}

struct VramRequestContextGuard {
    safety_margin_mb: u64,
}

impl Drop for VramRequestContextGuard {
    fn drop(&mut self) {
        if let Some((context, lows)) = crate::vram_monitor::end_request_context() {
            if lows.is_empty() {
                log::info!("Request VRAM low-water: {} lows=none", context);
            } else {
                crate::vram_monitor::record_request_lows_below_safety(
                    &context,
                    &lows,
                    self.safety_margin_mb,
                );
                let lows_text = lows
                    .iter()
                    .map(|(device, mb)| format!("cuda{}={}MB", device, mb))
                    .collect::<Vec<_>>()
                    .join(" ");
                log::info!("Request VRAM low-water: {} lows={}", context, lows_text);
            }
        }
    }
}

/// Parsed HTTP request.
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

fn prepare_store_for_rust_prefill(
    store: &mut GpuDecodeStore,
    engine: &mut crate::gpu_prefill::PrefillEngine,
    prompt_tokens: usize,
) -> Result<bool, String> {
    let has_hqq = store.prepare_runtime_for_prefill_rust(prompt_tokens)?;
    store.refresh_prefill_engine_kv_cache_rust(engine)?;
    if has_hqq {
        let patches = store.hqq_prefill_pointer_patches_rust()?;
        engine.refresh_hqq_prefill_tensor_pointers(&patches)?;
    }
    Ok(has_hqq)
}

fn prefill_entry_floor_bytes_for_server(
    rust_prefill: &Arc<std::sync::Mutex<Option<crate::gpu_prefill::PrefillEngine>>>,
    prompt_tokens: usize,
) -> Result<usize, String> {
    let guard = rust_prefill
        .lock()
        .map_err(|e| format!("Prefill engine lock poisoned: {}", e))?;
    Ok(guard
        .as_ref()
        .map(|engine| engine.minimum_prefill_entry_free_bytes(prompt_tokens))
        .unwrap_or(0))
}

fn create_prefill_engine_for_server(
    store: &mut GpuDecodeStore,
    max_context_tokens: usize,
) -> Result<crate::gpu_prefill::PrefillEngine, String> {
    let has_hqq = store.has_hqq_runtime_slots();
    if has_hqq {
        store.prepare_runtime_for_prefill_rust(max_context_tokens)?;
    }
    let engine = match store.create_prefill_engine(max_context_tokens) {
        Ok(engine) => engine,
        Err(e) => {
            if has_hqq {
                let _ = store.prepare_runtime_for_decode_rust();
            }
            return Err(e);
        }
    };
    if has_hqq {
        store.prepare_runtime_for_decode_rust()?;
    }
    Ok(engine)
}

fn restore_store_after_rust_prefill(
    store: &mut GpuDecodeStore,
    prompt_len: usize,
) -> Result<(), String> {
    store.prepare_dspark_context_after_prefill_rust(prompt_len)?;
    store.set_kv_position_rust(prompt_len);
    store.prepare_runtime_for_decode_rust()
}

/// Parse an HTTP request from a TCP stream.
fn parse_request(stream: &mut BufReader<TcpStream>) -> std::io::Result<HttpRequest> {
    // Request line
    let mut request_line = String::new();
    stream.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid request line",
        ));
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Headers
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        stream.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        } else if let Some(val) = trimmed.strip_prefix("content-length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
    }

    // Body
    let mut body = String::new();
    if content_length > 0 {
        let mut buf = vec![0u8; content_length];
        stream.read_exact(&mut buf)?;
        body = String::from_utf8_lossy(&buf).to_string();
    }

    Ok(HttpRequest { method, path, body })
}

/// Send a JSON response.
fn send_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        507 => "Insufficient Storage",
        _ => "Unknown",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        status_text,
        body.len(),
        body
    )?;
    stream.flush()
}

fn send_prometheus(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\n\
         Access-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    stream.flush()
}

fn is_models_endpoint(path: &str) -> bool {
    let path_no_query = path.split('?').next().unwrap_or(path);
    let normalized = path_no_query.trim_end_matches('/');
    normalized == "/v1/models" || normalized == "/models"
}

fn is_chat_completions_endpoint(path: &str) -> bool {
    let path_no_query = path.split('?').next().unwrap_or(path);
    let normalized = path_no_query.trim_end_matches('/');
    normalized == "/v1/chat/completions" || normalized == "/chat/completions"
}

fn format_models_response(
    model_name: &str,
    max_context_tokens: usize,
    supports_vision: bool,
) -> String {
    let model_id = if supports_vision {
        format!("{}-vision", model_name)
    } else {
        model_name.to_string()
    };
    let mut model = serde_json::json!({
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "krasis",
        "max_context_tokens": max_context_tokens,
        "capabilities": {
            "vision": supports_vision,
        },
    });
    if supports_vision {
        model["input_modalities"] = serde_json::json!(["text", "image"]);
    }
    serde_json::json!({
        "object": "list",
        "data": [model],
    })
    .to_string()
}

fn handle_front_connection(
    mut tcp_stream: TcpStream,
    server_info: ServerInfo,
    scheduler: Arc<FairModelScheduler>,
    test_endpoints: bool,
) {
    let cloned = match tcp_stream.try_clone() {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to clone TCP stream: {}", e);
            return;
        }
    };
    let mut reader = BufReader::new(cloned);

    let request = match parse_request(&mut reader) {
        Ok(r) => r,
        Err(e) => {
            if matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) {
                log::debug!("Ignoring incomplete HTTP request: {}", e);
                return;
            }
            log::error!("Failed to parse request: {}", e);
            let _ = send_json(&mut tcp_stream, 400, r#"{"error":"Bad request"}"#);
            return;
        }
    };

    if request.method == "OPTIONS" {
        let _ = write!(
            tcp_stream,
            "HTTP/1.1 204 No Content\r\n\
             Access-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
             Connection: close\r\n\r\n"
        );
        let _ = tcp_stream.flush();
        return;
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let body = format!(
                r#"{{"status":"ok","max_context_tokens":{}}}"#,
                server_info.max_context_tokens
            );
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("GET", "/metrics") => {
            let body = server_info.serving_metrics.render(scheduler.queued());
            let _ = send_prometheus(&mut tcp_stream, &body);
        }

        ("GET", path) if is_models_endpoint(path) => {
            let body = format_models_response(
                &server_info.model_name,
                server_info.max_context_tokens,
                server_info.supports_vision,
            );
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("GET", "/v1/session-cache/stats") => {
            if let Err(error) =
                scheduler.enqueue(ModelRequest::SessionCacheStats { stream: tcp_stream })
            {
                log::error!(
                    "Model worker is not available for session-cache stats: {}",
                    error
                );
            }
        }

        ("POST", path) if is_chat_completions_endpoint(path) => {
            if let Err(error) = scheduler.enqueue(ModelRequest::Chat {
                stream: tcp_stream,
                body: request.body,
                received_at: Instant::now(),
            }) {
                log::error!(
                    "Model worker is not available for /v1/chat/completions: {}",
                    error
                );
            }
        }

        ("POST", "/v1/internal/prefill_logits") => {
            if test_endpoints {
                if let Err(error) = scheduler.enqueue(ModelRequest::PrefillLogits {
                    stream: tcp_stream,
                    body: request.body,
                }) {
                    log::error!(
                        "Model worker is not available for /v1/internal/prefill_logits: {}",
                        error
                    );
                }
            } else {
                let _ = send_json(
                    &mut tcp_stream,
                    404,
                    r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#,
                );
            }
        }

        ("POST", "/v1/internal/teacher_forced_decode_logits") => {
            if test_endpoints {
                if let Err(error) = scheduler.enqueue(ModelRequest::TeacherForcedDecodeLogits {
                    stream: tcp_stream,
                    body: request.body,
                }) {
                    log::error!(
                        "Model worker is not available for /v1/internal/teacher_forced_decode_logits: {}",
                        error
                    );
                }
            } else {
                let _ = send_json(
                    &mut tcp_stream,
                    404,
                    r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#,
                );
            }
        }

        ("POST", "/v1/internal/reference_test") => {
            if test_endpoints {
                if let Err(error) = scheduler.enqueue(ModelRequest::ReferenceTest {
                    stream: tcp_stream,
                    body: request.body,
                }) {
                    log::error!(
                        "Model worker is not available for /v1/internal/reference_test: {}",
                        error
                    );
                }
            } else {
                let _ = send_json(
                    &mut tcp_stream,
                    404,
                    r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#,
                );
            }
        }

        ("POST", "/v1/internal/sequence_state_inventory") => {
            if test_endpoints {
                if let Err(error) = scheduler.enqueue(ModelRequest::SequenceStateInventory {
                    stream: tcp_stream,
                    body: request.body,
                }) {
                    log::error!(
                        "Model worker is not available for /v1/internal/sequence_state_inventory: {}",
                        error
                    );
                }
            } else {
                let _ = send_json(
                    &mut tcp_stream,
                    404,
                    r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#,
                );
            }
        }

        ("POST", "/v1/internal/sequence_state_transfer_measurement") => {
            if test_endpoints {
                if let Err(error) =
                    scheduler.enqueue(ModelRequest::SequenceStateTransferMeasurement {
                        stream: tcp_stream,
                        body: request.body,
                    })
                {
                    log::error!(
                        "Model worker is not available for /v1/internal/sequence_state_transfer_measurement: {}",
                        error
                    );
                }
            } else {
                let _ = send_json(
                    &mut tcp_stream,
                    404,
                    r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#,
                );
            }
        }

        _ => {
            let _ = send_json(&mut tcp_stream, 404, r#"{"error":"Not found"}"#);
        }
    }
}

fn checked_metric_sum(values: &[u64], name: &str) -> u64 {
    match values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
    {
        Some(total) => total,
        None => {
            log::error!("Session-cache metric total {} overflowed u64", name);
            u64::MAX
        }
    }
}

fn handle_session_cache_stats(stream: &mut TcpStream, state: &ServerState) {
    let metrics = &state.session_cache.metrics;
    let ram = state
        .session_cache
        .ram_store
        .as_ref()
        .map(crate::session_cache::RamSessionStore::stats)
        .unwrap_or_default();
    let hit_total = checked_metric_sum(&[metrics.active_hits, metrics.ram_hits], "hits");
    let miss_values = [
        metrics.no_match_misses,
        metrics.signature_mismatch_misses,
        metrics.evicted_misses,
        metrics.restore_not_worth_it_misses,
        metrics.divergence_misses,
        metrics.request_disabled_misses,
        metrics.image_input_misses,
        metrics.multi_gpu_pending_misses,
        metrics.speculative_decode_misses,
        metrics.no_suffix_misses,
        metrics.restore_failed_misses,
    ];
    let miss_total = checked_metric_sum(&miss_values, "misses");
    let save_average_ms = if metrics.save_count > 0 {
        metrics.save_total_ms / metrics.save_count as f64
    } else {
        0.0
    };
    let restore_average_ms = if metrics.restore_count > 0 {
        metrics.restore_total_ms / metrics.restore_count as f64
    } else {
        0.0
    };
    let body = serde_json::json!({
        "object": "krasis_session_cache_stats",
        "enabled": state.session_cache.enabled,
        "config": {
            "ram_fraction": state.session_cache.ram_fraction,
            "budget_bytes": ram.last_budget_bytes,
            "effective_available_bytes": ram.last_effective_available_bytes,
        },
        "capabilities": {
            "exact_mid_prefill_boundary_capture": unsafe {
                (&*(state.gpu_store_addr as *const GpuDecodeStore))
                    .exact_mid_prefill_boundary_capture_supported_rust()
            },
            "multi_gpu_pending": session_cache_multi_gpu_pending(
                state.session_cache.enabled,
                &state.aux_gpu_store_addrs,
            ),
            "mid_prefill_boundary_skipped": metrics.mid_prefill_boundary_skipped,
        },
        "hits": {
            "total": hit_total,
            "active_gpu": metrics.active_hits,
            "pageable_ram": metrics.ram_hits,
        },
        "misses": {
            "total": miss_total,
            "no_match": metrics.no_match_misses,
            "signature_mismatch": metrics.signature_mismatch_misses,
            "evicted": metrics.evicted_misses,
            "restore_not_worth_it": metrics.restore_not_worth_it_misses,
            "divergence": metrics.divergence_misses,
            "request_disabled": metrics.request_disabled_misses,
            "image_input_uncacheable": metrics.image_input_misses,
            "multi_gpu_pending": metrics.multi_gpu_pending_misses,
            "speculative_decode_uncacheable": metrics.speculative_decode_misses,
            "no_suffix": metrics.no_suffix_misses,
            "restore_failed": metrics.restore_failed_misses,
        },
        "resident": {
            "snapshots": ram.resident_snapshots,
            "bytes": ram.resident_bytes,
            "snapshot_bytes": ram.snapshot_bytes,
            "index_bytes": ram.index_bytes,
            "reserved_bytes": ram.reserved_bytes,
            "active_gpu_tokens": state.session_cache.active.as_ref().map_or(0, |active| active.consumed_token_ids.len()),
        },
        "evictions": ram.evictions,
        "timing": {
            "save": {
                "count": metrics.save_count,
                "bytes": metrics.save_bytes,
                "total_ms": metrics.save_total_ms,
                "last_ms": metrics.save_last_ms,
                "average_ms": save_average_ms,
            },
            "restore": {
                "count": metrics.restore_count,
                "bytes": metrics.restore_bytes,
                "total_ms": metrics.restore_total_ms,
                "last_ms": metrics.restore_last_ms,
                "average_ms": restore_average_ms,
            },
        },
        "runtime_measurements": {
            "prefill_samples": state.session_cache.prefill_samples.len(),
            "restore_samples": state.session_cache.restore_samples.len(),
        },
    });
    let _ = send_json(stream, 200, &body.to_string());
}

fn publish_session_cache_metrics(state: &ServerState) {
    let metrics = &state.session_cache.metrics;
    let ram = state
        .session_cache
        .ram_store
        .as_ref()
        .map(crate::session_cache::RamSessionStore::stats)
        .unwrap_or_default();
    let hits = checked_metric_sum(&[metrics.active_hits, metrics.ram_hits], "hits");
    let misses = checked_metric_sum(
        &[
            metrics.no_match_misses,
            metrics.signature_mismatch_misses,
            metrics.evicted_misses,
            metrics.restore_not_worth_it_misses,
            metrics.divergence_misses,
            metrics.request_disabled_misses,
            metrics.image_input_misses,
            metrics.multi_gpu_pending_misses,
            metrics.speculative_decode_misses,
            metrics.no_suffix_misses,
            metrics.restore_failed_misses,
        ],
        "misses",
    );
    state.serving_metrics.with_inner(|inner| {
        inner.session_enabled = state.session_cache.enabled;
        inner.session_budget_bytes = ram.last_budget_bytes;
        inner.session_resident_bytes = ram.resident_bytes;
        inner.session_resident_snapshots = ram.resident_snapshots;
        inner.session_active_gpu_tokens = state
            .session_cache
            .active
            .as_ref()
            .map_or(0, |active| active.consumed_token_ids.len());
        inner.session_hits = hits;
        inner.session_misses = misses;
        inner.session_evictions = ram.evictions;
        inner.session_save_count = metrics.save_count;
        inner.session_save_bytes = metrics.save_bytes;
        inner.session_save_seconds = metrics.save_total_ms / 1000.0;
        inner.session_restore_count = metrics.restore_count;
        inner.session_restore_bytes = metrics.restore_bytes;
        inner.session_restore_seconds = metrics.restore_total_ms / 1000.0;
    });
}

fn handle_sequence_state_inventory(stream: &mut TcpStream, body: &str, state: &mut ServerState) {
    let request: serde_json::Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(error) => {
            let response = serde_json::json!({"error": format!("invalid JSON: {}", error)});
            let _ = send_json(stream, 400, &response.to_string());
            return;
        }
    };
    let logical_tokens = match request.get("logical_tokens") {
        Some(value) => match value.as_u64().and_then(|value| usize::try_from(value).ok()) {
            Some(value) => value,
            None => {
                let response = serde_json::json!({
                    "error": "logical_tokens must be a non-negative platform-sized integer"
                });
                let _ = send_json(stream, 400, &response.to_string());
                return;
            }
        },
        None => {
            let response = serde_json::json!({"error": "logical_tokens is required"});
            let _ = send_json(stream, 400, &response.to_string());
            return;
        }
    };
    let mut devices = Vec::with_capacity(1 + state.aux_gpu_store_addrs.len());
    if state.gpu_store_addr != 0 {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        devices.push(store.sequence_state_inventory_value(logical_tokens));
    }
    for &address in &state.aux_gpu_store_addrs {
        if address == 0 {
            continue;
        }
        let store = unsafe { &*(address as *const GpuDecodeStore) };
        devices.push(store.sequence_state_inventory_value(logical_tokens));
    }
    let response = serde_json::json!({
        "logical_tokens": logical_tokens,
        "device_count": devices.len(),
        "allocated_bytes": devices.iter().filter_map(|device| device.get("allocated_bytes").and_then(|value| value.as_u64())).sum::<u64>(),
        "used_bytes": devices.iter().filter_map(|device| device.get("used_bytes").and_then(|value| value.as_u64())).sum::<u64>(),
        "devices": devices,
    });
    let _ = send_json(stream, 200, &response.to_string());
}

fn handle_sequence_state_transfer_measurement(
    stream: &mut TcpStream,
    body: &str,
    state: &mut ServerState,
) {
    let request: serde_json::Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(error) => {
            let response = serde_json::json!({"error": format!("invalid JSON: {}", error)});
            let _ = send_json(stream, 400, &response.to_string());
            return;
        }
    };
    let parse_positive = |name: &str| -> Result<usize, String> {
        request
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{} must be a positive platform-sized integer", name))
    };
    let logical_tokens = match parse_positive("logical_tokens") {
        Ok(value) => value,
        Err(error) => {
            let response = serde_json::json!({"error": error});
            let _ = send_json(stream, 400, &response.to_string());
            return;
        }
    };
    let iterations = match parse_positive("iterations") {
        Ok(value) => value,
        Err(error) => {
            let response = serde_json::json!({"error": error});
            let _ = send_json(stream, 400, &response.to_string());
            return;
        }
    };

    let mut devices = Vec::with_capacity(1 + state.aux_gpu_store_addrs.len());
    if state.gpu_store_addr != 0 {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        match store.sequence_state_transfer_measurement_value(logical_tokens, iterations) {
            Ok(value) => devices.push(value),
            Err(error) => {
                let response = serde_json::json!({"error": error});
                let _ = send_json(stream, 500, &response.to_string());
                return;
            }
        }
    }
    for &address in &state.aux_gpu_store_addrs {
        if address == 0 {
            continue;
        }
        let store = unsafe { &mut *(address as *mut GpuDecodeStore) };
        match store.sequence_state_transfer_measurement_value(logical_tokens, iterations) {
            Ok(value) => devices.push(value),
            Err(error) => {
                let response = serde_json::json!({"error": error});
                let _ = send_json(stream, 500, &response.to_string());
                return;
            }
        }
    }
    let response = serde_json::json!({
        "logical_tokens": logical_tokens,
        "iterations": iterations,
        "device_count": devices.len(),
        "devices": devices,
    });
    let _ = send_json(stream, 200, &response.to_string());
}

fn fnv1a_token_hash(token_ids: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &token in token_ids {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn mamba2_state_lifecycle_point(
    store: &GpuDecodeStore,
    phase: &str,
    layer_idx: usize,
) -> serde_json::Value {
    let raw = store.mamba2_state_debug_summary_json(phase, layer_idx);
    serde_json::from_str(&raw).unwrap_or_else(|e| {
        serde_json::json!({
            "phase": phase,
            "layer": layer_idx,
            "available": false,
            "error": format!("parse_failed: {}", e),
            "raw": raw,
        })
    })
}

fn reference_logit_trace_json(
    logits: &[f32],
    vocab_size: usize,
    selected_token: usize,
    top_n: usize,
) -> serde_json::Value {
    let vocab_size = vocab_size.min(logits.len());
    if vocab_size == 0 {
        return serde_json::json!({
            "available": false,
            "reason": "empty_logits",
        });
    }

    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_token = 0usize;
    let mut min_logit = f32::INFINITY;
    let mut min_token = 0usize;

    for (idx, &value) in logits[..vocab_size].iter().enumerate() {
        if value.is_nan() {
            nan_count += 1;
            continue;
        }
        if value == f32::INFINITY {
            pos_inf_count += 1;
        } else if value == f32::NEG_INFINITY {
            neg_inf_count += 1;
        } else {
            finite_count += 1;
        }
        if value > max_logit {
            max_logit = value;
            max_token = idx;
        }
        if value < min_logit {
            min_logit = value;
            min_token = idx;
        }
    }

    let sum_exp: f64 = logits[..vocab_size]
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&x| ((x - max_logit) as f64).exp())
        .sum();
    let log_sum_exp = max_logit as f64 + sum_exp.ln();
    let selected_raw_logit = logits.get(selected_token).copied().unwrap_or(f32::NAN);
    let selected_logprob = selected_raw_logit as f64 - log_sum_exp;

    let mut top_logits: Vec<(usize, f32)> = Vec::with_capacity(top_n.saturating_add(1));
    for (idx, &value) in logits[..vocab_size].iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if top_logits.len() < top_n {
            top_logits.push((idx, value));
            if top_logits.len() == top_n {
                top_logits
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if top_n > 0 && value > top_logits[top_n - 1].1 {
            top_logits[top_n - 1] = (idx, value);
            top_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    top_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_entries: Vec<serde_json::Value> = top_logits
        .iter()
        .enumerate()
        .map(|(rank, &(token_id, raw_logit))| {
            serde_json::json!({
                "rank": rank + 1,
                "token_id": token_id,
                "raw_logit": raw_logit as f64,
                "logprob": raw_logit as f64 - log_sum_exp,
                "softmax_prob": (raw_logit as f64 - log_sum_exp).exp(),
            })
        })
        .collect();

    serde_json::json!({
        "available": true,
        "source": "prefill_engine.h_logits_after_lm_head_download_and_suppression",
        "dtype": "f32",
        "device_before_download": "cuda",
        "host_buffer": "engine.h_logits",
        "vocab_size": vocab_size,
        "selected_token_id": selected_token,
        "selected_raw_logit": selected_raw_logit as f64,
        "selected_logprob_from_raw": selected_logprob,
        "selected_softmax_prob_from_raw": selected_logprob.exp(),
        "max_logit": max_logit as f64,
        "max_token_id": max_token,
        "min_logit": min_logit as f64,
        "min_token_id": min_token,
        "sum_exp_shifted": sum_exp,
        "logsumexp": log_sum_exp,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "top_logits_before_logprob": top_entries,
    })
}

/// Begin an SSE stream (send headers, return stream for data).
fn begin_sse(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: keep-alive\r\n\r\n"
    )?;
    stream.flush()
}

/// Send one SSE data chunk.
fn send_sse_chunk(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(stream, "data: {}\n\n", data)?;
    stream.flush()
}

/// Format an SSE chunk as OpenAI chat.completion.chunk JSON.
fn format_sse_token(
    request_id: &str,
    model_name: &str,
    text: &str,
    finish_reason: Option<&str>,
    created: u64,
    logprobs: Option<&[(u32, f32)]>,
) -> String {
    let delta = if text.is_empty() {
        "{}".to_string()
    } else {
        format!(r#"{{"content":{}}}"#, json_string(text))
    };
    let fr = match finish_reason {
        Some(r) => json_string(r),
        None => "null".to_string(),
    };
    let logprobs_str = if let Some(lps) = logprobs {
        // OpenAI format: {"content": [{"token": "...", "logprob": -0.5, "top_logprobs": [{"token": "...", "logprob": -0.5}, ...]}]}
        let mut top_entries = Vec::new();
        for &(tid, lp) in lps.iter() {
            top_entries.push(format!(r#"{{"token_id":{},"logprob":{:.6}}}"#, tid, lp));
        }
        let top_str = top_entries.join(",");
        // The first entry is the selected token
        let selected_lp = if !lps.is_empty() { lps[0].1 } else { 0.0 };
        format!(
            r#","logprobs":{{"content":[{{"logprob":{:.6},"top_logprobs":[{}]}}]}}"#,
            selected_lp, top_str
        )
    } else {
        String::new()
    };
    format!(
        r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{},"finish_reason":{}{}}}]}}"#,
        json_string(request_id),
        created,
        json_string(model_name),
        delta,
        fr,
        logprobs_str
    )
}

/// Format a complete (non-streaming) chat completion response.
fn format_completion(
    request_id: &str,
    model_name: &str,
    text: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
    created: u64,
) -> String {
    format!(
        r#"{{"id":{},"object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":{}}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        json_string(request_id),
        created,
        json_string(model_name),
        json_string(text),
        json_string(finish_reason),
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens
    )
}

// ── Tool use support ──────────────────────────────────────────────

/// A parsed tool call extracted from model output.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ParsedToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

type RawToolCall = (String, serde_json::Value);

fn new_tool_call_id(call_idx: u64) -> String {
    format!("call_{:016x}", {
        let mut s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s ^= call_idx;
        s
    })
}

fn value_from_untyped_tool_text(value: &str) -> serde_json::Value {
    serde_json::from_str(value).unwrap_or_else(|_| serde_json::Value::String(value.to_string()))
}

fn parse_delimited_tool_blocks(
    text: &str,
    start_marker: &str,
    end_marker: &str,
    parse_block: fn(&str) -> Option<Vec<RawToolCall>>,
) -> (String, Vec<ParsedToolCall>) {
    let mut tool_calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;
    let mut call_idx = 0u64;

    while let Some(start) = remaining.find(start_marker) {
        content.push_str(&remaining[..start]);
        let after_start = &remaining[start + start_marker.len()..];

        if let Some(end) = after_start.find(end_marker) {
            let block = &after_start[..end];
            let raw_end = end + end_marker.len();
            let raw_block = &remaining[start..start + start_marker.len() + raw_end];
            remaining = &after_start[raw_end..];

            if let Some(parsed) = parse_block(block) {
                for (name, arguments) in parsed {
                    let id = new_tool_call_id(call_idx);
                    call_idx += 1;
                    tool_calls.push(ParsedToolCall {
                        id,
                        name,
                        arguments_json: arguments.to_string(),
                    });
                }
            } else {
                // A complete but malformed block remains visible as assistant
                // text. Never discard model output or fabricate a tool call.
                content.push_str(raw_block);
            }
        } else {
            // A truncated block remains visible as assistant text.
            content.push_str(&remaining[start..]);
            remaining = "";
        }
    }

    content.push_str(remaining);
    (content.trim().to_string(), tool_calls)
}

fn parse_qwen_json_block(block: &str) -> Option<Vec<RawToolCall>> {
    let value: serde_json::Value = serde_json::from_str(block.trim()).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let arguments = match object.get("arguments") {
        Some(serde_json::Value::String(encoded)) => serde_json::from_str(encoded).ok()?,
        Some(value) => value.clone(),
        None => serde_json::json!({}),
    };
    if !arguments.is_object() {
        return None;
    }
    Some(vec![(name.to_string(), arguments)])
}

fn parse_function_xml_block(block: &str) -> Option<Vec<RawToolCall>> {
    let block = block.trim();
    let after = block.strip_prefix("<function=")?;
    let name_end = after.find('>')?;
    let name = after[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let inner = &after[name_end + 1..];
    let function_end = inner.rfind("</function>")?;
    if !inner[function_end + "</function>".len()..]
        .trim()
        .is_empty()
    {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut remaining = inner[..function_end].trim();
    while !remaining.is_empty() {
        let after_param = remaining.strip_prefix("<parameter=")?;
        let name_end = after_param.find('>')?;
        let param_name = after_param[..name_end].trim();
        if param_name.is_empty() {
            return None;
        }
        let value_text = &after_param[name_end + 1..];
        let value_end = value_text.find("</parameter>")?;
        let value = value_text[..value_end]
            .trim_start_matches(['\n', '\r'])
            .trim_end_matches(['\n', '\r']);
        args.insert(param_name.to_string(), value_from_untyped_tool_text(value));
        remaining = value_text[value_end + "</parameter>".len()..].trim();
    }
    Some(vec![(name.to_string(), serde_json::Value::Object(args))])
}

fn parse_glm_xml_block(block: &str) -> Option<Vec<RawToolCall>> {
    let block = block.trim();
    let first_arg = block.find("<arg_key>");
    let name = block[..first_arg.unwrap_or(block.len())].trim();
    if name.is_empty() || name.contains(['<', '>']) {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut remaining = &block[first_arg.unwrap_or(block.len())..];
    while !remaining.trim().is_empty() {
        remaining = remaining.trim();
        let after_key = remaining.strip_prefix("<arg_key>")?;
        let key_end = after_key.find("</arg_key>")?;
        let key = after_key[..key_end].trim();
        if key.is_empty() {
            return None;
        }
        let after_key = after_key[key_end + "</arg_key>".len()..].trim_start();
        let after_value = after_key.strip_prefix("<arg_value>")?;
        let value_end = after_value.find("</arg_value>")?;
        let value = after_value[..value_end].trim();
        args.insert(key.to_string(), value_from_untyped_tool_text(value));
        remaining = &after_value[value_end + "</arg_value>".len()..];
    }
    Some(vec![(name.to_string(), serde_json::Value::Object(args))])
}

/// GLM can repeat its exact tool-call container special token once, producing
/// `<tool_call><tool_call>...body...</tool_call></tool_call>`. Normalize only
/// that complete, balanced duplicate wrapper. Incomplete or differently
/// malformed markup remains untouched and visible under the fail-safe parser
/// contract.
fn parse_glm_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";
    const DOUBLE_START: &str = "<tool_call><tool_call>";
    const DOUBLE_END: &str = "</tool_call></tool_call>";

    if !text.contains(DOUBLE_START) {
        return parse_delimited_tool_blocks(text, START, END, parse_glm_xml_block);
    }

    let mut normalized = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find(DOUBLE_START) {
        normalized.push_str(&remaining[..start]);
        let after_double_start = &remaining[start + DOUBLE_START.len()..];
        let Some(end) = after_double_start.find(DOUBLE_END) else {
            normalized.push_str(&remaining[start..]);
            remaining = "";
            break;
        };
        normalized.push_str(START);
        normalized.push_str(&after_double_start[..end]);
        normalized.push_str(END);
        remaining = &after_double_start[end + DOUBLE_END.len()..];
    }
    normalized.push_str(remaining);
    parse_delimited_tool_blocks(&normalized, START, END, parse_glm_xml_block)
}

fn xml_attribute<'a>(opening_tag: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!(" {}=\"", name);
    let start = opening_tag.find(&needle)? + needle.len();
    let end = opening_tag[start..].find('"')? + start;
    Some(&opening_tag[start..end])
}

fn parse_invoke_blocks(
    block: &str,
    invoke_start: &str,
    invoke_end: &str,
    parameter_start: &str,
    parameter_end: &str,
    typed_strings: bool,
) -> Option<Vec<RawToolCall>> {
    let mut calls = Vec::new();
    let mut remaining = block.trim();
    while !remaining.is_empty() {
        let after_invoke = remaining.strip_prefix(invoke_start)?;
        let opening_end = after_invoke.find('>')?;
        let opening_tag = &remaining[..invoke_start.len() + opening_end + 1];
        let name = xml_attribute(opening_tag, "name")?.trim();
        if name.is_empty() {
            return None;
        }
        let invoke_body = &after_invoke[opening_end + 1..];
        let invoke_end_at = invoke_body.find(invoke_end)?;
        let mut parameters = invoke_body[..invoke_end_at].trim();
        let mut args = serde_json::Map::new();
        while !parameters.is_empty() {
            let after_parameter = parameters.strip_prefix(parameter_start)?;
            let parameter_open_end = after_parameter.find('>')?;
            let opening_tag = &parameters[..parameter_start.len() + parameter_open_end + 1];
            let parameter_name = xml_attribute(opening_tag, "name")?.trim();
            if parameter_name.is_empty() {
                return None;
            }
            let value_text = &after_parameter[parameter_open_end + 1..];
            let value_end = value_text.find(parameter_end)?;
            let raw_value = &value_text[..value_end];
            let value = if typed_strings {
                match xml_attribute(opening_tag, "string")? {
                    "true" => serde_json::Value::String(raw_value.to_string()),
                    "false" => serde_json::from_str(raw_value.trim()).ok()?,
                    _ => return None,
                }
            } else {
                value_from_untyped_tool_text(raw_value.trim())
            };
            args.insert(parameter_name.to_string(), value);
            parameters = value_text[value_end + parameter_end.len()..].trim();
        }
        calls.push((name.to_string(), serde_json::Value::Object(args)));
        remaining = invoke_body[invoke_end_at + invoke_end.len()..].trim();
    }
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn parse_deepseek_dsml_block(block: &str) -> Option<Vec<RawToolCall>> {
    parse_invoke_blocks(
        block,
        "<｜DSML｜invoke",
        "</｜DSML｜invoke>",
        "<｜DSML｜parameter",
        "</｜DSML｜parameter>",
        true,
    )
    .or_else(|| {
        // DeepSeek-V4-Flash can abbreviate only the inner closing/parameter
        // tags even though its template specifies the fully-prefixed DSML
        // grammar. The canonical outer DSML container and invoke opener still
        // identify the grammar unambiguously. Accept that observed form while
        // retaining fail-safe rejection for incomplete or otherwise malformed
        // blocks.
        parse_invoke_blocks(
            block,
            "<｜DSML｜invoke",
            "</invoke>",
            "<parameter",
            "</parameter>",
            false,
        )
    })
}

fn parse_minimax_block(block: &str) -> Option<Vec<RawToolCall>> {
    parse_invoke_blocks(
        block,
        "<invoke",
        "</invoke>",
        "<parameter",
        "</parameter>",
        false,
    )
}

struct GemmaValueParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> GemmaValueParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.remaining().chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            self.pos += ch.len_utf8();
        }
    }

    fn consume(&mut self, expected: &str) -> bool {
        self.skip_ws();
        if self.remaining().starts_with(expected) {
            self.pos += expected.len();
            true
        } else {
            false
        }
    }

    fn parse_string(&mut self) -> Option<serde_json::Value> {
        const QUOTE: &str = "<|\"|>";
        if !self.consume(QUOTE) {
            return None;
        }
        let end = self.remaining().find(QUOTE)?;
        let value = self.remaining()[..end].to_string();
        self.pos += end + QUOTE.len();
        Some(serde_json::Value::String(value))
    }

    fn parse_key(&mut self) -> Option<String> {
        self.skip_ws();
        if self.remaining().starts_with("<|\"|>") {
            return self.parse_string()?.as_str().map(String::from);
        }
        let end = self.remaining().find(':')?;
        let key = self.remaining()[..end].trim();
        if key.is_empty() || key.contains([',', '{', '}', '[', ']']) {
            return None;
        }
        self.pos += end;
        Some(key.to_string())
    }

    fn parse_object(&mut self) -> Option<serde_json::Value> {
        if !self.consume("{") {
            return None;
        }
        let mut object = serde_json::Map::new();
        self.skip_ws();
        if self.consume("}") {
            return Some(serde_json::Value::Object(object));
        }
        loop {
            let key = self.parse_key()?;
            if !self.consume(":") {
                return None;
            }
            object.insert(key, self.parse_value()?);
            if self.consume("}") {
                break;
            }
            if !self.consume(",") {
                return None;
            }
        }
        Some(serde_json::Value::Object(object))
    }

    fn parse_array(&mut self) -> Option<serde_json::Value> {
        if !self.consume("[") {
            return None;
        }
        let mut values = Vec::new();
        self.skip_ws();
        if self.consume("]") {
            return Some(serde_json::Value::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            if self.consume("]") {
                break;
            }
            if !self.consume(",") {
                return None;
            }
        }
        Some(serde_json::Value::Array(values))
    }

    fn parse_scalar(&mut self) -> Option<serde_json::Value> {
        self.skip_ws();
        let end = self
            .remaining()
            .find([',', '}', ']'])
            .unwrap_or(self.remaining().len());
        let token = self.remaining()[..end].trim();
        if token.is_empty() {
            return None;
        }
        self.pos += end;
        serde_json::from_str(token).ok()
    }

    fn parse_value(&mut self) -> Option<serde_json::Value> {
        self.skip_ws();
        if self.remaining().starts_with("<|\"|>") {
            self.parse_string()
        } else if self.remaining().starts_with('{') {
            self.parse_object()
        } else if self.remaining().starts_with('[') {
            self.parse_array()
        } else {
            self.parse_scalar()
        }
    }
}

fn parse_gemma_block(block: &str) -> Option<Vec<RawToolCall>> {
    let body = block.trim().strip_prefix("call:")?;
    let args_start = body.find('{')?;
    let name = body[..args_start].trim();
    if name.is_empty() {
        return None;
    }
    let mut parser = GemmaValueParser::new(&body[args_start..]);
    let arguments = parser.parse_object()?;
    parser.skip_ws();
    if !parser.remaining().is_empty() {
        return None;
    }
    Some(vec![(name.to_string(), arguments)])
}

/// Parse the native tool-call grammar selected from the loaded chat template.
/// Returns (content_text, tool_calls). Malformed or truncated blocks remain in
/// content rather than being dropped or partially interpreted.
fn parse_tool_calls(
    text: &str,
    format: crate::chat_template::ToolCallFormat,
) -> (String, Vec<ParsedToolCall>) {
    use crate::chat_template::ToolCallFormat;
    match format {
        ToolCallFormat::Unsupported => (text.to_string(), Vec::new()),
        ToolCallFormat::QwenJson => {
            parse_delimited_tool_blocks(text, "<tool_call>", "</tool_call>", parse_qwen_json_block)
        }
        ToolCallFormat::FunctionXml => parse_delimited_tool_blocks(
            text,
            "<tool_call>",
            "</tool_call>",
            parse_function_xml_block,
        ),
        ToolCallFormat::GlmXml => parse_glm_tool_calls(text),
        ToolCallFormat::DeepseekDsml => parse_delimited_tool_blocks(
            text,
            "<｜DSML｜tool_calls>",
            "</｜DSML｜tool_calls>",
            parse_deepseek_dsml_block,
        ),
        ToolCallFormat::Gemma => {
            parse_delimited_tool_blocks(text, "<|tool_call>", "<tool_call|>", parse_gemma_block)
        }
        ToolCallFormat::Minimax => parse_delimited_tool_blocks(
            text,
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            parse_minimax_block,
        ),
    }
}

/// Stream ordinary assistant text while retaining the shortest suffix that
/// could still become a native tool-call marker on the next decoded token.
/// Once the marker is complete, all remaining text is buffered for structured
/// parsing. This handles markers split at arbitrary UTF-8 token boundaries.
fn push_tool_stream_text(
    marker: &str,
    pending: &mut String,
    captured: &mut String,
    found_marker: &mut bool,
    text: &str,
) -> String {
    if *found_marker {
        captured.push_str(text);
        return String::new();
    }

    pending.push_str(text);
    if let Some(start) = pending.find(marker) {
        let visible = pending[..start].to_string();
        captured.push_str(&pending[start..]);
        pending.clear();
        *found_marker = true;
        return visible;
    }

    let max_suffix = pending.len().min(marker.len().saturating_sub(1));
    let mut retained = 0usize;
    for len in (1..=max_suffix).rev() {
        let start = pending.len() - len;
        if pending.is_char_boundary(start)
            && marker.is_char_boundary(len)
            && pending[start..] == marker[..len]
        {
            retained = len;
            break;
        }
    }
    let visible_end = pending.len() - retained;
    let visible = pending[..visible_end].to_string();
    let suffix = pending[visible_end..].to_string();
    *pending = suffix;
    visible
}

/// Serialize a string as a complete JSON string literal, including quotes.
///
/// Keeping quoting and escaping together prevents callers from accidentally
/// embedding raw Windows paths or control characters in response envelopes.
#[inline]
fn json_string(s: &str) -> String {
    serde_json::to_string(s).expect("serializing a Rust string to JSON cannot fail")
}

/// Escape a string for legacy response templates that provide their own quotes.
fn json_escape(s: &str) -> String {
    let quoted = json_string(s);
    quoted[1..quoted.len() - 1].to_string()
}

fn hide_synthetic_think_stop_text(
    token_id: usize,
    finish_reason: Option<&str>,
    hidden_think_stop_id: Option<usize>,
) -> bool {
    finish_reason == Some("stop") && hidden_think_stop_id == Some(token_id)
}

fn push_eos_token_id_from_json(value: &serde_json::Value, ids: &mut Vec<usize>) {
    let Some(eos) = value.get("eos_token_id") else {
        return;
    };
    match eos {
        serde_json::Value::Number(n) => {
            if let Some(id) = n.as_u64() {
                let id = id as usize;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                if let Some(id) = v.as_u64() {
                    let id = id as usize;
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }
        _ => {}
    }
}

fn collect_eos_stop_ids(tokenizer_path: &str) -> Vec<usize> {
    let p = std::path::Path::new(tokenizer_path);
    let model_dir = p.parent().unwrap_or(p);
    let mut ids = Vec::new();

    // Match Python config parsing order: generation_config.json is
    // authoritative, then config.json top level, then nested text_config.
    let gen_cfg_path = model_dir.join("generation_config.json");
    if let Ok(data) = std::fs::read_to_string(&gen_cfg_path) {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
            push_eos_token_id_from_json(&cfg, &mut ids);
        }
    }

    let config_path = model_dir.join("config.json");
    if let Ok(data) = std::fs::read_to_string(&config_path) {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
            push_eos_token_id_from_json(&cfg, &mut ids);
            if let Some(text_cfg) = cfg.get("text_config") {
                push_eos_token_id_from_json(text_cfg, &mut ids);
            }
        }
    }

    ids
}

fn image_vram_error_body(err: &str) -> Option<String> {
    let marker = "VRAM is too constrained";
    let start = err.find(marker)?;
    let message = err[start..].lines().next().unwrap_or(&err[start..]).trim();
    Some(format!(
        r#"{{"error":{{"message":"{}","type":"insufficient_resources","code":"insufficient_vram"}}}}"#,
        json_escape(message)
    ))
}

/// Format SSE chunk: tool call start (name + empty args).
fn format_sse_tool_call_start(
    request_id: &str,
    model_name: &str,
    call_index: usize,
    call_id: &str,
    function_name: &str,
    created: u64,
) -> String {
    format!(
        r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"id":{},"type":"function","function":{{"name":{},"arguments":""}}}}]}},"finish_reason":null}}]}}"#,
        json_string(request_id),
        created,
        json_string(model_name),
        call_index,
        json_string(call_id),
        json_string(function_name)
    )
}

/// Format SSE chunk: tool call arguments fragment.
fn format_sse_tool_call_args(
    request_id: &str,
    model_name: &str,
    call_index: usize,
    arguments_json: &str,
    created: u64,
) -> String {
    format!(
        r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"function":{{"arguments":{}}}}}]}},"finish_reason":null}}]}}"#,
        json_string(request_id),
        created,
        json_string(model_name),
        call_index,
        json_string(arguments_json)
    )
}

/// Format non-streaming response with tool calls.
fn format_completion_with_tool_calls(
    request_id: &str,
    model_name: &str,
    content: &str,
    tool_calls: &[ParsedToolCall],
    prompt_tokens: usize,
    completion_tokens: usize,
    created: u64,
) -> String {
    let mut tc_parts = Vec::new();
    for tc in tool_calls {
        tc_parts.push(format!(
            r#"{{"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#,
            json_string(&tc.id),
            json_string(&tc.name),
            json_string(&tc.arguments_json)
        ));
    }
    let content_field = if content.is_empty() {
        "null".to_string()
    } else {
        json_string(content)
    };
    format!(
        r#"{{"id":{},"object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{},"tool_calls":[{}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        json_string(request_id),
        created,
        json_string(model_name),
        content_field,
        tc_parts.join(","),
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens
    )
}

/// Overhead timings collected during request setup (before decode).
struct RequestOverhead {
    parse_ms: f64,           // HTTP parse + JSON parse + tokenization
    evict_ms: f64,           // HCS soft-tier eviction
    prefill_ms: f64,         // GIL acquire + Python prefill
    reload_ms: f64,          // HCS soft-tier reload (wall-clock, includes sync if enabled)
    real_reload_dma_ms: f64, // Actual DMA time when sync is on (0.0 if async)
}

#[allow(clippy::too_many_arguments)]
fn format_sse_timing(
    request_id: &str,
    model_name: &str,
    created: u64,
    decode_tokens: usize,
    decode_time_ms: f64,
    decode_tok_s: f64,
    thinking_tokens: usize,
    answer_tokens: usize,
    total_generated: usize,
    prompt_tokens: usize,
    prefill_tok_s: f64,
    overhead_ms: f64,
    overhead: &RequestOverhead,
) -> String {
    format!(
        r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[],"krasis_timing":{{"decode_tokens":{},"decode_time_ms":{:.1},"decode_tok_s":{:.2},"thinking_tokens":{},"answer_tokens":{},"total_generated":{},"prompt_tokens":{},"prefill_tok_s":{:.1},"overhead_ms":{:.1},"overhead":{{"parse_ms":{:.1},"evict_ms":{:.1},"prefill_ms":{:.1},"reload_ms":{:.1},"real_reload_dma_ms":{:.1}}}}}}}"#,
        json_string(request_id),
        created,
        json_string(model_name),
        decode_tokens,
        decode_time_ms,
        decode_tok_s,
        thinking_tokens,
        answer_tokens,
        total_generated,
        prompt_tokens,
        prefill_tok_s,
        overhead_ms,
        overhead.parse_ms,
        overhead.evict_ms,
        overhead.prefill_ms,
        overhead.reload_ms,
        overhead.real_reload_dma_ms
    )
}

fn format_completion_with_debug(
    request_id: &str,
    model_name: &str,
    text: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
    created: u64,
    debug: Option<&serde_json::Value>,
) -> String {
    let mut response = format_completion(
        request_id,
        model_name,
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        created,
    );
    if let Some(debug_value) = debug {
        response.pop();
        response.push_str(&format!(r#","krasis_debug":{}"#, debug_value));
        response.push('}');
    }
    response
}

struct MultimodalPrefillInputs {
    token_ids: Vec<u32>,
    inputs_embeds_ptr: u64,
    mrope_cos_ptr: u64,
    mrope_sin_ptr: u64,
    mrope_half_dim: usize,
    rope_delta: i32,
    vision_block_ids_ptr: u64,
    vision_visible_left_ptr: u64,
    vision_visible_right_ptr: u64,
    vision_max_tokens: usize,
    image_count: usize,
    image_tokens: usize,
}

/// Handle /v1/chat/completions request.
fn handle_chat_completion(
    stream: &mut TcpStream,
    body: &str,
    state: &mut ServerState,
    received_at: Instant,
) {
    let t_request = received_at;
    let serving_metrics = Arc::clone(&state.serving_metrics);
    let mut metrics_guard = ServingRequestGuard::new(Arc::clone(&serving_metrics), received_at);
    publish_session_cache_metrics(state);

    // Parse request
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = send_json(
                stream,
                400,
                &format!(r#"{{"error":"Invalid JSON: {}"}}"#, e),
            );
            return;
        }
    };

    // Log full request body if request logging is enabled (for IDE debugging)
    if let Some(ref dir) = state.log_requests_dir {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let filename = format!("{}/{}.json", dir, ts.as_millis());
        if let Ok(pretty) = serde_json::to_string_pretty(&req) {
            std::fs::write(&filename, &pretty).ok();
        } else {
            std::fs::write(&filename, body).ok();
        }
    }

    let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(8192) as usize;
    let min_new_tokens = req
        .get("min_new_tokens")
        .or_else(|| req.get("min_completion_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
        .min(max_tokens);
    let temperature = req
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.6) as f32;
    let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32;
    let presence_penalty = req
        .get("presence_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let req_logprobs = req
        .get("logprobs")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let req_top_logprobs = req
        .get("top_logprobs")
        .and_then(|v| v.as_u64())
        .unwrap_or(5) as usize;
    let logprobs_top_n = if req_logprobs { req_top_logprobs } else { 0 };
    let enable_thinking = req
        .get("enable_thinking")
        .and_then(|v| v.as_bool())
        .unwrap_or(state.default_enable_thinking);
    let debug_first_token_boundary = req
        .get("debug_first_token_boundary")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let request_prefix_cache = match req.get("prefix_cache") {
        Some(value) => match value.as_bool() {
            Some(enabled) => enabled,
            None => {
                let _ = send_json(
                    stream,
                    400,
                    r#"{"error":"prefix_cache must be a boolean when provided"}"#,
                );
                return;
            }
        },
        None => true,
    };

    let request_id = format!("chatcmpl-{:016x}", {
        let mut s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    });
    crate::vram_monitor::begin_request_context(&format!(
        "route=/v1/chat/completions request_id={} model={} max_new={} stream={} phase=parse",
        request_id, state.model_name, max_tokens, is_stream,
    ));
    let _vram_context_guard = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        VramRequestContextGuard {
            safety_margin_mb: store.hcs_safety_margin_mb() as u64,
        }
    };
    drain_vram_pressure_for_state(state, "chat_request_entry", false);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Extract messages JSON for Python
    let (messages_json, has_images) = match req.get("messages") {
        Some(m) => {
            let has_images = crate::text_only_messages::messages_have_image_parts(m);
            let validation = if has_images {
                crate::text_only_messages::validate_image_only_messages(m)
            } else {
                crate::text_only_messages::validate_text_only_messages(m)
            };
            if let Err(e) = validation {
                let _ = send_json(
                    stream,
                    400,
                    &format!(r#"{{"error":"{}"}}"#, json_escape(&e)),
                );
                return;
            }
            (m.to_string(), has_images)
        }
        None => {
            let _ = send_json(stream, 400, r#"{"error":"Missing messages"}"#);
            return;
        }
    };

    // Custom stop tokens
    let stop_tokens: Vec<String> = match req.get("stop") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => vec![],
    };

    // Tool use: extract tools array and tool_choice
    // tool_choice can be a string ("auto", "none", "required") or an object
    // {"type": "function", "function": {"name": "..."}} — we pass tools through
    // unless tool_choice is explicitly "none".
    let tools_json = match req.get("tools") {
        Some(t) if t.is_array() => {
            let is_none = match req.get("tool_choice") {
                Some(serde_json::Value::String(s)) => s == "none",
                _ => false, // object form or missing = allow tools
            };
            if is_none {
                String::new()
            } else {
                t.to_string()
            }
        }
        _ => String::new(),
    };
    let has_tools = !tools_json.is_empty();
    let tool_call_format = state.chat_template.tool_call_format();
    if has_tools && tool_call_format.start_marker().is_none() {
        let _ = send_json(
            stream,
            400,
            r#"{"error":"The loaded chat template does not declare a supported native tool-call grammar"}"#,
        );
        return;
    }

    // ── Render chat template (reused for both token estimation and Rust prefill) ──
    let rendered_result = if has_images {
        state.chat_template.apply_multimodal_with_tools(
            &messages_json,
            &tools_json,
            true,
            enable_thinking,
        )
    } else {
        state
            .chat_template
            .apply_with_tools(&messages_json, &tools_json, true, enable_thinking)
    };
    let rendered = match rendered_result {
        Ok(r) => r,
        Err(e) => {
            log::error!("Chat template failed: {}", e);
            let _ = send_json(
                stream,
                500,
                &format!(
                    r#"{{"error":"Chat template failed: {}. This indicates a broken model setup."}}"#,
                    e
                ),
            );
            return;
        }
    };
    // A later turn normally replaces the assistant-generation suffix rather
    // than appending after it. Templates whose disabled-thinking scaffold is
    // explicitly history-stable can safely capture the ordinary full prompt;
    // all others render without the suffix and must prove an exact token
    // boundary below. Rendering alone is never treated as a correctness
    // guarantee.
    let stable_rendered = if state.session_cache.enabled && request_prefix_cache && !has_images {
        if !enable_thinking
            && state
                .chat_template
                .disabled_thinking_generation_prompt_is_history_stable()
        {
            Some(rendered.clone())
        } else {
            match state.chat_template.apply_with_tools(
                &messages_json,
                &tools_json,
                false,
                enable_thinking,
            ) {
                Ok(stable) => Some(stable),
                Err(error) => {
                    log::error!(
                        "Request {} prefix cache miss: stable_template_render_failed error={}",
                        request_id,
                        error,
                    );
                    None
                }
            }
        }
    } else {
        None
    };
    let estimated_tokens = if has_images {
        log::info!(
            "Soft HCS: image request pre-evicting for configured context window {} (rendered_len={})",
            state.max_context_tokens,
            rendered.len()
        );
        crate::vram_monitor::update_request_context(&format!(
            "route=/v1/chat/completions request_id={} model={} estimated_prompt_tokens={} rendered_len={} max_new={} stream={} phase=prefill_setup multimodal=image",
            request_id, state.model_name, state.max_context_tokens, rendered.len(), max_tokens, is_stream,
        ));
        state.max_context_tokens
    } else {
        let token_count = match state.tokenizer.encode(rendered.as_str(), false) {
            Ok(e) => e.len(),
            Err(e) => {
                log::error!("Tokenizer failed to encode prompt: {}", e);
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Tokenizer failed: {}. This indicates a broken model setup."}}"#,
                        e
                    ),
                );
                return;
            }
        };
        log::info!(
            "Soft HCS: estimated {} tokens (rendered_len={})",
            token_count,
            rendered.len()
        );
        crate::vram_monitor::update_request_context(&format!(
            "route=/v1/chat/completions request_id={} model={} estimated_prompt_tokens={} rendered_len={} max_new={} stream={} phase=prefill_setup",
            request_id, state.model_name, token_count, rendered.len(), max_tokens, is_stream,
        ));
        token_count
    };
    let parse_ms = t_request.elapsed().as_secs_f64() * 1000.0;

    if !has_images && !context_window_fits(estimated_tokens, max_tokens, state.max_context_tokens) {
        let requested_total = estimated_tokens.saturating_add(max_tokens);
        log::warn!(
            "Request {} rejected before prefill: estimated prompt {} + max_new {} = {} tokens exceeds context {}",
            request_id,
            estimated_tokens,
            max_tokens,
            requested_total,
            state.max_context_tokens,
        );
        let _ = send_json(
            stream,
            413,
            &format!(
                r#"{{"error":{{"message":"Requested prompt and output total {} tokens ({} prompt + {} max output), exceeding context capacity of {} tokens","type":"invalid_request_error","code":"context_length_exceeded","prompt_tokens":{},"max_output_tokens":{},"max_context_tokens":{}}}}}"#,
                requested_total,
                estimated_tokens,
                max_tokens,
                state.max_context_tokens,
                estimated_tokens,
                max_tokens,
                state.max_context_tokens,
            ),
        );
        return;
    }

    // ── Evict soft HCS before prefill to free VRAM ──
    crate::vram_monitor::report_event("evict_start");
    let t_evict = Instant::now();
    let prefill_entry_floor_bytes =
        match prefill_entry_floor_bytes_for_server(&state.rust_prefill, estimated_tokens) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::error!(
                    "Prefill engine floor unavailable before HCS eviction: {}",
                    e
                );
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Prefill engine floor unavailable: {}"}}"#,
                        json_escape(&e)
                    ),
                );
                return;
            }
        };
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (_evicted, _freed_mb) = store_for_evict
        .hcs_evict_for_prefill_with_engine_floor(estimated_tokens, prefill_entry_floor_bytes);
    // NOTE: aux GPU never does prefill, so no eviction needed there
    let evict_ms = t_evict.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("evict_end");

    // ── Snapshot VRAM before prefill ──
    log::info!(
        "VRAM before prefill: {} MB free",
        store_for_evict.query_vram_free_mb()
    );

    // ── Prefill: Rust path (text-only token IDs, or image embeddings handoff) ──
    crate::vram_monitor::report_event("prefill_start");
    crate::vram_monitor::reset_request_lows();
    let t_prefill_gil = Instant::now();

    let mut prompt_hcs_snapshot: Option<(Vec<u64>, usize, usize, usize)> = None;
    let mut chat_debug_input_token_ids: Option<Vec<u32>> = None;
    let mut request_token_ids: Vec<u32> = Vec::new();
    let mut cache_sequence_start = 0usize;
    let mut cache_request_eligible = false;
    // The restore source may extend beyond a rewindable shared KV boundary.
    // Keep it separate from the incremental writeback base, which must be an
    // exact prefix of the newly published branch.
    let mut cache_restore_snapshot_id = None;
    let mut cache_base_snapshot_id = None;
    let mut cache_ram_restore: Option<(usize, f64)> = None;
    let mut cache_miss_reason: Option<SessionCacheMissReason> = None;
    let mut cache_active_device_restore = false;
    let mut cache_active_stage_restore = false;
    let mut cache_prefill_stage_required = false;
    let mut cache_stable_boundary_tokens: Option<Vec<u32>> = None;
    let mut cache_image_miss_recorded = false;
    let mut cache_boundary_reservation = None;
    let mut cache_boundary_incremental_base: Option<Arc<crate::session_cache::SessionSnapshot>> =
        None;
    let mut cache_session_lease: Option<SessionRequestLease> = None;
    let mut cache_stage_reservation_extended = false;
    let mut cache_ram_boundary_checkpoint_armed = false;
    let mut pending_boundary_snapshot: Option<PendingBoundarySnapshot> = None;
    let prefill_result: Result<
        (usize, usize, Vec<usize>, bool, Option<serde_json::Value>),
        String,
    > = {
        // Image/recurrent multimodal state does not yet have an exact host
        // snapshot contract. Record that limitation before Python image
        // preparation so even a fail-closed setup error remains observable.
        if state.session_cache.enabled && request_prefix_cache && has_images {
            state
                .session_cache
                .metrics
                .record_miss(SessionCacheMissReason::ImageInput);
            cache_image_miss_recorded = true;
            log::info!(
                "Request {} prefix cache miss: image_input_uncacheable",
                request_id,
            );
        }
        // ── Rust prefill: text requests stay token-id only; image requests
        // build BF16 inputs_embeds once before the Rust prefill run.
        let mut multimodal_inputs: Option<MultimodalPrefillInputs> = None;
        let token_ids: Vec<u32> = if has_images {
            let built = Python::with_gil(|py| -> Result<MultimodalPrefillInputs, String> {
                let obj = state
                    .py_model
                    .call_method1(
                        py,
                        "build_multimodal_prefill_inputs",
                        (messages_json.as_str(), rendered.as_str()),
                    )
                    .map_err(|e| format!("image prefill setup failed: {}", e))?;
                let mm = obj.bind(py);
                let token_ids: Vec<u32> = mm
                    .get_item("token_ids")
                    .map_err(|e| format!("image prefill token_ids read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill token_ids extract failed: {}", e))?;
                let inputs_embeds_ptr: u64 = mm
                    .get_item("inputs_embeds_ptr")
                    .map_err(|e| format!("image prefill inputs_embeds_ptr read failed: {}", e))?
                    .extract()
                    .map_err(|e| {
                        format!("image prefill inputs_embeds_ptr extract failed: {}", e)
                    })?;
                let mrope_cos_ptr: u64 = mm
                    .get_item("mrope_cos_ptr")
                    .map_err(|e| format!("image prefill mrope_cos_ptr read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill mrope_cos_ptr extract failed: {}", e))?;
                let mrope_sin_ptr: u64 = mm
                    .get_item("mrope_sin_ptr")
                    .map_err(|e| format!("image prefill mrope_sin_ptr read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill mrope_sin_ptr extract failed: {}", e))?;
                let mrope_half_dim: usize = mm
                    .get_item("mrope_half_dim")
                    .map_err(|e| format!("image prefill mrope_half_dim read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill mrope_half_dim extract failed: {}", e))?;
                let rope_delta: i32 = mm
                    .get_item("rope_delta")
                    .map_err(|e| format!("image prefill rope_delta read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill rope_delta extract failed: {}", e))?;
                let vision_block_ids_ptr: u64 = mm
                    .get_item("vision_block_ids_ptr")
                    .map_err(|e| format!("image prefill vision_block_ids_ptr read failed: {}", e))?
                    .extract()
                    .map_err(|e| {
                        format!("image prefill vision_block_ids_ptr extract failed: {}", e)
                    })?;
                let vision_visible_left_ptr: u64 = mm
                    .get_item("vision_visible_left_ptr")
                    .map_err(|e| {
                        format!("image prefill vision_visible_left_ptr read failed: {}", e)
                    })?
                    .extract()
                    .map_err(|e| {
                        format!(
                            "image prefill vision_visible_left_ptr extract failed: {}",
                            e
                        )
                    })?;
                let vision_visible_right_ptr: u64 = mm
                    .get_item("vision_visible_right_ptr")
                    .map_err(|e| {
                        format!("image prefill vision_visible_right_ptr read failed: {}", e)
                    })?
                    .extract()
                    .map_err(|e| {
                        format!(
                            "image prefill vision_visible_right_ptr extract failed: {}",
                            e
                        )
                    })?;
                let vision_max_tokens: usize = mm
                    .get_item("vision_max_tokens")
                    .map_err(|e| format!("image prefill vision_max_tokens read failed: {}", e))?
                    .extract()
                    .map_err(|e| {
                        format!("image prefill vision_max_tokens extract failed: {}", e)
                    })?;
                let image_count: usize = mm
                    .get_item("image_count")
                    .map_err(|e| format!("image prefill image_count read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill image_count extract failed: {}", e))?;
                let image_tokens: usize = mm
                    .get_item("image_tokens")
                    .map_err(|e| format!("image prefill image_tokens read failed: {}", e))?
                    .extract()
                    .map_err(|e| format!("image prefill image_tokens extract failed: {}", e))?;
                Ok(MultimodalPrefillInputs {
                    token_ids,
                    inputs_embeds_ptr,
                    mrope_cos_ptr,
                    mrope_sin_ptr,
                    mrope_half_dim,
                    rope_delta,
                    vision_block_ids_ptr,
                    vision_visible_left_ptr,
                    vision_visible_right_ptr,
                    vision_max_tokens,
                    image_count,
                    image_tokens,
                })
            });
            match built {
                Ok(mm) => {
                    log::info!(
                        "Request {}: image prefill inputs ready: images={} image_tokens={} prompt_tokens={} rope_delta={}",
                        request_id,
                        mm.image_count,
                        mm.image_tokens,
                        mm.token_ids.len(),
                        mm.rope_delta,
                    );
                    let ids = mm.token_ids.clone();
                    multimodal_inputs = Some(mm);
                    ids
                }
                Err(e) => {
                    if let Some(body) = image_vram_error_body(&e) {
                        let _ = send_json(stream, 507, &body);
                    } else {
                        let _ = send_json(
                            stream,
                            500,
                            &format!(r#"{{"error":"{}"}}"#, json_escape(&e)),
                        );
                    }
                    return;
                }
            }
        } else {
            match state.tokenizer.encode(rendered.as_str(), false) {
                Ok(e) => e.get_ids().to_vec(),
                Err(e) => {
                    let _ = send_json(stream, 500, &format!(r#"{{"error":"Tokenize: {}"}}"#, e));
                    return;
                }
            }
        };
        request_token_ids.clone_from(&token_ids);
        let multi_gpu_pending = session_cache_multi_gpu_pending(
            state.session_cache.enabled,
            &state.aux_gpu_store_addrs,
        );
        cache_request_eligible = state.session_cache.enabled
            && request_prefix_cache
            && !has_images
            && !multi_gpu_pending
            && {
                let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
                !store.speculative_decode_enabled_rust()
            };
        if cache_request_eligible {
            if let Some(stable_rendered) = stable_rendered.as_ref() {
                match state.tokenizer.encode(stable_rendered.as_str(), false) {
                    Ok(encoding) => {
                        let stable_ids = encoding.get_ids();
                        let matched_boundary =
                            crate::session_cache::common_token_prefix(stable_ids, &token_ids);
                        if matched_boundary == token_ids.len() && matched_boundary > 0 {
                            cache_stable_boundary_tokens = Some(token_ids.clone());
                            log::info!(
                                "Request {} prefix cache terminal template-stable boundary: tokens={}",
                                request_id,
                                matched_boundary,
                            );
                        } else if let Some(boundary_alignment) = unsafe {
                            (&*(state.gpu_store_addr as *const GpuDecodeStore))
                                .exact_mid_prefill_boundary_alignment_rust()
                        } {
                            let boundary = crate::session_cache::align_exact_boundary_down(
                                matched_boundary,
                                boundary_alignment,
                            )
                            .unwrap_or(0);
                            if boundary > 0 && boundary < token_ids.len() {
                                cache_stable_boundary_tokens = Some(token_ids[..boundary].to_vec());
                                log::info!(
                                    "Request {} prefix cache stable boundary: tokens={} matched_tokens={} alignment={} stable_render_tokens={} full_prompt_tokens={}",
                                    request_id,
                                    boundary,
                                    matched_boundary,
                                    boundary_alignment,
                                    stable_ids.len(),
                                    token_ids.len(),
                                );
                            } else {
                                log::info!(
                                    "Request {} prefix cache miss: no_internal_stable_template_boundary matched={} aligned={} alignment={} stable_render_tokens={} full_prompt_tokens={}",
                                    request_id,
                                    matched_boundary,
                                    boundary,
                                    boundary_alignment,
                                    stable_ids.len(),
                                    token_ids.len(),
                                );
                            }
                        } else {
                            increment_metric(
                                &mut state.session_cache.metrics.mid_prefill_boundary_skipped,
                                "mid_prefill_boundary_skipped",
                            );
                            // Preserve the ordinary model-math path for lossy
                            // stage formats: snapshot the exact terminal
                            // prefill state after the normal chunk plan rather
                            // than introducing a synthetic internal split.
                            cache_stable_boundary_tokens = Some(token_ids.clone());
                            log::info!(
                                "Request {} prefix cache internal boundary skipped: runtime prefill path is not exact across synthetic chunk splits; terminal prefill boundary planned at {} tokens",
                                request_id,
                                token_ids.len(),
                            );
                        }
                    }
                    Err(error) => log::error!(
                        "Request {} prefix cache miss: stable_template_tokenize_failed error={}",
                        request_id,
                        error,
                    ),
                }
            }
        }
        if state.session_cache.enabled && !cache_request_eligible {
            let (reason, miss_reason) = if !request_prefix_cache {
                ("request_disabled", SessionCacheMissReason::RequestDisabled)
            } else if has_images {
                (
                    "image_input_uncacheable",
                    SessionCacheMissReason::ImageInput,
                )
            } else if multi_gpu_pending {
                (
                    "multi_gpu_active_handoff_pending",
                    SessionCacheMissReason::MultiGpuPending,
                )
            } else {
                (
                    "speculative_decode_uncacheable",
                    SessionCacheMissReason::SpeculativeDecode,
                )
            };
            if !(cache_image_miss_recorded
                && matches!(miss_reason, SessionCacheMissReason::ImageInput))
            {
                state.session_cache.metrics.record_miss(miss_reason);
            }
            log::info!("Request {} prefix cache miss: {}", request_id, reason);
            invalidate_active_sequence(state, reason);
        }
        if cache_request_eligible {
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let active_plan = state.session_cache.active.as_ref().map(|active| {
                crate::session_cache::plan_active_prefix(
                    &active.consumed_token_ids,
                    &token_ids,
                    store.sequence_state_has_non_rewindable_rust(),
                )
            });
            let active_reusable_tokens = match active_plan {
                Some(crate::session_cache::ActivePrefixPlan::Append { matched_tokens })
                | Some(crate::session_cache::ActivePrefixPlan::TruncateKvAndAppend {
                    matched_tokens,
                }) => matched_tokens,
                _ => 0,
            };

            // Active GPU state and inactive RAM snapshots must compete on the
            // longest exact prefix. A tiny shared chat-template prefix in the
            // active conversation must never hide a much longer RAM match.
            let compatibility =
                state.session_cache.compatibility.clone().ok_or_else(|| {
                    "session cache compatibility signature is unavailable".to_string()
                });
            match compatibility.and_then(|compatibility| {
                let ram_store = state
                    .session_cache
                    .ram_store
                    .as_mut()
                    .ok_or_else(|| "RAM session store is unavailable".to_string())?;
                if store.sequence_state_has_non_rewindable_rust() {
                    ram_store.longest_proper_prefix(&token_ids, &compatibility)
                } else {
                    ram_store.longest_rewindable_prefix(&token_ids, &compatibility)
                }
            }) {
                Ok(crate::session_cache::PrefixLookupResult::Hit {
                    snapshot_id,
                    matched_tokens,
                }) if matched_tokens < token_ids.len()
                    && crate::session_cache::ram_prefix_is_longer(
                        active_reusable_tokens,
                        matched_tokens,
                    ) =>
                {
                    let snapshot_meta = match state.session_cache.ram_store.as_mut() {
                        Some(ram_store) => match ram_store.get(snapshot_id) {
                            Ok(Some(snapshot)) => Some((
                                snapshot.memory_cost_bytes(),
                                snapshot.consumed_token_ids.len(),
                            )),
                            Ok(None) => {
                                log::info!("Request {} prefix cache miss: evicted", request_id);
                                cache_miss_reason = Some(SessionCacheMissReason::Evicted);
                                None
                            }
                            Err(error) => {
                                log::error!(
                                    "Request {} prefix cache snapshot lookup failed: {}",
                                    request_id,
                                    error,
                                );
                                cache_miss_reason = Some(SessionCacheMissReason::RestoreFailed);
                                None
                            }
                        },
                        None => {
                            log::error!("Request {} RAM session store is unavailable", request_id);
                            cache_miss_reason = Some(SessionCacheMissReason::RestoreFailed);
                            None
                        }
                    };
                    let restore_ms = snapshot_meta.map(|(bytes, _)| bytes).and_then(|bytes| {
                        predicted_restore_ms(&state.session_cache.restore_samples, bytes)
                    });
                    let avoided_ms = predicted_avoided_prefill_ms(
                        &state.session_cache.prefill_samples,
                        token_ids.len(),
                        matched_tokens,
                    );
                    if restore_ms
                        .zip(avoided_ms)
                        .is_some_and(|(restore, avoided)| restore < avoided)
                    {
                        match restore_snapshot_by_id(state, snapshot_id) {
                            Ok((restored_tokens, bytes, measured_ms)) => {
                                if restored_tokens.len() < matched_tokens
                                    || restored_tokens[..matched_tokens]
                                        != token_ids[..matched_tokens]
                                {
                                    log::error!(
                                        "Request {} prefix cache restore exact-token verification failed",
                                        request_id
                                    );
                                    cache_miss_reason = Some(SessionCacheMissReason::RestoreFailed);
                                } else {
                                    let rewound = restored_tokens.len() > matched_tokens;
                                    if rewound {
                                        for address in std::iter::once(state.gpu_store_addr)
                                            .chain(state.aux_gpu_store_addrs.iter().copied())
                                        {
                                            let restored_store =
                                                unsafe { &mut *(address as *mut GpuDecodeStore) };
                                            restored_store.set_kv_position_rust(matched_tokens);
                                        }
                                    }
                                    cache_sequence_start = matched_tokens;
                                    cache_restore_snapshot_id = Some(snapshot_id);
                                    cache_base_snapshot_id = (!rewound).then_some(snapshot_id);
                                    cache_ram_restore = Some((bytes, measured_ms));
                                    cache_miss_reason = None;
                                    log::info!(
                                            "Request {} prefix cache RAM hit: reused={} suffix={} restored_tokens={} rewound={} bytes={} restore_ms={:.3}",
                                            request_id,
                                            matched_tokens,
                                            token_ids.len().saturating_sub(matched_tokens),
                                            restored_tokens.len(),
                                            rewound,
                                            bytes,
                                            measured_ms,
                                        );
                                }
                            }
                            Err(error) => {
                                cache_miss_reason = Some(SessionCacheMissReason::RestoreFailed);
                                log::error!(
                                    "Request {} prefix cache restore failed; using full prefill: {}",
                                    request_id,
                                    error,
                                );
                            }
                        }
                    } else {
                        cache_miss_reason = Some(SessionCacheMissReason::RestoreNotWorthIt);
                        log::info!(
                                "Request {} prefix cache miss: restore_not_worth_it predicted_restore_ms={:?} predicted_avoided_prefill_ms={:?}",
                                request_id,
                                restore_ms,
                                avoided_ms,
                            );
                    }
                }
                Ok(crate::session_cache::PrefixLookupResult::Hit {
                    snapshot_id,
                    matched_tokens,
                }) => log::info!(
                    "Request {} prefix cache RAM candidate not selected: snapshot={:?} matched={} active_reusable={} request_tokens={}",
                    request_id,
                    snapshot_id,
                    matched_tokens,
                    active_reusable_tokens,
                    token_ids.len(),
                ),
                Ok(crate::session_cache::PrefixLookupResult::SignatureMismatch {
                    matched_tokens,
                }) => {
                    cache_miss_reason = Some(SessionCacheMissReason::SignatureMismatch);
                    log::info!(
                        "Request {} prefix cache miss: signature_mismatch matched={}",
                        request_id,
                        matched_tokens,
                    );
                }
                Ok(crate::session_cache::PrefixLookupResult::NoMatch) => {
                    cache_miss_reason.get_or_insert(SessionCacheMissReason::NoMatch);
                    log::info!("Request {} prefix cache miss: no_match", request_id)
                }
                Err(error) => {
                    cache_miss_reason = Some(SessionCacheMissReason::RestoreFailed);
                    log::error!(
                        "Request {} prefix cache lookup failed; using full prefill: {}",
                        request_id,
                        error,
                    );
                }
            }

            if cache_sequence_start == 0 {
                match active_plan {
                    Some(crate::session_cache::ActivePrefixPlan::Append { matched_tokens }) => {
                        cache_miss_reason = None;
                        cache_sequence_start = matched_tokens;
                        cache_base_snapshot_id = state
                            .session_cache
                            .active
                            .as_ref()
                            .map(|active| active.snapshot_id);
                        cache_active_device_restore = active_plan_requires_device_checkpoint(
                            crate::session_cache::ActivePrefixPlan::Append { matched_tokens },
                            state
                                .session_cache
                                .active
                                .as_ref()
                                .is_some_and(|active| active.requires_device_checkpoint),
                        );
                        cache_active_stage_restore = active_plan_requires_stage_restore(
                            crate::session_cache::ActivePrefixPlan::Append { matched_tokens },
                            state
                                .session_cache
                                .active
                                .as_ref()
                                .is_some_and(|active| active.requires_device_checkpoint),
                        );
                        if !cache_active_device_restore {
                            store.set_kv_position_rust(matched_tokens);
                        }
                        log::info!(
                            "Request {} prefix cache active hit: reused={} suffix={}",
                            request_id,
                            matched_tokens,
                            token_ids.len().saturating_sub(matched_tokens),
                        );
                    }
                    Some(crate::session_cache::ActivePrefixPlan::TruncateKvAndAppend {
                        matched_tokens,
                    }) => {
                        cache_miss_reason = None;
                        cache_sequence_start = matched_tokens;
                        cache_active_device_restore = active_plan_requires_device_checkpoint(
                            crate::session_cache::ActivePrefixPlan::TruncateKvAndAppend {
                                matched_tokens,
                            },
                            state
                                .session_cache
                                .active
                                .as_ref()
                                .is_some_and(|active| active.requires_device_checkpoint),
                        );
                        cache_active_stage_restore = active_plan_requires_stage_restore(
                            crate::session_cache::ActivePrefixPlan::TruncateKvAndAppend {
                                matched_tokens,
                            },
                            state
                                .session_cache
                                .active
                                .as_ref()
                                .is_some_and(|active| active.requires_device_checkpoint),
                        );
                        if !cache_active_device_restore {
                            store.set_kv_position_rust(matched_tokens);
                        }
                        log::info!(
                            "Request {} prefix cache active KV truncation: reused={} suffix={}",
                            request_id,
                            matched_tokens,
                            token_ids.len().saturating_sub(matched_tokens),
                        );
                    }
                    Some(crate::session_cache::ActivePrefixPlan::RequiresBoundarySnapshot {
                        matched_tokens,
                    }) => {
                        cache_miss_reason = Some(SessionCacheMissReason::Divergence);
                        log::info!(
                            "Request {} prefix cache miss: divergence_requires_exact_boundary_snapshot matched={}",
                            request_id,
                            matched_tokens,
                        );
                    }
                    Some(crate::session_cache::ActivePrefixPlan::NoReusablePrefix) => {
                        cache_miss_reason.get_or_insert(SessionCacheMissReason::NoMatch);
                        log::info!("Request {} prefix cache miss: no_match", request_id)
                    }
                    Some(crate::session_cache::ActivePrefixPlan::NoSuffixToCompute) => {
                        cache_miss_reason = Some(SessionCacheMissReason::NoSuffix);
                        log::info!(
                            "Request {} prefix cache miss: no_suffix_to_compute",
                            request_id,
                        );
                    }
                    None => {
                        cache_miss_reason.get_or_insert(SessionCacheMissReason::NoMatch);
                        log::info!("Request {} prefix cache miss: no_active_state", request_id)
                    }
                }
            }
            let session_lock_key = cache_restore_snapshot_id
                .or(cache_base_snapshot_id)
                .or_else(|| {
                    (cache_sequence_start > 0)
                        .then(|| {
                            state
                                .session_cache
                                .active
                                .as_ref()
                                .map(|active| active.snapshot_id)
                        })
                        .flatten()
                })
                .map(SessionLockKey::Snapshot)
                .unwrap_or_else(|| {
                    let exact_boundary = cache_stable_boundary_tokens
                        .as_deref()
                        .unwrap_or(token_ids.as_slice());
                    SessionLockKey::ExactBoundary(Arc::from(exact_boundary))
                });
            match state.session_cache.session_locks.acquire(session_lock_key) {
                Ok(lease) => cache_session_lease = Some(lease),
                Err(error) => {
                    let _ = send_json(
                        stream,
                        500,
                        &format!(
                            r#"{{"error":"Session scheduling failed closed: {}"}}"#,
                            json_escape(&error)
                        ),
                    );
                    return;
                }
            }
            // GPU state is mutated below. It is republished only after decode
            // and response delivery succeed.
            state.session_cache.active = None;
        }
        let snapshot_boundary = cache_stable_boundary_tokens
            .as_ref()
            .map(Vec::len)
            .filter(|&boundary| boundary > cache_sequence_start && boundary <= token_ids.len());
        if let Some(boundary) = snapshot_boundary {
            match session_snapshot_reservation_bytes(state, boundary).and_then(|required_bytes| {
                state
                    .session_cache
                    .ram_store
                    .as_mut()
                    .ok_or_else(|| "RAM session store is unavailable".to_string())?
                    .reserve_protecting(
                        required_bytes,
                        cache_base_snapshot_id
                            .as_ref()
                            .map_or(&[], std::slice::from_ref),
                    )
            }) {
                Ok(reservation) => cache_boundary_reservation = Some(reservation),
                Err(error) => {
                    cache_stable_boundary_tokens = None;
                    log::info!(
                        "Request {} prefix cache miss: stable_boundary_reservation_failed boundary={} error={}",
                        request_id,
                        boundary,
                        error,
                    );
                }
            }
        }
        let capture_boundary = internal_capture_boundary(
            cache_boundary_reservation
                .is_some()
                .then_some(snapshot_boundary)
                .flatten(),
            cache_sequence_start,
            token_ids.len(),
        );
        if cache_boundary_reservation.is_some() {
            if let Some(base_id) = cache_base_snapshot_id {
                let base_result = state
                    .session_cache
                    .ram_store
                    .as_mut()
                    .ok_or_else(|| "RAM session store is unavailable".to_string())
                    .and_then(|store| {
                        store.get(base_id)?.ok_or_else(|| {
                            "protected incremental boundary base was evicted".to_string()
                        })
                    })
                    .and_then(|base| {
                        let boundary_tokens =
                            cache_stable_boundary_tokens.as_ref().ok_or_else(|| {
                                "incremental boundary has no exact token IDs".to_string()
                            })?;
                        if base.consumed_token_ids.len() > boundary_tokens.len()
                            || base.consumed_token_ids.as_slice()
                                != &boundary_tokens[..base.consumed_token_ids.len()]
                        {
                            return Err("incremental boundary base is not an exact token prefix"
                                .to_string());
                        }
                        Ok(base)
                    });
                match base_result {
                    Ok(base) => cache_boundary_incremental_base = Some(base),
                    Err(error) => {
                        cancel_boundary_reservation(
                            &mut state.session_cache,
                            &mut cache_boundary_reservation,
                        );
                        if let Ok(mut engine_guard) = state.rust_prefill.lock() {
                            if let Some(engine) = engine_guard.as_mut() {
                                engine.discard_active_stage_state();
                            }
                        }
                        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                        store.discard_active_sequence_checkpoint_rust();
                        let _ = send_json(
                            stream,
                            500,
                            &format!(
                                r#"{{"error":"Incremental prefix-cache boundary failed closed: {}"}}"#,
                                json_escape(&error)
                            ),
                        );
                        return;
                    }
                }
            }
        }
        if debug_first_token_boundary {
            chat_debug_input_token_ids = Some(token_ids.clone());
        }
        let mut engine_guard = state.rust_prefill.lock().unwrap();
        let engine = engine_guard.as_mut().unwrap();
        if cache_active_stage_restore {
            if let Err(error) = engine.arm_active_stage_continuation(cache_sequence_start) {
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                store.discard_active_sequence_checkpoint_rust();
                engine.discard_active_stage_state();
                cancel_boundary_reservation(
                    &mut state.session_cache,
                    &mut cache_boundary_reservation,
                );
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Active prefix-cache stage restore failed closed: {}"}}"#,
                        json_escape(&error)
                    ),
                );
                return;
            }
        } else {
            engine.discard_active_stage_state();
        }
        if cache_active_device_restore {
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            match store.restore_active_sequence_checkpoint_rust(cache_sequence_start) {
                Ok(restore_ms) => log::info!(
                    "Request {} prefix cache active device restore: reused={} d2d_restore_ms={:.3}",
                    request_id,
                    cache_sequence_start,
                    restore_ms,
                ),
                Err(error) => {
                    store.discard_active_sequence_checkpoint_rust();
                    engine.discard_active_stage_state();
                    cancel_boundary_reservation(
                        &mut state.session_cache,
                        &mut cache_boundary_reservation,
                    );
                    let _ = send_json(
                        stream,
                        500,
                        &format!(
                            r#"{{"error":"Active prefix-cache device restore failed closed: {}"}}"#,
                            json_escape(&error)
                        ),
                    );
                    return;
                }
            }
        } else {
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            store.discard_active_sequence_checkpoint_rust();
        }
        // Warmup/calibration calls disable prefill pinning through the shared engine.
        // Normal request prefill must not inherit that one-shot state.
        engine.set_prefill_pinning_disabled(false);

        // Update HCS snapshot so prefill can use GPU-resident experts directly
        {
            let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
            let (cache_fast, ne) = store.export_hcs_snapshot();
            engine.update_hcs_snapshot(cache_fast, ne);
        }

        let kv_max_seq = engine.kv_max_seq;
        let kv_overflow = token_ids.len() > kv_max_seq;

        let _has_hqq_runtime_slots = {
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            match prepare_store_for_rust_prefill(store, engine, token_ids.len()) {
                Ok(has_hqq) => has_hqq,
                Err(e) => {
                    engine.clear_external_prefill_inputs();
                    engine.discard_active_stage_state();
                    store.discard_active_sequence_checkpoint_rust();
                    cancel_boundary_reservation(
                        &mut state.session_cache,
                        &mut cache_boundary_reservation,
                    );
                    if has_images {
                        Python::with_gil(|py| {
                            let _ = state
                                .py_model
                                .call_method0(py, "clear_multimodal_prefill_inputs");
                        });
                    }
                    let _ = send_json(
                        stream,
                        500,
                        &format!(r#"{{"error":"Prefill prepare failed: {}"}}"#, e),
                    );
                    return;
                }
            }
        };

        // Image visibility changes only the request-scoped V4 sparse-index
        // geometry, so publish it before dynamic scratch sizing.
        if let Some(mm) = multimodal_inputs.as_ref() {
            engine.set_external_prefill_inputs(
                mm.inputs_embeds_ptr,
                mm.mrope_cos_ptr,
                mm.mrope_sin_ptr,
                mm.mrope_half_dim,
                mm.vision_block_ids_ptr,
                mm.vision_visible_left_ptr,
                mm.vision_visible_right_ptr,
                mm.vision_max_tokens,
            );
        } else {
            engine.clear_external_prefill_inputs();
        }

        engine.set_prefill_hcs_guard_store_addr(state.gpu_store_addr);

        let mut retry_cap: Option<usize> = None;
        let mut retry_attempt = 0usize;
        let result = loop {
            engine.set_prefill_runtime_chunk_cap(retry_cap);

            // Dynamically allocate scratch sized for this prompt.
            // Scratch contains prompt-wide state for several attention
            // backends even when only a suffix is computed. Size it from the
            // exact logical sequence length; run_prefill_from below still
            // executes only the uncached suffix.
            let prepare_result = if cache_request_eligible {
                engine.prepare_for_prefill_session_cache_exact(token_ids.len())
            } else {
                engine.prepare_for_prefill(token_ids.len())
            };
            if let Err(e) = prepare_result {
                engine.clear_external_prefill_inputs();
                engine.clear_prefill_hcs_guard_store_addr();
                engine.set_optional_pinning_budget_mb(None);
                engine.clear_prefill_runtime_chunk_cap();
                engine.discard_active_stage_state();
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                store.discard_active_sequence_checkpoint_rust();
                let _ = store.prepare_runtime_for_decode_rust();
                cancel_boundary_reservation(
                    &mut state.session_cache,
                    &mut cache_boundary_reservation,
                );
                if has_images {
                    Python::with_gil(|py| {
                        let _ = state
                            .py_model
                            .call_method0(py, "clear_multimodal_prefill_inputs");
                    });
                }
                if has_images {
                    let body = format!(
                        r#"{{"error":{{"message":"VRAM is too constrained for this image request. Multimodal prefill scratch allocation failed: {}","type":"insufficient_resources","code":"insufficient_vram"}}}}"#,
                        json_escape(&e)
                    );
                    let _ = send_json(stream, 507, &body);
                    return;
                }
                let _ = send_json(
                    stream,
                    500,
                    &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e),
                );
                return;
            }
            cache_prefill_stage_required = match engine.stage_exact_snapshot_cost_estimate(1) {
                Ok(bytes) => bytes > 0,
                Err(error) => {
                    break Err(format!("inspect live prefill-stage state: {error}"));
                }
            };

            if cache_sequence_start > 0 && cache_ram_restore.is_some() {
                let stage_restore = cache_restore_snapshot_id
                    .ok_or_else(|| "RAM-restored continuation has no base snapshot ID".to_string())
                    .and_then(|snapshot_id| {
                        let snapshot = state
                            .session_cache
                            .ram_store
                            .as_mut()
                            .ok_or_else(|| "RAM session store is unavailable".to_string())?
                            .get(snapshot_id)?
                            .ok_or_else(|| {
                                "RAM-restored continuation snapshot was evicted".to_string()
                            })?;
                        engine.restore_stage_exact_sequence_state(
                            snapshot.as_ref(),
                            cache_sequence_start,
                        )
                    });
                match stage_restore {
                    Ok(stage_ms) => {
                        if let Some((bytes, decode_restore_ms)) = cache_ram_restore.as_mut() {
                            *decode_restore_ms += stage_ms;
                            state
                                .session_cache
                                .restore_samples
                                .push((*bytes, *decode_restore_ms));
                            log::info!(
                                "Request {} prefix cache exact stage restore: stage_ms={:.3} total_restore_ms={:.3}",
                                request_id,
                                stage_ms,
                                *decode_restore_ms,
                            );
                        }
                        let reuses_exact_stable_boundary = cache_prefill_stage_required
                            && cache_base_snapshot_id.is_some()
                            && cache_stable_boundary_tokens
                                .as_ref()
                                .is_some_and(|tokens| tokens.len() == cache_sequence_start);
                        if reuses_exact_stable_boundary && !cache_ram_boundary_checkpoint_armed {
                            let store =
                                unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                            match store.capture_pending_active_sequence_checkpoint_rust(
                                cache_sequence_start,
                            ) {
                                Ok((bytes, capture_ms)) => {
                                    cache_ram_boundary_checkpoint_armed = true;
                                    log::info!(
                                        "Request {} prefix cache RAM boundary device checkpoint armed: tokens={} bytes={} d2d_capture_ms={:.3}",
                                        request_id,
                                        cache_sequence_start,
                                        bytes,
                                        capture_ms,
                                    );
                                }
                                Err(error) => {
                                    break Err(format!(
                                        "arm RAM-restored active boundary checkpoint: {error}"
                                    ));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        log::info!(
                            "Request {} prefix cache miss: exact_stage_restore_failed error={}",
                            request_id,
                            error,
                        );
                        cache_sequence_start = 0;
                        cache_base_snapshot_id = None;
                        cache_ram_restore = None;
                        cache_miss_reason = Some(SessionCacheMissReason::RestoreFailed);
                    }
                }
            }

            if !cache_stage_reservation_extended {
                if let (Some(reservation), Some(boundary)) = (
                    cache_boundary_reservation,
                    cache_stable_boundary_tokens.as_ref().map(Vec::len),
                ) {
                    let extra_bytes = match engine.stage_exact_snapshot_cost_estimate(boundary) {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            break Err(format!(
                                "measure exact prefill-stage snapshot size: {error}"
                            ));
                        }
                    };
                    if extra_bytes > 0 {
                        let protected = cache_base_snapshot_id
                            .as_ref()
                            .map_or(&[][..], std::slice::from_ref);
                        let extension = match state.session_cache.ram_store.as_mut() {
                            Some(store) => {
                                store.extend_reservation(reservation, extra_bytes, protected)
                            }
                            None => Err("RAM session store is unavailable".to_string()),
                        };
                        if let Err(error) = extension {
                            if let Some(reservation_id) = cache_boundary_reservation.take() {
                                if let Some(ram_store) = state.session_cache.ram_store.as_mut() {
                                    if let Err(cancel_error) =
                                        ram_store.cancel_reservation(reservation_id)
                                    {
                                        log::error!(
                                        "Request {} failed to cancel exact-stage reservation: {}",
                                        request_id,
                                        cancel_error,
                                    );
                                    }
                                }
                            }
                            cache_stable_boundary_tokens = None;
                            log::info!(
                            "Request {} prefix cache miss: exact_stage_reservation_failed boundary={} additional_bytes={} error={}",
                            request_id,
                            boundary,
                            extra_bytes,
                            error,
                        );
                        } else {
                            cache_stage_reservation_extended = true;
                        }
                    } else {
                        cache_stage_reservation_extended = true;
                    }
                }
            }
            let pinning_budget_mb = {
                let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
                store.prefill_optional_pinning_budget_mb(
                    token_ids.len(),
                    engine.last_prepare_post_alloc_free_mb(),
                )
            };
            engine.set_optional_pinning_budget_mb(pinning_budget_mb);

            let suppress_tokens = {
                let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
                store.suppress_tokens_clone()
            };
            if let Some(mm) = multimodal_inputs.as_ref() {
                engine.set_external_prefill_inputs(
                    mm.inputs_embeds_ptr,
                    mm.mrope_cos_ptr,
                    mm.mrope_sin_ptr,
                    mm.mrope_half_dim,
                    mm.vision_block_ids_ptr,
                    mm.vision_visible_left_ptr,
                    mm.vision_visible_right_ptr,
                    mm.vision_max_tokens,
                );
            } else {
                engine.clear_external_prefill_inputs();
            }

            let attempt_result = match match (cache_sequence_start, capture_boundary) {
                (sequence_start, Some(boundary)) if sequence_start > 0 => engine
                    .run_prefill_continuation_capturing_boundary(
                        &token_ids[sequence_start..],
                        sequence_start,
                        boundary,
                        cache_boundary_incremental_base.as_deref(),
                        temperature,
                        &suppress_tokens,
                    ),
                (0, Some(boundary)) => engine.run_prefill_capturing_boundary(
                    &token_ids,
                    boundary,
                    cache_boundary_incremental_base.as_deref(),
                    temperature,
                    &suppress_tokens,
                ),
                (sequence_start, None) if sequence_start > 0 => engine
                    .run_prefill_continuation(
                        &token_ids[sequence_start..],
                        sequence_start,
                        temperature,
                        &suppress_tokens,
                    )
                    .map(|result| (result, None)),
                (_, None) => engine
                    .run_prefill(&token_ids, temperature, &suppress_tokens)
                    .map(|result| (result, None)),
                (_, Some(_)) => unreachable!(),
            } {
                Ok((r, capture)) => {
                    let terminal_capture_planned = capture.is_none()
                        && cache_boundary_reservation.is_some()
                        && cache_stable_boundary_tokens
                            .as_ref()
                            .is_some_and(|tokens| tokens.len() == r.prompt_len);
                    if terminal_capture_planned {
                        engine
                            .capture_completed_prefill_boundary(
                                r.prompt_len,
                                cache_sequence_start,
                                cache_boundary_incremental_base.as_deref(),
                            )
                            .map(|terminal| {
                                log::info!(
                                    "Request {} prefix cache terminal prefill boundary captured: tokens={} allocations={} save_ms={:.3} restore_ms={:.3} active_device_bytes={} active_device_ms={:.3}",
                                    request_id,
                                    terminal.token_count,
                                    terminal.state_blobs.len(),
                                    terminal.save_ms,
                                    terminal.restore_ms,
                                    terminal.active_device_checkpoint_bytes,
                                    terminal.active_device_checkpoint_ms,
                                );
                                (r, Some(terminal))
                            })
                    } else {
                        let finalize = if cache_sequence_start > 0 {
                            engine.finalize_stage_exact_prefill_kv_continuation(
                                r.prompt_len,
                                cache_sequence_start,
                            )
                        } else {
                            engine.finalize_stage_exact_prefill_kv(r.prompt_len)
                        };
                        finalize
                            .map(|()| (r, capture))
                            .map_err(|e| format!("KV stage export failed: {}", e))
                    }
                }
                Err(e) => Err(e),
            };

            match attempt_result {
                Ok(result) => break Ok(result),
                Err(e) => {
                    if cache_sequence_start > 0 {
                        break Err(format!(
                            "continuation prefill failed after mutating live sequence state; retry requires the committed boundary snapshot: {}",
                            e
                        ));
                    }
                    let current_chunk = engine.scratch.max_tokens;
                    let next_retry_cap = engine.cold_staging_retry_chunk_cap();
                    if let Some(next_cap) = next_retry_cap {
                        if next_cap < current_chunk && current_chunk > 128 {
                            retry_attempt += 1;
                            if let Some(failure) = engine.last_cold_staging_failure {
                                log::info!(
                                    "Retrying chat prefill with measured cold-staging chunk cap: attempt={} prompt_tokens={} failed_chunk={} requested_slots={} max_safe_slots={} free_before_mb={} safety_mb={} current_chunk={} next_chunk_cap={} error={}",
                                    retry_attempt,
                                    token_ids.len(),
                                    failure.chunk_tokens,
                                    failure.requested_slots,
                                    failure.max_safe_slots,
                                    failure.free_before_mb,
                                    failure.safety_mb,
                                    current_chunk,
                                    next_cap,
                                    e,
                                );
                            } else {
                                log::info!(
                                    "Retrying chat prefill with measured cold-staging chunk cap: attempt={} prompt_tokens={} current_chunk={} next_chunk_cap={} error={}",
                                    retry_attempt,
                                    token_ids.len(),
                                    current_chunk,
                                    next_cap,
                                    e,
                                );
                            }
                            engine.set_optional_pinning_budget_mb(None);
                            if let Err(release_err) = engine.release_scratch() {
                                log::error!(
                                    "Failed to release scratch before chat prefill retry: {}",
                                    release_err
                                );
                                abort_if_cuda_context_poisoned(
                                    "chat retry release_scratch",
                                    &release_err,
                                );
                                break Err(release_err);
                            }
                            engine.clear_external_prefill_inputs();
                            retry_cap = Some(next_cap);
                            continue;
                        }
                    }
                    break Err(e);
                }
            }
        };

        prompt_hcs_snapshot = engine.prompt_hcs_shadow_snapshot();

        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let primary_device = store.device_ordinal();
        let free_now_mb = store.query_vram_free_mb();
        let prefill_min_free_mb = crate::vram_monitor::current_request_lows()
            .into_iter()
            .find(|(device, _)| *device == primary_device)
            .map(|(_, free_mb)| free_mb as usize)
            .unwrap_or(free_now_mb);
        engine.update_measured_prefill_runtime_overhead_mb(
            engine.last_prepare_post_alloc_free_mb(),
            prefill_min_free_mb,
        );

        let retain_active_stage = cache_prefill_stage_required
            && (cache_active_stage_restore
                || cache_ram_boundary_checkpoint_armed
                || result
                    .as_ref()
                    .ok()
                    .and_then(|(_, capture)| capture.as_ref())
                    .is_some());
        engine.retain_active_stage_after_prefill(retain_active_stage);

        // Release scratch to free VRAM for decode/HCS. The exact stage K/V is
        // retained only when a transactional active boundary was captured.
        if let Err(e) = engine.release_scratch() {
            log::error!("Failed to release scratch: {}", e);
            abort_if_cuda_context_poisoned("chat release_scratch", &e);
        }
        engine.clear_external_prefill_inputs();
        engine.clear_prefill_hcs_guard_store_addr();
        engine.set_optional_pinning_budget_mb(None);
        engine.clear_prefill_runtime_chunk_cap();
        if has_images {
            Python::with_gil(|py| {
                let _ = state
                    .py_model
                    .call_method0(py, "clear_multimodal_prefill_inputs");
            });
        }

        // Convert stop token strings to IDs, and always include model's EOS tokens
        let mut stop_ids: Vec<usize> = state.eos_stop_ids.clone();
        for s in &stop_tokens {
            if let Some(id) = state.tokenizer.token_to_id(s) {
                let id = id as usize;
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            }
        }
        if !enable_thinking {
            if let Some(id) = state.thinking_end_token {
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            }
        }

        match result {
            Ok((r, capture)) => {
                if let Some(capture) = capture {
                    match (
                        cache_boundary_reservation.take(),
                        cache_stable_boundary_tokens
                            .as_ref()
                            .filter(|tokens| tokens.len() == capture.token_count)
                            .cloned(),
                    ) {
                        (Some(reservation), Some(consumed_token_ids)) => {
                            pending_boundary_snapshot = Some(PendingBoundarySnapshot {
                                reservation,
                                consumed_token_ids,
                                capture,
                            });
                        }
                        (reservation, _) => {
                            cache_boundary_reservation = reservation;
                            cache_request_eligible = false;
                            log::error!(
                                "Request {} prefix cache boundary capture contract failed: capture_tokens={} stable_boundary={:?} reservation_present={}",
                                request_id,
                                capture.token_count,
                                cache_stable_boundary_tokens.as_ref().map(Vec::len),
                                cache_boundary_reservation.is_some(),
                            );
                        }
                    }
                }
                let debug_payload = if debug_first_token_boundary {
                    let debug_ids = chat_debug_input_token_ids.clone().unwrap_or_default();
                    let selected_token_text = state
                        .tokenizer
                        .decode(&[r.first_token], true)
                        .unwrap_or_default();
                    Some(serde_json::json!({
                        "schema": "krasis_chat_first_token_boundary_debug_v1",
                        "route": "/v1/chat/completions",
                        "rendered_prompt": rendered.as_str(),
                        "rendered_len": rendered.len(),
                        "enable_thinking": enable_thinking,
                        "has_tools": has_tools,
                        "input_token_count": debug_ids.len(),
                        "input_token_hash_fnv1a64": format!("0x{:016x}", fnv1a_token_hash(&debug_ids)),
                        "input_token_ids": debug_ids,
                        "selected_token_id": r.first_token as usize,
                        "selected_token_text": selected_token_text,
                        "first_token_logits": reference_logit_trace_json(
                            &engine.h_logits,
                            engine.h_logits.len(),
                            r.first_token as usize,
                            req_top_logprobs,
                        ),
                    }))
                } else {
                    None
                };
                // Set KV cache position on decode store so decode knows where to continue
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                if let Err(e) = restore_store_after_rust_prefill(store, r.prompt_len) {
                    log::error!("Failed to restore decode runtime after prefill: {}", e);
                }
                store.set_rope_position_delta(
                    multimodal_inputs
                        .as_ref()
                        .map(|mm| mm.rope_delta)
                        .unwrap_or(0),
                );
                Ok((
                    r.first_token as usize,
                    r.prompt_len,
                    stop_ids,
                    kv_overflow,
                    debug_payload,
                ))
            }
            Err(e) => {
                engine.clear_external_prefill_inputs();
                if has_images {
                    Python::with_gil(|py| {
                        let _ = state
                            .py_model
                            .call_method0(py, "clear_multimodal_prefill_inputs");
                    });
                }
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                store.set_rope_position_delta(0);
                let _ = store.prepare_runtime_for_decode_rust();
                Err(e)
            }
        }
    };
    // Boundary capture is complete. Release the old snapshot lease before
    // decode/post-generation admission so normal LRU policy can reclaim it.
    drop(cache_boundary_incremental_base.take());

    if !cache_request_eligible {
        rollback_pending_boundary_snapshot(
            state,
            &mut pending_boundary_snapshot,
            &mut cache_boundary_reservation,
        );
    }

    let prefill_gil_ms = t_prefill_gil.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("prefill_end");

    let (first_token, prompt_len, stop_ids, kv_overflow, chat_debug_payload) = match prefill_result
    {
        Ok(v) => v,
        Err(e) => {
            rollback_pending_boundary_snapshot(
                state,
                &mut pending_boundary_snapshot,
                &mut cache_boundary_reservation,
            );
            let err_str = e.to_string();
            log::error!("Prefill failed: {}", err_str);
            abort_if_cuda_context_poisoned("chat prefill", &err_str);
            // Return 413 with structured error for KV cache exhaustion
            let (status, body) = if err_str.contains("KV cache exhausted") {
                (
                    413,
                    format!(
                        r#"{{"error":{{"message":"Context length exceeds KV cache capacity ({} tokens max). Reduce context or start a new conversation.","type":"invalid_request_error","code":"context_length_exceeded","max_context_tokens":{}}}}}"#,
                        state.max_context_tokens, state.max_context_tokens
                    ),
                )
            } else if has_images {
                if let Some(body) = image_vram_error_body(&err_str) {
                    (507, body)
                } else if err_str.to_ascii_lowercase().contains("out of memory") {
                    (
                        507,
                        format!(
                            r#"{{"error":{{"message":"VRAM is too constrained for this image request. Multimodal prefill failed: {}","type":"insufficient_resources","code":"insufficient_vram"}}}}"#,
                            json_escape(&err_str)
                        ),
                    )
                } else {
                    (
                        500,
                        format!(
                            r#"{{"error":{{"message":"Prefill failed: {}","type":"server_error"}}}}"#,
                            err_str
                        ),
                    )
                }
            } else {
                (
                    500,
                    format!(
                        r#"{{"error":{{"message":"Prefill failed: {}","type":"server_error"}}}}"#,
                        err_str
                    ),
                )
            };
            let _ = send_json(stream, status, &body);
            // Cleanup on error
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };
    if cache_request_eligible && cache_sequence_start == 0 && prefill_gil_ms > 0.0 {
        state
            .session_cache
            .prefill_samples
            .push((prompt_len, prefill_gil_ms));
    }

    // If prompt exceeded Rust KV cache, return error (not a silent 200 with truncated output)
    if kv_overflow {
        rollback_pending_boundary_snapshot(
            state,
            &mut pending_boundary_snapshot,
            &mut cache_boundary_reservation,
        );
        log::error!(
            "Request {}: prompt {} tokens exceeds Rust KV cache capacity",
            request_id,
            prompt_len
        );
        let _ = send_json(
            stream,
            507,
            &format!(
                r#"{{"error":{{"message":"Prompt ({} tokens) exceeds KV cache capacity. Increase CFG_KV_CACHE_MB or reduce prompt length.","type":"insufficient_storage","code":"kv_cache_overflow","prompt_tokens":{}}}}}"#,
                prompt_len, prompt_len,
            ),
        );
        Python::with_gil(|py| {
            let _ = state.py_model.call_method0(py, "server_cleanup");
        });
        return;
    }

    if cache_request_eligible {
        if cache_sequence_start > 0 {
            let ram_hit = cache_ram_restore.is_some();
            state.session_cache.metrics.record_hit(ram_hit);
            if let Some((bytes, restore_ms)) = cache_ram_restore {
                state
                    .session_cache
                    .metrics
                    .record_restore(bytes, restore_ms);
            }
        } else {
            state
                .session_cache
                .metrics
                .record_miss(cache_miss_reason.unwrap_or(SessionCacheMissReason::NoMatch));
        }
    }

    {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let free_now_mb = store.query_vram_free_mb();
        let primary_device = store.device_ordinal();
        let prefill_min_free_mb = crate::vram_monitor::current_request_lows()
            .into_iter()
            .find(|(device, _)| *device == primary_device)
            .map(|(_, free_mb)| free_mb as usize)
            .unwrap_or(free_now_mb);
        let prefill_secs = prefill_gil_ms / 1000.0;
        let prefill_tok_s = if prefill_secs > 0.0 && prompt_len > 0 {
            prompt_len as f64 / prefill_secs
        } else {
            0.0
        };
        eprintln!(
            "  \x1b[32mprefill: {} tokens in {:.2}s ({:.1} tok/s)  VRAM: {} MB free now, {} MB min free during prefill\x1b[0m",
            prompt_len,
            prefill_secs,
            prefill_tok_s,
            free_now_mb,
            prefill_min_free_mb,
        );
        log::info!(
            "Request {} prefill: {} tokens in {:.2}s ({:.1} tok/s), free_now={} MB, min_free_prefill={} MB",
            request_id,
            prompt_len,
            prefill_secs,
            prefill_tok_s,
            free_now_mb,
            prefill_min_free_mb,
        );
    }

    // Check context length
    if !context_window_fits(prompt_len, max_tokens, state.max_context_tokens) {
        rollback_pending_boundary_snapshot(
            state,
            &mut pending_boundary_snapshot,
            &mut cache_boundary_reservation,
        );
        let requested_total = prompt_len.saturating_add(max_tokens);
        let _ = send_json(
            stream,
            413,
            &format!(
                r#"{{"error":{{"message":"Requested prompt and output total {} tokens ({} prompt + {} max output), exceeding context capacity of {} tokens","type":"invalid_request_error","code":"context_length_exceeded","prompt_tokens":{},"max_output_tokens":{},"max_context_tokens":{}}}}}"#,
                requested_total,
                prompt_len,
                max_tokens,
                state.max_context_tokens,
                prompt_len,
                max_tokens,
                state.max_context_tokens,
            ),
        );
        Python::with_gil(|py| {
            let _ = state.py_model.call_method0(py, "server_cleanup");
        });
        return;
    }

    log::info!(
        "Request {}: {} prompt tokens, max_new={}, stream={}, decode=gpu",
        request_id,
        prompt_len,
        max_tokens,
        is_stream
    );
    crate::vram_monitor::update_request_context(&format!(
        "route=/v1/chat/completions request_id={} model={} prompt_tokens={} max_new={} stream={} phase=decode_setup",
        request_id, state.model_name, prompt_len, max_tokens, is_stream,
    ));

    // ── Reload soft HCS after prefill ──
    // Always attempt reload — soft pool may have been cancelled by a prior operation
    // even if we didn't evict anything this time.
    crate::vram_monitor::report_event("reload_start");
    let t_reload = Instant::now();
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
        log::info!(
            "Request {}: prompt-HCS snapshot ready: prompt_tokens={} layers={} experts={}",
            request_id,
            prompt_tokens,
            layers,
            experts,
        );
        store.install_prompt_hcs_counts(counts.clone(), *layers, *experts, *prompt_tokens);
    } else {
        log::warn!(
            "Request {}: prompt-HCS snapshot missing before reload",
            request_id
        );
        store.clear_prompt_hcs_counts();
    }
    // Decode must never begin with an incomplete HCS.  Use the bounded
    // synchronous reload here: async queue+sync can still create CUDA/DMA
    // transients after the pre-allocation free checks and before pressure
    // drain gets a chance to run.
    let (activated, real_reload_ms) = store.hcs_reload_after_prefill(prompt_len);
    if activated > 0 {
        log::info!(
            "Request {}: HCS reload complete: {} experts, {:.1}ms",
            request_id,
            activated,
            real_reload_ms
        );
    }
    if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
        store.install_prompt_hcs_shadow(counts.clone(), *layers, *experts, *prompt_tokens);
    } else {
        store.clear_prompt_hcs_shadow();
    }
    // NOTE: aux GPUs have no soft tier (100% hard), no eviction/reload needed
    // ── Multi-GPU: copy KV, LA, and DSA prompt state after prefill ──
    if !state.aux_gpu_store_addrs.is_empty() {
        let t_kvcopy = Instant::now();
        let num_aux = state.aux_gpu_store_addrs.len();
        let num_layers = store.num_layers();
        let mut dsa_owner_copies = 0usize;
        let mut dsa_key_bytes = 0usize;
        for i in 0..num_aux {
            let aux_store = unsafe { &mut *(state.aux_gpu_store_addrs[i] as *mut GpuDecodeStore) };
            let layer_start = state.multi_gpu_split_layers[i];
            let layer_end = if i + 1 < num_aux {
                state.multi_gpu_split_layers[i + 1]
            } else {
                num_layers
            };
            if let Err(e) = store.copy_kv_to_aux(
                aux_store,
                layer_start,
                layer_end,
                state.multi_gpu_gqa_offsets[i],
                prompt_len,
            ) {
                log::error!(
                    "Request {}: KV cache copy to aux GPU{} failed: {}",
                    request_id,
                    i + 1,
                    e
                );
            }
            // Copy LA recurrent state (conv_state + recur_state) for linear attention layers
            if let Err(e) = store.copy_la_states_to_aux(aux_store, layer_start, layer_end) {
                log::error!(
                    "Request {}: LA state copy to aux GPU{} failed: {}",
                    request_id,
                    i + 1,
                    e
                );
            }
            match store.copy_dsa_prompt_keys_to_aux(aux_store, prompt_len) {
                Ok((owners, bytes)) => {
                    dsa_owner_copies += owners;
                    dsa_key_bytes += bytes;
                }
                Err(e) => {
                    let message = format!(
                        "Request {}: DSA prompt key-cache copy to aux GPU{} failed: {}",
                        request_id,
                        i + 1,
                        e
                    );
                    log::error!("{}", message);
                    let _ = send_json(
                        stream,
                        500,
                        &format!(
                            r#"{{"error":{{"message":"{}","type":"server_error"}}}}"#,
                            json_escape(&message)
                        ),
                    );
                    Python::with_gil(|py| {
                        let _ = state.py_model.call_method0(py, "server_cleanup");
                    });
                    return;
                }
            }
        }
        let kvcopy_ms = t_kvcopy.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "Request {}: KV+LA+DSA prompt state copied to {} aux GPUs in {:.1}ms (DSA owners={}, bytes={})",
            request_id,
            num_aux,
            kvcopy_ms,
            dsa_owner_copies,
            dsa_key_bytes,
        );
    }
    let (pressure_evicted, pressure_freed_mb, pressure_final_free_mb) =
        store.hcs_drain_vram_pressure("request_before_decode", true);
    if pressure_evicted > 0 {
        log::warn!(
            "Request {}: VRAM pressure eviction before decode evicted {} soft experts, freed {:.1} MB, final_free={} MB",
            request_id,
            pressure_evicted,
            pressure_freed_mb,
            pressure_final_free_mb,
        );
        let (pressure_reload_activated, pressure_reload_ms) =
            store.hcs_reload_after_prefill(prompt_len);
        if pressure_reload_activated > 0 {
            log::info!(
                "Request {}: HCS reload after pressure drain: {} experts, {:.1}ms",
                request_id,
                pressure_reload_activated,
                pressure_reload_ms,
            );
            let (post_reload_evicted, post_reload_freed_mb, post_reload_final_free_mb) =
                store.hcs_drain_vram_pressure("request_before_decode_after_pressure_reload", true);
            if post_reload_evicted > 0 {
                log::warn!(
                    "Request {}: post-reload pressure eviction before decode evicted {} soft experts, freed {:.1} MB, final_free={} MB",
                    request_id,
                    post_reload_evicted,
                    post_reload_freed_mb,
                    post_reload_final_free_mb,
                );
            }
        }
    }
    let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;
    {
        let (min_free_vram_mb, hcs_loaded, hcs_total, hcs_pct) = store.benchmark_stats();
        crate::vram_monitor::update_request_context(&format!(
            "route=/v1/chat/completions request_id={} model={} prompt_tokens={} max_new={} stream={} phase=decode hcs_loaded={}/{} hcs_pct={:.1} hcs_min_free_mb={} safety_margin_mb={}",
            request_id,
            state.model_name,
            prompt_len,
            max_tokens,
            is_stream,
            hcs_loaded,
            hcs_total,
            hcs_pct,
            min_free_vram_mb,
            store.hcs_safety_margin_mb(),
        ));
    }

    let overhead = RequestOverhead {
        parse_ms,
        evict_ms,
        prefill_ms: prefill_gil_ms,
        reload_ms,                          // includes sync wait
        real_reload_dma_ms: real_reload_ms, // actual DMA time (0 if async)
    };

    // ── Thinking suppression: prevent EOS before </think> ──
    // When thinking is enabled, the model must generate </think> before it can
    // terminate with <|im_end|>. Without this, the model puts its answer inside
    // the thinking block and bails to EOS, resulting in 0 visible answer tokens.
    let min_stop_suppress_steps = min_new_tokens.saturating_sub(1);
    let min_stop_suppress_ids = if min_stop_suppress_steps > 0 {
        stop_ids.to_vec()
    } else {
        vec![]
    };
    if enable_thinking {
        if let Some(te_id) = state.thinking_end_token {
            // Budget = max 4096 thinking tokens. If the model hasn't produced </think>
            // by then, it's stuck in a loop. 4096 is generous for real reasoning.
            let think_budget = 4096;
            store.set_think_end_suppress(Some(te_id), think_budget);
            store.set_min_new_tokens_ext(min_stop_suppress_steps, min_stop_suppress_ids.clone());
        } else {
            store.set_think_end_suppress(None, 0);
            store.set_min_new_tokens_ext(min_stop_suppress_steps, min_stop_suppress_ids.clone());
        }
    } else {
        store.set_think_end_suppress(None, 0);
        store.set_min_new_tokens_ext(min_stop_suppress_steps, min_stop_suppress_ids);
    }

    let tokenizer = &state.tokenizer;

    // ── GPU decode: GIL-free Rust decode via GpuDecodeStore ──
    serving_metrics.set_active_kv_tokens(prompt_len);
    serving_metrics.observe_ttft(t_request.elapsed().as_secs_f64());
    crate::vram_monitor::report_event("decode_start");
    let decode_outcome = handle_gpu_decode(
        stream,
        is_stream,
        state,
        store,
        tokenizer,
        first_token,
        prompt_len,
        max_tokens,
        temperature,
        top_k,
        top_p,
        presence_penalty,
        &stop_ids,
        &request_id,
        &state.model_name,
        created,
        &overhead,
        has_tools,
        tool_call_format,
        enable_thinking,
        logprobs_top_n,
        chat_debug_payload,
        &serving_metrics,
    );
    crate::vram_monitor::report_event("decode_end");

    // ── Transactional sequence-state publication and cleanup ──
    let t_cleanup = Instant::now();
    if cache_request_eligible {
        if decode_outcome.completed {
            let mut post_snapshot_base = cache_base_snapshot_id;
            // A duplicate continuation can reuse an already committed exact
            // boundary, so it has no newly captured pending snapshot. In that
            // case the base snapshot and stable token boundary are the
            // transaction's publication source. Never infer a boundary from
            // length alone: the lookup/active plan already exact-matched these
            // stable tokens to the identified base snapshot.
            let mut pending_active_boundary = active_boundary_tokens_for_publication(
                cache_prefill_stage_required,
                pending_boundary_snapshot
                    .as_ref()
                    .map(|pending| pending.consumed_token_ids.as_slice()),
                cache_stable_boundary_tokens.as_deref(),
                cache_sequence_start,
                cache_base_snapshot_id.is_some(),
            );
            if let Some(pending) = pending_boundary_snapshot.take() {
                let boundary_tokens = pending.consumed_token_ids.len();
                match commit_pending_boundary_snapshot(state, pending) {
                    Ok((snapshot_id, bytes, save_ms, calibration_restore_ms)) => {
                        post_snapshot_base = Some(snapshot_id);
                        state.session_cache.metrics.record_save(bytes, save_ms);
                        log::info!(
                            "Request {} prefix cache stable boundary committed: id={:?} tokens={} bytes={} save_ms={:.3} validation_restore_ms={:.3}",
                            request_id,
                            snapshot_id,
                            boundary_tokens,
                            bytes,
                            save_ms,
                            calibration_restore_ms,
                        );
                    }
                    Err(error) => {
                        // The old base remains a valid RAM snapshot, but the
                        // live GPU state has advanced to the new boundary and
                        // must never be published under that old identity.
                        pending_active_boundary = None;
                        post_snapshot_base = None;
                        log::error!(
                            "Request {} prefix cache stable boundary commit failed: {}",
                            request_id,
                            error,
                        );
                    }
                }
            } else {
                cancel_boundary_reservation(
                    &mut state.session_cache,
                    &mut cache_boundary_reservation,
                );
            }
            let mut consumed_token_ids = request_token_ids;
            consumed_token_ids.extend_from_slice(&decode_outcome.consumed_generation_tokens);
            if cache_prefill_stage_required && cache_stable_boundary_tokens.is_some() {
                let publish_active = pending_active_boundary
                    .zip(post_snapshot_base)
                    .ok_or_else(|| {
                        "compressed-stage request completed without a committed active boundary"
                            .to_string()
                    })
                    .and_then(|(boundary_tokens, snapshot_id)| {
                        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                        let checkpoint_bytes =
                            store.commit_active_sequence_checkpoint_rust(boundary_tokens.len())?;
                        Ok((boundary_tokens, snapshot_id, checkpoint_bytes))
                    });
                match publish_active {
                    Ok((boundary_tokens, snapshot_id, checkpoint_bytes)) => {
                        let boundary_len = boundary_tokens.len();
                        state.session_cache.active = Some(ActiveSequenceState {
                            consumed_token_ids: boundary_tokens,
                            snapshot_id,
                            requires_device_checkpoint: true,
                        });
                        log::info!(
                            "Request {} prefix cache active compressed boundary committed: id={:?} tokens={} device_checkpoint_bytes={} host_transfer_bytes=0",
                            request_id,
                            snapshot_id,
                            boundary_len,
                            checkpoint_bytes,
                        );
                    }
                    Err(error) => {
                        state.session_cache.active = None;
                        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                        store.discard_active_sequence_checkpoint_rust();
                        if let Ok(mut engine_guard) = state.rust_prefill.lock() {
                            if let Some(engine) = engine_guard.as_mut() {
                                engine.discard_active_stage_state();
                            }
                        }
                        log::error!(
                            "Request {} prefix cache active compressed boundary publication failed: {}",
                            request_id,
                            error,
                        );
                    }
                }
            } else {
                match snapshot_current_sequence_to_ram(
                    state,
                    &consumed_token_ids,
                    post_snapshot_base,
                ) {
                    Ok((snapshot_id, bytes, save_ms, calibration_restore_ms)) => {
                        state.session_cache.metrics.record_save(bytes, save_ms);
                        state.session_cache.active = Some(ActiveSequenceState {
                            consumed_token_ids,
                            snapshot_id,
                            requires_device_checkpoint: false,
                        });
                        log::info!(
                        "Request {} prefix cache snapshot committed: id={:?} bytes={} save_ms={:.3} validation_restore_ms={:.3}",
                        request_id,
                        snapshot_id,
                        bytes,
                        save_ms,
                        calibration_restore_ms,
                    );
                    }
                    Err(error) => {
                        state.session_cache.active = None;
                        log::error!(
                        "Request {} prefix cache post-generation snapshot failed; any committed stable boundary remains valid: {}",
                        request_id,
                        error,
                    );
                    }
                }
            }
        } else {
            state.session_cache.active = None;
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            store.discard_active_sequence_checkpoint_rust();
            if let Ok(mut engine_guard) = state.rust_prefill.lock() {
                if let Some(engine) = engine_guard.as_mut() {
                    engine.discard_active_stage_state();
                }
            }
            rollback_pending_boundary_snapshot(
                state,
                &mut pending_boundary_snapshot,
                &mut cache_boundary_reservation,
            );
            log::warn!(
                "Request {} prefix cache transaction rolled back: {}",
                request_id,
                decode_outcome
                    .failure_reason
                    .as_deref()
                    .unwrap_or("request did not complete"),
            );
        }
        let cleanup_store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        if let Err(error) = cleanup_store.swap_to_marlin_rust() {
            state.session_cache.active = None;
            log::error!(
                "Request {} Rust cleanup failed; active state invalidated: {}",
                request_id,
                error,
            );
        }
    } else {
        Python::with_gil(|py| {
            let _ = state.py_model.call_method0(py, "server_cleanup");
        });
        state.session_cache.active = None;
    }
    let cleanup_gil_ms = t_cleanup.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("cleanup_end");
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (cleanup_pressure_evicted, cleanup_pressure_freed_mb, cleanup_pressure_final_free_mb) =
        store.hcs_drain_vram_pressure("request_cleanup_end", true);
    if cleanup_pressure_evicted > 0 {
        log::warn!(
            "Request {}: VRAM pressure eviction after cleanup evicted {} soft experts, freed {:.1} MB, final_free={} MB",
            request_id,
            cleanup_pressure_evicted,
            cleanup_pressure_freed_mb,
            cleanup_pressure_final_free_mb,
        );
    }
    // The lease covers GPU restore/prefill/decode and transactional snapshot
    // publication. Release only after all session-visible mutation is done.
    drop(cache_session_lease.take());

    let total_ms = t_request.elapsed().as_secs_f64() * 1000.0;
    log::info!(
        "Request {} complete: total={:.0}ms | parse={:.1}ms evict={:.1}ms prefill={:.0}ms reload={:.0}ms cleanup={:.1}ms",
        request_id, total_ms, parse_ms, evict_ms, prefill_gil_ms, reload_ms, cleanup_gil_ms
    );
    publish_session_cache_metrics(state);
    let outcome = if decode_outcome.completed {
        decode_outcome.finish_reason.as_str()
    } else if decode_outcome.client_aborted {
        "abort"
    } else {
        "error"
    };
    metrics_guard.finish(
        prompt_len,
        decode_outcome.generated_tokens,
        cache_sequence_start,
        outcome,
    );
}

/// Handle /v1/internal/prefill_logits endpoint.
/// Runs a full prefill pass and extracts top-k logprobs at sampled positions.
fn finish_prefill_logits_runtime(
    state: &mut ServerState,
    token_count: usize,
    invalidate_graph: bool,
    reason: &str,
) {
    crate::vram_monitor::report_event("prefill_logits_restore_start");
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    if let Err(error) = store.prepare_runtime_for_decode_rust() {
        log::error!(
            "prefill_logits: failed to restore decode runtime after {}: {}",
            reason,
            error,
        );
        abort_if_cuda_context_poisoned("prefill_logits decode-runtime restore", &error);
    }
    let _ = store.hcs_reload_after_prefill(token_count);
    if invalidate_graph {
        store.invalidate_cuda_graph();
        log::info!(
            "prefill_logits: invalidated CUDA graphs after {} restore",
            reason,
        );
    }
    crate::vram_monitor::report_event("prefill_logits_restore_end");

    // Match the normal reference/inference cleanup path so diagnostic prefill
    // requests do not leak sequence state into the next prompt.
    crate::vram_monitor::report_event("prefill_logits_cleanup_start");
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
    crate::vram_monitor::report_event("prefill_logits_cleanup_end");

    // The monitor may observe a transient peak between named phase snapshots.
    // Convert that measured deficit into the same runtime-sized HCS pressure
    // cap used by ordinary inference before another diagnostic request starts.
    let pressure_evicted = drain_vram_pressure_for_state(state, "prefill_logits_cleanup_end", true);
    if pressure_evicted > 0 {
        log::warn!(
            "prefill_logits: VRAM pressure feedback after {} evicted {} soft experts",
            reason,
            pressure_evicted,
        );
    }
}

fn handle_prefill_logits(stream: &mut TcpStream, body: &str, state: &mut ServerState) {
    invalidate_active_sequence(state, "prefill_logits_request");
    // Parse request
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = send_json(
                stream,
                400,
                &format!(r#"{{"error":"Invalid JSON: {}"}}"#, e),
            );
            return;
        }
    };

    let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let sample_every = req
        .get("sample_every")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    if sample_every == 0 {
        let _ = send_json(
            stream,
            400,
            r#"{"error":"sample_every must be greater than zero"}"#,
        );
        return;
    }
    let target_token_ids: Option<Vec<u32>> = match req.get("target_token_ids") {
        Some(serde_json::Value::Array(arr)) => {
            let mut parsed = Vec::with_capacity(arr.len());
            for v in arr {
                match v.as_u64() {
                    Some(tid) if tid <= u32::MAX as u64 => parsed.push(tid as u32),
                    Some(_) | None => {
                        let _ = send_json(
                            stream,
                            400,
                            r#"{"error":"target_token_ids must be an array of non-negative integers"}"#,
                        );
                        return;
                    }
                }
            }
            Some(parsed)
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"target_token_ids must be an array"}"#,
            );
            return;
        }
        None => None,
    };

    // Accept either raw input_token_ids or messages (with chat template + tokenization)
    let token_ids: Vec<u32> =
        if let Some(serde_json::Value::Array(arr)) = req.get("input_token_ids") {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect()
        } else if let Some(messages) = req.get("messages") {
            if let Err(e) = crate::text_only_messages::validate_text_only_messages(messages) {
                let _ = send_json(
                    stream,
                    400,
                    &format!(r#"{{"error":"{}"}}"#, json_escape(&e)),
                );
                return;
            }
            let messages_json = messages.to_string();
            let enable_thinking = req
                .get("enable_thinking")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let rendered = match state
                .chat_template
                .apply(&messages_json, true, enable_thinking)
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = send_json(
                        stream,
                        500,
                        &format!(r#"{{"error":"Chat template: {}"}}"#, e),
                    );
                    return;
                }
            };
            match state.tokenizer.encode(rendered.as_str(), false) {
                Ok(e) => e.get_ids().to_vec(),
                Err(e) => {
                    let _ = send_json(stream, 500, &format!(r#"{{"error":"Tokenize: {}"}}"#, e));
                    return;
                }
            }
        } else {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"Missing input_token_ids or messages"}"#,
            );
            return;
        };
    if let Some(ref targets) = target_token_ids {
        if targets.len() != token_ids.len() {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"target_token_ids length must match input token length"}"#,
            );
            return;
        }
    }

    crate::vram_monitor::begin_request_context(&format!(
        "route=/v1/internal/prefill_logits tokens={} target_logprobs={}",
        token_ids.len(),
        target_token_ids.is_some(),
    ));
    let _vram_context_guard = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        VramRequestContextGuard {
            safety_margin_mb: store.hcs_safety_margin_mb() as u64,
        }
    };

    log::info!(
        "prefill_logits: {} tokens, top_k={}, sample_every={}, target_logprobs={}",
        token_ids.len(),
        top_k,
        sample_every,
        target_token_ids.is_some()
    );

    // Evict soft HCS before diagnostic prefill so this endpoint uses the same
    // conservative VRAM budget as the production and reference-test paths.
    crate::vram_monitor::report_event("prefill_logits_evict_start");
    let prefill_entry_floor_bytes =
        match prefill_entry_floor_bytes_for_server(&state.rust_prefill, token_ids.len()) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::error!(
                    "Prefill logits engine floor unavailable before HCS eviction: {}",
                    e
                );
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Prefill engine floor unavailable: {}"}}"#,
                        json_escape(&e)
                    ),
                );
                return;
            }
        };
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (_evicted, _freed_mb) = store_for_evict
        .hcs_evict_for_prefill_with_engine_floor(token_ids.len(), prefill_entry_floor_bytes);
    crate::vram_monitor::report_event("prefill_logits_evict_end");

    // Run prefill logits extraction
    let mut engine_guard = state.rust_prefill.lock().unwrap();
    let engine = match engine_guard.as_mut() {
        Some(e) => e,
        None => {
            drop(engine_guard);
            finish_prefill_logits_runtime(
                state,
                token_ids.len(),
                true,
                "missing Rust prefill engine",
            );
            let _ = send_json(
                stream,
                500,
                r#"{"error":"Rust prefill engine not available"}"#,
            );
            return;
        }
    };

    // Update HCS snapshot
    {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let (cache_fast, ne) = store.export_hcs_snapshot();
        engine.update_hcs_snapshot(cache_fast, ne);
    }

    let _has_hqq_runtime_slots = {
        crate::vram_monitor::report_event("prefill_logits_prepare_runtime_start");
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        match prepare_store_for_rust_prefill(store, engine, token_ids.len()) {
            Ok(has_hqq) => has_hqq,
            Err(e) => {
                drop(engine_guard);
                finish_prefill_logits_runtime(
                    state,
                    token_ids.len(),
                    true,
                    "failed prefill preparation",
                );
                let _ = send_json(
                    stream,
                    500,
                    &format!(r#"{{"error":"Prefill prepare failed: {}"}}"#, e),
                );
                return;
            }
        }
    };
    crate::vram_monitor::report_event("prefill_logits_prepare_runtime_end");

    // Dynamically allocate scratch for this prompt
    // run_prefill_logits needs scratch sized for all tokens (no chunking)
    crate::vram_monitor::report_event("prefill_logits_scratch_alloc_start");
    if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
        drop(engine_guard);
        finish_prefill_logits_runtime(state, token_ids.len(), true, "failed scratch allocation");
        let _ = send_json(
            stream,
            500,
            &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e),
        );
        return;
    }
    crate::vram_monitor::report_event("prefill_logits_scratch_alloc_end");

    crate::vram_monitor::report_event("prefill_logits_run_start");
    let positions = match engine.run_prefill_logits(
        &token_ids,
        top_k,
        sample_every,
        target_token_ids.as_deref(),
    ) {
        Ok(p) => p,
        Err(e) => {
            // Release scratch even on error
            let _ = engine.release_scratch();
            drop(engine_guard);
            finish_prefill_logits_runtime(
                state,
                token_ids.len(),
                true,
                "failed diagnostic prefill",
            );
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Prefill logits: {}"}}"#, e),
            );
            return;
        }
    };
    crate::vram_monitor::report_event("prefill_logits_run_end");

    // Release scratch after logits extraction
    crate::vram_monitor::report_event("prefill_logits_scratch_release_start");
    if let Err(e) = engine.release_scratch() {
        log::error!("Failed to release scratch after prefill_logits: {}", e);
        abort_if_cuda_context_poisoned("prefill_logits release_scratch", &e);
    }
    crate::vram_monitor::report_event("prefill_logits_scratch_release_end");

    drop(engine_guard);
    finish_prefill_logits_runtime(state, token_ids.len(), false, "successful prefill");

    // Format response: {positions: [{position, target_token_id, target_logprob, top_k: [...]}]}
    let mut pos_json = Vec::new();
    for p in &positions {
        let mut tk_json = Vec::new();
        for &(tid, lp) in &p.top_k {
            tk_json.push(format!(r#"{{"token_id":{},"logprob":{:.6}}}"#, tid, lp));
        }
        let target_token_json = match p.target_token_id {
            Some(tid) => tid.to_string(),
            None => "null".to_string(),
        };
        let target_logprob_json = match p.target_logprob {
            Some(lp) => format!("{:.9}", lp),
            None => "null".to_string(),
        };
        pos_json.push(format!(
            r#"{{"position":{},"target_token_id":{},"target_logprob":{},"top_k":[{}]}}"#,
            p.position,
            target_token_json,
            target_logprob_json,
            tk_json.join(",")
        ));
    }
    let response = format!(r#"{{"positions":[{}]}}"#, pos_json.join(","));
    crate::vram_monitor::report_event("prefill_logits_response");
    let _ = send_json(stream, 200, &response);
}

/// Test-only exact-checkpoint diagnostic: process an existing prompt one token
/// at a time through the Rust/CUDA decode path and report logits after each
/// token.  This is deliberately separate from production generation and from
/// batch prefill so the two execution strategies can be compared directly.
fn handle_teacher_forced_decode_logits(
    stream: &mut TcpStream,
    body: &str,
    state: &mut ServerState,
) {
    invalidate_active_sequence(state, "teacher_forced_decode_logits_request");
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(error) => {
            let _ = send_json(
                stream,
                400,
                &format!(
                    r#"{{"error":"Invalid JSON: {}"}}"#,
                    json_escape(&error.to_string())
                ),
            );
            return;
        }
    };
    let token_ids: Vec<u32> = match req.get("input_token_ids") {
        Some(serde_json::Value::Array(values)) => {
            let mut parsed = Vec::with_capacity(values.len());
            for value in values {
                match value.as_u64().and_then(|token| u32::try_from(token).ok()) {
                    Some(token) => parsed.push(token),
                    None => {
                        let _ = send_json(
                            stream,
                            400,
                            r#"{"error":"input_token_ids must contain only u32 token IDs"}"#,
                        );
                        return;
                    }
                }
            }
            parsed
        }
        _ => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"Missing or invalid input_token_ids array"}"#,
            );
            return;
        }
    };
    if token_ids.is_empty() || token_ids.len() > state.max_context_tokens {
        let _ = send_json(
            stream,
            400,
            &format!(
                r#"{{"error":"input_token_ids length must be in 1..={}"}}"#,
                state.max_context_tokens
            ),
        );
        return;
    }
    let top_k = req
        .get("top_k")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(10);
    if !(1..=100).contains(&top_k) {
        let _ = send_json(stream, 400, r#"{"error":"top_k must be in 1..=100"}"#);
        return;
    }
    let debug_decode_early_trace = req
        .get("debug_decode_early_trace")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let debug_decode_early_detail_dims: Vec<usize> = match req.get("debug_decode_early_detail_dims")
    {
        Some(serde_json::Value::Array(values)) => {
            let mut dims = Vec::with_capacity(values.len());
            for value in values {
                let Some(dim) = value.as_u64().and_then(|dim| usize::try_from(dim).ok()) else {
                    let _ = send_json(
                        stream,
                        400,
                        r#"{"error":"debug_decode_early_detail_dims must contain only unsigned integer dimensions"}"#,
                    );
                    return;
                };
                dims.push(dim);
            }
            dims
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_decode_early_detail_dims must be an array of unsigned integer dimensions"}"#,
            );
            return;
        }
        None => Vec::new(),
    };

    crate::vram_monitor::begin_request_context(&format!(
        "route=/v1/internal/teacher_forced_decode_logits tokens={}",
        token_ids.len(),
    ));
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let _vram_context_guard = VramRequestContextGuard {
        safety_margin_mb: store.hcs_safety_margin_mb() as u64,
    };

    let reset_bytes = match store.reset_sequence_state_for_diagnostic_rust() {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = send_json(
                stream,
                500,
                &format!(
                    r#"{{"error":"Teacher-forced state reset failed: {}"}}"#,
                    json_escape(&error)
                ),
            );
            return;
        }
    };
    if let Err(error) = store.prepare_runtime_for_decode_rust() {
        let _ = send_json(
            stream,
            500,
            &format!(
                r#"{{"error":"Teacher-forced decode preparation failed: {}"}}"#,
                json_escape(&error)
            ),
        );
        return;
    }
    if debug_decode_early_trace {
        if let Err(error) = store.begin_debug_decode_early_trace_rust(
            token_ids.len() as u64,
            debug_decode_early_detail_dims,
        ) {
            let cleanup_error = store.reset_sequence_state_for_diagnostic_rust().err();
            let _ = send_json(
                stream,
                500,
                &format!(
                    r#"{{"error":"Teacher-forced trace setup failed: {}","cleanup_error":{}}}"#,
                    json_escape(&error),
                    cleanup_error
                        .map(|value| format!(r#""{}""#, json_escape(&value)))
                        .unwrap_or_else(|| "null".to_string()),
                ),
            );
            return;
        }
    }

    let mut positions = Vec::with_capacity(token_ids.len());
    for (position, &token_id) in token_ids.iter().enumerate() {
        if let Err(error) = store.gpu_decode_step(token_id as usize, position) {
            let cleanup_error = store.reset_sequence_state_for_diagnostic_rust().err();
            let _ = send_json(
                stream,
                500,
                &format!(
                    r#"{{"error":"Teacher-forced decode failed at position {}: {}","cleanup_error":{}}}"#,
                    position,
                    json_escape(&error),
                    cleanup_error
                        .map(|value| format!(r#""{}""#, json_escape(&value)))
                        .unwrap_or_else(|| "null".to_string()),
                ),
            );
            return;
        }
        let logits = match store.logits_snapshot_rust() {
            Ok(logits) => logits,
            Err(error) => {
                let cleanup_error = store.reset_sequence_state_for_diagnostic_rust().err();
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Teacher-forced logits unavailable at position {}: {}","cleanup_error":{}}}"#,
                        position,
                        json_escape(&error),
                        cleanup_error
                            .map(|value| format!(r#""{}""#, json_escape(&value)))
                            .unwrap_or_else(|| "null".to_string()),
                    ),
                );
                return;
            }
        };
        let top = crate::decode::extract_top_logprobs(&logits, logits.len(), top_k)
            .into_iter()
            .map(
                |(token_id, logprob)| serde_json::json!({"token_id": token_id, "logprob": logprob}),
            )
            .collect::<Vec<_>>();
        positions.push(serde_json::json!({
            "position": position,
            "input_token_id": token_id,
            "top_k": top,
        }));
        if let Err(error) = store.complete_diagnostic_decode_step_rust() {
            let trace_cleanup_error = if debug_decode_early_trace {
                store.take_debug_decode_early_trace_rust().err()
            } else {
                None
            };
            let state_cleanup_error = store.reset_sequence_state_for_diagnostic_rust().err();
            let _ = send_json(
                stream,
                500,
                &format!(
                    r#"{{"error":"Teacher-forced trace step advance failed at position {}: {}","trace_cleanup_error":{},"state_cleanup_error":{}}}"#,
                    position,
                    json_escape(&error),
                    trace_cleanup_error
                        .map(|value| format!(r#""{}""#, json_escape(&value)))
                        .unwrap_or_else(|| "null".to_string()),
                    state_cleanup_error
                        .map(|value| format!(r#""{}""#, json_escape(&value)))
                        .unwrap_or_else(|| "null".to_string()),
                ),
            );
            return;
        }
    }

    let debug_decode_early_entries = if debug_decode_early_trace {
        match store.take_debug_decode_early_trace_rust() {
            Ok(entries) => Some(entries),
            Err(error) => {
                let cleanup_error = store.reset_sequence_state_for_diagnostic_rust().err();
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Teacher-forced trace collection failed: {}","cleanup_error":{}}}"#,
                        json_escape(&error),
                        cleanup_error
                            .map(|value| format!(r#""{}""#, json_escape(&value)))
                            .unwrap_or_else(|| "null".to_string()),
                    ),
                );
                return;
            }
        }
    } else {
        None
    };
    let cleanup_error = store.reset_sequence_state_for_diagnostic_rust().err();
    let response = serde_json::json!({
        "positions": positions,
        "reset_bytes": reset_bytes,
        "cleanup_error": cleanup_error,
        "debug_decode_early_trace": debug_decode_early_entries.map(|entries| serde_json::json!({
            "entry_count": entries.len(),
            "entries": entries,
        })),
    });
    match serde_json::to_string(&response) {
        Ok(body) => {
            let _ = send_json(stream, 200, &body);
        }
        Err(error) => {
            let _ = send_json(
                stream,
                500,
                &format!(
                    r#"{{"error":"Serialize response: {}"}}"#,
                    json_escape(&error.to_string())
                ),
            );
        }
    }
}

/// Handle /v1/internal/reference_test endpoint.
/// Accepts raw input_token_ids, runs greedy prefill + decode, returns output tokens with logprobs.
/// Used for comparing engine output against BF16 reference data.
fn handle_reference_test(stream: &mut TcpStream, body: &str, state: &mut ServerState) {
    invalidate_active_sequence(state, "reference_test_request");
    let t_start = Instant::now();
    state.reference_test_request_order = state.reference_test_request_order.saturating_add(1);
    let reference_request_order = state.reference_test_request_order;
    crate::vram_monitor::begin_request_context(&format!(
        "route=/v1/internal/reference_test request_order={} model={} phase=parse",
        reference_request_order, state.model_name,
    ));
    let _vram_context_guard = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        VramRequestContextGuard {
            safety_margin_mb: store.hcs_safety_margin_mb() as u64,
        }
    };

    // Parse request
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = send_json(
                stream,
                400,
                &format!(r#"{{"error":"Invalid JSON: {}"}}"#, e),
            );
            return;
        }
    };

    // Required: input_token_ids (raw token IDs, no tokenization or template applied)
    let input_token_ids: Vec<u32> = match req.get("input_token_ids") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"Missing or invalid input_token_ids array"}"#,
            );
            return;
        }
    };

    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(200) as usize;
    let top_logprobs = req
        .get("top_logprobs")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    let debug_reference_trace = req
        .get("debug_reference_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_prompt_trace = req
        .get("debug_prompt_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_prefill_device_trace = req
        .get("debug_prefill_device_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_prefill_device_trace_all_layers = req
        .get("debug_prefill_device_trace_all_layers")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_prefill_device_trace_mla_only = match req.get("debug_prefill_device_trace_mla_only") {
        Some(serde_json::Value::Bool(value)) => *value,
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_prefill_device_trace_mla_only must be a boolean"}"#,
            );
            return;
        }
        None => false,
    };
    let debug_prefill_device_trace_full_pre_out_proj =
        match req.get("debug_prefill_device_trace_full_pre_out_proj") {
            Some(serde_json::Value::Bool(value)) => *value,
            Some(_) => {
                let _ = send_json(
                    stream,
                    400,
                    r#"{"error":"debug_prefill_device_trace_full_pre_out_proj must be a boolean"}"#,
                );
                return;
            }
            None => false,
        };
    let debug_prefill_device_trace_layer = req
        .get("debug_prefill_device_trace_layer")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(if debug_prefill_device_trace_all_layers {
            crate::gpu_prefill::PREFILL_DEVICE_TRACE_NO_SELECTED_LAYER
        } else {
            4usize
        });
    let debug_prefill_device_trace_dims: Vec<usize> = match req
        .get("debug_prefill_device_trace_dims")
    {
        Some(serde_json::Value::Array(values)) => {
            let mut dims = Vec::with_capacity(values.len());
            for value in values {
                let Some(dim) = value.as_u64() else {
                    let _ = send_json(
                        stream,
                        400,
                        r#"{"error":"debug_prefill_device_trace_dims must contain only unsigned integer dimensions"}"#,
                    );
                    return;
                };
                dims.push(dim as usize);
            }
            dims
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_prefill_device_trace_dims must be an array of unsigned integer dimensions"}"#,
            );
            return;
        }
        None => Vec::new(),
    };
    let debug_prefill_device_trace_rows: Vec<usize> = match req
        .get("debug_prefill_device_trace_rows")
    {
        Some(serde_json::Value::Array(values)) => {
            let mut rows = Vec::with_capacity(values.len());
            for value in values {
                let Some(row) = value.as_u64() else {
                    let _ = send_json(
                        stream,
                        400,
                        r#"{"error":"debug_prefill_device_trace_rows must contain only unsigned integer row indices"}"#,
                    );
                    return;
                };
                rows.push(row as usize);
            }
            rows
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_prefill_device_trace_rows must be an array of unsigned integer row indices"}"#,
            );
            return;
        }
        None => Vec::new(),
    };
    let debug_prefill_device_trace_local_scan_token = match req
        .get("debug_prefill_device_trace_local_scan_token")
    {
        Some(value) => match value.as_u64() {
            Some(token) => Some(token as usize),
            None => {
                let _ = send_json(
                    stream,
                    400,
                    r#"{"error":"debug_prefill_device_trace_local_scan_token must be an unsigned integer"}"#,
                );
                return;
            }
        },
        None => None,
    };
    let debug_prefill_device_trace_experts: Vec<usize> = match req
        .get("debug_prefill_device_trace_experts")
    {
        Some(serde_json::Value::Array(values)) => {
            let mut experts = Vec::with_capacity(values.len());
            for value in values {
                let Some(expert) = value.as_u64() else {
                    let _ = send_json(
                        stream,
                        400,
                        r#"{"error":"debug_prefill_device_trace_experts must contain only unsigned integer expert IDs"}"#,
                    );
                    return;
                };
                experts.push(expert as usize);
            }
            experts
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_prefill_device_trace_experts must be an array of unsigned integer expert IDs"}"#,
            );
            return;
        }
        None => Vec::new(),
    };
    let debug_router_variant_requested = req.get("debug_router_variant").is_some();
    let debug_router_variant = match req.get("debug_router_variant") {
        Some(serde_json::Value::String(value)) => {
            match crate::gpu_prefill::ReferenceRouterVariant::from_request_str(value) {
                Some(variant) => variant,
                None => {
                    let _ = send_json(
                        stream,
                        400,
                        r#"{"error":"debug_router_variant must be one of: raw, corrected_hf_unsorted, corrected_sorted, corrected_set_raw_slot_weights"}"#,
                    );
                    return;
                }
            }
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_router_variant must be a string"}"#,
            );
            return;
        }
        None => crate::gpu_prefill::ReferenceRouterVariant::RawBaseline,
    };
    let debug_router_variant_layers: Vec<usize> = match req.get("debug_router_variant_layers") {
        Some(serde_json::Value::Array(values)) => {
            let mut layers = Vec::with_capacity(values.len());
            for value in values {
                let Some(layer_idx) = value.as_u64() else {
                    let _ = send_json(
                        stream,
                        400,
                        r#"{"error":"debug_router_variant_layers must contain only unsigned integer layer indices"}"#,
                    );
                    return;
                };
                layers.push(layer_idx as usize);
            }
            layers.sort_unstable();
            layers.dedup();
            layers
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_router_variant_layers must be an array of unsigned integer layer indices"}"#,
            );
            return;
        }
        None => Vec::new(),
    };
    let debug_router_e_score_corr_by_layer: Vec<Option<Vec<f32>>> = match req
        .get("debug_router_e_score_correction_by_layer")
    {
        Some(serde_json::Value::Array(layers)) => {
            let mut parsed = Vec::with_capacity(layers.len());
            for (layer_idx, layer_value) in layers.iter().enumerate() {
                match layer_value {
                    serde_json::Value::Null => parsed.push(None),
                    serde_json::Value::Array(values) => {
                        let mut layer_values = Vec::with_capacity(values.len());
                        for value in values {
                            let Some(v) = value.as_f64() else {
                                let _ = send_json(
                                    stream,
                                    400,
                                    &format!(
                                        r#"{{"error":"debug_router_e_score_correction_by_layer[{}] must contain only numbers"}}"#,
                                        layer_idx,
                                    ),
                                );
                                return;
                            };
                            layer_values.push(v as f32);
                        }
                        parsed.push(Some(layer_values));
                    }
                    _ => {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_router_e_score_correction_by_layer[{}] must be null or an array of numbers"}}"#,
                                layer_idx,
                            ),
                        );
                        return;
                    }
                }
            }
            parsed
        }
        Some(_) => {
            let _ = send_json(
                stream,
                400,
                r#"{"error":"debug_router_e_score_correction_by_layer must be an array"}"#,
            );
            return;
        }
        None => Vec::new(),
    };
    let debug_router_forced_slot_orders_requested =
        req.get("debug_router_forced_slot_orders").is_some();
    let debug_router_forced_slot_orders: Vec<crate::gpu_prefill::ReferenceRouterForcedSlotOrder> =
        match req.get("debug_router_forced_slot_orders") {
            Some(serde_json::Value::Array(entries)) => {
                let mut parsed = Vec::with_capacity(entries.len());
                for (entry_idx, entry) in entries.iter().enumerate() {
                    let serde_json::Value::Object(obj) = entry else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_router_forced_slot_orders[{}] must be an object"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let Some(layer_idx) = obj.get("layer").and_then(|v| v.as_u64()) else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_router_forced_slot_orders[{}].layer must be an unsigned integer"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let Some(row_idx) = obj.get("row").and_then(|v| v.as_u64()) else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_router_forced_slot_orders[{}].row must be an unsigned integer"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let expert_values = obj
                        .get("expert_ids")
                        .or_else(|| obj.get("slot_order"))
                        .or_else(|| obj.get("slot_expert_ids"));
                    let Some(serde_json::Value::Array(expert_values)) = expert_values else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_router_forced_slot_orders[{}] must include expert_ids as an array"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let mut expert_ids = Vec::with_capacity(expert_values.len());
                    for (slot_idx, value) in expert_values.iter().enumerate() {
                        let Some(expert_id) = value.as_u64() else {
                            let _ = send_json(
                                stream,
                                400,
                                &format!(
                                    r#"{{"error":"debug_router_forced_slot_orders[{}].expert_ids[{}] must be an unsigned integer"}}"#,
                                    entry_idx, slot_idx,
                                ),
                            );
                            return;
                        };
                        expert_ids.push(expert_id as usize);
                    }
                    if parsed.iter().any(
                        |existing: &crate::gpu_prefill::ReferenceRouterForcedSlotOrder| {
                            existing.layer_idx == layer_idx as usize
                                && existing.row_idx == row_idx as usize
                        },
                    ) {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"duplicate debug_router_forced_slot_orders entry for layer {} row {}"}}"#,
                                layer_idx, row_idx,
                            ),
                        );
                        return;
                    }
                    parsed.push(crate::gpu_prefill::ReferenceRouterForcedSlotOrder {
                        layer_idx: layer_idx as usize,
                        row_idx: row_idx as usize,
                        expert_ids,
                    });
                }
                parsed
            }
            Some(_) => {
                let _ = send_json(
                    stream,
                    400,
                    r#"{"error":"debug_router_forced_slot_orders must be an array"}"#,
                );
                return;
            }
            None => Vec::new(),
        };
    let debug_mamba2_gated_norm_replay_requested =
        req.get("debug_mamba2_gated_norm_replay").is_some();
    let debug_mamba2_gated_norm_replay: Vec<crate::gpu_prefill::ReferenceMamba2GatedNormReplay> =
        match req.get("debug_mamba2_gated_norm_replay") {
            Some(serde_json::Value::Array(entries)) => {
                let mut parsed = Vec::with_capacity(entries.len());
                for (entry_idx, entry) in entries.iter().enumerate() {
                    let serde_json::Value::Object(obj) = entry else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_mamba2_gated_norm_replay[{}] must be an object"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let Some(layer_idx) = obj.get("layer").and_then(|v| v.as_u64()) else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_mamba2_gated_norm_replay[{}].layer must be an unsigned integer"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let Some(row_idx) = obj.get("row").and_then(|v| v.as_u64()) else {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"debug_mamba2_gated_norm_replay[{}].row must be an unsigned integer"}}"#,
                                entry_idx,
                            ),
                        );
                        return;
                    };
                    let mode = match obj.get("mode").and_then(|v| v.as_str()) {
                        Some(value) => {
                            match crate::gpu_prefill::ReferenceMamba2GatedNormReplayMode::from_request_str(value) {
                                Some(mode) => mode,
                                None => {
                                    let _ = send_json(
                                        stream,
                                        400,
                                        r#"{"error":"debug_mamba2_gated_norm_replay mode must be sqrt_approx_div_rn"}"#,
                                    );
                                    return;
                                }
                            }
                        }
                        None => {
                            let _ = send_json(
                                stream,
                                400,
                                &format!(
                                    r#"{{"error":"debug_mamba2_gated_norm_replay[{}].mode is required"}}"#,
                                    entry_idx,
                                ),
                            );
                            return;
                        }
                    };
                    if parsed.iter().any(
                        |existing: &crate::gpu_prefill::ReferenceMamba2GatedNormReplay| {
                            existing.layer_idx == layer_idx as usize
                                && existing.row_idx == row_idx as usize
                        },
                    ) {
                        let _ = send_json(
                            stream,
                            400,
                            &format!(
                                r#"{{"error":"duplicate debug_mamba2_gated_norm_replay entry for layer {} row {}"}}"#,
                                layer_idx, row_idx,
                            ),
                        );
                        return;
                    }
                    parsed.push(crate::gpu_prefill::ReferenceMamba2GatedNormReplay {
                        layer_idx: layer_idx as usize,
                        row_idx: row_idx as usize,
                        mode,
                    });
                }
                parsed
            }
            Some(_) => {
                let _ = send_json(
                    stream,
                    400,
                    r#"{"error":"debug_mamba2_gated_norm_replay must be an array"}"#,
                );
                return;
            }
            None => Vec::new(),
        };
    let debug_decode_state_trace_requested = req
        .get("debug_decode_state_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_decode_hcs_equiv_trace = req
        .get("debug_decode_hcs_equiv_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_decode_early_trace = req
        .get("debug_decode_early_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_decode_early_trace_max_steps = match req.get("debug_decode_early_trace_max_steps") {
        Some(value) => match value.as_u64() {
            Some(steps) if (1..=64).contains(&steps) => steps,
            _ => {
                let _ = send_json(
                    stream,
                    400,
                    r#"{"error":"debug_decode_early_trace_max_steps must be an integer from 1 to 64"}"#,
                );
                return;
            }
        },
        None => 3,
    };
    let debug_hcs_transition_trace = req
        .get("debug_hcs_transition_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_mamba2_state_lifecycle_trace = req
        .get("debug_mamba2_state_lifecycle_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug_decode_hcs_equiv_layer = req
        .get("debug_decode_hcs_equiv_layer")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(1usize);
    let debug_mamba2_state_layer = req
        .get("debug_mamba2_state_layer")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(0usize);
    let debug_decode_state_trace = debug_decode_state_trace_requested
        || debug_decode_hcs_equiv_trace
        || debug_decode_early_trace
        || debug_hcs_transition_trace
        || debug_mamba2_state_lifecycle_trace;
    let client_request_id = req
        .get("debug_request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let input_token_hash = fnv1a_token_hash(&input_token_ids);

    // Stop token IDs (from reference data's eos_token_ids)
    let stop_ids: Vec<usize> = match req.get("stop_token_ids") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect(),
        _ => state.eos_stop_ids.clone(),
    };

    log::info!(
        "reference_test: {} input tokens, max_tokens={}, top_logprobs={}, stop_ids={:?}",
        input_token_ids.len(),
        max_tokens,
        top_logprobs,
        stop_ids
    );
    crate::vram_monitor::update_request_context(&format!(
        "route=/v1/internal/reference_test request_order={} model={} prompt_tokens={} max_new={} phase=prefill",
        reference_request_order,
        state.model_name,
        input_token_ids.len(),
        max_tokens,
    ));

    let mut debug_hcs_transition_points: Vec<serde_json::Value> = Vec::new();
    let mut debug_mamba2_state_lifecycle_points: Vec<serde_json::Value> = Vec::new();

    // ── Evict soft HCS before prefill ──
    let prefill_entry_floor_bytes =
        match prefill_entry_floor_bytes_for_server(&state.rust_prefill, input_token_ids.len()) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::error!(
                    "Reference-test prefill engine floor unavailable before HCS eviction: {}",
                    e
                );
                let _ = send_json(
                    stream,
                    500,
                    &format!(
                        r#"{{"error":"Prefill engine floor unavailable: {}"}}"#,
                        json_escape(&e)
                    ),
                );
                return;
            }
        };
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    if debug_mamba2_state_lifecycle_trace {
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store_for_evict,
            "request_start_before_hcs_evict_for_prefill",
            debug_mamba2_state_layer,
        ));
    }
    if debug_hcs_transition_trace {
        let raw =
            store_for_evict.hcs_debug_summary_json("request_start_before_hcs_evict_for_prefill");
        debug_hcs_transition_points.push(serde_json::from_str(&raw).unwrap_or_else(|e| {
            serde_json::json!({
                "phase": "request_start_before_hcs_evict_for_prefill",
                "available": false,
                "error": format!("parse_failed: {}", e),
                "raw": raw,
            })
        }));
    }
    let (evicted, freed_mb) = store_for_evict
        .hcs_evict_for_prefill_with_engine_floor(input_token_ids.len(), prefill_entry_floor_bytes);
    if debug_hcs_transition_trace {
        let raw = store_for_evict.hcs_debug_summary_json("after_hcs_evict_for_prefill");
        let mut value = serde_json::from_str(&raw).unwrap_or_else(|e| {
            serde_json::json!({
                "phase": "after_hcs_evict_for_prefill",
                "available": false,
                "error": format!("parse_failed: {}", e),
                "raw": raw,
            })
        });
        if let Some(obj) = value.as_object_mut() {
            obj.insert("evicted".to_string(), serde_json::json!(evicted));
            obj.insert("freed_mb".to_string(), serde_json::json!(freed_mb));
        }
        debug_hcs_transition_points.push(value);
    }
    if debug_mamba2_state_lifecycle_trace {
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store_for_evict,
            "after_hcs_evict_for_prefill",
            debug_mamba2_state_layer,
        ));
    }

    // ── Prefill with raw token IDs (no tokenization, no chat template) ──
    let mut engine_guard = state.rust_prefill.lock().unwrap();
    let engine = match engine_guard.as_mut() {
        Some(e) => e,
        None => {
            let _ = send_json(
                stream,
                500,
                r#"{"error":"Rust prefill engine not available"}"#,
            );
            return;
        }
    };

    // Update HCS snapshot
    {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let (cache_fast, ne) = store.export_hcs_snapshot();
        engine.update_hcs_snapshot(cache_fast, ne);
    }
    // Warmup/calibration calls disable prefill pinning through the shared engine.
    // Raw prefill-logits requests should use the normal prefill policy.
    engine.set_prefill_pinning_disabled(false);

    let (hcs_snapshot_entries, hcs_num_experts_per_layer) = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let (cache_fast, ne) = store.export_hcs_snapshot();
        (cache_fast.len(), ne)
    };

    let has_hqq_runtime_slots = {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        match prepare_store_for_rust_prefill(store, engine, input_token_ids.len()) {
            Ok(has_hqq) => has_hqq,
            Err(e) => {
                let _ = send_json(
                    stream,
                    500,
                    &format!(r#"{{"error":"Prefill prepare failed: {}"}}"#, e),
                );
                return;
            }
        }
    };

    let hqq_prefill_materialized = false;

    let suppress_tokens = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        store.suppress_tokens_clone()
    };
    engine.set_reference_debug_trace_enabled(debug_reference_trace);
    engine.set_first_token_margin_projection_request_enabled(true);
    engine.set_read_only_checkpoint_request_enabled(true);
    engine.set_reference_router_variant_override(
        debug_router_variant,
        debug_router_variant_layers.clone(),
        debug_router_e_score_corr_by_layer.clone(),
        debug_router_forced_slot_orders.clone(),
    );
    engine.set_reference_mamba2_gated_norm_replay(debug_mamba2_gated_norm_replay.clone());

    engine.set_prefill_hcs_guard_store_addr(state.gpu_store_addr);
    let mut retry_cap: Option<usize> = None;
    let mut retry_attempt = 0usize;
    let mut scratch_tokens_after_prepare = 0usize;
    let mut prefill_chunk_size_after_prepare = engine.config.prefill_chunk_size;

    crate::vram_monitor::reset_request_lows();
    let prefill_result = loop {
        engine.set_prefill_runtime_chunk_cap(retry_cap);

        // Dynamically allocate scratch for this prompt.
        if let Err(e) = engine.prepare_for_prefill(input_token_ids.len()) {
            engine.clear_prefill_hcs_guard_store_addr();
            engine.set_optional_pinning_budget_mb(None);
            engine.clear_prefill_runtime_chunk_cap();
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e),
            );
            return;
        }
        let pinning_budget_mb = {
            let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
            store.prefill_optional_pinning_budget_mb(
                input_token_ids.len(),
                engine.last_prepare_post_alloc_free_mb(),
            )
        };
        engine.set_optional_pinning_budget_mb(pinning_budget_mb);
        scratch_tokens_after_prepare = engine.scratch.max_tokens;
        prefill_chunk_size_after_prepare = engine.config.prefill_chunk_size;

        if let Err(e) = engine.set_prefill_device_trace_enabled(
            debug_prefill_device_trace,
            debug_prefill_device_trace_mla_only,
            debug_prefill_device_trace_layer,
            debug_prefill_device_trace_all_layers,
            debug_prefill_device_trace_full_pre_out_proj,
            debug_prefill_device_trace_dims.clone(),
            debug_prefill_device_trace_rows.clone(),
            debug_prefill_device_trace_experts.clone(),
            debug_prefill_device_trace_local_scan_token,
        ) {
            engine.clear_prefill_hcs_guard_store_addr();
            engine.set_read_only_checkpoint_request_enabled(false);
            engine.set_first_token_margin_projection_request_enabled(false);
            engine.set_reference_debug_trace_enabled(false);
            engine.set_optional_pinning_budget_mb(None);
            engine.clear_prefill_runtime_chunk_cap();
            let _ = engine.release_scratch();
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Prefill device trace setup failed: {}"}}"#, e),
            );
            return;
        }
        if debug_mamba2_state_lifecycle_trace {
            let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
            debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
                store,
                "before_prefill_run",
                debug_mamba2_state_layer,
            ));
        }

        let attempt_result = match engine.run_prefill(
            &input_token_ids,
            0.0, // temperature=0 for greedy
            &suppress_tokens,
        ) {
            Ok(r) => match engine.finalize_stage_exact_prefill_kv(r.prompt_len) {
                Ok(()) => Ok(r),
                Err(e) => Err(format!("KV stage export failed: {}", e)),
            },
            Err(e) => Err(e),
        };

        match attempt_result {
            Ok(r) => break Ok(r),
            Err(e) => {
                let current_chunk = engine.scratch.max_tokens;
                let next_retry_cap = engine.cold_staging_retry_chunk_cap();
                if let Some(next_cap) = next_retry_cap {
                    if next_cap < current_chunk && current_chunk > 128 {
                        retry_attempt += 1;
                        if let Some(failure) = engine.last_cold_staging_failure {
                            log::info!(
                                "Retrying reference_test prefill with measured cold-staging chunk cap: attempt={} prompt_tokens={} failed_chunk={} requested_slots={} max_safe_slots={} free_before_mb={} safety_mb={} current_chunk={} next_chunk_cap={} error={}",
                                retry_attempt,
                                input_token_ids.len(),
                                failure.chunk_tokens,
                                failure.requested_slots,
                                failure.max_safe_slots,
                                failure.free_before_mb,
                                failure.safety_mb,
                                current_chunk,
                                next_cap,
                                e,
                            );
                        } else {
                            log::info!(
                                "Retrying reference_test prefill with measured cold-staging chunk cap: attempt={} prompt_tokens={} current_chunk={} next_chunk_cap={} error={}",
                                retry_attempt,
                                input_token_ids.len(),
                                current_chunk,
                                next_cap,
                                e,
                            );
                        }
                        let _ = engine.set_prefill_device_trace_enabled(
                            false,
                            false,
                            debug_prefill_device_trace_layer,
                            false,
                            false,
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            None,
                        );
                        engine.set_optional_pinning_budget_mb(None);
                        if let Err(release_err) = engine.release_scratch() {
                            log::error!(
                                "reference_test: failed to release scratch before retry: {}",
                                release_err
                            );
                            abort_if_cuda_context_poisoned(
                                "reference_test retry release_scratch",
                                &release_err,
                            );
                            break Err(release_err);
                        }
                        retry_cap = Some(next_cap);
                        continue;
                    }
                }
                break Err(e);
            }
        }
    };
    if prefill_result.is_ok() {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let primary_device = store.device_ordinal();
        let free_now_mb = store.query_vram_free_mb();
        let prefill_min_free_mb = crate::vram_monitor::current_request_lows()
            .into_iter()
            .find(|(device, _)| *device == primary_device)
            .map(|(_, free_mb)| free_mb as usize)
            .unwrap_or(free_now_mb);
        engine.update_measured_prefill_runtime_overhead_mb(
            engine.last_prepare_post_alloc_free_mb(),
            prefill_min_free_mb,
        );
    }
    let debug_prefill_stage_trace = if debug_reference_trace {
        engine.take_reference_debug_trace()
    } else {
        None
    };
    let debug_prefill_device_trace_json = if debug_prefill_device_trace {
        engine.take_prefill_device_trace()
    } else {
        None
    };
    engine.set_read_only_checkpoint_request_enabled(false);
    engine.set_first_token_margin_projection_request_enabled(false);
    engine.set_reference_debug_trace_enabled(false);
    engine.clear_reference_router_variant_override();
    engine.clear_reference_mamba2_gated_norm_replay();
    if debug_mamba2_state_lifecycle_trace {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store,
            "after_prefill_before_result_handling",
            debug_mamba2_state_layer,
        ));
    }

    let (first_token, prompt_len, first_token_top_k, debug_prefill_logits) = match prefill_result {
        Ok(r) => {
            let first_token = r.first_token as usize;
            let first_token_top_k = crate::decode::extract_top_logprobs(
                &engine.h_logits,
                engine.h_logits.len(),
                top_logprobs,
            );
            let debug_prefill_logits = if debug_reference_trace || debug_prompt_trace {
                Some(reference_logit_trace_json(
                    &engine.h_logits,
                    engine.h_logits.len(),
                    first_token,
                    top_logprobs,
                ))
            } else {
                None
            };
            (
                first_token,
                r.prompt_len,
                first_token_top_k,
                debug_prefill_logits,
            )
        }
        Err(e) => {
            abort_if_cuda_context_poisoned("reference_test prefill", &e);
            engine.clear_prefill_hcs_guard_store_addr();
            let _ = engine.set_prefill_device_trace_enabled(
                false,
                false,
                debug_prefill_device_trace_layer,
                false,
                false,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
            );
            engine.clear_reference_mamba2_gated_norm_replay();
            let _ = engine.release_scratch();
            engine.set_optional_pinning_budget_mb(None);
            engine.clear_prefill_runtime_chunk_cap();
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Prefill failed: {}"}}"#, e),
            );
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };

    // Release scratch to free VRAM for decode/HCS
    if let Err(e) = engine.release_scratch() {
        log::error!("reference_test: Failed to release scratch: {}", e);
        abort_if_cuda_context_poisoned("reference_test release_scratch", &e);
    }
    engine.set_optional_pinning_budget_mb(None);
    engine.clear_prefill_hcs_guard_store_addr();
    engine.clear_prefill_runtime_chunk_cap();
    if debug_mamba2_state_lifecycle_trace {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store,
            "after_prefill_scratch_release_before_decode_restore",
            debug_mamba2_state_layer,
        ));
    }

    let prompt_hcs_snapshot = engine.prompt_hcs_shadow_snapshot();

    // Set KV position and swap to simple INT4 for decode
    {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        if let Err(e) = restore_store_after_rust_prefill(store, prompt_len) {
            log::error!("reference_test: Failed to restore decode runtime: {}", e);
        }
        store.set_rope_position_delta(0);
    }

    let prefill_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    // ── Reload soft HCS after prefill ──
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
        log::info!(
            "reference_test: prompt-HCS snapshot ready: prompt_tokens={} layers={} experts={}",
            prompt_tokens,
            layers,
            experts,
        );
        store.install_prompt_hcs_counts(counts.clone(), *layers, *experts, *prompt_tokens);
    } else {
        log::warn!("reference_test: prompt-HCS snapshot missing before reload");
        store.clear_prompt_hcs_counts();
    }
    let (activated, dma_ms) = store.hcs_reload_after_prefill(prompt_len);
    let queued = activated;
    let alloc_mb = store.last_soft_reload_alloc_mb();
    if activated > 0 {
        log::info!(
            "reference_test: HCS reload complete: {} experts, {:.1}ms",
            activated,
            dma_ms
        );
    }
    if debug_hcs_transition_trace {
        let raw = store.hcs_debug_summary_json("after_hcs_reload_after_prefill_before_decode");
        let mut value = serde_json::from_str(&raw).unwrap_or_else(|e| {
            serde_json::json!({
                "phase": "after_hcs_reload_after_prefill_before_decode",
                "available": false,
                "error": format!("parse_failed: {}", e),
                "raw": raw,
            })
        });
        if let Some(obj) = value.as_object_mut() {
            obj.insert("reload_activated".to_string(), serde_json::json!(activated));
            obj.insert("reload_dma_ms".to_string(), serde_json::json!(dma_ms));
            obj.insert("reload_alloc_mb".to_string(), serde_json::json!(alloc_mb));
        }
        debug_hcs_transition_points.push(value);
    }
    if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
        store.install_prompt_hcs_shadow(counts.clone(), *layers, *experts, *prompt_tokens);
    } else {
        store.clear_prompt_hcs_shadow();
    }
    if debug_mamba2_state_lifecycle_trace {
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store,
            "after_hcs_reload_before_decode",
            debug_mamba2_state_layer,
        ));
    }

    // Disable thinking suppression for reference test (greedy, no thinking budget logic)
    store.set_think_end_suppress(None, 0);
    store.set_min_new_tokens_ext(0, vec![]);
    let gqa_diag_layer = std::env::var("KRASIS_GQA_DIAG_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(layer_idx) = gqa_diag_layer {
        store.set_debug_gqa_diag_layer(Some(layer_idx));
        log::info!(
            "reference_test: enabled GQA decode diagnostic capture for layer {}",
            layer_idx
        );
    }

    // ── Greedy decode with logprobs collection ──
    let t_decode = Instant::now();
    let tokenizer = &state.tokenizer;

    // Collect all output tokens and their top-k logprobs
    let mut output_tokens: Vec<(usize, Vec<(u32, f32)>)> = Vec::new();
    let mut all_text = String::new();
    let mut finish_reason = "length".to_string();

    // First token
    let first_text = tokenizer
        .decode(&[first_token as u32], true)
        .unwrap_or_default();
    all_text.push_str(&first_text);
    output_tokens.push((first_token, first_token_top_k.clone()));

    let decode_budget = max_tokens.saturating_sub(1);
    let reload_pending_at_decode_start = store.hcs_soft_reload_pending();
    if debug_decode_state_trace {
        store.set_debug_decode_state_trace_once(true);
    }
    if debug_decode_hcs_equiv_trace {
        store.set_debug_decode_hcs_equiv_trace_once(Some(debug_decode_hcs_equiv_layer));
    }
    if debug_decode_early_trace {
        store.set_debug_decode_early_trace_once(true);
        store.set_debug_decode_early_max_steps_once(debug_decode_early_trace_max_steps);
        store.set_debug_decode_early_detail_dims_once(debug_prefill_device_trace_dims.clone());
    }
    if debug_hcs_transition_trace {
        store.set_debug_hcs_transition_trace_once(true);
    }

    {
        let mut on_token = |token_id: usize,
                            text: &str,
                            fr: Option<&str>,
                            token_logprobs: Option<&[(u32, f32)]>|
         -> bool {
            all_text.push_str(text);
            let lps = token_logprobs.map(|s| s.to_vec()).unwrap_or_default();
            output_tokens.push((token_id, lps));
            if let Some(r) = fr {
                finish_reason = r.to_string();
            }
            true
        };

        store.gpu_generate_stream(
            first_token,
            prompt_len,
            decode_budget,
            0.0, // temperature=0 (greedy)
            1,   // top_k=1 (greedy)
            1.0, // top_p=1.0
            &stop_ids,
            tokenizer,
            &[],
            0.0, // no presence penalty
            top_logprobs,
            Some("reference_test".to_string()),
            on_token,
        );
    }
    if debug_mamba2_state_lifecycle_trace {
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store,
            "after_decode_before_cleanup",
            debug_mamba2_state_layer,
        ));
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let mut debug_decode_state = if debug_decode_state_trace {
        let raw =
            store.config_validation_snapshot_json(prompt_len, true, reload_pending_at_decode_start);
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(value) => Some(value),
            Err(e) => Some(serde_json::json!({
                "available": false,
                "error": format!("decode state trace parse failed: {}", e),
                "raw": raw,
            })),
        }
    } else {
        None
    };
    if debug_hcs_transition_trace {
        if let Some(value) = debug_decode_state.as_mut() {
            value["server_hcs_transition_points"] = serde_json::json!(debug_hcs_transition_points);
        }
    }
    if gqa_diag_layer.is_some() {
        if let Ok(path) = std::env::var("KRASIS_GQA_DIAG_DUMP") {
            match store.debug_gqa_diag_json() {
                Ok(payload) => {
                    if let Err(e) = std::fs::write(&path, payload) {
                        log::error!(
                            "reference_test: failed to write GQA diagnostic {}: {}",
                            path,
                            e
                        );
                    } else {
                        log::info!("reference_test: wrote GQA diagnostic {}", path);
                    }
                }
                Err(e) => {
                    log::error!("reference_test: failed to capture GQA diagnostic: {}", e);
                }
            }
        }
        store.set_debug_gqa_diag_layer(None);
    }

    // ── Cleanup ──
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
    let server_cleanup_called = true;
    if debug_mamba2_state_lifecycle_trace {
        debug_mamba2_state_lifecycle_points.push(mamba2_state_lifecycle_point(
            store,
            "after_server_cleanup",
            debug_mamba2_state_layer,
        ));
        if let Some(value) = debug_decode_state.as_mut() {
            value["mamba2_state_lifecycle_trace"] = serde_json::json!({
                "active": true,
                "layer": debug_mamba2_state_layer,
                "entry_count": debug_mamba2_state_lifecycle_points.len(),
                "entries": debug_mamba2_state_lifecycle_points,
            });
        }
    }

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::update_request_context(&format!(
        "route=/v1/internal/reference_test request_order={} model={} prompt_tokens={} max_new={} phase=complete",
        reference_request_order, state.model_name, prompt_len, max_tokens,
    ));
    let vram_low_water = serde_json::Value::Array(
        crate::vram_monitor::current_request_lows()
            .into_iter()
            .map(|(device, min_free_mb)| {
                serde_json::json!({
                    "device": device,
                    "min_free_mb": min_free_mb,
                })
            })
            .collect(),
    );
    let safety_margin_mb = store.hcs_safety_margin_mb();

    // ── Format response ──
    let mut per_token_json = Vec::new();
    for (tid, logprobs) in &output_tokens {
        let mut tk_json = Vec::new();
        for &(lp_tid, lp_val) in logprobs {
            tk_json.push(format!(
                r#"{{"token_id":{},"log_prob":{:.6}}}"#,
                lp_tid, lp_val
            ));
        }
        // Get log_prob for the selected token (first in top-k if available)
        let selected_lp = logprobs
            .iter()
            .find(|&&(t, _)| t == *tid as u32)
            .map(|&(_, lp)| lp)
            .unwrap_or(0.0);
        per_token_json.push(format!(
            r#"{{"token_id":{},"log_prob":{:.6},"top_k":[{}]}}"#,
            tid,
            selected_lp,
            tk_json.join(",")
        ));
    }

    // Escape text for JSON
    let text_escaped = serde_json::to_string(&all_text).unwrap_or_else(|_| "\"\"".to_string());

    let mut first_topk_json = Vec::new();
    for &(lp_tid, lp_val) in &first_token_top_k {
        first_topk_json.push(format!(
            r#"{{"token_id":{},"log_prob":{:.6}}}"#,
            lp_tid, lp_val
        ));
    }

    let reference_prompt_debug = if debug_prompt_trace {
        Some(serde_json::json!({
            "schema": "krasis_reference_first_token_boundary_debug_v1",
            "route": "/v1/internal/reference_test",
            "input_source": "input_token_ids",
            "input_token_count": input_token_ids.len(),
            "input_token_hash_fnv1a64": format!("0x{:016x}", input_token_hash),
            "input_token_ids": input_token_ids.clone(),
            "selected_token_id": first_token,
            "selected_token_text": tokenizer.decode(&[first_token as u32], true).unwrap_or_default(),
            "prompt_len": prompt_len,
            "debug_reference_trace_enabled": debug_reference_trace,
            "first_token_logits": debug_prefill_logits
                .clone()
                .unwrap_or_else(|| serde_json::json!({"available": false})),
        }))
    } else {
        None
    };

    let debug_router_variant_json = if debug_router_variant_requested {
        let override_layer_count = debug_router_e_score_corr_by_layer
            .iter()
            .filter(|entry| entry.is_some())
            .count();
        Some(serde_json::json!({
            "schema": "krasis_reference_test_router_variant_v1",
            "scope": "/v1/internal/reference_test",
            "variant": debug_router_variant.as_str(),
            "layer_scope": if debug_router_variant_layers.is_empty() {
                serde_json::json!("all")
            } else {
                serde_json::json!(debug_router_variant_layers)
            },
            "production_default": "raw",
            "enabled_by_default": false,
            "e_score_correction_override_layers": override_layer_count,
            "e_score_correction_override_source": if override_layer_count > 0 {
                "request_fp32_by_layer"
            } else {
                "registered_graph_ptr"
            },
        }))
    } else {
        None
    };
    let debug_router_forced_slot_orders_json = if debug_router_forced_slot_orders_requested {
        let entries = debug_router_forced_slot_orders
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "layer": entry.layer_idx,
                    "row": entry.row_idx,
                    "expert_ids": entry.expert_ids,
                    "weight_source": "raw_sigmoid_score_for_forced_expert",
                })
            })
            .collect::<Vec<_>>();
        Some(serde_json::json!({
            "schema": "krasis_reference_test_router_forced_slot_orders_v1",
            "scope": "/v1/internal/reference_test",
            "enabled_by_default": false,
            "production_default": "raw",
            "entries": entries,
        }))
    } else {
        None
    };
    let debug_mamba2_gated_norm_replay_json = if debug_mamba2_gated_norm_replay_requested {
        let entries = debug_mamba2_gated_norm_replay
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "layer": entry.layer_idx,
                    "row": entry.row_idx,
                    "mode": entry.mode.as_str(),
                    "operation": "sqrt.approx.ftz.f32 + div.rn.f32",
                })
            })
            .collect::<Vec<_>>();
        Some(serde_json::json!({
            "schema": "krasis_reference_test_mamba2_gated_norm_replay_v1",
            "scope": "/v1/internal/reference_test",
            "enabled_by_default": false,
            "production_default": "mamba2_gated_group_rmsnorm_kernel",
            "entries": entries,
        }))
    } else {
        None
    };

    let mut debug_json_suffix = String::new();
    if let Some(prompt_debug) = reference_prompt_debug.as_ref() {
        debug_json_suffix.push_str(&format!(r#","debug_prompt_trace":{}"#, prompt_debug));
    }
    if let Some(router_variant) = debug_router_variant_json.as_ref() {
        debug_json_suffix.push_str(&format!(r#","debug_router_variant":{}"#, router_variant));
    }
    if let Some(forced_slots) = debug_router_forced_slot_orders_json.as_ref() {
        debug_json_suffix.push_str(&format!(
            r#","debug_router_forced_slot_orders":{}"#,
            forced_slots
        ));
    }
    if let Some(replay) = debug_mamba2_gated_norm_replay_json.as_ref() {
        debug_json_suffix.push_str(&format!(r#","debug_mamba2_gated_norm_replay":{}"#, replay));
    }
    if let Some(prefill_device_trace) = debug_prefill_device_trace_json.as_ref() {
        debug_json_suffix.push_str(&format!(
            r#","debug_prefill_device_trace":{}"#,
            prefill_device_trace
        ));
    }
    if let Some(decode_state) = debug_decode_state.as_ref() {
        debug_json_suffix.push_str(&format!(r#","debug_decode_state_trace":{}"#, decode_state));
    }

    if debug_reference_trace {
        let final_top_logprobs: Vec<serde_json::Value> = first_token_top_k
            .iter()
            .enumerate()
            .map(|(rank, &(token_id, log_prob))| {
                serde_json::json!({
                    "rank": rank + 1,
                    "token_id": token_id,
                    "log_prob": log_prob as f64,
                })
            })
            .collect();
        let selected_logprob_from_endpoint = first_token_top_k
            .iter()
            .find(|&&(token_id, _)| token_id == first_token as u32)
            .map(|&(_, log_prob)| log_prob as f64);
        let trace = serde_json::json!({
            "schema": "krasis_reference_test_debug_v1",
            "request_order": reference_request_order,
            "client_request_id": client_request_id,
            "input_token_count": input_token_ids.len(),
            "input_token_hash_fnv1a64": format!("0x{:016x}", input_token_hash),
            "max_tokens": max_tokens,
            "top_logprobs": top_logprobs,
            "stop_token_ids": stop_ids,
            "debug_router_variant": debug_router_variant_json.clone().unwrap_or_else(|| serde_json::json!({"available": false})),
            "selected_token_id": first_token,
            "prompt_len": prompt_len,
            "state_reset_proof": {
                "fresh_prefill_run": true,
                "run_prefill_zeroes_la_state": true,
                "hcs_evict_for_prefill_called": true,
                "hcs_evicted_experts": evicted,
                "hcs_freed_mb": freed_mb,
                "hcs_snapshot_entries": hcs_snapshot_entries,
                "hcs_num_experts_per_layer": hcs_num_experts_per_layer,
                "prepare_runtime_for_prefill_called": true,
                "has_hqq_runtime_slots": has_hqq_runtime_slots,
                "hqq_prefill_materialized": hqq_prefill_materialized,
                "prepare_for_prefill_prompt_tokens": input_token_ids.len(),
                "scratch_tokens_after_prepare": scratch_tokens_after_prepare,
                "prefill_chunk_size_after_prepare": prefill_chunk_size_after_prepare,
                "release_scratch_called": true,
                "restore_runtime_for_decode_called": true,
                "decode_kv_position_set_to_prompt_len": prompt_len,
                "hcs_reload_after_prefill_queued": queued,
                "hcs_reload_after_prefill_alloc_mb": alloc_mb,
                "hcs_sync_soft_reload_activated": activated,
                "hcs_sync_soft_reload_dma_ms": dma_ms,
                "server_cleanup_called": server_cleanup_called
            },
            "prefill_stage_trace": debug_prefill_stage_trace.unwrap_or_else(|| serde_json::json!({"available": false})),
            "prefill_logits": debug_prefill_logits.unwrap_or_else(|| serde_json::json!({"available": false})),
            "prompt_debug": reference_prompt_debug.clone().unwrap_or_else(|| serde_json::json!({"available": false})),
            "final_top_logprobs": final_top_logprobs,
            "selected_logprob_from_endpoint": selected_logprob_from_endpoint,
            "timing": {
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "prompt_tokens": prompt_len,
                "vram_low_water": vram_low_water.clone(),
                "safety_margin_mb": safety_margin_mb,
            }
        });
        debug_json_suffix.push_str(&format!(r#","debug_reference_trace":{}"#, trace));
    }

    let response = format!(
        r#"{{"token_ids":[{}],"text":{},"num_tokens":{},"per_token_data":[{}],"first_token_top_k":[{}],"finish_reason":"{}","timing":{{"prefill_ms":{:.1},"decode_ms":{:.1},"total_ms":{:.1},"prompt_tokens":{},"vram_low_water":{},"safety_margin_mb":{}}}{}}}"#,
        output_tokens
            .iter()
            .map(|(t, _)| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        text_escaped,
        output_tokens.len(),
        per_token_json.join(","),
        first_topk_json.join(","),
        finish_reason,
        prefill_ms,
        decode_ms,
        total_ms,
        prompt_len,
        vram_low_water,
        safety_margin_mb,
        debug_json_suffix
    );

    log::info!(
        "reference_test: {} output tokens in {:.0}ms (prefill={:.0}ms decode={:.0}ms), finish={}",
        output_tokens.len(),
        total_ms,
        prefill_ms,
        decode_ms,
        finish_reason
    );

    let _ = send_json(stream, 200, &response);
}

/// GPU decode: GIL-free Rust decode loop via GpuDecodeStore.
/// Pure Rust, zero Python per token.
#[derive(Debug)]
struct DecodeTransactionOutcome {
    /// Tokens which were actually consumed by decode and are therefore
    /// represented by the live GPU sequence state. The final emitted token is
    /// intentionally absent because it has not yet been fed back into decode.
    consumed_generation_tokens: Vec<u32>,
    completed: bool,
    failure_reason: Option<String>,
    generated_tokens: usize,
    finish_reason: String,
    client_aborted: bool,
}

impl DecodeTransactionOutcome {
    fn failed(reason: impl Into<String>) -> Self {
        Self {
            consumed_generation_tokens: Vec::new(),
            completed: false,
            failure_reason: Some(reason.into()),
            generated_tokens: 0,
            finish_reason: "error".to_string(),
            client_aborted: false,
        }
    }
}

fn consumed_generation_boundary(
    first_token: usize,
    emitted_token_ids: &[u32],
    reported_generated: usize,
) -> Result<Vec<u32>, String> {
    if reported_generated != emitted_token_ids.len() {
        return Err(format!(
            "decode reported {} generated tokens but callback observed {}",
            reported_generated,
            emitted_token_ids.len()
        ));
    }
    if reported_generated == 0 {
        return Ok(Vec::new());
    }
    let first_token = u32::try_from(first_token)
        .map_err(|_| format!("first generated token ID {first_token} exceeds u32"))?;
    let mut consumed = Vec::with_capacity(reported_generated);
    consumed.push(first_token);
    consumed.extend_from_slice(&emitted_token_ids[..reported_generated - 1]);
    Ok(consumed)
}

#[allow(clippy::too_many_arguments)]
fn handle_gpu_decode(
    stream: &mut TcpStream,
    is_stream: bool,
    state: &ServerState,
    store: &mut GpuDecodeStore,
    tokenizer: &tokenizers::Tokenizer,
    first_token: usize,
    prompt_len: usize,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    presence_penalty: f32,
    stop_ids: &[usize],
    request_id: &str,
    model_name: &str,
    created: u64,
    overhead: &RequestOverhead,
    has_tools: bool,
    tool_call_format: crate::chat_template::ToolCallFormat,
    enable_thinking: bool,
    logprobs_top_n: usize,
    chat_debug_payload: Option<serde_json::Value>,
    serving_metrics: &Arc<ServingMetrics>,
) -> DecodeTransactionOutcome {
    let mut chat_debug_payload = chat_debug_payload;
    // Resolve thinking end token early — used by both streaming and non-streaming paths
    let think_end_id = if enable_thinking {
        state.thinking_end_token
    } else {
        None
    };
    let hidden_think_stop_id = if enable_thinking {
        None
    } else {
        state.thinking_end_token
    };
    let preserved_tool_special_tokens = if has_tools {
        tool_call_format.preserved_special_tokens()
    } else {
        &[]
    };

    if is_stream {
        if let Err(e) = begin_sse(stream) {
            log::error!("Failed to send SSE headers: {}", e);
            return DecodeTransactionOutcome::failed(format!("send SSE headers: {e}"));
        }

        let first_text = decode_token_preserving_tool_specials(
            tokenizer,
            first_token as u32,
            preserved_tool_special_tokens,
        );

        // When thinking is enabled, inject <think> at start of stream.
        // The prompt already includes <think>, but the client needs it in the
        // output to know this is a thinking block (for display suppression).
        if think_end_id.is_some() {
            let think_chunk =
                format_sse_token(request_id, model_name, "<think>", None, created, None);
            if let Err(error) = send_sse_chunk(stream, &think_chunk) {
                return DecodeTransactionOutcome::failed(format!(
                    "send initial thinking SSE chunk: {error}"
                ));
            }
        }

        // When tool use is active, buffer first token (might need tool call parsing).
        // Otherwise send immediately for lowest latency.
        if !has_tools {
            let chunk = format_sse_token(request_id, model_name, &first_text, None, created, None);
            if let Err(error) = send_sse_chunk(stream, &chunk) {
                return DecodeTransactionOutcome::failed(format!(
                    "send initial token SSE chunk: {error}"
                ));
            }
        }

        let (tx, rx) = mpsc::channel::<String>();
        let writer_disconnected = Arc::new(AtomicBool::new(false));
        let writer_disc_clone = writer_disconnected.clone();

        let mut writer_stream = match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to clone stream for writer: {}", e);
                return DecodeTransactionOutcome::failed(format!("clone SSE stream: {e}"));
            }
        };

        let writer_handle = std::thread::spawn(move || {
            let flush_interval = std::time::Duration::from_millis(100);
            let mut buf = String::new();
            let mut last_flush = Instant::now();
            let mut is_first = true;
            loop {
                match rx.recv_timeout(flush_interval) {
                    Ok(chunk) => {
                        buf.push_str(&chunk);
                        if is_first || last_flush.elapsed() >= flush_interval || buf.len() > 8192 {
                            if writer_stream.write_all(buf.as_bytes()).is_err()
                                || writer_stream.flush().is_err()
                            {
                                writer_disc_clone.store(true, Ordering::Release);
                                return;
                            }
                            buf.clear();
                            last_flush = Instant::now();
                            is_first = false;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if !buf.is_empty() {
                            if writer_stream.write_all(buf.as_bytes()).is_err()
                                || writer_stream.flush().is_err()
                            {
                                writer_disc_clone.store(true, Ordering::Release);
                                return;
                            }
                            buf.clear();
                            last_flush = Instant::now();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        if !buf.is_empty() {
                            if writer_stream.write_all(buf.as_bytes()).is_err()
                                || writer_stream.flush().is_err()
                            {
                                writer_disc_clone.store(true, Ordering::Release);
                            }
                        }
                        return;
                    }
                }
            }
        });

        let decode_start = Instant::now();
        let mut last_token_at = Instant::now();
        let mut decode_token_count = 0usize;
        let mut emitted_token_ids = Vec::new();
        let mut delivery_failed = false;

        // ── Thinking budget tracking ──
        // When thinking is enabled, tokens inside <think>...</think> are exempt
        // from max_tokens. We track the state and only count answer tokens.
        let mut in_thinking = think_end_id.is_some(); // start in thinking if enabled
        let mut answer_token_count = 0usize;
        let mut thinking_token_count = 0usize;
        // Also check first_token — it could be </think> for trivial thinking
        if in_thinking && Some(first_token) == think_end_id {
            in_thinking = false;
        } else if in_thinking {
            thinking_token_count += 1;
        }

        // ── Tool call detection state ──
        // Stream safe content immediately, retain a possible split marker,
        // then buffer from the first complete native marker for structured
        // parsing after generation.
        let tool_marker = tool_call_format.start_marker().unwrap_or("");
        let mut tc_pending = String::new();
        let mut tc_captured = String::new();
        let mut tc_found = false;
        let mut tc_finish = String::new();
        let mut terminal_finish_reason = "length".to_string();
        let mut emitted_tool_calls = false;

        if has_tools {
            let visible = push_tool_stream_text(
                tool_marker,
                &mut tc_pending,
                &mut tc_captured,
                &mut tc_found,
                &first_text,
            );
            if !visible.is_empty() {
                let chunk = format_sse_token(request_id, model_name, &visible, None, created, None);
                if tx.send(format!("data: {}\n\n", chunk)).is_err() {
                    delivery_failed = true;
                }
            }
        }

        // Shared callback for both single-GPU and multi-GPU decode
        let mut on_token = |token_id: usize,
                            text: &str,
                            finish_reason: Option<&str>,
                            token_logprobs: Option<&[(u32, f32)]>|
         -> bool {
            let token_at = Instant::now();
            serving_metrics.token_generated(
                token_at.duration_since(last_token_at).as_secs_f64(),
                prompt_len
                    .saturating_add(decode_token_count)
                    .saturating_add(1),
            );
            last_token_at = token_at;
            decode_token_count += 1;
            emitted_token_ids.push(token_id as u32);

            // ── Track thinking state ──
            // Tokens before </think> are "thinking" and don't count against max_tokens.
            if think_end_id.is_some() {
                if in_thinking {
                    thinking_token_count += 1;
                    if Some(token_id) == think_end_id {
                        in_thinking = false;
                        log::info!("Thinking complete: {} tokens", thinking_token_count);
                    }
                } else {
                    answer_token_count += 1;
                }
            }

            // Override finish_reason if answer token limit reached
            let effective_finish = if finish_reason.is_some() {
                finish_reason
            } else if think_end_id.is_some() && !in_thinking && answer_token_count >= max_tokens {
                Some("length")
            } else {
                None
            };
            if let Some(reason) = effective_finish {
                terminal_finish_reason = reason.to_string();
            }
            let hide_text =
                hide_synthetic_think_stop_text(token_id, effective_finish, hidden_think_stop_id);
            let visible_text = if hide_text { "" } else { text };

            if has_tools {
                if let Some(fr) = effective_finish {
                    tc_finish = fr.to_string();
                }

                let visible = push_tool_stream_text(
                    tool_marker,
                    &mut tc_pending,
                    &mut tc_captured,
                    &mut tc_found,
                    visible_text,
                );
                if !visible.is_empty() {
                    let chunk = format_sse_token(
                        request_id,
                        model_name,
                        &visible,
                        None,
                        created,
                        token_logprobs,
                    );
                    if tx.send(format!("data: {}\n\n", chunk)).is_err() {
                        delivery_failed = true;
                        return false;
                    }
                }

                if writer_disconnected.load(Ordering::Acquire) {
                    return false;
                }
                if effective_finish.is_some() {
                    return false;
                }
                true
            } else {
                // Original non-tool path
                let chunk = format_sse_token(
                    request_id,
                    model_name,
                    visible_text,
                    effective_finish,
                    created,
                    token_logprobs,
                );
                let formatted = format!("data: {}\n\n", chunk);
                if tx.send(formatted).is_err() || writer_disconnected.load(Ordering::Acquire) {
                    delivery_failed = true;
                    return false;
                }
                // Stop if answer limit reached
                if effective_finish.is_some() {
                    return false;
                }
                true
            }
        };

        // When thinking is enabled, give the decode loop extra budget for thinking tokens.
        // The on_token callback enforces the real max_tokens on answer tokens only.
        let requested_decode_budget = if think_end_id.is_some() {
            max_tokens.saturating_add(32768).saturating_sub(1)
        } else {
            max_tokens.saturating_sub(1)
        };
        let context_decode_budget = state
            .max_context_tokens
            .saturating_sub(prompt_len)
            .saturating_sub(1);
        let decode_budget = requested_decode_budget.min(context_decode_budget);

        let generated = if !state.aux_gpu_store_addrs.is_empty() {
            // Multi-GPU decode: pipeline across N GPUs
            store.gpu_generate_stream_multi(
                &state.aux_gpu_store_addrs,
                &state.multi_gpu_split_layers,
                &state.multi_gpu_gqa_offsets,
                first_token,
                prompt_len,
                decode_budget,
                temperature,
                top_k,
                top_p,
                stop_ids,
                tokenizer,
                preserved_tool_special_tokens,
                presence_penalty,
                logprobs_top_n,
                Some(format!("chat_{}", request_id)),
                &mut on_token,
            )
        } else {
            // Single-GPU decode
            store.gpu_generate_stream(
                first_token,
                prompt_len,
                decode_budget,
                temperature,
                top_k,
                top_p,
                stop_ids,
                tokenizer,
                preserved_tool_special_tokens,
                presence_penalty,
                logprobs_top_n,
                Some(format!("chat_{}", request_id)),
                &mut on_token,
            )
        };

        // Capture decode timing BEFORE post-generation processing (tool call parsing etc.)
        let decode_elapsed = decode_start.elapsed().as_secs_f64();

        // ── Post-generation: emit tool calls or finish ──
        if has_tools {
            let (unstreamed_content, tool_calls) = if tc_found {
                parse_tool_calls(&tc_captured, tool_call_format)
            } else {
                (tc_pending.clone(), Vec::new())
            };
            if !unstreamed_content.is_empty() {
                let content_chunk = format_sse_token(
                    request_id,
                    model_name,
                    &unstreamed_content,
                    None,
                    created,
                    None,
                );
                if tx.send(format!("data: {}\n\n", content_chunk)).is_err() {
                    delivery_failed = true;
                }
            }
            if !tool_calls.is_empty() {
                emitted_tool_calls = true;
                for (i, tc) in tool_calls.iter().enumerate() {
                    let start_chunk = format_sse_tool_call_start(
                        request_id, model_name, i, &tc.id, &tc.name, created,
                    );
                    if tx.send(format!("data: {}\n\n", start_chunk)).is_err() {
                        delivery_failed = true;
                    }
                    let args_chunk = format_sse_tool_call_args(
                        request_id,
                        model_name,
                        i,
                        &tc.arguments_json,
                        created,
                    );
                    if tx.send(format!("data: {}\n\n", args_chunk)).is_err() {
                        delivery_failed = true;
                    }
                }
                let finish_chunk = format_sse_token(
                    request_id,
                    model_name,
                    "",
                    Some("tool_calls"),
                    created,
                    None,
                );
                if tx.send(format!("data: {}\n\n", finish_chunk)).is_err() {
                    delivery_failed = true;
                }
                log::info!(
                    "Request {}: {} tool call(s) detected",
                    request_id,
                    tool_calls.len()
                );
            } else {
                let fr = if tc_finish.is_empty() {
                    "stop"
                } else {
                    &tc_finish
                };
                let finish_chunk =
                    format_sse_token(request_id, model_name, "", Some(fr), created, None);
                if tx.send(format!("data: {}\n\n", finish_chunk)).is_err() {
                    delivery_failed = true;
                }
            }
        }

        let elapsed = decode_elapsed;
        let total_gen = decode_token_count + 1;
        let (reported_thinking_tokens, reported_answer_tokens) = if think_end_id.is_some() {
            (thinking_token_count, answer_token_count)
        } else {
            // With thinking disabled every generated token is an answer token.
            // total_gen includes the first token produced by prefill.
            (0, total_gen)
        };
        let decode_tok_s = if elapsed > 0.0 && decode_token_count > 0 {
            decode_token_count as f64 / elapsed
        } else {
            0.0
        };
        let decode_ms = elapsed * 1000.0;
        let prefill_tok_s = if overhead.prefill_ms > 0.0 && prompt_len > 0 {
            prompt_len as f64 / (overhead.prefill_ms / 1000.0)
        } else {
            0.0
        };
        let overhead_total_ms =
            overhead.parse_ms + overhead.evict_ms + overhead.prefill_ms + overhead.reload_ms;
        let timing_chunk = format_sse_timing(
            request_id,
            model_name,
            created,
            decode_token_count,
            decode_ms,
            decode_tok_s,
            reported_thinking_tokens,
            reported_answer_tokens,
            total_gen,
            prompt_len,
            prefill_tok_s,
            overhead_total_ms,
            overhead,
        );
        if tx.send(format!("data: {}\n\n", timing_chunk)).is_err() {
            delivery_failed = true;
        }
        if tx.send("data: [DONE]\n\n".to_string()).is_err() {
            delivery_failed = true;
        }
        drop(tx);
        let _ = writer_handle.join();

        let engine_failure = store.last_stream_failure_rust().map(str::to_string);
        let disconnected = writer_disconnected.load(Ordering::Acquire);
        let consumed_boundary =
            consumed_generation_boundary(first_token, &emitted_token_ids, generated);
        let completed = engine_failure.is_none()
            && !delivery_failed
            && !disconnected
            && consumed_boundary.is_ok();
        let failure_reason = if let Some(failure) = engine_failure {
            Some(failure)
        } else if delivery_failed {
            Some("SSE response delivery failed".to_string())
        } else if disconnected {
            Some("client disconnected during SSE response".to_string())
        } else if let Err(error) = consumed_boundary.as_ref() {
            Some(error.clone())
        } else {
            None
        };
        let consumed_generation_tokens = if completed {
            consumed_boundary.unwrap_or_default()
        } else {
            Vec::new()
        };

        log::info!(
            "Request {} complete: decode={:.2}s ({} tok, {:.1} tok/s) | overhead={:.0}ms (parse={:.1} evict={:.1} prefill={:.0} reload={:.0})",
            request_id, elapsed, total_gen, decode_tok_s,
            overhead_total_ms, overhead.parse_ms, overhead.evict_ms, overhead.prefill_ms, overhead.reload_ms
        );
        DecodeTransactionOutcome {
            consumed_generation_tokens,
            completed,
            failure_reason,
            generated_tokens: total_gen,
            finish_reason: if emitted_tool_calls {
                "tool_calls".to_string()
            } else {
                terminal_finish_reason
            },
            client_aborted: disconnected,
        }
    } else {
        // ── Non-streaming path ──
        let mut all_text = String::new();
        // Inject <think> prefix so clients can identify thinking blocks
        if enable_thinking && state.thinking_end_token.is_some() {
            all_text.push_str("<think>");
        }
        let first_text = decode_token_preserving_tool_specials(
            tokenizer,
            first_token as u32,
            preserved_tool_special_tokens,
        );
        all_text.push_str(&first_text);
        let mut total_tokens = 1usize;
        let mut last_token_at = Instant::now();
        let mut emitted_token_ids = Vec::new();
        let decode_reported_generated;
        let mut finish = "length".to_string();
        let mut debug_output_tokens: Vec<(usize, Vec<(u32, f32)>)> = Vec::new();
        if chat_debug_payload.is_some() {
            debug_output_tokens.push((first_token, Vec::new()));
        }

        // Thinking budget for non-streaming
        let ns_think_end_id = if enable_thinking {
            state.thinking_end_token
        } else {
            None
        };
        let mut ns_in_thinking = ns_think_end_id.is_some();
        let mut ns_answer_tokens = 0usize;
        if ns_in_thinking && Some(first_token) == ns_think_end_id {
            ns_in_thinking = false;
        }

        let requested_ns_decode_budget = if ns_think_end_id.is_some() {
            max_tokens.saturating_add(32768).saturating_sub(1)
        } else {
            max_tokens.saturating_sub(1)
        };
        let context_decode_budget = state
            .max_context_tokens
            .saturating_sub(prompt_len)
            .saturating_sub(1);
        let ns_decode_budget = requested_ns_decode_budget.min(context_decode_budget);

        {
            let mut on_token = |token_id: usize,
                                text: &str,
                                finish_reason: Option<&str>,
                                token_logprobs: Option<&[(u32, f32)]>|
             -> bool {
                let token_at = Instant::now();
                serving_metrics.token_generated(
                    token_at.duration_since(last_token_at).as_secs_f64(),
                    prompt_len.saturating_add(total_tokens),
                );
                last_token_at = token_at;
                emitted_token_ids.push(token_id as u32);
                let hide_text =
                    hide_synthetic_think_stop_text(token_id, finish_reason, hidden_think_stop_id);
                if !hide_text {
                    all_text.push_str(text);
                }
                total_tokens += 1;
                if chat_debug_payload.is_some() {
                    debug_output_tokens.push((
                        token_id,
                        token_logprobs.map(|s| s.to_vec()).unwrap_or_default(),
                    ));
                }

                // Track thinking state
                if ns_think_end_id.is_some() {
                    if ns_in_thinking {
                        if Some(token_id) == ns_think_end_id {
                            ns_in_thinking = false;
                        }
                    } else {
                        ns_answer_tokens += 1;
                    }
                }

                if let Some(fr) = finish_reason {
                    finish = fr.to_string();
                }

                // Stop if answer limit reached
                if ns_think_end_id.is_some() && !ns_in_thinking && ns_answer_tokens >= max_tokens {
                    finish = "length".to_string();
                    return false;
                }

                true
            };
            decode_reported_generated = if !state.aux_gpu_store_addrs.is_empty() {
                store.gpu_generate_stream_multi(
                    &state.aux_gpu_store_addrs,
                    &state.multi_gpu_split_layers,
                    &state.multi_gpu_gqa_offsets,
                    first_token,
                    prompt_len,
                    ns_decode_budget,
                    temperature,
                    top_k,
                    top_p,
                    stop_ids,
                    tokenizer,
                    preserved_tool_special_tokens,
                    presence_penalty,
                    logprobs_top_n,
                    Some(format!("chat_{}_nosse", request_id)),
                    &mut on_token,
                )
            } else {
                store.gpu_generate_stream(
                    first_token,
                    prompt_len,
                    ns_decode_budget,
                    temperature,
                    top_k,
                    top_p,
                    stop_ids,
                    tokenizer,
                    preserved_tool_special_tokens,
                    presence_penalty,
                    logprobs_top_n,
                    Some(format!("chat_{}_nosse", request_id)),
                    &mut on_token,
                )
            };

            if decode_reported_generated != emitted_token_ids.len() {
                log::error!(
                    "decode reported {} generated tokens but callback observed {}",
                    decode_reported_generated,
                    emitted_token_ids.len()
                );
            }
        }

        if let Some(serde_json::Value::Object(debug)) = chat_debug_payload.as_mut() {
            let token_ids: Vec<usize> = debug_output_tokens.iter().map(|(tid, _)| *tid).collect();
            let per_token: Vec<serde_json::Value> = debug_output_tokens
                .iter()
                .enumerate()
                .map(|(step, (token_id, logprobs))| {
                    let token_text = tokenizer
                        .decode(&[*token_id as u32], true)
                        .unwrap_or_default();
                    let top_k: Vec<serde_json::Value> = logprobs
                        .iter()
                        .map(|&(tid, lp)| {
                            serde_json::json!({
                                "token_id": tid,
                                "log_prob": lp as f64,
                            })
                        })
                        .collect();
                    let selected_log_prob = logprobs
                        .iter()
                        .find(|&&(tid, _)| tid == *token_id as u32)
                        .map(|&(_, lp)| lp as f64);
                    serde_json::json!({
                        "step": step,
                        "source": if step == 0 { "prefill_first_token" } else { "decode" },
                        "token_id": token_id,
                        "token_text": token_text,
                        "selected_log_prob": selected_log_prob,
                        "top_k": top_k,
                    })
                })
                .collect();
            debug.insert(
                "completion_token_ids".to_string(),
                serde_json::json!(token_ids),
            );
            debug.insert(
                "completion_token_count".to_string(),
                serde_json::json!(total_tokens),
            );
            debug.insert(
                "completion_finish_reason".to_string(),
                serde_json::json!(finish),
            );
            debug.insert(
                "completion_decode_trace".to_string(),
                serde_json::json!(per_token),
            );
        }

        let response_delivery = if has_tools {
            let (content, tool_calls) = parse_tool_calls(&all_text, tool_call_format);
            if !tool_calls.is_empty() {
                let response = format_completion_with_tool_calls(
                    request_id,
                    model_name,
                    &content,
                    &tool_calls,
                    prompt_len,
                    total_tokens,
                    created,
                );
                let delivered = send_json(stream, 200, &response);
                log::info!(
                    "Request {}: {} tool call(s) (non-streaming)",
                    request_id,
                    tool_calls.len()
                );
                delivered
            } else {
                let response = format_completion_with_debug(
                    request_id,
                    model_name,
                    &all_text,
                    prompt_len,
                    total_tokens,
                    &finish,
                    created,
                    chat_debug_payload.as_ref(),
                );
                send_json(stream, 200, &response)
            }
        } else {
            let response = format_completion_with_debug(
                request_id,
                model_name,
                &all_text,
                prompt_len,
                total_tokens,
                &finish,
                created,
                chat_debug_payload.as_ref(),
            );
            send_json(stream, 200, &response)
        };

        let engine_failure = store.last_stream_failure_rust().map(str::to_string);
        let consumed_boundary = consumed_generation_boundary(
            first_token,
            &emitted_token_ids,
            decode_reported_generated,
        );
        let completed =
            engine_failure.is_none() && consumed_boundary.is_ok() && response_delivery.is_ok();
        let client_aborted = response_delivery.as_ref().err().is_some_and(|error| {
            matches!(
                error.kind(),
                std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
            )
        });
        let consumed_generation_tokens = if completed {
            consumed_boundary.as_ref().cloned().unwrap_or_default()
        } else {
            Vec::new()
        };
        DecodeTransactionOutcome {
            consumed_generation_tokens,
            completed,
            failure_reason: engine_failure
                .or_else(|| {
                    response_delivery
                        .err()
                        .map(|error| format!("send JSON response: {error}"))
                })
                .or_else(|| consumed_boundary.err()),
            generated_tokens: total_tokens,
            finish_reason: finish,
            client_aborted,
        }
    }
}

/// The Rust HTTP server, exposed to Python via PyO3.
#[pyclass]
pub struct RustServer {
    host: String,
    port: u16,
    model_name: String,
    tokenizer_path: String,
    max_context_tokens: usize,
    default_enable_thinking: bool,
    /// Token ID for `</think>` passed from Python (0 = not available).
    thinking_end_token_id: usize,
    gpu_store_addr: usize,
    py_model: Py<PyAny>,
    running: Arc<AtomicBool>,
    aux_gpu_store_addrs: Vec<usize>,
    multi_gpu_split_layers: Vec<usize>,
    multi_gpu_gqa_offsets: Vec<usize>,
    supports_vision: bool,
    /// Shared Rust prefill engine — used by both serve_forever (HTTP requests)
    /// and benchmark_request (engine benchmarks). Arc+Mutex allows both paths
    /// to share the single pre-allocated engine without moving it.
    prefill_engine: Arc<std::sync::Mutex<Option<crate::gpu_prefill::PrefillEngine>>>,
    /// Enable test-only endpoints (/v1/internal/prefill_logits)
    test_endpoints: bool,
    /// RAM-backed multi-conversation cache. Disabled by default.
    prefix_cache: bool,
    /// Fraction of live cgroup-aware host availability admitted to the cache.
    prefix_cache_ram_fraction: f64,
}

#[pymethods]
impl RustServer {
    #[new]
    #[pyo3(signature = (py_model, host, port, model_name, tokenizer_path, max_context_tokens, enable_thinking=true, thinking_end_token_id=0, gpu_store_addr=0, aux_gpu_store_addrs=Vec::new(), multi_gpu_split_layers=Vec::new(), multi_gpu_gqa_offsets=Vec::new(), supports_vision=false, test_endpoints=false, prefix_cache=true, prefix_cache_ram_fraction=0.25))]
    fn new(
        py_model: PyObject,
        host: String,
        port: u16,
        model_name: String,
        tokenizer_path: String,
        max_context_tokens: usize,
        enable_thinking: bool,
        thinking_end_token_id: usize,
        gpu_store_addr: usize,
        aux_gpu_store_addrs: Vec<usize>,
        multi_gpu_split_layers: Vec<usize>,
        multi_gpu_gqa_offsets: Vec<usize>,
        supports_vision: bool,
        test_endpoints: bool,
        prefix_cache: bool,
        prefix_cache_ram_fraction: f64,
    ) -> Self {
        // Take the pre-allocated Rust prefill engine from the decode store.
        // The engine was pre-allocated from Python (before HCS pool loading)
        // so it already has its VRAM allocated. Creating a new one here would
        // fail because HCS has consumed most remaining VRAM.
        let prefill_engine = if gpu_store_addr != 0 {
            let store = unsafe { &mut *(gpu_store_addr as *mut GpuDecodeStore) };
            match store.take_prefill_engine() {
                Some(engine) => {
                    log::info!("RustServer: took pre-allocated prefill engine for benchmarks");
                    Some(engine)
                }
                None => {
                    log::warn!("RustServer: no pre-allocated prefill engine, creating on demand");
                    match create_prefill_engine_for_server(store, max_context_tokens) {
                        Ok(engine) => {
                            log::info!(
                                "RustServer: prefill engine created on demand (max_tokens={})",
                                max_context_tokens
                            );
                            Some(engine)
                        }
                        Err(e) => {
                            log::error!("RustServer: prefill engine failed: {}", e);
                            None
                        }
                    }
                }
            }
        } else {
            None
        };

        Self {
            host,
            port,
            model_name,
            tokenizer_path,
            max_context_tokens,
            default_enable_thinking: enable_thinking,
            thinking_end_token_id,
            gpu_store_addr,
            py_model: py_model.into(),
            running: Arc::new(AtomicBool::new(false)),
            aux_gpu_store_addrs,
            multi_gpu_split_layers,
            multi_gpu_gqa_offsets,
            supports_vision,
            prefill_engine: Arc::new(std::sync::Mutex::new(prefill_engine)),
            test_endpoints,
            prefix_cache,
            prefix_cache_ram_fraction,
        }
    }

    /// Start the HTTP server. Blocks until stop() is called.
    /// Releases the GIL so Python remains responsive for prefill calls.
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        self.running.store(true, Ordering::Release);

        let addr = format!("{}:{}", self.host, self.port);
        let py_model = self.py_model.clone_ref(py);
        let model_name = self.model_name.clone();
        let tokenizer_path = self.tokenizer_path.clone();
        let max_context_tokens = self.max_context_tokens;
        let default_enable_thinking = self.default_enable_thinking;
        let thinking_end_token_id = self.thinking_end_token_id;
        let gpu_store_addr = self.gpu_store_addr;
        let aux_gpu_store_addrs = self.aux_gpu_store_addrs.clone();
        let multi_gpu_split_layers = self.multi_gpu_split_layers.clone();
        let multi_gpu_gqa_offsets = self.multi_gpu_gqa_offsets.clone();
        let test_endpoints = self.test_endpoints;
        let prefix_cache_enabled = self.prefix_cache;
        let prefix_cache_ram_fraction = self.prefix_cache_ram_fraction;
        let running = self.running.clone();

        // Install raw SIGINT + SIGTERM handlers BEFORE releasing the GIL.
        // Python's signal.signal handlers only dispatch between bytecodes,
        // but run() enters allow_threads (native Rust) so Python never gets
        // a chance to run the handler.  The raw handler sets `running` to
        // false directly, and the accept loop exits on the next 10ms poll.
        // SIGTERM is needed because the release test (and systemd) send
        // SIGTERM for clean shutdown; without a raw handler, the server
        // never stops and gets SIGKILL'd, skipping VRAM report CSV write.
        #[cfg(unix)]
        let running_ptr = Arc::as_ptr(&self.running) as *mut AtomicBool;
        #[cfg(unix)]
        SIGNAL_FLAG_PTR.store(running_ptr, Ordering::Release);

        // Save previous handlers so we can restore them
        #[cfg(unix)]
        let prev_sigint;
        #[cfg(unix)]
        let prev_sigterm;
        #[cfg(unix)]
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = shutdown_signal_handler as *const () as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = libc::SA_RESTART;

            let mut old_int: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGINT, &sa, &mut old_int);
            prev_sigint = old_int;

            let mut old_term: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGTERM, &sa, &mut old_term);
            prev_sigterm = old_term;
        }

        // Release GIL — server loop runs without it.
        // GIL is reacquired inside model-worker request handlers only for
        // Python cleanup calls.
        py.allow_threads(move || {
            // Load tokenizer once at startup (not per-request)
            let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Failed to load tokenizer: {}", e);
                    return;
                }
            };

            // Load EOS token IDs from generation_config.json and config.json.
            // Step-family models may ship no generation_config.json and place
            // the EOS list only under config.json text_config.
            let eos_stop_ids = {
                let ids = collect_eos_stop_ids(&tokenizer_path);
                if ids.is_empty() {
                    log::warn!(
                        "No eos_token_id found in generation_config.json/config.json — decode may not stop"
                    );
                } else {
                    log::info!("EOS stop tokens: {:?}", ids);
                }
                ids
            };

            // Load chat template from tokenizer_config.json (same directory as tokenizer.json)
            let tokenizer_config_path = {
                let p = std::path::Path::new(&tokenizer_path);
                p.parent().unwrap_or(p).join("tokenizer_config.json")
            };
            let chat_template = match crate::chat_template::ChatTemplateEngine::from_config(
                tokenizer_config_path.to_str().unwrap_or(""),
            ) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Failed to load chat template: {}", e);
                    return;
                }
            };

            let listener = match TcpListener::bind(&addr) {
                Ok(l) => l,
                Err(e) => {
                    log::error!("Failed to bind {}: {}", addr, e);
                    return;
                }
            };

            // Set non-blocking so we can check the running flag
            listener
                .set_nonblocking(true)
                .expect("Cannot set non-blocking");

            log::info!("Rust HTTP server listening on {}", addr);

            let gil_timing = std::env::var("KRASIS_GIL_TIMING")
                .map(|v| v == "1")
                .unwrap_or(false);
            if gil_timing {
                log::info!("GIL timing enabled (KRASIS_GIL_TIMING=1)");
            }

            let log_requests_dir = if std::env::var("KRASIS_LOG_REQUESTS")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                let dir = "logs/requests".to_string();
                std::fs::create_dir_all(&dir).ok();
                log::info!("Request logging enabled → {}/", dir);
                Some(dir)
            } else {
                None
            };

            // </think> token ID passed from Python (0 = not available)
            let thinking_end_token = if thinking_end_token_id > 0 {
                log::info!("Thinking end token: </think> = {}", thinking_end_token_id);
                Some(thinking_end_token_id)
            } else {
                None
            };

            let prefix_cache_ram_fraction =
                match validate_prefix_cache_ram_fraction(prefix_cache_ram_fraction) {
                    Ok(fraction) => fraction,
                    Err(error) => {
                        log::error!("Cannot start server: {}", error);
                        return;
                    }
                };
            log::info!(
                "RAM-backed prefix cache: {}",
                if prefix_cache_enabled {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            let prefix_cache_multi_gpu_pending =
                session_cache_multi_gpu_pending(prefix_cache_enabled, &aux_gpu_store_addrs);
            if prefix_cache_multi_gpu_pending {
                log::warn!(
                    "Conversation caching is enabled but unavailable on this multi-GPU pipeline configuration; every request will miss (stats counter: misses.multi_gpu_pending)."
                );
            }
            let materialize_prefix_cache_runtime = session_cache_runtime_materialization_enabled(
                prefix_cache_enabled,
                &aux_gpu_store_addrs,
            );
            let prefix_cache_ram_store = if materialize_prefix_cache_runtime {
                match crate::session_cache::RamSessionStore::new(
                    prefix_cache_ram_fraction,
                    Arc::new(crate::session_cache::SystemMemoryAvailabilityProbe),
                ) {
                    Ok(store) => Some(store),
                    Err(error) => {
                        log::error!("Cannot start RAM-backed prefix cache: {}", error);
                        return;
                    }
                }
            } else {
                None
            };

            // Share the prefill engine from the RustServer via Arc clone.
            // If RustServer::new() took the pre-allocated engine (it should have),
            // it's already in the shared Mutex. If not, try the decode store.
            let rust_prefill = {
                let has_engine = self.prefill_engine.lock().unwrap().is_some();
                if has_engine {
                    log::info!("Rust prefill engine shared via Arc (was pre-allocated)");
                    self.prefill_engine.clone()
                } else {
                    // Not in the shared Mutex — try the decode store
                    let store = unsafe { &mut *(gpu_store_addr as *mut GpuDecodeStore) };
                    match store.take_prefill_engine() {
                        Some(engine) => {
                            log::info!(
                                "Rust prefill engine taken from decode store pre-allocated slot"
                            );
                            let arc = Arc::new(std::sync::Mutex::new(Some(engine)));
                            arc
                        }
                        None => {
                            log::warn!("No pre-allocated prefill engine — creating on demand");
                            match create_prefill_engine_for_server(store, max_context_tokens) {
                                Ok(engine) => {
                                    log::info!(
                                        "Rust prefill engine created on demand (max_tokens={})",
                                        max_context_tokens
                                    );
                                    Arc::new(std::sync::Mutex::new(Some(engine)))
                                }
                                Err(e) => {
                                    log::error!("Rust prefill engine failed: {}", e);
                                    log::error!("Cannot start server without Rust prefill engine");
                                    return;
                                }
                            }
                        }
                    }
                }
            };

            let prefix_cache_compatibility = if materialize_prefix_cache_runtime {
                match build_session_compatibility_signature(
                    &model_name,
                    &tokenizer_path,
                    &chat_template,
                    gpu_store_addr,
                    &aux_gpu_store_addrs,
                ) {
                    Ok(signature) => Some(signature),
                    Err(error) => {
                        log::error!(
                            "Cannot start server with RAM-backed prefix cache enabled: {}",
                            error
                        );
                        return;
                    }
                }
            } else {
                None
            };

            let serving_metrics = Arc::new(ServingMetrics::new(max_context_tokens));
            let state = ServerState {
                py_model,
                model_name,
                tokenizer,
                chat_template,
                max_context_tokens,
                default_enable_thinking,
                thinking_end_token,
                gpu_store_addr,
                log_requests_dir,
                aux_gpu_store_addrs,
                multi_gpu_split_layers,
                multi_gpu_gqa_offsets,
                rust_prefill,
                eos_stop_ids,
                reference_test_request_order: 0,
                session_cache: SessionCacheRuntime {
                    enabled: prefix_cache_enabled,
                    ram_fraction: prefix_cache_ram_fraction,
                    active: None,
                    compatibility: prefix_cache_compatibility,
                    ram_store: prefix_cache_ram_store,
                    prefill_samples: Vec::new(),
                    restore_samples: Vec::new(),
                    session_locks: Arc::new(SessionLockTable::default()),
                    metrics: SessionCacheMetrics::default(),
                },
                serving_metrics: Arc::clone(&serving_metrics),
            };

            publish_session_cache_metrics(&state);

            let server_info = ServerInfo {
                model_name: state.model_name.clone(),
                max_context_tokens: state.max_context_tokens,
                supports_vision: self.supports_vision,
                serving_metrics,
            };
            let (model_tx, model_rx) = mpsc::channel::<QueuedModelRequest>();
            let scheduler = Arc::new(FairModelScheduler::new(model_tx));
            let worker_scheduler = Arc::clone(&scheduler);
            let worker_running = running.clone();
            let worker_handle = std::thread::Builder::new()
                .name("krasis-model-worker".to_string())
                .spawn(move || {
                    let mut state = state;
                    let mut expected_ticket = 0u64;
                    while worker_running.load(Ordering::Acquire) {
                        match model_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(queued) => {
                                let remaining = worker_scheduler.mark_dequeued();
                                let wait_ms = queued.enqueued_at.elapsed().as_secs_f64() * 1000.0;
                                if queued.ticket != expected_ticket {
                                    let error = format!(
                                        "model scheduler ticket mismatch: expected {}, received {}",
                                        expected_ticket, queued.ticket
                                    );
                                    log::error!("{}", error);
                                    reject_model_request(queued.request, &error);
                                } else {
                                    log::info!(
                                        "Model scheduler dispatch: ticket={} wait_ms={:.3} remaining={}",
                                        queued.ticket,
                                        wait_ms,
                                        remaining,
                                    );
                                    handle_model_request(queued.request, &mut state);
                                }
                                expected_ticket = expected_ticket.checked_add(1).unwrap_or_else(|| {
                                    log::error!("model scheduler worker ticket counter exhausted");
                                    u64::MAX
                                });
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                drain_vram_pressure_for_state(&mut state, "idle", false);
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }

                    while let Ok(queued) = model_rx.try_recv() {
                        let remaining = worker_scheduler.mark_dequeued();
                        if queued.ticket != expected_ticket {
                            let error = format!(
                                "model scheduler shutdown ticket mismatch: expected {}, received {}",
                                expected_ticket, queued.ticket
                            );
                            log::error!("{}", error);
                            reject_model_request(queued.request, &error);
                        } else {
                            log::info!(
                                "Model scheduler shutdown dispatch: ticket={} wait_ms={:.3} remaining={}",
                                queued.ticket,
                                queued.enqueued_at.elapsed().as_secs_f64() * 1000.0,
                                remaining,
                            );
                            handle_model_request(queued.request, &mut state);
                        }
                        expected_ticket = expected_ticket.checked_add(1).unwrap_or(u64::MAX);
                    }

                    log::info!("Rust HTTP model worker stopped");
                });
            let worker_handle = match worker_handle {
                Ok(handle) => handle,
                Err(e) => {
                    log::error!("Failed to start model worker: {}", e);
                    return;
                }
            };

            while running.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _addr)) => {
                        // Set blocking for the actual request handling
                        stream.set_nonblocking(false).ok();
                        // Disable Nagle's algorithm for immediate SSE chunk delivery
                        stream.set_nodelay(true).ok();
                        // Set read timeout to prevent hanging on malformed requests
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                            .ok();
                        let info = server_info.clone();
                        let request_scheduler = Arc::clone(&scheduler);
                        let endpoints_enabled = test_endpoints;
                        if let Err(e) = std::thread::Builder::new()
                            .name("krasis-http-connection".to_string())
                            .spawn(move || {
                                handle_front_connection(
                                    stream,
                                    info,
                                    request_scheduler,
                                    endpoints_enabled,
                                );
                            })
                        {
                            log::error!("Failed to spawn connection handler: {}", e);
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No connection ready, sleep briefly and retry
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(e) => {
                        log::error!("Accept error: {}", e);
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            }

            drop(scheduler);
            let _ = worker_handle.join();
            log::info!("Rust HTTP server stopped");
        });

        // Restore previous signal handlers and clear global pointer
        #[cfg(unix)]
        SIGNAL_FLAG_PTR.store(std::ptr::null_mut(), Ordering::Release);
        #[cfg(unix)]
        unsafe {
            libc::sigaction(libc::SIGINT, &prev_sigint, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &prev_sigterm, std::ptr::null_mut());
        }

        Ok(())
    }

    /// Run a single benchmark request through the engine (no HTTP/SSE).
    /// Same operations as handle_chat_completion but without network I/O.
    /// Returns JSON string with engine-internal timing breakdown.
    ///
    /// Safety: assumes no concurrent HTTP requests during benchmark.
    #[pyo3(signature = (messages_json, max_new_tokens, temperature=0.6, enable_thinking=false))]
    fn benchmark_request(
        &self,
        py: Python<'_>,
        messages_json: String,
        max_new_tokens: usize,
        temperature: f32,
        enable_thinking: bool,
    ) -> PyResult<String> {
        let benchmark_prefill_breakdown =
            std::env::var("KRASIS_BENCHMARK_PREFILL_BREAKDOWN").is_ok();

        // Load tokenizer and chat template (same as server path)
        let tokenizer = tokenizers::Tokenizer::from_file(&self.tokenizer_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e))
        })?;
        let tokenizer_config_path = {
            let p = std::path::Path::new(&self.tokenizer_path);
            p.parent().unwrap_or(p).join("tokenizer_config.json")
        };
        let chat_template = crate::chat_template::ChatTemplateEngine::from_config(
            tokenizer_config_path.to_str().unwrap_or(""),
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to load chat template: {}",
                e
            ))
        })?;

        let messages_value: serde_json::Value =
            serde_json::from_str(&messages_json).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Invalid messages JSON: {}", e))
            })?;
        crate::text_only_messages::validate_text_only_messages(&messages_value)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Estimate tokens by applying the same chat template mode the request will use.
        let estimated_tokens = {
            let rendered = chat_template
                .apply(&messages_json, true, enable_thinking)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Chat template failed: {}",
                        e
                    ))
                })?;
            tokenizer
                .encode(rendered.as_str(), false)
                .map(|e| e.len())
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenizer failed: {}", e))
                })?
        };

        // Evict soft HCS before prefill (both stores in multi-GPU)
        let prefill_entry_floor_bytes =
            prefill_entry_floor_bytes_for_server(&self.prefill_engine, estimated_tokens).map_err(
                |e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Prefill engine floor unavailable before HCS eviction: {}",
                        e
                    ))
                },
            )?;
        let store = unsafe { &mut *(self.gpu_store_addr as *mut GpuDecodeStore) };
        let t_evict = Instant::now();
        let (evicted, _) = store
            .hcs_evict_for_prefill_with_engine_floor(estimated_tokens, prefill_entry_floor_bytes);
        // NOTE: aux GPU never does prefill, so no eviction needed there
        let evict_ms = t_evict.elapsed().as_secs_f64() * 1000.0;

        // Prefill (Rust, zero GIL)
        crate::vram_monitor::report_event("prefill_start");
        let t_prefill = Instant::now();
        let mut prefill_lock_ms = 0.0f64;
        let mut prefill_hcs_snapshot_ms = 0.0f64;
        let mut prefill_tokenize_ms = 0.0f64;
        let mut prefill_prepare_runtime_ms = 0.0f64;
        let mut prefill_scratch_ms = 0.0f64;
        let mut prefill_run_ms = 0.0f64;
        let mut prefill_shadow_ms = 0.0f64;
        let mut prefill_release_ms = 0.0f64;
        let mut prefill_stop_ids_ms = 0.0f64;
        let mut prefill_restore_ms = 0.0f64;

        let t_phase = Instant::now();
        let mut engine_guard = self.prefill_engine.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Prefill engine lock poisoned: {}",
                e
            ))
        })?;
        let engine = engine_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Rust prefill engine not available for benchmark",
            )
        })?;
        prefill_lock_ms = t_phase.elapsed().as_secs_f64() * 1000.0;
        // Warmup/calibration calls disable prefill pinning through the shared engine.
        // Benchmarks should exercise the same normal prefill path as requests.
        engine.set_prefill_pinning_disabled(false);

        // Update HCS snapshot
        let t_phase = Instant::now();
        {
            let store_ref = unsafe { &*(self.gpu_store_addr as *const GpuDecodeStore) };
            let (cache_fast, ne) = store_ref.export_hcs_snapshot();
            engine.update_hcs_snapshot(cache_fast, ne);
        }
        prefill_hcs_snapshot_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        // Tokenize using Rust tokenizer (always with generation prompt)
        let t_phase = Instant::now();
        let token_ids: Vec<u32> = {
            let rendered = chat_template
                .apply(&messages_json, true, enable_thinking)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Chat template failed: {}",
                        e
                    ))
                })?;
            let encoding = tokenizer.encode(rendered.as_str(), false).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenizer failed: {}", e))
            })?;
            encoding.get_ids().to_vec()
        };
        prefill_tokenize_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        let kv_overflow = token_ids.len() > engine.kv_max_seq;

        let t_phase = Instant::now();
        let _has_hqq_runtime_slots = prepare_store_for_rust_prefill(store, engine, token_ids.len())
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to prepare runtime for prefill: {}",
                    e
                ))
            })?;
        prefill_prepare_runtime_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        engine.set_prefill_hcs_guard_store_addr(self.gpu_store_addr);

        // Dynamically allocate scratch for this prompt
        let t_phase = Instant::now();
        if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
            engine.clear_prefill_hcs_guard_store_addr();
            engine.set_optional_pinning_budget_mb(None);
            let _ = store.prepare_runtime_for_decode_rust();
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Scratch alloc failed: {}",
                e
            )));
        }
        prefill_scratch_ms = t_phase.elapsed().as_secs_f64() * 1000.0;
        let pinning_budget_mb = store.prefill_optional_pinning_budget_mb(
            token_ids.len(),
            engine.last_prepare_post_alloc_free_mb(),
        );
        engine.set_optional_pinning_budget_mb(pinning_budget_mb);

        let suppress_tokens = store.suppress_tokens_clone();
        let t_phase = Instant::now();
        let prefill_result = match engine.run_prefill(&token_ids, temperature, &suppress_tokens) {
            Ok(r) => match engine.finalize_stage_exact_prefill_kv(r.prompt_len) {
                Ok(()) => Ok(r),
                Err(e) => Err(format!("KV stage export failed: {}", e)),
            },
            Err(e) => Err(e),
        }
        .map_err(|e| {
            abort_if_cuda_context_poisoned("benchmark prefill", &e);
            engine.clear_prefill_hcs_guard_store_addr();
            engine.set_optional_pinning_budget_mb(None);
            let _ = engine.release_scratch();
            let _ = store.prepare_runtime_for_decode_rust();
            pyo3::exceptions::PyRuntimeError::new_err(format!("Rust prefill failed: {}", e))
        })?;
        prefill_run_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        let t_phase = Instant::now();
        let prompt_hcs_snapshot = engine.prompt_hcs_shadow_snapshot();
        prefill_shadow_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        // Release scratch to free VRAM for decode/HCS
        let t_phase = Instant::now();
        if let Err(e) = engine.release_scratch() {
            log::error!("Failed to release scratch: {}", e);
            abort_if_cuda_context_poisoned("benchmark release_scratch", &e);
        }
        engine.clear_prefill_hcs_guard_store_addr();
        engine.set_optional_pinning_budget_mb(None);
        prefill_release_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        let first_token = prefill_result.first_token as usize;
        let prompt_len = prefill_result.prompt_len;
        // Load EOS tokens for benchmark path (same logic as serve_forever)
        let t_phase = Instant::now();
        let stop_ids: Vec<usize> = collect_eos_stop_ids(&self.tokenizer_path);
        prefill_stop_ids_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        let t_phase = Instant::now();
        if let Err(e) = restore_store_after_rust_prefill(store, prompt_len) {
            log::error!("Failed to restore decode runtime after prefill: {}", e);
        }
        store.set_rope_position_delta(0);
        prefill_restore_ms = t_phase.elapsed().as_secs_f64() * 1000.0;

        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        crate::vram_monitor::report_event("prefill_end");
        let prefill_accounted_ms = prefill_lock_ms
            + prefill_hcs_snapshot_ms
            + prefill_tokenize_ms
            + prefill_prepare_runtime_ms
            + prefill_scratch_ms
            + prefill_run_ms
            + prefill_shadow_ms
            + prefill_release_ms
            + prefill_stop_ids_ms
            + prefill_restore_ms;
        let prefill_unaccounted_ms = (prefill_ms - prefill_accounted_ms).max(0.0);
        let prefill_breakdown = serde_json::json!({
            "total_ms": prefill_ms,
            "lock_ms": prefill_lock_ms,
            "hcs_snapshot_ms": prefill_hcs_snapshot_ms,
            "tokenize_ms": prefill_tokenize_ms,
            "prepare_runtime_ms": prefill_prepare_runtime_ms,
            "scratch_ms": prefill_scratch_ms,
            "run_finalize_ms": prefill_run_ms,
            "prompt_hcs_shadow_ms": prefill_shadow_ms,
            "release_scratch_ms": prefill_release_ms,
            "stop_ids_ms": prefill_stop_ids_ms,
            "restore_runtime_ms": prefill_restore_ms,
            "unaccounted_ms": prefill_unaccounted_ms,
        });
        if benchmark_prefill_breakdown {
            log::info!(
                "BENCH_PREFILL_BREAKDOWN tokens={} total_ms={:.1} lock_ms={:.1} hcs_snapshot_ms={:.1} tokenize_ms={:.1} prepare_runtime_ms={:.1} scratch_ms={:.1} run_finalize_ms={:.1} prompt_hcs_shadow_ms={:.1} release_scratch_ms={:.1} stop_ids_ms={:.1} restore_runtime_ms={:.1} unaccounted_ms={:.1}",
                prompt_len,
                prefill_ms,
                prefill_lock_ms,
                prefill_hcs_snapshot_ms,
                prefill_tokenize_ms,
                prefill_prepare_runtime_ms,
                prefill_scratch_ms,
                prefill_run_ms,
                prefill_shadow_ms,
                prefill_release_ms,
                prefill_stop_ids_ms,
                prefill_restore_ms,
                prefill_unaccounted_ms,
            );
        }

        if kv_overflow || max_new_tokens <= 1 {
            crate::vram_monitor::report_event("hcs_soft_load_start");
            let t_reload = Instant::now();
            if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
                log::info!(
                    "Benchmark prefill-only: prompt-HCS snapshot ready: prompt_tokens={} layers={} experts={}",
                    prompt_tokens,
                    layers,
                    experts,
                );
                store.install_prompt_hcs_counts(counts.clone(), *layers, *experts, *prompt_tokens);
            } else {
                log::warn!("Benchmark prefill-only: prompt-HCS snapshot missing before reload");
                store.clear_prompt_hcs_counts();
            }
            let (activated, real_reload_dma_ms) = store.hcs_reload_after_prefill(prompt_len);
            if activated > 0 {
                log::info!(
                    "Benchmark prefill-only: HCS reload complete: {} experts, {:.1}ms ({} tokens)",
                    activated,
                    real_reload_dma_ms,
                    prompt_len,
                );
            }
            if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
                store.install_prompt_hcs_shadow(counts.clone(), *layers, *experts, *prompt_tokens);
            } else {
                store.clear_prompt_hcs_shadow();
            }
            let (pressure_evicted, pressure_freed_mb, pressure_final_free_mb) =
                store.hcs_drain_vram_pressure("benchmark_prefill_only_after_reload", true);
            if pressure_evicted > 0 {
                log::warn!(
                    "Benchmark prefill-only: VRAM pressure eviction after reload evicted {} soft experts, freed {:.1} MB, final_free={} MB",
                    pressure_evicted,
                    pressure_freed_mb,
                    pressure_final_free_mb,
                );
            }
            let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;
            crate::vram_monitor::report_event("hcs_soft_load_end");

            self.py_model.call_method0(py, "server_cleanup")?;

            let prefill_tok_s = if prefill_ms > 0.0 {
                prompt_len as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            };
            let (min_free_vram_mb, mut hcs_loaded, mut hcs_total, _) = store.benchmark_stats();
            let safety_margin_mb = store.hcs_safety_margin_mb();
            if !self.aux_gpu_store_addrs.is_empty() {
                log::info!(
                    "  GPU0: min_free={} MB, HCS {} loaded",
                    min_free_vram_mb,
                    hcs_loaded
                );
            }
            for (i, &aux_addr) in self.aux_gpu_store_addrs.iter().enumerate() {
                let aux_store = unsafe { &*(aux_addr as *const GpuDecodeStore) };
                let (aux_min_free, aux_loaded, aux_total, aux_pct) = aux_store.benchmark_stats();
                hcs_loaded += aux_loaded;
                hcs_total += aux_total;
                if !self.aux_gpu_store_addrs.is_empty() {
                    log::info!(
                        "  GPU{}: min_free={} MB, HCS {}/{} ({:.1}%)",
                        i + 1,
                        aux_min_free,
                        aux_loaded,
                        aux_total,
                        aux_pct
                    );
                }
            }
            let hcs_pct = if hcs_total > 0 {
                hcs_loaded as f64 / hcs_total as f64 * 100.0
            } else {
                0.0
            };

            let mut result = serde_json::json!({
                "prefill_ms": prefill_ms,
                "prefill_tok_s": prefill_tok_s,
                "prompt_tokens": prompt_len,
                "decode_ms": 0.0,
                "decode_tok_s": 0.0,
                "decode_tokens": 1,
                "evict_ms": evict_ms,
                "reload_ms": reload_ms,
                "real_reload_dma_ms": real_reload_dma_ms,
                "min_free_vram_mb": min_free_vram_mb,
                "hcs_loaded": hcs_loaded,
                "hcs_total": hcs_total,
                "hcs_pct": hcs_pct,
                "safety_margin_mb": safety_margin_mb,
            });
            if benchmark_prefill_breakdown {
                result["prefill_breakdown"] = prefill_breakdown.clone();
            }

            return Ok(result.to_string());
        }

        // Reload soft HCS after prefill
        crate::vram_monitor::report_event("hcs_soft_load_start");
        let t_reload = Instant::now();
        if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
            log::info!(
                "Benchmark: prompt-HCS snapshot ready: prompt_tokens={} layers={} experts={}",
                prompt_tokens,
                layers,
                experts,
            );
            store.install_prompt_hcs_counts(counts.clone(), *layers, *experts, *prompt_tokens);
        } else {
            log::warn!("Benchmark: prompt-HCS snapshot missing before reload");
            store.clear_prompt_hcs_counts();
        }
        let (activated, real_reload_dma_ms) = store.hcs_reload_after_prefill(prompt_len);
        if activated > 0 {
            log::info!(
                "Benchmark: HCS reload complete: {} experts, {:.1}ms",
                activated,
                real_reload_dma_ms
            );
        }
        if let Some((counts, layers, experts, prompt_tokens)) = prompt_hcs_snapshot.as_ref() {
            store.install_prompt_hcs_shadow(counts.clone(), *layers, *experts, *prompt_tokens);
        } else {
            store.clear_prompt_hcs_shadow();
        }
        let reload_pending_at_decode_start = store.hcs_soft_reload_pending();
        // NOTE: aux GPUs have no soft tier (100% hard), no eviction/reload needed
        let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;
        crate::vram_monitor::report_event("hcs_soft_load_end");

        // Match the live request path's per-request decode suppression setup.
        let benchmark_min_stop_suppress_steps = max_new_tokens.saturating_sub(1);
        if enable_thinking {
            if self.thinking_end_token_id > 0 {
                store.set_think_end_suppress(Some(self.thinking_end_token_id), 4096);
                store.set_min_new_tokens_ext(benchmark_min_stop_suppress_steps, stop_ids.clone());
            } else {
                store.set_think_end_suppress(None, 0);
                store.set_min_new_tokens_ext(benchmark_min_stop_suppress_steps, stop_ids.clone());
            }
        } else {
            store.set_think_end_suppress(None, 0);
            store.set_min_new_tokens_ext(benchmark_min_stop_suppress_steps, stop_ids.clone());
        }

        // Copy KV cache to aux stores (multi-GPU) — after async reload starts
        if !self.aux_gpu_store_addrs.is_empty() {
            let num_aux = self.aux_gpu_store_addrs.len();
            let num_layers = store.num_layers();
            for i in 0..num_aux {
                let aux_store =
                    unsafe { &mut *(self.aux_gpu_store_addrs[i] as *mut GpuDecodeStore) };
                let layer_start = self.multi_gpu_split_layers[i];
                let layer_end = if i + 1 < num_aux {
                    self.multi_gpu_split_layers[i + 1]
                } else {
                    num_layers
                };
                if let Err(e) = store.copy_kv_to_aux(
                    aux_store,
                    layer_start,
                    layer_end,
                    self.multi_gpu_gqa_offsets[i],
                    prompt_len,
                ) {
                    log::error!(
                        "benchmark_request: KV copy to aux GPU{} failed: {}",
                        i + 1,
                        e
                    );
                }
                if let Err(e) = store.copy_la_states_to_aux(aux_store, layer_start, layer_end) {
                    log::error!(
                        "benchmark_request: LA state copy to aux GPU{} failed: {}",
                        i + 1,
                        e
                    );
                }
                store
                    .copy_dsa_prompt_keys_to_aux(aux_store, prompt_len)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "benchmark_request: DSA prompt key-cache copy to aux GPU{} failed: {}",
                            i + 1,
                            e
                        ))
                    })?;
            }
        }
        let (pressure_evicted, pressure_freed_mb, pressure_final_free_mb) =
            store.hcs_drain_vram_pressure("benchmark_before_decode", true);
        if pressure_evicted > 0 {
            log::warn!(
                "Benchmark: VRAM pressure eviction before decode evicted {} soft experts, freed {:.1} MB, final_free={} MB",
                pressure_evicted,
                pressure_freed_mb,
                pressure_final_free_mb,
            );
            let (pressure_reload_activated, pressure_reload_ms) =
                store.hcs_reload_after_prefill(prompt_len);
            if pressure_reload_activated > 0 {
                log::info!(
                    "Benchmark: HCS reload after pressure drain: {} experts, {:.1}ms",
                    pressure_reload_activated,
                    pressure_reload_ms,
                );
                let (post_reload_evicted, post_reload_freed_mb, post_reload_final_free_mb) = store
                    .hcs_drain_vram_pressure("benchmark_before_decode_after_pressure_reload", true);
                if post_reload_evicted > 0 {
                    log::warn!(
                        "Benchmark: post-reload pressure eviction before decode evicted {} soft experts, freed {:.1} MB, final_free={} MB",
                        post_reload_evicted,
                        post_reload_freed_mb,
                        post_reload_final_free_mb,
                    );
                }
            }
        }

        // Decode (pure Rust, GIL held but unused by decode loop)
        crate::vram_monitor::report_event("decode_start");
        let decode_start = Instant::now();
        let mut count = 0usize;
        if !self.aux_gpu_store_addrs.is_empty() {
            store.gpu_generate_stream_multi(
                &self.aux_gpu_store_addrs,
                &self.multi_gpu_split_layers,
                &self.multi_gpu_gqa_offsets,
                first_token,
                prompt_len,
                max_new_tokens.saturating_sub(1),
                temperature,
                50,   // top_k
                0.95, // top_p
                &stop_ids,
                &tokenizer,
                &[],
                0.0, // presence_penalty
                0,   // logprobs_top_n
                Some("benchmark".to_string()),
                |_token_id: usize,
                 _text: &str,
                 _finish_reason: Option<&str>,
                 _logprobs: Option<&[(u32, f32)]>| {
                    count += 1;
                    true
                },
            );
        } else {
            store.gpu_generate_stream(
                first_token,
                prompt_len,
                max_new_tokens.saturating_sub(1),
                temperature,
                50,   // top_k
                0.95, // top_p
                &stop_ids,
                &tokenizer,
                &[],
                0.0, // presence_penalty
                0,   // logprobs_top_n
                Some("benchmark".to_string()),
                |_token_id, _text, _finish_reason, _logprobs: Option<&[(u32, f32)]>| {
                    count += 1;
                    true
                },
            );
        }
        let elapsed = decode_start.elapsed().as_secs_f64();
        let decode_tokens = count + 1; // includes first_token from prefill
        let decode_tok_s = if elapsed > 0.0 && count > 0 {
            count as f64 / elapsed
        } else {
            0.0
        };
        let decode_ms = elapsed * 1000.0;

        crate::vram_monitor::report_event("decode_end");

        // Cleanup
        self.py_model.call_method0(py, "server_cleanup")?;

        let prefill_tok_s = if prefill_ms > 0.0 {
            prompt_len as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };

        // Collect HCS stats from primary store
        let (min_free_vram_mb, mut hcs_loaded, mut hcs_total, _) = store.benchmark_stats();
        let safety_margin_mb = store.hcs_safety_margin_mb();

        // Aggregate HCS stats from all aux stores (multi-GPU)
        // Also log per-GPU VRAM stats
        if !self.aux_gpu_store_addrs.is_empty() {
            log::info!(
                "  GPU0: min_free={} MB, HCS {} loaded",
                min_free_vram_mb,
                hcs_loaded
            );
        }
        for (i, &aux_addr) in self.aux_gpu_store_addrs.iter().enumerate() {
            let aux_store = unsafe { &*(aux_addr as *const GpuDecodeStore) };
            let (aux_min_free, aux_loaded, aux_total, aux_pct) = aux_store.benchmark_stats();
            hcs_loaded += aux_loaded;
            hcs_total += aux_total;
            if !self.aux_gpu_store_addrs.is_empty() {
                log::info!(
                    "  GPU{}: min_free={} MB, HCS {}/{} ({:.1}%)",
                    i + 1,
                    aux_min_free,
                    aux_loaded,
                    aux_total,
                    aux_pct
                );
            }
        }
        let hcs_pct = if hcs_total > 0 {
            hcs_loaded as f64 / hcs_total as f64 * 100.0
        } else {
            0.0
        };
        let state_validation_env = std::env::var("KRASIS_STATE_VALIDATION").ok();
        let config_validation_env = std::env::var("KRASIS_CONFIG_VALIDATION").ok();
        let state_validation_enabled = state_validation_env
            .as_deref()
            .map(|v| v != "0")
            .unwrap_or(false)
            || config_validation_env
                .as_deref()
                .map(|v| v != "0")
                .unwrap_or(false);
        let state_validation = if state_validation_enabled {
            let raw = store.config_validation_snapshot_json(
                prompt_len,
                true, // sync is always on
                reload_pending_at_decode_start,
            );
            match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(v) => {
                    log::info!("STATE_VALIDATION {}", v);
                    Some(v)
                }
                Err(e) => {
                    log::warn!("STATE_VALIDATION parse failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let mut result = serde_json::json!({
            "prefill_ms": prefill_ms,
            "prefill_tok_s": prefill_tok_s,
            "prompt_tokens": prompt_len,
            "decode_ms": decode_ms,
            "decode_tok_s": decode_tok_s,
            "decode_tokens": decode_tokens,
            "evict_ms": evict_ms,
            "reload_ms": reload_ms,
            "real_reload_dma_ms": real_reload_dma_ms,
            "min_free_vram_mb": min_free_vram_mb,
            "hcs_loaded": hcs_loaded,
            "hcs_total": hcs_total,
            "hcs_pct": hcs_pct,
            "safety_margin_mb": safety_margin_mb,
        });
        if benchmark_prefill_breakdown {
            result["prefill_breakdown"] = prefill_breakdown;
        }
        if let Some(v) = state_validation {
            result["state_validation"] = v;
        }

        Ok(result.to_string())
    }

    /// Signal the server to stop.
    fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// Check if server is running.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        active_boundary_tokens_for_publication, active_plan_requires_device_checkpoint,
        active_plan_requires_stage_restore, consumed_generation_boundary, context_window_fits,
        format_completion, format_completion_with_debug, format_completion_with_tool_calls,
        format_models_response, format_sse_timing, format_sse_token, format_sse_tool_call_args,
        format_sse_tool_call_start, hide_synthetic_think_stop_text, internal_capture_boundary,
        is_chat_completions_endpoint, is_models_endpoint, parse_tool_calls, push_tool_stream_text,
        session_cache_multi_gpu_pending, session_cache_runtime_materialization_enabled,
        validate_prefix_cache_ram_fraction, FairModelScheduler, ModelRequest, ParsedToolCall,
        RequestOverhead, ServingMetrics, ServingRequestGuard, SessionCacheMetrics,
        SessionCacheMissReason, SessionLockKey, SessionLockTable, StreamDetokenizer,
    };
    use crate::chat_template::{ChatTemplateEngine, ToolCallFormat};
    use std::fs;
    use std::net::{TcpListener, TcpStream};
    use std::sync::{mpsc, Arc};
    use std::time::Instant;

    const WINDOWS_MODEL_PATH: &str = r#"C:\Users\stoate\.krasis\models\Qwen3.6-35B-A3B"#;

    #[test]
    fn terminal_snapshot_never_enters_internal_boundary_path() {
        assert_eq!(internal_capture_boundary(Some(64), 0, 128), Some(64));
        assert_eq!(internal_capture_boundary(Some(128), 0, 128), None);
        assert_eq!(internal_capture_boundary(Some(64), 64, 128), None);
        assert_eq!(internal_capture_boundary(None, 0, 128), None);
    }

    #[test]
    fn kv_truncation_never_restores_a_longer_device_checkpoint() {
        use crate::session_cache::ActivePrefixPlan;

        assert!(active_plan_requires_device_checkpoint(
            ActivePrefixPlan::Append { matched_tokens: 32 },
            true,
        ));
        assert!(!active_plan_requires_device_checkpoint(
            ActivePrefixPlan::TruncateKvAndAppend { matched_tokens: 4 },
            true,
        ));
        assert!(!active_plan_requires_device_checkpoint(
            ActivePrefixPlan::Append { matched_tokens: 32 },
            false,
        ));
        assert!(active_plan_requires_stage_restore(
            ActivePrefixPlan::Append { matched_tokens: 32 },
            true,
        ));
        assert!(active_plan_requires_stage_restore(
            ActivePrefixPlan::TruncateKvAndAppend { matched_tokens: 4 },
            true,
        ));
        assert!(!active_plan_requires_stage_restore(
            ActivePrefixPlan::TruncateKvAndAppend { matched_tokens: 4 },
            false,
        ));
    }

    fn parse_response(body: &str) -> serde_json::Value {
        serde_json::from_str(body).unwrap_or_else(|e| {
            panic!("response is not valid JSON: {e}\nbody: {body}");
        })
    }

    fn parsed_arguments(call: &ParsedToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments_json).unwrap()
    }

    fn tcp_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        (server, client)
    }

    #[test]
    fn compressed_publication_reuses_an_exact_committed_boundary() {
        let stable = [10u32, 20, 30];
        let selected =
            active_boundary_tokens_for_publication(true, None, Some(&stable), stable.len(), true);
        assert_eq!(selected.as_deref(), Some(stable.as_slice()));
        let pending = [40u32, 50];
        assert_eq!(
            active_boundary_tokens_for_publication(
                true,
                Some(&pending),
                Some(&stable),
                stable.len(),
                false,
            )
            .as_deref(),
            Some(pending.as_slice()),
        );

        assert!(active_boundary_tokens_for_publication(
            true,
            None,
            Some(&stable),
            stable.len() - 1,
            true,
        )
        .is_none());
        assert!(active_boundary_tokens_for_publication(
            true,
            None,
            Some(&stable),
            stable.len(),
            false,
        )
        .is_none());
    }

    #[test]
    fn session_cache_metrics_keep_operational_timing_separate_from_miss_reasons() {
        let mut metrics = SessionCacheMetrics::default();
        metrics.record_hit(false);
        metrics.record_hit(true);
        metrics.record_miss(SessionCacheMissReason::NoMatch);
        metrics.record_miss(SessionCacheMissReason::SignatureMismatch);
        metrics.record_miss(SessionCacheMissReason::Evicted);
        metrics.record_miss(SessionCacheMissReason::RestoreNotWorthIt);
        metrics.record_miss(SessionCacheMissReason::Divergence);
        metrics.record_save(1_024, 2.5);
        metrics.record_restore(768, 1.25);

        assert_eq!(metrics.active_hits, 1);
        assert_eq!(metrics.ram_hits, 1);
        assert_eq!(metrics.no_match_misses, 1);
        assert_eq!(metrics.signature_mismatch_misses, 1);
        assert_eq!(metrics.evicted_misses, 1);
        assert_eq!(metrics.restore_not_worth_it_misses, 1);
        assert_eq!(metrics.divergence_misses, 1);
        assert_eq!(metrics.save_count, 1);
        assert_eq!(metrics.save_bytes, 1_024);
        assert_eq!(metrics.save_total_ms, 2.5);
        assert_eq!(metrics.restore_count, 1);
        assert_eq!(metrics.restore_bytes, 768);
        assert_eq!(metrics.restore_total_ms, 1.25);

        metrics.record_save(9_999, f64::NAN);
        metrics.record_restore(9_999, -1.0);
        assert_eq!(metrics.save_count, 1);
        assert_eq!(metrics.save_bytes, 1_024);
        assert_eq!(metrics.restore_count, 1);
        assert_eq!(metrics.restore_bytes, 768);

        assert_eq!(validate_prefix_cache_ram_fraction(0.25).unwrap(), 0.25);
        assert!(validate_prefix_cache_ram_fraction(0.0).is_err());
        assert!(validate_prefix_cache_ram_fraction(f64::NAN).is_err());
        assert!(validate_prefix_cache_ram_fraction(1.01).is_err());
    }

    #[test]
    fn session_cache_multi_gpu_pending_uses_auxiliary_store_topology() {
        assert!(!session_cache_multi_gpu_pending(true, &[]));
        assert!(!session_cache_multi_gpu_pending(false, &[0x1234]));
        assert!(session_cache_multi_gpu_pending(true, &[0x1234]));
        assert!(session_cache_multi_gpu_pending(true, &[0x1234, 0x5678]));

        assert!(session_cache_runtime_materialization_enabled(true, &[]));
        assert!(!session_cache_runtime_materialization_enabled(false, &[]));
        assert!(!session_cache_runtime_materialization_enabled(
            true,
            &[0x1234]
        ));
        assert!(!session_cache_runtime_materialization_enabled(
            true,
            &[0x1234, 0x5678]
        ));
    }

    #[test]
    fn fair_model_scheduler_dispatches_concurrent_admission_in_ticket_order() {
        let (sender, receiver) = mpsc::channel();
        let scheduler = Arc::new(FairModelScheduler::new(sender));
        let mut peers = Vec::new();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let (stream, peer) = tcp_pair();
            peers.push(peer);
            let scheduler = Arc::clone(&scheduler);
            handles.push(std::thread::spawn(move || {
                scheduler
                    .enqueue(ModelRequest::Chat {
                        stream,
                        body: String::new(),
                        received_at: Instant::now(),
                    })
                    .unwrap()
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let mut tickets = Vec::new();
        for expected_remaining in (0..8).rev() {
            let queued = receiver.recv().unwrap();
            tickets.push(queued.ticket);
            assert_eq!(scheduler.mark_dequeued(), expected_remaining);
        }
        assert_eq!(tickets, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn serving_metrics_render_exact_cache_and_outcome_semantics() {
        let metrics = ServingMetrics::new(1_000);
        metrics.request_started();
        metrics.set_active_kv_tokens(250);
        metrics.observe_ttft(0.5);
        metrics.token_generated(0.02, 250);
        let live = metrics.render(3);
        assert!(live.contains("vllm:num_requests_running 1\n"));
        assert!(live.contains("vllm:num_requests_waiting 3\n"));
        assert!(live.contains("vllm:kv_cache_usage_perc 0.25\n"));

        metrics.request_finished(1.25, 100, 10, 40, "tool_calls");
        let finished = metrics.render(0);
        assert!(finished.contains("vllm:num_requests_running 0\n"));
        assert!(finished.contains("vllm:prompt_tokens_total 100\n"));
        assert!(finished.contains("vllm:generation_tokens_total 10\n"));
        assert!(finished.contains("vllm:prefix_cache_queries_total 100\n"));
        assert!(finished.contains("vllm:prefix_cache_hits_total 40\n"));
        assert!(finished.contains("vllm:request_success_total{finished_reason=\"tool_calls\"} 1\n"));
        assert!(finished.contains("vllm:time_to_first_token_seconds_count 1\n"));
        assert!(finished.contains("vllm:inter_token_latency_seconds_count 1\n"));
        assert!(finished.contains("vllm:e2e_request_latency_seconds_count 1\n"));
    }

    #[test]
    fn unfinished_serving_request_is_recorded_as_error() {
        let metrics = Arc::new(ServingMetrics::new(1_000));
        {
            let _guard = ServingRequestGuard::new(Arc::clone(&metrics), Instant::now());
        }
        let rendered = metrics.render(0);
        assert!(rendered.contains("vllm:num_requests_running 0\n"));
        assert!(rendered.contains("vllm:request_success_total{finished_reason=\"error\"} 1\n"));
    }

    #[test]
    fn exact_session_lease_excludes_same_session_until_commit_boundary() {
        let table = Arc::new(SessionLockTable::default());
        let key = SessionLockKey::ExactBoundary(Arc::from([1u32, 2, 3].as_slice()));
        let first = table.acquire(key.clone()).unwrap();
        let (attempted_tx, attempted_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let waiting_table = Arc::clone(&table);
        let waiting_key = key.clone();
        let handle = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            let lease = waiting_table.acquire(waiting_key).unwrap();
            acquired_tx.send(()).unwrap();
            lease
        });
        attempted_rx.recv().unwrap();
        assert!(acquired_rx.try_recv().is_err());

        let unrelated = table
            .acquire(SessionLockKey::ExactBoundary(Arc::from([9u32].as_slice())))
            .unwrap();
        drop(unrelated);
        drop(first);
        acquired_rx.recv().unwrap();
        drop(handle.join().unwrap());
    }

    #[test]
    fn transaction_boundary_excludes_the_unconsumed_final_token() {
        assert_eq!(
            consumed_generation_boundary(10, &[], 0).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(
            consumed_generation_boundary(10, &[11], 1).unwrap(),
            vec![10]
        );
        assert_eq!(
            consumed_generation_boundary(10, &[11, 12, 13], 3).unwrap(),
            vec![10, 11, 12]
        );
        assert!(consumed_generation_boundary(10, &[11, 12], 1).is_err());
    }

    #[test]
    fn tool_detokenizer_preserves_grammar_specials_and_hides_stop_ids() {
        use tokenizers::models::bpe::BPE;
        use tokenizers::{AddedToken, Tokenizer};

        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|tool_call>", true),
            AddedToken::from("<|\"|>", true),
            AddedToken::from("<|channel>", true),
            AddedToken::from("<eos>", true),
        ]);
        let marker = tokenizer.token_to_id("<|tool_call>").unwrap();
        let quote = tokenizer.token_to_id("<|\"|>").unwrap();
        let channel = tokenizer.token_to_id("<|channel>").unwrap();
        let eos = tokenizer.token_to_id("<eos>").unwrap();

        let mut ordinary = StreamDetokenizer::new(&tokenizer);
        assert_eq!(ordinary.add(marker), "");

        let mut tools = StreamDetokenizer::for_tool_calls(
            &tokenizer,
            &[eos as usize],
            &["<|tool_call>", "<|\"|>"],
        );
        assert_eq!(tools.add(marker), "<|tool_call>");
        assert_eq!(tools.add(quote), "<|\"|>");
        assert_eq!(tools.add(channel), "");
        assert_eq!(tools.add(eos), "");
    }

    #[test]
    fn models_endpoint_accepts_openai_base_url_variants() {
        assert!(is_models_endpoint("/v1/models"));
        assert!(is_models_endpoint("/v1/models/"));
        assert!(is_models_endpoint("/v1/models?refresh=1"));
        assert!(is_models_endpoint("/models"));
        assert!(is_models_endpoint("/models/"));
        assert!(is_models_endpoint("/models?refresh=1"));
        assert!(!is_models_endpoint("/v1/chat/completions"));
        assert!(!is_models_endpoint("/foo/models"));
    }

    #[test]
    fn chat_endpoint_accepts_openai_base_url_variants() {
        assert!(is_chat_completions_endpoint("/v1/chat/completions"));
        assert!(is_chat_completions_endpoint("/v1/chat/completions/"));
        assert!(is_chat_completions_endpoint("/v1/chat/completions?x=1"));
        assert!(is_chat_completions_endpoint("/chat/completions"));
        assert!(is_chat_completions_endpoint("/chat/completions/"));
        assert!(is_chat_completions_endpoint("/chat/completions?x=1"));
        assert!(!is_chat_completions_endpoint("/v1/models"));
        assert!(!is_chat_completions_endpoint("/foo/chat/completions"));
    }

    #[test]
    fn context_window_accounts_for_prompt_and_requested_output() {
        assert!(context_window_fits(2047, 1, 2048));
        assert!(context_window_fits(1024, 1024, 2048));
        assert!(!context_window_fits(2048, 0, 2048));
        assert!(!context_window_fits(2047, 2, 2048));
        assert!(!context_window_fits(usize::MAX, usize::MAX, 2048));
    }

    #[test]
    fn hides_only_synthetic_thinking_stop_text() {
        assert!(hide_synthetic_think_stop_text(123, Some("stop"), Some(123)));
        assert!(!hide_synthetic_think_stop_text(
            123,
            Some("length"),
            Some(123)
        ));
        assert!(!hide_synthetic_think_stop_text(
            123,
            Some("stop"),
            Some(456)
        ));
        assert!(!hide_synthetic_think_stop_text(123, Some("stop"), None));
        assert!(!hide_synthetic_think_stop_text(123, None, Some(123)));
    }

    #[test]
    fn windows_model_path_round_trips_in_models_and_completion_responses() {
        let models = format_models_response(WINDOWS_MODEL_PATH, 32_768, false);
        let models_json = parse_response(&models);
        assert_eq!(models_json["data"][0]["id"], WINDOWS_MODEL_PATH);
        assert!(models.contains(r#""id":"C:\\Users\\stoate\\.krasis\\models\\"#));

        let vision_models = format_models_response(WINDOWS_MODEL_PATH, 32_768, true);
        let vision_json = parse_response(&vision_models);
        assert_eq!(
            vision_json["data"][0]["id"],
            format!("{}-vision", WINDOWS_MODEL_PATH)
        );

        let npc_reply = "Elara says, \"Good day.\"\nLooking to buy?";
        let completion = format_completion(
            "chatcmpl-test",
            WINDOWS_MODEL_PATH,
            npc_reply,
            17,
            9,
            "stop",
            123,
        );
        let completion_json = parse_response(&completion);
        assert_eq!(completion_json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(
            completion_json["choices"][0]["message"]["content"],
            npc_reply
        );
        assert!(
            completion.contains(r#""model":"C:\\Users\\stoate\\.krasis\\models\\Qwen3.6-35B-A3B""#)
        );

        let debug = serde_json::json!({"path": WINDOWS_MODEL_PATH});
        let with_debug = format_completion_with_debug(
            "chatcmpl-test",
            WINDOWS_MODEL_PATH,
            npc_reply,
            17,
            9,
            "stop",
            123,
            Some(&debug),
        );
        let debug_json = parse_response(&with_debug);
        assert_eq!(debug_json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(debug_json["krasis_debug"]["path"], WINDOWS_MODEL_PATH);
    }

    #[test]
    fn windows_model_path_round_trips_in_every_stream_chunk_type() {
        let token = format_sse_token(
            "chatcmpl-test",
            WINDOWS_MODEL_PATH,
            "quoted \"text\" with \\ and \u{0008}",
            Some("stop"),
            123,
            Some(&[(42, -0.25)]),
        );
        let token_json = parse_response(&token);
        assert_eq!(token_json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(
            token_json["choices"][0]["delta"]["content"],
            "quoted \"text\" with \\ and \u{0008}"
        );

        let start = format_sse_tool_call_start(
            "chatcmpl-test",
            WINDOWS_MODEL_PATH,
            0,
            "call_\"quoted\"",
            "inspect\\npc",
            123,
        );
        let start_json = parse_response(&start);
        assert_eq!(start_json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(
            start_json["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "inspect\\npc"
        );

        let arguments = r#"{"path":"C:\\Users\\stoate\\npc.json"}"#;
        let args =
            format_sse_tool_call_args("chatcmpl-test", WINDOWS_MODEL_PATH, 0, arguments, 123);
        let args_json = parse_response(&args);
        assert_eq!(args_json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(
            args_json["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            arguments
        );

        let overhead = RequestOverhead {
            parse_ms: 1.0,
            evict_ms: 2.0,
            prefill_ms: 3.0,
            reload_ms: 4.0,
            real_reload_dma_ms: 5.0,
        };
        let timing = format_sse_timing(
            "chatcmpl-test",
            WINDOWS_MODEL_PATH,
            123,
            7,
            70.0,
            100.0,
            0,
            8,
            8,
            17,
            200.0,
            10.0,
            &overhead,
        );
        let timing_json = parse_response(&timing);
        assert_eq!(timing_json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(timing_json["krasis_timing"]["decode_tokens"], 7);
    }

    #[test]
    fn windows_model_path_round_trips_in_nonstreaming_tool_response() {
        let tool_calls = vec![ParsedToolCall {
            id: "call_\"quoted\"".to_string(),
            name: "inspect\\npc".to_string(),
            arguments_json: r#"{"path":"C:\\Users\\stoate\\npc.json"}"#.to_string(),
        }];
        let response = format_completion_with_tool_calls(
            "chatcmpl-test",
            WINDOWS_MODEL_PATH,
            "Using \"inspect\".",
            &tool_calls,
            17,
            9,
            123,
        );
        let json = parse_response(&response);
        assert_eq!(json["model"], WINDOWS_MODEL_PATH);
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "inspect\\npc"
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            tool_calls[0].arguments_json
        );
    }

    #[test]
    fn parses_function_xml_with_multiple_calls_and_preserves_content() {
        let text = concat!(
            "Before\n<tool_call><function=weather><parameter=city>\nLondon\n</parameter>",
            "<parameter=days>2</parameter></function></tool_call>",
            "<tool_call><function=notify></function></tool_call>\nAfter",
        );
        let (content, calls) = parse_tool_calls(text, ToolCallFormat::FunctionXml);
        assert_eq!(content, "Before\n\nAfter");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(
            parsed_arguments(&calls[0]),
            serde_json::json!({"city":"London","days":2})
        );
        assert_eq!(calls[1].name, "notify");
        assert_eq!(parsed_arguments(&calls[1]), serde_json::json!({}));
    }

    #[test]
    fn parses_qwen_json_and_glm_xml_grammars() {
        let qwen =
            r#"<tool_call>{"name":"search","arguments":{"q":"Krasis","limit":3}}</tool_call>"#;
        let (_, qwen_calls) = parse_tool_calls(qwen, ToolCallFormat::QwenJson);
        assert_eq!(qwen_calls.len(), 1);
        assert_eq!(qwen_calls[0].name, "search");
        assert_eq!(parsed_arguments(&qwen_calls[0])["limit"], 3);

        let glm = concat!(
            "<tool_call>weather",
            "<arg_key>city</arg_key><arg_value>London</arg_value>",
            "<arg_key>days</arg_key><arg_value>2</arg_value>",
            "</tool_call>",
        );
        let (_, glm_calls) = parse_tool_calls(glm, ToolCallFormat::GlmXml);
        assert_eq!(glm_calls.len(), 1);
        assert_eq!(glm_calls[0].name, "weather");
        assert_eq!(
            parsed_arguments(&glm_calls[0]),
            serde_json::json!({"city":"London","days":2})
        );
    }

    #[test]
    fn parses_balanced_duplicate_glm_container_without_accepting_markup_as_name() {
        let duplicated = concat!(
            "Before<tool_call><tool_call>read",
            "<arg_key>filePath</arg_key>",
            "<arg_value>/tmp/proof.txt</arg_value>",
            "</tool_call></tool_call>After",
        );
        let (content, calls) = parse_tool_calls(duplicated, ToolCallFormat::GlmXml);
        assert_eq!(content, "BeforeAfter");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(
            parsed_arguments(&calls[0]),
            serde_json::json!({"filePath":"/tmp/proof.txt"})
        );

        let malformed = concat!(
            "Before<tool_call><tool_call>read",
            "<arg_key>filePath</arg_key>",
            "<arg_value>/tmp/proof.txt</arg_value>",
            "</tool_call>After",
        );
        let (content, calls) = parse_tool_calls(malformed, ToolCallFormat::GlmXml);
        assert!(calls.is_empty());
        assert_eq!(content, malformed);
    }

    #[test]
    fn parses_deepseek_dsml_typed_parameters_and_multiple_invokes() {
        let text = concat!(
            "Reasoning complete.\n<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"weather\">",
            "<｜DSML｜parameter name=\"city\" string=\"true\">London</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"units\" string=\"false\">[\"C\",\"F\"]</｜DSML｜parameter>",
            "</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"toggle\">",
            "<｜DSML｜parameter name=\"enabled\" string=\"false\">true</｜DSML｜parameter>",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>\nAfter",
        );
        let (content, calls) = parse_tool_calls(text, ToolCallFormat::DeepseekDsml);
        assert_eq!(content, "Reasoning complete.\n\nAfter");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(
            parsed_arguments(&calls[0]),
            serde_json::json!({"city":"London","days":2,"units":["C","F"]})
        );
        assert_eq!(
            parsed_arguments(&calls[1]),
            serde_json::json!({"enabled":true})
        );
    }

    #[test]
    fn parses_deepseek_observed_abbreviated_inner_tags() {
        let text = concat!(
            "\n\n<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"webfetch\">\n",
            "<parameter name=\"url\">https://github.com/brontoguana/krasis</parameter>\n",
            "</invoke>\n",
            "</｜DSML｜tool_calls>",
        );
        let (content, calls) = parse_tool_calls(text, ToolCallFormat::DeepseekDsml);
        assert!(content.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "webfetch");
        assert_eq!(
            parsed_arguments(&calls[0]),
            serde_json::json!({"url":"https://github.com/brontoguana/krasis"})
        );
    }

    #[test]
    fn parses_gemma_and_minimax_native_grammars() {
        let gemma = concat!(
            "<|tool_call>call:weather{city:<|\"|>London<|\"|>,days:2,",
            "options:{units:<|\"|>C<|\"|>,alerts:true},ids:[1,2]}<tool_call|>",
        );
        let (_, gemma_calls) = parse_tool_calls(gemma, ToolCallFormat::Gemma);
        assert_eq!(gemma_calls.len(), 1);
        assert_eq!(gemma_calls[0].name, "weather");
        assert_eq!(
            parsed_arguments(&gemma_calls[0]),
            serde_json::json!({
                "city":"London", "days":2,
                "options":{"units":"C","alerts":true}, "ids":[1,2]
            })
        );

        let minimax = concat!(
            "<minimax:tool_call><invoke name=\"weather\">",
            "<parameter name=\"city\">London</parameter>",
            "<parameter name=\"days\">2</parameter>",
            "</invoke></minimax:tool_call>",
        );
        let (_, minimax_calls) = parse_tool_calls(minimax, ToolCallFormat::Minimax);
        assert_eq!(minimax_calls.len(), 1);
        assert_eq!(
            parsed_arguments(&minimax_calls[0]),
            serde_json::json!({"city":"London","days":2})
        );
    }

    #[test]
    fn malformed_and_truncated_blocks_remain_text() {
        let malformed = concat!(
            "x<｜DSML｜tool_calls><｜DSML｜invoke name=\"bad\">",
            "<｜DSML｜parameter name=\"n\" string=\"false\">not-json</｜DSML｜parameter>",
            "</｜DSML｜invoke></｜DSML｜tool_calls>y",
        );
        let (content, calls) = parse_tool_calls(malformed, ToolCallFormat::DeepseekDsml);
        assert_eq!(content, malformed);
        assert!(calls.is_empty());

        let truncated = "x<tool_call><function=weather>";
        let (content, calls) = parse_tool_calls(truncated, ToolCallFormat::FunctionXml);
        assert_eq!(content, truncated);
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_detector_handles_utf8_marker_split_at_every_char_boundary() {
        let marker = "<｜DSML｜tool_calls>";
        for split in marker.char_indices().map(|(idx, _)| idx).skip(1) {
            let mut pending = String::new();
            let mut captured = String::new();
            let mut found = false;
            let first = format!("before{}", &marker[..split]);
            let visible =
                push_tool_stream_text(marker, &mut pending, &mut captured, &mut found, &first);
            assert_eq!(visible, "before");
            assert!(!found);
            let visible = push_tool_stream_text(
                marker,
                &mut pending,
                &mut captured,
                &mut found,
                &format!("{}body", &marker[split..]),
            );
            assert!(visible.is_empty());
            assert!(found);
            assert_eq!(captured, format!("{}body", marker));
        }
    }

    #[test]
    fn bundled_deepseek_template_round_trips_tool_history_and_result() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "krasis_server_dsml_roundtrip_{}_{}",
            std::process::id(),
            nonce
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::json!({"model_type":"deepseek_v4"}).to_string(),
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template":null,
                "bos_token":{"content":"<｜begin▁of▁sentence｜>"},
                "eos_token":{"content":"<｜end▁of▁sentence｜>"}
            })
            .to_string(),
        )
        .unwrap();
        let engine =
            ChatTemplateEngine::from_config(dir.join("tokenizer_config.json").to_str().unwrap())
                .unwrap();
        assert_eq!(engine.tool_call_format(), ToolCallFormat::DeepseekDsml);
        let rendered = engine
            .apply_with_tools(
                r#"[{"role":"user","content":"Check weather"},{"role":"assistant","content":"","tool_calls":[{"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"city\":\"London\",\"days\":2}"}}]},{"role":"tool","tool_call_id":"call_weather","content":"sunny"}]"#,
                r#"[{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]"#,
                true,
                false,
            )
            .unwrap();
        assert!(rendered.contains("<tool_result>sunny</tool_result>"));
        let (_, calls) = parse_tool_calls(&rendered, engine.tool_call_format());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(
            parsed_arguments(&calls[0]),
            serde_json::json!({"city":"London","days":2})
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn template_engine_dispatches_all_supported_tool_grammars() {
        let cases = [
            (
                r#"<tool_call>{"name": <function-name>, "arguments": {}}</tool_call>"#,
                ToolCallFormat::QwenJson,
            ),
            (
                "<tool_call><function=example_function_name><parameter=x></parameter></function></tool_call>",
                ToolCallFormat::FunctionXml,
            ),
            (
                "<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>",
                ToolCallFormat::GlmXml,
            ),
            (
                "{% set dsml_token = '｜DSML｜' %}<invoke name=\"tool\">",
                ToolCallFormat::DeepseekDsml,
            ),
            (
                "<|tool_call>call:name{}<tool_call|>",
                ToolCallFormat::Gemma,
            ),
            (
                "{% set x = '<minimax:tool_call>' %}<invoke name=\"tool\">",
                ToolCallFormat::Minimax,
            ),
            ("User: {{ message.content }}", ToolCallFormat::Unsupported),
        ];
        for (idx, (template, expected)) in cases.into_iter().enumerate() {
            let dir = std::env::temp_dir().join(format!(
                "krasis_server_tool_dispatch_{}_{}",
                std::process::id(),
                idx
            ));
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("tokenizer_config.json"),
                serde_json::json!({
                    "chat_template":template,
                    "bos_token":"<s>",
                    "eos_token":"</s>"
                })
                .to_string(),
            )
            .unwrap();
            let engine = ChatTemplateEngine::from_config(
                dir.join("tokenizer_config.json").to_str().unwrap(),
            )
            .unwrap();
            assert_eq!(engine.tool_call_format(), expected, "{template}");
            let _ = fs::remove_dir_all(dir);
        }
    }
}
