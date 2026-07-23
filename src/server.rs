//! Rust HTTP server for Krasis — replaces Python FastAPI/uvicorn.
//!
//! Handles tokenization, HTTP parsing, and SSE streaming entirely in Rust.
//! Prefill and decode run in Rust on the production request path.
//! Python remains for startup/orchestration and model ownership.
//!
//! Single-request at a time (matches our hardware constraint).

use crate::gpu_decode::GpuDecodeStore;

/// Streaming detokenizer that buffers incomplete UTF-8 sequences.
///
/// Some characters (emojis, CJK, etc.) span multiple tokens in byte-level BPE.
/// Decoding each token individually produces incomplete UTF-8 bytes → U+FFFD.
/// This struct buffers tokens until the decoded text contains no trailing FFFD,
/// then emits the complete text.
pub struct StreamDetokenizer<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    pending: Vec<u32>,
}

impl<'a> StreamDetokenizer<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::new(),
        }
    }

    /// Add a token. Returns the decoded text if the sequence is complete UTF-8,
    /// or an empty string if we're still buffering incomplete bytes.
    pub fn add(&mut self, token_id: u32) -> String {
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
use pyo3::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

fn abort_if_cuda_context_poisoned(context: &str, err: &str) {
    if err.contains("CUDA_ERROR_ILLEGAL_ADDRESS")
        || err.to_ascii_lowercase().contains("illegal address")
    {
        crate::vram_monitor::fatal_cuda_context_error(context, err);
    }
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
    /// Safety: single-request guarantee means no concurrent access.
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
}

#[derive(Clone)]
struct ServerInfo {
    model_name: String,
    max_context_tokens: usize,
    supports_vision: bool,
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
    Chat { stream: TcpStream, body: String },
    PrefillLogits { stream: TcpStream, body: String },
    ReferenceTest { stream: TcpStream, body: String },
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

fn handle_front_connection(
    mut tcp_stream: TcpStream,
    server_info: ServerInfo,
    model_tx: mpsc::Sender<ModelRequest>,
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

        ("GET", path) if is_models_endpoint(path) => {
            let data = if server_info.supports_vision {
                let vision_id = json_escape(&format!("{}-vision", server_info.model_name));
                format!(
                    r#"{{"id":"{}","object":"model","created":0,"owned_by":"krasis","max_context_tokens":{},"capabilities":{{"vision":true}},"input_modalities":["text","image"]}}"#,
                    vision_id, server_info.max_context_tokens
                )
            } else {
                let model_id = json_escape(&server_info.model_name);
                format!(
                    r#"{{"id":"{}","object":"model","created":0,"owned_by":"krasis","max_context_tokens":{},"capabilities":{{"vision":false}}}}"#,
                    model_id, server_info.max_context_tokens
                )
            };
            let body = format!(r#"{{"object":"list","data":[{}]}}"#, data);
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("POST", path) if is_chat_completions_endpoint(path) => {
            if model_tx
                .send(ModelRequest::Chat {
                    stream: tcp_stream,
                    body: request.body,
                })
                .is_err()
            {
                log::error!("Model worker is not available for /v1/chat/completions");
            }
        }

        ("POST", "/v1/internal/prefill_logits") => {
            if test_endpoints {
                if model_tx
                    .send(ModelRequest::PrefillLogits {
                        stream: tcp_stream,
                        body: request.body,
                    })
                    .is_err()
                {
                    log::error!("Model worker is not available for /v1/internal/prefill_logits");
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
                if model_tx
                    .send(ModelRequest::ReferenceTest {
                        stream: tcp_stream,
                        body: request.body,
                    })
                    .is_err()
                {
                    log::error!("Model worker is not available for /v1/internal/reference_test");
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
        let escaped = text
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        format!(r#"{{"content":"{}"}}"#, escaped)
    };
    let fr = match finish_reason {
        Some(r) => format!(r#""{}""#, r),
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
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{},"finish_reason":{}{}}}]}}"#,
        request_id, created, model_name, delta, fr, logprobs_str
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
    let escaped = text
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!(
        r#"{{"id":"{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"{}"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        request_id,
        created,
        model_name,
        escaped,
        finish_reason,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens
    )
}

// ── Tool use support ──────────────────────────────────────────────

/// A parsed tool call extracted from model output.
struct ParsedToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

/// Parse tool calls from model-generated text (Qwen XML format).
/// Returns (content_text, tool_calls).
/// Content is everything outside `<tool_call>...</tool_call>` blocks.
fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    let mut tool_calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;
    let mut call_idx = 0u64;

    while let Some(start) = remaining.find("<tool_call>") {
        content.push_str(&remaining[..start]);
        remaining = &remaining[start + "<tool_call>".len()..];

        if let Some(end) = remaining.find("</tool_call>") {
            let block = remaining[..end].trim();
            remaining = &remaining[end + "</tool_call>".len()..];

            // Parse <function=name>
            if let Some(fn_start) = block.find("<function=") {
                let after = &block[fn_start + "<function=".len()..];
                if let Some(fn_end) = after.find('>') {
                    let name = after[..fn_end].to_string();
                    let inner = &after[fn_end + 1..];

                    // Find </function> boundary
                    let params_text = if let Some(fe) = inner.find("</function>") {
                        &inner[..fe]
                    } else {
                        inner
                    };

                    // Parse <parameter=name>value</parameter> pairs
                    let mut args = serde_json::Map::new();
                    let mut param_rem = params_text;
                    while let Some(p_start) = param_rem.find("<parameter=") {
                        let after_p = &param_rem[p_start + "<parameter=".len()..];
                        if let Some(p_name_end) = after_p.find('>') {
                            let param_name = after_p[..p_name_end].to_string();
                            let value_text = &after_p[p_name_end + 1..];
                            if let Some(p_end) = value_text.find("</parameter>") {
                                let value = value_text[..p_end]
                                    .trim_start_matches('\n')
                                    .trim_end_matches('\n');
                                // Try JSON parse (objects, arrays, numbers, bools)
                                let json_value = serde_json::from_str(value).unwrap_or_else(|_| {
                                    serde_json::Value::String(value.to_string())
                                });
                                args.insert(param_name, json_value);
                                param_rem = &value_text[p_end + "</parameter>".len()..];
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    // Generate unique call ID
                    let id = format!("call_{:016x}", {
                        let mut s = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        s ^= call_idx;
                        s
                    });
                    call_idx += 1;

                    tool_calls.push(ParsedToolCall {
                        id,
                        name,
                        arguments_json: serde_json::Value::Object(args).to_string(),
                    });
                }
            }
        } else {
            // No closing tag — treat as content
            content.push_str("<tool_call>");
            content.push_str(remaining);
            remaining = "";
        }
    }

    content.push_str(remaining);
    (content.trim().to_string(), tool_calls)
}

/// Escape a string for embedding inside a JSON string value.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
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
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"id":"{}","type":"function","function":{{"name":"{}","arguments":""}}}}]}},"finish_reason":null}}]}}"#,
        request_id, created, model_name, call_index, call_id, function_name
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
    let escaped = json_escape(arguments_json);
    format!(
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"function":{{"arguments":"{}"}}}}]}},"finish_reason":null}}]}}"#,
        request_id, created, model_name, call_index, escaped
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
        let escaped_args = json_escape(&tc.arguments_json);
        tc_parts.push(format!(
            r#"{{"id":"{}","type":"function","function":{{"name":"{}","arguments":"{}"}}}}"#,
            tc.id, tc.name, escaped_args
        ));
    }
    let content_field = if content.is_empty() {
        "null".to_string()
    } else {
        format!(r#""{}""#, json_escape(content))
    };
    format!(
        r#"{{"id":"{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":{},"tool_calls":[{}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        request_id,
        created,
        model_name,
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
    image_count: usize,
    image_tokens: usize,
}

/// Handle /v1/chat/completions request.
fn handle_chat_completion(stream: &mut TcpStream, body: &str, state: &mut ServerState) {
    let t_request = Instant::now();

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

    if !has_images && estimated_tokens >= state.max_context_tokens {
        log::warn!(
            "Request {} rejected before prefill: estimated prompt {} tokens exceeds context {}",
            request_id,
            estimated_tokens,
            state.max_context_tokens,
        );
        let _ = send_json(
            stream,
            413,
            &format!(
                r#"{{"error":{{"message":"Prompt too long: {} tokens exceeds KV cache capacity of {} tokens","type":"invalid_request_error","code":"context_length_exceeded","prompt_tokens":{},"max_context_tokens":{}}}}}"#,
                estimated_tokens,
                state.max_context_tokens,
                estimated_tokens,
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
    let prefill_result: Result<
        (usize, usize, Vec<usize>, bool, Option<serde_json::Value>),
        String,
    > = {
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
        if debug_first_token_boundary {
            chat_debug_input_token_ids = Some(token_ids.clone());
        }
        let mut engine_guard = state.rust_prefill.lock().unwrap();
        let engine = engine_guard.as_mut().unwrap();
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

        engine.set_prefill_hcs_guard_store_addr(state.gpu_store_addr);

        let mut retry_cap: Option<usize> = None;
        let mut retry_attempt = 0usize;
        let result = loop {
            engine.set_prefill_runtime_chunk_cap(retry_cap);

            // Dynamically allocate scratch sized for this prompt.
            if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
                engine.clear_external_prefill_inputs();
                engine.clear_prefill_hcs_guard_store_addr();
                engine.set_optional_pinning_budget_mb(None);
                engine.clear_prefill_runtime_chunk_cap();
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                let _ = store.prepare_runtime_for_decode_rust();
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
                );
            } else {
                engine.clear_external_prefill_inputs();
            }

            let attempt_result = match engine.run_prefill(&token_ids, temperature, &suppress_tokens)
            {
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

        // Release scratch to free VRAM for decode/HCS
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
            Ok(r) => {
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

    let prefill_gil_ms = t_prefill_gil.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("prefill_end");

    let (first_token, prompt_len, stop_ids, kv_overflow, chat_debug_payload) = match prefill_result
    {
        Ok(v) => v,
        Err(e) => {
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

    // If prompt exceeded Rust KV cache, return error (not a silent 200 with truncated output)
    if kv_overflow {
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
    if prompt_len >= state.max_context_tokens {
        let _ = send_json(
            stream,
            413,
            &format!(
                r#"{{"error":{{"message":"Prompt too long: {} tokens exceeds KV cache capacity of {} tokens","type":"invalid_request_error","code":"context_length_exceeded","prompt_tokens":{},"max_context_tokens":{}}}}}"#,
                prompt_len, state.max_context_tokens, prompt_len, state.max_context_tokens
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

    let tokenizer = &state.tokenizer;

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
    // ── Multi-GPU: copy KV+LA state from primary to all aux GPUs after prefill ──
    if !state.aux_gpu_store_addrs.is_empty() {
        let t_kvcopy = Instant::now();
        let num_aux = state.aux_gpu_store_addrs.len();
        let num_layers = store.num_layers();
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
        }
        let kvcopy_ms = t_kvcopy.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "Request {}: KV+LA state copied to {} aux GPUs in {:.1}ms",
            request_id,
            num_aux,
            kvcopy_ms
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

    // ── GPU decode: GIL-free Rust decode via GpuDecodeStore ──
    crate::vram_monitor::report_event("decode_start");
    handle_gpu_decode(
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
        enable_thinking,
        logprobs_top_n,
        chat_debug_payload,
    );
    crate::vram_monitor::report_event("decode_end");

    // ── Cleanup (GIL required) ──
    let t_cleanup_gil = Instant::now();
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
    let cleanup_gil_ms = t_cleanup_gil.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("cleanup_end");
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

    let total_ms = t_request.elapsed().as_secs_f64() * 1000.0;
    log::info!(
        "Request {} complete: total={:.0}ms | parse={:.1}ms evict={:.1}ms prefill={:.0}ms reload={:.0}ms cleanup={:.1}ms",
        request_id, total_ms, parse_ms, evict_ms, prefill_gil_ms, reload_ms, cleanup_gil_ms
    );
}

/// Handle /v1/internal/prefill_logits endpoint.
/// Runs a full prefill pass and extracts top-k logprobs at sampled positions.
fn handle_prefill_logits(stream: &mut TcpStream, body: &str, state: &mut ServerState) {
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

    log::info!(
        "prefill_logits: {} tokens, top_k={}, sample_every={}, target_logprobs={}",
        token_ids.len(),
        top_k,
        sample_every,
        target_token_ids.is_some()
    );

    // Evict soft HCS before diagnostic prefill so this endpoint uses the same
    // conservative VRAM budget as the production and reference-test paths.
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

    // Run prefill logits extraction
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

    let _has_hqq_runtime_slots = {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        match prepare_store_for_rust_prefill(store, engine, token_ids.len()) {
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

    // Dynamically allocate scratch for this prompt
    // run_prefill_logits needs scratch sized for all tokens (no chunking)
    if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        let _ = store.prepare_runtime_for_decode_rust();
        store.invalidate_cuda_graph();
        log::info!(
            "prefill_logits: invalidated CUDA graphs after failed scratch allocation restore"
        );
        let _ = send_json(
            stream,
            500,
            &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e),
        );
        return;
    }

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
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = store.hcs_reload_after_prefill(token_ids.len());
            store.invalidate_cuda_graph();
            log::info!(
                "prefill_logits: invalidated CUDA graphs after failed diagnostic prefill restore"
            );
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Prefill logits: {}"}}"#, e),
            );
            return;
        }
    };

    // Release scratch after logits extraction
    if let Err(e) = engine.release_scratch() {
        log::error!("Failed to release scratch after prefill_logits: {}", e);
        abort_if_cuda_context_poisoned("prefill_logits release_scratch", &e);
    }

    // Restore evicted soft HCS so the next decode/reference request starts
    // from the normal steady-state cache residency.
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let _ = store.prepare_runtime_for_decode_rust();
    let _ = store.hcs_reload_after_prefill(token_ids.len());
    log::info!("prefill_logits: restored decode runtime after diagnostic prefill");

    // Match the normal reference/inference cleanup path so diagnostic prefill
    // requests do not leak sequence state into the next prompt.
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });

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
    let _ = send_json(stream, 200, &response);
}

/// Handle /v1/internal/reference_test endpoint.
/// Accepts raw input_token_ids, runs greedy prefill + decode, returns output tokens with logprobs.
/// Used for comparing engine output against BF16 reference data.
fn handle_reference_test(stream: &mut TcpStream, body: &str, state: &mut ServerState) {
    let t_start = Instant::now();
    state.reference_test_request_order = state.reference_test_request_order.saturating_add(1);
    let reference_request_order = state.reference_test_request_order;

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
                "prompt_tokens": prompt_len
            }
        });
        debug_json_suffix.push_str(&format!(r#","debug_reference_trace":{}"#, trace));
    }

    let response = format!(
        r#"{{"token_ids":[{}],"text":{},"num_tokens":{},"per_token_data":[{}],"first_token_top_k":[{}],"finish_reason":"{}","timing":{{"prefill_ms":{:.1},"decode_ms":{:.1},"total_ms":{:.1},"prompt_tokens":{}}}{}}}"#,
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
    enable_thinking: bool,
    logprobs_top_n: usize,
    chat_debug_payload: Option<serde_json::Value>,
) {
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

    if is_stream {
        if let Err(e) = begin_sse(stream) {
            log::error!("Failed to send SSE headers: {}", e);
            return;
        }

        let first_text = tokenizer
            .decode(&[first_token as u32], true)
            .unwrap_or_default();

        // When thinking is enabled, inject <think> at start of stream.
        // The prompt already includes <think>, but the client needs it in the
        // output to know this is a thinking block (for display suppression).
        if think_end_id.is_some() {
            let think_chunk =
                format_sse_token(request_id, model_name, "<think>", None, created, None);
            let _ = send_sse_chunk(stream, &think_chunk);
        }

        // When tool use is active, buffer first token (might need tool call parsing).
        // Otherwise send immediately for lowest latency.
        if !has_tools {
            let chunk = format_sse_token(request_id, model_name, &first_text, None, created, None);
            let _ = send_sse_chunk(stream, &chunk);
        }

        let (tx, rx) = mpsc::channel::<String>();
        let writer_disconnected = Arc::new(AtomicBool::new(false));
        let writer_disc_clone = writer_disconnected.clone();

        let mut writer_stream = match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to clone stream for writer: {}", e);
                return;
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
                            let _ = writer_stream.write_all(buf.as_bytes());
                            let _ = writer_stream.flush();
                        }
                        return;
                    }
                }
            }
        });

        let decode_start = Instant::now();
        let mut decode_token_count = 0usize;

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
        // When tools are present: stream content normally, detect <tool_call>,
        // buffer everything from that point, then send structured tool_calls
        // at the end.  Content before tool calls streams with full latency.
        let mut tc_all_text = String::new();
        let mut tc_in_tool_call = false;
        let mut tc_found = false;
        let mut tc_finish = String::new();

        if has_tools {
            tc_all_text.push_str(&first_text);
            // Send first token if it's safe (doesn't contain tool call marker)
            if first_text.contains("<tool_call>") {
                tc_in_tool_call = true;
                tc_found = true;
                // Send content before the marker
                if let Some(idx) = first_text.find("<tool_call>") {
                    let before = &first_text[..idx];
                    if !before.is_empty() {
                        let chunk =
                            format_sse_token(request_id, model_name, before, None, created, None);
                        let _ = tx.send(format!("data: {}\n\n", chunk));
                    }
                }
            } else if !first_text.is_empty() {
                let chunk =
                    format_sse_token(request_id, model_name, &first_text, None, created, None);
                let _ = tx.send(format!("data: {}\n\n", chunk));
            }
        }

        // Shared callback for both single-GPU and multi-GPU decode
        let mut on_token = |token_id: usize,
                            text: &str,
                            finish_reason: Option<&str>,
                            token_logprobs: Option<&[(u32, f32)]>|
         -> bool {
            decode_token_count += 1;

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
            let hide_text =
                hide_synthetic_think_stop_text(token_id, effective_finish, hidden_think_stop_id);
            let visible_text = if hide_text { "" } else { text };

            if has_tools {
                tc_all_text.push_str(visible_text);
                if let Some(fr) = effective_finish {
                    tc_finish = fr.to_string();
                }

                if tc_in_tool_call {
                    // Inside a tool call block — buffer silently
                } else if visible_text.contains("<tool_call>") {
                    // Entering tool call territory
                    tc_in_tool_call = true;
                    tc_found = true;
                    // Send any content before the marker in this text
                    if let Some(idx) = visible_text.find("<tool_call>") {
                        let before = &visible_text[..idx];
                        if !before.is_empty() {
                            let chunk = format_sse_token(
                                request_id, model_name, before, None, created, None,
                            );
                            let _ = tx.send(format!("data: {}\n\n", chunk));
                        }
                    }
                } else {
                    // Normal content — stream it (no finish_reason; handled post-generation)
                    if !visible_text.is_empty() {
                        let chunk = format_sse_token(
                            request_id,
                            model_name,
                            visible_text,
                            None,
                            created,
                            token_logprobs,
                        );
                        let _ = tx.send(format!("data: {}\n\n", chunk));
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
        let decode_budget = if think_end_id.is_some() {
            max_tokens.saturating_add(32768).saturating_sub(1)
        } else {
            max_tokens.saturating_sub(1)
        };

        if !state.aux_gpu_store_addrs.is_empty() {
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
                presence_penalty,
                logprobs_top_n,
                Some(format!("chat_{}", request_id)),
                &mut on_token,
            );
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
                presence_penalty,
                logprobs_top_n,
                Some(format!("chat_{}", request_id)),
                on_token,
            );
        }

        // Capture decode timing BEFORE post-generation processing (tool call parsing etc.)
        let decode_elapsed = decode_start.elapsed().as_secs_f64();

        // ── Post-generation: emit tool calls or finish ──
        if has_tools {
            let (_content, tool_calls) = parse_tool_calls(&tc_all_text);
            if !tool_calls.is_empty() {
                // Content before tool calls was already streamed in the callback.
                // Now send the structured tool_call chunks.
                for (i, tc) in tool_calls.iter().enumerate() {
                    let start_chunk = format_sse_tool_call_start(
                        request_id, model_name, i, &tc.id, &tc.name, created,
                    );
                    let _ = tx.send(format!("data: {}\n\n", start_chunk));
                    let args_chunk = format_sse_tool_call_args(
                        request_id,
                        model_name,
                        i,
                        &tc.arguments_json,
                        created,
                    );
                    let _ = tx.send(format!("data: {}\n\n", args_chunk));
                }
                let finish_chunk = format_sse_token(
                    request_id,
                    model_name,
                    "",
                    Some("tool_calls"),
                    created,
                    None,
                );
                let _ = tx.send(format!("data: {}\n\n", finish_chunk));
                log::info!(
                    "Request {}: {} tool call(s) detected",
                    request_id,
                    tool_calls.len()
                );
            } else {
                // No tool calls — send finish with original reason
                let fr = if tc_finish.is_empty() {
                    "stop"
                } else {
                    &tc_finish
                };
                let finish_chunk =
                    format_sse_token(request_id, model_name, "", Some(fr), created, None);
                let _ = tx.send(format!("data: {}\n\n", finish_chunk));
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
        let timing_chunk = format!(
            r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[],"krasis_timing":{{"decode_tokens":{},"decode_time_ms":{:.1},"decode_tok_s":{:.2},"thinking_tokens":{},"answer_tokens":{},"total_generated":{},"prompt_tokens":{},"prefill_tok_s":{:.1},"overhead_ms":{:.1},"overhead":{{"parse_ms":{:.1},"evict_ms":{:.1},"prefill_ms":{:.1},"reload_ms":{:.1},"real_reload_dma_ms":{:.1}}}}}}}"#,
            request_id,
            created,
            model_name,
            decode_token_count,
            decode_ms,
            decode_tok_s,
            reported_thinking_tokens,
            reported_answer_tokens,
            total_gen,
            prompt_len,
            prefill_tok_s,
            overhead_total_ms,
            overhead.parse_ms,
            overhead.evict_ms,
            overhead.prefill_ms,
            overhead.reload_ms,
            overhead.real_reload_dma_ms
        );
        let _ = tx.send(format!("data: {}\n\n", timing_chunk));
        let _ = tx.send("data: [DONE]\n\n".to_string());
        drop(tx);
        let _ = writer_handle.join();

        log::info!(
            "Request {} complete: decode={:.2}s ({} tok, {:.1} tok/s) | overhead={:.0}ms (parse={:.1} evict={:.1} prefill={:.0} reload={:.0})",
            request_id, elapsed, total_gen, decode_tok_s,
            overhead_total_ms, overhead.parse_ms, overhead.evict_ms, overhead.prefill_ms, overhead.reload_ms
        );
    } else {
        // ── Non-streaming path ──
        let mut all_text = String::new();
        // Inject <think> prefix so clients can identify thinking blocks
        if enable_thinking && state.thinking_end_token.is_some() {
            all_text.push_str("<think>");
        }
        let first_text = tokenizer
            .decode(&[first_token as u32], true)
            .unwrap_or_default();
        all_text.push_str(&first_text);
        let mut total_tokens = 1usize;
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

        let ns_decode_budget = if ns_think_end_id.is_some() {
            max_tokens.saturating_add(32768).saturating_sub(1)
        } else {
            max_tokens.saturating_sub(1)
        };

        {
            let mut on_token = |token_id: usize,
                                text: &str,
                                finish_reason: Option<&str>,
                                token_logprobs: Option<&[(u32, f32)]>|
             -> bool {
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
            if !state.aux_gpu_store_addrs.is_empty() {
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
                    presence_penalty,
                    logprobs_top_n,
                    Some(format!("chat_{}_nosse", request_id)),
                    &mut on_token,
                );
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
                    presence_penalty,
                    logprobs_top_n,
                    Some(format!("chat_{}_nosse", request_id)),
                    on_token,
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

        if has_tools {
            let (content, tool_calls) = parse_tool_calls(&all_text);
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
                let _ = send_json(stream, 200, &response);
                log::info!(
                    "Request {}: {} tool call(s) (non-streaming)",
                    request_id,
                    tool_calls.len()
                );
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
                let _ = send_json(stream, 200, &response);
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
            let _ = send_json(stream, 200, &response);
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
}

#[pymethods]
impl RustServer {
    #[new]
    #[pyo3(signature = (py_model, host, port, model_name, tokenizer_path, max_context_tokens, enable_thinking=true, thinking_end_token_id=0, gpu_store_addr=0, aux_gpu_store_addrs=Vec::new(), multi_gpu_split_layers=Vec::new(), multi_gpu_gqa_offsets=Vec::new(), supports_vision=false, test_endpoints=false))]
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
            };

            let server_info = ServerInfo {
                model_name: state.model_name.clone(),
                max_context_tokens: state.max_context_tokens,
                supports_vision: self.supports_vision,
            };
            let (model_tx, model_rx) = mpsc::channel::<ModelRequest>();
            let worker_running = running.clone();
            let worker_handle = std::thread::Builder::new()
                .name("krasis-model-worker".to_string())
                .spawn(move || {
                    let mut state = state;
                    while worker_running.load(Ordering::Acquire) {
                        match model_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(ModelRequest::Chat { mut stream, body }) => {
                                handle_chat_completion(&mut stream, &body, &mut state);
                            }
                            Ok(ModelRequest::PrefillLogits { mut stream, body }) => {
                                handle_prefill_logits(&mut stream, &body, &mut state);
                            }
                            Ok(ModelRequest::ReferenceTest { mut stream, body }) => {
                                handle_reference_test(&mut stream, &body, &mut state);
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                drain_vram_pressure_for_state(&mut state, "idle", false);
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }

                    while let Ok(req) = model_rx.try_recv() {
                        match req {
                            ModelRequest::Chat { mut stream, body } => {
                                handle_chat_completion(&mut stream, &body, &mut state);
                            }
                            ModelRequest::PrefillLogits { mut stream, body } => {
                                handle_prefill_logits(&mut stream, &body, &mut state);
                            }
                            ModelRequest::ReferenceTest { mut stream, body } => {
                                handle_reference_test(&mut stream, &body, &mut state);
                            }
                        }
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
                        let tx = model_tx.clone();
                        let endpoints_enabled = test_endpoints;
                        if let Err(e) = std::thread::Builder::new()
                            .name("krasis-http-connection".to_string())
                            .spawn(move || {
                                handle_front_connection(stream, info, tx, endpoints_enabled);
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

            drop(model_tx);
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
    use super::{hide_synthetic_think_stop_text, is_chat_completions_endpoint, is_models_endpoint};

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
}
