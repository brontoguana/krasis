//! Rust HTTP server for Krasis — replaces Python FastAPI/uvicorn.
//!
//! Handles tokenization, HTTP parsing, and SSE streaming entirely in Rust.
//! Python is called only for GPU prefill (unavoidable — PyTorch/CUDA).
//! The decode loop runs GIL-free with zero Python involvement.
//!
//! Single-request at a time (matches our hardware constraint).

use crate::decode::CpuDecodeStore;
use pyo3::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Server state shared across request handling.
struct ServerState {
    py_model: Py<PyAny>,
    model_name: String,
    tokenizer_path: String,
    max_context_tokens: usize,
    running: Arc<AtomicBool>,
}

/// Parsed HTTP request.
struct HttpRequest {
    method: String,
    path: String,
    body: String,
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
) -> String {
    let delta = if finish_reason.is_some() {
        "{}".to_string()
    } else {
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"")
            .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
        format!(r#"{{"content":"{}"}}"#, escaped)
    };
    let fr = match finish_reason {
        Some(r) => format!(r#""{}""#, r),
        None => "null".to_string(),
    };
    format!(
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{},"finish_reason":{}}}]}}"#,
        request_id, created, model_name, delta, fr
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
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"")
        .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
    format!(
        r#"{{"id":"{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"{}"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        request_id, created, model_name, escaped, finish_reason,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
    )
}

/// Handle a single HTTP request.
fn handle_request(
    mut tcp_stream: TcpStream,
    state: &ServerState,
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
            log::error!("Failed to parse request: {}", e);
            let _ = send_json(&mut tcp_stream, 400, r#"{"error":"Bad request"}"#);
            return;
        }
    };

    // Handle CORS preflight
    if request.method == "OPTIONS" {
        let _ = write!(
            tcp_stream,
            "HTTP/1.1 204 No Content\r\n\
             Access-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
             Connection: close\r\n\r\n"
        );
        return;
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let body = format!(
                r#"{{"status":"ok","max_context_tokens":{}}}"#,
                state.max_context_tokens
            );
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("GET", "/v1/models") => {
            let body = format!(
                r#"{{"object":"list","data":[{{"id":"{}","object":"model","owned_by":"krasis"}}]}}"#,
                state.model_name
            );
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("POST", "/v1/chat/completions") => {
            handle_chat_completion(&mut tcp_stream, &request.body, state);
        }

        _ => {
            let _ = send_json(&mut tcp_stream, 404, r#"{"error":"Not found"}"#);
        }
    }
}

/// Handle /v1/chat/completions request.
fn handle_chat_completion(
    stream: &mut TcpStream,
    body: &str,
    state: &ServerState,
) {
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

    let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    let temperature = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.6) as f32;
    let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32;
    let presence_penalty = req.get("presence_penalty").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let enable_thinking = req.get("enable_thinking").and_then(|v| v.as_bool()).unwrap_or(true);

    let request_id = format!("chatcmpl-{:016x}", {
        let mut s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        s
    });
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Extract messages JSON for Python
    let messages_json = match req.get("messages") {
        Some(m) => m.to_string(),
        None => {
            let _ = send_json(stream, 400, r#"{"error":"Missing messages"}"#);
            return;
        }
    };

    // Custom stop tokens
    let stop_tokens: Vec<String> = match req.get("stop") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => {
            arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
        }
        _ => vec![],
    };

    // ── Call Python for prefill (GIL required) ──
    let prefill_result = Python::with_gil(|py| -> PyResult<(usize, usize, Vec<usize>, usize)> {
        let result = state.py_model.call_method(
            py,
            "server_prefill",
            (
                &messages_json,
                max_tokens,
                temperature,
                top_k,
                top_p,
                presence_penalty,
                enable_thinking,
                stop_tokens.clone(),
            ),
            None,
        )?;
        let first_token: usize = result.getattr(py, "first_token")?.extract(py)?;
        let prompt_len: usize = result.getattr(py, "prompt_len")?.extract(py)?;
        let stop_ids: Vec<usize> = result.getattr(py, "stop_ids")?.extract(py)?;
        let store_addr: usize = result.getattr(py, "store_addr")?.extract(py)?;
        Ok((first_token, prompt_len, stop_ids, store_addr))
    });

    let (first_token, prompt_len, stop_ids, store_addr) = match prefill_result {
        Ok(v) => v,
        Err(e) => {
            log::error!("Prefill failed: {}", e);
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Prefill failed: {}"}}"#, e),
            );
            // Cleanup on error
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };

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
        "Request {}: {} prompt tokens, max_new={}, stream={}",
        request_id, prompt_len, max_tokens, is_stream
    );

    // ── Load tokenizer for decode (pure Rust, ~10ms one-time) ──
    let tokenizer = match tokenizers::Tokenizer::from_file(&state.tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            log::error!("Failed to load tokenizer: {}", e);
            let _ = send_json(stream, 500, r#"{"error":"Tokenizer load failed"}"#);
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };

    // ── Decode (GIL-free!) ──
    // Safety: single-request guarantee. store_addr is a valid *mut CpuDecodeStore
    // obtained from Python during prefill. No other code touches it until cleanup.
    let store = unsafe { &mut *(store_addr as *mut CpuDecodeStore) };

    if is_stream {
        // ── Streaming SSE ──
        if let Err(e) = begin_sse(stream) {
            log::error!("Failed to send SSE headers: {}", e);
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }

        // Emit first token
        let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
        let chunk = format_sse_token(&request_id, &state.model_name, &first_text, None, created);
        let _ = send_sse_chunk(stream, &chunk);

        let decode_start = Instant::now();

        store.generate_stream(
            first_token,
            prompt_len,
            max_tokens.saturating_sub(1),
            temperature,
            top_k,
            top_p,
            &stop_ids,
            &tokenizer,
            presence_penalty,
            |_token_id, text, finish_reason| {
                let chunk = format_sse_token(
                    &request_id,
                    &state.model_name,
                    text,
                    finish_reason,
                    created,
                );
                if send_sse_chunk(stream, &chunk).is_err() {
                    return false; // Client disconnected
                }
                true
            },
        );

        let _ = send_sse_chunk(stream, "[DONE]");

        let elapsed = decode_start.elapsed().as_secs_f64();
        log::info!("Request {} streaming complete ({:.2}s)", request_id, elapsed);
    } else {
        // ── Non-streaming ──
        let mut all_text = String::new();
        let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
        all_text.push_str(&first_text);
        let mut total_tokens = 1usize;
        let mut finish = "length".to_string();

        store.generate_stream(
            first_token,
            prompt_len,
            max_tokens.saturating_sub(1),
            temperature,
            top_k,
            top_p,
            &stop_ids,
            &tokenizer,
            presence_penalty,
            |_token_id, text, finish_reason| {
                all_text.push_str(text);
                total_tokens += 1;
                if let Some(fr) = finish_reason {
                    finish = fr.to_string();
                }
                true
            },
        );

        let response = format_completion(
            &request_id,
            &state.model_name,
            &all_text,
            prompt_len,
            total_tokens,
            &finish,
            created,
        );
        let _ = send_json(stream, 200, &response);
    }

    // ── Cleanup (GIL required) ──
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
}

/// The Rust HTTP server, exposed to Python via PyO3.
#[pyclass]
pub struct RustServer {
    host: String,
    port: u16,
    model_name: String,
    tokenizer_path: String,
    max_context_tokens: usize,
    py_model: Py<PyAny>,
    running: Arc<AtomicBool>,
}

#[pymethods]
impl RustServer {
    #[new]
    #[pyo3(signature = (py_model, host, port, model_name, tokenizer_path, max_context_tokens))]
    fn new(
        py_model: PyObject,
        host: String,
        port: u16,
        model_name: String,
        tokenizer_path: String,
        max_context_tokens: usize,
    ) -> Self {
        Self {
            host,
            port,
            model_name,
            tokenizer_path,
            max_context_tokens,
            py_model: py_model.into(),
            running: Arc::new(AtomicBool::new(false)),
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
        let running = self.running.clone();

        // Release GIL — server loop runs without it.
        // GIL is reacquired inside handle_request only for Python prefill/cleanup.
        py.allow_threads(move || {
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

            let state = ServerState {
                py_model,
                model_name,
                tokenizer_path,
                max_context_tokens,
                running: running.clone(),
            };

            while running.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _addr)) => {
                        // Set blocking for the actual request handling
                        stream.set_nonblocking(false).ok();
                        // Set read timeout to prevent hanging on malformed requests
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                            .ok();
                        handle_request(stream, &state);
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

            log::info!("Rust HTTP server stopped");
        });

        Ok(())
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
