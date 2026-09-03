//! Krasis model manager with localhost-only default and explicit LAN access.
//!
//! The long-running control plane, HTTP/API surface, GPU/process discovery,
//! lifecycle coordination, operation state, and browser UI are all Rust. The
//! manager deliberately delegates model capability and budget validation to a
//! short-lived invocation of the existing launcher so there is one source of
//! truth for validated model/runtime combinations.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(windows)]
use std::os::windows::process::CommandExt;

const DEFAULT_MANAGER_PORT: u16 = 8090;
const MAX_REQUEST_BYTES: usize = 256 * 1024;
const MAX_OPERATION_LOG_LINES: usize = 500;
const VALIDATION_TIMEOUT: Duration = Duration::from_secs(180);
const STOP_TIMEOUT: Duration = Duration::from_secs(180);
static MANAGER_SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn manager_signal_handler(_signal: libc::c_int) {
    MANAGER_SHUTDOWN.store(true, Ordering::SeqCst);
}

#[cfg(unix)]
struct ShutdownHandler {
    previous_interrupt: libc::sighandler_t,
    previous_terminate: libc::sighandler_t,
}

#[cfg(unix)]
impl Drop for ShutdownHandler {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.previous_interrupt);
            libc::signal(libc::SIGTERM, self.previous_terminate);
        }
    }
}

#[cfg(unix)]
fn install_shutdown_handler() -> Result<ShutdownHandler, String> {
    let handler = manager_signal_handler as libc::sighandler_t;
    let previous_interrupt = unsafe { libc::signal(libc::SIGINT, handler) };
    if previous_interrupt == libc::SIG_ERR {
        return Err(format!(
            "cannot install Manager SIGINT handler: {}",
            std::io::Error::last_os_error()
        ));
    }
    let previous_terminate = unsafe { libc::signal(libc::SIGTERM, handler) };
    if previous_terminate == libc::SIG_ERR {
        unsafe {
            libc::signal(libc::SIGINT, previous_interrupt);
        }
        return Err(format!(
            "cannot install Manager SIGTERM handler: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(ShutdownHandler {
        previous_interrupt,
        previous_terminate,
    })
}

#[cfg(windows)]
unsafe extern "system" fn manager_console_handler(control: u32) -> i32 {
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_C_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT,
    };
    if matches!(
        control,
        CTRL_C_EVENT
            | CTRL_BREAK_EVENT
            | CTRL_CLOSE_EVENT
            | CTRL_LOGOFF_EVENT
            | CTRL_SHUTDOWN_EVENT
    ) {
        MANAGER_SHUTDOWN.store(true, Ordering::SeqCst);
        1
    } else {
        0
    }
}

#[cfg(windows)]
struct ShutdownHandler;

#[cfg(windows)]
impl Drop for ShutdownHandler {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
        unsafe {
            SetConsoleCtrlHandler(Some(manager_console_handler), 0);
        }
    }
}

#[cfg(windows)]
fn install_shutdown_handler() -> Result<ShutdownHandler, String> {
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
    let result = unsafe { SetConsoleCtrlHandler(Some(manager_console_handler), 1) };
    if result == 0 {
        Err(format!(
            "cannot install Manager console handler: {}",
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(ShutdownHandler)
    }
}

#[derive(Clone)]
struct ManagerState {
    inner: Arc<ManagerInner>,
}

struct ManagerInner {
    python_executable: PathBuf,
    manager_dir: PathBuf,
    models_dir: PathBuf,
    nvidia_smi: PathBuf,
    token: String,
    port: u16,
    lan: bool,
    mutable: Mutex<MutableState>,
}

#[derive(Default)]
struct MutableState {
    operations: HashMap<String, Operation>,
    gpu_locks: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
struct Operation {
    id: String,
    kind: String,
    status: String,
    message: String,
    gpu_uuids: Vec<String>,
    model_path: Option<String>,
    pid: Option<u32>,
    port: Option<u16>,
    created_at_unix_ms: u128,
    updated_at_unix_ms: u128,
    terminal: bool,
    error: Option<String>,
    logs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct GpuInfo {
    index: u32,
    uuid: String,
    name: String,
    memory_total_mb: u64,
    memory_used_mb: u64,
    memory_free_mb: u64,
    processes: Vec<GpuProcess>,
    operation: Option<OperationSummary>,
}

#[derive(Clone, Debug, Serialize)]
struct GpuProcess {
    pid: u32,
    used_memory_mb: u64,
    is_krasis: bool,
    command: Option<String>,
    model_path: Option<String>,
    port: Option<u16>,
    gpu_uuids: Vec<String>,
    config: BTreeMap<String, String>,
    manager_config: Option<ManagerConfig>,
}

#[derive(Clone, Debug, Serialize)]
struct OperationSummary {
    id: String,
    kind: String,
    status: String,
    message: String,
}

#[derive(Clone, Debug, Serialize)]
struct InstalledModel {
    name: String,
    path: String,
    model_type: String,
    architecture: String,
    has_safetensors: bool,
}

fn normalize_dynamic_hcs_tail_blocks(value: &str) -> Result<String, String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "auto" {
        return Ok(normalized);
    }
    let blocks = normalized
        .parse::<u32>()
        .map_err(|_| "dynamic_hcs_tail_blocks must be auto or an integer in 1..5".to_string())?;
    if !(1..=5).contains(&blocks) {
        return Err("dynamic_hcs_tail_blocks must be auto or an integer in 1..5".to_string());
    }
    Ok(blocks.to_string())
}

fn deserialize_dynamic_hcs_tail_blocks<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TailPolicyInput {
        Text(String),
        Blocks(u32),
    }

    let raw = match TailPolicyInput::deserialize(deserializer)? {
        TailPolicyInput::Text(value) => value,
        TailPolicyInput::Blocks(value) => value.to_string(),
    };
    normalize_dynamic_hcs_tail_blocks(&raw).map_err(serde::de::Error::custom)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct ManagerConfig {
    model_path: String,
    gpu_uuids: Vec<String>,
    host: String,
    port: u16,
    attention_quant: String,
    vision_quant: String,
    hqq_cache_profile: String,
    hqq_group_size: u32,
    hqq_auto_budget_pct: f64,
    hqq_sidecar_manifest: String,
    kv_dtype: String,
    kv_cache_mb: u64,
    max_context_tokens: u64,
    vram_safety_margin_mb: u64,
    layer_group_size: u32,
    expert_group_size: u32,
    gpu_expert_int4_calib: String,
    shared_expert_quant: String,
    dense_mlp_quant: String,
    lm_head_quant: String,
    krasis_threads: u32,
    hcs: bool,
    dynamic_hcs: bool,
    #[serde(deserialize_with = "deserialize_dynamic_hcs_tail_blocks")]
    dynamic_hcs_tail_blocks: String,
    hcs_host_cache_mode: String,
    multi_gpu_mode: String,
    dynamic_peer: bool,
    adaptive_cold_mass_pruning: String,
    prefix_cache: bool,
    prefix_cache_ram_fraction: f64,
    enable_thinking: bool,
    gpu_prefill_threshold: u32,
    pp_partition: String,
    heatmap_path: String,
    gguf_path: String,
    expert_compression: bool,
    expert_compression_sidecar: String,
    expert_compression_pipeline: String,
    dspark_mode: String,
    ssh_tunnel: String,
    ssh_key_path: String,
    force_rebuild_cache: bool,
    force_rebuild_hqq_cache: bool,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            gpu_uuids: Vec::new(),
            host: "0.0.0.0".to_string(),
            port: 8012,
            attention_quant: "hqq6".to_string(),
            vision_quant: "int4".to_string(),
            hqq_cache_profile: "baseline".to_string(),
            hqq_group_size: 128,
            hqq_auto_budget_pct: 0.0,
            hqq_sidecar_manifest: String::new(),
            kv_dtype: "k6v6".to_string(),
            kv_cache_mb: 1000,
            max_context_tokens: 0,
            vram_safety_margin_mb: 600,
            layer_group_size: 2,
            expert_group_size: 128,
            gpu_expert_int4_calib: "amax".to_string(),
            shared_expert_quant: "int8".to_string(),
            dense_mlp_quant: "int8".to_string(),
            lm_head_quant: "int8".to_string(),
            krasis_threads: 40,
            hcs: true,
            dynamic_hcs: true,
            dynamic_hcs_tail_blocks: "auto".to_string(),
            hcs_host_cache_mode: "source".to_string(),
            multi_gpu_mode: "auto".to_string(),
            dynamic_peer: false,
            adaptive_cold_mass_pruning: "off".to_string(),
            prefix_cache: true,
            prefix_cache_ram_fraction: 0.25,
            enable_thinking: true,
            gpu_prefill_threshold: 300,
            pp_partition: String::new(),
            heatmap_path: String::new(),
            gguf_path: String::new(),
            expert_compression: false,
            expert_compression_sidecar: String::new(),
            expert_compression_pipeline: "grouped".to_string(),
            dspark_mode: "off".to_string(),
            ssh_tunnel: String::new(),
            ssh_key_path: String::new(),
            force_rebuild_cache: false,
            force_rebuild_hqq_cache: false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DescribeRequest {
    model_path: String,
    gpu_uuids: Vec<String>,
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    target: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
    local_addr: Option<SocketAddr>,
    peer_addr: Option<SocketAddr>,
}

#[derive(Debug)]
struct ApiError {
    status: u16,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            code: "bad_request",
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: 409,
            code: "conflict",
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            code: "not_found",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            code: "internal_error",
            message: message.into(),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (python_executable, port=DEFAULT_MANAGER_PORT, open_browser=true, lan=false))]
pub fn run_manager(
    py: Python<'_>,
    python_executable: String,
    port: u16,
    open_browser: bool,
    lan: bool,
) -> PyResult<()> {
    py.allow_threads(move || run_manager_inner(&python_executable, port, open_browser, lan))
        .map_err(PyRuntimeError::new_err)
}

fn run_manager_inner(
    python_executable: &str,
    port: u16,
    open_browser: bool,
    lan: bool,
) -> Result<(), String> {
    if port == 0 {
        return Err("manager port must be between 1 and 65535".to_string());
    }
    let krasis_home = krasis_home()?;
    let manager_dir = krasis_home.join("manager");
    let models_dir = krasis_home.join("models");
    fs::create_dir_all(&manager_dir)
        .map_err(|error| format!("cannot create {}: {error}", manager_dir.display()))?;
    fs::create_dir_all(&models_dir)
        .map_err(|error| format!("cannot create {}: {error}", models_dir.display()))?;
    let token = load_or_create_token(&manager_dir)?;
    let nvidia_smi = find_nvidia_smi()?;
    let bind_host = if lan { "0.0.0.0" } else { "127.0.0.1" };
    let listener = TcpListener::bind((bind_host, port))
        .map_err(|error| format!("cannot bind Krasis Manager to {bind_host}:{port}: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("cannot configure Manager listener: {error}"))?;
    MANAGER_SHUTDOWN.store(false, Ordering::SeqCst);
    let _shutdown_handler = install_shutdown_handler()?;
    let state = ManagerState {
        inner: Arc::new(ManagerInner {
            python_executable: PathBuf::from(python_executable),
            manager_dir,
            models_dir,
            nvidia_smi,
            token,
            port,
            lan,
            mutable: Mutex::new(MutableState::default()),
        }),
    };

    let url = format!("http://127.0.0.1:{port}/");
    println!("Krasis Manager is running at {url}");
    if lan {
        println!(
            "LAN access enabled on 0.0.0.0:{port}; connect using this machine's LAN IPv4 address."
        );
        println!("Every LAN API request requires the owner token.");
        println!("The host firewall must permit this port; Krasis does not change firewall rules.");
    }
    println!(
        "Local API token: {}",
        state.inner.manager_dir.join("token").display()
    );
    println!("Press Ctrl-C to stop the manager. Running models are not stopped.");
    if open_browser {
        open_browser_url(&url);
    }

    while !MANAGER_SHUTDOWN.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let request_state = state.clone();
                thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, request_state) {
                        eprintln!("Krasis Manager request error: {error}");
                    }
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                eprintln!("Krasis Manager accept error: {error}");
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    println!("Krasis Manager stopped. Running models were left untouched.");
    Ok(())
}

fn krasis_home() -> Result<PathBuf, String> {
    if let Some(value) = env::var_os("KRASIS_HOME") {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    for key in ["HOME", "USERPROFILE"] {
        if let Some(value) = env::var_os(key) {
            if !value.is_empty() {
                return Ok(PathBuf::from(value).join(".krasis"));
            }
        }
    }
    Err("cannot determine Krasis home; set KRASIS_HOME".to_string())
}

fn load_or_create_token(manager_dir: &Path) -> Result<String, String> {
    let path = manager_dir.join("token");
    if path.is_file() {
        let token = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?
            .trim()
            .to_string();
        if token.len() == 64 && token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(token);
        }
        return Err(format!(
            "{} is not a valid Krasis Manager token; remove it and restart the manager",
            path.display()
        ));
    }
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    let token = hex_bytes(&bytes);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    writeln!(file, "{token}")
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(token)
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn find_nvidia_smi() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("KRASIS_NVIDIA_SMI") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "KRASIS_NVIDIA_SMI does not name a file: {}",
            path.display()
        ));
    }
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            candidates.push(directory.join(if cfg!(windows) {
                "nvidia-smi.exe"
            } else {
                "nvidia-smi"
            }));
        }
    }
    candidates.push(PathBuf::from("/usr/lib/wsl/lib/nvidia-smi"));
    #[cfg(windows)]
    {
        for key in ["ProgramW6432", "ProgramFiles"] {
            if let Some(root) = env::var_os(key) {
                candidates.push(
                    PathBuf::from(root)
                        .join("NVIDIA Corporation")
                        .join("NVSMI")
                        .join("nvidia-smi.exe"),
                );
            }
        }
    }
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            "nvidia-smi was not found; install an NVIDIA driver or set KRASIS_NVIDIA_SMI"
                .to_string()
        })
}

fn open_browser_url(url: &str) {
    #[cfg(target_os = "windows")]
    let result = Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let result = Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = Command::new("xdg-open").arg(url).spawn();
    if let Err(error) = result {
        eprintln!("Could not open a browser automatically: {error}");
    }
}

fn handle_connection(mut stream: TcpStream, state: ManagerState) -> Result<(), String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|error| error.to_string())?;
    let request = match read_http_request(&mut stream) {
        Ok(request) => request,
        Err(error) => {
            write_api_error(&mut stream, &error)?;
            return Ok(());
        }
    };
    let response = route_request(&request, &state);
    match response {
        Ok((status, content_type, body)) => {
            write_http_response(&mut stream, status, content_type, body.as_bytes())?
        }
        Err(error) => write_api_error(&mut stream, &error)?,
    }
    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, ApiError> {
    let local_addr = stream.local_addr().ok();
    let peer_addr = stream.peer_addr().ok();
    let mut data = Vec::new();
    let mut buffer = [0u8; 8192];
    let header_end = loop {
        let count = stream
            .read(&mut buffer)
            .map_err(|error| ApiError::bad_request(format!("cannot read request: {error}")))?;
        if count == 0 {
            return Err(ApiError::bad_request(
                "connection closed before request completed",
            ));
        }
        data.extend_from_slice(&buffer[..count]);
        if data.len() > MAX_REQUEST_BYTES {
            return Err(ApiError {
                status: 413,
                code: "request_too_large",
                message: "request exceeds 256 KiB".to_string(),
            });
        }
        if let Some(position) = find_bytes(&data, b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers_text = std::str::from_utf8(&data[..header_end - 4])
        .map_err(|_| ApiError::bad_request("request headers are not UTF-8"))?;
    let mut lines = headers_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| ApiError::bad_request("missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| ApiError::bad_request("missing method"))?
        .to_string();
    let target = request_parts
        .next()
        .ok_or_else(|| ApiError::bad_request("missing target"))?
        .to_string();
    if request_parts.next() != Some("HTTP/1.1") || request_parts.next().is_some() {
        return Err(ApiError::bad_request("only HTTP/1.1 is supported"));
    }
    let mut headers = HashMap::new();
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| ApiError::bad_request("malformed header"))?;
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
    }
    let content_length = match headers.get("content-length") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| ApiError::bad_request("invalid Content-Length"))?,
        None => 0,
    };
    if content_length > MAX_REQUEST_BYTES {
        return Err(ApiError {
            status: 413,
            code: "request_too_large",
            message: "request exceeds 256 KiB".to_string(),
        });
    }
    while data.len() - header_end < content_length {
        let count = stream
            .read(&mut buffer)
            .map_err(|error| ApiError::bad_request(format!("cannot read body: {error}")))?;
        if count == 0 {
            return Err(ApiError::bad_request("request body ended early"));
        }
        data.extend_from_slice(&buffer[..count]);
        if data.len() > header_end + MAX_REQUEST_BYTES {
            return Err(ApiError {
                status: 413,
                code: "request_too_large",
                message: "request exceeds 256 KiB".to_string(),
            });
        }
    }
    Ok(HttpRequest {
        method,
        target,
        headers,
        body: data[header_end..header_end + content_length].to_vec(),
        local_addr,
        peer_addr,
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn route_request(
    request: &HttpRequest,
    state: &ManagerState,
) -> Result<(u16, &'static str, String), ApiError> {
    validate_host(request, state)?;
    let path = request.target.split('?').next().unwrap_or(&request.target);
    if request.method == "GET" && path == "/" {
        let is_local_client = request
            .peer_addr
            .is_some_and(|address| address.ip().is_loopback());
        let page_token = if state.inner.lan && !is_local_client {
            ""
        } else {
            &state.inner.token
        };
        let request_host = request
            .headers
            .get("host")
            .map(String::as_str)
            .unwrap_or("");
        let html = include_str!("manager.html")
            .replace("__KRASIS_MANAGER_TOKEN__", page_token)
            .replace("__KRASIS_MANAGER_BASE__", &format!("http://{request_host}"))
            .replace(
                "__KRASIS_MANAGER_NETWORK_MODE__",
                if state.inner.lan { "true" } else { "false" },
            )
            .replace(
                "__KRASIS_MANAGER_NETWORK_LABEL__",
                if state.inner.lan { "LAN enabled" } else { "localhost only" },
            )
            .replace(
                "__KRASIS_MANAGER_NETWORK_DESCRIPTION__",
                if state.inner.lan {
                    "LAN mode requires the owner token for every API request and accepts only the exact destination interface address."
                } else {
                    "Manager binds only to 127.0.0.1 and rejects non-local Host and Origin values."
                },
            );
        return Ok((200, "text/html; charset=utf-8", html));
    }
    if request.method == "GET" && path == "/favicon.ico" {
        return Ok((204, "image/x-icon", String::new()));
    }
    if state.inner.lan && path.starts_with("/api/") {
        validate_token(request, state)?;
    }
    if request.method == "GET" && path == "/api/v1/status" {
        let mutable = state.inner.mutable.lock().unwrap();
        return json_response(
            200,
            json!({
                "manager": "krasis",
                "api_version": 1,
                "bind": format!("{}:{}", if state.inner.lan { "0.0.0.0" } else { "127.0.0.1" }, state.inner.port),
                "network_mode": if state.inner.lan { "lan" } else { "localhost" },
                "api_authentication": if state.inner.lan { "all" } else { "mutations" },
                "models_dir": state.inner.models_dir,
                "operations": mutable.operations.values().cloned().collect::<Vec<_>>(),
            }),
        );
    }
    if request.method == "GET" && path == "/api/v1/gpus" {
        let gpus = discover_gpus(state).map_err(ApiError::internal)?;
        return json_response(200, json!({"gpus": gpus}));
    }
    if request.method == "GET" && path == "/api/v1/models" {
        let models = scan_models(&state.inner.models_dir).map_err(ApiError::internal)?;
        return json_response(200, json!({"models": models}));
    }
    if request.method == "GET" && path.starts_with("/api/v1/operations/") {
        let id = path.trim_start_matches("/api/v1/operations/");
        if id.is_empty() || id.contains('/') {
            return Err(ApiError::not_found("operation not found"));
        }
        let mutable = state.inner.mutable.lock().unwrap();
        let operation = mutable
            .operations
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError::not_found(format!("operation {id} not found")))?;
        return json_response(200, json!({"operation": operation}));
    }

    if request.method == "POST" {
        validate_mutation_request(request, state)?;
    }
    if request.method == "POST" && path == "/api/v1/configs/describe" {
        let payload: DescribeRequest = parse_json(&request.body)?;
        let value = describe_config(state, &payload)?;
        return json_response(200, value);
    }
    if request.method == "POST" && path == "/api/v1/configs/validate" {
        let config: ManagerConfig = parse_json(&request.body)?;
        let result = validate_config_request(state, &config, None)?;
        return json_response(200, result);
    }
    if request.method == "POST" && path.starts_with("/api/v1/gpus/") && path.ends_with("/apply") {
        let uuid = route_gpu_uuid(path, "/apply")?;
        let config: ManagerConfig = parse_json(&request.body)?;
        let operation = start_apply(state.clone(), uuid, config)?;
        return json_response(202, json!({"operation": operation}));
    }
    if request.method == "POST" && path.starts_with("/api/v1/gpus/") && path.ends_with("/stop") {
        let uuid = route_gpu_uuid(path, "/stop")?;
        if !request.body.is_empty() && request.body != b"{}" {
            let _: BTreeMap<String, Value> = parse_json(&request.body)?;
        }
        let operation = start_stop(state.clone(), uuid)?;
        return json_response(202, json!({"operation": operation}));
    }
    Err(ApiError::not_found(format!(
        "no route for {} {}",
        request.method, path
    )))
}

fn json_response(status: u16, value: Value) -> Result<(u16, &'static str, String), ApiError> {
    serde_json::to_string_pretty(&value)
        .map(|body| (status, "application/json; charset=utf-8", body))
        .map_err(|error| ApiError::internal(format!("cannot serialize response: {error}")))
}

fn parse_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body)
        .map_err(|error| ApiError::bad_request(format!("invalid JSON: {error}")))
}

fn validate_host(request: &HttpRequest, state: &ManagerState) -> Result<(), ApiError> {
    let host = request
        .headers
        .get("host")
        .ok_or_else(|| ApiError::bad_request("Host header is required"))?;
    if !state.inner.lan {
        let allowed = [
            format!("127.0.0.1:{}", state.inner.port),
            format!("localhost:{}", state.inner.port),
            format!("[::1]:{}", state.inner.port),
        ];
        if allowed
            .iter()
            .any(|candidate| host.eq_ignore_ascii_case(candidate))
        {
            return Ok(());
        }
        return Err(ApiError {
            status: 403,
            code: "localhost_only",
            message: "Krasis Manager accepts only localhost Host headers".to_string(),
        });
    }

    let (host_ip, host_port) = parse_ip_authority(host).ok_or_else(|| ApiError {
        status: 403,
        code: "invalid_lan_host",
        message: "LAN Manager Host must be the exact destination IP address and port".to_string(),
    })?;
    let destination = request.local_addr.ok_or_else(|| {
        ApiError::internal("cannot determine the Manager connection's destination address")
    })?;
    if host_port != state.inner.port || host_ip != destination.ip() {
        return Err(ApiError {
            status: 403,
            code: "invalid_lan_host",
            message: "LAN Manager Host does not match the destination interface and port"
                .to_string(),
        });
    }
    Ok(())
}

fn parse_ip_authority(authority: &str) -> Option<(IpAddr, u16)> {
    if authority.starts_with('[') {
        let close = authority.find(']')?;
        let ip = authority.get(1..close)?.parse().ok()?;
        let port = authority
            .get(close + 1..)?
            .strip_prefix(':')?
            .parse()
            .ok()?;
        return Some((ip, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    if host.is_empty() || host.contains(':') {
        return None;
    }
    Some((host.parse().ok()?, port.parse().ok()?))
}

fn validate_token(request: &HttpRequest, state: &ManagerState) -> Result<(), ApiError> {
    let token = request
        .headers
        .get("x-krasis-manager-token")
        .ok_or_else(|| ApiError {
            status: 401,
            code: "token_required",
            message: "X-Krasis-Manager-Token is required".to_string(),
        })?;
    if token != &state.inner.token {
        return Err(ApiError {
            status: 403,
            code: "invalid_token",
            message: "invalid Krasis Manager token".to_string(),
        });
    }
    Ok(())
}

fn validate_mutation_request(request: &HttpRequest, state: &ManagerState) -> Result<(), ApiError> {
    validate_token(request, state)?;
    if let Some(origin) = request.headers.get("origin") {
        let origin_allowed = if state.inner.lan {
            request
                .headers
                .get("host")
                .is_some_and(|host| origin.eq_ignore_ascii_case(&format!("http://{host}")))
        } else {
            [
                format!("http://127.0.0.1:{}", state.inner.port),
                format!("http://localhost:{}", state.inner.port),
            ]
            .iter()
            .any(|candidate| origin.eq_ignore_ascii_case(candidate))
        };
        if !origin_allowed {
            return Err(ApiError {
                status: 403,
                code: "invalid_origin",
                message: "state changes are accepted only from the Manager page's exact origin"
                    .to_string(),
            });
        }
    }
    Ok(())
}

fn route_gpu_uuid(path: &str, suffix: &str) -> Result<String, ApiError> {
    let inner = path
        .strip_prefix("/api/v1/gpus/")
        .and_then(|value| value.strip_suffix(suffix))
        .ok_or_else(|| ApiError::not_found("GPU route not found"))?;
    let uuid = inner.trim_end_matches('/');
    if uuid.is_empty() || uuid.contains('/') {
        return Err(ApiError::bad_request("invalid GPU UUID in route"));
    }
    percent_decode(uuid)
}

fn percent_decode(value: &str) -> Result<String, ApiError> {
    let bytes = value.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err(ApiError::bad_request("invalid percent encoding"));
            }
            let high = hex_value(bytes[index + 1])
                .ok_or_else(|| ApiError::bad_request("invalid percent encoding"))?;
            let low = hex_value(bytes[index + 2])
                .ok_or_else(|| ApiError::bad_request("invalid percent encoding"))?;
            result.push((high << 4) | low);
            index += 3;
        } else {
            result.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(result).map_err(|_| ApiError::bad_request("route is not UTF-8"))
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn write_api_error(stream: &mut TcpStream, error: &ApiError) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(&json!({
        "error": {"code": error.code, "message": error.message}
    }))
    .map_err(|serialization_error| serialization_error.to_string())?;
    write_http_response(
        stream,
        error.status,
        "application/json; charset=utf-8",
        &body,
    )
}

fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "Response",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\nContent-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|_| stream.write_all(body))
        .map_err(|error| format!("cannot write response: {error}"))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn operation_id() -> String {
    let mut bytes = [0u8; 8];
    OsRng.fill_bytes(&mut bytes);
    format!("op-{}-{}", now_unix_ms(), hex_bytes(&bytes))
}

fn update_operation(
    state: &ManagerState,
    id: &str,
    status: &str,
    message: impl Into<String>,
    terminal: bool,
) {
    let mut mutable = state.inner.mutable.lock().unwrap();
    if let Some(operation) = mutable.operations.get_mut(id) {
        operation.status = status.to_string();
        operation.message = message.into();
        operation.updated_at_unix_ms = now_unix_ms();
        operation.terminal = terminal;
    }
}

fn fail_operation(state: &ManagerState, id: &str, message: impl Into<String>) {
    let message = message.into();
    let gpu_uuids = {
        let mut mutable = state.inner.mutable.lock().unwrap();
        let Some(operation) = mutable.operations.get_mut(id) else {
            return;
        };
        operation.status = "failed".to_string();
        operation.message = message.clone();
        operation.error = Some(message.clone());
        operation.updated_at_unix_ms = now_unix_ms();
        operation.terminal = true;
        operation.gpu_uuids.clone()
    };
    release_gpu_locks(state, id, &gpu_uuids);
}

fn set_operation_pid(state: &ManagerState, id: &str, pid: u32) {
    let mut mutable = state.inner.mutable.lock().unwrap();
    if let Some(operation) = mutable.operations.get_mut(id) {
        operation.pid = Some(pid);
        operation.updated_at_unix_ms = now_unix_ms();
    }
}

fn append_operation_log(state: &ManagerState, id: &str, line: impl Into<String>) {
    let line = strip_ansi(&line.into());
    if line.trim().is_empty() {
        return;
    }
    let mut mutable = state.inner.mutable.lock().unwrap();
    if let Some(operation) = mutable.operations.get_mut(id) {
        operation.logs.push(line);
        if operation.logs.len() > MAX_OPERATION_LOG_LINES {
            let remove = operation.logs.len() - MAX_OPERATION_LOG_LINES;
            operation.logs.drain(..remove);
        }
        operation.updated_at_unix_ms = now_unix_ms();
    }
}

fn strip_ansi(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

fn release_gpu_locks(state: &ManagerState, id: &str, gpu_uuids: &[String]) {
    let mut mutable = state.inner.mutable.lock().unwrap();
    for uuid in gpu_uuids {
        if mutable.gpu_locks.get(uuid).is_some_and(|owner| owner == id) {
            mutable.gpu_locks.remove(uuid);
        }
    }
}

fn discover_gpus(state: &ManagerState) -> Result<Vec<GpuInfo>, String> {
    let output = Command::new(&state.inner.nvidia_smi)
        .args([
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("nvidia-smi GPU query failed to start: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi GPU query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut gpus = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 6 {
            return Err(format!("unexpected nvidia-smi GPU row: {line}"));
        }
        gpus.push(GpuInfo {
            index: fields[0]
                .parse()
                .map_err(|_| format!("invalid GPU index in nvidia-smi row: {line}"))?,
            uuid: fields[1].to_string(),
            name: fields[2].to_string(),
            memory_total_mb: fields[3]
                .parse()
                .map_err(|_| format!("invalid total memory in nvidia-smi row: {line}"))?,
            memory_used_mb: fields[4]
                .parse()
                .map_err(|_| format!("invalid used memory in nvidia-smi row: {line}"))?,
            memory_free_mb: fields[5]
                .parse()
                .map_err(|_| format!("invalid free memory in nvidia-smi row: {line}"))?,
            processes: Vec::new(),
            operation: None,
        });
    }

    let compute_rows = discover_compute_processes(state)?;
    let mut pid_gpus: HashMap<u32, BTreeSet<String>> = HashMap::new();
    let mut pid_memory: HashMap<(u32, String), u64> = HashMap::new();
    for (pid, uuid, memory_mb) in compute_rows {
        pid_gpus.entry(pid).or_default().insert(uuid.clone());
        pid_memory.insert((pid, uuid), memory_mb);
    }
    let mut process_details = HashMap::new();
    for (pid, gpu_set) in &pid_gpus {
        let command = process_command_line(*pid);
        let (is_krasis, config) = match &command {
            Some(command) if is_krasis_server_command(command) => {
                let config = extract_config_path(command)
                    .and_then(|path| read_krasis_config(&path).ok())
                    .unwrap_or_default();
                (true, config)
            }
            _ => (false, BTreeMap::new()),
        };
        let model_path = config.get("MODEL_PATH").cloned();
        let port = config.get("CFG_PORT").and_then(|value| value.parse().ok());
        process_details.insert(
            *pid,
            (
                command,
                is_krasis,
                config,
                model_path,
                port,
                gpu_set.iter().cloned().collect::<Vec<_>>(),
            ),
        );
    }
    for gpu in &mut gpus {
        for (pid, gpu_set) in &pid_gpus {
            if !gpu_set.contains(&gpu.uuid) {
                continue;
            }
            let (command, is_krasis, config, model_path, port, gpu_uuids) =
                process_details.get(pid).cloned().unwrap_or_default();
            let manager_config = is_krasis
                .then(|| manager_config_from_saved(&config, &gpu_uuids))
                .flatten();
            gpu.processes.push(GpuProcess {
                pid: *pid,
                used_memory_mb: *pid_memory.get(&(*pid, gpu.uuid.clone())).unwrap_or(&0),
                is_krasis,
                command,
                model_path,
                port,
                gpu_uuids,
                config,
                manager_config,
            });
        }
    }
    let mutable = state.inner.mutable.lock().unwrap();
    for gpu in &mut gpus {
        if let Some(operation_id) = mutable.gpu_locks.get(&gpu.uuid) {
            if let Some(operation) = mutable.operations.get(operation_id) {
                gpu.operation = Some(OperationSummary {
                    id: operation.id.clone(),
                    kind: operation.kind.clone(),
                    status: operation.status.clone(),
                    message: operation.message.clone(),
                });
            }
        }
    }
    Ok(gpus)
}

fn discover_compute_processes(state: &ManagerState) -> Result<Vec<(u32, String, u64)>, String> {
    let output = Command::new(&state.inner.nvidia_smi)
        .args([
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("nvidia-smi process query failed to start: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No running processes found") {
            return Ok(Vec::new());
        }
        return Err(format!(
            "nvidia-smi process query failed: {}",
            stderr.trim()
        ));
    }
    let mut rows = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.trim().is_empty() || line.contains("No running processes found") {
            continue;
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 3 {
            return Err(format!("unexpected nvidia-smi process row: {line}"));
        }
        let pid = fields[0]
            .parse()
            .map_err(|_| format!("invalid PID in nvidia-smi row: {line}"))?;
        let memory = fields[2].parse().unwrap_or(0);
        rows.push((pid, fields[1].to_string(), memory));
    }
    Ok(rows)
}

#[cfg(unix)]
fn process_command_line(pid: u32) -> Option<String> {
    let bytes = fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let parts = bytes
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).to_string())
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| join_command_parts(&parts))
}

#[cfg(windows)]
fn process_command_line(pid: u32) -> Option<String> {
    let script = format!("(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine");
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let command = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!command.is_empty()).then_some(command)
}

fn join_command_parts(parts: &[String]) -> String {
    parts
        .iter()
        .map(|part| {
            if part.chars().any(char::is_whitespace) {
                format!("\"{}\"", part.replace('"', "\\\""))
            } else {
                part.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_command_line(command: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for ch in command.chars() {
        if matches!(ch, '\'' | '"') {
            if quote == Some(ch) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(ch);
            } else {
                current.push(ch);
            }
            continue;
        }
        if ch.is_whitespace() && quote.is_none() {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn is_krasis_server_command(command: &str) -> bool {
    let parts = split_command_line(command);
    parts
        .windows(2)
        .any(|window| window[0] == "-m" && window[1] == "krasis.server")
}

fn extract_config_path(command: &str) -> Option<PathBuf> {
    let parts = split_command_line(command);
    parts
        .windows(2)
        .find(|window| window[0] == "--config")
        .map(|window| PathBuf::from(&window[1]))
}

fn read_krasis_config(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut config = BTreeMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value
            .trim()
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| {
                value
                    .trim()
                    .strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
            })
            .unwrap_or(value.trim());
        config.insert(key.trim().to_string(), value.to_string());
    }
    Ok(config)
}

fn manager_config_from_saved(
    saved: &BTreeMap<String, String>,
    gpu_uuids: &[String],
) -> Option<ManagerConfig> {
    let mut config = ManagerConfig::default();
    config.model_path = saved.get("MODEL_PATH")?.clone();
    config.gpu_uuids = gpu_uuids.to_vec();
    macro_rules! string_value {
        ($field:ident, $key:literal) => {
            if let Some(value) = saved.get($key) {
                config.$field = value.clone();
            }
        };
    }
    macro_rules! number_value {
        ($field:ident, $key:literal) => {
            if let Some(value) = saved.get($key).and_then(|value| value.parse().ok()) {
                config.$field = value;
            }
        };
    }
    macro_rules! bool_value {
        ($field:ident, $key:literal) => {
            if let Some(value) = saved.get($key) {
                config.$field = value == "1" || value.eq_ignore_ascii_case("true");
            }
        };
    }
    string_value!(host, "CFG_HOST");
    number_value!(port, "CFG_PORT");
    string_value!(attention_quant, "CFG_ATTENTION_QUANT");
    string_value!(vision_quant, "CFG_VISION_QUANT");
    string_value!(hqq_cache_profile, "CFG_HQQ_CACHE_PROFILE");
    number_value!(hqq_group_size, "CFG_HQQ_GROUP_SIZE");
    number_value!(hqq_auto_budget_pct, "CFG_HQQ_AUTO_BUDGET_PCT");
    string_value!(hqq_sidecar_manifest, "CFG_HQQ_SIDECAR_MANIFEST");
    string_value!(kv_dtype, "CFG_KV_DTYPE");
    number_value!(kv_cache_mb, "CFG_KV_CACHE_MB");
    number_value!(max_context_tokens, "CFG_MAX_CONTEXT_TOKENS");
    number_value!(vram_safety_margin_mb, "CFG_VRAM_SAFETY_MARGIN");
    number_value!(layer_group_size, "CFG_LAYER_GROUP_SIZE");
    number_value!(expert_group_size, "CFG_EXPERT_GROUP_SIZE");
    string_value!(gpu_expert_int4_calib, "CFG_GPU_EXPERT_INT4_CALIB");
    string_value!(shared_expert_quant, "CFG_SHARED_EXPERT_QUANT");
    string_value!(dense_mlp_quant, "CFG_DENSE_MLP_QUANT");
    string_value!(lm_head_quant, "CFG_LM_HEAD_QUANT");
    number_value!(krasis_threads, "CFG_KRASIS_THREADS");
    bool_value!(hcs, "CFG_HCS");
    bool_value!(dynamic_hcs, "CFG_DYNAMIC_HCS");
    if let Some(value) = saved.get("CFG_DYNAMIC_HCS_TAIL_BLOCKS") {
        config.dynamic_hcs_tail_blocks = normalize_dynamic_hcs_tail_blocks(value).ok()?;
    }
    string_value!(hcs_host_cache_mode, "CFG_HCS_HOST_CACHE_MODE");
    string_value!(multi_gpu_mode, "CFG_MULTI_GPU_MODE");
    bool_value!(dynamic_peer, "CFG_DYNAMIC_PEER");
    string_value!(adaptive_cold_mass_pruning, "CFG_ADAPTIVE_COLD_MASS_PRUNING");
    bool_value!(prefix_cache, "CFG_PREFIX_CACHE");
    number_value!(prefix_cache_ram_fraction, "CFG_PREFIX_CACHE_RAM_FRACTION");
    bool_value!(enable_thinking, "CFG_ENABLE_THINKING");
    number_value!(gpu_prefill_threshold, "CFG_GPU_PREFILL_THRESHOLD");
    string_value!(pp_partition, "CFG_PP_PARTITION");
    string_value!(heatmap_path, "CFG_HEATMAP_PATH");
    string_value!(gguf_path, "CFG_GGUF_PATH");
    bool_value!(expert_compression, "CFG_EXPERT_COMPRESSION");
    string_value!(expert_compression_sidecar, "CFG_EXPERT_COMPRESSION_SIDECAR");
    string_value!(
        expert_compression_pipeline,
        "CFG_EXPERT_COMPRESSION_PIPELINE"
    );
    string_value!(dspark_mode, "CFG_DSPARK_MODE");
    string_value!(ssh_tunnel, "CFG_SSH_TUNNEL");
    string_value!(ssh_key_path, "CFG_SSH_KEY_PATH");
    bool_value!(force_rebuild_cache, "CFG_FORCE_REBUILD_CACHE");
    bool_value!(force_rebuild_hqq_cache, "CFG_FORCE_REBUILD_HQQ_CACHE");
    Some(config)
}

fn scan_models(models_dir: &Path) -> Result<Vec<InstalledModel>, String> {
    let root = models_dir
        .canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", models_dir.display()))?;
    let mut pending = vec![root.clone()];
    let mut models = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        let mut has_config = false;
        let mut has_safetensors = false;
        let mut children = Vec::new();
        for entry in entries.flatten() {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                children.push(entry.path());
            } else if entry.file_name() == "config.json" {
                has_config = true;
            } else if entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
            {
                has_safetensors = true;
            }
        }
        if has_config {
            let config_path = directory.join("config.json");
            let value: Value = fs::read_to_string(&config_path)
                .ok()
                .and_then(|content| serde_json::from_str(&content).ok())
                .unwrap_or(Value::Null);
            let relative = directory.strip_prefix(&root).unwrap_or(&directory);
            let name = if relative.as_os_str().is_empty() {
                directory
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            } else {
                relative.to_string_lossy().replace('\\', "/")
            };
            let model_type = value
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let architecture = value
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            models.push(InstalledModel {
                name,
                path: directory.to_string_lossy().to_string(),
                model_type,
                architecture,
                has_safetensors,
            });
        }
        pending.extend(children);
    }
    models.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase()));
    Ok(models)
}

fn validate_model_path(state: &ManagerState, model_path: &str) -> Result<PathBuf, ApiError> {
    reject_config_text("model_path", model_path)?;
    let root =
        state.inner.models_dir.canonicalize().map_err(|error| {
            ApiError::internal(format!("cannot resolve models directory: {error}"))
        })?;
    let model = PathBuf::from(model_path)
        .canonicalize()
        .map_err(|error| ApiError::bad_request(format!("cannot resolve model_path: {error}")))?;
    if !model.starts_with(&root) || model == root {
        return Err(ApiError::bad_request(format!(
            "model_path must be an installed model beneath {}",
            root.display()
        )));
    }
    if !model.join("config.json").is_file() {
        return Err(ApiError::bad_request("model_path has no config.json"));
    }
    Ok(model)
}

fn validate_config_syntax(
    state: &ManagerState,
    config: &ManagerConfig,
    anchor_uuid: Option<&str>,
) -> Result<PathBuf, ApiError> {
    let model = validate_model_path(state, &config.model_path)?;
    if config.gpu_uuids.is_empty() {
        return Err(ApiError::bad_request(
            "gpu_uuids must contain at least one GPU",
        ));
    }
    let unique: BTreeSet<_> = config.gpu_uuids.iter().collect();
    if unique.len() != config.gpu_uuids.len() {
        return Err(ApiError::bad_request("gpu_uuids contains a duplicate"));
    }
    if let Some(anchor_uuid) = anchor_uuid {
        if !config.gpu_uuids.iter().any(|uuid| uuid == anchor_uuid) {
            return Err(ApiError::bad_request(
                "the GPU selected in the Apply URL must be included in gpu_uuids",
            ));
        }
    }
    if config.port == 0 {
        return Err(ApiError::bad_request("port must be between 1 and 65535"));
    }
    if config.host.trim().is_empty() {
        return Err(ApiError::bad_request("host cannot be empty"));
    }
    if config.vram_safety_margin_mb < 500 {
        return Err(ApiError::bad_request(
            "vram_safety_margin_mb must be at least 500; the launcher default is 600",
        ));
    }
    if config.layer_group_size == 0 {
        return Err(ApiError::bad_request("layer_group_size must be positive"));
    }
    if config.kv_cache_mb == 0 {
        return Err(ApiError::bad_request("kv_cache_mb must be positive"));
    }
    if config.krasis_threads == 0 {
        return Err(ApiError::bad_request("krasis_threads must be positive"));
    }
    normalize_dynamic_hcs_tail_blocks(&config.dynamic_hcs_tail_blocks)
        .map_err(ApiError::bad_request)?;
    if !(0.0..=1.0).contains(&config.prefix_cache_ram_fraction) {
        return Err(ApiError::bad_request(
            "prefix_cache_ram_fraction must be between 0 and 1",
        ));
    }
    if !matches!(config.expert_group_size, 32 | 64 | 128) {
        return Err(ApiError::bad_request(
            "expert_group_size must be 32, 64, or 128",
        ));
    }
    if !matches!(config.vision_quant.as_str(), "bf16" | "int4") {
        return Err(ApiError::bad_request("vision_quant must be bf16 or int4"));
    }
    for (name, value) in config_text_fields(config) {
        reject_config_text(name, value)?;
    }
    let known: BTreeSet<_> = discover_gpus(state)
        .map_err(ApiError::internal)?
        .into_iter()
        .map(|gpu| gpu.uuid)
        .collect();
    let missing: Vec<_> = config
        .gpu_uuids
        .iter()
        .filter(|uuid| !known.contains(*uuid))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(ApiError::bad_request(format!(
            "unknown GPU UUID(s): {}",
            missing.join(", ")
        )));
    }
    Ok(model)
}

fn config_text_fields(config: &ManagerConfig) -> Vec<(&'static str, &str)> {
    vec![
        ("host", &config.host),
        ("attention_quant", &config.attention_quant),
        ("vision_quant", &config.vision_quant),
        ("hqq_cache_profile", &config.hqq_cache_profile),
        ("hqq_sidecar_manifest", &config.hqq_sidecar_manifest),
        ("kv_dtype", &config.kv_dtype),
        ("gpu_expert_int4_calib", &config.gpu_expert_int4_calib),
        ("shared_expert_quant", &config.shared_expert_quant),
        ("dense_mlp_quant", &config.dense_mlp_quant),
        ("lm_head_quant", &config.lm_head_quant),
        ("hcs_host_cache_mode", &config.hcs_host_cache_mode),
        ("multi_gpu_mode", &config.multi_gpu_mode),
        (
            "adaptive_cold_mass_pruning",
            &config.adaptive_cold_mass_pruning,
        ),
        ("pp_partition", &config.pp_partition),
        ("heatmap_path", &config.heatmap_path),
        ("gguf_path", &config.gguf_path),
        (
            "expert_compression_sidecar",
            &config.expert_compression_sidecar,
        ),
        (
            "expert_compression_pipeline",
            &config.expert_compression_pipeline,
        ),
        ("dspark_mode", &config.dspark_mode),
        ("ssh_tunnel", &config.ssh_tunnel),
        ("ssh_key_path", &config.ssh_key_path),
    ]
}

fn reject_config_text(name: &str, value: &str) -> Result<(), ApiError> {
    if value.contains(['\n', '\r', '\0', '"']) {
        return Err(ApiError::bad_request(format!(
            "{name} contains a character that cannot be serialized safely"
        )));
    }
    Ok(())
}

fn config_text(config: &ManagerConfig, model_path: &Path) -> String {
    let mut values = vec![
        ("MODEL_PATH", model_path.to_string_lossy().to_string()),
        ("CFG_SELECTED_GPUS", config.gpu_uuids.join(",")),
        ("CFG_NUM_GPUS", config.gpu_uuids.len().to_string()),
        ("CFG_PP_PARTITION", config.pp_partition.clone()),
        ("CFG_LAYER_GROUP_SIZE", config.layer_group_size.to_string()),
        ("CFG_KV_CACHE_MB", config.kv_cache_mb.to_string()),
        (
            "CFG_MAX_CONTEXT_TOKENS",
            config.max_context_tokens.to_string(),
        ),
        ("CFG_KV_DTYPE", config.kv_dtype.clone()),
        ("CFG_GPU_EXPERT_BITS", "4".to_string()),
        ("CFG_CPU_EXPERT_BITS", "4".to_string()),
        (
            "CFG_EXPERT_GROUP_SIZE",
            config.expert_group_size.to_string(),
        ),
        (
            "CFG_GPU_EXPERT_INT4_CALIB",
            config.gpu_expert_int4_calib.clone(),
        ),
        ("CFG_ATTENTION_QUANT", config.attention_quant.clone()),
        ("CFG_VISION_QUANT", config.vision_quant.clone()),
        ("CFG_HQQ_CACHE_PROFILE", config.hqq_cache_profile.clone()),
        ("CFG_HQQ_GROUP_SIZE", config.hqq_group_size.to_string()),
        (
            "CFG_HQQ_AUTO_BUDGET_PCT",
            config.hqq_auto_budget_pct.to_string(),
        ),
        (
            "CFG_HQQ_SIDECAR_MANIFEST",
            config.hqq_sidecar_manifest.clone(),
        ),
        (
            "CFG_SHARED_EXPERT_QUANT",
            config.shared_expert_quant.clone(),
        ),
        ("CFG_DENSE_MLP_QUANT", config.dense_mlp_quant.clone()),
        ("CFG_LM_HEAD_QUANT", config.lm_head_quant.clone()),
        ("CFG_KRASIS_THREADS", config.krasis_threads.to_string()),
        ("CFG_HOST", config.host.clone()),
        ("CFG_PORT", config.port.to_string()),
        ("CFG_SSH_TUNNEL", config.ssh_tunnel.clone()),
        ("CFG_SSH_KEY_PATH", config.ssh_key_path.clone()),
        (
            "CFG_GPU_PREFILL_THRESHOLD",
            config.gpu_prefill_threshold.to_string(),
        ),
        ("CFG_GGUF_PATH", config.gguf_path.clone()),
        ("CFG_HEATMAP_PATH", config.heatmap_path.clone()),
        (
            "CFG_VRAM_SAFETY_MARGIN",
            config.vram_safety_margin_mb.to_string(),
        ),
        ("CFG_HCS", bool_config(config.hcs)),
        ("CFG_MULTI_GPU_HCS", bool_config(config.gpu_uuids.len() > 1)),
        ("CFG_MULTI_GPU_MODE", config.multi_gpu_mode.clone()),
        ("CFG_DYNAMIC_PEER", bool_config(config.dynamic_peer)),
        (
            "CFG_HCS_HOST_CACHE_MODE",
            config.hcs_host_cache_mode.clone(),
        ),
        ("CFG_DYNAMIC_HCS", bool_config(config.dynamic_hcs)),
        (
            "CFG_DYNAMIC_HCS_TAIL_BLOCKS",
            config.dynamic_hcs_tail_blocks.clone(),
        ),
        (
            "CFG_ADAPTIVE_COLD_MASS_PRUNING",
            config.adaptive_cold_mass_pruning.clone(),
        ),
        (
            "CFG_EXPERT_COMPRESSION",
            bool_config(config.expert_compression),
        ),
        (
            "CFG_EXPERT_COMPRESSION_SIDECAR",
            config.expert_compression_sidecar.clone(),
        ),
        (
            "CFG_EXPERT_COMPRESSION_PIPELINE",
            config.expert_compression_pipeline.clone(),
        ),
        ("CFG_DSPARK_MODE", config.dspark_mode.clone()),
        ("CFG_ENABLE_THINKING", bool_config(config.enable_thinking)),
        ("CFG_PREFIX_CACHE", bool_config(config.prefix_cache)),
        (
            "CFG_PREFIX_CACHE_RAM_FRACTION",
            config.prefix_cache_ram_fraction.to_string(),
        ),
        (
            "CFG_FORCE_REBUILD_CACHE",
            bool_config(config.force_rebuild_cache),
        ),
        (
            "CFG_FORCE_REBUILD_HQQ_CACHE",
            bool_config(config.force_rebuild_hqq_cache),
        ),
    ];
    values.sort_by(|left, right| left.0.cmp(right.0));
    let mut output = "# Generated by Krasis Manager; validated before Apply\n".to_string();
    for (key, value) in values {
        output.push_str(&format!("{key}=\"{value}\"\n"));
    }
    output
}

fn bool_config(value: bool) -> String {
    if value { "1" } else { "0" }.to_string()
}

fn describe_config(state: &ManagerState, payload: &DescribeRequest) -> Result<Value, ApiError> {
    let model = validate_model_path(state, &payload.model_path)?;
    if payload.gpu_uuids.is_empty() {
        return Err(ApiError::bad_request("gpu_uuids must not be empty"));
    }
    let known: BTreeSet<_> = discover_gpus(state)
        .map_err(ApiError::internal)?
        .into_iter()
        .map(|gpu| gpu.uuid)
        .collect();
    if payload.gpu_uuids.iter().any(|uuid| !known.contains(uuid)) {
        return Err(ApiError::bad_request("gpu_uuids contains an unknown GPU"));
    }
    let output = run_with_timeout(
        Command::new(&state.inner.python_executable)
            .args(["-m", "krasis.launcher", "_manager-schema", "--model-path"])
            .arg(&model)
            .arg("--selected-gpus")
            .arg(payload.gpu_uuids.join(",")),
        VALIDATION_TIMEOUT,
    )
    .map_err(ApiError::internal)?;
    if !output.status.success() {
        return Err(ApiError::bad_request(format!(
            "launcher capability resolution failed: {}",
            combined_output(&output)
        )));
    }
    let marker = "KRASIS_MANAGER_SCHEMA=";
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(marker))
        .ok_or_else(|| ApiError::internal("launcher did not return a manager schema"))?;
    serde_json::from_str(line).map_err(|error| {
        ApiError::internal(format!("launcher returned invalid schema JSON: {error}"))
    })
}

fn validate_config_request(
    state: &ManagerState,
    config: &ManagerConfig,
    anchor_uuid: Option<&str>,
) -> Result<Value, ApiError> {
    let model = validate_config_syntax(state, config, anchor_uuid)?;
    let validation_dir = state.inner.manager_dir.join("validation");
    fs::create_dir_all(&validation_dir).map_err(|error| {
        ApiError::internal(format!("cannot create validation directory: {error}"))
    })?;
    let path = validation_dir.join(format!("{}.conf", operation_id()));
    fs::write(&path, config_text(config, &model))
        .map_err(|error| ApiError::internal(format!("cannot write validation config: {error}")))?;
    let result = run_launcher_validation(state, &path);
    let _ = fs::remove_file(&path);
    let output = result.map_err(ApiError::internal)?;
    if !output.status.success() {
        return Err(ApiError::bad_request(format!(
            "launcher validation failed: {}",
            combined_output(&output)
        )));
    }
    Ok(json!({
        "valid": true,
        "message": "Configuration passed the existing Krasis launcher capability, topology, and budget gates.",
        "launcher_output": combined_output(&output),
    }))
}

fn run_launcher_validation(state: &ManagerState, config_path: &Path) -> Result<Output, String> {
    run_with_timeout(
        Command::new(&state.inner.python_executable)
            .args(["-m", "krasis.launcher", "--config"])
            .arg(config_path)
            .args(["--non-interactive", "--validate-only"]),
        VALIDATION_TIMEOUT,
    )
}

fn run_with_timeout(command: &mut Command, timeout: Duration) -> Result<Output, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot start subprocess: {error}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "subprocess stdout was not captured".to_string())?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| "subprocess stderr was not captured".to_string())?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.read_to_end(&mut bytes);
        (result, bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stderr.read_to_end(&mut bytes);
        (result, bytes)
    });
    let started = Instant::now();
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("cannot poll subprocess: {error}"))?
        {
            let (stdout_result, stdout) = stdout_reader
                .join()
                .map_err(|_| "subprocess stdout reader panicked".to_string())?;
            let (stderr_result, stderr) = stderr_reader
                .join()
                .map_err(|_| "subprocess stderr reader panicked".to_string())?;
            stdout_result.map_err(|error| format!("cannot read subprocess stdout: {error}"))?;
            stderr_result.map_err(|error| format!("cannot read subprocess stderr: {error}"))?;
            return Ok(Output {
                status,
                stdout,
                stderr,
            });
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("subprocess exceeded {} seconds", timeout.as_secs()));
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn combined_output(output: &Output) -> String {
    let mut text = strip_ansi(&String::from_utf8_lossy(&output.stdout));
    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    if !stderr.trim().is_empty() {
        if !text.ends_with('\n') && !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&stderr);
    }
    let trimmed = text.trim();
    if trimmed.len() > 16_000 {
        let target = trimmed.len() - 16_000;
        let boundary = trimmed
            .char_indices()
            .find_map(|(index, _)| (index >= target).then_some(index))
            .unwrap_or(0);
        trimmed[boundary..].to_string()
    } else {
        trimmed.to_string()
    }
}

fn start_apply(
    state: ManagerState,
    anchor_uuid: String,
    config: ManagerConfig,
) -> Result<Operation, ApiError> {
    let model = validate_config_syntax(&state, &config, Some(&anchor_uuid))?;
    let gpus = discover_gpus(&state).map_err(ApiError::internal)?;
    let selected: BTreeSet<_> = config.gpu_uuids.iter().cloned().collect();
    let mut affected = selected.clone();
    let mut current_pids = BTreeSet::new();
    for gpu in &gpus {
        if !selected.contains(&gpu.uuid) {
            continue;
        }
        for process in &gpu.processes {
            if !process.is_krasis {
                return Err(ApiError::conflict(format!(
                    "GPU {} is used by unrelated PID {}; Krasis Manager will not stop it",
                    gpu.uuid, process.pid
                )));
            }
            current_pids.insert(process.pid);
            affected.extend(process.gpu_uuids.iter().cloned());
        }
    }
    if port_is_open(config.port)
        && !gpus.iter().any(|gpu| {
            selected.contains(&gpu.uuid)
                && gpu
                    .processes
                    .iter()
                    .any(|process| process.is_krasis && process.port == Some(config.port))
        })
    {
        return Err(ApiError::conflict(format!(
            "port {} is already accepting connections and is not owned by the Krasis process being replaced",
            config.port
        )));
    }
    let id = operation_id();
    let gpu_uuids: Vec<_> = affected.into_iter().collect();
    let operation = Operation {
        id: id.clone(),
        kind: "apply".to_string(),
        status: "queued".to_string(),
        message: "Apply request queued".to_string(),
        gpu_uuids: gpu_uuids.clone(),
        model_path: Some(model.to_string_lossy().to_string()),
        pid: None,
        port: Some(config.port),
        created_at_unix_ms: now_unix_ms(),
        updated_at_unix_ms: now_unix_ms(),
        terminal: false,
        error: None,
        logs: Vec::new(),
    };
    {
        let mut mutable = state.inner.mutable.lock().unwrap();
        for uuid in &gpu_uuids {
            if let Some(owner) = mutable.gpu_locks.get(uuid) {
                return Err(ApiError::conflict(format!(
                    "GPU {uuid} already has active operation {owner}"
                )));
            }
        }
        for uuid in &gpu_uuids {
            mutable.gpu_locks.insert(uuid.clone(), id.clone());
        }
        mutable.operations.insert(id.clone(), operation.clone());
    }
    thread::spawn(move || apply_worker(state, id, config, model, current_pids));
    Ok(operation)
}

fn verify_apply_ownership(
    state: &ManagerState,
    selected_uuids: &[String],
    expected_pids: &BTreeSet<u32>,
) -> Result<(), String> {
    let selected: BTreeSet<_> = selected_uuids.iter().map(String::as_str).collect();
    let mut actual_pids = BTreeSet::new();
    for gpu in discover_gpus(state)? {
        if !selected.contains(gpu.uuid.as_str()) {
            continue;
        }
        for process in gpu.processes {
            if !process.is_krasis {
                return Err(format!(
                    "GPU {} gained unrelated PID {} during validation; no process was stopped",
                    gpu.uuid, process.pid
                ));
            }
            actual_pids.insert(process.pid);
        }
    }
    if &actual_pids != expected_pids {
        return Err(format!(
            "GPU ownership changed during validation (expected Krasis PID(s) [{}], now [{}]); no process was stopped",
            expected_pids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            actual_pids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(())
}

fn wait_for_gpus_clear(
    state: &ManagerState,
    gpu_uuids: &[String],
    timeout: Duration,
) -> Result<(), String> {
    let selected: BTreeSet<_> = gpu_uuids.iter().map(String::as_str).collect();
    let started = Instant::now();
    loop {
        let occupied = discover_gpus(state)?
            .into_iter()
            .filter(|gpu| selected.contains(gpu.uuid.as_str()) && !gpu.processes.is_empty())
            .map(|gpu| {
                format!(
                    "{} by PID(s) {}",
                    gpu.uuid,
                    gpu.processes
                        .iter()
                        .map(|process| process.pid.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect::<Vec<_>>();
        if occupied.is_empty() {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            return Err(format!(
                "selected GPU allocation did not clear after stop: {}",
                occupied.join("; ")
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn wait_for_gpu_process_records_clear(
    state: &ManagerState,
    pids: &BTreeSet<u32>,
    timeout: Duration,
) -> Result<(), String> {
    let started = Instant::now();
    loop {
        let remaining = discover_compute_processes(state)?
            .into_iter()
            .filter_map(|(pid, gpu_uuid, used_memory_mb)| {
                pids.contains(&pid)
                    .then_some(format!("PID {pid} on {gpu_uuid} ({used_memory_mb} MiB)"))
            })
            .collect::<Vec<_>>();
        if remaining.is_empty() {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            return Err(format!(
                "NVIDIA still reports CUDA allocation(s) after process exit: {}",
                remaining.join(", ")
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

#[cfg(unix)]
fn isolate_managed_model(command: &mut Command) {
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(windows)]
fn isolate_managed_model(command: &mut Command) {
    use windows_sys::Win32::System::Threading::{CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS};
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
}

fn apply_worker(
    state: ManagerState,
    id: String,
    config: ManagerConfig,
    model: PathBuf,
    current_pids: BTreeSet<u32>,
) {
    let run_dir = state.inner.manager_dir.join("runs").join(&id);
    if let Err(error) = fs::create_dir_all(&run_dir) {
        fail_operation(
            &state,
            &id,
            format!("cannot create operation directory: {error}"),
        );
        return;
    }
    let config_path = run_dir.join("krasis.conf");
    let log_path = run_dir.join("startup.log");
    if let Err(error) = fs::write(&config_path, config_text(&config, &model)) {
        fail_operation(&state, &id, format!("cannot write launch config: {error}"));
        return;
    }

    update_operation(
        &state,
        &id,
        "validating",
        "Validating model capabilities, topology, and computed budgets",
        false,
    );
    let validation = match run_launcher_validation(&state, &config_path) {
        Ok(output) => output,
        Err(error) => {
            fail_operation(
                &state,
                &id,
                format!("launcher validation could not run: {error}"),
            );
            return;
        }
    };
    for line in combined_output(&validation).lines() {
        append_operation_log(&state, &id, format!("validate: {line}"));
    }
    if !validation.status.success() {
        fail_operation(
            &state,
            &id,
            "Configuration failed launcher validation; the existing model was left running",
        );
        return;
    }
    if let Err(error) = verify_apply_ownership(&state, &config.gpu_uuids, &current_pids) {
        fail_operation(&state, &id, error);
        return;
    }

    if current_pids.is_empty() {
        update_operation(
            &state,
            &id,
            "waiting_for_gpu",
            "Selected GPU allocation is clear",
            false,
        );
    } else {
        update_operation(
            &state,
            &id,
            "stopping",
            format!(
                "Stopping existing Krasis process{} {}",
                if current_pids.len() == 1 { "" } else { "es" },
                current_pids
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            false,
        );
        for pid in &current_pids {
            if process_exists(*pid)
                && !process_command_line(*pid)
                    .as_deref()
                    .is_some_and(is_krasis_server_command)
            {
                fail_operation(
                    &state,
                    &id,
                    format!("PID {pid} changed identity before stop; refusing to terminate it"),
                );
                return;
            }
            if let Err(error) = terminate_process(*pid, false) {
                fail_operation(
                    &state,
                    &id,
                    format!("could not stop Krasis PID {pid}: {error}"),
                );
                return;
            }
        }
        update_operation(
            &state,
            &id,
            "waiting_for_gpu",
            "Waiting for the old process and CUDA allocations to release",
            false,
        );
        if let Err(error) = wait_for_processes_exit(&current_pids, STOP_TIMEOUT) {
            fail_operation(&state, &id, error);
            return;
        }
    }
    if let Err(error) = wait_for_gpus_clear(&state, &config.gpu_uuids, STOP_TIMEOUT) {
        fail_operation(&state, &id, error);
        return;
    }

    if port_is_open(config.port) && !current_pids.is_empty() {
        thread::sleep(Duration::from_secs(1));
    }
    if port_is_open(config.port) {
        fail_operation(
            &state,
            &id,
            format!(
                "port {} is already accepting connections after the selected Krasis process stopped",
                config.port
            ),
        );
        return;
    }

    update_operation(
        &state,
        &id,
        "launching",
        "Starting the validated configuration through the normal Krasis launcher",
        false,
    );
    let log_stdout = match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(file) => file,
        Err(error) => {
            fail_operation(&state, &id, format!("cannot create startup log: {error}"));
            return;
        }
    };
    let log_stderr = match log_stdout.try_clone() {
        Ok(file) => file,
        Err(error) => {
            fail_operation(&state, &id, format!("cannot clone startup log: {error}"));
            return;
        }
    };
    let mut command = Command::new(&state.inner.python_executable);
    command
        .args(["-m", "krasis.launcher", "--config"])
        .arg(&config_path)
        .arg("--non-interactive")
        .env("KRASIS_MANAGER_OPERATION_ID", &id)
        .stdout(Stdio::from(log_stdout))
        .stderr(Stdio::from(log_stderr));
    isolate_managed_model(&mut command);
    let child = command.spawn();
    let child = match child {
        Ok(child) => child,
        Err(error) => {
            fail_operation(&state, &id, format!("cannot start Krasis: {error}"));
            return;
        }
    };
    let pid = child.id();
    set_operation_pid(&state, &id, pid);
    update_operation(
        &state,
        &id,
        "loading",
        format!(
            "Krasis PID {pid} is loading; waiting for health on port {}",
            config.port
        ),
        false,
    );
    monitor_startup(&state, &id, &log_path, child, config.port);
}

fn monitor_startup(state: &ManagerState, id: &str, log_path: &Path, mut child: Child, port: u16) {
    let mut offset = 0u64;
    loop {
        read_log_updates(state, id, log_path, &mut offset);
        match child.try_wait() {
            Ok(Some(status)) => {
                read_log_updates(state, id, log_path, &mut offset);
                fail_operation(
                    state,
                    id,
                    format!("Krasis exited before readiness with status {status}"),
                );
                return;
            }
            Ok(None) => {}
            Err(error) => {
                fail_operation(state, id, format!("cannot monitor Krasis process: {error}"));
                return;
            }
        }
        if health_ready(port) {
            update_operation(
                state,
                id,
                "ready",
                format!("Model is healthy and ready on port {port}"),
                true,
            );
            let gpu_uuids = {
                let mutable = state.inner.mutable.lock().unwrap();
                mutable
                    .operations
                    .get(id)
                    .map(|operation| operation.gpu_uuids.clone())
                    .unwrap_or_default()
            };
            release_gpu_locks(state, id, &gpu_uuids);
            let monitor_state = state.clone();
            let monitor_id = id.to_string();
            let monitor_log = log_path.to_path_buf();
            thread::spawn(move || {
                let status = child.wait();
                let mut tail_offset = offset;
                read_log_updates(&monitor_state, &monitor_id, &monitor_log, &mut tail_offset);
                match status {
                    Ok(status) => append_operation_log(
                        &monitor_state,
                        &monitor_id,
                        format!("Managed server later exited with status {status}"),
                    ),
                    Err(error) => append_operation_log(
                        &monitor_state,
                        &monitor_id,
                        format!("Could not wait for managed server exit: {error}"),
                    ),
                }
            });
            return;
        }
        thread::sleep(Duration::from_secs(1));
    }
}

fn read_log_updates(state: &ManagerState, id: &str, log_path: &Path, offset: &mut u64) {
    let Ok(mut file) = File::open(log_path) else {
        return;
    };
    if file.seek(SeekFrom::Start(*offset)).is_err() {
        return;
    }
    let mut content = String::new();
    if file.read_to_string(&mut content).is_err() {
        return;
    }
    *offset += content.len() as u64;
    for line in content.lines() {
        append_operation_log(state, id, line);
        let stripped = strip_ansi(line).to_lowercase();
        let message = if stripped.contains("vram calibration complete") {
            Some("VRAM calibration completed")
        } else if stripped.contains("long calibration: probing") {
            Some("Running long-prompt VRAM calibration")
        } else if stripped.contains("short calibration: probing") {
            Some("Running short-prompt VRAM calibration")
        } else if stripped.contains("building expert heatmap") {
            Some("Building the exact runtime route heatmap")
        } else if stripped.contains("quick heatmap prompt") {
            Some("Building the exact runtime route heatmap")
        } else if stripped.contains("hcs pool:") {
            Some("Admitting measured HCS expert residency")
        } else if stripped.contains("rust http server listening") {
            Some("HTTP listener started; waiting for health")
        } else {
            None
        };
        if let Some(message) = message {
            update_operation(state, id, "loading", message, false);
        }
    }
}

fn start_stop(state: ManagerState, anchor_uuid: String) -> Result<Operation, ApiError> {
    let gpus = discover_gpus(&state).map_err(ApiError::internal)?;
    let gpu = gpus
        .iter()
        .find(|gpu| gpu.uuid == anchor_uuid)
        .ok_or_else(|| ApiError::not_found(format!("GPU {anchor_uuid} not found")))?;
    let mut pids = BTreeSet::new();
    let mut gpu_uuids = BTreeSet::from([anchor_uuid.clone()]);
    for process in &gpu.processes {
        if !process.is_krasis {
            return Err(ApiError::conflict(format!(
                "GPU {anchor_uuid} is used by unrelated PID {}; Krasis Manager will not stop it",
                process.pid
            )));
        }
        pids.insert(process.pid);
        gpu_uuids.extend(process.gpu_uuids.iter().cloned());
    }
    let id = operation_id();
    let gpu_uuids: Vec<_> = gpu_uuids.into_iter().collect();
    let operation = Operation {
        id: id.clone(),
        kind: "stop".to_string(),
        status: "queued".to_string(),
        message: "Stop request queued".to_string(),
        gpu_uuids: gpu_uuids.clone(),
        model_path: gpu
            .processes
            .first()
            .and_then(|process| process.model_path.clone()),
        pid: pids.iter().next().copied(),
        port: gpu.processes.first().and_then(|process| process.port),
        created_at_unix_ms: now_unix_ms(),
        updated_at_unix_ms: now_unix_ms(),
        terminal: false,
        error: None,
        logs: Vec::new(),
    };
    {
        let mut mutable = state.inner.mutable.lock().unwrap();
        for uuid in &gpu_uuids {
            if let Some(owner) = mutable.gpu_locks.get(uuid) {
                return Err(ApiError::conflict(format!(
                    "GPU {uuid} already has active operation {owner}"
                )));
            }
        }
        for uuid in &gpu_uuids {
            mutable.gpu_locks.insert(uuid.clone(), id.clone());
        }
        mutable.operations.insert(id.clone(), operation.clone());
    }
    thread::spawn(move || stop_worker(state, id, pids));
    Ok(operation)
}

fn stop_worker(state: ManagerState, id: String, pids: BTreeSet<u32>) {
    if pids.is_empty() {
        update_operation(&state, &id, "stopped", "GPU was already idle", true);
        let gpu_uuids = {
            let mutable = state.inner.mutable.lock().unwrap();
            mutable
                .operations
                .get(&id)
                .map(|operation| operation.gpu_uuids.clone())
                .unwrap_or_default()
        };
        release_gpu_locks(&state, &id, &gpu_uuids);
        return;
    }
    update_operation(
        &state,
        &id,
        "stopping",
        format!(
            "Stopping Krasis process{} {}",
            if pids.len() == 1 { "" } else { "es" },
            pids.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        false,
    );
    for pid in &pids {
        if process_exists(*pid)
            && !process_command_line(*pid)
                .as_deref()
                .is_some_and(is_krasis_server_command)
        {
            fail_operation(
                &state,
                &id,
                format!("PID {pid} changed identity before stop; refusing to terminate it"),
            );
            return;
        }
        if let Err(error) = terminate_process(*pid, false) {
            fail_operation(
                &state,
                &id,
                format!("could not stop Krasis PID {pid}: {error}"),
            );
            return;
        }
    }
    update_operation(
        &state,
        &id,
        "waiting_for_gpu",
        "Waiting for the process and CUDA allocations to release",
        false,
    );
    if let Err(error) = wait_for_processes_exit(&pids, STOP_TIMEOUT) {
        fail_operation(&state, &id, error);
        return;
    }
    update_operation(
        &state,
        &id,
        "waiting_for_gpu",
        "Process exited; verifying CUDA allocations are released",
        false,
    );
    if let Err(error) = wait_for_gpu_process_records_clear(&state, &pids, STOP_TIMEOUT) {
        fail_operation(&state, &id, error);
        return;
    }
    update_operation(
        &state,
        &id,
        "stopped",
        "Verified Krasis model stopped",
        true,
    );
    let gpu_uuids = {
        let mutable = state.inner.mutable.lock().unwrap();
        mutable
            .operations
            .get(&id)
            .map(|operation| operation.gpu_uuids.clone())
            .unwrap_or_default()
    };
    release_gpu_locks(&state, &id, &gpu_uuids);
}

#[cfg(unix)]
fn terminate_process(pid: u32, force: bool) -> Result<(), String> {
    let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
    let result = unsafe { libc::kill(pid as libc::pid_t, signal) };
    if result == 0 {
        Ok(())
    } else {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error.to_string())
        }
    }
}

#[cfg(windows)]
fn terminate_process(pid: u32, force: bool) -> Result<(), String> {
    fn taskkill(pid: u32, force: bool) -> Result<Output, String> {
        let mut command = Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T"]);
        if force {
            command.arg("/F");
        }
        command
            .output()
            .map_err(|error| format!("taskkill failed to start: {error}"))
    }

    let mut output = taskkill(pid, force)?;
    if !force && !output.status.success() {
        // Console processes do not always accept taskkill's window-close path.
        // The target was re-verified as krasis.server immediately before this
        // call, so escalate to terminating only that process tree.
        output = taskkill(pid, true)?;
    }
    if output.status.success() || !process_exists(pid) {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    let filter = format!("PID eq {pid}");
    Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .ok()
        .is_some_and(|output| {
            let needle = format!(",\"{pid}\",");
            String::from_utf8_lossy(&output.stdout).contains(&needle)
        })
}

fn wait_for_processes_exit(pids: &BTreeSet<u32>, timeout: Duration) -> Result<(), String> {
    let started = Instant::now();
    loop {
        let remaining: Vec<_> = pids
            .iter()
            .copied()
            .filter(|pid| process_exists(*pid))
            .collect();
        if remaining.is_empty() {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            for pid in &remaining {
                let _ = terminate_process(*pid, true);
            }
            let forced_at = Instant::now();
            while forced_at.elapsed() < Duration::from_secs(10) {
                if remaining.iter().all(|pid| !process_exists(*pid)) {
                    return Ok(());
                }
                thread::sleep(Duration::from_millis(250));
            }
            return Err(format!(
                "Krasis process(es) {} did not exit after graceful and forced stop",
                remaining
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn port_is_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_millis(250),
    )
    .is_ok()
}

fn health_ready(port: u16) -> bool {
    let address = match format!("127.0.0.1:{port}").parse() {
        Ok(address) => address,
        Err(_) => return false,
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return false;
    }
    if !response.starts_with("HTTP/1.1 200") {
        return false;
    }
    let Some((_, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|status| status == "ok")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state(port: u16, lan: bool) -> ManagerState {
        ManagerState {
            inner: Arc::new(ManagerInner {
                python_executable: PathBuf::from("python"),
                manager_dir: PathBuf::from("/tmp/krasis-manager-test"),
                models_dir: PathBuf::from("/tmp/krasis-manager-test/models"),
                nvidia_smi: PathBuf::from("nvidia-smi"),
                token: "test-owner-token".to_string(),
                port,
                lan,
                mutable: Mutex::new(MutableState::default()),
            }),
        }
    }

    #[test]
    fn command_classification_requires_python_module_server_pair() {
        assert!(is_krasis_server_command(
            "python -m krasis.server --config /tmp/model.conf"
        ));
        assert!(is_krasis_server_command(
            "\"C:\\Krasis Runtime\\python.exe\" -m krasis.server --config \"C:\\Users\\Name\\model.conf\""
        ));
        assert!(!is_krasis_server_command(
            "python helper.py /tmp/krasis.server.log"
        ));
        assert!(!is_krasis_server_command("krasis manager"));
    }

    #[test]
    fn extracts_quoted_config_paths() {
        assert_eq!(
            extract_config_path(
                "python -m krasis.server --config \"/tmp/path with spaces/model.conf\""
            ),
            Some(PathBuf::from("/tmp/path with spaces/model.conf"))
        );
        assert_eq!(
            extract_config_path(
                "\"C:\\Krasis Runtime\\python.exe\" -m krasis.server --config \"C:\\Users\\Name\\model config.conf\""
            ),
            Some(PathBuf::from("C:\\Users\\Name\\model config.conf"))
        );
    }

    #[test]
    fn generated_config_is_int4_and_contains_manager_fields() {
        let config = ManagerConfig {
            model_path: "/models/test".to_string(),
            gpu_uuids: vec!["GPU-one".to_string(), "GPU-two".to_string()],
            attention_quant: "hqq6".to_string(),
            kv_dtype: "k6v6".to_string(),
            ssh_tunnel: "user@example".to_string(),
            ..ManagerConfig::default()
        };
        let text = config_text(&config, Path::new("/models/test"));
        assert!(text.contains("CFG_GPU_EXPERT_BITS=\"4\""));
        assert!(text.contains("CFG_CPU_EXPERT_BITS=\"4\""));
        assert!(text.contains("CFG_SELECTED_GPUS=\"GPU-one,GPU-two\""));
        assert!(text.contains("CFG_VISION_QUANT=\"int4\""));
        assert!(text.contains("CFG_SSH_TUNNEL=\"user@example\""));
        assert!(text.contains("CFG_DYNAMIC_HCS_TAIL_BLOCKS=\"auto\""));
        assert!(!text.to_lowercase().contains("awq"));
        assert!(!text.to_lowercase().contains("fp8"));
    }

    #[test]
    fn manager_tail_policy_accepts_auto_and_legacy_numeric_payloads() {
        let automatic: ManagerConfig =
            serde_json::from_str(r#"{"dynamic_hcs_tail_blocks":"auto"}"#).unwrap();
        assert_eq!(automatic.dynamic_hcs_tail_blocks, "auto");

        let legacy: ManagerConfig =
            serde_json::from_str(r#"{"dynamic_hcs_tail_blocks":5}"#).unwrap();
        assert_eq!(legacy.dynamic_hcs_tail_blocks, "5");

        let invalid = serde_json::from_str::<ManagerConfig>(r#"{"dynamic_hcs_tail_blocks":6}"#);
        assert!(invalid.is_err());
    }

    #[test]
    fn saved_server_config_round_trips_into_the_editor() {
        let saved = BTreeMap::from([
            ("MODEL_PATH".to_string(), "/models/current".to_string()),
            ("CFG_PORT".to_string(), "8123".to_string()),
            ("CFG_ATTENTION_QUANT".to_string(), "hqq4".to_string()),
            ("CFG_VISION_QUANT".to_string(), "bf16".to_string()),
            ("CFG_KV_DTYPE".to_string(), "k4v4".to_string()),
            ("CFG_VRAM_SAFETY_MARGIN".to_string(), "600".to_string()),
            ("CFG_DYNAMIC_HCS".to_string(), "0".to_string()),
            ("CFG_SSH_TUNNEL".to_string(), "user@host".to_string()),
        ]);
        let config = manager_config_from_saved(&saved, &["GPU-stable".to_string()]).unwrap();
        assert_eq!(config.model_path, "/models/current");
        assert_eq!(config.gpu_uuids, ["GPU-stable"]);
        assert_eq!(config.port, 8123);
        assert_eq!(config.attention_quant, "hqq4");
        assert_eq!(config.vision_quant, "bf16");
        assert_eq!(config.kv_dtype, "k4v4");
        assert!(!config.dynamic_hcs);
        assert_eq!(config.dynamic_hcs_tail_blocks, "auto");
        assert_eq!(config.ssh_tunnel, "user@host");
    }

    #[test]
    fn parser_rejects_ambiguous_or_injected_config_json() {
        let unknown = br#"{"model_path":"x","gpu_uuids":["GPU-x"],"fallback":true}"#;
        assert!(serde_json::from_slice::<ManagerConfig>(unknown).is_err());
        assert!(reject_config_text("host", "ok\nCFG_PORT=1").is_err());
        assert!(reject_config_text("host", "contains\"quote").is_err());
    }

    #[test]
    fn http_parser_honors_body_length() {
        let request = b"POST /api/v1/configs/validate HTTP/1.1\r\nHost: 127.0.0.1:8090\r\nContent-Length: 2\r\n\r\n{}";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let sender = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream.write_all(request).unwrap();
        });
        let (mut stream, _) = listener.accept().unwrap();
        let parsed = read_http_request(&mut stream).unwrap();
        sender.join().unwrap();
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.body, b"{}");
    }

    #[test]
    fn page_uses_actual_port_and_mutations_are_fail_closed() {
        let state = test_state(18123, false);
        let get = HttpRequest {
            method: "GET".to_string(),
            target: "/".to_string(),
            headers: HashMap::from([("host".to_string(), "127.0.0.1:18123".to_string())]),
            body: Vec::new(),
            local_addr: Some("127.0.0.1:18123".parse().unwrap()),
            peer_addr: Some("127.0.0.1:42000".parse().unwrap()),
        };
        let (_, _, html) = route_request(&get, &state).unwrap();
        assert!(html.contains("http://127.0.0.1:18123"));
        assert!(html.contains("test-owner-token"));
        assert!(!html.contains("__KRASIS_MANAGER_BASE__"));

        let unauthenticated = HttpRequest {
            method: "POST".to_string(),
            target: "/api/v1/configs/validate".to_string(),
            headers: get.headers.clone(),
            body: b"{}".to_vec(),
            local_addr: get.local_addr,
            peer_addr: get.peer_addr,
        };
        assert_eq!(
            route_request(&unauthenticated, &state).unwrap_err().status,
            401
        );

        let foreign_host = HttpRequest {
            method: "GET".to_string(),
            target: "/api/v1/status".to_string(),
            headers: HashMap::from([("host".to_string(), "manager.example:18123".to_string())]),
            body: Vec::new(),
            local_addr: get.local_addr,
            peer_addr: get.peer_addr,
        };
        assert_eq!(
            route_request(&foreign_host, &state).unwrap_err().status,
            403
        );
    }

    #[test]
    fn lan_mode_requires_exact_destination_and_authenticates_every_api() {
        let state = test_state(18080, true);
        let request = HttpRequest {
            method: "GET".to_string(),
            target: "/api/v1/status".to_string(),
            headers: HashMap::from([("host".to_string(), "192.168.1.181:18080".to_string())]),
            body: Vec::new(),
            local_addr: Some("192.168.1.181:18080".parse().unwrap()),
            peer_addr: Some("192.168.1.50:42000".parse().unwrap()),
        };
        assert_eq!(route_request(&request, &state).unwrap_err().status, 401);

        let mut authenticated = request;
        authenticated.headers.insert(
            "x-krasis-manager-token".to_string(),
            "test-owner-token".to_string(),
        );
        let (_, _, status) = route_request(&authenticated, &state).unwrap();
        assert!(status.contains("\"network_mode\": \"lan\""));
        assert!(status.contains("\"api_authentication\": \"all\""));

        authenticated
            .headers
            .insert("host".to_string(), "192.168.1.182:18080".to_string());
        assert_eq!(
            route_request(&authenticated, &state).unwrap_err().status,
            403
        );
    }

    #[test]
    fn lan_page_omits_token_for_remote_peers_and_enforces_exact_origin() {
        let state = test_state(18080, true);
        let page = HttpRequest {
            method: "GET".to_string(),
            target: "/".to_string(),
            headers: HashMap::from([("host".to_string(), "192.168.1.181:18080".to_string())]),
            body: Vec::new(),
            local_addr: Some("192.168.1.181:18080".parse().unwrap()),
            peer_addr: Some("192.168.1.50:42000".parse().unwrap()),
        };
        let (_, _, html) = route_request(&page, &state).unwrap();
        assert!(!html.contains("test-owner-token"));
        assert!(html.contains("http://192.168.1.181:18080"));
        assert!(html.contains("LAN enabled"));
        assert!(!html.contains("__KRASIS_MANAGER_"));

        let mut mutation = page;
        mutation.method = "POST".to_string();
        mutation.headers.insert(
            "x-krasis-manager-token".to_string(),
            "test-owner-token".to_string(),
        );
        mutation.headers.insert(
            "origin".to_string(),
            "http://192.168.1.181:18080".to_string(),
        );
        assert!(validate_mutation_request(&mutation, &state).is_ok());
        mutation.headers.insert(
            "origin".to_string(),
            "http://192.168.1.182:18080".to_string(),
        );
        assert_eq!(
            validate_mutation_request(&mutation, &state)
                .unwrap_err()
                .status,
            403
        );
    }

    #[test]
    fn embedded_page_documents_agent_lifecycle_and_stop() {
        let html = include_str!("manager.html");
        for required in [
            "curl",
            "X-Krasis-Manager-Token",
            "/api/v1/gpus",
            "/apply",
            "/stop",
            "/api/v1/operations/",
            "ACTIVE_CONFIG",
            "visionQuantField",
            "vision_quant",
            "attention_presets",
            "attentionChoice",
            "attentionSelection",
            "attention_quant:attention.mode",
            "hqq_auto_budget_pct:attention.pct",
            "syncAttentionBudget",
            "Defaults to the larger of 60,000 or one-quarter of the model limit, capped at that limit.",
            "contextInput.max=String(modelLimit)",
        ] {
            assert!(html.contains(required), "missing {required}");
        }
    }
}
