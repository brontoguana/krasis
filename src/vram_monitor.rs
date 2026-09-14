//! VRAM monitor — background thread polling cudaMemGetInfo every ~50ms.
//!
//! Tracks min-free-VRAM (peak usage) per GPU device. Used to:
//! 1. Measure actual VRAM headroom after warmup (for HCS budget)
//! 2. Warn at runtime when free VRAM hits new lows below safety margin
//! 3. Record VRAM report (periodic samples + named events) when enabled
//!
//! CUDA functions are loaded at runtime — no link-time dependency on
//! libcudart. PyTorch normally loads it first, but native Windows builds may
//! resolve the bundled CUDA runtime DLL from the package directory.

use cudarc::driver::sys as cuda_sys;
use pyo3::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const VRAM_HARD_EXIT_FLOOR_MB: u64 = 125;
pub const CUDA_CONTEXT_POISONED_EXIT_CODE: i32 = 138;
const PRESSURE_DEVICE_SLOTS: usize = u64::BITS as usize;

#[derive(Clone, Copy, Debug)]
pub struct VramPressure {
    pub device_id: i32,
    pub free_mb: u64,
    pub safety_margin_mb: u64,
    pub deficit_mb: u64,
}

static PRESSURE_PENDING_MASK: AtomicU64 = AtomicU64::new(0);
static PRESSURE_FREE_MB: [AtomicU64; PRESSURE_DEVICE_SLOTS] =
    [const { AtomicU64::new(u64::MAX) }; PRESSURE_DEVICE_SLOTS];
static PRESSURE_MARGIN_MB: [AtomicU64; PRESSURE_DEVICE_SLOTS] =
    [const { AtomicU64::new(0) }; PRESSURE_DEVICE_SLOTS];
static PRESSURE_DEFICIT_MB: [AtomicU64; PRESSURE_DEVICE_SLOTS] =
    [const { AtomicU64::new(0) }; PRESSURE_DEVICE_SLOTS];

fn pressure_slot(device_id: i32) -> Option<usize> {
    if device_id >= 0 && (device_id as usize) < PRESSURE_DEVICE_SLOTS {
        Some(device_id as usize)
    } else {
        None
    }
}

pub fn mark_pressure(device_id: i32, free_mb: u64, safety_margin_mb: u64) {
    let Some(slot) = pressure_slot(device_id) else {
        return;
    };
    let worst_free_mb = PRESSURE_FREE_MB[slot]
        .fetch_min(free_mb, Ordering::Relaxed)
        .min(free_mb);
    let max_margin_mb = PRESSURE_MARGIN_MB[slot]
        .fetch_max(safety_margin_mb, Ordering::Relaxed)
        .max(safety_margin_mb);
    PRESSURE_DEFICIT_MB[slot].store(
        max_margin_mb.saturating_sub(worst_free_mb),
        Ordering::Relaxed,
    );
    PRESSURE_PENDING_MASK.fetch_or(1u64 << slot, Ordering::Release);
}

pub fn clear_pressure(device_id: i32) {
    let Some(slot) = pressure_slot(device_id) else {
        return;
    };
    PRESSURE_PENDING_MASK.fetch_and(!(1u64 << slot), Ordering::AcqRel);
    PRESSURE_FREE_MB[slot].store(u64::MAX, Ordering::Relaxed);
    PRESSURE_MARGIN_MB[slot].store(0, Ordering::Relaxed);
    PRESSURE_DEFICIT_MB[slot].store(0, Ordering::Relaxed);
}

pub fn pressure_pending(device_id: i32) -> Option<VramPressure> {
    let slot = pressure_slot(device_id)?;
    let mask = PRESSURE_PENDING_MASK.load(Ordering::Acquire);
    if mask & (1u64 << slot) == 0 {
        return None;
    }
    Some(VramPressure {
        device_id,
        free_mb: PRESSURE_FREE_MB[slot].load(Ordering::Relaxed),
        safety_margin_mb: PRESSURE_MARGIN_MB[slot].load(Ordering::Relaxed),
        deficit_mb: PRESSURE_DEFICIT_MB[slot].load(Ordering::Relaxed),
    })
}

pub fn fatal_cuda_context_error(context: &str, err: &str) -> ! {
    log::error!(
        "Fatal CUDA context error in {}: {}. Exiting because the CUDA context is unsafe to continue.",
        context,
        err,
    );
    eprintln!(
        "\x1b[1;31mKRASIS FATAL: CUDA context error in {}: {}. Exiting; restart Krasis to recover.\x1b[0m",
        context,
        err,
    );
    #[cfg(unix)]
    unsafe {
        libc::_exit(CUDA_CONTEXT_POISONED_EXIT_CODE);
    }
    #[cfg(not(unix))]
    std::process::exit(CUDA_CONTEXT_POISONED_EXIT_CODE);
}

// CUDA runtime function signatures (resolved dynamically)
type CudaSetDeviceFn = unsafe extern "C" fn(i32) -> i32;
type CudaMemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
static CUDA_RUNTIME_FNS: OnceLock<(CudaSetDeviceFn, CudaMemGetInfoFn)> = OnceLock::new();

// Startup calibration needs to observe brief CUDA allocation low-waters at the
// scratch-release boundary. A millisecond polling cadence can miss those
// transients, so the model thread opens a calibration-only precision window and
// waits until this monitor thread is actively sampling before releasing scratch.
static PRECISION_WINDOWS_ENABLED: AtomicBool = AtomicBool::new(false);
static PRECISION_MONITOR_ACTIVE: AtomicBool = AtomicBool::new(false);
static PRECISION_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
static PRECISION_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static PRECISION_REQUEST_GENERATION: AtomicU64 = AtomicU64::new(0);
static PRECISION_READY_GENERATION: AtomicU64 = AtomicU64::new(0);
static PRECISION_FINISH_GENERATION: AtomicU64 = AtomicU64::new(0);
static PRECISION_DONE_GENERATION: AtomicU64 = AtomicU64::new(0);
static PRECISION_DEVICE: AtomicU64 = AtomicU64::new(0);
static PRECISION_MIN_FREE_BYTES: AtomicU64 = AtomicU64::new(u64::MAX);
static PRECISION_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
static PRECISION_QUERY_FAILED: AtomicBool = AtomicBool::new(false);
static PRECISION_READY_TIMESTAMP_MS: AtomicU64 = AtomicU64::new(0);
static PRECISION_DONE_TIMESTAMP_MS: AtomicU64 = AtomicU64::new(0);

const PRECISION_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

pub struct PrecisionVramWindow {
    generation: u64,
    active: bool,
}

impl PrecisionVramWindow {
    pub fn finish(mut self) -> Result<u64, String> {
        let result = self.finish_inner();
        self.active = false;
        result
    }

    fn finish_inner(&mut self) -> Result<u64, String> {
        PRECISION_FINISH_GENERATION.store(self.generation, Ordering::Release);
        let deadline = Instant::now() + PRECISION_HANDSHAKE_TIMEOUT;
        while PRECISION_DONE_GENERATION.load(Ordering::Acquire) != self.generation {
            if Instant::now() >= deadline {
                PRECISION_IN_FLIGHT.store(false, Ordering::Release);
                return Err(format!(
                    "VRAM precision monitor did not finish generation {}",
                    self.generation
                ));
            }
            thread::yield_now();
        }
        PRECISION_IN_FLIGHT.store(false, Ordering::Release);
        if PRECISION_QUERY_FAILED.load(Ordering::Acquire) {
            return Err(format!(
                "VRAM precision monitor CUDA query failed for generation {}",
                self.generation
            ));
        }
        let samples = PRECISION_SAMPLE_COUNT.load(Ordering::Acquire);
        let min_free_bytes = PRECISION_MIN_FREE_BYTES.load(Ordering::Acquire);
        if samples == 0 || min_free_bytes == u64::MAX {
            return Err(format!(
                "VRAM precision monitor captured no samples for generation {}",
                self.generation
            ));
        }
        append_precision_window_dump(
            self.generation,
            PRECISION_DEVICE.load(Ordering::Relaxed) as i32,
            PRECISION_READY_TIMESTAMP_MS.load(Ordering::Acquire),
            PRECISION_DONE_TIMESTAMP_MS.load(Ordering::Acquire),
            samples,
            min_free_bytes / (1024 * 1024),
        );
        Ok(min_free_bytes)
    }
}

impl Drop for PrecisionVramWindow {
    fn drop(&mut self) {
        if self.active {
            let _ = self.finish_inner();
        }
    }
}

pub fn begin_precision_vram_window(device_id: i32) -> Result<Option<PrecisionVramWindow>, String> {
    if !PRECISION_WINDOWS_ENABLED.load(Ordering::Acquire) {
        return Ok(None);
    }
    if device_id < 0 {
        return Err(format!(
            "VRAM precision monitor received invalid CUDA device {}",
            device_id
        ));
    }
    if !PRECISION_MONITOR_ACTIVE.load(Ordering::Acquire) {
        return Err("VRAM precision monitor is not active".to_string());
    }
    PRECISION_IN_FLIGHT
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| "VRAM precision monitor already has an active window".to_string())?;

    let generation = PRECISION_SEQUENCE.fetch_add(1, Ordering::AcqRel) + 1;
    PRECISION_DEVICE.store(device_id as u64, Ordering::Relaxed);
    PRECISION_MIN_FREE_BYTES.store(u64::MAX, Ordering::Relaxed);
    PRECISION_SAMPLE_COUNT.store(0, Ordering::Relaxed);
    PRECISION_QUERY_FAILED.store(false, Ordering::Relaxed);
    PRECISION_READY_TIMESTAMP_MS.store(0, Ordering::Relaxed);
    PRECISION_DONE_TIMESTAMP_MS.store(0, Ordering::Relaxed);
    PRECISION_REQUEST_GENERATION.store(generation, Ordering::Release);

    let deadline = Instant::now() + PRECISION_HANDSHAKE_TIMEOUT;
    while PRECISION_READY_GENERATION.load(Ordering::Acquire) != generation {
        if Instant::now() >= deadline {
            PRECISION_FINISH_GENERATION.store(generation, Ordering::Release);
            PRECISION_IN_FLIGHT.store(false, Ordering::Release);
            return Err(format!(
                "VRAM precision monitor did not acknowledge generation {}",
                generation
            ));
        }
        thread::yield_now();
    }
    Ok(Some(PrecisionVramWindow {
        generation,
        active: true,
    }))
}

/// Load cudaSetDevice + cudaMemGetInfo from the already-loaded libcudart.
/// Returns None if the library isn't loaded or symbols aren't found.
#[cfg(unix)]
fn load_cuda_fns() -> Option<(CudaSetDeviceFn, CudaMemGetInfoFn)> {
    if let Some(funcs) = CUDA_RUNTIME_FNS.get() {
        return Some(*funcs);
    }
    unsafe {
        // Try common names — RTLD_NOLOAD means "only find already-loaded lib"
        let lib_names: &[&[u8]] = &[
            b"libcudart.so\0",
            b"libcudart.so.12\0",
            b"libcudart.so.11\0",
        ];
        let mut lib = std::ptr::null_mut();
        for name in lib_names {
            lib = libc::dlopen(
                name.as_ptr() as *const libc::c_char,
                libc::RTLD_NOW | libc::RTLD_NOLOAD,
            );
            if !lib.is_null() {
                break;
            }
        }
        if lib.is_null() {
            return None;
        }
        let set_device = libc::dlsym(lib, b"cudaSetDevice\0".as_ptr() as *const libc::c_char);
        let mem_get_info = libc::dlsym(lib, b"cudaMemGetInfo\0".as_ptr() as *const libc::c_char);
        if set_device.is_null() || mem_get_info.is_null() {
            return None;
        }
        let funcs = (
            std::mem::transmute(set_device),
            std::mem::transmute(mem_get_info),
        );
        let _ = CUDA_RUNTIME_FNS.set(funcs);
        Some(funcs)
    }
}

/// Load cudaSetDevice + cudaMemGetInfo from the CUDA runtime DLL.
/// The Library is intentionally leaked after symbol resolution because the
/// returned function pointers must remain valid for the process lifetime.
#[cfg(windows)]
fn load_cuda_fns() -> Option<(CudaSetDeviceFn, CudaMemGetInfoFn)> {
    if let Some(funcs) = CUDA_RUNTIME_FNS.get() {
        return Some(*funcs);
    }
    let lib_names = ["cudart64_12.dll", "cudart64_110.dll"];
    for name in lib_names {
        let Ok(lib) = (unsafe { libloading::Library::new(name) }) else {
            continue;
        };
        let funcs = unsafe {
            let set_device = lib.get::<CudaSetDeviceFn>(b"cudaSetDevice\0").ok()?;
            let mem_get_info = lib.get::<CudaMemGetInfoFn>(b"cudaMemGetInfo\0").ok()?;
            (*set_device, *mem_get_info)
        };
        std::mem::forget(lib);
        let _ = CUDA_RUNTIME_FNS.set(funcs);
        return Some(funcs);
    }
    None
}

/// Query free VRAM in bytes for a specific device.
fn query_free_bytes(
    set_device: CudaSetDeviceFn,
    mem_get_info: CudaMemGetInfoFn,
    device_id: i32,
) -> Option<usize> {
    unsafe {
        if (set_device)(device_id) != 0 {
            return None;
        }
        let mut free: usize = 0;
        let mut total: usize = 0;
        if (mem_get_info)(&mut free, &mut total) != 0 {
            return None;
        }
        Some(free)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VRAM Report — global state for recording time-series + events
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct ReportEntry {
    timestamp_ms: u64,
    event: String, // empty for periodic samples
    gpu_free_mb: Vec<u64>,
}

struct VramReportState {
    start_time: Instant,
    entries: Vec<ReportEntry>,
    device_ids: Vec<i32>,
}

static REPORT: Mutex<Option<VramReportState>> = Mutex::new(None);

#[derive(Clone)]
struct ActiveRequestVram {
    context: String,
    lows_mb: Vec<(i32, u64)>,
}

static CURRENT_EVENT: Mutex<String> = Mutex::new(String::new());
static ACTIVE_REQUEST_VRAM: Mutex<Option<ActiveRequestVram>> = Mutex::new(None);
static DEVICE_UUIDS: Mutex<Vec<(i32, String)>> = Mutex::new(Vec::new());

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn current_event() -> String {
    CURRENT_EVENT.lock().unwrap().clone()
}

fn current_request_context() -> String {
    ACTIVE_REQUEST_VRAM
        .lock()
        .unwrap()
        .as_ref()
        .map(|ctx| ctx.context.clone())
        .unwrap_or_default()
}

/// Update the lifecycle label captured by any subsequent low-water dump
/// without forcing an extra CUDA query. This keeps the production-disabled
/// path cheap while allowing the opt-in VRAM report to bracket request
/// teardown and worker-idle transitions precisely.
pub fn set_lifecycle_event(event: &str) {
    if let Ok(mut current) = CURRENT_EVENT.lock() {
        *current = event.to_string();
    }
}

fn format_cuda_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

fn cache_current_cuda_device_uuid(logical_device_id: i32) -> Option<String> {
    let mut context_device = 0i32;
    if unsafe { cuda_sys::lib().cuCtxGetDevice(&mut context_device) }
        != cuda_sys::CUresult::CUDA_SUCCESS
    {
        return None;
    }
    let mut uuid = cuda_sys::CUuuid::default();
    if unsafe { cuda_sys::lib().cuDeviceGetUuid_v2(&mut uuid, context_device) }
        != cuda_sys::CUresult::CUDA_SUCCESS
    {
        return None;
    }
    let bytes: [u8; 16] = uuid.bytes.map(|value| value as u8);
    let formatted = format_cuda_uuid(&bytes);
    let mut cache = DEVICE_UUIDS.lock().unwrap();
    if let Some((_, existing)) = cache
        .iter_mut()
        .find(|(device_id, _)| *device_id == logical_device_id)
    {
        *existing = formatted.clone();
    } else {
        cache.push((logical_device_id, formatted.clone()));
    }
    Some(formatted)
}

fn cached_cuda_device_uuid(device_id: i32) -> Option<String> {
    DEVICE_UUIDS
        .lock()
        .unwrap()
        .iter()
        .find(|(cached_device_id, _)| *cached_device_id == device_id)
        .map(|(_, uuid)| uuid.clone())
}

fn update_active_request_low(device_id: i32, free_mb: u64) -> bool {
    let mut active = ACTIVE_REQUEST_VRAM.lock().unwrap();
    let Some(ref mut ctx) = *active else {
        return false;
    };
    if let Some((_, existing)) = ctx
        .lows_mb
        .iter_mut()
        .find(|(existing_device, _)| *existing_device == device_id)
    {
        if free_mb < *existing {
            *existing = free_mb;
        }
    } else {
        ctx.lows_mb.push((device_id, free_mb));
    }
    true
}

fn append_vram_dump(
    file_name: &str,
    kind: &str,
    device_id: i32,
    free_mb: u64,
    total_mb: Option<u64>,
    safety_margin_mb: u64,
    deficit_mb: u64,
    poll_interval_ms: Option<u64>,
) {
    let Ok(run_dir) = std::env::var("KRASIS_RUN_DIR") else {
        return;
    };
    let path = std::path::Path::new(&run_dir).join(file_name);
    let event = current_event();
    let request_context = current_request_context();
    let device_uuid = cached_cuda_device_uuid(device_id).unwrap_or_default();
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_string();
    let sample_scope = if request_context.is_empty() {
        "global"
    } else {
        "active_request"
    };
    let line = format!(
        "{{\"timestamp_ms\":{},\"kind\":\"{}\",\"pid\":{},\"thread\":\"{}\",\"sample_scope\":\"{}\",\"device\":{},\"device_uuid\":\"{}\",\"free_mb\":{},\"total_mb\":{},\"poll_interval_ms\":{},\"safety_margin_mb\":{},\"deficit_mb\":{},\"hard_exit_floor_mb\":{},\"current_event\":\"{}\",\"request_context\":\"{}\"}}\n",
        now_millis(),
        json_escape(kind),
        std::process::id(),
        json_escape(&thread_name),
        sample_scope,
        device_id,
        json_escape(&device_uuid),
        free_mb,
        total_mb.map_or_else(|| "null".to_string(), |value| value.to_string()),
        poll_interval_ms.map_or_else(|| "null".to_string(), |value| value.to_string()),
        safety_margin_mb,
        deficit_mb,
        VRAM_HARD_EXIT_FLOOR_MB,
        json_escape(&event),
        json_escape(&request_context),
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
        let _ = f.flush();
    }
}

fn append_precision_window_dump(
    generation: u64,
    device_id: i32,
    started_at_ms: u64,
    completed_at_ms: u64,
    samples: u64,
    min_free_mb: u64,
) {
    if std::env::var_os("KRASIS_VRAM_LEDGER").is_none() {
        return;
    }
    let Ok(run_dir) = std::env::var("KRASIS_RUN_DIR") else {
        return;
    };
    let path = std::path::Path::new(&run_dir).join("vram-precision-windows.log");
    let device_uuid = cached_cuda_device_uuid(device_id).unwrap_or_default();
    let line = format!(
        "{{\"generation\":{},\"device\":{},\"device_uuid\":\"{}\",\"source\":\"cuda_runtime_monitor\",\"started_at_ms\":{},\"completed_at_ms\":{},\"duration_ms\":{},\"samples\":{},\"min_free_mb\":{},\"current_event\":\"{}\"}}\n",
        generation,
        device_id,
        json_escape(&device_uuid),
        started_at_ms,
        completed_at_ms,
        completed_at_ms.saturating_sub(started_at_ms),
        samples,
        min_free_mb,
        json_escape(&current_event()),
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
        let _ = f.flush();
    }
}

fn append_safety_limit_dump(
    kind: &str,
    device_id: i32,
    free_mb: u64,
    total_mb: Option<u64>,
    safety_margin_mb: u64,
    deficit_mb: u64,
    poll_interval_ms: Option<u64>,
) {
    append_vram_dump(
        "below-vram-safety-limit.log",
        kind,
        device_id,
        free_mb,
        total_mb,
        safety_margin_mb,
        deficit_mb,
        poll_interval_ms,
    );
}

pub fn begin_request_context(context: &str) {
    let mut active = ACTIVE_REQUEST_VRAM.lock().unwrap();
    *active = Some(ActiveRequestVram {
        context: context.to_string(),
        lows_mb: Vec::new(),
    });
}

pub fn update_request_context(context: &str) {
    let mut active = ACTIVE_REQUEST_VRAM.lock().unwrap();
    if let Some(ref mut ctx) = *active {
        ctx.context = context.to_string();
    }
}

pub fn reset_request_lows() {
    let mut active = ACTIVE_REQUEST_VRAM.lock().unwrap();
    if let Some(ref mut ctx) = *active {
        ctx.lows_mb.clear();
    }
}

pub fn current_request_lows() -> Vec<(i32, u64)> {
    ACTIVE_REQUEST_VRAM
        .lock()
        .unwrap()
        .as_ref()
        .map(|ctx| ctx.lows_mb.clone())
        .unwrap_or_default()
}

pub fn end_request_context() -> Option<(String, Vec<(i32, u64)>)> {
    ACTIVE_REQUEST_VRAM
        .lock()
        .unwrap()
        .take()
        .map(|ctx| (ctx.context, ctx.lows_mb))
}

pub fn record_request_lows_below_safety(
    context: &str,
    lows_mb: &[(i32, u64)],
    safety_margin_mb: u64,
) {
    if safety_margin_mb == 0 {
        return;
    }
    for (device_id, free_mb) in lows_mb {
        if *free_mb < safety_margin_mb {
            {
                let mut active = ACTIVE_REQUEST_VRAM.lock().unwrap();
                *active = Some(ActiveRequestVram {
                    context: context.to_string(),
                    lows_mb: lows_mb.to_vec(),
                });
            }
            append_safety_limit_dump(
                "request_low_water_below_safety",
                *device_id,
                *free_mb,
                None,
                safety_margin_mb,
                safety_margin_mb.saturating_sub(*free_mb),
                None,
            );
            let mut active = ACTIVE_REQUEST_VRAM.lock().unwrap();
            *active = None;
        }
    }
}

/// Enable VRAM reporting. Called once at startup.
pub fn report_enable(device_ids: Vec<i32>) {
    let mut report = REPORT.lock().unwrap();
    *report = Some(VramReportState {
        start_time: Instant::now(),
        entries: Vec::with_capacity(10000),
        device_ids,
    });
}

/// Check if reporting is enabled.
pub fn report_is_enabled() -> bool {
    REPORT.lock().unwrap().is_some()
}

/// Record a periodic VRAM sample (called from poll thread, no CUDA query needed).
fn report_sample(gpu_free_mb: Vec<u64>) {
    let mut report = REPORT.lock().unwrap();
    if let Some(ref mut state) = *report {
        let ts = state.start_time.elapsed().as_millis() as u64;
        state.entries.push(ReportEntry {
            timestamp_ms: ts,
            event: String::new(),
            gpu_free_mb,
        });
    }
}

/// Query free VRAM (in MB) for all report devices.
fn query_all_free_mb(device_ids: &[i32]) -> Vec<u64> {
    let Some((set_device, mem_get_info)) = load_cuda_fns() else {
        return vec![0; device_ids.len()];
    };
    device_ids
        .iter()
        .map(|&id| {
            query_free_bytes(set_device, mem_get_info, id)
                .map(|f| (f as u64) / (1024 * 1024))
                .unwrap_or(0)
        })
        .collect()
}

/// Record a named event with current VRAM snapshot.
/// No-op if reporting is not enabled.
pub fn report_event(event: &str) {
    if let Ok(mut current) = CURRENT_EVENT.lock() {
        *current = event.to_string();
    }

    // Peek at device_ids without holding lock during CUDA query
    let device_ids = {
        let report = REPORT.lock().unwrap();
        match *report {
            Some(ref state) => state.device_ids.clone(),
            None => return,
        }
    };

    let gpu_free_mb = query_all_free_mb(&device_ids);

    let mut report = REPORT.lock().unwrap();
    if let Some(ref mut state) = *report {
        let ts = state.start_time.elapsed().as_millis() as u64;
        state.entries.push(ReportEntry {
            timestamp_ms: ts,
            event: event.to_string(),
            gpu_free_mb,
        });
    }
}

/// Write VRAM report CSV to file.
pub fn report_write(path: &str) -> std::io::Result<()> {
    let report = REPORT.lock().unwrap();
    let Some(ref state) = *report else {
        return Ok(());
    };

    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    // Header
    write!(f, "timestamp_ms,event")?;
    for &id in &state.device_ids {
        write!(f, ",gpu{}_free_mb", id)?;
    }
    writeln!(f)?;

    // Data
    for entry in &state.entries {
        write!(f, "{},{}", entry.timestamp_ms, entry.event)?;
        for &mb in &entry.gpu_free_mb {
            write!(f, ",{}", mb)?;
        }
        writeln!(f)?;
    }

    Ok(())
}

/// Get summary of key events as list of (event, timestamp_ms, [gpu_free_mb, ...]).
pub fn report_summary() -> Vec<(String, u64, Vec<u64>)> {
    let report = REPORT.lock().unwrap();
    let Some(ref state) = *report else {
        return vec![];
    };

    state
        .entries
        .iter()
        .filter(|e| !e.event.is_empty())
        .map(|e| (e.event.clone(), e.timestamp_ms, e.gpu_free_mb.clone()))
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VramMonitor — background polling + min-free tracking
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct DeviceState {
    device_id: i32,
    total_bytes: AtomicU64,
    min_free_bytes: AtomicU64,
}

/// Background VRAM monitor that polls cudaMemGetInfo and tracks peak usage.
///
/// Usage:
///   monitor = VramMonitor([0, 1], poll_interval_ms=50, safety_margin_mb=600)
///   monitor.start()
///   # ... run prefill + decode warmup ...
///   min_free = monitor.min_free_mb(0)  # measured min free during warmup
///   hcs_budget = min_free - 600
///   # ... load HCS experts ...
///   monitor.enable_warnings()          # warns on each new low below margin
///   # ... run server (monitor stays on, no reset needed) ...
///   monitor.stop()
#[pyclass]
pub struct VramMonitor {
    devices: Arc<Vec<DeviceState>>,
    running: Arc<AtomicBool>,
    warn_enabled: Arc<AtomicBool>,
    safety_margin_bytes: Arc<AtomicU64>,
    poll_interval_ms: Arc<AtomicU64>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl VramMonitor {
    #[new]
    #[pyo3(signature = (device_indices, poll_interval_ms=50, safety_margin_mb=600))]
    fn new(
        device_indices: Vec<i32>,
        poll_interval_ms: u64,
        safety_margin_mb: u64,
    ) -> PyResult<Self> {
        if !(1..=1000).contains(&poll_interval_ms) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "VRAM monitor poll interval must be between 1 and 1000 milliseconds",
            ));
        }
        let devices: Vec<DeviceState> = device_indices
            .into_iter()
            .map(|id| DeviceState {
                device_id: id,
                total_bytes: AtomicU64::new(0),
                min_free_bytes: AtomicU64::new(u64::MAX),
            })
            .collect();

        Ok(Self {
            devices: Arc::new(devices),
            running: Arc::new(AtomicBool::new(false)),
            warn_enabled: Arc::new(AtomicBool::new(false)),
            safety_margin_bytes: Arc::new(AtomicU64::new(safety_margin_mb * 1024 * 1024)),
            poll_interval_ms: Arc::new(AtomicU64::new(poll_interval_ms)),
            thread_handle: None,
        })
    }

    /// Start the background monitoring thread.
    fn start(&mut self) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Verify CUDA is available before spawning thread
        let _ = load_cuda_fns().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "VRAM monitor: failed to load CUDA runtime (libcudart.so not loaded)",
            )
        })?;

        PRECISION_WINDOWS_ENABLED.store(false, Ordering::Release);
        PRECISION_MONITOR_ACTIVE.store(false, Ordering::Release);
        PRECISION_IN_FLIGHT.store(false, Ordering::Release);
        PRECISION_REQUEST_GENERATION.store(0, Ordering::Release);
        PRECISION_READY_GENERATION.store(0, Ordering::Release);
        PRECISION_FINISH_GENERATION.store(0, Ordering::Release);
        PRECISION_DONE_GENERATION.store(0, Ordering::Release);
        PRECISION_READY_TIMESTAMP_MS.store(0, Ordering::Release);
        PRECISION_DONE_TIMESTAMP_MS.store(0, Ordering::Release);
        self.running.store(true, Ordering::Release);

        let devices = self.devices.clone();
        let running = self.running.clone();
        let warn_enabled = self.warn_enabled.clone();
        let safety_margin = self.safety_margin_bytes.clone();
        let poll_interval_ms = self.poll_interval_ms.clone();

        let handle = thread::Builder::new()
            .name("vram-monitor".into())
            .spawn(move || {
                // Load CUDA fns on this thread (avoids Send issues with fn pointers)
                let Some((set_device, mem_get_info)) = load_cuda_fns() else {
                    log::error!("VRAM monitor thread: failed to load CUDA runtime");
                    return;
                };

                // Initial reading to populate total_bytes
                for dev in devices.iter() {
                    unsafe {
                        if (set_device)(dev.device_id) == 0 {
                            let mut free: usize = 0;
                            let mut total: usize = 0;
                            if (mem_get_info)(&mut free, &mut total) == 0 {
                                dev.total_bytes.store(total as u64, Ordering::Relaxed);
                                dev.min_free_bytes.store(free as u64, Ordering::Relaxed);
                                match cache_current_cuda_device_uuid(dev.device_id) {
                                    Some(uuid) => log::info!(
                                        "VRAM monitor logical cuda:{} is physical {}",
                                        dev.device_id,
                                        uuid,
                                    ),
                                    None => log::warn!(
                                        "VRAM monitor could not resolve physical UUID for logical cuda:{}",
                                        dev.device_id,
                                    ),
                                }
                            }
                        }
                    }
                }

                log::info!(
                    "VRAM monitor started: {} device(s), poll interval {}ms",
                    devices.len(),
                    poll_interval_ms.load(Ordering::Relaxed),
                );

                PRECISION_MONITOR_ACTIVE.store(true, Ordering::Release);

                let mut report_elapsed_ms = 0u64;
                let mut handled_precision_generation = 0u64;

                while running.load(Ordering::Acquire) {
                    let mut readings = Vec::with_capacity(devices.len());

                    for dev in devices.iter() {
                        if let Some(free) = query_free_bytes(set_device, mem_get_info, dev.device_id)
                        {
                            let free_u64 = free as u64;
                            let free_mb = free_u64 / (1024 * 1024);
                            readings.push(free_mb);
                            update_active_request_low(dev.device_id, free_mb);

                            if warn_enabled.load(Ordering::Relaxed) {
                                let margin = safety_margin.load(Ordering::Relaxed);
                                if free_u64 < margin {
                                    mark_pressure(dev.device_id, free_mb, margin / (1024 * 1024));
                                }
                                // Do not clear pressure just because idle free recovered.
                                // A transient below-safety low is evidence that the next
                                // decode step needs more idle headroom; the HCS drain path
                                // clears this after it has reacted to the recorded deficit.
                            }

                            let prev_min = dev.min_free_bytes.load(Ordering::Relaxed);

                            if free_u64 < prev_min {
                                dev.min_free_bytes.store(free_u64, Ordering::Relaxed);

                                // The opt-in ledger preserves every observed new low
                                // separately from the safety-violation log. This lets
                                // calibration diagnostics attribute a safe-but-close
                                // transient to a request phase without changing the
                                // configured margin or adding CUDA calls to the model
                                // thread.
                                if std::env::var_os("KRASIS_VRAM_LEDGER").is_some() {
                                    let margin_mb = safety_margin.load(Ordering::Relaxed)
                                        / (1024 * 1024);
                                    append_vram_dump(
                                        "vram-low-water-events.log",
                                        "new_low_water",
                                        dev.device_id,
                                        free_mb,
                                        Some(
                                            dev.total_bytes.load(Ordering::Relaxed)
                                                / (1024 * 1024),
                                        ),
                                        margin_mb,
                                        margin_mb.saturating_sub(free_mb),
                                        Some(poll_interval_ms.load(Ordering::Relaxed)),
                                    );
                                }

                                if free_u64 < VRAM_HARD_EXIT_FLOOR_MB * 1024 * 1024 {
                                    let margin_mb =
                                        safety_margin.load(Ordering::Relaxed) / (1024 * 1024);
                                    append_safety_limit_dump(
                                        "critical_low_floor",
                                        dev.device_id,
                                        free_mb,
                                        Some(dev.total_bytes.load(Ordering::Relaxed) / (1024 * 1024)),
                                        margin_mb,
                                        margin_mb.saturating_sub(free_mb),
                                        Some(poll_interval_ms.load(Ordering::Relaxed)),
                                    );
                                    eprintln!(
                                        "\x1b[1;31mVRAM MONITOR: cuda:{} free VRAM dropped to {} MB, below critical floor {} MB. Marking pressure for immediate HCS drain.\x1b[0m",
                                        dev.device_id,
                                        free_mb,
                                        VRAM_HARD_EXIT_FLOOR_MB,
                                    );
                                }

                                // Warn on new lows below safety margin (when enabled).
                                // Pressure state is updated every poll above; warning output
                                // remains new-low only to avoid log spam.
                                if warn_enabled.load(Ordering::Relaxed) {
                                    let margin = safety_margin.load(Ordering::Relaxed);
                                    if free_u64 < margin {
                                        let margin_mb = margin / (1024 * 1024);
                                        let deficit_mb = margin_mb.saturating_sub(free_mb);
                                        append_safety_limit_dump(
                                            "below_safety_margin",
                                            dev.device_id,
                                            free_mb,
                                            Some(dev.total_bytes.load(Ordering::Relaxed) / (1024 * 1024)),
                                            margin_mb,
                                            deficit_mb,
                                            Some(poll_interval_ms.load(Ordering::Relaxed)),
                                        );
                                        eprintln!(
                                            "\x1b[1;33m⚠ VRAM MONITOR: new low on cuda:{} — \
                                             {} MB free (safety margin: {} MB, deficit: {} MB)\x1b[0m",
                                            dev.device_id,
                                            free_mb,
                                            margin_mb,
                                            deficit_mb,
                                        );
                                    }
                                }
                            }
                        } else {
                            readings.push(0);
                        }
                    }

                    let requested = PRECISION_REQUEST_GENERATION.load(Ordering::Acquire);
                    if requested != 0 && requested != handled_precision_generation {
                        let requested_device = PRECISION_DEVICE.load(Ordering::Relaxed) as i32;
                        let target = devices
                            .iter()
                            .find(|dev| dev.device_id == requested_device);
                        let capture_sample = |dev: &DeviceState| {
                            match query_free_bytes(set_device, mem_get_info, dev.device_id) {
                                Some(free) => {
                                    let free_u64 = free as u64;
                                    dev.min_free_bytes.fetch_min(free_u64, Ordering::Relaxed);
                                    PRECISION_MIN_FREE_BYTES.fetch_min(free_u64, Ordering::Relaxed);
                                    PRECISION_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed);
                                }
                                None => {
                                    PRECISION_QUERY_FAILED.store(true, Ordering::Release);
                                }
                            }
                        };
                        if let Some(dev) = target {
                            // Prove that this thread has entered the exact-device Runtime
                            // sampling loop before releasing the model thread to launch.
                            capture_sample(dev);
                        } else {
                            PRECISION_QUERY_FAILED.store(true, Ordering::Release);
                        }
                        PRECISION_READY_TIMESTAMP_MS
                            .store(now_millis() as u64, Ordering::Release);
                        PRECISION_READY_GENERATION.store(requested, Ordering::Release);
                        while running.load(Ordering::Acquire)
                            && PRECISION_FINISH_GENERATION.load(Ordering::Acquire) != requested
                        {
                            if let Some(dev) = target {
                                capture_sample(dev);
                            }
                            std::hint::spin_loop();
                        }
                        if !running.load(Ordering::Acquire) {
                            PRECISION_QUERY_FAILED.store(true, Ordering::Release);
                        }
                        handled_precision_generation = requested;
                        PRECISION_DONE_TIMESTAMP_MS
                            .store(now_millis() as u64, Ordering::Release);
                        PRECISION_DONE_GENERATION.store(requested, Ordering::Release);
                    }

                    // Record periodic sample for VRAM report (every ~200ms)
                    let poll_ms = poll_interval_ms.load(Ordering::Relaxed).clamp(1, 1000);
                    report_elapsed_ms = report_elapsed_ms.saturating_add(poll_ms);
                    if report_elapsed_ms >= 200 {
                        report_sample(readings);
                        report_elapsed_ms %= 200;
                    }
                    thread::sleep(Duration::from_millis(poll_ms));
                }

                PRECISION_MONITOR_ACTIVE.store(false, Ordering::Release);
                log::info!("VRAM monitor stopped");
            })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to spawn VRAM monitor thread: {}",
                    e
                ))
            })?;

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Stop the background monitoring thread.
    fn stop(&mut self) {
        PRECISION_WINDOWS_ENABLED.store(false, Ordering::Release);
        self.running.store(false, Ordering::Release);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Get the minimum free VRAM observed on a device (in MB).
    fn min_free_mb(&self, device_index: i32) -> u64 {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                let bytes = dev.min_free_bytes.load(Ordering::Relaxed);
                if bytes == u64::MAX {
                    return 0;
                }
                return bytes / (1024 * 1024);
            }
        }
        0
    }

    /// Get the total VRAM on a device (in MB).
    fn total_mb(&self, device_index: i32) -> u64 {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                return dev.total_bytes.load(Ordering::Relaxed) / (1024 * 1024);
            }
        }
        0
    }

    /// Get the peak VRAM used on a device (in MB) = total - min_free.
    fn peak_used_mb(&self, device_index: i32) -> u64 {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                let total = dev.total_bytes.load(Ordering::Relaxed);
                let min_free = dev.min_free_bytes.load(Ordering::Relaxed);
                if min_free == u64::MAX || min_free > total {
                    return 0;
                }
                return (total - min_free) / (1024 * 1024);
            }
        }
        0
    }

    /// Reset min-free tracking for a single device (used by 4-point VRAM calibration).
    fn reset(&self, device_index: i32) {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                dev.min_free_bytes.store(u64::MAX, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Reset min-free tracking on all devices (e.g. after warmup, before runtime).
    fn reset_min_free(&self) {
        for dev in self.devices.iter() {
            dev.min_free_bytes.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Get current free VRAM on a device (in MB) — live reading, not tracked min.
    fn current_free_mb(&self, device_index: i32) -> u64 {
        let Some((set_device, mem_get_info)) = load_cuda_fns() else {
            return 0;
        };
        match query_free_bytes(set_device, mem_get_info, device_index) {
            Some(free) => (free as u64) / (1024 * 1024),
            None => 0,
        }
    }

    /// Enable runtime warnings when free VRAM drops below safety margin.
    /// Resets min-free tracking so the next poll immediately captures current state
    /// and warns if already below the margin (e.g. right after HCS allocation).
    fn enable_warnings(&self) {
        // Reset min_free so the very next poll sets a fresh baseline.
        // If post-HCS free is already below safety margin, the first poll
        // will detect it as a "new low" and fire the warning immediately.
        for dev in self.devices.iter() {
            dev.min_free_bytes.store(u64::MAX, Ordering::Relaxed);
        }
        self.warn_enabled.store(true, Ordering::Release);
    }

    /// Disable runtime warnings.
    fn disable_warnings(&self) {
        self.warn_enabled.store(false, Ordering::Release);
    }

    /// Update safety margin (in MB).
    fn set_safety_margin_mb(&self, margin_mb: u64) {
        self.safety_margin_bytes
            .store(margin_mb * 1024 * 1024, Ordering::Relaxed);
    }

    /// Change the polling cadence and return the previous value. Startup
    /// calibration uses a high-resolution cadence, then restores the normal
    /// runtime cadence before serving requests or collecting speed evidence.
    fn set_poll_interval_ms(&self, poll_interval_ms: u64) -> PyResult<u64> {
        if !(1..=1000).contains(&poll_interval_ms) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "VRAM monitor poll interval must be between 1 and 1000 milliseconds",
            ));
        }
        Ok(self
            .poll_interval_ms
            .swap(poll_interval_ms, Ordering::AcqRel))
    }

    /// Enable exact scratch-release allocation windows for startup calibration only.
    fn enable_precision_windows(&self) -> PyResult<()> {
        if !self.running.load(Ordering::Acquire)
            || !PRECISION_MONITOR_ACTIVE.load(Ordering::Acquire)
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "VRAM precision monitor is not active",
            ));
        }
        PRECISION_WINDOWS_ENABLED.store(true, Ordering::Release);
        Ok(())
    }

    /// Disable startup-only exact allocation windows before normal serving.
    fn disable_precision_windows(&self) -> PyResult<()> {
        if PRECISION_IN_FLIGHT.load(Ordering::Acquire) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot disable VRAM precision windows while one is active",
            ));
        }
        PRECISION_WINDOWS_ENABLED.store(false, Ordering::Release);
        Ok(())
    }

    // ── VRAM Report methods ──

    /// Enable VRAM reporting. Periodic samples (~200ms) and named events
    /// are recorded to an in-memory buffer. Call write_report() to flush to CSV.
    fn enable_report(&self) {
        let device_ids: Vec<i32> = self.devices.iter().map(|d| d.device_id).collect();
        report_enable(device_ids);
    }

    /// Log a named event with current VRAM snapshot. No-op if report not enabled.
    fn report_event(&self, event: &str) {
        crate::vram_monitor::report_event(event);
    }

    /// Write VRAM report CSV to file. Contains periodic samples and events.
    fn write_report(&self, path: &str) -> PyResult<()> {
        report_write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write VRAM report: {}", e))
        })
    }

    /// Get summary: list of (event_name, timestamp_ms, [gpu_free_mb, ...]) for all events.
    fn report_summary(&self) -> Vec<(String, u64, Vec<u64>)> {
        crate::vram_monitor::report_summary()
    }
}

impl Drop for VramMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::format_cuda_uuid;

    #[test]
    fn cuda_uuid_matches_nvidia_canonical_format() {
        let bytes = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
            0xdc, 0xfe,
        ];
        assert_eq!(
            format_cuda_uuid(&bytes),
            "GPU-01234567-89ab-cdef-1032-547698badcfe"
        );
    }
}
