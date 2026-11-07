//! Bounded, observation-only fixed-history route journal. No GPU calls or
//! per-layer filesystem writes. Export only after the decode timer ends.
use serde::Serialize;
use std::{
    cell::RefCell,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    },
    time::Instant,
};
static ACTIVE: AtomicBool = AtomicBool::new(false);
thread_local! {static JOURNAL:RefCell<Option<Journal>>=const{RefCell::new(None)};}
#[derive(Serialize)]
pub(crate) struct Cold {
    pub slot: usize,
    pub expert: usize,
    pub weight: f32,
    pub component_bytes: usize,
    pub allocation_bytes: usize,
}
#[derive(Serialize)]
struct Route {
    position: usize,
    layer: usize,
    ids: Vec<i32>,
    weights: Vec<f32>,
    hot_count: usize,
    cold: Vec<Cold>,
    host_us: u128,
}
#[derive(Serialize)]
struct Forward {
    position: usize,
    input: usize,
    path: &'static str,
    begin_us: u128,
}
struct Journal {
    label: String,
    streamed_kv_initial:Vec<serde_json::Value>,
    start: Instant,
    epoch_monotonic_ns: Option<u128>,
    routes: Vec<Route>,
    forwards: Vec<Forward>,
}
pub(crate) struct Scope {
    enabled: bool,
}
fn enabled() -> Result<bool, String> {
    static VALUE: OnceLock<Result<bool, String>> = OnceLock::new();
    VALUE
        .get_or_init(|| match std::env::var("KRASIS_DEFERRED_ROUTE_AUDIT") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(v) => match v.as_str() {
                "0" => Ok(false),
                "1" => Ok(true),
                _ => Err("KRASIS_DEFERRED_ROUTE_AUDIT must be0 or1".into()),
            },
            Err(e) => Err(e.to_string()),
        })
        .clone()
}
pub(crate) fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}
fn monotonic_ns() -> Option<u128> {
    #[cfg(target_os = "linux")]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } == 0 {
            return Some(ts.tv_sec as u128 * 1_000_000_000 + ts.tv_nsec as u128);
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
impl Scope {
    pub fn begin(label: Option<&str>, deferred: bool) -> Result<Self, String> {
        if !enabled()? || label.is_none() {
            return Ok(Self { enabled: false });
        }
        if !deferred {
            return Err("route audit requires deferred teacher receipts; numerical proof writes are not a timing path".into());
        }
        let streamed_kv_initial=crate::streamed_kv::diagnostic_snapshot()?;
        ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "route audit supports one diagnostic invocation at a time")?;
        JOURNAL.with(|j| {
            *j.borrow_mut() = Some(Journal {
                label: label.unwrap().to_string(),
                streamed_kv_initial,
                start: Instant::now(),
                epoch_monotonic_ns: monotonic_ns(),
                routes: Vec::with_capacity(4096),
                forwards: Vec::with_capacity(256),
            })
        });
        Ok(Self { enabled: true })
    }
    pub fn finish(&mut self, directory: &Path) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        let data = JOURNAL
            .with(|j| j.borrow_mut().take())
            .ok_or("missing route journal")?;
        ACTIVE.store(false, Ordering::Release);
        self.enabled = false;
        let bytes=serde_json::to_vec(&serde_json::json!({"diagnostic_only":true,"qualifies_candidate":false,"observation_only":true,
   "label":data.label,"clock":"monotonic microseconds since journal begin","epoch_monotonic_ns":data.epoch_monotonic_ns,"route_scope":"actual graph-replay host-visible classification; eager forward has no per-layer route journal",
   "cold_scope":"initial demand classification before CPU-tail resolution; reconcile actual saved and DMA bytes separately",
   "forwards":data.forwards,"routes":data.routes,"streamed_kv_initial":data.streamed_kv_initial,"streamed_kv_cumulative":crate::streamed_kv::diagnostic_snapshot()?})).map_err(|e|e.to_string())?;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("deferred-routes.json"))
            .map_err(|e| e.to_string())?;
        f.write_all(&bytes)
            .and_then(|_| f.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::File::open(directory)
            .and_then(|p| p.sync_all())
            .map_err(|e| e.to_string())
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if self.enabled {
            JOURNAL.with(|j| j.borrow_mut().take());
            ACTIVE.store(false, Ordering::Release);
        }
    }
}
pub(crate) fn forward(position: usize, input: usize, path: &'static str) -> Result<(), String> {
    if !active() {
        return Ok(());
    }
    JOURNAL.with(|j| {
        let mut j = j.borrow_mut();
        let j = j
            .as_mut()
            .ok_or("route journal is on a different inference thread")?;
        if j.forwards.len() >= 4096 {
            return Err("route journal forward bound exceeded".into());
        }
        j.forwards.push(Forward {
            position,
            input,
            path,
            begin_us: j.start.elapsed().as_micros(),
        });
        Ok(())
    })
}
pub(crate) fn record(
    position: usize,
    layer: usize,
    ids: &[i32],
    weights: &[f32],
    hot_count: usize,
    cold: Vec<Cold>,
) -> Result<(), String> {
    if !active() {
        return Ok(());
    }
    if ids.len() != weights.len() || ids.len() > 512 || weights.iter().any(|x| !x.is_finite()) {
        return Err("invalid observed route dimensions/weights".into());
    }
    JOURNAL.with(|j| {
        let mut j = j.borrow_mut();
        let j = j
            .as_mut()
            .ok_or("route journal is on a different inference thread")?;
        if j.routes.len() >= 65536 {
            return Err("route journal row bound exceeded".into());
        }
        j.routes.push(Route {
            position,
            layer,
            ids: ids.to_vec(),
            weights: weights.to_vec(),
            hot_count,
            cold,
            host_us: j.start.elapsed().as_micros(),
        });
        Ok(())
    })
}
