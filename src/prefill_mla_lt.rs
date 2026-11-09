//! Bounded cold selection for the existing gathered FP32 attention operands.
//! Hot enqueue retains prepared plans and allocation owners without allocation,
//! synchronization, heuristic calls or Python. Unselected shapes use cuBLAS.
use crate::stable_mla_gemm::{Candidates, Preparation, Prepared, Shape, Span};
use cudarc::driver::{sys, CudaDevice};
use serde_json::{json, Value};
use std::{collections::HashMap, ffi::c_void, sync::Arc};

pub(crate) const WORKSPACE_CAP: usize = 16 << 20;
#[derive(Default)]
pub(crate) struct Cache {
    plans: HashMap<(Shape, bool), Option<Prepared>>,
    reports: Vec<Value>,
    sealed: bool,
}
impl Cache {
    pub(crate) fn finalize(&mut self) {
        self.sealed = true;
    }
    pub(crate) fn identity(&self) -> Value {
        let mut plans:Vec<_>=self.plans.iter().map(|((shape,reassociated),plan)| {
            // Keep the existing exact-only identity byte-compatible. The new
            // treatment is explicit only when its additional plan is prepared.
            let mut entry=json!({"shape":format!("{shape:?}"),"plan":plan.as_ref().map(|p|p.identity.as_str())});
            if *reassociated { entry["reassociated"]=json!(true); }
            entry
        }).collect();
        plans.sort_by_key(|p| (p["shape"].as_str().unwrap_or("").to_owned(),p["reassociated"].as_bool()));
        json!({"format":"prefill_mla_f32_v1","sealed":self.sealed,"plans":plans})
    }
    /// The caller owns every real span until main-stream completion, including
    /// errors. Calibration is admitted only on representative full selections.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &mut self,
        device: Arc<CudaDevice>,
        shape: Shape,
        a: &Span<'_>,
        b: &Span<'_>,
        output: &Span<'_>,
        workspace: &Span<'_>,
        alpha: f32,
        beta: f32,
        stream: sys::CUstream,
        allow_value_reassociation: bool,
        can_select: bool,
        reference: &mut dyn FnMut() -> Result<(), String>,
    ) -> Result<bool, String> {
        let allow_value_reassociation = crate::optimisations::VALUE_REASSOCIATION && allow_value_reassociation;
        if allow_value_reassociation && (shape.transpose_a || shape.transpose_b
            || shape.a.rows != 512 || shape.a.cols != 2051 || shape.b.cols != 64
            || shape.a.batch != 32) {
            return Err("reassociated prefill plan requires the admitted value shape".into());
        }
        let key = (shape, allow_value_reassociation);
        // Prepare the exact control in the same cold process, so later paired
        // requests never tune or allocate after the cache has been sealed.
        if allow_value_reassociation && !self.sealed && can_select
            && !self.plans.contains_key(&(shape, false)) {
            self.enqueue(device.clone(), shape, a, b, output, workspace, alpha, beta,
                stream, false, can_select, reference)?;
        }
        if !self.plans.contains_key(&key) {
            if self.sealed || !can_select {
                return Ok(false);
            }
            device.bind_to_thread().map_err(|e| e.to_string())?;
            let align = [
                a.alignment()?.min(256),
                b.alignment()?.min(256),
                output.alignment()?.min(256),
                output.alignment()?.min(256),
            ];
            let Some(candidates) =
                Candidates::discover_bounded(device.clone(), shape, WORKSPACE_CAP, align, 4)?
            else {
                self.plans.insert(key, None);
                return Ok(false);
            };
            // beta=0 for both stages: repetitions never consume prior output.
            if beta != 0.0 {
                return Err("prefill Lt cold selection requires beta zero".into());
            }
            reference()?;
            let elements = shape.c.elements(true)?;
            let expected = readback(&device, output, elements, stream)?;
            if expected.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite FP32 calibration reference".into());
            }
            let mut clock = Clock::new(device.clone(), stream)?;
            let baseline = clock.measure(|| reference())?;
            let mut best = baseline * 0.95;
            let mut selected = None;
            let mut selected_exact = true;
            let mut rows = Vec::new();
            for candidate in candidates.values.iter() {
                let plan = match candidate.admit(shape, WORKSPACE_CAP)? {
                    Preparation::Ready(p) => p,
                    Preparation::Unsupported(reason) => {
                        rows.push(json!({"unsupported":reason}));
                        continue;
                    }
                };
                let _completion = Completion {
                    device: device.clone(),
                    stream,
                };
                // The checked prepared operation validates allocation context,
                // extent/alignment and aliases before any launch.
                plan.enqueue(a, b, output, output, Some(workspace), alpha, beta, stream)?;
                let actual = readback(&device, output, elements, stream)?;
                let exact = actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                let finite = actual.iter().all(|v| v.is_finite());
                let maximum = actual.iter().zip(&expected)
                    .map(|(a,b)| (*a as f64-*b as f64).abs()).fold(0.0f64,f64::max);
                let admitted = exact || (allow_value_reassociation && finite
                    && actual.len() == expected.len() && maximum <= 2e-4);
                if !admitted {
                    rows.push(json!({"identity":plan.identity,"exact":exact,"finite":finite,"max_abs":maximum}));
                    continue;
                }
                let ms = clock.measure(|| {
                    plan.enqueue(a, b, output, output, Some(workspace), alpha, beta, stream)
                })?;
                rows.push(json!({"identity":plan.identity,"exact":exact,"max_abs":maximum,"median_ms":ms,"workspace":plan.workspace_bytes}));
                if (ms < best && (exact || !selected_exact || ms < best * 0.95))
                    || (exact && !selected_exact && ms <= best * 1.05 && ms < baseline * 0.95) {
                    best = ms;
                    selected = Some(plan);
                    selected_exact = exact;
                }
            }
            let report = json!({"shape":format!("{shape:?}"),"baseline_ms":baseline,"selected_ms":selected.as_ref().map(|_|best),"selected":selected.as_ref().map(|p|p.identity.as_str()),"candidates":rows,"reference_values":elements,"reassociated_value":allow_value_reassociation,"criterion":if allow_value_reassociation {"complete finite actual-output error <=2e-4; prefer exact within five percent; at least five percent baseline timing margin"} else {"complete actual-output bitwise equality; at least five percent timing margin"}});
            eprintln!("[prefill-mla-lt] {report}");
            self.reports.push(report);
            self.plans.insert(key, selected);
        }
        let Some(plan) = self.plans.get(&key).and_then(Option::as_ref) else {
            return Ok(false);
        };
        plan.enqueue(a, b, output, output, Some(workspace), alpha, beta, stream)?;
        Ok(true)
    }
}

// Declared after each cold prepared plan, so it completes before that plan
// can drop on success, error or unwind. Hot plans stay owned by the engine.
struct Completion {
    device: Arc<CudaDevice>,
    stream: sys::CUstream,
}
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.device.bind_to_thread() {
            crate::vram_monitor::fatal_cuda_context_error(
                "prefill Lt cold context",
                &e.to_string(),
            );
        }
        let status = unsafe { sys::lib().cuStreamSynchronize(self.stream) };
        if status != sys::CUresult::CUDA_SUCCESS {
            crate::vram_monitor::fatal_cuda_context_error(
                "prefill Lt cold completion",
                &format!("{status:?}"),
            );
        }
    }
}

// Pinned async readback is explicit cold validation, confined to the owning
// stream. Error cleanup also completes that stream before freeing host memory.
struct HostReadback {
    pointer: *mut c_void,
    stream: sys::CUstream,
    complete: bool,
}
impl Drop for HostReadback {
    fn drop(&mut self) {
        unsafe {
            if !self.complete {
                let status = sys::lib().cuStreamSynchronize(self.stream);
                if status != sys::CUresult::CUDA_SUCCESS {
                    crate::vram_monitor::fatal_cuda_context_error(
                        "prefill Lt readback completion",
                        &format!("{status:?}"),
                    );
                }
            }
            let status = sys::lib().cuMemFreeHost(self.pointer);
            if status != sys::CUresult::CUDA_SUCCESS {
                eprintln!("ERROR: prefill Lt pinned readback release: {status:?}");
            }
        }
    }
}
fn readback(
    device: &Arc<CudaDevice>,
    span: &Span<'_>,
    elements: usize,
    stream: sys::CUstream,
) -> Result<Vec<f32>, String> {
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let bytes = elements
        .checked_mul(4)
        .ok_or("prefill Lt readback size overflow")?;
    span.range(bytes)?;
    let mut pointer = std::ptr::null_mut();
    unsafe { sys::lib().cuMemHostAlloc(&mut pointer, bytes, 0) }
        .result()
        .map_err(|e| format!("prefill Lt pinned readback: {e:?}"))?;
    let mut host = HostReadback {
        pointer,
        stream,
        complete: false,
    };
    unsafe { sys::lib().cuMemcpyDtoHAsync_v2(pointer, span.pointer(), bytes, stream) }
        .result()
        .map_err(|e| format!("prefill Lt async readback: {e:?}"))?;
    unsafe { sys::lib().cuStreamSynchronize(stream) }
        .result()
        .map_err(|e| format!("prefill Lt readback stream: {e:?}"))?;
    host.complete = true;
    Ok(unsafe { std::slice::from_raw_parts(pointer.cast::<f32>(), elements) }.to_vec())
}
struct Clock {
    start: sys::CUevent,
    end: sys::CUevent,
    device: Arc<CudaDevice>,
    stream: sys::CUstream,
}
impl Clock {
    fn new(device: Arc<CudaDevice>, stream: sys::CUstream) -> Result<Self, String> {
        let mut start = std::ptr::null_mut();
        let mut end = std::ptr::null_mut();
        unsafe { sys::lib().cuEventCreate(&mut start, 0) }
            .result()
            .map_err(|e| e.to_string())?;
        if let Err(e) = unsafe { sys::lib().cuEventCreate(&mut end, 0) }.result() {
            unsafe {
                let _ = sys::lib().cuEventDestroy_v2(start);
            };
            return Err(e.to_string());
        }
        Ok(Self {
            start,
            end,
            device,
            stream,
        })
    }
    fn measure(
        &mut self,
        mut operation: impl FnMut() -> Result<(), String>,
    ) -> Result<f32, String> {
        let mut times = Vec::with_capacity(5);
        for rep in 0..7 {
            unsafe { sys::lib().cuEventRecord(self.start, self.stream) }
                .result()
                .map_err(|e| e.to_string())?;
            operation()?;
            unsafe { sys::lib().cuEventRecord(self.end, self.stream) }
                .result()
                .map_err(|e| e.to_string())?;
            unsafe { sys::lib().cuEventSynchronize(self.end) }
                .result()
                .map_err(|e| e.to_string())?;
            let mut ms = 0.0;
            unsafe { sys::lib().cuEventElapsedTime(&mut ms, self.start, self.end) }
                .result()
                .map_err(|e| e.to_string())?;
            if rep >= 2 {
                times.push(ms);
            }
        }
        times.sort_by(f32::total_cmp);
        Ok(times[2])
    }
}
impl Drop for Clock {
    fn drop(&mut self) {
        unsafe {
            if self.device.bind_to_thread().is_err() {
                return;
            }
            let _ = sys::lib().cuEventDestroy_v2(self.start);
            let _ = sys::lib().cuEventDestroy_v2(self.end);
        }
    }
}
