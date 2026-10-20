//! Cold measured selection of one explicit algorithm across actual batch and
//! singleton layouts. Never called by token enqueue or graph replay.
use crate::stable_mla_gemm::{Candidate, Candidates, Preparation, Prepared, Shape, Span};
use crate::stable_mla_path::{Geometry, Inputs, Work};
use cudarc::driver::{sys, CudaDevice};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, mem::size_of, sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Stage {
    Score,
    Latent,
    Wvc,
}
impl Stage {
    fn index(self) -> usize {
        match self {
            Self::Score => 0,
            Self::Latent => 1,
            Self::Wvc => 2,
        }
    }
}
pub(crate) struct Selected {
    pub candidate: Candidate,
    pub report: Value,
}
struct Trial<'a> {
    shape: Shape,
    a: Span<'a>,
    b: Span<'a>,
    c: Span<'a>,
    output: Span<'a>,
    rows: usize,
}
fn bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(size_of::<f32>())
        .ok_or("MLA calibration byte extent overflow".into())
}
fn view<'a>(source: &'a Span<'_>, offset: usize, elements: usize) -> Result<Span<'a>, String> {
    source.subview(bytes(offset)?, bytes(elements)?)
}
fn trial<'a>(
    g: Geometry,
    stage: Stage,
    rows: usize,
    single: bool,
    input: &'a Inputs<'_>,
    work: &'a Work<'_>,
) -> Result<Trial<'a>, String> {
    trial_row(
        g,
        stage,
        rows,
        if single {
            Some(rows.checked_sub(1).ok_or("empty MLA calibration")?)
        } else {
            None
        },
        input,
        work,
    )
}
fn trial_row<'a>(
    g: Geometry,
    stage: Stage,
    rows: usize,
    single: Option<usize>,
    input: &'a Inputs<'_>,
    work: &'a Work<'_>,
) -> Result<Trial<'a>, String> {
    let row = single.unwrap_or(0);
    if row >= rows {
        return Err("MLA actual singleton row outside calibration batch".into());
    }
    let count = if single.is_some() { 1 } else { rows };
    let shape = g.shapes(count)?[stage.index()];
    let (a, b, c) = match stage {
        Stage::Score => (
            view(
                &work.gathered,
                row * g.slots * (g.ccd + g.rope),
                shape.a.elements(false)?,
            )?,
            view(
                &work.query,
                row * g.heads * (g.ccd + g.rope),
                shape.b.elements(false)?,
            )?,
            view(
                &work.weights,
                row * g.heads * g.slots,
                shape.c.elements(false)?,
            )?,
        ),
        Stage::Latent => (
            view(
                &work.gathered,
                row * g.slots * (g.ccd + g.rope),
                shape.a.elements(false)?,
            )?,
            view(
                &work.weights,
                row * g.heads * g.slots,
                shape.b.elements(false)?,
            )?,
            view(&work.attention, 0, shape.c.elements(false)?)?,
        ),
        Stage::Wvc => (
            view(&input.wvc, 0, shape.a.elements(false)?)?,
            view(
                &work.attention,
                row * g.heads * g.ccd,
                shape.b.elements(false)?,
            )?,
            view(&work.projected, 0, shape.c.elements(false)?)?,
        ),
    };
    let output_source = match stage {
        Stage::Score => &work.scores,
        Stage::Latent => &work.attention,
        Stage::Wvc => &work.projected,
    };
    Ok(Trial {
        shape,
        a,
        b,
        c,
        output: view(output_source, 0, shape.c.elements(true)?)?,
        rows: count,
    })
}
fn drain(device: &Arc<CudaDevice>, stream: sys::CUstream) -> Result<(), String> {
    device.bind_to_thread().map_err(|e| e.to_string())?;
    unsafe { sys::lib().cuStreamSynchronize(stream) }
        .result()
        .map_err(|e| format!("MLA calibration completion: {e}"))
}
fn download(source: &Span<'_>, elements: usize) -> Result<Vec<f32>, String> {
    source.range(bytes(elements)?)?;
    let mut out = vec![0f32; elements];
    unsafe {
        sys::lib().cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), source.pointer(), bytes(elements)?)
    }
    .result()
    .map_err(|e| format!("MLA calibration actual operand read: {e}"))?;
    Ok(out)
}
fn logical_row(
    g: Geometry,
    stage: Stage,
    trial: &Trial<'_>,
    row: usize,
) -> Result<Vec<f32>, String> {
    if row >= trial.rows {
        return Err("MLA calibration logical row outside actual trial".into());
    }
    let width = match stage {
        Stage::Score => g.slots,
        Stage::Latent => g.ccd,
        Stage::Wvc => g.values,
    };
    let elements = g
        .heads
        .checked_mul(width)
        .ok_or("MLA calibration output extent overflow")?;
    let mut values = Vec::with_capacity(elements);
    if matches!(stage, Stage::Wvc) {
        for head in 0..g.heads {
            let offset = (head * trial.rows + row) * g.values;
            values.extend(download(&view(&trial.output, offset, g.values)?, g.values)?);
        }
    } else {
        values = download(&view(&trial.output, row * elements, elements)?, elements)?;
    }
    Ok(values)
}
fn row_hash(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for value in values {
        h.update(value.to_bits().to_le_bytes())
    }
    format!("{:x}", h.finalize())
}
struct Reference {
    value: Vec<f64>,
    bound: Vec<f64>,
}
fn reference(trial: &Trial<'_>, alpha: f32, beta: f32) -> Result<Reference, String> {
    let a = download(&trial.a, trial.shape.a.elements(false)?)?;
    let b = download(&trial.b, trial.shape.b.elements(false)?)?;
    let c = if beta == 0.0 {
        None
    } else {
        Some(download(&trial.c, trial.shape.c.elements(false)?)?)
    };
    let shape = trial.shape;
    let contraction = if shape.transpose_a {
        shape.a.rows
    } else {
        shape.a.cols
    };
    let error_steps = contraction
        .checked_add(1)
        .ok_or("MLA contraction bound overflow")?;
    let e = error_steps as f64 * f32::EPSILON as f64;
    if e >= 1.0 {
        return Err("MLA F32 contraction has no finite relative error bound".into());
    }
    let gamma = e / (1.0 - e);
    let mut value = Vec::new();
    let mut bound = Vec::new();
    for batch in 0..shape.c.batch {
        for col in 0..shape.c.cols {
            for row in 0..shape.c.rows {
                let ci = batch * shape.c.stride + col * shape.c.leading + row;
                if c.as_ref().is_some_and(|v| v[ci] == f32::NEG_INFINITY) {
                    value.push(f64::NEG_INFINITY);
                    bound.push(0.0);
                    continue;
                }
                let mut sum = 0f64;
                let mut absolute = 0f64;
                for k in 0..contraction {
                    let (ar, ac) = if shape.transpose_a {
                        (k, row)
                    } else {
                        (row, k)
                    };
                    let (br, bc) = if shape.transpose_b {
                        (col, k)
                    } else {
                        (k, col)
                    };
                    let av = a[batch * shape.a.stride + ac * shape.a.leading + ar] as f64;
                    let bv = b[batch * shape.b.stride + bc * shape.b.leading + br] as f64;
                    if !av.is_finite() || !bv.is_finite() {
                        return Err("nonfinite actual MLA calibration input".into());
                    }
                    let product = av * bv;
                    sum += product;
                    absolute += product.abs();
                }
                let initial = c.as_ref().map_or(0.0, |v| v[ci] as f64 * beta as f64);
                if !initial.is_finite() {
                    return Err("nonfinite actual MLA calibration initial value".into());
                }
                value.push(sum * alpha as f64 + initial);
                bound.push(gamma * (absolute * (alpha as f64).abs() + initial.abs()));
            }
        }
    }
    Ok(Reference { value, bound })
}
fn validate(values: &[f32], reference: &Reference) -> Result<f64, String> {
    if values.len() != reference.value.len() {
        return Err("MLA calibration complete row length differs".into());
    }
    let mut maximum = 0f64;
    for (i, ((&actual, &expected), &bound)) in values
        .iter()
        .zip(&reference.value)
        .zip(&reference.bound)
        .enumerate()
    {
        if expected == f64::NEG_INFINITY {
            if actual != f32::NEG_INFINITY {
                return Err(format!("MLA masked calibration output differs at {i}"));
            }
            continue;
        }
        let error = (actual as f64 - expected).abs();
        if !expected.is_finite()
            || !bound.is_finite()
            || !actual.is_finite()
            || error > bound + f64::MIN_POSITIVE
        {
            return Err(format!("MLA F64 contraction rejected coordinate {i}: actual={actual} reference={expected} error={error} bound={bound}"));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
struct Event {
    raw: sys::CUevent,
    device: Arc<CudaDevice>,
}
impl Event {
    fn new(device: &Arc<CudaDevice>) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        unsafe { sys::lib().cuEventCreate(&mut raw, 0) }
            .result()
            .map_err(|e| format!("MLA calibration event creation: {e}"))?;
        Ok(Self {
            raw,
            device: device.clone(),
        })
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::lib()
                .cuCtxPushCurrent_v2(*self.device.cu_primary_ctx())
                .result()
            {
                eprintln!("ERROR: calibration event context: {e}");
                return;
            }
            if let Err(e) = sys::lib().cuEventDestroy_v2(self.raw).result() {
                eprintln!("ERROR: calibration event release: {e}")
            }
            let mut context = std::ptr::null_mut();
            if let Err(e) = sys::lib().cuCtxPopCurrent_v2(&mut context).result() {
                eprintln!("ERROR: calibration event context restoration: {e}")
            }
        }
    }
}
fn measure(
    device: &Arc<CudaDevice>,
    stream: sys::CUstream,
    prepared: &Prepared,
    trial: &Trial<'_>,
    workspace: Option<&Span<'_>>,
    alpha: f32,
    beta: f32,
    warmups: usize,
    samples: usize,
    start: &Event,
    stop: &Event,
) -> Result<Value, String> {
    for _ in 0..warmups {
        unsafe {
            prepared.enqueue(
                &trial.a,
                &trial.b,
                &trial.c,
                &trial.output,
                workspace,
                alpha,
                beta,
                stream,
            )?
        }
    }
    drain(device, stream)?;
    let mut cuda_ms = Vec::with_capacity(samples);
    let mut cpu_us = Vec::with_capacity(samples);
    for _ in 0..samples {
        unsafe { sys::lib().cuEventRecord(start.raw, stream) }
            .result()
            .map_err(|e| format!("MLA calibration event start: {e}"))?;
        let clock = Instant::now();
        unsafe {
            prepared.enqueue(
                &trial.a,
                &trial.b,
                &trial.c,
                &trial.output,
                workspace,
                alpha,
                beta,
                stream,
            )?
        }
        cpu_us.push(clock.elapsed().as_secs_f64() * 1e6);
        unsafe { sys::lib().cuEventRecord(stop.raw, stream) }
            .result()
            .map_err(|e| format!("MLA calibration event stop: {e}"))?;
        drain(device, stream)?;
        let mut ms = 0f32;
        unsafe { sys::lib().cuEventElapsedTime(&mut ms, start.raw, stop.raw) }
            .result()
            .map_err(|e| format!("MLA calibration event elapsed: {e}"))?;
        if !ms.is_finite() || ms <= 0.0 {
            return Err("MLA CUDA event measurement has no finite positive duration".into());
        }
        cuda_ms.push(ms as f64);
    }
    let mut sorted = cuda_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(samples - 1) / 2] + sorted[samples / 2]) * 0.5;
    Ok(
        json!({"rows":trial.rows,"cuda_stream_ms":cuda_ms,"cpu_enqueue_us":cpu_us,"median_cuda_stream_ms":median,"warmups":warmups,"samples":samples,"scope":"Cold actual GEMM CUDA stream spans and Rust CPU enqueue; not pure kernel/model/client throughput"}),
    )
}
fn alignments(trial: &Trial<'_>) -> Result<[usize; 4], String> {
    let spans = [&trial.a, &trial.b, &trial.c, &trial.output];
    let layouts = [trial.shape.a, trial.shape.b, trial.shape.c, trial.shape.c];
    let mut out = [0usize; 4];
    for i in 0..4 {
        let mut alignment = spans[i].alignment()?;
        if layouts[i].batch > 1 && layouts[i].stride > 0 {
            let stride = bytes(layouts[i].stride)?;
            alignment = alignment.min(stride & stride.wrapping_neg());
        }
        while alignment > u32::MAX as usize {
            alignment /= 2
        }
        out[i] = alignment;
    }
    Ok(out)
}

/// The caller retains real source/workspace/stream owners throughout this cold
/// operation. Only explicit NOT_SUPPORTED/workspace rejection excludes a shape;
/// CUDA/API errors remain visible. No replacement kernel or fixed cost is used.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn select_stage(
    device: Arc<CudaDevice>,
    g: Geometry,
    stage: Stage,
    rows: usize,
    input: &Inputs<'_>,
    work: &Work<'_>,
    workspace: Option<&Span<'_>>,
    workspace_capacity: usize,
    alpha: f32,
    warmups: usize,
    samples: usize,
    stream: sys::CUstream,
) -> Result<Selected, String> {
    if samples == 0 || warmups == 0 || rows == 0 || !alpha.is_finite() || alpha <= 0.0 {
        return Err("MLA cold measurement requires real observations and finite geometry".into());
    }
    let mut capture = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    sys::lib()
        .cuStreamIsCapturing(stream, &mut capture)
        .result()
        .map_err(|e| format!("MLA cold capture admission: {e}"))?;
    if capture != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err("cold MLA selection cannot run during CUDA graph capture".into());
    }
    let full = trial(g, stage, rows, false, input, work)?;
    let one = trial(g, stage, rows, true, input, work)?;
    let beta = if matches!(stage, Stage::Score) {
        1.0
    } else {
        0.0
    };
    drain(&device, stream)?;
    let expected = reference(&one, alpha, beta)?;
    let mut algorithms = Vec::new();
    let mut seen = HashSet::new();
    for t in [&full, &one] {
        for candidate in
            Candidates::discover(device.clone(), t.shape, workspace_capacity, alignments(t)?)?
                .values
        {
            if seen.insert(candidate.configuration()) {
                algorithms.push(candidate)
            }
        }
    }
    let start = Event::new(&device)?;
    let stop = Event::new(&device)?;
    let mut records = Vec::new();
    let mut best: Option<(usize, f64)> = None;
    for (index, candidate) in algorithms.iter().enumerate() {
        let (full_plan, one_plan) = match (
            candidate.admit(full.shape, workspace_capacity)?,
            candidate.admit(one.shape, workspace_capacity)?,
        ) {
            (Preparation::Ready(full), Preparation::Ready(one)) => (full, one),
            (a, b) => {
                let reason = |x: Preparation| match x {
                    Preparation::Ready(_) => None,
                    Preparation::Unsupported(reason) => Some(reason),
                };
                records.push(json!({"index":index,"algorithm":candidate.configuration(),"eligible":false,"shape_admission":[reason(a),reason(b)]}));
                continue;
            }
        };
        let retained = (
            &full_plan, &one_plan, &full, &one, &start, &stop, workspace, input, work, candidate,
        );
        let result = crate::sequence_copy_completion::with_copy_completion(
            &device,
            stream,
            &retained,
            |_| {
                full_plan.enqueue(
                    &full.a,
                    &full.b,
                    &full.c,
                    &full.output,
                    workspace,
                    alpha,
                    beta,
                    stream,
                )?;
                drain(&device, stream)?;
                let mut full_digest = Sha256::new();
                let mut single_digest = Sha256::new();
                let mut full_values = Vec::new();
                let mut one_values = Vec::new();
                for row in 0..rows {
                    // Singleton WVC writes a compact head-major plane, so renew the
                    // full expected output before reading its next physical row.
                    if row > 0 && matches!(stage, Stage::Wvc) {
                        full_plan.enqueue(
                            &full.a,
                            &full.b,
                            &full.c,
                            &full.output,
                            workspace,
                            alpha,
                            beta,
                            stream,
                        )?;
                        drain(&device, stream)?;
                    }
                    let expected_row = logical_row(g, stage, &full, row)?;
                    let actual_trial = trial_row(g, stage, rows, Some(row), input, work)?;
                    one_plan.enqueue(
                        &actual_trial.a,
                        &actual_trial.b,
                        &actual_trial.c,
                        &actual_trial.output,
                        workspace,
                        alpha,
                        beta,
                        stream,
                    )?;
                    drain(&device, stream)?;
                    let actual_row = logical_row(g, stage, &actual_trial, 0)?;
                    if expected_row.len() != actual_row.len()
                        || expected_row
                            .iter()
                            .zip(&actual_row)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Ok(
                            json!({"eligible":false,"reason":"actual full/singleton output bytes differ","row":row,"full_sha256":row_hash(&expected_row),"singleton_sha256":row_hash(&actual_row)}),
                        );
                    }
                    for value in &expected_row {
                        full_digest.update(value.to_bits().to_le_bytes());
                    }
                    for value in &actual_row {
                        single_digest.update(value.to_bits().to_le_bytes());
                    }
                    if row == rows - 1 {
                        full_values = expected_row;
                        one_values = actual_row;
                    }
                }
                let max_error = match validate(&full_values, &expected) {
                    Ok(error) => error,
                    Err(reason) => return Ok(json!({"eligible":false,"reason":reason})),
                };
                let full_timing = measure(
                    &device, stream, &full_plan, &full, workspace, alpha, beta, warmups, samples,
                    &start, &stop,
                )?;
                let after = logical_row(g, stage, &full, rows - 1)?;
                if row_hash(&after) != row_hash(&full_values) {
                    return Err("MLA measurement changed actual full output".into());
                }
                let one_timing = measure(
                    &device, stream, &one_plan, &one, workspace, alpha, beta, warmups, samples,
                    &start, &stop,
                )?;
                let after = logical_row(g, stage, &one, 0)?;
                if row_hash(&after) != row_hash(&one_values) {
                    return Err("MLA measurement changed actual singleton output".into());
                }
                let cost = full_timing["median_cuda_stream_ms"]
                    .as_f64()
                    .ok_or("missing measured full cost")?
                    + one_timing["median_cuda_stream_ms"]
                        .as_f64()
                        .ok_or("missing measured singleton cost")?;
                Ok(
                    json!({"eligible":true,"measured_pair_cost_ms":cost,"full":full_timing,"singleton":one_timing,"full_last_row_sha256":row_hash(&full_values),"singleton_sha256":row_hash(&one_values),"every_row_full_sha256":format!("{:x}",full_digest.finalize()),"every_row_singleton_sha256":format!("{:x}",single_digest.finalize()),"validated_rows":rows,"max_f64_contraction_error":max_error,"validated_elements":full_values.len(),"workspace_bytes":[full_plan.workspace_bytes,one_plan.workspace_bytes]}),
                )
            },
        )?;
        if result["eligible"] == true {
            let cost = result["measured_pair_cost_ms"]
                .as_f64()
                .ok_or("actual selection cost absent")?;
            if best.as_ref().is_none_or(|&(_, old)| cost < old) {
                best = Some((index, cost));
            }
        }
        records.push(json!({"index":index,"algorithm":candidate.configuration(),"result":result}));
    }
    let (index,cost)=best.ok_or("no actual measured MLA family passes singleton/full/F64 admission; refusing alternate arithmetic")?;
    let report = json!({"stage":format!("{stage:?}"),"selected_index":index,"selected_algorithm":algorithms[index].configuration(),"measured_pair_cost_ms":cost,"objective":"minimum sum of measured medians for actual full and last-row singleton workloads","actual_rows":rows,"workspace_capacity_bytes":workspace_capacity,"records":records,"scope":"Cold per-geometry runtime selection. All observations retained; full VRAM/model calibration remains independently mandatory."});
    let candidate = algorithms
        .into_iter()
        .nth(index)
        .ok_or("selected runtime candidate disappeared")?;
    Ok(Selected { candidate, report })
}
