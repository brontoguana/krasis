//! Per-device common MLA bank. The owning Store/Engine serialize access through
//! one Mutex; raw Lt descriptors never escape it. No Python call on execution.
use crate::{
    stable_mla_gemm::{Candidate, Span},
    stable_mla_ops::{Gather, Ops},
    stable_mla_owned::Layer,
    stable_mla_path::{Geometry, Inputs, Path, Work},
    stable_mla_select::{self, Stage},
};
use cudarc::driver::{sys, CudaDevice};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
struct Family {
    candidates: [Candidate; 3],
    paths: HashMap<usize, Path>,
    identity: String,
    reports: [Value; 3],
}
pub(crate) struct Bank {
    device: Arc<CudaDevice>,
    pub ops: Arc<Ops>,
    layers: BTreeMap<usize, Arc<Layer>>,
    families: HashMap<Geometry, Family>,
    pub workspace_capacity: usize,
}
// Every raw Lt object belongs exclusively to this bank, remains on its actual
// retained CUDA context, and is accessed under the Store/Engine's single Mutex.
// Immutable prepared objects do not escape that lock. GPU work retains the bank
// through ordinary request/graph completion; this does not permit concurrent FFI.
unsafe impl Send for Bank {}
impl Bank {
    pub(crate) fn new(
        device: Arc<CudaDevice>,
        ops: Arc<Ops>,
        layers: BTreeMap<usize, Arc<Layer>>,
        workspace_capacity: usize,
        context: usize,
        format: u32,
    ) -> Result<Self, String> {
        for layer in layers.values() {
            layer.validate_cache(context, format)?;
        }
        if layers.is_empty() {
            return Err("common MLA bank has no real model allocations".into());
        }
        Ok(Self {
            device,
            ops,
            layers,
            families: HashMap::new(),
            workspace_capacity,
        })
    }
    pub(crate) fn layer(&self, index: usize) -> Result<Arc<Layer>, String> {
        self.layers
            .get(&index)
            .cloned()
            .ok_or_else(|| format!("common MLA layer {index} actual owners are missing"))
    }
    pub(crate) fn resident_bytes(&self) -> Result<usize, String> {
        self.layers.values().try_fold(0usize, |n, l| {
            n.checked_add(l.resident_bytes()?)
                .ok_or("common MLA resident extent overflow".into())
        })
    }
    pub(crate) fn compatibility(&self) -> Result<Value, String> {
        if self.families.is_empty() {
            return Err(
                "common MLA arithmetic families have not completed startup measurement".into(),
            );
        }
        let mut values: Vec<_> = self
            .families
            .iter()
            .map(|(g, f)| (format!("{g:?}"), f.identity.clone()))
            .collect();
        values.sort();
        Ok(
            json!({"backend":"common_mla_f32_v1","kernels":self.ops.identity,"weight_precision":"exact_bf16_to_f32_resident_v1","families":values}),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn prepare(
        &mut self,
        layer: usize,
        g: Geometry,
        gather: Gather,
        input: &Inputs<'_>,
        work: &Work<'_>,
        workspace: Option<&Span<'_>>,
        scale: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let dimensions = self.layer(layer)?.dimensions;
        if dimensions.heads != g.heads
            || dimensions.ccd != g.ccd
            || dimensions.rope != g.rope
            || dimensions.values != g.values
        {
            return Err("common MLA actual layer geometry differs from prepared program".into());
        }
        let rows = gather.rows;
        if !self.families.contains_key(&g) {
            let retained = (input, work, workspace, &self.ops);
            let selected = crate::sequence_copy_completion::with_copy_completion(
                &self.device,
                stream,
                &retained,
                |_| {
                    // The weight plane holds the immutable causal mask during score profiling.
                    self.ops.gather(
                        gather,
                        &work.gathered,
                        &work.query,
                        &work.weights,
                        &input.q_abs,
                        input.q_rope.as_ref(),
                        &input.ckv,
                        input.kpe.as_ref(),
                        &input.selected,
                        &input.positions,
                        stream,
                    )?;
                    let score = stable_mla_select::select_stage(
                        self.device.clone(),
                        g,
                        Stage::Score,
                        rows,
                        input,
                        work,
                        workspace,
                        self.workspace_capacity,
                        scale,
                        8,
                        32,
                        stream,
                    )?;
                    let prepared = score
                        .candidate
                        .prepare(g.shapes(rows)?[0], self.workspace_capacity)?;
                    crate::sequence_copy_completion::with_copy_completion(
                        &self.device,
                        stream,
                        &(&prepared, input, work, workspace),
                        |_| {
                            prepared.enqueue(
                                &work.gathered,
                                &work.query,
                                &work.weights,
                                &work.scores,
                                workspace,
                                scale,
                                1.0,
                                stream,
                            )
                        },
                    )?;
                    self.ops.softmax(
                        &work.scores,
                        &work.weights,
                        rows,
                        g.heads,
                        g.slots,
                        stream,
                    )?;
                    let latent = stable_mla_select::select_stage(
                        self.device.clone(),
                        g,
                        Stage::Latent,
                        rows,
                        input,
                        work,
                        workspace,
                        self.workspace_capacity,
                        1.0,
                        8,
                        32,
                        stream,
                    )?;
                    let prepared = latent
                        .candidate
                        .prepare(g.shapes(rows)?[1], self.workspace_capacity)?;
                    crate::sequence_copy_completion::with_copy_completion(
                        &self.device,
                        stream,
                        &(&prepared, input, work, workspace),
                        |_| {
                            prepared.enqueue(
                                &work.gathered,
                                &work.weights,
                                &work.attention,
                                &work.attention,
                                workspace,
                                1.0,
                                0.0,
                                stream,
                            )
                        },
                    )?;
                    let wvc = stable_mla_select::select_stage(
                        self.device.clone(),
                        g,
                        Stage::Wvc,
                        rows,
                        input,
                        work,
                        workspace,
                        self.workspace_capacity,
                        1.0,
                        8,
                        32,
                        stream,
                    )?;
                    Ok([score, latent, wvc])
                },
            )?;
            let [score, latent, wvc] = selected;
            let reports = [score.report, latent.report, wvc.report];
            let candidates = [score.candidate, latent.candidate, wvc.candidate];
            let one = Path::prepare(
                self.ops.clone(),
                g,
                1,
                [&candidates[0], &candidates[1], &candidates[2]],
                self.workspace_capacity,
            )?;
            let identity = one.family_identity.clone();
            let mut paths = HashMap::new();
            paths.insert(1, one);
            log::info!(
                "Common MLA runtime measurement: {}",
                json!({"layer":layer,"geometry":format!("{g:?}"),"family_identity":identity,"reports":reports})
            );
            self.families.insert(
                g,
                Family {
                    candidates,
                    paths,
                    identity,
                    reports,
                },
            );
        }
        let family = self
            .families
            .get_mut(&g)
            .ok_or("measured MLA family absent")?;
        if !family.paths.contains_key(&rows) {
            family.paths.insert(
                rows,
                Path::prepare(
                    self.ops.clone(),
                    g,
                    rows,
                    [
                        &family.candidates[0],
                        &family.candidates[1],
                        &family.candidates[2],
                    ],
                    self.workspace_capacity,
                )?,
            );
        }
        Ok(())
    }
    pub(crate) unsafe fn enqueue(
        &self,
        g: Geometry,
        gather: Gather,
        input: &Inputs<'_>,
        work: &Work<'_>,
        workspace: Option<&Span<'_>>,
        scale: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        self.enqueue_observed(g,gather,input,work,workspace,scale,stream,None)
    }
    pub(crate) unsafe fn enqueue_observed(
        &self,
        g: Geometry,
        gather: Gather,
        input: &Inputs<'_>,
        work: &Work<'_>,
        workspace: Option<&Span<'_>>,
        scale: f32,
        stream: sys::CUstream,
        observer: Option<&crate::matched_mla_stage::Observer<'_>>,
    ) -> Result<(), String> {
        let program = self
            .families
            .get(&g)
            .and_then(|family| family.paths.get(&gather.rows))
            .ok_or("common MLA request was not prepared before enqueue/capture")?;
        program.enqueue_observed(gather, input, work, workspace, scale, stream, observer)
    }
    pub(crate) fn reports(&self) -> Value {
        let mut rows: Vec<_> = self
            .families
            .iter()
            .map(|(g, f)| {
                (
                    format!("{g:?}"),
                    json!({"identity":f.identity,"reports":f.reports}),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        json!(rows)
    }
}
