//! Calibration ownership for diagnostic between-request switching.
use super::*;

pub(super) struct Calibration {
    counts: Vec<usize>,
    receipt: Option<serde_json::Value>,
    workers: Vec<CpuTailRuntime>,
    w13: HashMap<(usize, usize, usize, bool), usize>,
    w2: HashMap<(usize, usize, bool), usize>,
    n13: HashMap<(usize, usize, usize, usize), bool>,
    n2: HashMap<(usize, usize, usize, u32, bool), bool>,
}
impl Calibration {
    fn take(g: &mut GpuDecodeGraph) -> Self {
        Self {counts: g.cpu_tail_planned_counts.clone(), receipt:g.cpu_tail_calibration.clone(),
            workers:std::mem::take(&mut g.cpu_tail_workers),
            w13:g.ksplit_autotune.clone(),w2:g.shared_w2_ksplit_autotune.clone(),
            n13:g.int4_w13_n32_autotune.clone(),n2:g.int4_w2_n32_autotune.clone()}
    }
    fn restore(self, g: &mut GpuDecodeGraph) {
        g.cpu_tail_planned_counts=self.counts;g.cpu_tail_calibration=self.receipt;
        g.cpu_tail_workers=self.workers;g.ksplit_autotune=self.w13;g.shared_w2_ksplit_autotune=self.w2;
        g.int4_w13_n32_autotune=self.n13;g.int4_w2_n32_autotune=self.n2;
        g.ksplit_autotune_complete=true;g.int4_n32_autotune_complete=true;
    }
}
impl GpuDecodeStore {
    pub(crate) fn switch_live(&mut self, next: crate::optimisations::Selection, execution_plan: Option<&serde_json::Value>) -> Result<serde_json::Value,String> {
        self.device.bind_to_thread().map_err(|e|format!("switch CUDA context: {e:?}"))?;
        let sync=unsafe {cuda_sys::lib().cuCtxSynchronize()};
        if sync != cuda_sys::CUresult::CUDA_SUCCESS {return Err(format!("switch synchronize: {sync:?}"));}
        self.invalidate_cuda_graph();
        let mut g=self.graph.take().ok_or("switch requires configured graph")?;
        let previous=g.live_switches;
        let previous_hash=crate::live_switches::hash(previous);
        let hash=crate::live_switches::hash(next);
        let start=std::time::Instant::now();
        let result=(|| -> Result<serde_json::Value,String> {
            if g.glm5_hc_mean_mult.is_none() || g.cpu_tail_calibration.is_none() {
                return Err("live harness requires the calibrated supported GLM profile".into());
            }
            let reused=if hash==previous_hash {true} else {
                let saved=Calibration::take(&mut g);g.live_calibrations.insert(previous_hash.clone(),saved);
                if let Some(saved)=g.live_calibrations.remove(&hash) {saved.restore(&mut g);true}
                else {
                    // New selection gets its own calibration, never an arm's cached costs.
                    let attempt=(|| -> Result<(),String> {
                        self.calibrate_live_cpu(&mut g,next)?;
                        g.ksplit_autotune_complete=false;g.int4_n32_autotune_complete=false;
                        g.ksplit_autotune.clear();g.shared_w2_ksplit_autotune.clear();
                        g.int4_w13_n32_autotune.clear();g.int4_w2_n32_autotune.clear();
                        self.ensure_marlin_dispatch_autotuned_for_graph(&mut g)
                    })();
                    if let Err(error)=attempt {
                        if let Some(saved)=g.live_calibrations.remove(&previous_hash) {saved.restore(&mut g);}
                        return Err(error);
                    }
                    false
                }
            };
            if let Some(plan)=execution_plan {
                // Freeze decisions, never relabel the measured calibration costs.
                let plan_hash=plan.get("switch_set_sha256").and_then(|x|x.as_str()).ok_or("execution plan lacks switch hash")?;
                if plan_hash != hash {return Err("execution plan belongs to another switch set".into());}
                let geometry=serde_json::json!({"hidden":g.hidden_size,"intermediate":g.moe_intermediate_size,"group_size":g.group_size});
                if plan.get("geometry") != Some(&geometry) {return Err("execution plan geometry mismatch".into());}
                let counts=plan.get("cpu_counts").and_then(|x|x.as_array()).ok_or("execution plan lacks CPU counts")?;
                let counts=counts.iter().map(|x|x.as_u64().map(|x|x as usize).ok_or("invalid CPU count".to_string())).collect::<Result<Vec<_>,_>>()?;
                if counts.len()!=g.max_experts_per_tok+1 || counts.iter().enumerate().any(|(depth,&n)|n>depth || n>g.cpu_tail_workers.len()) {
                    return Err("execution plan CPU counts exceed existing workers or routes".into());
                }
                let affinities=(0..g.cpu_tail_workers.len()).map(|i|crate::cpu_tail::compiled_lane_cpu_ids(i,g.cpu_tail_workers.len())).collect::<Result<Vec<_>,_>>()?;
                if plan.get("cpu_ids") != Some(&serde_json::json!(affinities)) {return Err("execution plan CPU topology mismatch".into());}
                let w2=plan.get("shared_w2").and_then(|x|x.as_array()).ok_or("execution plan lacks shared W2 selections")?;
                let mut choices=Vec::new();
                for row in w2 {
                    let k=row.get("k").and_then(|x|x.as_u64()).ok_or("missing shared k")? as usize;
                    let n=row.get("n").and_then(|x|x.as_u64()).ok_or("missing shared n")? as usize;
                    let int8=row.get("int8").and_then(|x|x.as_bool()).ok_or("missing shared format")?;
                    let splits=row.get("splits").and_then(|x|x.as_u64()).ok_or("missing shared splits")? as usize;
                    if !g.shared_w2_ksplit_autotune.contains_key(&(k,n,int8)) || ![2,4,8].contains(&splits) || k % splits != 0 {
                        return Err("shared W2 plan is not an admitted calibrated shape/launch".into());
                    }
                    choices.push(((k,n,int8),splits));
                }
                g.cpu_tail_planned_counts=counts;
                for (shape,choice) in choices {g.shared_w2_ksplit_autotune.insert(shape,choice);}
                if let Some(receipt)=g.cpu_tail_calibration.as_mut() {
                    receipt["diagnostic_execution_plan"]=plan.clone();
                }
            }
            g.live_switches=next;
            if let Some(receipt)=g.cpu_tail_calibration.as_mut() {receipt["switch_set_sha256"]=serde_json::json!(hash);receipt["optimisations"]=crate::live_switches::manifest(next);}
            g.mla_tensor=next.TENSOR_DECODE_ATTENTION();
            g.mla_split_exact=!g.mla_tensor;g.mla_split_unpack=!g.mla_tensor;
            if next.COLD_EXPERT_PRUNING_FIVE()!=previous.COLD_EXPERT_PRUNING_FIVE() {
                g.adaptive_cold_drop=AdaptiveColdDropRuntime::from_selection(next);
            }
            Ok(serde_json::json!({"switch_set_sha256":hash,"optimisations":crate::live_switches::manifest(next),
                "calibration_reused":reused,"calibration_seconds":start.elapsed().as_secs_f64(),
                "cpu_calibration":g.cpu_tail_calibration,"actual_cpu_counts":g.cpu_tail_planned_counts,"shared_w13":format!("{:?}",g.ksplit_autotune),
                "shared_w2":format!("{:?}",g.shared_w2_ksplit_autotune)}))
        })();
        self.graph=Some(g);
        result
    }
    fn calibrate_live_cpu(&self,g:&mut GpuDecodeGraph,next:crate::optimisations::Selection)->Result<(),String> {
        let mut sources=Vec::new();
        for (layer,moe) in g.moe_layers.iter().enumerate().filter_map(|(i,m)|m.as_ref().map(|m|(i,m))) {
            if moe.experts.is_empty(){continue;}
            let e=&moe.experts[(layer*37+11)%moe.experts.len()];
            if [e.w13_packed_ptr,e.w13_scales_ptr,e.w2_packed_ptr,e.w2_scales_ptr].contains(&0){continue;}
            sources.push(CpuTailExpertRef {format:CpuTailWeightFormat::Marlin,
                w13_packed_ptr:e.w13_packed_ptr,w13_packed_len:e.w13_packed_bytes/4,
                w13_scales_ptr:e.w13_scales_ptr,w13_scales_len:e.w13_scales_bytes/2,
                w2_packed_ptr:e.w2_packed_ptr,w2_packed_len:e.w2_packed_bytes/4,
                w2_scales_ptr:e.w2_scales_ptr,w2_scales_len:e.w2_scales_bytes/2,
                hidden_size:g.hidden_size,intermediate_size:g.moe_intermediate_size,
                group_size:g.group_size,swiglu_limit:moe.swiglu_limit,activation_alpha:1.,deepseek_clamp:moe.deepseek_v4_activation});
        }
        let count=if crate::optimisations::CPU_MORE_WORKERS && crate::cpu_tail::compiled_lane_cpu_ids(3,4).is_ok(){4}else{2};
        let mut workers=(0..count).map(|i|CpuTailRuntime::new_live_compiled(g.hidden_size,g.moe_intermediate_size,g.group_size,i,count,next.CPU_INT8_K_MAJOR())).collect::<Result<Vec<_>,_>>()?;
        let plan=crate::cpu_cold_lane::calibrate(&mut workers,&sources,g.max_experts_per_tok,g.streamed_cold.as_ref())?;
        g.cpu_tail_planned_counts=plan.counts;g.cpu_tail_calibration=Some(plan.receipt);g.cpu_tail_workers=workers;
        Ok(())
    }
}
