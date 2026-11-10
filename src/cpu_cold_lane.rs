//! Cold-start calibration for an upfront CPU/GPU split over existing host weights.
use crate::cpu_tail::{CpuTailExpertRef, CpuTailRuntime};
use cudarc::driver::sys as cuda;
use serde_json::{json, Value};
use std::time::Instant;

pub(crate) struct Plan {
    pub counts: Vec<usize>,
    pub receipt: Value,
}

fn checked(result: cuda::CUresult, label: &str) -> Result<(), String> {
    if result == cuda::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CPU lane calibration {label}: {result:?}"))
    }
}
fn sizes(expert: &CpuTailExpertRef) -> Result<[usize; 4], String> {
    Ok([
        expert
            .w13_packed_len
            .checked_mul(4)
            .ok_or("CPU lane W13 extent overflow")?,
        expert
            .w13_scales_len
            .checked_mul(2)
            .ok_or("CPU lane W13 scale extent overflow")?,
        expert
            .w2_packed_len
            .checked_mul(4)
            .ok_or("CPU lane W2 extent overflow")?,
        expert
            .w2_scales_len
            .checked_mul(2)
            .ok_or("CPU lane W2 scale extent overflow")?,
    ])
}
struct Scratch<'a> {
    tier: Option<&'a crate::streamed_cold::Runtime>,
    stream: cuda::CUstream,
    begin: cuda::CUevent,
    end: cuda::CUevent,
    pointer: u64,
    bytes: usize,
}
impl<'a> Scratch<'a> {
    fn new(bytes: usize, tier: Option<&'a crate::streamed_cold::Runtime>) -> Result<Self, String> {
        let mut out = Self {
            tier,
            stream: std::ptr::null_mut(),
            begin: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            pointer: 0,
            bytes,
        };
        unsafe {
            checked(
                cuda::lib().cuStreamCreate(
                    &mut out.stream,
                    cuda::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                ),
                "create stream",
            )?;
            checked(
                cuda::lib().cuEventCreate(&mut out.begin, 0),
                "create begin event",
            )?;
            checked(
                cuda::lib().cuEventCreate(&mut out.end, 0),
                "create end event",
            )?;
            let (mut free, mut total) = (0usize, 0usize);
            checked(
                cuda::lib().cuMemGetInfo_v2(&mut free, &mut total),
                "query reserve",
            )?;
            let required = bytes
                .checked_add(600 * 1024 * 1024)
                .ok_or("CPU lane reserve overflow")?;
            if free < required {
                return Err("CPU lane scratch would violate 600 MiB reserve".into());
            }
            checked(
                cuda::lib().cuMemAlloc_v2(&mut out.pointer, bytes),
                "allocate scratch",
            )?;
        }
        Ok(out)
    }
    fn enqueue(&self, expert: &CpuTailExpertRef, timed: bool) -> Result<(), String> {
        if let Some(tier) = self.tier.filter(|_| crate::optimisations::STREAMED_COLD_INT3) {
            unsafe { if timed { checked(cuda::lib().cuEventRecord(self.begin,self.stream),"compact begin")?; } }
            tier.enqueue(expert.w13_packed_ptr,self.pointer,self.stream)?;
            unsafe { if timed { checked(cuda::lib().cuEventRecord(self.end,self.stream),"compact end")?; } }
            return Ok(());
        }
        let lengths = sizes(expert)?;
        let hosts = [
            expert.w13_packed_ptr,
            expert.w13_scales_ptr,
            expert.w2_packed_ptr,
            expert.w2_scales_ptr,
        ];
        let mut offset = 0usize;
        unsafe {
            if timed {
                checked(
                    cuda::lib().cuEventRecord(self.begin, self.stream),
                    "record begin",
                )?;
            }
            for (host, bytes) in hosts.into_iter().zip(lengths) {
                if host == 0 || bytes == 0 {
                    return Err("CPU lane source is absent".into());
                }
                let next = offset
                    .checked_add(bytes)
                    .ok_or("CPU lane copy extent overflow")?;
                if next > self.bytes {
                    return Err("CPU lane scratch extent exceeded".into());
                }
                checked(
                    cuda::lib().cuMemcpyHtoDAsync_v2(
                        self.pointer + offset as u64,
                        host as *const std::ffi::c_void,
                        bytes,
                        self.stream,
                    ),
                    "weight copy",
                )?;
                offset = next;
            }
            if timed {
                checked(
                    cuda::lib().cuEventRecord(self.end, self.stream),
                    "record end",
                )?;
            }
        }
        Ok(())
    }
    fn drain(&self) -> Result<f64, String> {
        let mut ms = 0f32;
        unsafe {
            checked(
                cuda::lib().cuStreamSynchronize(self.stream),
                "drain owned stream",
            )?;
            checked(
                cuda::lib().cuEventElapsedTime(&mut ms, self.begin, self.end),
                "copy duration",
            )?;
        }
        if !ms.is_finite() || ms <= 0.0 {
            return Err("CPU lane invalid copy duration".into());
        }
        Ok(ms as f64)
    }
    fn enqueue_queue(&self, experts: &[CpuTailExpertRef]) -> Result<(), String> {
        if experts.is_empty() { return Err("CPU lane GPU queue is empty".into()); }
        unsafe { checked(cuda::lib().cuEventRecord(self.begin,self.stream),"queue begin")?; }
        for expert in experts { self.enqueue(expert,false)?; }
        unsafe { checked(cuda::lib().cuEventRecord(self.end,self.stream),"queue end")?; }
        Ok(())
    }
}
impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        unsafe {
            // The stream is private and all host/model sources outlive this owner.
            if !self.stream.is_null() {
                let _ = cuda::lib().cuStreamSynchronize(self.stream);
            }
            if self.pointer != 0 {
                let _ = cuda::lib().cuMemFree_v2(self.pointer);
            }
            if !self.begin.is_null() {
                let _ = cuda::lib().cuEventDestroy_v2(self.begin);
            }
            if !self.end.is_null() {
                let _ = cuda::lib().cuEventDestroy_v2(self.end);
            }
            if !self.stream.is_null() {
                let _ = cuda::lib().cuStreamDestroy_v2(self.stream);
            }
        }
    }
}
fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction).ceil() as usize]
}

/// Measure the queue that will actually remain on the GPU. In particular,
/// CPU service may accelerate after a short DMA queue has finished.
fn calibrate_full_queues(
    workers: &mut [CpuTailRuntime],
    experts: &[CpuTailExpertRef],
    maximum_cold: usize,
    scratch: &Scratch,
    bytes: usize,
) -> Result<Plan, String> {
    let samples = experts.len().min(40);
    let mut baseline_samples = vec![Vec::new(); maximum_cold + 1];
    let mut baselines = vec![0.0; maximum_cold + 1];
    for gpu in 1..=maximum_cold {
        for sample in 0..samples {
            let queue = (0..gpu).map(|i| experts[(sample+i)%experts.len()]).collect::<Vec<_>>();
            let start = Instant::now();
            scratch.enqueue_queue(&queue)?;
            scratch.drain()?;
            baseline_samples[gpu].push(start.elapsed().as_secs_f64()*1000.0);
        }
        baselines[gpu] = baseline_samples[gpu].iter().sum::<f64>()/samples as f64;
    }
    let mut measurements = Vec::new();
    let mut costs = Vec::new();
    for cpu in 1..=workers.len().min(maximum_cold.saturating_sub(1)) {
        for gpu in 1..=maximum_cold-cpu {
            let mut walls = Vec::new();
            let mut details = Vec::new();
            for sample in 0..samples {
                let mut cpu_indices = Vec::new();
                for ordinal in 0..cpu {
                    let mut index = (sample+ordinal*19)%experts.len();
                    while cpu_indices.contains(&index) { index = (index+1)%experts.len(); }
                    cpu_indices.push(index);
                }
                let gpu_indices = (0..experts.len())
                    .map(|i| (sample+11+i)%experts.len())
                    .filter(|i| !cpu_indices.contains(i))
                    .take(gpu).collect::<Vec<_>>();
                if gpu_indices.len()!=gpu { return Err("CPU lane lacks distinct GPU queue sources".into()); }
                let queue = gpu_indices.iter().map(|&i| experts[i]).collect::<Vec<_>>();
                let mut sequences = Vec::with_capacity(cpu);
                let mut offsets = Vec::with_capacity(cpu);
                let mut raw_worker_rows = Vec::with_capacity(cpu);
                let start = Instant::now();
                for (ordinal, worker) in workers.iter_mut().take(cpu).enumerate() {
                    offsets.push(start.elapsed().as_secs_f64()*1000.0);
                    sequences.push(worker.submit(experts[cpu_indices[ordinal]],0,Some(Instant::now()))?);
                }
                let gpu_offset = start.elapsed().as_secs_f64()*1000.0;
                scratch.enqueue_queue(&queue)?;
                let mut cpu_max = 0.0f64;
                for (ordinal, worker) in workers.iter_mut().take(cpu).enumerate() {
                    let result = worker.wait_result()?;
                    if result.sequence!=sequences[ordinal] || !result.completed {
                        return Err("CPU queue calibration sequence/cancellation mismatch".into());
                    }
                    if let Some(error) = result.error { return Err(error); }
                    let total = (result.dispatch_lag_s+result.elapsed_s+result.compute_to_result_visible_s)*1000.0;
                    if !total.is_finite() || total<=0.0 { return Err("CPU queue timing is absent/invalid".into()); }
                    cpu_max = cpu_max.max(offsets[ordinal]+total);
                    raw_worker_rows.push((total,offsets[ordinal],result.elapsed_s*1000.0,
                        result.dispatch_lag_s*1000.0,result.compute_to_result_visible_s*1000.0));
                    worker.finish(result.sequence)?;
                }
                let gpu_elapsed = scratch.drain()?;
                // Both lanes and all result handoffs have completed. Never
                // add overlapping CPU and DMA service intervals together.
                let wall = start.elapsed().as_secs_f64()*1000.0;
                if !wall.is_finite() || wall<=0.0 { return Err("CPU queue wall time is invalid".into()); }
                walls.push(wall);
                let worker_rows = raw_worker_rows.into_iter().map(|(total,offset,compute,dispatch,visible)|
                    json!({"total_ms":total,"dispatch_offset_ms":offset,"compute_ms":compute,
                        "dispatch_ms":dispatch,"result_visible_ms":visible})).collect::<Vec<_>>();
                details.push(json!({"source_sample":sample,"cpu_source_indices":cpu_indices,
                    "gpu_source_indices":gpu_indices,"worker_ms":worker_rows,"cpu_max_ms":cpu_max,
                    "gpu_dispatch_offset_ms":gpu_offset,"gpu_queue_ms":gpu_elapsed,"paired_wall_ms":wall}));
            }
            costs.push((cpu,gpu,walls));
            measurements.push(json!({"cpu_experts":cpu,"gpu_experts":gpu,"samples":details}));
        }
    }
    let handoff_ms = 0.1;
    let estimates = crate::cpu_cold_plan::choose_observed_queues(maximum_cold,&baselines,&costs,handoff_ms)?;
    let counts = estimates.iter().map(|p|p.cpu_count).collect::<Vec<_>>();
    let predictions = estimates.iter().map(|p|json!({"cold":p.cold,"cpu_count":p.cpu_count,
        "predicted_ms":p.critical_ms,"gpu_only_copy_ms":p.gpu_only_ms})).collect::<Vec<_>>();
    Ok(Plan { counts:counts.clone(), receipt:json!({
        "schema":3,"cost_model":"complete_cpu_gpu_queues","source_experts":experts.len(),
        "expert_bytes":bytes,"samples":samples,
        "geometry":{"hidden":experts[0].hidden_size,"intermediate":experts[0].intermediate_size,"group_size":experts[0].group_size},
        "deepseek_clamped_sources":experts.iter().filter(|e|e.deepseek_clamp).count(),
        "gpu_only_queue_samples_ms":baseline_samples,"gpu_only_queue_mean_ms":baselines,
        "selection_objective":"mean paired complete-queue wall time","handoff_allowance_ms":handoff_ms,
        "measurements":measurements,"predictions":predictions,"counts":counts,
        "ownership":"existing immutable host weights; private startup scratch drained before destruction",
        "adq":false
    }) })
}
/// Inputs are immutable registered model allocations, sampled across layers.
/// Worker pools are already bound to discovered, disjoint allowed cores.
/// There is no extra CPU weight representation or persistent GPU scratch.
pub(crate) fn calibrate(
    workers: &mut [CpuTailRuntime],
    experts: &[CpuTailExpertRef],
    maximum_cold: usize,
    tier: Option<&crate::streamed_cold::Runtime>,
) -> Result<Plan, String> {
    if workers.is_empty()
        || workers.len() > 4
        || experts.len() < 24
        || maximum_cold == 0
        || maximum_cold > 64
    {
        return Err("CPU lane calibration lacks supported workers/geometry/sources".into());
    }
    for worker in workers.iter_mut() {
        if !worker.is_idle()? {
            return Err("CPU lane calibration worker is busy".into());
        }
        let pointer = worker.input_ptr() as *mut u16;
        for index in 0..worker.hidden_size() {
            let value = ((index * 197 + 31) % 2048) as f32 / 1024.0 - 1.0;
            unsafe {
                *pointer.add(index) = crate::weights::marlin::f32_to_bf16(value);
            }
        }
    }
    let bytes = sizes(&experts[0])?.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b).ok_or("CPU lane size overflow")
    })?;
    if experts
        .iter()
        .any(|e| sizes(e).ok().map(|x| x.into_iter().sum::<usize>()) != Some(bytes))
    {
        return Err("CPU lane calibration needs equal loaded expert geometry".into());
    }
    let scratch = Scratch::new(bytes,tier)?;
    if crate::optimisations::CPU_FULL_QUEUE_PLANNER && maximum_cold <= 8 {
        return calibrate_full_queues(workers,experts,maximum_cold,&scratch,bytes);
    }
    let samples = experts.len().min(40);
    let mut dma_baseline = Vec::new();
    for expert in experts.iter().take(samples) {
        scratch.enqueue(expert, true)?;
        dma_baseline.push(scratch.drain()?);
    }
    let baseline_ms = percentile(&dma_baseline, 0.5);
    let baseline_mean_ms = dma_baseline.iter().sum::<f64>() / dma_baseline.len() as f64;
    let mut measurements = Vec::new();
    let mut costs = Vec::new();
    for count in 1..=workers.len() {
        let mut cpu_times = Vec::new();
        let mut dma_times = Vec::new();
        let mut details = Vec::new();
        for index in 0..samples {
            let mut sequences = Vec::new();
            for (ordinal, worker) in workers.iter_mut().take(count).enumerate() {
                let expert = experts[(index + ordinal * 19) % experts.len()];
                sequences.push(worker.submit(expert, 0, Some(Instant::now()))?);
            }
            // Keep H2D active throughout useful CPU calls. The first complete
            // copy is timed while the workers are already processing, retaining
            // host-bandwidth contention in the split cost model.
            for copy in 0..16 {
                scratch.enqueue(&experts[(index + copy + 11) % experts.len()], copy == 0)?;
            }
            let mut critical = 0f64;
            let mut worker_rows = Vec::new();
            for (ordinal, worker) in workers.iter_mut().take(count).enumerate() {
                let result = worker.wait_result()?;
                if result.sequence != sequences[ordinal] || !result.completed {
                    return Err("CPU lane calibration sequence/cancellation mismatch".into());
                }
                if let Some(error) = result.error {
                    return Err(error);
                }
                let ms =
                    (result.dispatch_lag_s + result.elapsed_s + result.compute_to_result_visible_s)
                        * 1000.0;
                if !ms.is_finite() || ms <= 0.0 {
                    return Err("CPU lane worker timing is absent/invalid".into());
                }
                critical = critical.max(ms);
                worker_rows.push(json!({"total_ms":ms,"compute_ms":result.elapsed_s*1000.0,
                    "dispatch_ms":result.dispatch_lag_s*1000.0,
                    "result_visible_ms":result.compute_to_result_visible_s*1000.0}));
                worker.finish(result.sequence)?;
            }
            let dma = scratch.drain()?;
            cpu_times.push(critical);
            dma_times.push(dma);
            details
                .push(json!({"source_sample":index,"worker_ms":worker_rows,"first_copy_ms":dma}));
        }
        let cpu_p95 = percentile(&cpu_times, 0.95);
        let dma_p95 = percentile(&dma_times, 0.95).max(baseline_ms);
        costs.push((count, cpu_times.iter().copied().zip(dma_times.iter().copied()).collect()));
        measurements.push(json!({"cpu_experts":count,"cpu_p95_ms":cpu_p95,"contended_copy_p95_ms":dma_p95,"samples":details}));
    }
    // Include an explicit activation/result handoff allowance. Measured CPU
    // time already includes dispatch and result visibility, not only GEMV.
    let handoff_ms = 0.1;
    let estimates = crate::cpu_cold_plan::choose_expected(maximum_cold, baseline_mean_ms, &costs, handoff_ms);
    let counts = estimates.iter().map(|p| p.cpu_count).collect::<Vec<_>>();
    let predictions = estimates
        .iter()
        .map(|p| {
            json!({"cold":p.cold,"cpu_count":p.cpu_count,
        "predicted_ms":p.critical_ms,"gpu_only_copy_ms":p.gpu_only_ms})
        })
        .collect::<Vec<_>>();
    Ok(Plan {
        counts: counts.clone(),
        receipt: json!({
            "schema":2,"source_experts":experts.len(),"expert_bytes":bytes,"samples":samples,
            "geometry":{"hidden":experts[0].hidden_size,"intermediate":experts[0].intermediate_size,"group_size":experts[0].group_size},
            "deepseek_clamped_sources":experts.iter().filter(|e|e.deepseek_clamp).count(),
            "dma_baseline_ms":dma_baseline,"baseline_copy_median_ms":baseline_ms,
            "baseline_copy_mean_ms":baseline_mean_ms,"selection_objective":"mean paired critical-path latency",
            "handoff_allowance_ms":handoff_ms,"measurements":measurements,"predictions":predictions,"counts":counts,
            "ownership":"existing immutable host weights; private startup scratch drained before destruction",
            "adq":false
        }),
    })
}
