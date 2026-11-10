//! Disjoint startup cost model for the native cold-expert split.
pub(crate) struct Prediction {
    pub cold: usize,
    pub cpu_count: usize,
    pub critical_ms: f64,
    pub gpu_only_ms: f64,
}

/// Use directly observed complete-queue wall time; CPU and DMA already overlap
/// within each sample. Keep one GPU cold route and fail closed on invalid data.
pub(crate) fn choose_observed_queues(
    maximum_cold: usize,
    gpu_only_ms: &[f64],
    observations: &[(usize, usize, Vec<f64>)],
    handoff_ms: f64,
) -> Result<Vec<Prediction>, String> {
    if gpu_only_ms.len() != maximum_cold + 1 || gpu_only_ms.first()!=Some(&0.0)
        || !handoff_ms.is_finite() || handoff_ms < 0.0
        || gpu_only_ms.iter().enumerate().any(|(n, &v)| !v.is_finite() || v < 0.0 || (n > 0 && v == 0.0))
    {
        return Err("invalid complete-queue baseline/handoff measurements".into());
    }
    for (cpu, gpu, samples) in observations {
        if *cpu == 0 || *gpu == 0 || cpu + gpu > maximum_cold || samples.is_empty()
            || samples.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("invalid complete-queue CPU/GPU observations".into());
        }
    }
    Ok((0..=maximum_cold).map(|cold| {
        let baseline = gpu_only_ms[cold];
        let mut best = baseline;
        let mut selected = 0;
        for (cpu, gpu, samples) in observations {
            if cpu + gpu != cold { continue; }
            let expected = samples.iter().sum::<f64>() / samples.len() as f64 + handoff_ms;
            if expected < best { best = expected; selected = *cpu; }
        }
        Prediction { cold, cpu_count: selected, critical_ms: best, gpu_only_ms: baseline }
    }).collect())
}

/// Minimise expected layer latency using paired CPU/copy observations. Tail
/// measurements remain diagnostics; independently combining their tails is not
/// an expected-throughput estimate.
pub(crate) fn choose_expected(
    maximum_cold: usize,
    baseline_copy_ms: f64,
    costs: &[(usize, Vec<(f64, f64)>)],
    handoff_ms: f64,
) -> Vec<Prediction> {
    (0..=maximum_cold).map(|cold| {
        let baseline = cold as f64 * baseline_copy_ms;
        let mut best = baseline;
        let mut selected = 0;
        for (count, samples) in costs {
            if *count == 0 || *count >= cold || samples.is_empty() { continue; }
            let expected = samples.iter().map(|&(cpu_ms, copy_ms)| {
                let transferred = cpu_ms / copy_ms.max(baseline_copy_ms);
                cpu_ms + ((cold - *count) as f64 - transferred).max(0.0)
                    * baseline_copy_ms + handoff_ms
            }).sum::<f64>() / samples.len() as f64;
            if expected < best { best = expected; selected = *count; }
        }
        Prediction { cold, cpu_count: selected, critical_ms: best, gpu_only_ms: baseline }
    }).collect()
}

#[cfg(test)]
pub(crate) fn choose(
    maximum_cold: usize,
    baseline_copy_ms: f64,
    costs: &[(usize, f64, f64)],
    handoff_ms: f64,
) -> Vec<Prediction> {
    (0..=maximum_cold)
        .map(|cold| {
            let baseline = cold as f64 * baseline_copy_ms;
            let mut best = baseline;
            let mut selected = 0;
            for &(count, cpu_ms, contended_copy_ms) in costs {
                // The existing graph contract retains at least one GPU cold
                // expert. Other routes stay on their established GPU path.
                if count == 0 || count >= cold {
                    continue;
                }
                let contended = contended_copy_ms.max(baseline_copy_ms);
                // H2D and CPU run concurrently until CPU completion. Only
                // the remaining copy work then uses the uncontended rate.
                let transferred = cpu_ms / contended;
                let remaining = ((cold - count) as f64 - transferred).max(0.0);
                let proposed = cpu_ms + remaining * baseline_copy_ms + handoff_ms;
                if proposed < best {
                    best = proposed;
                    selected = count;
                }
            }
            Prediction {
                cold,
                cpu_count: selected,
                critical_ms: best,
                gpu_only_ms: baseline,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_queue_measurements_select_the_useful_partition() {
        let costs = vec![(1,1,vec![0.9,1.0]),(1,2,vec![1.7,1.5]),(2,1,vec![1.2,1.3])];
        let result = choose_observed_queues(3, &[0.0,0.6,1.2,1.8], &costs, 0.1).unwrap();
        assert_eq!(result[1].cpu_count, 0);
        assert_eq!(result[2].cpu_count, 1);
        assert_eq!(result[3].cpu_count, 2);
        assert!((result[3].critical_ms-1.35).abs()<1e-12);
        // The constant-contention estimate misses the observed speed after
        // the short GPU queue drains and CPU work can use the bandwidth.
        assert_eq!(choose_expected(2,0.6,&[(1,vec![(1.3,0.65)])],0.1)[2].cpu_count,0);
        assert_eq!(choose_observed_queues(3, &[0.0,0.6,1.2,1.8], &[(2,1,vec![20.0])],0.1).unwrap()[3].cpu_count,0);
    }
    #[test]
    fn full_queue_measurements_reject_invalid_or_cpu_only_assignments() {
        assert!(choose_observed_queues(2,&[0.0,0.6,1.2],&[(2,0,vec![0.1])],0.1).is_err());
        assert!(choose_observed_queues(2,&[0.0,0.6,1.2],&[(1,1,vec![f64::NAN])],0.1).is_err());
        assert!(choose_observed_queues(2,&[0.0,0.6],&[],0.1).is_err());
    }
    #[test]
    fn expected_cost_does_not_combine_independent_tail_events() {
        // CPU and DMA slowdowns occur on different observations. Combining
        // their maxima incorrectly rejects useful concurrent work.
        let observations = vec![(2, vec![(1.0, 3.0), (3.0, 1.0)])];
        let result = choose_expected(4, 1.0, &observations, 0.1);
        assert_eq!(result[3].cpu_count, 2);
        assert!((result[3].critical_ms - 73.0 / 30.0).abs() < 1e-12);
        assert_eq!(choose(4, 1.0, &[(2, 3.0, 3.0)], 0.1)[3].cpu_count, 0);
    }
    #[test]
    fn expected_cost_retains_gpu_for_slow_or_unavailable_cpu() {
        let result = choose_expected(8, 0.5, &[(2, vec![(20.0, 1.0)])], 0.1);
        assert!(result.iter().all(|p| p.cpu_count == 0));
        assert!(choose_expected(8, 0.5, &[], 0.1).iter().all(|p| p.cpu_count == 0));
    }
    #[test]
    fn contention_ends_when_cpu_finishes() {
        // One CPU expert takes 1 ms. During it, half a GPU expert transfers;
        // the other 6.5 finish at 1 ms each: 7.6 ms including handoff < 8 ms.
        let plan = choose(8, 1.0, &[(1, 1.0, 2.0)], 0.1);
        assert_eq!(plan[8].cpu_count, 1);
        assert!((plan[8].critical_ms - 7.6).abs() < 1e-12);
    }
    #[test]
    fn slow_cpu_retains_gpu_and_never_claims_the_only_cold_route() {
        let plan = choose(8, 1.0, &[(1, 20.0, 2.0), (2, 30.0, 3.0)], 0.1);
        assert!(plan.iter().all(|p| p.cpu_count == 0));
        assert_eq!(choose(1, 1.0, &[(1, 0.01, 1.0)], 0.0)[1].cpu_count, 0);
    }
    #[test]
    fn only_profitable_available_assignments_are_selected() {
        let plan = choose(8, 0.6, &[(1, 0.8, 0.7), (2, 1.1, 0.8)], 0.1);
        assert_eq!(plan[2].cpu_count, 1);
        assert_eq!(plan[3].cpu_count, 2);
        assert_eq!(plan[5].cpu_count, 2);
        for p in plan {
            assert!(p.cpu_count == 0 || p.cpu_count < p.cold);
        }
    }
}
