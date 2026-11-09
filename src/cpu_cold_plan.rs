//! Disjoint startup cost model for the native cold-expert split.
pub(crate) struct Prediction {
    pub cold: usize,
    pub cpu_count: usize,
    pub critical_ms: f64,
    pub gpu_only_ms: f64,
}

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
