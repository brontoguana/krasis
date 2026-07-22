//! Shadow measurement and policy planning for adaptive demand-cold expert drop.
//!
//! This module is deliberately independent of CUDA and model-specific shapes.
//! A policy protects a configurable leading fraction of router ranks, then
//! considers only experts that would require demand DMA. Eligible experts are
//! examined from lowest router weight upward and admitted to the drop plan only
//! while the configured fraction of that layer's routed mass is not exceeded.

use std::cmp::Ordering;

const SHADOW_MASS_ENV: &str = "KRASIS_ADAPTIVE_COLD_DROP_SHADOW_MASS_PCTS";
const SHADOW_PROTECT_ENV: &str = "KRASIS_ADAPTIVE_COLD_DROP_SHADOW_PROTECT_PCTS";
const RUNTIME_ENABLE_ENV: &str = "KRASIS_ADAPTIVE_COLD_DROP";
const RUNTIME_MASS_ENV: &str = "KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT";
const RUNTIME_PROTECT_ENV: &str = "KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColdDropPolicy {
    pub protect_rank_fraction: f64,
    pub max_mass_fraction: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColdDropPlan {
    pub positions: Vec<usize>,
    pub dropped_bytes: u64,
    pub dropped_mass: f64,
    pub dropped_mass_fraction: f64,
}

#[derive(Debug, Default)]
struct ShadowPolicyStats {
    policy: Option<ColdDropPolicy>,
    route_events: u64,
    selected_experts: u64,
    cold_experts: u64,
    cold_bytes: u64,
    eligible_cold_experts: u64,
    routes_with_drop: u64,
    dropped_experts: u64,
    dropped_bytes: u64,
    routed_mass: f64,
    dropped_mass: f64,
    dropped_layer_fraction_sum: f64,
    max_dropped_layer_fraction: f64,
    dropped_rank_counts: Vec<u64>,
}

impl ShadowPolicyStats {
    fn new(policy: ColdDropPolicy) -> Self {
        Self {
            policy: Some(policy),
            ..Self::default()
        }
    }

    fn reset(&mut self) {
        let policy = self.policy;
        *self = Self {
            policy,
            ..Self::default()
        };
    }

    fn record(&mut self, weights: &[f32], cold_bytes: &[u64]) {
        let policy = self.policy.expect("shadow policy must be present");
        let plan = plan_cold_drop(weights, cold_bytes, policy);
        self.record_plan(weights, cold_bytes, &plan);
    }

    fn record_plan(&mut self, weights: &[f32], cold_bytes: &[u64], plan: &ColdDropPlan) {
        let policy = self.policy.expect("policy must be present");
        let n = weights.len().min(cold_bytes.len());
        let total_mass = routed_mass(&weights[..n]);
        let protected = protected_count(n, policy.protect_rank_fraction);
        let cold = cold_bytes[..n].iter().filter(|&&bytes| bytes > 0).count();
        let eligible = cold_bytes[..n]
            .iter()
            .enumerate()
            .filter(|(rank, bytes)| *rank >= protected && **bytes > 0)
            .count();

        self.route_events = self.route_events.saturating_add(1);
        self.selected_experts = self.selected_experts.saturating_add(n as u64);
        self.cold_experts = self.cold_experts.saturating_add(cold as u64);
        self.cold_bytes = self
            .cold_bytes
            .saturating_add(cold_bytes[..n].iter().copied().sum::<u64>());
        self.eligible_cold_experts = self.eligible_cold_experts.saturating_add(eligible as u64);
        self.routed_mass += total_mass;
        self.dropped_mass += plan.dropped_mass;
        self.dropped_layer_fraction_sum += plan.dropped_mass_fraction;
        self.max_dropped_layer_fraction = self
            .max_dropped_layer_fraction
            .max(plan.dropped_mass_fraction);

        if !plan.positions.is_empty() {
            self.routes_with_drop = self.routes_with_drop.saturating_add(1);
            self.dropped_experts = self
                .dropped_experts
                .saturating_add(plan.positions.len() as u64);
            self.dropped_bytes = self.dropped_bytes.saturating_add(plan.dropped_bytes);
            if self.dropped_rank_counts.len() < n {
                self.dropped_rank_counts.resize(n, 0);
            }
            for &rank in &plan.positions {
                self.dropped_rank_counts[rank] = self.dropped_rank_counts[rank].saturating_add(1);
            }
        }
    }

    fn emit_summary(&self, generated_tokens: usize) {
        let policy = self.policy.expect("shadow policy must be present");
        let tokens = generated_tokens.max(1) as f64;
        let routes = self.route_events.max(1) as f64;
        let routed_mass = self.routed_mass.max(f64::MIN_POSITIVE);
        let rank_counts = self
            .dropped_rank_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(rank, count)| format!("{}:{}", rank + 1, count))
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "  [adaptive-cold-drop-shadow] protect_rank_pct={:.3} mass_pct={:.3} routes={} cold={} eligible={} dropped={} dropped/tok={:.3} saved_mb/tok={:.3} dropped_mass_pct={:.6} avg_layer_mass_pct={:.6} max_layer_mass_pct={:.6} routes_with_drop={} rank_counts=[{}]",
            policy.protect_rank_fraction * 100.0,
            policy.max_mass_fraction * 100.0,
            self.route_events,
            self.cold_experts,
            self.eligible_cold_experts,
            self.dropped_experts,
            self.dropped_experts as f64 / tokens,
            self.dropped_bytes as f64 / tokens / (1024.0 * 1024.0),
            self.dropped_mass / routed_mass * 100.0,
            self.dropped_layer_fraction_sum / routes * 100.0,
            self.max_dropped_layer_fraction * 100.0,
            self.routes_with_drop,
            rank_counts,
        );
        log::info!(
            "ADAPTIVE COLD DROP SHADOW generated={} protect_rank_pct={:.6} mass_pct={:.6} routes={} selected={} cold={} cold_bytes={} eligible={} routes_with_drop={} dropped={} dropped_bytes={} dropped_per_token={:.6} saved_mb_per_token={:.6} dropped_mass={} routed_mass={} dropped_mass_pct={:.9} avg_layer_mass_pct={:.9} max_layer_mass_pct={:.9} rank_counts=[{}]",
            generated_tokens,
            policy.protect_rank_fraction * 100.0,
            policy.max_mass_fraction * 100.0,
            self.route_events,
            self.selected_experts,
            self.cold_experts,
            self.cold_bytes,
            self.eligible_cold_experts,
            self.routes_with_drop,
            self.dropped_experts,
            self.dropped_bytes,
            self.dropped_experts as f64 / tokens,
            self.dropped_bytes as f64 / tokens / (1024.0 * 1024.0),
            self.dropped_mass,
            self.routed_mass,
            self.dropped_mass / routed_mass * 100.0,
            self.dropped_layer_fraction_sum / routes * 100.0,
            self.max_dropped_layer_fraction * 100.0,
            rank_counts,
        );
    }
}

#[derive(Debug, Default)]
pub struct AdaptiveColdDropRuntime {
    stats: Option<ShadowPolicyStats>,
}

impl AdaptiveColdDropRuntime {
    pub fn from_env(shadow_enabled: bool) -> Result<Self, String> {
        let enabled = parse_flag(RUNTIME_ENABLE_ENV)?;
        let mass_is_set = env_value_is_set(RUNTIME_MASS_ENV);
        let protect_is_set = env_value_is_set(RUNTIME_PROTECT_ENV);
        if !enabled {
            if mass_is_set || protect_is_set {
                return Err(format!(
                    "{} and {} require {}=1; refusing to silently enable or ignore approximate decode settings",
                    RUNTIME_MASS_ENV, RUNTIME_PROTECT_ENV, RUNTIME_ENABLE_ENV
                ));
            }
            return Ok(Self::default());
        }
        if shadow_enabled {
            return Err(format!(
                "{}=1 cannot be combined with adaptive cold-drop shadow measurement",
                RUNTIME_ENABLE_ENV
            ));
        }
        let mass_raw = std::env::var(RUNTIME_MASS_ENV)
            .map_err(|_| format!("{}=1 requires {}", RUNTIME_ENABLE_ENV, RUNTIME_MASS_ENV))?;
        let protect_raw = std::env::var(RUNTIME_PROTECT_ENV)
            .map_err(|_| format!("{}=1 requires {}", RUNTIME_ENABLE_ENV, RUNTIME_PROTECT_ENV))?;
        let masses = parse_percentage_list(RUNTIME_MASS_ENV, &mass_raw, false)?;
        let protects = parse_percentage_list(RUNTIME_PROTECT_ENV, &protect_raw, true)?;
        if masses.len() != 1 || protects.len() != 1 {
            return Err(format!(
                "runtime adaptive cold drop requires one mass percentage and one protect percentage; use the shadow variables for sweeps"
            ));
        }
        let policy = ColdDropPolicy {
            protect_rank_fraction: protects[0],
            max_mass_fraction: masses[0],
        };
        log::warn!(
            "Adaptive cold drop ENABLED: approximate decode mode protect_rank_pct={:.3} max_layer_mass_pct={:.3} renormalize=false",
            policy.protect_rank_fraction * 100.0,
            policy.max_mass_fraction * 100.0,
        );
        Ok(Self {
            stats: Some(ShadowPolicyStats::new(policy)),
        })
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.stats.is_some()
    }

    pub fn reset(&mut self) {
        if let Some(stats) = &mut self.stats {
            stats.reset();
        }
    }

    pub fn plan_and_record(&mut self, weights: &[f32], cold_bytes: &[u64]) -> ColdDropPlan {
        let Some(stats) = &mut self.stats else {
            return ColdDropPlan::default();
        };
        let policy = stats.policy.expect("runtime policy must be present");
        let plan = plan_cold_drop(weights, cold_bytes, policy);
        stats.record_plan(weights, cold_bytes, &plan);
        plan
    }

    pub fn dropped_experts(&self) -> u64 {
        self.stats
            .as_ref()
            .map(|stats| stats.dropped_experts)
            .unwrap_or(0)
    }

    pub fn emit_summary(&self, generated_tokens: usize) {
        let Some(stats) = &self.stats else {
            return;
        };
        let policy = stats.policy.expect("runtime policy must be present");
        let tokens = generated_tokens.max(1) as f64;
        let routes = stats.route_events.max(1) as f64;
        let routed_mass = stats.routed_mass.max(f64::MIN_POSITIVE);
        let rank_counts = stats
            .dropped_rank_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(rank, count)| format!("{}:{}", rank + 1, count))
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "  [adaptive-cold-drop] protect_rank_pct={:.3} mass_pct={:.3} cold_before_drop={} dropped={} dropped/tok={:.3} saved_mb/tok={:.3} dropped_mass_pct={:.6} avg_layer_mass_pct={:.6} max_layer_mass_pct={:.6} rank_counts=[{}]",
            policy.protect_rank_fraction * 100.0,
            policy.max_mass_fraction * 100.0,
            stats.cold_experts,
            stats.dropped_experts,
            stats.dropped_experts as f64 / tokens,
            stats.dropped_bytes as f64 / tokens / (1024.0 * 1024.0),
            stats.dropped_mass / routed_mass * 100.0,
            stats.dropped_layer_fraction_sum / routes * 100.0,
            stats.max_dropped_layer_fraction * 100.0,
            rank_counts,
        );
        log::info!(
            "ADAPTIVE COLD DROP SUMMARY generated={} protect_rank_pct={:.6} mass_pct={:.6} routes={} cold_before_drop={} cold_bytes_before_drop={} dropped={} dropped_bytes={} dropped_per_token={:.6} saved_mb_per_token={:.6} dropped_mass={} routed_mass={} dropped_mass_pct={:.9} avg_layer_mass_pct={:.9} max_layer_mass_pct={:.9} renormalize=false rank_counts=[{}]",
            generated_tokens,
            policy.protect_rank_fraction * 100.0,
            policy.max_mass_fraction * 100.0,
            stats.route_events,
            stats.cold_experts,
            stats.cold_bytes,
            stats.dropped_experts,
            stats.dropped_bytes,
            stats.dropped_experts as f64 / tokens,
            stats.dropped_bytes as f64 / tokens / (1024.0 * 1024.0),
            stats.dropped_mass,
            stats.routed_mass,
            stats.dropped_mass / routed_mass * 100.0,
            stats.dropped_layer_fraction_sum / routes * 100.0,
            stats.max_dropped_layer_fraction * 100.0,
            rank_counts,
        );
    }
}

#[derive(Debug, Default)]
pub struct AdaptiveColdDropShadow {
    policies: Vec<ShadowPolicyStats>,
}

impl AdaptiveColdDropShadow {
    pub fn from_env() -> Result<Self, String> {
        let mass_raw = match std::env::var(SHADOW_MASS_ENV) {
            Ok(value) if !value.trim().is_empty() => value,
            _ => return Ok(Self::default()),
        };
        let protect_raw = std::env::var(SHADOW_PROTECT_ENV).map_err(|_| {
            format!(
                "{} requires {} so rank protection is explicit",
                SHADOW_MASS_ENV, SHADOW_PROTECT_ENV
            )
        })?;
        let masses = parse_percentage_list(SHADOW_MASS_ENV, &mass_raw, false)?;
        let protects = parse_percentage_list(SHADOW_PROTECT_ENV, &protect_raw, true)?;
        let mut policies = Vec::with_capacity(masses.len().saturating_mul(protects.len()));
        for protect_rank_fraction in protects {
            for &max_mass_fraction in &masses {
                policies.push(ShadowPolicyStats::new(ColdDropPolicy {
                    protect_rank_fraction,
                    max_mass_fraction,
                }));
            }
        }
        log::warn!(
            "Adaptive cold-drop shadow enabled with {} policy combinations; routes and outputs are unchanged",
            policies.len()
        );
        Ok(Self { policies })
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        !self.policies.is_empty()
    }

    pub fn reset(&mut self) {
        for policy in &mut self.policies {
            policy.reset();
        }
    }

    pub fn record(&mut self, weights: &[f32], cold_bytes: &[u64]) {
        for policy in &mut self.policies {
            policy.record(weights, cold_bytes);
        }
    }

    pub fn emit_summary(&self, generated_tokens: usize) {
        for policy in &self.policies {
            policy.emit_summary(generated_tokens);
        }
    }
}

pub fn plan_cold_drop(weights: &[f32], cold_bytes: &[u64], policy: ColdDropPolicy) -> ColdDropPlan {
    let n = weights.len().min(cold_bytes.len());
    let total_mass = routed_mass(&weights[..n]);
    if n == 0 || total_mass <= 0.0 || policy.max_mass_fraction <= 0.0 {
        return ColdDropPlan::default();
    }
    let protected = protected_count(n, policy.protect_rank_fraction);
    let mut candidates = (protected..n)
        .filter(|&rank| cold_bytes[rank] > 0)
        .collect::<Vec<_>>();
    candidates.sort_by(|&left, &right| {
        safe_weight(weights[left])
            .partial_cmp(&safe_weight(weights[right]))
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.cmp(&left))
    });

    let mass_budget = total_mass * policy.max_mass_fraction;
    // Router weights originate as f32. Compare an accumulated f64 view with a
    // tolerance at the source precision so an exact declared boundary (for
    // example 0.06 + 0.04 within a 0.10 budget) is not rejected by conversion
    // noise. This tolerance scales with the route's own total mass.
    let mass_tolerance = f64::from(f32::EPSILON) * total_mass.max(1.0);
    let mut plan = ColdDropPlan::default();
    for rank in candidates {
        let mass = safe_weight(weights[rank]);
        if plan.dropped_mass + mass > mass_budget + mass_tolerance {
            continue;
        }
        plan.positions.push(rank);
        plan.dropped_bytes = plan.dropped_bytes.saturating_add(cold_bytes[rank]);
        plan.dropped_mass += mass;
    }
    plan.positions.sort_unstable();
    plan.dropped_mass_fraction = plan.dropped_mass / total_mass;
    plan
}

fn protected_count(topk: usize, protect_rank_fraction: f64) -> usize {
    ((topk as f64) * protect_rank_fraction)
        .ceil()
        .clamp(0.0, topk as f64) as usize
}

fn routed_mass(weights: &[f32]) -> f64 {
    weights.iter().map(|&weight| safe_weight(weight)).sum()
}

fn safe_weight(weight: f32) -> f64 {
    if weight.is_finite() && weight > 0.0 {
        weight as f64
    } else {
        0.0
    }
}

fn parse_percentage_list(name: &str, raw: &str, allow_zero: bool) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err(format!("{} contains an empty percentage", name));
        }
        let value = item
            .parse::<f64>()
            .map_err(|error| format!("invalid {} percentage {:?}: {}", name, item, error))?;
        let lower_ok = if allow_zero {
            value >= 0.0
        } else {
            value > 0.0
        };
        if !value.is_finite() || !lower_ok || value >= 100.0 {
            return Err(format!(
                "{} percentage must be {} and <100, got {}",
                name,
                if allow_zero { ">=0" } else { ">0" },
                value
            ));
        }
        let fraction = value / 100.0;
        if !values.contains(&fraction) {
            values.push(fraction);
        }
    }
    if values.is_empty() {
        return Err(format!("{} must contain at least one percentage", name));
    }
    Ok(values)
}

fn env_value_is_set(name: &str) -> bool {
    std::env::var(name)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn parse_flag(name: &str) -> Result<bool, String> {
    let Ok(raw) = std::env::var(name) else {
        return Ok(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        _ => Err(format!(
            "{} must be one of 1/0, true/false, yes/no, or on/off; got {:?}",
            name, raw
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protects_head_and_drops_lowest_cold_mass_first() {
        let weights = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04];
        let cold = [0, 7, 7, 7, 7, 7];
        let plan = plan_cold_drop(
            &weights,
            &cold,
            ColdDropPolicy {
                protect_rank_fraction: 0.5,
                max_mass_fraction: 0.10,
            },
        );
        assert_eq!(plan.positions, vec![4, 5]);
        assert_eq!(plan.dropped_bytes, 14);
        assert!((plan.dropped_mass_fraction - 0.10).abs() < 1.0e-6);
    }

    #[test]
    fn hot_tail_never_consumes_drop_budget() {
        let weights = [0.50, 0.20, 0.15, 0.10, 0.05];
        let cold = [0, 0, 0, 9, 0];
        let plan = plan_cold_drop(
            &weights,
            &cold,
            ColdDropPolicy {
                protect_rank_fraction: 0.4,
                max_mass_fraction: 0.10,
            },
        );
        assert_eq!(plan.positions, vec![3]);
        assert_eq!(plan.dropped_bytes, 9);
    }

    #[test]
    fn mass_budget_is_relative_when_router_weights_are_not_normalized() {
        let weights = [2.0, 1.0, 0.5, 0.3, 0.2];
        let cold = [1, 1, 1, 1, 1];
        let plan = plan_cold_drop(
            &weights,
            &cold,
            ColdDropPolicy {
                protect_rank_fraction: 0.4,
                max_mass_fraction: 0.125,
            },
        );
        assert_eq!(plan.positions, vec![3, 4]);
        assert!((plan.dropped_mass_fraction - 0.125).abs() < 1.0e-6);
    }

    #[test]
    fn runtime_knobs_without_explicit_enable_fail_closed() {
        let previous_enable = std::env::var(RUNTIME_ENABLE_ENV).ok();
        let previous_mass = std::env::var(RUNTIME_MASS_ENV).ok();
        let previous_protect = std::env::var(RUNTIME_PROTECT_ENV).ok();
        std::env::remove_var(RUNTIME_ENABLE_ENV);
        std::env::set_var(RUNTIME_MASS_ENV, "5");
        std::env::set_var(RUNTIME_PROTECT_ENV, "75");
        let result = AdaptiveColdDropRuntime::from_env(false);
        assert!(result.is_err());
        restore_env(RUNTIME_ENABLE_ENV, previous_enable);
        restore_env(RUNTIME_MASS_ENV, previous_mass);
        restore_env(RUNTIME_PROTECT_ENV, previous_protect);
    }

    fn restore_env(name: &str, value: Option<String>) {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}
