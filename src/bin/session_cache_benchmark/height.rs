use super::{
    diagnostic::source_json,
    evidence::{save, Client},
};
use serde_json::{json, Value};
use std::path::Path;

fn input_ids(value: &Value) -> Result<Vec<Value>, String> {
    let ids = value
        .pointer("/krasis_debug/input_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing authoritative input IDs")?;
    if ids.len() < 2 || ids.iter().any(|id| id.as_u64().is_none()) {
        return Err("invalid input IDs".into());
    }
    Ok(ids.clone())
}

fn scores(position: &Value) -> Result<Vec<(u64, f64)>, String> {
    let entries = position
        .get("top_k")
        .and_then(Value::as_array)
        .ok_or("missing top-k")?;
    entries
        .iter()
        .map(|entry| {
            let id = entry
                .get("token_id")
                .and_then(Value::as_u64)
                .ok_or("missing token ID")?;
            let score = entry
                .get("log_prob")
                .or_else(|| entry.get("logprob"))
                .and_then(Value::as_f64)
                .filter(|score| score.is_finite())
                .ok_or("missing finite log-probability")?;
            Ok((id, score))
        })
        .collect()
}

pub fn run(client: &mut Client, source: &Path, regression: bool) -> Result<(), String> {
    let base_ids = input_ids(&source_json(source, 0, "response")?)?;
    let longer_ids = input_ids(&source_json(source, 2, "response")?)?;
    let shared = base_ids
        .iter()
        .zip(&longer_ids)
        .take_while(|(a, b)| a == b)
        .count();
    if shared < 2 {
        return Err("no useful shared prefix".into());
    }
    // Same bounded shared-position sampling as the existing live cache gate.
    let sample_every = ((shared - 1) / 32).max(1);
    client.context = json!({"diagnostic": "cache_disabled_height_baseline"});
    let before = client.json("GET", "/v1/session-cache/stats", None)?;
    client.context =
        json!({"diagnostic": "base_full_prefill_logits", "input_tokens":base_ids.len()});
    let base = client.json(
        "POST",
        "/v1/internal/prefill_logits",
        Some(&json!({"input_token_ids":base_ids,"top_k":10,"sample_every":sample_every})),
    )?;
    client.context =
        json!({"diagnostic": "longer_full_prefill_logits", "input_tokens":longer_ids.len()});
    let longer = client.json(
        "POST",
        "/v1/internal/prefill_logits",
        Some(&json!({"input_token_ids":longer_ids,"top_k":10,"sample_every":sample_every})),
    )?;
    let distinct_ids = input_ids(&source_json(source, 1, "response")?)?;
    client.context = json!({"diagnostic":"distinct_full_prefill_before_repeat","input_tokens":distinct_ids.len()});
    client.json(
        "POST",
        "/v1/internal/prefill_logits",
        Some(&json!({
        "input_token_ids":distinct_ids,"top_k":10,"sample_every":sample_every})),
    )?;
    // Repeating the same geometry after another request distinguishes stale
    // recurrent state from shape-dependent arithmetic.
    client.context =
        json!({"diagnostic":"base_full_prefill_repeat_after_longer","input_tokens":base_ids.len()});
    let repeated = client.json(
        "POST",
        "/v1/internal/prefill_logits",
        Some(&json!({"input_token_ids":base_ids,"top_k":10,"sample_every":sample_every})),
    )?;
    save(
        &client.root.join("identical-height-repeat.json"),
        &serde_json::to_vec_pretty(&json!({"diagnostic_only":true,
            "positions_exact":base["positions"]==repeated["positions"],
            "base_positions":base["positions"],"repeat_positions":repeated["positions"]}))
        .map_err(|e| e.to_string())?,
    )?;
    let after = client.json("GET", "/v1/session-cache/stats", None)?;
    for stats in [&before, &after] {
        if !stats.get("hits").is_some_and(Value::is_object)
            || stats
                .pointer("/resident/snapshots")
                .and_then(Value::as_u64)
                .is_none()
        {
            return Err("cache statistics contract missing".into());
        }
    }
    if before["hits"] != after["hits"]
        || before["resident"]["snapshots"] != after["resident"]["snapshots"]
    {
        return Err("cache-disabled logits calls changed cache hits or snapshot count".into());
    }
    let left_positions = base
        .get("positions")
        .and_then(Value::as_array)
        .ok_or("base logits missing positions")?;
    let right_positions = longer
        .get("positions")
        .and_then(Value::as_array)
        .ok_or("longer logits missing positions")?;
    let mut comparisons = Vec::new();
    for left in left_positions {
        let position = left
            .get("position")
            .and_then(Value::as_u64)
            .ok_or("invalid base position")?;
        if position as usize >= shared {
            continue;
        }
        let Some(right) = right_positions
            .iter()
            .find(|right| right.get("position").and_then(Value::as_u64) == Some(position))
        else {
            continue;
        };
        let a = scores(left)?;
        let b = scores(right)?;
        if a.is_empty() || b.is_empty() {
            return Err("empty shared-position distribution".into());
        }
        let deltas:Vec<_> = a.iter().filter_map(|(id,score)| b.iter().find(|(other,_)|id==other).map(|(_,other)|json!({"token_id":id,"base_log_prob":score,"longer_log_prob":other,"absolute_delta":(score-other).abs()}))).collect();
        if deltas.is_empty() {
            return Err("no shared top-k tokens".into());
        }
        let max_delta = deltas
            .iter()
            .filter_map(|v| v["absolute_delta"].as_f64())
            .fold(0.0, f64::max);
        comparisons.push(json!({"position":position,"top_token_changed":a[0].0 != b[0].0,"top_k_overlap":deltas.len(),"max_shared_log_prob_delta":max_delta,"shared_token_deltas":deltas}));
    }
    if comparisons.is_empty() {
        return Err("no shared sampled causal positions".into());
    }
    let max_delta = comparisons
        .iter()
        .filter_map(|v| v["max_shared_log_prob_delta"].as_f64())
        .fold(0.0, f64::max);
    let result = json!({"diagnostic_only":true,"clears_original_failed_lifecycle":false,"cache_path":"disabled_internal_full_prefill","base_input_tokens":base_ids.len(),"longer_input_tokens":longer_ids.len(),"shared_prefix_tokens":shared,"sample_every":sample_every,"compared_positions":comparisons.len(),"top_token_changed_positions":comparisons.iter().filter(|v|v["top_token_changed"]==true).count(),"max_shared_log_prob_delta":max_delta,"positions":comparisons});
    save(
        &client.root.join("height-comparisons.json"),
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    if regression && base["positions"] != repeated["positions"] {
        return Err("identical prefill logits changed after a distinct request".into());
    }
    Ok(())
}

pub fn trace(client: &mut Client, source: &Path) -> Result<(), String> {
    let base = input_ids(&source_json(source, 0, "response")?)?;
    let longer = input_ids(&source_json(source, 2, "response")?)?;
    trace_pair(client, base, longer)
}

// BEGIN authoritative boundary capture
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryCapturePlan {
    source: std::path::PathBuf,
    prefix_index: u64,
    full_index: u64,
    expected_prefix_tokens: usize,
    expected_full_tokens: usize,
    prefix_token_ids_sha256: String,
    full_token_ids_sha256: String,
}

fn token_ids_sha256(ids: &[Value]) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for id in ids {
        let value = id
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("authoritative token is not a u32")?;
        hash.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn admit_boundaries(
    plan: &BoundaryCapturePlan,
    prefix: &[Value],
    full: &[Value],
) -> Result<(), String> {
    if !plan.source.is_absolute()
        || plan.prefix_index == plan.full_index
        || prefix.is_empty()
        || prefix.len() >= full.len()
        || prefix.len() != plan.expected_prefix_tokens
        || full.len() != plan.expected_full_tokens
        || !full.starts_with(prefix)
        || token_ids_sha256(prefix)? != plan.prefix_token_ids_sha256
        || token_ids_sha256(full)? != plan.full_token_ids_sha256
    {
        return Err("authoritative boundary plan/input identity mismatch".into());
    }
    Ok(())
}

pub fn capture_boundaries(client: &mut Client, plan_path: &Path) -> Result<(), String> {
    let plan: BoundaryCapturePlan =
        serde_json::from_slice(&std::fs::read(plan_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let prefix = input_ids(&source_json(&plan.source, plan.prefix_index, "response")?)?;
    let full = input_ids(&source_json(&plan.source, plan.full_index, "response")?)?;
    admit_boundaries(&plan, &prefix, &full)?;
    for (role, index, ids) in [
        ("original_initial_prefill", plan.prefix_index, prefix),
        ("original_full_recomputation", plan.full_index, full),
    ] {
        client.context = json!({"diagnostic":"authoritative_boundary_capture",
            "role":role,"schedule_index":index,"input_tokens":ids.len(),
            "input_token_ids_sha256":token_ids_sha256(&ids)?,"plan":plan_path,
            "ordinary_cache_gate_qualified":false});
        client.json(
            "POST",
            "/v1/internal/reference_test",
            Some(&json!({
                "input_token_ids":ids,"max_tokens":1,"top_logprobs":10
            })),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    #[test]
    fn boundary_admission_rejects_substituted_tokens_and_invalid_sources() {
        let prefix = vec![json!(1), json!(2)];
        let full = vec![json!(1), json!(2), json!(3)];
        let mut plan = BoundaryCapturePlan {
            source: "/tmp/authoritative".into(),
            prefix_index: 0,
            full_index: 6,
            expected_prefix_tokens: 2,
            expected_full_tokens: 3,
            prefix_token_ids_sha256: token_ids_sha256(&prefix).unwrap(),
            full_token_ids_sha256: token_ids_sha256(&full).unwrap(),
        };
        assert!(admit_boundaries(&plan, &prefix, &full).is_ok());
        assert!(admit_boundaries(&plan, &[json!(1), json!(4)], &full).is_err());
        assert!(admit_boundaries(&plan, &prefix, &[json!(1), json!(2), json!(4)]).is_err());
        assert!(token_ids_sha256(&[json!(4294967296u64)]).is_err());
        plan.source = "relative".into();
        assert!(admit_boundaries(&plan, &prefix, &full).is_err());
    }
}
// END authoritative boundary capture

// Isolate the retained A3/A4 prompt-height change without replaying the
// completed cache diagnostic or changing any qualification result.
pub fn capture_matched_weights(client: &mut Client, source: &Path) -> Result<(), String> {
    for (label, index) in [("A3", 4), ("A4", 6)] {
        let ids = input_ids(&source_json(source, index, "response")?)?;
        client.context = json!({"diagnostic":"matched_weight_binary_capture","variant":label,"input_tokens":ids.len()});
        client.json(
            "POST",
            "/v1/internal/reference_test",
            Some(&json!({
                "input_token_ids":ids,"max_tokens":1,"top_logprobs":10
            })),
        )?;
    }
    Ok(())
}

pub fn trace_sequence(client: &mut Client, source: &Path) -> Result<(), String> {
    let base = input_ids(&source_json(source, 4, "response")?)?;
    let longer = input_ids(&source_json(source, 6, "response")?)?;
    trace_pair(client, base.clone(), longer.clone())?;
    // Capture the seven actual decode operations which produce A3's reusable
    // boundary, together with the corresponding rows of A4's full prefill.
    // The offline audit must admit their token paths before comparing stages.
    for (label, ids, rows, max_tokens) in [
        ("A3-boundary", &base, vec![base.len() - 1], 8),
        (
            "A4-boundary-rows",
            &longer,
            (base.len() - 1..base.len() + 7).collect(),
            1,
        ),
    ] {
        client.context = json!({"diagnostic":"sequence_boundary_operation_trace","variant":label});
        let response = client.json(
            "POST",
            "/v1/internal/reference_test",
            Some(&json!({
                "input_token_ids":ids,"max_tokens":max_tokens,"top_logprobs":10,
                "debug_reference_trace":true,"debug_prefill_device_trace":true,
                "debug_prefill_device_trace_all_layers":true,
                "debug_prefill_device_trace_rows":rows,
                "debug_decode_early_trace":max_tokens > 1,
                "debug_decode_early_trace_max_steps":7
            })),
        )?;
        if response.get("debug_prefill_device_trace").is_none()
            || response.get("debug_reference_trace").is_none()
            || (max_tokens > 1 && response.get("debug_decode_state_trace").is_none())
        {
            return Err("missing requested boundary trace".into());
        }
        save(
            &client.root.join(format!("{label}-trace.json")),
            &serde_json::to_vec_pretty(&response).map_err(|e| e.to_string())?,
        )?;
    }
    Ok(())
}

fn trace_pair(client: &mut Client, base: Vec<Value>, longer: Vec<Value>) -> Result<(), String> {
    let shared = base.iter().zip(&longer).take_while(|(a, b)| a == b).count();
    if shared < 2 {
        return Err("no useful trace prefix".into());
    }
    for (label, ids) in [("base", &base), ("longer", &longer), ("base-repeat", &base)] {
        client.context = json!({"diagnostic":"shared_causal_row_trace","variant":label});
        let response = client.json(
            "POST",
            "/v1/internal/reference_test",
            Some(&json!({
            "input_token_ids":ids,"max_tokens":1,"top_logprobs":10,
            "debug_reference_trace":true,"debug_prefill_device_trace":true,
            "debug_prefill_device_trace_all_layers":true,
            "debug_prefill_device_trace_rows":[0,shared/2,shared-1]})),
        )?;
        if response.get("debug_prefill_device_trace").is_none()
            || response.get("debug_reference_trace").is_none()
        {
            return Err("missing requested layer trace".into());
        }
        save(
            &client.root.join(format!("{label}-trace.json")),
            &serde_json::to_vec_pretty(&response).map_err(|e| e.to_string())?,
        )?;
    }
    Ok(())
}
