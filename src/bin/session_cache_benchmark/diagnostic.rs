use super::evidence::{save, Client};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

// Replay frozen wire requests, changing only the cache switch and explicit
// diagnostic fields. This never invokes or scores productive task agents.
fn source_request(root: &Path, schedule_index: u64) -> Result<Value, String> {
    source_json(root, schedule_index, "request")
}

pub(super) fn source_json(root: &Path, schedule_index: u64, part: &str) -> Result<Value, String> {
    let mut found = Vec::new();
    for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if !path.to_string_lossy().ends_with(".context.json") {
            continue;
        }
        let meta: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if meta["path"] != "/v1/chat/completions"
            || meta["context"]["mode"] != "cache_off_before"
            || meta["context"]["schedule_index"].as_u64() != Some(schedule_index)
        {
            continue;
        }
        let stem = path
            .file_name()
            .ok_or("missing filename")?
            .to_string_lossy()
            .replace(".context.json", "");
        let raw = fs::read(root.join(format!("{stem}.{part}.http"))).map_err(|e| e.to_string())?;
        let transport: Value = serde_json::from_slice(
            &fs::read(root.join(format!("{stem}.transport.json"))).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        use sha2::{Digest, Sha256};
        if transport[format!("{part}_sha256")] != format!("{:x}", Sha256::digest(&raw)) {
            return Err(format!("source request hash mismatch at {schedule_index}"));
        }
        let boundary = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or("source request missing HTTP boundary")?;
        found.push(serde_json::from_slice(&raw[boundary + 4..]).map_err(|e| e.to_string())?);
    }
    if found.len() != 1 {
        return Err(format!(
            "expected one frozen off-before request at {schedule_index}, got {}",
            found.len()
        ));
    }
    Ok(found.remove(0))
}

fn request(client: &mut Client, source: &Value, label: &str, cache: bool) -> Result<Value, String> {
    let mut payload = source.clone();
    payload["prefix_cache"] = json!(cache);
    payload["debug_first_token_boundary"] = json!(true);
    payload["logprobs"] = json!(true);
    payload["top_logprobs"] = json!(10);
    client.context = json!({"diagnostic": label, "source": "retained_cache_off_before_request", "prefix_cache": cache});
    client.json("GET", "/v1/session-cache/stats", None)?;
    let response = client.json("POST", "/v1/chat/completions", Some(&payload))?;
    client.json("GET", "/v1/session-cache/stats", None)?;
    for (ids_key, count_key) in [
        ("input_token_ids", "prompt_tokens"),
        ("completion_token_ids", "completion_tokens"),
    ] {
        let ids = response
            .pointer(&format!("/krasis_debug/{ids_key}"))
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing authoritative {ids_key}"))?;
        let count = response
            .pointer(&format!("/usage/{count_key}"))
            .and_then(Value::as_u64)
            .ok_or("missing authoritative usage count")?;
        if ids.len() as u64 != count || ids.iter().any(|id| id.as_u64().is_none()) {
            return Err(format!("invalid {ids_key}/usage count"));
        }
    }
    Ok(response)
}

fn compare(left: &Value, right: &Value) -> Result<Value, String> {
    let ids = |v: &Value| -> Result<Vec<Value>, String> {
        v.pointer("/krasis_debug/completion_token_ids")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| "missing IDs".into())
    };
    let a = ids(left)?;
    let b = ids(right)?;
    let common = a.iter().zip(&b).take_while(|(a, b)| a == b).count();
    let divergent = (common < a.len().max(b.len())).then_some(common);
    let distribution = |response: &Value| -> Value {
        match divergent {
            Some(0) => response
                .pointer("/krasis_debug/first_token_logits")
                .cloned()
                .unwrap_or(Value::Null),
            Some(step) => response
                .pointer(&format!("/krasis_debug/completion_decode_trace/{step}"))
                .cloned()
                .unwrap_or(Value::Null),
            None => Value::Null,
        }
    };
    Ok(json!({
        "input_ids_identical": left.pointer("/krasis_debug/input_token_ids") == right.pointer("/krasis_debug/input_token_ids"),
        "completion_ids_identical": a==b, "content_exact": left.pointer("/choices/0/message/content") == right.pointer("/choices/0/message/content"),
        "left_ids": a, "right_ids": b, "common_prefix_tokens": common, "first_divergent_decision": divergent,
        "left_distribution": distribution(left), "right_distribution": distribution(right),
        "first_token_distribution_exact": left.pointer("/krasis_debug/first_token_logits") == right.pointer("/krasis_debug/first_token_logits"),
        "decode_distributions_exact": left.pointer("/krasis_debug/completion_decode_trace") == right.pointer("/krasis_debug/completion_decode_trace"),
    }))
}

// Replay the retained turn-four failure without reseeding its snapshot between
// active and RAM comparisons. Original candidate verdicts are never altered.
pub fn run_sequence(client: &mut Client, root: &Path) -> Result<(), String> {
    let sequence = (0..=6)
        .map(|i| source_request(root, i))
        .collect::<Result<Vec<_>, _>>()?;
    save(
        &client.root.join("diagnostic-inputs.json"),
        &serde_json::to_vec_pretty(&json!({"source":root,"sequence":sequence}))
            .map_err(|e| e.to_string())?,
    )?;
    let mut controls = Vec::new();
    for sample in 0..8 {
        controls.push(request(
            client,
            &sequence[6],
            &format!("sequence_full_A4_{sample}"),
            false,
        )?);
    }
    let mut seeds = Vec::new();
    for (i, payload) in sequence.iter().take(5).enumerate() {
        seeds.push(request(
            client,
            payload,
            &format!("sequence_seed_{i}"),
            true,
        )?);
    }
    let active = request(client, &sequence[6], "sequence_active_A4", true)?;
    request(client, &sequence[5], "sequence_displace_B3", true)?;
    let restored = request(client, &sequence[6], "sequence_RAM_A4", true)?;
    let control_pairs = controls
        .iter()
        .enumerate()
        .flat_map(|(i, a)| controls.iter().skip(i + 1).map(move |b| (a, b)))
        .map(|(a, b)| compare(a, b))
        .collect::<Result<Vec<_>, _>>()?;
    let seed_pairs = seeds
        .iter()
        .enumerate()
        .map(|(i, seed)| {
            Ok(json!({"schedule_index":i,
            "original_vs_replay":compare(&source_json(root,i as u64,"response")?,seed)?}))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let result = json!({"diagnostic_only":true,"clears_original_failed_lifecycle":false,
        "source_evidence":root,"seed_comparisons":seed_pairs,
        "uncached_pairs":control_pairs,"active_vs_RAM":compare(&active,&restored)?,
        "original_full_vs_replay":compare(&source_json(root,6,"response")?,&controls[0])?,
        "full_vs_active":controls.iter().map(|v|compare(v,&active)).collect::<Result<Vec<_>,_>>()?,
        "full_vs_RAM":controls.iter().map(|v|compare(v,&restored)).collect::<Result<Vec<_>,_>>()?,
        "geometry_admission":"independently verify identical snapshot ID, 10551 boundary and 13 suffix in runtime log and cache stats"});
    save(
        &client.root.join("diagnostic-comparisons.json"),
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

pub fn run(client: &mut Client, root: &Path) -> Result<(), String> {
    let base = source_request(root, 0)?;
    let displaced = source_request(root, 1)?;
    let continuation = source_request(root, 2)?;
    save(
        &client.root.join("diagnostic-inputs.json"),
        &serde_json::to_vec_pretty(
            &json!({"source": root, "A1": base, "B1": displaced, "A2": continuation}),
        )
        .map_err(|e| e.to_string())?,
    )?;
    let mut bases = Vec::new();
    let mut controls = Vec::new();
    // Match the existing live gate's eight independent full-prefill samples.
    for sample in 0..8 {
        bases.push(request(
            client,
            &base,
            &format!("base_cache_disabled_{sample}"),
            false,
        )?);
        controls.push(request(
            client,
            &continuation,
            &format!("continuation_cache_disabled_{sample}"),
            false,
        )?);
    }
    let seed = request(client, &base, "seed_A1", true)?;
    let active = request(client, &continuation, "active_A2", true)?;
    // Retain the very same A1 snapshot for both paths. Reseeding A1 here
    // would create a new boundary and weaken the restoration discriminator.
    request(client, &displaced, "displace_B1", true)?;
    let restored = request(client, &continuation, "RAM_A2", true)?;
    let control_pairs = controls
        .iter()
        .enumerate()
        .flat_map(|(i, a)| controls.iter().skip(i + 1).map(move |b| (a, b)))
        .map(|(a, b)| compare(a, b))
        .collect::<Result<Vec<_>, _>>()?;
    let base_pairs = bases
        .iter()
        .enumerate()
        .flat_map(|(i, a)| bases.iter().skip(i + 1).map(move |b| (a, b)))
        .map(|(a, b)| compare(a, b))
        .collect::<Result<Vec<_>, _>>()?;
    let result = json!({
        "diagnostic_only": true, "clears_original_failed_lifecycle": false,
        "source_evidence": root, "base_cache_disabled_pairs": base_pairs, "continuation_cache_disabled_pairs": control_pairs,
        "seed_vs_control": compare(&bases[0], &seed)?, "active_vs_RAM": compare(&active, &restored)?,
        "full_prefill_vs_active": controls.iter().map(|v|compare(v,&active)).collect::<Result<Vec<_>,_>>()?,
        "full_prefill_vs_RAM": controls.iter().map(|v|compare(v,&restored)).collect::<Result<Vec<_>,_>>()?,
        "geometry_admission": "must independently verify active/RAM start boundaries and suffix heights from runtime evidence; no tolerance replaces exact campaign oracle",
    });
    save(
        &client.root.join("diagnostic-comparisons.json"),
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_rejects_a_modified_frozen_wire_request() {
        use sha2::{Digest, Sha256};
        let root = std::env::temp_dir().join(format!(
            "cache-source-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"temperature\":0}";
        fs::write(root.join("0000.context.json"), serde_json::to_vec(&json!({"path":"/v1/chat/completions","context":{"mode":"cache_off_before","schedule_index":0}})).unwrap()).unwrap();
        fs::write(root.join("0000.request.http"), raw).unwrap();
        fs::write(
            root.join("0000.transport.json"),
            serde_json::to_vec(&json!({"request_sha256":format!("{:x}",Sha256::digest(raw))}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(source_request(&root, 0).unwrap()["temperature"], 0);
        fs::write(root.join("0000.request.http"), b"changed").unwrap();
        assert!(source_request(&root, 0)
            .unwrap_err()
            .contains("hash mismatch"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn comparison_reports_only_the_first_divergent_decision() {
        let response = |ids: Value| {
            json!({
                "choices": [{"message": {"content": "same visible text"}}],
                "krasis_debug": {
                    "input_token_ids": [10,20], "completion_token_ids": ids,
                    "first_token_logits": {"marker": "prefill"},
                    "completion_decode_trace": [{"marker": "zero"},{"marker": "first-divergence"},{"marker": "different-history"}],
                }
            })
        };
        let pair = compare(&response(json!([1, 2, 3])), &response(json!([1, 4, 5]))).unwrap();
        assert_eq!(pair["first_divergent_decision"], 1);
        assert_eq!(pair["left_distribution"]["marker"], "first-divergence");
        assert_eq!(pair["content_exact"], true);
        assert_eq!(pair["completion_ids_identical"], false);
        let first = compare(&response(json!([1])), &response(json!([2]))).unwrap();
        assert_eq!(first["left_distribution"]["marker"], "prefill");
    }
}

fn state_request(client: &mut Client, source: &Value, label: &str) -> Result<Value, String> {
    let response = request(client, source, label, true)?;
    let prompt = response
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or("missing prompt usage")?;
    let completion = response
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .ok_or("missing completion usage")?;
    let logical = prompt
        .checked_add(completion)
        .and_then(|v| v.checked_sub(1))
        .ok_or("invalid consumed position")?;
    let inventory = client.json(
        "POST",
        "/v1/internal/sequence_state_inventory",
        Some(&json!({"logical_tokens":logical})),
    )?;
    let devices = inventory["devices"]
        .as_array()
        .ok_or("missing state inventory devices")?;
    if devices.is_empty()
        || devices.iter().any(|d| {
            d.get("error").is_some()
                || d["kv_absolute_position"].as_u64() != Some(logical)
                || !d["graph_path"].is_object()
        })
    {
        return Err(format!(
            "ordinary state inventory admission failed for {label}"
        ));
    }
    save(
        &client.root.join(format!("{label}-inventory.json")),
        &serde_json::to_vec_pretty(&inventory).map_err(|e| e.to_string())?,
    )?;
    Ok(response)
}

// Capture only the retained seed chain and ordinary active/RAM continuation.
// The completed full-recomputation control campaign is not repeated.
pub fn run_state(client: &mut Client, root: &Path) -> Result<(), String> {
    let sequence = (0..=6)
        .map(|i| source_request(root, i))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seeds = Vec::new();
    for (i, payload) in sequence.iter().take(5).enumerate() {
        let seed = state_request(client, payload, &format!("state_seed_{i}"))?;
        let original = source_json(root, i as u64, "response")?;
        let comparison = compare(&original, &seed)?;
        if comparison["input_ids_identical"] != true
            || comparison["completion_ids_identical"] != true
        {
            return Err(format!(
                "ordinary state seed {i} changed the retained token path"
            ));
        }
        seeds.push(comparison);
    }
    let active = state_request(client, &sequence[6], "state_active_A4")?;
    state_request(client, &sequence[5], "state_displace_B3")?;
    let restored = state_request(client, &sequence[6], "state_RAM_A4")?;
    let report = json!({"diagnostic_only":true,"collection_does_not_qualify_candidate":true,"source":root,
        "seed_comparisons":seeds,"active_vs_RAM":compare(&active,&restored)?,
        "geometry_admission":"requires same retained snapshot, 10551-token boundary and 13-token GPU prefill suffix; binary state and graph counters are authoritative"});
    save(
        &client.root.join("state-comparisons.json"),
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    let original = source_json(root, 6, "response")?;
    let ids = original
        .pointer("/krasis_debug/input_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing retained KDA source input IDs")?;
    client.context = json!({"diagnostic":"matched_kda_binary_capture","input_tokens":ids.len()});
    client.json(
        "POST",
        "/v1/internal/reference_test",
        Some(&json!({"input_token_ids":ids,"max_tokens":1,"top_logprobs":10})),
    )?;
    Ok(())
}

// A bounded ordinary full-vs-cache replay, not another restoration campaign.
pub fn run_trajectory(client: &mut Client, root: &Path) -> Result<(), String> {
    let sequence = (0..=6)
        .map(|i| source_request(root, i))
        .collect::<Result<Vec<_>, _>>()?;
    let original = source_json(root, 6, "response")?;
    let original_ids = original
        .pointer("/krasis_debug/input_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing original prompt IDs")?;
    let completion = original
        .pointer("/krasis_debug/completion_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing original completion IDs")?;
    if original_ids.len() != 10564 || completion.len() != 8 {
        return Err("saved failure geometry changed".into());
    }
    let full = request(client, &sequence[6], "trajectory_full_A4", false)?;
    let full_comparison = compare(&original, &full)?;
    if full_comparison["input_ids_identical"] != true
        || full_comparison["completion_ids_identical"] != true
    {
        return Err("full trajectory did not reproduce authoritative failure".into());
    }
    let mut seeds = Vec::new();
    for (i, payload) in sequence.iter().take(5).enumerate() {
        let seed = state_request(client, payload, &format!("trajectory_seed_{i}"))?;
        let comparison = compare(&source_json(root, i as u64, "response")?, &seed)?;
        if comparison["input_ids_identical"] != true
            || comparison["completion_ids_identical"] != true
        {
            return Err(format!("trajectory seed {i} changed authoritative IDs"));
        }
        seeds.push(comparison);
    }
    let cached = state_request(client, &sequence[6], "trajectory_cached_A4")?;
    let comparison = compare(&full, &cached)?;
    save(&client.root.join("trajectory-comparison.json"), &serde_json::to_vec_pretty(&json!({
        "diagnostic_only":true,"collection_does_not_qualify_candidate":true,
        "source":root,"original_vs_full":full_comparison,"seed_admission":seeds,
        "full_vs_ordinary_cache":comparison,
        "common_five_input_ids":completion[..5],
        "forcing":"ordinary greedy replay: both paths naturally feed these same five authoritative IDs; fail if history differs",
        "boundary_admission":"independently verify reused=10551 suffix=13 and ordinary graphs in server evidence",
    })).map_err(|e|e.to_string())?)?;
    if comparison["input_ids_identical"] != true || comparison["common_prefix_tokens"] != 5 {
        return Err(
            "trajectory did not reproduce token-six reversal with identical first five inputs"
                .into(),
        );
    }
    // Prefix-only recomputation isolates the original prefill/decode boundary.
    // It does not claim to expose a 10551 state inside a full-height 10564 run.
    client.context = json!({"diagnostic":"trajectory_full_prefix10551","authoritative_source":root,"prefix_tokens":10551});
    client.json(
        "POST",
        "/v1/internal/reference_test",
        Some(&json!({
            "input_token_ids":original_ids[..10551], "max_tokens":1, "top_logprobs":10,
        })),
    )?;
    Ok(())
}

// This client never scores ADQ. Server admission also enforces completed controls.
pub(super) fn run_replay(client: &mut Client, plan_path: &Path) -> Result<(), String> {
    let plan: Value = serde_json::from_slice(&fs::read(plan_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let input_path = plan_path
        .parent()
        .ok_or("missing plan directory")?
        .join("SAVED_GAP.json");
    let input: Value = serde_json::from_slice(&fs::read(input_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let forced = plan["forced_inputs"]
        .as_array()
        .ok_or("missing forced IDs")?;
    let trials = plan["trials"].as_array().ok_or("missing replay trials")?;
    for trial in trials {
        let label = trial["label"].as_str().ok_or("missing trial label")?;
        client.context = json!({"diagnostic":"causal_state_replay","label":label,"plan":plan_path,"diagnostic_only":true,"not_adq_scoring":true});
        let response = client.json(
            "POST",
            "/v1/internal/reference_test",
            Some(&json!({
                "input_token_ids":input["authoritative_prompt_ids"],"max_tokens":forced.len()+1,
                "top_logprobs":10,"debug_matched_replay":label,
            })),
        )?;
        let ids = response["token_ids"]
            .as_array()
            .ok_or("missing replay token IDs")?;
        if ids.len() != forced.len() + 1 || ids[..forced.len()] != forced[..] {
            return Err(format!(
                "replay {label} did not feed every authoritative common input"
            ));
        }
        let directory = Path::new(
            plan["directory"]
                .as_str()
                .ok_or("missing replay output directory")?,
        )
        .join(label);
        for step in 0..forced.len() {
            let proof: Value = serde_json::from_slice(
                &fs::read(directory.join(format!("step-{step}.json")))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if proof["input_token"] != forced[step]
                || (!trial["expected_logits_sha256"].is_null() && proof["control_matches"] != true)
            {
                return Err(format!(
                    "replay {label} control/input failure at step {step}"
                ));
            }
        }
        eprintln!(
            "Replay {label} captured with all authoritative inputs; whole-state-control={}",
            !trial["expected_logits_sha256"].is_null()
        );
    }
    Ok(())
}

// BEGIN bounded KDA model treatment trajectory
// Separate diagnostic intervention: the historical control above remains strict.
// Collection never changes the failed gate, scores ADQ, or qualifies a runtime.
fn save_kda_treatment_stage(root: &Path, stage: &str, report: &Value) -> Result<(), String> {
    if ![
        "full", "seed-0", "seed-1", "seed-2", "seed-3", "seed-4", "cached",
    ]
    .contains(&stage)
    {
        return Err("invalid KDA treatment evidence stage".into());
    }
    let mut snapshot = report.clone();
    snapshot["capture_stage"] = json!(stage);
    save(
        &root.join(format!("kda-treatment-{stage}-comparison.json")),
        &serde_json::to_vec_pretty(&snapshot).map_err(|e| e.to_string())?,
    )
}

// BEGIN explicit raw KDA treatment identity
fn raw_kda_treatment_identity(
    raw: Option<&str>,
    legacy: Option<&str>,
    serial: Option<&str>,
    module: Option<&str>,
) -> Result<Value, String> {
    if raw != Some("1") || legacy.is_some() || serial.is_some() || module.is_some() {
        return Err("raw-prefill treatment requires explicit KRASIS_GLM53_KDA_RECURRENT_ORACLE=1 with legacy, serial-convolution and intra-module variables absent".into());
    }
    Ok(
        json!({"kind":"raw_prefill_only", "prefill_operator":"kimi_delta_recurrent_scan_tiled_bf16", "decode_operator":"kimi_delta_recurrent_scan_bf16", "production_adoption":false}),
    )
}

pub fn run_raw_kda_treatment(client: &mut Client, root: &Path) -> Result<(), String> {
    let raw = std::env::var("KRASIS_GLM53_KDA_RECURRENT_ORACLE").ok();
    let legacy = std::env::var_os("KRASIS_GLM53_KDA_LEGACY_RECURRENT_ORACLE").map(|_| "present");
    let serial = std::env::var_os("KRASIS_GLM53_KDA_SERIAL_CONV_ORACLE").map(|_| "present");
    let module = std::env::var_os("KRASIS_MATCHED_KDA_INTRA_CONFIG").map(|_| "present");
    let identity = raw_kda_treatment_identity(raw.as_deref(), legacy, serial, module)?;
    run_kda_treatment_identified(client, root, Value::Null, identity)
}
// END explicit raw KDA treatment identity

pub fn run_kda_treatment(client: &mut Client, root: &Path) -> Result<(), String> {
    let module_plan = std::env::var("KRASIS_MATCHED_KDA_INTRA_CONFIG")
        .map_err(|_| "explicit diagnostic KDA intervention required")?;
    run_kda_treatment_identified(
        client,
        root,
        json!(module_plan),
        json!({"kind":"intra_module", "production_adoption":false}),
    )
}

fn run_kda_treatment_identified(
    client: &mut Client,
    root: &Path,
    module_plan: Value,
    intervention: Value,
) -> Result<(), String> {
    let sequence = (0..=6)
        .map(|i| source_request(root, i))
        .collect::<Result<Vec<_>, _>>()?;
    let original = source_json(root, 6, "response")?;
    let inputs = original
        .pointer("/krasis_debug/input_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing authoritative inputs")?;
    let generated = original
        .pointer("/krasis_debug/completion_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing authoritative generated IDs")?;
    if inputs.len() != 10564 || generated.len() != 8 {
        return Err("retained failure geometry changed".into());
    }
    let full = request(client, &sequence[6], "treatment_full_A4", false)?;
    let full_comparison = compare(&original, &full)?;
    let mut report = json!({"diagnostic_only":true,"qualifies_candidate":false,"mandatory_consistency_gate_remains_failed":true,
        "original_control_is_not_waived":true,"module_plan":module_plan,"intervention":intervention,"source":root,"original_vs_treatment_full":full_comparison,"seed_admission":[],
        "required_common_five":generated[..5],"ordinary_greedy_responses_retained_separately":true});
    let report_root = client.root.clone();
    save_kda_treatment_stage(&report_root, "full", &report)?;
    if full_comparison["input_ids_identical"] != true
        || full
            .pointer("/krasis_debug/completion_token_ids")
            .and_then(Value::as_array)
            .is_none_or(|ids| ids.len() != 8 || ids[..5] != generated[..5])
    {
        return Err("KDA treatment full path changed retained prompt or common-five inputs; bounded replay stopped".into());
    }
    for (i, payload) in sequence.iter().take(5).enumerate() {
        let seed = state_request(client, payload, &format!("treatment_seed_{i}"))?;
        let comparison = compare(&source_json(root, i as u64, "response")?, &seed)?;
        report["seed_admission"]
            .as_array_mut()
            .unwrap()
            .push(json!({"index":i,"comparison":comparison}));
        save_kda_treatment_stage(&report_root, &format!("seed-{i}"), &report)?;
        if comparison["input_ids_identical"] != true
            || comparison["completion_ids_identical"] != true
        {
            return Err(format!(
                "KDA treatment seed {i} changed authoritative history; cache path not manufactured"
            ));
        }
    }
    let cached = state_request(client, &sequence[6], "treatment_cached_A4")?;
    report["treatment_full_vs_ordinary_cache"] = compare(&full, &cached)?;
    save_kda_treatment_stage(&report_root, "cached", &report)?;
    if report["treatment_full_vs_ordinary_cache"]["input_ids_identical"] != true
        || cached
            .pointer("/krasis_debug/completion_token_ids")
            .and_then(Value::as_array)
            .is_none_or(|ids| ids.len() != 8 || ids[..5] != generated[..5])
    {
        return Err("KDA treatment cached path changed prompt or common-five inputs; no same-input endpoint attribution".into());
    }
    Ok(())
}

#[cfg(test)]
mod kda_treatment_evidence_tests {
    use super::*;
    // BEGIN explicit raw treatment admission tests
    #[test]
    fn raw_treatment_rejects_missing_or_combined_interventions() {
        for raw in [None, Some("0"), Some("true"), Some("")] {
            assert!(raw_kda_treatment_identity(raw, None, None, None).is_err());
        }
        for combined in [
            (Some("0"), None, None),
            (None, Some("0"), None),
            (None, None, Some("unused")),
        ] {
            assert!(
                raw_kda_treatment_identity(Some("1"), combined.0, combined.1, combined.2).is_err()
            );
        }
        let id = raw_kda_treatment_identity(Some("1"), None, None, None).unwrap();
        assert_eq!(id["kind"], "raw_prefill_only");
        assert_eq!(id["decode_operator"], "kimi_delta_recurrent_scan_bf16");
        assert_eq!(id["production_adoption"], false);
    }
    // END explicit raw treatment admission tests

    #[test]
    fn incremental_treatment_evidence_preserves_every_completed_stage() {
        let root = std::env::temp_dir().join(format!(
            "kda-stages-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let mut report =
            json!({"diagnostic_only":true,"qualifies_candidate":false,"seed_admission":[]});
        save_kda_treatment_stage(&root, "full", &report).unwrap();
        let original = fs::read(root.join("kda-treatment-full-comparison.json")).unwrap();
        for i in 0..5 {
            report["seed_admission"]
                .as_array_mut()
                .unwrap()
                .push(json!({"index":i}));
            save_kda_treatment_stage(&root, &format!("seed-{i}"), &report).unwrap();
        }
        report["treatment_full_vs_ordinary_cache"] = json!({"completion_ids_identical":false});
        save_kda_treatment_stage(&root, "cached", &report).unwrap();
        assert_eq!(fs::read_dir(&root).unwrap().count(), 7);
        assert_eq!(
            fs::read(root.join("kda-treatment-full-comparison.json")).unwrap(),
            original
        );
        let final_report: Value = serde_json::from_slice(
            &fs::read(root.join("kda-treatment-cached-comparison.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(final_report["capture_stage"], "cached");
        assert_eq!(final_report["seed_admission"].as_array().unwrap().len(), 5);
        assert!(save_kda_treatment_stage(&root, "full", &report)
            .unwrap_err()
            .contains("create "));
        assert!(save_kda_treatment_stage(&root, "../escape", &report)
            .unwrap_err()
            .contains("invalid"));
        fs::remove_dir_all(root).unwrap();
    }
}
// END bounded KDA model treatment trajectory

// BEGIN explicit teacher-forced saved sequence client
pub fn run_teacher_sequence(client: &mut Client, root: &Path) -> Result<(), String> {
    run_teacher_sequence_with_seeds(client, root, 5)
}
pub fn run_teacher_original_sequence(client: &mut Client, root: &Path) -> Result<(), String> {
    run_teacher_sequence_with_seeds(client, root, 6)
}
fn admitted_seed_history(report: &Value, required: usize) -> bool {
    report.as_array().is_some_and(|rows| {
        rows.len() == required
            && rows.iter().enumerate().all(|(i, r)| {
                r["index"].as_u64() == Some(i as u64)
                    && r["forced_ids_match"] == true
                    && r["prompt_ids_match"] == true
            })
    })
}
fn run_teacher_sequence_with_seeds(
    client: &mut Client,
    root: &Path,
    seed_count: usize,
) -> Result<(), String> {
    if ![5, 6].contains(&seed_count) {
        return Err("unadmitted saved seed schedule".into());
    }
    if std::env::var_os("KRASIS_MATCHED_TEACHER_CONFIG").is_none() {
        return Err("explicit teacher-forcing plan required".into());
    }
    let sequence = (0..=6)
        .map(|i| source_request(root, i))
        .collect::<Result<Vec<_>, _>>()?;
    let original = source_json(root, 6, "response")?;
    let original_ids = original
        .pointer("/krasis_debug/input_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing authoritative prompt IDs")?;
    let common = original
        .pointer("/krasis_debug/completion_token_ids")
        .and_then(Value::as_array)
        .ok_or("missing authoritative generation IDs")?;
    if original_ids.len() != 10564 || common.len() != 8 {
        return Err("retained failing conversation geometry changed".into());
    }
    let ordinary = request(client, &sequence[6], "teacher_ordinary_full_A4", false)?;
    if ordinary.pointer("/krasis_debug/input_token_ids")
        != original.pointer("/krasis_debug/input_token_ids")
    {
        return Err("ordinary full prompt changed".into());
    }
    let force = |client: &mut Client,
                 payload: &Value,
                 label: &str,
                 cache: bool|
     -> Result<Value, String> {
        let mut payload = payload.clone();
        payload["debug_matched_teacher_forcing"] = json!(label);
        let response = request(client, &payload, label, cache)?;
        if response.pointer("/krasis_debug/teacher_forcing/diagnostic_only") != Some(&json!(true))
            || response.pointer("/krasis_debug/teacher_forcing/qualifies_candidate")
                != Some(&json!(false))
        {
            return Err("teacher-forced response lacks explicit diagnostic-only provenance".into());
        }
        Ok(response)
    };
    let full = force(client, &sequence[6], "teacher_full_A4", false)?;
    let mut report = json!({"diagnostic_only":true,"qualifies_candidate":false,"mandatory_equality":"FAILED","ordinary_cache_equality_evidence":false,
        "ordinary_full_vs_teacher_full":compare(&ordinary,&full)?,"seed_admission":[],"known_common_five":common[..5]});
    save(
        &client.root.join("teacher-sequence-full.json"),
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    for (i, payload) in sequence.iter().take(seed_count).enumerate() {
        let label = format!("teacher_seed_{i}");
        let response = force(client, payload, &label, true)?;
        let control = source_json(root, i as u64, "response")?;
        let comparison = compare(&control, &response)?;
        report["seed_admission"].as_array_mut().unwrap().push(json!({"index":i,"forced_ids_match":comparison["completion_ids_identical"],"prompt_ids_match":comparison["input_ids_identical"]}));
        save(
            &client.root.join(format!("teacher-sequence-seed-{i}.json")),
            &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )?;
        if comparison["input_ids_identical"] != true
            || comparison["completion_ids_identical"] != true
        {
            return Err(
                "teacher sequence did not execute its checksum-bound saved seed inputs".into(),
            );
        }
    }
    if !admitted_seed_history(&report["seed_admission"], seed_count) {
        return Err("incomplete or invalid admitted seed history".into());
    }
    report["source_seed_count"] = json!(seed_count);
    report["complete_original_seed_sequence"] = json!(seed_count == 6);
    let cached = force(client, &sequence[6], "teacher_cached_A4", true)?;
    if full.pointer("/krasis_debug/input_token_ids")
        != original.pointer("/krasis_debug/input_token_ids")
        || cached.pointer("/krasis_debug/input_token_ids")
            != original.pointer("/krasis_debug/input_token_ids")
    {
        return Err("teacher paths changed authoritative final prompt".into());
    }
    for response in [&full, &cached] {
        let ids = response
            .pointer("/krasis_debug/completion_token_ids")
            .and_then(Value::as_array)
            .ok_or("missing teacher IDs")?;
        if ids.len() != 8 || ids[..5] != common[..5] {
            return Err("teacher path did not feed the exact common five inputs".into());
        }
    }
    report["teacher_full_vs_teacher_cache"] = compare(&full, &cached)?;
    save(
        &client.root.join("teacher-sequence-final.json"),
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    Ok(())
}
// END explicit teacher-forced saved sequence client

// Bounded saved-conversation test, with no state capture or token forcing.
pub fn run_unforced_saved_sequence(client: &mut Client, root: &Path) -> Result<(), String> {
    run_saved_sequence(client, root, false)
}

// Same actual requests, with explicitly admitted read-only observer. Never a final ordinary gate.
pub fn run_observed_saved_sequence(client: &mut Client, root: &Path) -> Result<(), String> {
    run_saved_sequence(client, root, true)
}

fn run_saved_sequence(client: &mut Client, root: &Path, observed: bool) -> Result<(), String> {
    for key in [
        "KRASIS_MATCHED_TEACHER_CONFIG",
        "KRASIS_MATCHED_PROBE_CONFIG",
        "KRASIS_MATCHED_REPLAY_CONFIG",
        "KRASIS_TRACE",
        "KRASIS_CAPTURE_PHASE_CONFIG",
    ] {
        if std::env::var_os(key).is_some() && !(observed && key == "KRASIS_MATCHED_PROBE_CONFIG") {
            return Err(format!("unforced consistency command forbids {key}"));
        }
    }
    if observed && std::env::var_os("KRASIS_MATCHED_PROBE_CONFIG").is_none() {
        return Err("observed sequence requires explicit probe configuration".into());
    }
    let sequence = (0..=6)
        .map(|i| source_request(root, i))
        .collect::<Result<Vec<_>, _>>()?;
    let expected = source_json(root, 6, "response")?;
    let full = request(client, &sequence[6], "unforced_full_A4", false)?;
    let repeated = request(client, &sequence[6], "unforced_full_A4_repeat", false)?;
    let repetition = compare(&full, &repeated)?;
    let mut seeds = Vec::new();
    for (i, payload) in sequence.iter().take(6).enumerate() {
        let response = request(client, payload, &format!("unforced_seed_{i}"), true)?;
        let old = source_json(root, i as u64, "response")?;
        seeds.push(json!({"index":i,"original_history_comparison":compare(&old,&response)?}));
    }
    let cached = request(client, &sequence[6], "unforced_cached_A4", true)?;
    let comparison = compare(&full, &cached)?;
    let prompt_match = full.pointer("/krasis_debug/input_token_ids")
        == expected.pointer("/krasis_debug/input_token_ids")
        && cached.pointer("/krasis_debug/input_token_ids")
            == expected.pointer("/krasis_debug/input_token_ids");
    let equal = prompt_match
        && comparison["completion_ids_identical"] == true
        && comparison["content_exact"] == true
        && repetition["completion_ids_identical"] == true
        && repetition["content_exact"] == true;
    let report = json!({"forced_inputs":false,"matched_capture":observed,"source":root,"source_seed_count":6,"complete_original_seed_sequence":true,"original_prompt_exact":prompt_match,"seed_history":seeds,"repetition":repetition,"full_vs_cached":comparison,"ordinary_consistency":if equal {if observed {"OBSERVED_EXACT_THIS_CASE_ONLY"} else {"PASS_THIS_CASE_ONLY"}} else {"FAILED"},"qualifies_ADQ":false,"actual_cache_reuse_requires_server_log_verification":true});
    save(
        &client.root.join(if observed {
            "observed-saved-sequence.json"
        } else {
            "unforced-saved-sequence.json"
        }),
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    if !equal {
        return Err(
            "unforced saved-conversation consistency failed; complete result retained".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod original_seed_history_tests {
    use super::*;
    #[test]
    fn full_seed_admission_rejects_omission_duplicate_and_divergent_history() {
        let rows = (0..6)
            .map(|i| json!({"index":i,"forced_ids_match":true,"prompt_ids_match":true}))
            .collect::<Vec<_>>();
        assert!(admitted_seed_history(&json!(rows), 6));
        assert!(!admitted_seed_history(&json!(&rows[..5]), 6));
        let mut duplicate = rows.clone();
        duplicate[5]["index"] = json!(4);
        assert!(!admitted_seed_history(&json!(duplicate), 6));
        let mut bad = rows.clone();
        bad[5]["forced_ids_match"] = json!(false);
        assert!(!admitted_seed_history(&json!(bad), 6));
        let mut bad = rows;
        bad[5]["prompt_ids_match"] = json!(false);
        assert!(!admitted_seed_history(&json!(bad), 6));
    }
}
