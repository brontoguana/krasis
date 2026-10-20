//! Reproducible end-to-end benchmark for RAM-backed multi-conversation reuse.
//!
//! Run through `./dev session-cache-benchmark <port>`. The server must be
//! started with session caching enabled. Two canonical
//! Gutenberg conversations are interleaved so every continuation exercises a
//! real pageable-RAM restore rather than the active-GPU fast path. Cache-off
//! runs bracket the cache-on run to expose process-local timing drift.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
#[path = "session_cache_benchmark/diagnostic.rs"]
mod diagnostic;
#[path = "session_cache_benchmark/evidence.rs"]
mod evidence;
#[path = "session_cache_benchmark/height.rs"]
mod height;
use evidence::Client;

const PROMPT_CHARS: usize = 40_000;
const TURNS_PER_CONVERSATION: usize = 4;
const COMPLETION_TOKENS: usize = 8;

const QUESTIONS_A: [&str; TURNS_PER_CONVERSATION] = [
    "Answer briefly: who commands the Pequod?",
    "Answer briefly: what is the ship called?",
    "Answer briefly: what animal is central to the voyage?",
    "Answer briefly: name the first mate.",
];
const QUESTIONS_B: [&str; TURNS_PER_CONVERSATION] = [
    "Answer briefly: who wrote this novel?",
    "Answer briefly: name one central family.",
    "Answer briefly: during which war is much of the story set?",
    "Answer briefly: name one major city in the story.",
];

#[derive(Clone)]
struct TurnResult {
    elapsed_ms: f64,
    input_tokens: usize,
    completion_tokens: usize,
    content: String,
}

struct ConversationRun {
    turns: Vec<TurnResult>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("session-cache benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let port: u16 = args
        .next()
        .ok_or_else(|| "usage: session-cache-benchmark <port>".to_string())?
        .parse()
        .map_err(|error| format!("invalid port: {error}"))?;
    let diagnostic_source = match args.next() {
        None => None,
        Some(flag)
            if [
                "--diagnostic",
                "--diagnostic-sequence",
                "--height-baseline",
                "--height-regression",
                "--height-trace",
                "--sequence-height-trace",
                "--matched-weight-capture",
                "--matched-boundary-capture",
                "--matched-state-capture",
                "--matched-kda-treatment",
                "--matched-raw-kda-treatment",
                "--matched-teacher-sequence",
                "--matched-teacher-original-sequence",
                "--unforced-saved-sequence",
                "--observed-saved-sequence",
                "--consistency-trajectory",
                "--matched-state-replay",
            ]
            .contains(&flag.as_str()) =>
        {
            Some((
                flag,
                PathBuf::from(args.next().ok_or("missing source evidence directory")?),
            ))
        }
        Some(_) => {
            return Err(
                "usage: session-cache-benchmark <port> [--diagnostic <source-evidence>]".into(),
            )
        }
    };
    if args.next().is_some() {
        return Err("unexpected arguments".into());
    }

    let evidence_root = std::env::var_os("KRASIS_CACHE_BENCH_EVIDENCE")
        .ok_or("Run through ./dev session-cache-benchmark to select a fresh evidence directory")?;
    let mut client = Client::new(port, PathBuf::from(evidence_root))?;
    eprintln!("Evidence dir: {}", client.root.display());
    let outcome = (|| -> Result<(), String> {
        if let Some((mode, source)) = diagnostic_source {
            return match mode.as_str() {
                "--diagnostic" => diagnostic::run(&mut client, &source),
                "--diagnostic-sequence" => diagnostic::run_sequence(&mut client, &source),
                "--height-trace" => height::trace(&mut client, &source),
                "--sequence-height-trace" => height::trace_sequence(&mut client, &source),
                "--matched-weight-capture" => height::capture_matched_weights(&mut client, &source),
                "--matched-boundary-capture" => height::capture_boundaries(&mut client, &source),
                "--matched-state-capture" => diagnostic::run_state(&mut client, &source),
                "--matched-kda-treatment" => diagnostic::run_kda_treatment(&mut client, &source),
                "--matched-raw-kda-treatment" => {
                    diagnostic::run_raw_kda_treatment(&mut client, &source)
                }
                "--matched-teacher-sequence" => {
                    diagnostic::run_teacher_sequence(&mut client, &source)
                }
                "--matched-teacher-original-sequence" => {
                    diagnostic::run_teacher_original_sequence(&mut client, &source)
                }
                "--unforced-saved-sequence" => {
                    diagnostic::run_unforced_saved_sequence(&mut client, &source)
                }
                "--observed-saved-sequence" => {
                    diagnostic::run_observed_saved_sequence(&mut client, &source)
                }
                "--consistency-trajectory" => diagnostic::run_trajectory(&mut client, &source),
                "--matched-state-replay" => diagnostic::run_replay(&mut client, &source),
                _ => height::run(&mut client, &source, mode == "--height-regression"),
            };
        }
        let model_response = client.json("GET", "/v1/models", None)?;
        let model = model_response
            .pointer("/data/0/id")
            .and_then(Value::as_str)
            .ok_or_else(|| "server did not expose exactly one model id".to_string())?;
        let initial_stats = client.json("GET", "/v1/session-cache/stats", None)?;
        if !initial_stats
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err("server session cache is disabled".to_string());
        }

        let moby = read_prefix(
            Path::new("benchmarks/prompts/prompt1_moby_dick.txt"),
            PROMPT_CHARS,
        )?;
        let war = read_prefix(
            Path::new("benchmarks/prompts/prompt2_war_and_peace.txt"),
            PROMPT_CHARS,
        )?;
        let prompts = [moby, war];
        let questions = [&QUESTIONS_A, &QUESTIONS_B];

        // Warm the local HTTP/model path without creating cache state.
        let warmup = json!({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word: ready."}],
            "temperature": 0,
            "top_k": 1,
            "max_tokens": 2,
            "min_new_tokens": 2,
            "stream": false,
            "prefix_cache": false,
        });
        client.json("POST", "/v1/chat/completions", Some(&warmup))?;

        let stats_before = client.json("GET", "/v1/session-cache/stats", None)?;
        let off_before = run_interleaved(
            &mut client,
            model,
            &prompts,
            questions,
            false,
            None,
            "cache_off_before",
        )?;
        let stats_after_off_before = client.json("GET", "/v1/session-cache/stats", None)?;
        ensure_no_snapshot_growth(&stats_before, &stats_after_off_before)?;

        let cache_on = run_interleaved(
            &mut client,
            model,
            &prompts,
            questions,
            true,
            Some(&off_before),
            "cache_on",
        )?;
        let stats_after_cache = client.json("GET", "/v1/session-cache/stats", None)?;
        let expected_ram_hits = ((TURNS_PER_CONVERSATION - 1) * prompts.len()) as u64;
        let ram_hit_delta = metric(&stats_after_cache, "/hits/pageable_ram")?
            .checked_sub(metric(&stats_after_off_before, "/hits/pageable_ram")?)
            .ok_or_else(|| "pageable-RAM hit counter moved backwards".to_string())?;
        if ram_hit_delta < expected_ram_hits {
            return Err(format!(
                "expected at least {expected_ram_hits} pageable-RAM hits, observed {ram_hit_delta}"
            ));
        }
        let restore_failure_delta = metric(&stats_after_cache, "/misses/restore_failed")?
            .checked_sub(metric(&stats_after_off_before, "/misses/restore_failed")?)
            .ok_or_else(|| "restore-failure counter moved backwards".to_string())?;
        if restore_failure_delta != 0 {
            return Err(format!(
                "cache-on run recorded {restore_failure_delta} restore failures"
            ));
        }

        let off_after = run_interleaved(
            &mut client,
            model,
            &prompts,
            questions,
            false,
            Some(&off_before),
            "cache_off_after",
        )?;

        let off_before_total = elapsed_total(&off_before.turns);
        let off_after_total = elapsed_total(&off_after.turns);
        let off_bracket_total = (off_before_total + off_after_total) / 2.0;
        let cache_on_total = elapsed_total(&cache_on.turns);
        let off_before_continuations = elapsed_continuations(&off_before.turns);
        let off_after_continuations = elapsed_continuations(&off_after.turns);
        let off_bracket_continuations = (off_before_continuations + off_after_continuations) / 2.0;
        let cache_on_continuations = elapsed_continuations(&cache_on.turns);

        let result = json!({
            "benchmark": "ram_backed_interleaved_agent_conversations_v1",
            "evidence": {"directory": client.root, "diagnostic_distributions": std::env::var_os("KRASIS_CACHE_BENCH_DEBUG").is_some(), "elapsed_definition": "HTTP connect/send/read; durable evidence writes and cache-statistics GETs excluded"},
            "model": model,
            "workload": {
                "canonical_prompts": [
                    "benchmarks/prompts/prompt1_moby_dick.txt",
                    "benchmarks/prompts/prompt2_war_and_peace.txt"
                ],
                "prompt_chars_each": PROMPT_CHARS,
                "conversations": prompts.len(),
                "turns_per_conversation": TURNS_PER_CONVERSATION,
                "completion_tokens_per_turn": COMPLETION_TOKENS,
                "schedule": "interleaved A1,B1,A2,B2,...",
                "cache_off_bracketing": true,
            },
            "identity": {
                "completion_text_equal_on_all_three_runs": true,
                "completion_token_counts_equal_on_all_three_runs": true,
            },
            "cache_evidence": {
                "expected_minimum_pageable_ram_hits": expected_ram_hits,
                "pageable_ram_hit_delta": ram_hit_delta,
                "active_gpu_hit_delta": metric(&stats_after_cache, "/hits/active_gpu")?
                    .saturating_sub(metric(&stats_after_off_before, "/hits/active_gpu")?),
                "restore_failures_delta": restore_failure_delta,
            },
            "wall_clock_ms": {
                "cache_off_before_total": off_before_total,
                "cache_off_after_total": off_after_total,
                "cache_off_bracket_mean_total": off_bracket_total,
                "cache_on_total": cache_on_total,
                "whole_session_speedup": off_bracket_total / cache_on_total,
                "cache_off_bracket_mean_continuations": off_bracket_continuations,
                "cache_on_continuations": cache_on_continuations,
                "continuation_speedup": off_bracket_continuations / cache_on_continuations,
            },
            "turns": {
                "cache_off_before": turn_json(&off_before.turns),
                "cache_on": turn_json(&cache_on.turns),
                "cache_off_after": turn_json(&off_after.turns),
            },
        });
        evidence::save(
            &client.root.join("success.json"),
            &serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?,
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(&result)
                .map_err(|error| format!("failed to serialize benchmark result: {error}"))?
        );
        Ok(())
    })();
    if let Err(error) = &outcome {
        evidence::save(
            &client.root.join("failure.json"),
            &serde_json::to_vec_pretty(&json!({"error": error})).map_err(|e| e.to_string())?,
        )?;
    }
    outcome
}

fn run_interleaved(
    client: &mut Client,
    model: &str,
    prompts: &[String; 2],
    questions: [&[&str; TURNS_PER_CONVERSATION]; 2],
    prefix_cache: bool,
    expected: Option<&ConversationRun>,
    mode: &str,
) -> Result<ConversationRun, String> {
    let mut histories = [Vec::<Value>::new(), Vec::<Value>::new()];
    let mut turns = Vec::with_capacity(TURNS_PER_CONVERSATION * histories.len());
    for turn in 0..TURNS_PER_CONVERSATION {
        for conversation in 0..histories.len() {
            let content = if turn == 0 {
                format!(
                    "{}\n\n{}",
                    prompts[conversation], questions[conversation][turn]
                )
            } else {
                questions[conversation][turn].to_string()
            };
            histories[conversation].push(json!({"role": "user", "content": content}));
            let mut payload = json!({
                "model": model,
                "messages": histories[conversation],
                "temperature": 0,
                "top_k": 1,
                "max_tokens": COMPLETION_TOKENS,
                "min_new_tokens": COMPLETION_TOKENS,
                "stream": false,
                "prefix_cache": prefix_cache,
            });
            if std::env::var_os("KRASIS_CACHE_BENCH_DEBUG").is_some() {
                payload["debug_first_token_boundary"] = json!(true);
                payload["logprobs"] = json!(true);
                payload["top_logprobs"] = json!(10);
            }
            client.context = json!({"mode": mode, "schedule_index": turns.len(), "conversation": conversation, "turn": turn + 1});
            let response = client.json("POST", "/v1/chat/completions", Some(&payload))?;
            let elapsed_ms = client.elapsed_ms;
            client.json("GET", "/v1/session-cache/stats", None)?;
            if std::env::var_os("KRASIS_CACHE_BENCH_DEBUG").is_some() {
                for (pointer, usage) in [
                    ("/krasis_debug/input_token_ids", "/usage/prompt_tokens"),
                    (
                        "/krasis_debug/completion_token_ids",
                        "/usage/completion_tokens",
                    ),
                ] {
                    let ids = response
                        .pointer(pointer)
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("diagnostic response is missing {pointer}"))?;
                    if ids.len() as u64 != metric(&response, usage)?
                        || ids.iter().any(|id| id.as_u64().is_none())
                    {
                        return Err(format!("invalid authoritative token IDs at {pointer}"));
                    }
                }
            }
            if let Some(error) = response.get("error") {
                return Err(format!("server returned an error: {error}"));
            }
            let content = response
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .ok_or_else(|| "response is missing assistant content".to_string())?
                .to_string();
            let input_tokens = metric(&response, "/usage/prompt_tokens")? as usize;
            let completion_tokens = metric(&response, "/usage/completion_tokens")? as usize;
            let index = turns.len();
            if let Some(reference) = expected {
                let reference_turn = reference
                    .turns
                    .get(index)
                    .ok_or_else(|| format!("reference is missing turn {index}"))?;
                if completion_tokens != reference_turn.completion_tokens {
                    return Err(format!(
                        "completion-token count mismatch at interleaved turn {index}: expected {}, got {}",
                        reference_turn.completion_tokens, completion_tokens
                    ));
                }
                if content != reference_turn.content {
                    return Err(format!(
                        "completion text mismatch at interleaved turn {index}"
                    ));
                }
            }
            histories[conversation].push(json!({"role": "assistant", "content": content}));
            turns.push(TurnResult {
                elapsed_ms,
                input_tokens,
                completion_tokens,
                content,
            });
        }
    }
    Ok(ConversationRun { turns })
}

fn turn_json(turns: &[TurnResult]) -> Vec<Value> {
    turns
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            json!({
                "schedule_index": index,
                "conversation": if index % 2 == 0 { "A" } else { "B" },
                "conversation_turn": index / 2 + 1,
                "input_tokens": turn.input_tokens,
                "completion_tokens": turn.completion_tokens,
                "elapsed_ms": turn.elapsed_ms,
            })
        })
        .collect()
}

fn elapsed_total(turns: &[TurnResult]) -> f64 {
    turns.iter().map(|turn| turn.elapsed_ms).sum()
}

fn elapsed_continuations(turns: &[TurnResult]) -> f64 {
    turns
        .iter()
        .enumerate()
        .filter(|(index, _)| *index >= 2)
        .map(|(_, turn)| turn.elapsed_ms)
        .sum()
}

fn ensure_no_snapshot_growth(before: &Value, after: &Value) -> Result<(), String> {
    let before_entries = metric(before, "/resident/snapshots")?;
    let after_entries = metric(after, "/resident/snapshots")?;
    if before_entries != after_entries {
        return Err(format!(
            "cache-off control changed snapshot residency: {before_entries} -> {after_entries}"
        ));
    }
    Ok(())
}

fn metric(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("session-cache stats missing integer {pointer}"))
}

fn read_prefix(path: &Path, chars: usize) -> Result<String, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    Ok(text.chars().take(chars).collect())
}
