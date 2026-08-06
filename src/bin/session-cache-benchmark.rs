//! Reproducible end-to-end benchmark for RAM-backed multi-conversation reuse.
//!
//! Run through `./dev session-cache-benchmark <port>`. The server must be
//! started with session caching enabled. Two canonical
//! Gutenberg conversations are interleaved so every continuation exercises a
//! real pageable-RAM restore rather than the active-GPU fast path. Cache-off
//! runs bracket the cache-on run to expose process-local timing drift.

use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::{Duration, Instant};

const PROMPT_CHARS: usize = 40_000;
const TURNS_PER_CONVERSATION: usize = 4;
const COMPLETION_TOKENS: usize = 8;
const REQUEST_TIMEOUT_SECS: u64 = 1_800;

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
    if args.next().is_some() {
        return Err("usage: session-cache-benchmark <port>".to_string());
    }

    let model_response = http_json(port, "GET", "/v1/models", None)?;
    let model = model_response
        .pointer("/data/0/id")
        .and_then(Value::as_str)
        .ok_or_else(|| "server did not expose exactly one model id".to_string())?;
    let initial_stats = http_json(port, "GET", "/v1/session-cache/stats", None)?;
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
    http_json(port, "POST", "/v1/chat/completions", Some(&warmup))?;

    let stats_before = http_json(port, "GET", "/v1/session-cache/stats", None)?;
    let off_before = run_interleaved(port, model, &prompts, questions, false, None)?;
    let stats_after_off_before = http_json(port, "GET", "/v1/session-cache/stats", None)?;
    ensure_no_snapshot_growth(&stats_before, &stats_after_off_before)?;

    let cache_on = run_interleaved(port, model, &prompts, questions, true, Some(&off_before))?;
    let stats_after_cache = http_json(port, "GET", "/v1/session-cache/stats", None)?;
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

    let off_after = run_interleaved(port, model, &prompts, questions, false, Some(&off_before))?;

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
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|error| format!("failed to serialize benchmark result: {error}"))?
    );
    Ok(())
}

fn run_interleaved(
    port: u16,
    model: &str,
    prompts: &[String; 2],
    questions: [&[&str; TURNS_PER_CONVERSATION]; 2],
    prefix_cache: bool,
    expected: Option<&ConversationRun>,
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
            let payload = json!({
                "model": model,
                "messages": histories[conversation],
                "temperature": 0,
                "top_k": 1,
                "max_tokens": COMPLETION_TOKENS,
                "min_new_tokens": COMPLETION_TOKENS,
                "stream": false,
                "prefix_cache": prefix_cache,
            });
            let started = Instant::now();
            let response = http_json(port, "POST", "/v1/chat/completions", Some(&payload))?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
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

fn http_json(port: u16, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
    let address = ("127.0.0.1", port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve server address: {error}"))?
        .next()
        .ok_or_else(|| "server address did not resolve".to_string())?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))
        .map_err(|error| format!("failed to connect to server: {error}"))?;
    let timeout = Some(Duration::from_secs(REQUEST_TIMEOUT_SECS));
    stream
        .set_read_timeout(timeout)
        .map_err(|error| format!("failed to set read timeout: {error}"))?;
    stream
        .set_write_timeout(timeout)
        .map_err(|error| format!("failed to set write timeout: {error}"))?;

    let body_text = match body {
        Some(value) => serde_json::to_string(value)
            .map_err(|error| format!("failed to serialize request: {error}"))?,
        None => String::new(),
    };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body_text.len(),
        body_text,
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("failed to write HTTP request: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("failed to read HTTP response: {error}"))?;
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "HTTP response is missing its header separator".to_string())?;
    let header = std::str::from_utf8(&response[..separator])
        .map_err(|error| format!("HTTP response header is not UTF-8: {error}"))?;
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| "HTTP response is missing a status code".to_string())?;
    let response_body = &response[separator + 4..];
    if status != 200 {
        return Err(format!(
            "server returned HTTP {status}: {}",
            String::from_utf8_lossy(response_body)
        ));
    }
    serde_json::from_slice(response_body)
        .map_err(|error| format!("server returned invalid JSON: {error}"))
}
