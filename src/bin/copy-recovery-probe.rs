//! Installed HTTP controls. Run through ./dev copy-recovery-probe only.
#[path = "session_cache_benchmark/evidence.rs"]
mod evidence;
use evidence::{save, Client};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;
fn main() {
    if let Err(e) = run() {
        eprintln!("copy recovery probe: {e}");
        std::process::exit(1)
    }
}
fn decisions(v: &Value) -> Result<Value, String> {
    let d = v
        .get("krasis_debug")
        .ok_or("missing genuine token debug evidence")?;
    let ids = d
        .get("completion_token_ids")
        .ok_or("missing generated token IDs")?;
    Ok(json!({"ids":ids,"content":v["choices"][0]["message"]["content"]}))
}
fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev copy-recovery-probe".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("expected port, genuine seed request and fresh output directory".into());
    }
    let port: u16 = args[0].parse().map_err(|e| format!("port: {e}"))?;
    let mut request: Value =
        serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    request["prefix_cache"] = json!(false);
    let root = PathBuf::from(&args[2]);
    let mut client = Client::new(port, root.clone())?;
    client.context = json!({"phase":"before","note":"genuine seed, forced full recomputation; copy error control is separate from normal pickup gate"});
    let before = client.json("POST", "/v1/chat/completions", Some(&request))?;
    let baseline = decisions(&before)?;
    let endpoint = "/v1/internal/sequence_state_transfer_measurement";
    let poison = client.json(
        "POST",
        endpoint,
        Some(&json!({"logical_tokens":1,"iterations":1,"copy_error_action":"poison"})),
    )?;
    let devices = poison["devices"]
        .as_array()
        .ok_or("poison devices absent")?;
    if devices.is_empty()
        || !devices
            .iter()
            .all(|d| d["poisoned"] == true && d["query_after_return"] == "CUDA_SUCCESS")
    {
        return Err("poison/completion evidence failed".into());
    }
    client.context = json!({"phase":"poisoned ordinary request"});
    let rejected = client
        .json("POST", "/v1/chat/completions", Some(&request))
        .expect_err("ordinary request accepted partial restore");
    if !rejected.contains("HTTP 500") || !rejected.contains("invalid after an interrupted restore")
    {
        return Err(format!("unexpected rejection: {rejected}"));
    }
    let recovery = client.json(
        "POST",
        endpoint,
        Some(&json!({"logical_tokens":1,"iterations":1,"copy_error_action":"recover"})),
    )?;
    if !recovery["devices"]
        .as_array()
        .ok_or("recovery devices absent")?
        .iter()
        .all(|d| d["recovered"] == true && d["exact_original_bytes"] == true)
    {
        return Err("explicit recovery evidence failed".into());
    }
    client.context = json!({"phase":"after explicit recovery"});
    let after = client.json("POST", "/v1/chat/completions", Some(&request))?;
    if decisions(&after)? != baseline {
        return Err("ordinary seed output changed after explicit recovery".into());
    }
    // Disconnect only after a real nonempty token has been received, not on a timer.
    let mut streaming = request.clone();
    streaming["stream"] = json!(true);
    streaming["max_tokens"] = json!(256);
    let text = serde_json::to_string(&streaming).map_err(|e| e.to_string())?;
    let wire=format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",text.len());
    save(&root.join("disconnect-request.http"), wire.as_bytes())?;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(1800)))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(wire.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    let mut content_received = false;
    while !content_received {
        let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        for line in String::from_utf8_lossy(&raw).lines() {
            if let Some(s) = line.strip_prefix("data: ") {
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    content_received = v["choices"][0]["delta"]["content"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty());
                    if content_received {
                        break;
                    }
                }
            }
        }
        if raw.len() > 1024 * 1024 {
            return Err("disconnect stream exceeded bounded evidence before content".into());
        }
    }
    save(&root.join("disconnect-response-partial.http"), &raw)?;
    drop(stream);
    if !content_received {
        return Err("stream ended before genuine content; cancellation untested".into());
    }
    client.context = json!({"phase":"after real disconnect"});
    let final_response = client.json("POST", "/v1/chat/completions", Some(&request))?;
    if decisions(&final_response)? != baseline {
        return Err("ordinary seed output changed after disconnect".into());
    }
    save(&root.join("REPORT.json"),&serde_json::to_vec_pretty(&json!({"passed":true,"post_enqueue_error_completed":true,"ordinary_poisoned_request_rejected":true,"explicit_original_restore_exact":true,"post_disconnect_seed_exact":true,"device_count":devices.len(),"scope":"installed real Store callback event failure and actual HTTP rejection/recovery/disconnect; no real CUDA context failure injected","serving_qualified":false})).map_err(|e|e.to_string())?)?;
    Ok(())
}
