//! Offline real-corpus next-token likelihood client. Run through ./dev.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Component, Path};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, String>;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Corpus preparation and the capture client bind sorted, ensure_ascii JSON.
fn canonical(value: &Value) -> Result<Vec<u8>> {
    use std::fmt::Write as _;
    fn ordered(value: &Value) -> Result<Value> {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                let mut result = serde_json::Map::new();
                for key in keys {
                    result.insert(key.clone(), ordered(&map[key])?);
                }
                Ok(Value::Object(result))
            }
            Value::Array(array) => Ok(Value::Array(
                array.iter().map(ordered).collect::<Result<_>>()?,
            )),
            Value::Number(number) if !number.is_i64() && !number.is_u64() => {
                Err("corpus identity requires integer numbers".into())
            }
            _ => Ok(value.clone()),
        }
    }
    let serialized = serde_json::to_string(&ordered(value)?).map_err(|e| e.to_string())?;
    let mut ascii = String::new();
    for character in serialized.chars() {
        if character.is_ascii() && character != '\u{7f}' {
            ascii.push(character);
        } else {
            let mut utf16 = [0u16; 2];
            for word in character.encode_utf16(&mut utf16) {
                write!(&mut ascii, "\\u{word:04x}").map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(ascii.into_bytes())
}

fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn corpus_source(entry: &Value, archive: Option<&Path>) -> Result<(Vec<u8>, String)> {
    let sha = string(entry, "source_sha256")?;
    if sha.len() != 64 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid source SHA256".into());
    }
    let original = string(entry, "source")?;
    let path = archive
        .map(|root| root.join(sha))
        .unwrap_or_else(|| Path::new(original).to_path_buf());
    // Explicit archive mode never reads the mutable original or substitutes text.
    let source = read(&path)?;
    if source.len() as u64 != number(entry, "source_bytes")? || digest(&source) != sha {
        return Err(format!("source identity mismatch: {}", path.display()));
    }
    Ok((source, path.display().to_string()))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn number(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing integer {key}"))
}

fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let filename = path
        .file_name()
        .ok_or("missing publication filename")?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{filename}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    // Atomic no-overwrite publication: incomplete writes cannot become scores.
    fs::hard_link(&temporary, path).map_err(|e| e.to_string())?;
    fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    fs::File::open(path.parent().ok_or("missing publication parent")?)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn parse_response(raw: &[u8]) -> Result<Vec<u8>> {
    let boundary = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("missing HTTP header boundary")?;
    let header = std::str::from_utf8(&raw[..boundary]).map_err(|e| e.to_string())?;
    let mut lines = header.split("\r\n");
    let status = lines
        .next()
        .ok_or("missing HTTP status")?
        .split_whitespace()
        .nth(1)
        .ok_or("invalid HTTP status")?;
    let mut length = None;
    for line in lines {
        let (key, value) = line.split_once(':').ok_or("invalid HTTP header")?;
        if key.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return Err("duplicate Content-Length".into());
            }
            length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| "invalid Content-Length")?,
            );
        }
        if key.eq_ignore_ascii_case("transfer-encoding") {
            return Err("unexpected transfer encoding".into());
        }
    }
    let body = &raw[boundary + 4..];
    if Some(body.len()) != length {
        return Err("HTTP body length mismatch".into());
    }
    if status != "200" {
        return Err(format!("HTTP {status}: {}", String::from_utf8_lossy(body)));
    }
    Ok(body.to_vec())
}

fn post(address: SocketAddr, payload: &[u8], timeout: Duration) -> Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, timeout).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|_| stream.set_write_timeout(Some(timeout)))
        .map_err(|e| e.to_string())?;
    write!(stream, "POST /v1/internal/prefill_logits HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", payload.len()).map_err(|e| e.to_string())?;
    stream.write_all(payload).map_err(|e| e.to_string())?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| e.to_string())?;
    parse_response(&response)
}

fn score(response: &Value, targets: &[u32]) -> Result<f64> {
    let positions = response
        .get("positions")
        .and_then(Value::as_array)
        .ok_or("missing positions")?;
    if targets.is_empty() || positions.len() != targets.len() {
        return Err("target position count mismatch".into());
    }
    let mut nll = 0.0;
    for (index, (position, target)) in positions.iter().zip(targets).enumerate() {
        if number(position, "position")? != index as u64
            || number(position, "target_token_id")? != u64::from(*target)
        {
            return Err(format!("position/target mismatch at {index}"));
        }
        let logprob = position
            .get("target_logprob")
            .and_then(Value::as_f64)
            .ok_or("missing target log probability")?;
        if !logprob.is_finite() || logprob > 0.0 {
            return Err("invalid target log probability".into());
        }
        nll -= logprob;
    }
    if !nll.is_finite() {
        return Err("nonfinite negative log likelihood".into());
    }
    Ok(nll)
}

fn run(args: &[String]) -> Result<()> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Run through ./dev heldout-nll".into());
    }
    if args.len() != 6 && !(args.len() == 8 && args[6] == "--source-archive") {
        return Err("Usage: ./dev heldout-nll <model-dir> <corpus-dir> <server-address:port> <runtime.conf> <new-output-dir> <timeout-seconds> [--source-archive <directory>]".into());
    }
    let source_archive = args.get(7).map(Path::new);
    let model = Path::new(&args[0]);
    let corpus = Path::new(&args[1]);
    let address: SocketAddr = args[2].parse().map_err(|_| "invalid server address")?;
    let conf = read(Path::new(&args[3]))?;
    let output = Path::new(&args[4]);
    let timeout_seconds: u64 = args[5].parse().map_err(|_| "invalid timeout")?;
    if timeout_seconds == 0 {
        return Err("timeout must be positive".into());
    }
    let manifest_raw = read(&corpus.join("manifest.json"))?;
    let mut manifest: Value = serde_json::from_slice(&manifest_raw).map_err(|e| e.to_string())?;
    let corpus_sha = string(&manifest, "corpus_sha256")?.to_owned();
    manifest
        .as_object_mut()
        .ok_or("manifest must be an object")?
        .remove("corpus_sha256");
    if digest(&canonical(&manifest)?) != corpus_sha {
        return Err("corpus SHA256 mismatch".into());
    }
    let tokenizer_bytes = read(&model.join("tokenizer.json"))?;
    if digest(&tokenizer_bytes) != string(&manifest, "tokenizer_sha256")? {
        return Err("corpus tokenizer identity mismatch".into());
    }
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| e.to_string())?;
    let entries = manifest
        .get("entries")
        .and_then(Value::as_array)
        .ok_or("missing corpus entries")?;
    if entries.is_empty() {
        return Err("empty corpus".into());
    }
    let mut prepared = Vec::new();
    let mut provenance = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        if number(entry, "index")? != index as u64 {
            return Err("noncontiguous corpus entry index".into());
        }
        let name = string(entry, "file")?;
        if Path::new(name).components().count() != 1
            || !matches!(
                Path::new(name).components().next(),
                Some(Component::Normal(_))
            )
        {
            return Err("unsafe corpus filename".into());
        }
        let (source, resolved_source) = corpus_source(entry, source_archive)?;
        provenance.push(json!({"original_source":string(entry, "source")?,"resolved_source":resolved_source,"source_sha256":string(entry, "source_sha256")?}));
        let slice = read(&corpus.join(name))?;
        let start =
            usize::try_from(number(entry, "slice_start")?).map_err(|_| "slice start overflow")?;
        let end = usize::try_from(number(entry, "slice_end")?).map_err(|_| "slice end overflow")?;
        if source.get(start..end) != Some(slice.as_slice())
            || slice.len() as u64 != number(entry, "slice_bytes")?
            || digest(&slice) != string(entry, "slice_sha256")?
        {
            return Err(format!("slice identity mismatch: {name}"));
        }
        let text = std::str::from_utf8(&slice).map_err(|e| e.to_string())?;
        let ids = tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if ids.len() < 2 {
            return Err(format!("insufficient real tokens: {name}"));
        }
        prepared.push((name.to_owned(), digest(&slice), ids));
    }
    fs::create_dir(output).map_err(|e| format!("new output directory: {e}"))?;
    let mut documents = Vec::new();
    let mut total_nll = 0.0;
    let mut total_targets = 0usize;
    for (index, (name, slice_sha, ids)) in prepared.iter().enumerate() {
        let targets = &ids[1..];
        let request = serde_json::to_vec(&json!({"input_token_ids":&ids[..ids.len()-1],"target_token_ids":targets,"sample_every":1,"top_k":1})).map_err(|e| e.to_string())?;
        publish(&output.join(format!("{index:02}.request.json")), &request)?;
        let started = Instant::now();
        let raw = post(address, &request, Duration::from_secs(timeout_seconds))?;
        publish(&output.join(format!("{index:02}.response.json")), &raw)?;
        let response: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        let nll = score(&response, targets)?;
        let token_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let document = json!({"index":index,"file":name,"slice_sha256":slice_sha,"token_ids_sha256":digest(&token_bytes),"target_tokens":targets.len(),"sum_nll":nll,"mean_nll":nll/targets.len() as f64,"request_sha256":digest(&request),"response_sha256":digest(&raw),"http_elapsed_seconds":started.elapsed().as_secs_f64()});
        println!("{document}");
        documents.push(document);
        total_nll += nll;
        total_targets = total_targets
            .checked_add(targets.len())
            .ok_or("target count overflow")?;
    }
    if !total_nll.is_finite() {
        return Err("nonfinite pooled NLL".into());
    }
    let worst = documents
        .iter()
        .map(|d| d["mean_nll"].as_f64().unwrap())
        .reduce(f64::max)
        .ok_or("no document scores")?;
    let result = json!({"format":"Krasis held-out raw next-token likelihood","schema_version":1,"corpus_sha256":corpus_sha,"corpus_manifest_sha256":digest(&manifest_raw),"tokenizer_sha256":digest(&tokenizer_bytes),"runtime_config_sha256":digest(&conf),"server_address":address.to_string(),"source_provenance":provenance,"sum_nll":total_nll,"target_tokens":total_targets,"pooled_mean_nll":total_nll/total_targets as f64,"worst_document_mean_nll":worst,"documents":documents});
    publish(
        &output.join("result.json"),
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    #[cfg(unix)]
    fs::File::open(output)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("heldout-nll: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_source_archive_preserves_identity_and_never_uses_live_source() {
        let root =
            std::env::temp_dir().join(format!("krasis-source-archive-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let original = root.join("original.txt");
        let bytes = b"frozen real source";
        let sha = digest(bytes);
        fs::write(&original, bytes).unwrap();
        let entry = json!({"source":original,"source_sha256":sha,"source_bytes":bytes.len()});
        assert_eq!(corpus_source(&entry, None).unwrap().0, bytes);
        assert!(corpus_source(&entry, Some(&root)).is_err());
        fs::write(root.join(&sha), bytes).unwrap();
        fs::write(&original, b"changed source").unwrap();
        assert!(corpus_source(&entry, None).is_err());
        assert_eq!(corpus_source(&entry, Some(&root)).unwrap().0, bytes);
        fs::write(root.join(&sha), b"corrupt archive").unwrap();
        assert!(corpus_source(&entry, Some(&root)).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn corpus_identity_uses_preparation_unicode_contract() {
        assert_eq!(
            canonical(&json!({"z": ["café🙂", "\u{7f}"], "a": 1})).unwrap(),
            br#"{"a":1,"z":["caf\u00e9\ud83d\ude42","\u007f"]}"#
        );
        assert!(canonical(&json!({"a": 1.5})).is_err());
    }
    #[test]
    fn scoring_requires_every_exact_target_and_real_probability() {
        let valid = json!({"positions":[{"position":0,"target_token_id":8,"target_logprob":-0.5},{"position":1,"target_token_id":9,"target_logprob":-1.25}]});
        assert_eq!(score(&valid, &[8, 9]).unwrap(), 1.75);
        assert!(score(&valid, &[8]).is_err());
        assert!(score(&valid, &[8, 10]).is_err());
        let mut bad = valid.clone();
        bad["positions"][1]["position"] = json!(0);
        assert!(score(&bad, &[8, 9]).is_err());
        bad = valid.clone();
        bad["positions"][1]["target_logprob"] = Value::Null;
        assert!(score(&bad, &[8, 9]).is_err());
        bad = valid;
        bad["positions"][1]["target_logprob"] = json!(0.1);
        assert!(score(&bad, &[8, 9]).is_err());
    }
    #[test]
    fn transport_rejects_truncated_or_ambiguous_responses() {
        assert_eq!(
            parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}").unwrap(),
            b"{}"
        );
        assert!(parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{}").is_err());
        assert!(parse_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}"
        )
        .is_err());
        assert!(parse_response(b"HTTP/1.1 500 Error\r\nContent-Length: 2\r\n\r\n{}").is_err());
        assert!(parse_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}"
        )
        .is_err());
    }

    #[test]
    fn publication_cannot_replace_an_existing_score() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "krasis-heldout-publication-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("result.json");
        publish(&path, b"first verified result").unwrap();
        assert!(publish(&path, b"replacement").is_err());
        assert_eq!(fs::read(path).unwrap(), b"first verified result");
        fs::remove_dir_all(directory).unwrap();
    }
}
