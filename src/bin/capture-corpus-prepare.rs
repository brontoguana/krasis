#[path = "../chat_template.rs"]
mod chat_template;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_digest(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn load(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn integer(value: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        value
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("missing integer {key}"))?,
    )
    .map_err(|_| format!("integer {key} overflows usize"))
}

// Match the existing corpus client's sorted, ensure_ascii JSON identity.
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
            Value::Number(n) if !n.is_i64() && !n.is_u64() => {
                Err("corpus identity requires integer numbers".into())
            }
            _ => Ok(value.clone()),
        }
    }
    let serialized = serde_json::to_string(&ordered(value)?).map_err(|e| e.to_string())?;
    let mut ascii = String::new();
    for c in serialized.chars() {
        if c.is_ascii() && c != '\u{7f}' {
            ascii.push(c);
        } else {
            let mut utf16 = [0u16; 2];
            for word in c.encode_utf16(&mut utf16) {
                write!(&mut ascii, "\\u{word:04x}").map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(ascii.into_bytes())
}

fn token_digest(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn request_digest(corpus: &str, ids: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(b"Krasis KTC1 transactional request v1\0");
    h.update(corpus.as_bytes());
    h.update(0u64.to_le_bytes());
    h.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn shorten(
    text: &str,
    full: &[u32],
    rows: usize,
    ceiling: usize,
    mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<(usize, Vec<u32>)> {
    if rows == 0 || full.len() < rows || ceiling < rows {
        return Err(format!("invalid token-prefix capture quota or ceiling: source_tokens={} capture_rows={rows} ceiling={ceiling}", full.len()));
    }
    let boundaries: Vec<usize> = std::iter::once(0)
        .chain(text.char_indices().skip(1).map(|(i, _)| i))
        .chain(std::iter::once(text.len()))
        .collect();
    let mut low = 0;
    let mut high = boundaries.len() - 1;
    while low < high {
        let mid = low + (high - low) / 2;
        if encode(&text[..boundaries[mid]])?.len() >= rows {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    for &end in &boundaries[low..] {
        let ids = encode(&text[..end])?;
        if ids.starts_with(&full[..rows]) {
            if ids.len() > ceiling || end == text.len() {
                return Err(format!(
                    "no shorter identical-prefix prompt fits {ceiling} tokens (got {})",
                    ids.len()
                ));
            }
            return Ok((end, ids));
        }
        if ids.len() > ceiling {
            return Err("token prefix changed before a bounded candidate was found".into());
        }
    }
    Err("no shorter prompt preserves the requested token prefix".into())
}

fn publish_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)
            .and_then(|f| f.sync_all())
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(0x02000000)
            .open(path)
            .and_then(|f| f.sync_all())
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn preparation_reference(args: &HashMap<String, String>) -> Result<Option<PathBuf>> {
    match args
        .get("preparation-mode")
        .map(String::as_str)
        .unwrap_or("matched")
    {
        "matched" => args
            .get("reference-capture")
            .map(|s| Some(PathBuf::from(s)))
            .ok_or_else(|| "matched preparation requires --reference-capture".into()),
        "independent" => {
            if args.contains_key("reference-capture") {
                return Err("independent preparation cannot claim a reference capture".into());
            }
            Ok(None)
        }
        _ => Err("preparation-mode must be matched or independent".into()),
    }
}

fn prepare(args: &HashMap<String, String>) -> Result<()> {
    let required = |key: &str| {
        args.get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("required --{key}"))
    };
    let input = PathBuf::from(required("input-corpus")?)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let reference = preparation_reference(args)?;
    let model = PathBuf::from(required("model-dir")?)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let requested_output = PathBuf::from(required("output-corpus")?);
    let parent = requested_output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()
        .map_err(|e| format!("output parent: {e}"))?;
    let output = parent.join(
        requested_output
            .file_name()
            .ok_or("invalid output filename")?,
    );
    let rows: usize = required("capture-rows")?
        .parse()
        .map_err(|_| "capture-rows must be a positive integer")?;
    let thinking = match required("thinking")? {
        "enabled" => true,
        "disabled" => false,
        _ => return Err("thinking must be enabled or disabled".into()),
    };
    if rows == 0 {
        return Err("capture-rows must be nonzero".into());
    }
    let config = model.join("tokenizer_config.json");
    let config_json = load(&config)?;
    let embedded_template = match config_json.get("chat_template") {
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(templates)) => templates.iter().any(|t| {
            t.get("name").and_then(Value::as_str).is_some()
                && t.get("template")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty())
        }),
        _ => false,
    };
    if !embedded_template && !model.join("chat_template.jinja").is_file() {
        return Err(
            "explicit checkpoint chat template required; no default-template fallback".into(),
        );
    }
    let template = chat_template::ChatTemplateEngine::from_config(
        config.to_str().ok_or("non-UTF8 model path")?,
    )?;
    let tokenizer_path = model.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
    let encode = |text: &str| -> Result<Vec<u32>> {
        let messages = json!([{"role":"user", "content":text}]).to_string();
        let rendered = template.apply(&messages, true, thinking)?;
        Ok(tokenizer
            .encode(rendered, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec())
    };
    let ceiling = rows
        .checked_add(encode("")?.len())
        .ok_or("token ceiling overflow")?;
    let mut original = load(&input.join("manifest.json"))?;
    let corpus_hash = string(&original, "corpus_sha256")?.to_string();
    original
        .as_object_mut()
        .ok_or("corpus must be an object")?
        .remove("corpus_sha256");
    if digest(&canonical(&original)?) != corpus_hash {
        return Err("input corpus SHA-256 mismatch".into());
    }
    let mut accepted = HashSet::new();
    if let Some(reference) = &reference {
        let ref_metadata = load(&reference.join("capture.json"))?;
        if string(&ref_metadata, "calibration_sha256")? != corpus_hash {
            return Err("reference capture/corpus binding mismatch".into());
        }
        for entry in std::fs::read_dir(reference.join("requests")).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("non-UTF8 reference request path")?;
            if name.starts_with(".request-") && name.ends_with(".tmp") {
                continue;
            }
            let manifest = load(&path.join("request.json"))?;
            if integer(&manifest, "sequence_start")? != 0
                || string(&manifest, "calibration_sha256")? != corpus_hash
            {
                return Err("reference request has incompatible sequence/corpus".into());
            }
            accepted.insert(string(&manifest, "request_sha256")?.to_string());
        }
    }
    let temporary = output.with_file_name(format!(
        "{}.tmp",
        output
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("invalid output path")?
    ));
    if output.exists() || temporary.exists() {
        return Err("refusing to overwrite corpus or partial corpus".into());
    }
    // Verify durability support before creating any corpus files.
    sync_dir(&parent)?;
    std::fs::create_dir(&temporary).map_err(|e| e.to_string())?;
    let mut entries = Vec::new();
    let mut requests = Vec::new();
    for (index, entry) in original
        .get("entries")
        .and_then(Value::as_array)
        .ok_or("missing corpus entries")?
        .iter()
        .enumerate()
    {
        if integer(entry, "index")? != index {
            return Err("corpus entry order mismatch".into());
        }
        let source = PathBuf::from(string(entry, "source")?);
        let source_raw =
            std::fs::read(&source).map_err(|e| format!("{}: {e}", source.display()))?;
        if source_raw.len() != integer(entry, "source_bytes")?
            || digest(&source_raw) != string(entry, "source_sha256")?
        {
            return Err(format!("source identity changed: {}", source.display()));
        }
        let name = string(entry, "file")?;
        if Path::new(name).components().count() != 1 {
            return Err("corpus slice must be a local filename".into());
        }
        let raw = std::fs::read(input.join(name)).map_err(|e| e.to_string())?;
        if raw.len() != integer(entry, "slice_bytes")?
            || digest(&raw) != string(entry, "slice_sha256")?
            || source_raw.get(integer(entry, "slice_start")?..integer(entry, "slice_end")?)
                != Some(raw.as_slice())
        {
            return Err(format!("corpus slice identity changed: {name}"));
        }
        let text = std::str::from_utf8(&raw).map_err(|e| e.to_string())?;
        let full = encode(text)?;
        let full_request = request_digest(&corpus_hash, &full);
        if reference.is_some() && !accepted.contains(&full_request) {
            return Err(format!("Rust tokenizer/template does not match captured full request: {name} hash={full_request}"));
        }
        let (end, ids) = shorten(text, &full, rows, ceiling, &encode)
            .map_err(|error| format!("{name}: {error}"))?;
        let short = &raw[..end];
        publish_file(&temporary.join(name), short)?;
        let mut new_entry = entry.clone();
        new_entry["slice"] = json!(if reference.is_some() {
            "identical-token-prefix"
        } else {
            "token-bounded-real-source"
        });
        new_entry["slice_end"] = json!(integer(entry, "slice_start")?
            .checked_add(end)
            .ok_or("slice offset overflow")?);
        new_entry["slice_bytes"] = json!(end);
        new_entry["slice_sha256"] = json!(digest(short));
        new_entry["rendered_prompt_tokens"] = json!(ids.len());
        new_entry["rendered_tokens_sha256"] = json!(token_digest(&ids));
        new_entry["capture_prefix_tokens"] = json!(rows);
        new_entry["capture_prefix_sha256"] = json!(token_digest(&ids[..rows]));
        if reference.is_some() {
            new_entry["full_reference_request_sha256"] = json!(full_request);
            new_entry["full_reference_prompt_tokens"] = json!(full.len());
        } else {
            new_entry["source_slice_prompt_tokens"] = json!(full.len());
        }
        println!(
            "{}",
            json!({"index":index,"file":name,"full_tokens":full.len(),"short_tokens":ids.len(),"capture_rows":rows,"prefix_exact":true})
        );
        requests.push((index, name.to_string(), ids));
        entries.push(new_entry);
    }
    if entries.is_empty() {
        return Err("input corpus has no entries".into());
    }
    let mut manifest = json!({
        "schema_version":1, "format":"Krasis TileQ calibration corpus",
        "half":if reference.is_some() {"identical-token-prefix"} else {"token-bounded-real-source"},
        "preparation":if reference.is_some() {"rust-exact-token-prefix-v1"} else {"rust-independent-token-bounded-v1"},
        "capture_rows_per_request":rows,
        "rendered_token_ceiling":ceiling, "enable_thinking":thinking,
        "tokenizer_sha256":file_digest(&tokenizer_path)?, "tokenizer_config_sha256":file_digest(&config)?,
        "chat_template_file_sha256":if model.join("chat_template.jinja").is_file() { Some(file_digest(&model.join("chat_template.jinja"))?) } else { None },
        "entries":entries,
    });
    if reference.is_some() {
        manifest["reference_corpus_sha256"] = json!(corpus_hash);
    } else {
        manifest["source_corpus_sha256"] = json!(corpus_hash);
    }
    manifest["corpus_sha256"] = json!(digest(&canonical(&manifest)?));
    let mut bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    publish_file(&temporary.join("manifest.json"), &bytes)?;
    let identities: Vec<_> = requests
        .iter()
        .map(|(index, name, ids)| {
            json!({"index":index,"file":name,"prompt_tokens":ids.len(),
            "request_sha256":request_digest(manifest["corpus_sha256"].as_str().unwrap(),ids),
            "token_ids_sha256":token_digest(ids)})
        })
        .collect();
    let identity_bytes = serde_json::to_vec_pretty(&json!({
        "format":"Krasis KTC1 prepared request identities v1",
        "corpus_sha256":manifest["corpus_sha256"],"requests":identities,
    }))
    .map_err(|e| e.to_string())?;
    publish_file(&temporary.join("request-identities.json"), &identity_bytes)?;
    sync_dir(&temporary)?;
    std::fs::rename(&temporary, &output).map_err(|e| e.to_string())?;
    sync_dir(output.parent().unwrap_or(Path::new(".")))?;
    println!(
        "{}",
        json!({"status":"complete","corpus":output,"corpus_sha256":manifest["corpus_sha256"]})
    );
    Ok(())
}

fn run() -> Result<()> {
    let mut args = HashMap::new();
    let mut cli = std::env::args().skip(1);
    while let Some(key) = cli.next() {
        let key = key
            .strip_prefix("--")
            .ok_or("arguments must be --key value pairs")?;
        if ![
            "input-corpus",
            "reference-capture",
            "preparation-mode",
            "model-dir",
            "output-corpus",
            "capture-rows",
            "thinking",
        ]
        .contains(&key)
        {
            return Err(format!("unknown option --{key}"));
        }
        let value = cli
            .next()
            .ok_or_else(|| format!("missing value for --{key}"))?;
        if args.insert(key.to_string(), value).is_some() {
            return Err(format!("duplicate option --{key}"));
        }
    }
    prepare(&args)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;
    fn encode(s: &str) -> Result<Vec<u32>> {
        Ok(std::iter::once(1)
            .chain(s.chars().map(|c| c as u32))
            .chain(std::iter::once(2))
            .collect())
    }
    #[test]
    fn unicode_prefix_and_template_suffix_are_preserved() {
        let text = "café🙂more actual text";
        let full = encode(text).unwrap();
        let (bytes, ids) = shorten(text, &full, 6, 8, encode).unwrap();
        assert_eq!(&text[..bytes], "café🙂");
        assert!(ids.starts_with(&full[..6]));
        assert_eq!(ids.len(), 7);
        assert!(shorten(text, &full, 6, 6, encode).is_err());
        assert!(shorten(text, &full, full.len(), full.len() + 2, encode).is_err());
        assert!(shorten(text, &full, 0, 5, encode).is_err());
    }
    #[test]
    fn canonical_identity_matches_existing_ascii_unicode_contract() {
        assert_eq!(
            canonical(&json!(["café🙂", "line\n"])).unwrap(),
            br#"["caf\u00e9\ud83d\ude42","line\n"]"#
        );
        assert_eq!(
            request_digest(&"11".repeat(32), &[1, 2]),
            "b2a0b65212c293b7a88c30764a362de4ac49bf03f57b04e843afad87c2f676b7"
        );
    }
    #[test]
    fn independent_preparation_never_silently_drops_reference_validation() {
        let mut args = HashMap::new();
        assert!(preparation_reference(&args).is_err());
        args.insert("reference-capture".into(), "accepted".into());
        assert_eq!(
            preparation_reference(&args).unwrap(),
            Some(PathBuf::from("accepted"))
        );
        args.insert("preparation-mode".into(), "independent".into());
        assert!(preparation_reference(&args).is_err());
        args.remove("reference-capture");
        assert_eq!(preparation_reference(&args).unwrap(), None);
        args.insert("preparation-mode".into(), "unknown".into());
        assert!(preparation_reference(&args).is_err());
    }
}
