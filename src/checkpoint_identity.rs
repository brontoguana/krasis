//! Immutable source-checkpoint identity shared with Python cache selection.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const IDENTITY_DOMAIN: &[u8] = b"krasis-checkpoint-identity-v1";
const IDENTITY_ENV: &str = "KRASIS_CHECKPOINT_IDENTITIES";

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn weight_names(model_dir: &Path) -> Result<(Vec<String>, Option<PathBuf>), String> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let raw = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("read {}: {e}", index_path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parse {}: {e}", index_path.display()))?;
        let weight_map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                format!(
                    "checkpoint index has no weight_map: {}",
                    index_path.display()
                )
            })?;
        let mut names = BTreeSet::new();
        for value in weight_map.values() {
            let name = value.as_str().ok_or_else(|| {
                format!(
                    "checkpoint index has a non-string shard path: {}",
                    index_path.display()
                )
            })?;
            if !name.ends_with(".safetensors")
                || Path::new(name).file_name().and_then(|v| v.to_str()) != Some(name)
            {
                return Err(format!("checkpoint index has an unsafe shard path: {name}"));
            }
            names.insert(name.to_string());
        }
        if names.is_empty() {
            return Err(format!(
                "checkpoint index has no shards: {}",
                index_path.display()
            ));
        }
        return Ok((names.into_iter().collect(), Some(index_path)));
    }
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok((vec!["model.safetensors".to_string()], None));
    }
    Err(format!(
        "no complete safetensors checkpoint in {}",
        model_dir.display()
    ))
}

fn metadata_record(model_dir: &Path, name: &str) -> Option<(String, String, f64)> {
    let path = model_dir
        .join(".cache")
        .join("huggingface")
        .join("download")
        .join(format!("{name}.metadata"));
    let raw = std::fs::read_to_string(path).ok()?;
    let mut lines = raw.lines();
    let revision = lines.next()?.trim().to_ascii_lowercase();
    let object_id = lines.next()?.trim().trim_matches('"').to_ascii_lowercase();
    let completed_at = lines.next()?.trim().parse::<f64>().ok()?;
    let is_hex = |value: &str| value.bytes().all(|ch| ch.is_ascii_hexdigit());
    if revision.len() != 40 || !is_hex(&revision) {
        return None;
    }
    if !matches!(object_id.len(), 40 | 64) || !is_hex(&object_id) {
        return None;
    }
    Some((revision, object_id, completed_at))
}

fn feed_field(digest: &mut Sha256, label: &str, value: &str) {
    digest.update((label.len() as u64).to_be_bytes());
    digest.update(label.as_bytes());
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value.as_bytes());
}

fn registered_identity(model_dir: &Path) -> Option<String> {
    let raw = std::env::var(IDENTITY_ENV).ok()?;
    let values: Value = serde_json::from_str(&raw).ok()?;
    let canonical = model_dir.canonicalize().ok()?;
    values
        .get(canonical.to_string_lossy().as_ref())?
        .as_str()
        .filter(|value| value.len() == 64 && value.bytes().all(|ch| ch.is_ascii_hexdigit()))
        .map(str::to_ascii_lowercase)
}

pub fn checkpoint_identity_sha256(model_dir: &Path) -> Result<String, String> {
    if let Some(identity) = registered_identity(model_dir) {
        return Ok(identity);
    }
    let model_dir = model_dir
        .canonicalize()
        .map_err(|e| format!("resolve {}: {e}", model_dir.display()))?;
    let (names, _index_path) = weight_names(&model_dir)?;

    let mut metadata = Vec::new();
    let mut revision: Option<String> = None;
    let mut complete_hf = true;
    for name in &names {
        let path = model_dir.join(name);
        if !path.is_file() {
            return Err(format!("checkpoint shard is missing: {}", path.display()));
        }
        let Some((file_revision, object_id, completed_at)) = metadata_record(&model_dir, name)
        else {
            complete_hf = false;
            break;
        };
        let modified = path
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|value| value.as_secs_f64());
        if modified.is_none_or(|value| {
            let completion_delta = completed_at - value;
            completion_delta < -0.001
        }) {
            complete_hf = false;
            break;
        }
        if revision
            .as_ref()
            .is_some_and(|current| current != &file_revision)
        {
            complete_hf = false;
            break;
        }
        revision.get_or_insert(file_revision);
        metadata.push(object_id);
    }

    if !(complete_hf && revision.is_some() && metadata.iter().all(|value| value.len() == 64)) {
        metadata.clear();
        complete_hf = false;
    }

    let mut digest = Sha256::new();
    digest.update(IDENTITY_DOMAIN);
    for (index, name) in names.iter().enumerate() {
        let path = model_dir.join(name);
        let size = path
            .metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        let object_id = if complete_hf {
            metadata[index].clone()
        } else {
            sha256_file(&path)?
        };
        feed_field(&mut digest, "weight_name", name);
        feed_field(&mut digest, "weight_size", &size.to_string());
        feed_field(&mut digest, "weight_object_id", &object_id);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn cache_namespace(model_dir: &Path) -> Result<String, String> {
    let name = model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let safe_name: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let checkpoint_sha256 = checkpoint_identity_sha256(model_dir)?;
    let canonical = model_dir
        .canonicalize()
        .map_err(|e| format!("resolve {}: {e}", model_dir.display()))?;
    let (_names, index_path) = weight_names(&canonical)?;
    let mut controls = vec![canonical.join("config.json")];
    if let Some(index_path) = index_path {
        controls.push(index_path);
    }
    controls.sort_by_key(|path| path.file_name().map(|value| value.to_os_string()));
    let mut cache_digest = Sha256::new();
    cache_digest.update(b"krasis-cache-identity-v1");
    feed_field(&mut cache_digest, "checkpoint_sha256", &checkpoint_sha256);
    for path in controls {
        if !path.is_file() {
            return Err(format!(
                "checkpoint control file is missing: {}",
                path.display()
            ));
        }
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        let size = path
            .metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        feed_field(&mut cache_digest, "control_name", name);
        feed_field(&mut cache_digest, "control_size", &size.to_string());
        feed_field(&mut cache_digest, "control_sha256", &sha256_file(&path)?);
    }
    let cache_sha256 = format!("{:x}", cache_digest.finalize());
    Ok(format!(
        "{}--{}",
        if safe_name.is_empty() {
            "model"
        } else {
            &safe_name
        },
        cache_sha256,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture(payload: &[u8], with_metadata: bool) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir()
            .join(format!("krasis-checkpoint-{nonce}"))
            .join("model");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), b"{\"model_type\":\"test\"}\n").unwrap();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            b"{\"weight_map\": {\"tensor.0\": \"model.safetensors\"}}",
        )
        .unwrap();
        std::fs::write(root.join("model.safetensors"), payload).unwrap();
        if with_metadata {
            let metadata_dir = root.join(".cache").join("huggingface").join("download");
            std::fs::create_dir_all(&metadata_dir).unwrap();
            let object_id = format!("{:x}", Sha256::digest(payload));
            let weight_path = root.join("model.safetensors");
            let completed_at = weight_path
                .metadata()
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            std::fs::write(
                metadata_dir.join("model.safetensors.metadata"),
                format!("{}\n{}\n{completed_at}\n", "a".repeat(40), object_id),
            )
            .unwrap();
        }
        root
    }

    #[test]
    fn hf_and_full_content_paths_produce_the_same_identity() {
        std::env::remove_var(IDENTITY_ENV);
        let hf = fixture(b"payload", true);
        let local = fixture(b"payload", false);
        let hf_identity = checkpoint_identity_sha256(&hf).unwrap();
        let local_identity = checkpoint_identity_sha256(&local).unwrap();
        assert_eq!(hf_identity, local_identity);
        assert_eq!(
            hf_identity,
            "4daab4e4e3a31165b08ea5172f8baf9cb2fe2934ce2311be03b7f2c755b4c833"
        );
        assert_eq!(
            cache_namespace(&hf).unwrap(),
            "model--f8b7c37d8aa31473fa24ca46e340feba6e3393aeb0b0026059f83cf3d5a65743"
        );
        let _ = std::fs::remove_dir_all(hf.parent().unwrap());
        let _ = std::fs::remove_dir_all(local.parent().unwrap());
    }

    #[test]
    fn same_size_content_change_changes_identity() {
        std::env::remove_var(IDENTITY_ENV);
        let root = fixture(b"aaaa", false);
        let before = checkpoint_identity_sha256(&root).unwrap();
        std::fs::write(root.join("model.safetensors"), b"bbbb").unwrap();
        let after = checkpoint_identity_sha256(&root).unwrap();
        assert_ne!(before, after);
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }
}
