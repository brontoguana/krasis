//! Request-admission authentication. No authentication work runs during generation.
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;
use std::sync::Mutex;

const MAX_FILE_BYTES: u64 = 1024 * 1024;

pub(super) struct ApiKeys {
    path: PathBuf,
    // Once observed (or explicitly configured), disappearing files fail closed.
    required: Mutex<bool>,
}

impl ApiKeys {
    pub(super) fn new(path: Option<&str>) -> Result<Self, &'static str> {
        let explicit = path.filter(|p| !p.trim().is_empty());
        let path = match explicit {
            Some(path) => PathBuf::from(path),
            None => {
                let home = match std::env::var_os("KRASIS_HOME").filter(|p| !p.is_empty()) {
                    Some(path) => PathBuf::from(path),
                    None => PathBuf::from(
                        std::env::var_os("HOME")
                            .or_else(|| std::env::var_os("USERPROFILE"))
                            .ok_or("Cannot locate home directory for API key file")?,
                    )
                    .join(".krasis"),
                };
                home.join("api-keys.json")
            }
        };
        let policy = Self {
            path,
            required: Mutex::new(explicit.is_some()),
        };
        policy.load()?;
        Ok(policy)
    }

    #[cfg(test)]
    pub(super) fn for_test_open() -> Self {
        Self {
            path: std::env::temp_dir().join(format!(
                "krasis-no-auth-{}-missing/keys.json",
                std::process::id()
            )),
            required: Mutex::new(false),
        }
    }

    fn load(&self) -> Result<Option<Vec<[u8; 32]>>, &'static str> {
        let mut required = self
            .required
            .lock()
            .map_err(|_| "API key state unavailable")?;
        let file = match std::fs::File::open(&self.path) {
            Ok(file) => {
                *required = true;
                file
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && !*required => {
                // A dangling link is configured authentication, not absence.
                match std::fs::symlink_metadata(&self.path) {
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    _ => {
                        *required = true;
                        return Err("API key file unavailable");
                    }
                }
            }
            Err(_) => {
                *required = true;
                return Err("API key file unavailable");
            }
        };
        let mut bytes = Vec::new();
        file.take(MAX_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "Cannot read API key file")?;
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Err("API key file exceeds 1 MiB");
        }
        // Never include parser errors: they can contain key material.
        let values: std::collections::BTreeMap<String, String> = serde_json::from_slice(&bytes)
            .map_err(|_| "API key file must be a JSON object of names to keys")?;
        let mut hashes = Vec::with_capacity(values.len());
        for (name, key) in values {
            if name.trim().is_empty()
                || key.is_empty()
                || !key.bytes().all(|b| b.is_ascii_graphic())
            {
                return Err("API key names must be nonempty and keys must contain only visible ASCII without spaces");
            }
            hashes.push(Sha256::digest(key.as_bytes()).into());
        }
        Ok(Some(hashes))
    }

    pub(super) fn authorize(&self, method: &str, path: &str, authorization: Option<&str>) -> bool {
        if method == "OPTIONS" || (method == "GET" && path == "/health") {
            return true;
        }
        let hashes = match self.load() {
            Ok(None) => return true,
            Ok(Some(hashes)) => hashes,
            Err(_) => return false,
        };
        let token = match authorization.and_then(|header| header.split_once(' ')) {
            Some((scheme, token))
                if scheme.eq_ignore_ascii_case("Bearer")
                    && !token.is_empty()
                    && token.bytes().all(|b| b.is_ascii_graphic()) =>
            {
                token
            }
            _ => return false,
        };
        let candidate: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let mut matched = 0u8;
        for hash in hashes {
            let mut difference = 0u8;
            for (a, b) in hash.iter().zip(candidate.iter()) {
                difference |= a ^ b;
            }
            // Fixed-length hash comparisons, scanning every configured key.
            matched |= (std::hint::black_box(difference) == 0) as u8;
        }
        matched != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PathBuf, ApiKeys) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "krasis-auth-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let policy = ApiKeys {
            path: path.clone(),
            required: Mutex::new(false),
        };
        (path, policy)
    }
    #[test]
    fn auto_detection_reload_revocation_and_deletion() {
        let (path, policy) = fixture();
        assert!(policy.authorize("POST", "/v1/chat/completions", None));
        std::fs::write(&path, r#"{"desktop":"secret-one","laptop":"secret-two"}"#).unwrap();
        for route in [
            "/v1/chat/completions",
            "/chat/completions/",
            "/v1/models?x=1",
            "/v1/internal/prefill_logits",
        ] {
            assert!(!policy.authorize("POST", route, None));
            assert!(!policy.authorize("POST", route, Some("Bearer bad")));
            assert!(policy.authorize("POST", route, Some("bEaReR secret-one")));
        }
        std::fs::write(&path, r#"{"laptop":"secret-two"}"#).unwrap();
        assert!(!policy.authorize("POST", "/v1/chat/completions", Some("Bearer secret-one")));
        assert!(policy.authorize("GET", "/v1/models", Some("Bearer secret-two")));
        std::fs::write(&path, "{}").unwrap();
        assert!(!policy.authorize("GET", "/v1/models", Some("Bearer secret-two")));
        std::fs::remove_file(&path).unwrap();
        assert!(!policy.authorize("GET", "/v1/models", None));
        assert!(policy.authorize("GET", "/health", None));
        assert!(policy.authorize("OPTIONS", "/v1/chat/completions", None));
    }
    #[cfg(unix)]
    #[test]
    fn dangling_link_and_unreadable_file_cannot_disable_authentication() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let (path, policy) = fixture();
        symlink(path.with_extension("missing"), &path).unwrap();
        assert!(!policy.authorize("GET", "/v1/models", None));
        assert!(ApiKeys::new(Some(path.to_str().unwrap())).is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, r#"{"client":"secret"}"#).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0)).unwrap();
        // Root bypasses Unix permission bits; normal serving users cannot.
        if std::fs::read(&path).is_err() {
            assert!(!policy.authorize("GET", "/v1/models", Some("Bearer secret")));
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(policy.authorize("GET", "/v1/models", Some("Bearer secret")));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_and_missing_explicit_files_fail_closed() {
        let (path, policy) = fixture();
        assert!(ApiKeys::new(Some(path.to_str().unwrap())).is_err());
        for invalid in [
            "null",
            "[]",
            "{",
            r#"{"client":1}"#,
            r#"{"client":""}"#,
            r#"{"":"secret"}"#,
            r#"{"client":"two words"}"#,
        ] {
            std::fs::write(&path, invalid).unwrap();
            assert!(!policy.authorize("GET", "/v1/models", Some("Bearer secret")));
            assert!(ApiKeys::new(Some(path.to_str().unwrap())).is_err());
        }
        std::fs::write(&path, r#"{"client":"secret"}"#).unwrap();
        assert!(ApiKeys::new(Some(path.to_str().unwrap())).is_ok());
        for header in [
            "secret",
            "Basic secret",
            "Bearer ",
            "Bearer secret extra",
            "Bearer  secret",
        ] {
            assert!(!policy.authorize("GET", "/v1/models", Some(header)));
        }
        std::fs::write(&path, vec![b' '; MAX_FILE_BYTES as usize + 1]).unwrap();
        assert!(!policy.authorize("GET", "/v1/models", Some("Bearer secret")));
        std::fs::remove_file(&path).unwrap();
    }
}
