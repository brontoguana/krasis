//! Explicit provenance for one actual prefill invocation. No CUDA work.
use crate::matched_probe::{self, Target};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct Invocation {
    namespace: String,
    logical_tokens: usize,
    metadata: Vec<u8>,
}

impl Invocation {
    pub(crate) fn begin(
        device: usize,
        request: Option<&str>,
        tokens: &[u32],
        start: usize,
    ) -> Result<Option<Self>, String> {
        if !matched_probe::enabled()? {
            return Ok(None);
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "prefill capture invocation counter overflow".to_string())?;
        Self::new(sequence, device, request, tokens, start).map(Some)
    }

    fn new(
        sequence: u64,
        device: usize,
        request: Option<&str>,
        tokens: &[u32],
        start: usize,
    ) -> Result<Self, String> {
        if tokens.is_empty() {
            return Err("empty prefill capture history".into());
        }
        let logical = start
            .checked_add(tokens.len())
            .ok_or("prefill capture position overflow")?;
        if request.is_some_and(|r| {
            r.is_empty()
                || !r
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        }) {
            return Err("unsafe prefill capture request identity".into());
        }
        let mut hash = Sha256::new();
        for token in tokens {
            hash.update(token.to_le_bytes());
        }
        let namespace = match request {
            Some(r) => format!("request-{r}-prefill-{sequence}-device-{device}"),
            None => format!("prefill-{sequence}-device-{device}"),
        };
        let metadata = serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version":1,"namespace":namespace,"invocation":sequence,
            "request_id":request,"device_ordinal":device,"sequence_start":start,
            "suffix_tokens":tokens.len(),"logical_tokens":logical,
            "input_token_ids_sha256":format!("{:x}",hash.finalize()),
            "input_encoding":"little-endian u32; actual consumed prefill suffix",
            "device_diagnostic_allocation_bytes":0,"qualifies_candidate":false
        }))
        .map_err(|e| e.to_string())?;
        Ok(Self {
            namespace,
            logical_tokens: logical,
            metadata,
        })
    }

    pub(crate) fn logical_tokens(&self) -> usize { self.logical_tokens }

    fn scope(&self, target: Target) -> Result<Target, String> {
        let target = target.child(&self.namespace)?;
        target.write("prefill-invocation.json", &self.metadata)?;
        Ok(target)
    }
}

pub(crate) fn target(
    invocation: Option<&Invocation>,
    operation: &str,
    layer: usize,
    height: usize,
    selection_label: Option<&str>,
) -> Result<Option<Target>, String> {
    let Some(target) =
        matched_probe::target_for_request(operation, layer, height, selection_label)?
    else {
        return Ok(None);
    };
    scope(invocation, target).map(Some)
}

fn scope(invocation: Option<&Invocation>, target: Target) -> Result<Target, String> {
    invocation
        .ok_or("selected prefill capture has no actual invocation identity")?
        .scope(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_prefill_namespace_separates_repeats_retries_and_devices() {
        let root = std::env::temp_dir().join(format!(
            "prefill-capture-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir(&root).unwrap();
        let make = |n, d, r| Invocation::new(n, d, r, &[1, 0x12345678], 17).unwrap();
        let first = make(0, 0, Some("chatcmpl-test"));
        let second = make(1, 0, Some("chatcmpl-test"));
        let device = make(0, 2, Some("chatcmpl-test"));
        let other = make(0, 0, Some("chatcmpl-other"));
        let mut paths = std::collections::HashSet::new();
        for (i, inv) in [&first, &second, &device, &other].into_iter().enumerate() {
            let t = inv
                .scope(Target {
                    row: 0,
                    directory: root.clone(),
                })
                .unwrap();
            assert!(paths.insert(t.directory.clone()));
            t.write("descriptor.json", format!("pointer-{i}").as_bytes())
                .unwrap();
            assert!(t.write("descriptor.json", b"changed-pointer").is_err());
            let meta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(t.directory.join("prefill-invocation.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(meta["logical_tokens"], 19);
            assert_eq!(meta["sequence_start"], 17);
            let mut h = Sha256::new();
            h.update([1, 0, 0, 0, 0x78, 0x56, 0x34, 0x12]);
            assert_eq!(
                meta["input_token_ids_sha256"],
                format!("{:x}", h.finalize())
            );
        }
        assert!(scope(
            None,
            Target {
                row: 0,
                directory: root.clone()
            }
        )
        .is_err());
        assert!(!root.join("descriptor.json").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn prefill_capture_rejects_invalid_provenance_before_writes() {
        for r in ["", "../escape", "a/b", "a.b", "has space"] {
            assert!(Invocation::new(0, 0, Some(r), &[1], 0).is_err());
        }
        assert!(Invocation::new(0, 0, None, &[], 0).is_err());
        assert!(Invocation::new(0, 0, None, &[1], usize::MAX).is_err());
        assert!(Invocation::new(0, 17, None, &[1], 0).is_ok());
    }
}

#[cfg(test)] #[test] fn first_prefill_observer_uses_actual_logical_position() { let inv=Invocation::new(0,3,Some("chatcmpl-test"),&[2,7],9625).unwrap();assert_eq!(inv.logical_tokens(),9627);let meta:serde_json::Value=serde_json::from_slice(&inv.metadata).unwrap();assert_eq!(meta["sequence_start"],9625);assert_eq!(meta["suffix_tokens"],2);assert_eq!(meta["logical_tokens"],9627); }
