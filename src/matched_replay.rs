//! Opt-in, checksum-bound causal state replay for internal reference requests.
//! Does not change ordinary request arithmetic or session-cache admission.
use crate::session_cache::{SequenceStateAllocation, SequenceStateBlob};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::Read,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

static PASSED_CONTROLS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedFile {
    path: PathBuf,
    sha256: String,
}

impl CheckedFile {
    fn verify(&self) -> Result<(), String> {
        if !self.path.is_absolute() || self.sha256.len() != 64 {
            return Err("invalid replay file identity".into());
        }
        let mut file = std::fs::File::open(&self.path).map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        if format!("{:x}", hash.finalize()) != self.sha256 {
            return Err(format!(
                "replay file checksum mismatch: {}",
                self.path.display()
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateFile {
    allocation: SequenceStateAllocation,
    used_bytes: usize,
    file: CheckedFile,
}

impl StateFile {
    fn read_blob(&self, position: usize) -> Result<SequenceStateBlob, String> {
        self.allocation.validate()?;
        if !self.file.path.is_absolute() {
            return Err("replay state requires an absolute path".into());
        }
        if self.allocation.used_bytes(position) != self.used_bytes {
            return Err("replay state metadata/length mismatch".into());
        }
        let bytes = std::fs::read(&self.file.path).map_err(|e| e.to_string())?;
        if bytes.len() != self.used_bytes
            || format!("{:x}", Sha256::digest(&bytes)) != self.file.sha256
        {
            return Err(format!(
                "replay state checksum mismatch: {}",
                self.allocation.name
            ));
        }
        let shape = self.allocation.snapshot_shape(bytes.len());
        let blob = SequenceStateBlob {
            allocation_name: self.allocation.name.clone(),
            kind: self.allocation.kind.clone(),
            layer_idx: self.allocation.layer_idx,
            device_ordinal: self.allocation.device_ordinal,
            dtype: self.allocation.dtype.clone(),
            element_size: self.allocation.element_size,
            shape,
            strides_bytes: self.allocation.strides_bytes.clone(),
            bytes,
        };
        blob.validate()?;
        Ok(blob)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Trial {
    label: String,
    states: Vec<StateFile>,
    /// Whole-state controls must match every original ordinary step exactly.
    expected_logits_sha256: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    directory: PathBuf,
    input_ids_sha256: String,
    logical_tokens: usize,
    rope_position_delta: i32,
    forced_inputs: Vec<u32>,
    loaded_native: CheckedFile,
    identities: Vec<CheckedFile>,
    required_controls: Vec<String>,
    trials: Vec<Trial>,
}

fn config() -> Result<Option<&'static Plan>, String> {
    static PLAN: OnceLock<Result<Option<Plan>, String>> = OnceLock::new();
    PLAN.get_or_init(|| {
        let path = match std::env::var("KRASIS_MATCHED_REPLAY_CONFIG") {
            Ok(path) => path,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        let plan: Plan = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if !plan.directory.is_absolute() || plan.forced_inputs.is_empty() || plan.trials.is_empty()
        {
            return Err("replay requires an absolute output directory, inputs and trials".into());
        }
        let mut labels = HashSet::new();
        for trial in &plan.trials {
            if trial.label.is_empty()
                || trial.label.len() + "matched_replay_".len() > 48
                || !trial
                    .label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                || !labels.insert(&trial.label)
                || trial.states.is_empty()
                || trial
                    .expected_logits_sha256
                    .as_ref()
                    .is_some_and(|v| v.len() != plan.forced_inputs.len())
            {
                return Err("invalid/duplicate replay trial or control length".into());
            }
        }
        if plan.required_controls.len() < 2
            || plan.required_controls.iter().collect::<HashSet<_>>().len()
                != plan.required_controls.len()
            || plan.required_controls.iter().any(|label| {
                !plan
                    .trials
                    .iter()
                    .any(|t| &t.label == label && t.expected_logits_sha256.is_some())
            })
        {
            return Err(
                "replay interventions require distinct admitted whole-state controls".into(),
            );
        }
        plan.loaded_native.verify()?;
        // This diagnostic is Linux-only and must identify the actually mapped
        // extension, rather than merely a same-named file supplied by the client.
        let maps = std::fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
        let native = plan
            .loaded_native
            .path
            .to_str()
            .ok_or("non-UTF8 replay native path")?;
        if !maps
            .lines()
            .any(|line| line.split_whitespace().last() == Some(native))
        {
            return Err("replay native identity is not mapped in this server".into());
        }
        if plan.identities.is_empty() {
            return Err("replay requires loaded bank/config/source identities".into());
        }
        for identity in &plan.identities {
            identity.verify()?;
        }
        Ok(Some(plan))
    })
    .as_ref()
    .map(|p| p.as_ref())
    .map_err(Clone::clone)
}

fn trial<'a>(plan: &'a Plan, label: &str) -> Result<&'a Trial, String> {
    plan.trials
        .iter()
        .find(|t| t.label == label)
        .ok_or_else(|| format!("unadmitted replay label {label}"))
}

pub(crate) fn validate_request(
    label: &str,
    input_ids: &[u32],
    max_tokens: usize,
) -> Result<usize, String> {
    let plan = config()?.ok_or("matched replay is not configured")?;
    let trial = trial(plan, label)?;
    if trial.expected_logits_sha256.is_none() {
        let passed = PASSED_CONTROLS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .map_err(|e| e.to_string())?;
        if plan
            .required_controls
            .iter()
            .any(|label| !passed.contains(label))
        {
            return Err("whole-state controls must pass before any intervention".into());
        }
    }
    let mut hash = Sha256::new();
    for id in input_ids {
        hash.update(id.to_le_bytes());
    }
    if input_ids.len() != plan.logical_tokens
        || format!("{:x}", hash.finalize()) != plan.input_ids_sha256
        || max_tokens != plan.forced_inputs.len() + 1
    {
        return Err("replay prompt IDs or generation length do not match the frozen plan".into());
    }
    Ok(plan.forced_inputs[0] as usize)
}

pub(crate) struct Replay {
    pub forced_inputs: Vec<u32>,
    directory: PathBuf,
    expected: Option<Vec<String>>,
    label: String,
}

pub(crate) fn load(
    label: &str,
    position: usize,
    first: usize,
    max_tokens: usize,
) -> Result<Option<(Replay, Vec<SequenceStateBlob>, i32)>, String> {
    // Ordinary requests must remain ordinary, including when an experiment is configured.
    let Some(label) = label.strip_prefix("matched_replay_") else {
        return Ok(None);
    };
    let plan = config()?.ok_or("explicit matched replay is not configured")?;
    let trial = trial(plan, label)?;
    if position != plan.logical_tokens
        || first != plan.forced_inputs[0] as usize
        || max_tokens != plan.forced_inputs.len()
    {
        return Err("replay generation inputs differ from admitted reference request".into());
    }
    let mut blobs = Vec::with_capacity(trial.states.len());
    let mut names = HashSet::new();
    for state in &trial.states {
        if !names.insert(&state.allocation.name) {
            return Err("duplicate replay state name".into());
        }
        let blob = state.read_blob(position)?;
        blobs.push(blob);
    }
    Ok(Some((
        Replay {
            forced_inputs: plan.forced_inputs.clone(),
            directory: plan.directory.join(label),
            expected: trial.expected_logits_sha256.clone(),
            label: label.into(),
        },
        blobs,
        plan.rope_position_delta,
    )))
}

impl Replay {
    pub fn selection(&self, step: usize, greedy: usize) -> Result<(), String> {
        let target = crate::matched_probe::Target {
            row: 0,
            directory: self.directory.clone(),
        };
        target.write(&format!("step-{step}-selection.json"), serde_json::to_vec_pretty(&serde_json::json!({"ordinary_greedy_token":greedy,"next_forced_input":self.forced_inputs.get(step+1)})).map_err(|e|e.to_string())?.as_slice())
    }
    pub fn step(&self, step: usize, input: usize, logits: &[f32]) -> Result<(), String> {
        if self.forced_inputs.get(step).copied().map(|x| x as usize) != Some(input) {
            return Err("replay did not feed the admitted common input".into());
        }
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            return Err("empty/nonfinite replay logits; attribution prohibited".into());
        }
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let target = crate::matched_probe::Target {
            row: 0,
            directory: self.directory.clone(),
        };
        target.write(&format!("step-{step}-logits.f32.bin"), &bytes)?;
        target.write(
            &format!("step-{step}.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "step":step,"input_token":input,
                "next_forced_input":self.forced_inputs.get(step+1),"logits_sha256":hash,
                "whole_state_control":self.expected.is_some(),
                "control_matches":self.expected.as_ref().map(|v|v[step]==hash),
                "diagnostic_only":true,"qualifies_candidate":false,
            }))
            .map_err(|e| e.to_string())?
            .as_slice(),
        )?;
        if self.expected.as_ref().is_some_and(|v| v[step] != hash) {
            return Err(format!(
                "whole-state replay control failed at step {step}; attribution prohibited"
            ));
        }
        if self.expected.is_some() && step + 1 == self.forced_inputs.len() {
            PASSED_CONTROLS
                .get_or_init(|| Mutex::new(HashSet::new()))
                .lock()
                .map_err(|e| e.to_string())?
                .insert(self.label.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_decode_never_resolves_diagnostic_config() {
        // Isolate environment and OnceLock state from other concurrent tests.
        if std::env::var_os("KRASIS_REPLAY_GUARD_TEST_CHILD").is_some() {
            assert!(load("ordinary_calibration", 12, 17, 5).unwrap().is_none());
            assert!(load("matched_replay_explicit", 12, 17, 5).is_err());
            return;
        }
        let missing =
            std::env::temp_dir().join(format!("missing-replay-plan-{}.json", std::process::id()));
        assert!(!missing.exists());
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "matched_replay::tests::ordinary_decode_never_resolves_diagnostic_config",
                "--nocapture",
            ])
            .env("KRASIS_REPLAY_GUARD_TEST_CHILD", "1")
            .env("KRASIS_MATCHED_REPLAY_CONFIG", missing)
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[test]
    fn rejects_mismatched_file_bytes_and_records_control_failure() {
        let dir = std::env::temp_dir().join(format!("matched-replay-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("identity");
        std::fs::write(&file, b"same bytes").unwrap();
        assert!(CheckedFile {
            path: file,
            sha256: "0".repeat(64)
        }
        .verify()
        .is_err());
        let replay = Replay {
            forced_inputs: vec![17],
            directory: dir.join("control"),
            expected: Some(vec!["0".repeat(64)]),
            label: "control".into(),
        };
        assert!(replay
            .step(0, 17, &[1.0, 2.0])
            .unwrap_err()
            .contains("attribution prohibited"));
        assert!(dir.join("control/step-0.json").exists());
        assert!(replay.step(0, 18, &[1.0, 2.0]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn replay_pooled_live_prefix_matches_snapshot_shape_and_rejects_invalid_bytes() {
        use crate::session_cache::SequenceStateGrowth;
        let dir = std::env::temp_dir().join(format!("replay-growth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (index, growth, rows) in [
            (0, SequenceStateGrowth::Fixed, 8),
            (
                1,
                SequenceStateGrowth::TokenRows {
                    logical_tokens_per_row: 4,
                    capacity_rows: 8,
                    row_bytes: 8,
                },
                3,
            ),
            (
                2,
                SequenceStateGrowth::CompletedTokenRows {
                    logical_tokens_per_row: 4,
                    capacity_rows: 8,
                    row_bytes: 8,
                },
                2,
            ),
        ] {
            let bytes = vec![0u8; rows * 8];
            let path = dir.join(format!("state-{index}"));
            std::fs::write(&path, &bytes).unwrap();
            let state = StateFile {
                allocation: SequenceStateAllocation {
                    name: format!("cache{index}"),
                    kind: "actual_growth_fixture".into(),
                    layer_idx: Some(3),
                    device_ordinal: 0,
                    ptr: 1,
                    storage_bytes: 64,
                    dtype: "bfloat16".into(),
                    element_size: 2,
                    shape: vec![8, 4],
                    strides_bytes: vec![8, 2],
                    growth,
                },
                used_bytes: bytes.len(),
                file: CheckedFile {
                    path: path.clone(),
                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                },
            };
            let blob = state.read_blob(10).unwrap();
            assert_eq!(blob.shape, vec![rows, 4]);
            assert_eq!(blob.bytes, bytes);
            blob.validate().unwrap();
            std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
            assert!(state.read_blob(10).is_err());
            let mut extra = bytes.clone();
            extra.push(0);
            std::fs::write(&path, extra).unwrap();
            assert!(state.read_blob(10).is_err());
            std::fs::write(&path, &bytes).unwrap();
            assert!(state.read_blob(14).is_ok() == (index == 0));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
