//! Explicit, checksum-bound token forcing for numerical diagnostics only.
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    ffi::OsStr,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedFile {
    path: PathBuf,
    sha256: String,
}
impl CheckedFile {
    fn verify(&self) -> Result<(), String> {
        if !self.path.is_absolute()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid teacher-forcing file identity".into());
        }
        let mut f = fs::File::open(&self.path).map_err(|e| e.to_string())?;
        let mut h = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let n = f.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            h.update(&buffer[..n]);
        }
        if format!("{:x}", h.finalize()) != self.sha256 {
            return Err(format!(
                "teacher-forcing checksum mismatch: {}",
                self.path.display()
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Trial {
    #[serde(default)]
    raw_input: bool,
    label: String,
    input_ids_sha256: String,
    logical_tokens: usize,
    expected_sequence_start: usize,
    max_tokens: usize,
    forced_inputs: Vec<u32>,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Arithmetic {
    Native,
    CenteredNativeProjection,
    RawPrefillOnly,
    RawSharedPrefill,
    HqqBf16Decode,
}
impl Arithmetic {
    fn validate_environment(
        self,
        raw: Option<&OsStr>,
        shared: Option<&OsStr>,
    ) -> Result<(), String> {
        let enabled = Some(OsStr::new("1"));
        let valid = match self {
            Self::Native | Self::CenteredNativeProjection | Self::HqqBf16Decode => raw.is_none() && shared.is_none(),
            Self::RawPrefillOnly => raw == enabled && shared.is_none(),
            Self::RawSharedPrefill => raw == enabled && shared == enabled,
        };
        if valid {
            Ok(())
        } else {
            Err("teacher-forcing arithmetic identity does not match runtime environment".into())
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    arithmetic: Arithmetic,
    schema: u32,
    directory: PathBuf,
    loaded_native: CheckedFile,
    identities: Vec<CheckedFile>,
    trials: Vec<Trial>,
    #[serde(default)]
    centered_module: Option<CheckedFile>,
}

fn token_hash(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn parse(raw: &[u8]) -> Result<Plan, String> {
    let p: Plan = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    if ![1,2,3].contains(&p.schema) || !p.directory.is_absolute() || p.identities.is_empty() || p.trials.is_empty()
    {
        return Err(
            "teacher-forcing plan requires schema1/template, schema2/raw or schema3/combined diagnostics, absolute output, identities and trials".into(),
        );
    }
    if p.schema>=2 && !matches!(p.arithmetic,Arithmetic::Native|Arithmetic::HqqBf16Decode) { return Err("raw teacher requires a single native or HQQ reconstruction arithmetic contract".into()); }
    if p.schema==3 && (!p.trials.iter().any(|t|t.raw_input)||!p.trials.iter().any(|t|!t.raw_input)) { return Err("schema3 requires both explicitly admitted template controls and raw quality trials".into()); }
    if matches!(p.arithmetic,Arithmetic::CenteredNativeProjection) != p.centered_module.is_some() || (p.centered_module.is_some() && p.schema!=1) { return Err("centered projection teacher requires schema1 and explicit center identity; older modes forbid it".into()); }
    let mut labels = HashSet::new();
    for t in &p.trials {
        if (p.schema==1 && t.raw_input) || (p.schema==2 && !t.raw_input) { return Err("teacher plan schema/input contract mismatch".into()); }
        if t.label.is_empty()
            || t.label.len() > 40
            || !t
                .label
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            || !labels.insert(&t.label)
            || t.input_ids_sha256.len() != 64
            || !t.input_ids_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || t.logical_tokens == 0
            || t.expected_sequence_start > t.logical_tokens
            || t.max_tokens < 2
            || t.logical_tokens.checked_add(t.max_tokens).is_none()
            || t.forced_inputs.is_empty()
            || t.forced_inputs.len() > t.max_tokens
        {
            return Err("invalid teacher-forcing trial/history/budget".into());
        }
    }
    Ok(p)
}

fn validate_center_environment(file:Option<&CheckedFile>,center:Option<&OsStr>,transfer:bool)->Result<(),String>{
    match (file,center,transfer) {
        (Some(file),Some(path),true) if std::path::Path::new(path)==file.path => Ok(()),
        (None,None,false)=>Ok(()),
        _=>Err("teacher center mode/environment identity mismatch".into()),
    }
}

fn plan() -> Result<&'static Plan, String> {
    static PLAN: OnceLock<Result<Plan, String>> = OnceLock::new();
    PLAN.get_or_init(|| {
        #[cfg(not(target_os = "linux"))]
        return Err(
            "matched teacher forcing currently requires Linux mapped-native identity verification"
                .into(),
        );
        #[cfg(target_os = "linux")]
        {
            let path = std::env::var_os("KRASIS_MATCHED_TEACHER_CONFIG")
                .ok_or("teacher forcing requires an explicit diagnostic plan")?;
            let p = parse(&fs::read(path).map_err(|e| e.to_string())?)?;
            let conflicts = [
                "KRASIS_GLM53_KDA_SERIAL_CONV_ORACLE",
                "KRASIS_MATCHED_MOE_LANE_CONFIG",
                "KRASIS_MATCHED_MOE_PARTITION_CONFIG",
                "KRASIS_MATCHED_REPLAY_CONFIG",
            ];
            if conflicts.iter().any(|key| std::env::var_os(key).is_some()) {
                return Err(
                    "teacher-forcing contract forbids combined arithmetic/state interventions"
                        .into(),
                );
            }
            let center=std::env::var_os("KRASIS_MATCHED_KDA_INTRA_CONFIG");
            validate_center_environment(p.centered_module.as_ref(),center.as_deref(),std::env::var_os("KRASIS_MATCHED_PROJECTION_TRANSFER_CONFIG").is_some())?;
            if let Some(file)=&p.centered_module { file.verify()?; }
            let raw = std::env::var_os("KRASIS_GLM53_KDA_RECURRENT_ORACLE");
            let shared = std::env::var_os("KRASIS_GLM53_KDA_LEGACY_RECURRENT_ORACLE");
            p.arithmetic.validate_environment(raw.as_deref(), shared.as_deref())?;
            let hqq = std::env::var_os("KRASIS_MATCHED_HQQ_ROUNDING_CONFIG");
            if matches!(p.arithmetic,Arithmetic::HqqBf16Decode) != hqq.is_some() {
                return Err("teacher HQQ reconstruction arithmetic identity mismatch".into());
            }
            p.loaded_native.verify()?;
            let maps = fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
            let native = p
                .loaded_native
                .path
                .to_str()
                .ok_or("non-UTF8 native identity")?;
            if !maps
                .lines()
                .any(|l| l.split_whitespace().last() == Some(native))
            {
                return Err("teacher-forcing native is not mapped in this server".into());
            }
            for f in &p.identities {
                f.verify()?;
            }
            Ok(p)
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn validate_raw_trial(t: &Trial, ids: &[u32], max_tokens: usize, vocab: usize, context: usize) -> Result<(), String> {
    if !t.raw_input || vocab == 0 || ids.is_empty() || ids.iter().any(|&id| id as usize >= vocab)
        || t.forced_inputs.iter().any(|&id|id as usize >= vocab)
        || ids.len().checked_add(max_tokens).map_or(true, |n|n>context) {
        return Err("raw teacher request requires admitted exact history, vocabulary and context bounds".into());
    }
    validate_inputs(t,ids,t.expected_sequence_start,max_tokens)
}
pub(crate) fn admit_raw_request(label: &str, value: &Value, max_tokens: usize, vocab: usize, context: usize) -> Result<Vec<u32>, String> {
    let ids: Vec<u32> = serde_json::from_value(value.clone()).map_err(|e|format!("raw teacher token array: {e}"))?;
    let p=plan()?;
    let t=p.trials.iter().find(|t|t.label==label).ok_or("unadmitted raw teacher label")?;
    validate_raw_trial(t,&ids,max_tokens,vocab,context)?;
    Ok(ids)
}
fn validate_first_logits(logits: &[f32]) -> Result<(), String> {
    if logits.is_empty() || logits.iter().any(|v|!v.is_finite()) { return Err("raw teacher first logits are missing or nonfinite".into()); }
    Ok(())
}
pub(crate) struct Session {
    trial: Trial,
    directory: PathBuf,
    next_step: usize,
    first_logits_len: Option<usize>,
}
impl Session {
    pub(crate) fn prepare(
        label: &str,
        ids: &[u32],
        sequence_start: usize,
        max_tokens: usize,
    ) -> Result<Self, String> {
        let p = plan()?;
        let t = p
            .trials
            .iter()
            .find(|t| t.label == label)
            .ok_or("unadmitted teacher-forcing label")?;
        validate_inputs(t, ids, sequence_start, max_tokens)?;
        fs::create_dir_all(&p.directory).map_err(|e| e.to_string())?;
        let directory = p.directory.join(label);
        fs::create_dir(&directory)
            .map_err(|e| format!("fresh teacher-forcing evidence required: {e}"))?;
        Ok(Self {
            trial: t.clone(),
            directory,
            next_step: 0,
            first_logits_len: None,
        })
    }
    pub(crate) fn raw_first(&mut self, logits: &[f32]) -> Result<(), String> {
        if !self.trial.raw_input || self.first_logits_len.is_some() { return Err("raw teacher first-logit contract/order mismatch".into()); }
        validate_first_logits(logits)?;
        persist(&self.directory.join("first-logits.f32"),&logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
        persist(&self.directory.join("first-logits.json"),&serde_json::to_vec_pretty(&json!({
            "input_ids_sha256":self.trial.input_ids_sha256,"logical_tokens":self.trial.logical_tokens,
            "sequence_start":self.trial.expected_sequence_start,"vocab_size":logits.len(),
            "input_source":"explicit_checksum_bound_raw_tokens","logits_source":"actual_prefill_engine_h_logits",
            "diagnostic_only":true,"qualifies_candidate":false
        })).map_err(|e|e.to_string())?)?;
        self.first_logits_len=Some(logits.len());Ok(())
    }
    pub(crate) fn admit_decode(
        &self,
        vocab: usize,
        stop_ids: &[usize],
        greedy_first: usize,
    ) -> Result<usize, String> {
        if self.trial.raw_input && self.first_logits_len != Some(vocab) { return Err("raw teacher first-logit vocabulary mismatch".into()); }
        if greedy_first >= vocab
            || self
                .trial
                .forced_inputs
                .iter()
                .any(|&id| id as usize >= vocab || stop_ids.contains(&(id as usize)))
        {
            return Err("teacher-forcing token is outside vocabulary or is a stop token".into());
        }
        persist(&self.directory.join("first-selection.json"),&serde_json::to_vec_pretty(&json!({
            "diagnostic_only":true,"qualifies_candidate":false,"ordinary_greedy_first":greedy_first,
            "forced_first":self.trial.forced_inputs[0],"forced_inputs":self.trial.forced_inputs,
            "logical_tokens":self.trial.logical_tokens,"sequence_start":self.trial.expected_sequence_start,
            "ordinary_cache_equality_evidence":false,
        })).map_err(|e|e.to_string())?)?;
        Ok(self.trial.forced_inputs[0] as usize)
    }
    pub(crate) fn validate_start(
        &self,
        position: usize,
        first: usize,
        budget: usize,
    ) -> Result<(), String> {
        if position != self.trial.logical_tokens
            || first != self.trial.forced_inputs[0] as usize
            || budget.checked_add(1) != Some(self.trial.max_tokens)
        {
            return Err(
                "teacher-forcing decode start does not match admitted history/budget".into(),
            );
        }
        Ok(())
    }
    pub(crate) fn raw_step(&self, step: usize, input: usize, logits: &[f32]) -> Result<(), String> {
        if step >= self.trial.max_tokens - 1
            || step != self.next_step
            || (step < self.trial.forced_inputs.len()
                && input != self.trial.forced_inputs[step] as usize)
            || logits.is_empty()
            || logits.iter().any(|x| !x.is_finite())
        {
            return Err("teacher-forcing step/input/logits admission failed".into());
        }
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        persist(
            &self.directory.join(format!("step-{step}-logits.f32")),
            &bytes,
        )
    }
    pub(crate) fn select(
        &mut self,
        step: usize,
        input: usize,
        greedy: usize,
    ) -> Result<usize, String> {
        if step >= self.trial.max_tokens - 1 || step != self.next_step {
            return Err("teacher-forcing selection order changed".into());
        }
        let selected = match self.trial.forced_inputs.get(step + 1) {
            Some(&id) => id as usize,
            None => greedy,
        };
        persist(&self.directory.join(format!("step-{step}.json")),&serde_json::to_vec_pretty(&json!({
            "diagnostic_only":true,"qualifies_candidate":false,"step":step,"input_token":input,
            "ordinary_greedy_under_forced_prefix":greedy,"actually_fed_next":selected,
            "next_token_forced":step+1<self.trial.forced_inputs.len(),"logical_tokens":self.trial.logical_tokens+step+1,
        })).map_err(|e|e.to_string())?)?;
        self.next_step += 1;
        Ok(selected)
    }
    pub(crate) fn diagnostic_label(&self) -> &str { &self.trial.label }
    pub(crate) fn metadata(&self) -> Value {
        json!({"diagnostic_only":true,"qualifies_candidate":false,"label":self.trial.label,
        "history":"explicit_teacher_forced_saved_tokens","ordinary_cache_equality_evidence":false,"forced_inputs":self.trial.forced_inputs})
    }
}
fn validate_inputs(t: &Trial, ids: &[u32], start: usize, max: usize) -> Result<(), String> {
    if ids.len() != t.logical_tokens
        || start != t.expected_sequence_start
        || max != t.max_tokens
        || token_hash(ids) != t.input_ids_sha256
    {
        return Err("teacher-forcing authoritative prompt/boundary/budget mismatch".into());
    }
    Ok(())
}
fn persist(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    f.write_all(bytes)
        .and_then(|_| f.sync_all())
        .map_err(|e| e.to_string())?;
    fs::File::open(path.parent().ok_or("missing diagnostic parent")?)
        .and_then(|p| p.sync_all())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arithmetic_modes_reject_mixed_missing_and_disabled_flags() {
        let enabled = OsStr::new("1");
        assert!(Arithmetic::Native.validate_environment(None, None).is_ok());
        assert!(Arithmetic::RawPrefillOnly.validate_environment(Some(enabled), None).is_ok());
        assert!(Arithmetic::RawSharedPrefill.validate_environment(Some(enabled), Some(enabled)).is_ok());
        assert!(Arithmetic::Native.validate_environment(Some(enabled), None).is_err());
        assert!(Arithmetic::Native.validate_environment(None, Some(enabled)).is_err());
        assert!(Arithmetic::RawPrefillOnly.validate_environment(Some(enabled), Some(enabled)).is_err());
        assert!(Arithmetic::RawSharedPrefill.validate_environment(Some(enabled), None).is_err());
        assert!(Arithmetic::RawSharedPrefill.validate_environment(None, Some(enabled)).is_err());
        for invalid in [OsStr::new("0"), OsStr::new(""), OsStr::new("true")] {
            assert!(Arithmetic::Native.validate_environment(Some(invalid), None).is_err());
            assert!(Arithmetic::RawPrefillOnly.validate_environment(Some(invalid), None).is_err());
            assert!(Arithmetic::RawSharedPrefill.validate_environment(Some(enabled), Some(invalid)).is_err());
        }
    }
    #[cfg(unix)]
    #[test]
    fn non_unicode_arithmetic_flags_cannot_hide_an_intervention() {
        use std::os::unix::ffi::OsStrExt;
        let invalid = OsStr::from_bytes(&[0xff]);
        assert!(Arithmetic::Native.validate_environment(None, Some(invalid)).is_err());
        assert!(Arithmetic::RawPrefillOnly.validate_environment(Some(OsStr::new("1")), Some(invalid)).is_err());
        assert!(Arithmetic::RawSharedPrefill.validate_environment(Some(invalid), Some(OsStr::new("1"))).is_err());
    }
    #[test]
    fn raw_input_cannot_substitute_template_history_or_exceed_loaded_bounds() {
        let mut t=trial();
        assert!(validate_raw_trial(&t,&[3,7],3,10,10).is_err());t.raw_input=true;
        assert!(validate_raw_trial(&t,&[3,7],3,10,10).is_ok());
        for (ids,max,vocab,context) in [(vec![7,3],3,10,10),(vec![3,7],4,10,10),(vec![3,7],3,7,10),(vec![3,7],3,10,4),(vec![],3,10,10)] {
            assert!(validate_raw_trial(&t,&ids,max,vocab,context).is_err());
        }
        assert!(validate_first_logits(&[]).is_err());assert!(validate_first_logits(&[f32::NAN]).is_err());assert!(validate_first_logits(&[f32::INFINITY]).is_err());assert!(validate_first_logits(&[-1.,0.,1.]).is_ok());
    }
    #[test]
    fn combined_plan_preserves_template_controls_and_admits_only_declared_raw_trials() {
        let base=serde_json::json!({"schema":3,"arithmetic":"native","directory":"/tmp/quality","loaded_native":{"path":"/tmp/native","sha256":"0".repeat(64)},"identities":[{"path":"/tmp/source","sha256":"0".repeat(64)}],"trials":[
            {"label":"control","input_ids_sha256":"0".repeat(64),"logical_tokens":2,"expected_sequence_start":0,"max_tokens":3,"forced_inputs":[4,5]},
            {"label":"quality","raw_input":true,"input_ids_sha256":"0".repeat(64),"logical_tokens":2,"expected_sequence_start":0,"max_tokens":3,"forced_inputs":[4,5]}
        ]});
        assert!(parse(&serde_json::to_vec(&base).unwrap()).is_ok());
        for schema in [1,2,4]{let mut p=base.clone();p["schema"]=serde_json::json!(schema);assert!(parse(&serde_json::to_vec(&p).unwrap()).is_err());}
        for raw in [false,true]{let mut p=base.clone();for t in p["trials"].as_array_mut().unwrap(){t["raw_input"]=serde_json::json!(raw);}assert!(parse(&serde_json::to_vec(&p).unwrap()).is_err());}
        let mut p=base.clone();p["arithmetic"]=serde_json::json!("raw_prefill_only");assert!(parse(&serde_json::to_vec(&p).unwrap()).is_err());
    }
    #[test]
    fn centered_teacher_preserves_old_mode_and_environment_contracts() {
        let mut p=json!({"schema":1,"arithmetic":"centered_native_projection","directory":"/tmp/conditional","loaded_native":{"path":"/tmp/native","sha256":"a".repeat(64)},"identities":[{"path":"/tmp/source","sha256":"b".repeat(64)}],"trials":[{"label":"seed","logical_tokens":2,"expected_sequence_start":0,"max_tokens":3,"input_ids_sha256":"c".repeat(64),"forced_inputs":[4,5]}]});
        assert!(parse(&serde_json::to_vec(&p).unwrap()).is_err());
        p["centered_module"]=json!({"path":"/tmp/center","sha256":"d".repeat(64)});
        assert!(parse(&serde_json::to_vec(&p).unwrap()).is_ok());
        for mode in ["native","raw_prefill_only","raw_shared_prefill","hqq_bf16_decode"] {let mut q=p.clone();q["arithmetic"]=json!(mode);assert!(parse(&serde_json::to_vec(&q).unwrap()).is_err());}
        for schema in [2,3]{let mut q=p.clone();q["schema"]=json!(schema);assert!(parse(&serde_json::to_vec(&q).unwrap()).is_err());}
        let f=CheckedFile{path:PathBuf::from("/tmp/center"),sha256:"d".repeat(64)};
        assert!(validate_center_environment(Some(&f),Some(OsStr::new("/tmp/center")),true).is_ok());
        assert!(validate_center_environment(None,None,false).is_ok());
        for (file,path,transfer) in [(Some(&f),None,true),(Some(&f),Some(OsStr::new("/tmp/other")),true),(Some(&f),Some(OsStr::new("/tmp/center")),false),(None,None,true),(None,Some(OsStr::new("/tmp/center")),false)] {assert!(validate_center_environment(file,path,transfer).is_err());}
        assert!(Arithmetic::CenteredNativeProjection.validate_environment(None,None).is_ok());
        assert!(Arithmetic::CenteredNativeProjection.validate_environment(Some(OsStr::new("1")),None).is_err());
    }
    fn trial() -> Trial {
        Trial {
            raw_input: false,
            label: "control".into(),
            input_ids_sha256: token_hash(&[3, 7]),
            logical_tokens: 2,
            expected_sequence_start: 1,
            max_tokens: 3,
            forced_inputs: vec![4, 5],
        }
    }
    #[test]
    fn exact_authoritative_inputs_boundary_and_budget_are_required() {
        let t = trial();
        assert!(validate_inputs(&t, &[3, 7], 1, 3).is_ok());
        for (ids, start, max) in [(vec![7, 3], 1, 3), (vec![3, 7], 0, 3), (vec![3, 7], 1, 4)] {
            assert!(validate_inputs(&t, &ids, start, max).is_err());
        }
    }
    #[test]
    fn malformed_history_and_duplicate_labels_are_rejected() {
        let t = trial();
        let base = json!({"arithmetic":"native","schema":1,"directory":std::env::temp_dir().join("teacher-test"),"loaded_native":{"path":std::env::temp_dir().join("native"),"sha256":"a".repeat(64)},"identities":[{"path":std::env::temp_dir().join("config"),"sha256":"b".repeat(64)}],"trials":[{"label":t.label,"input_ids_sha256":t.input_ids_sha256,"logical_tokens":2,"expected_sequence_start":1,"max_tokens":3,"forced_inputs":[4,5]}]});
        assert!(parse(&serde_json::to_vec(&base).unwrap()).is_ok());
        for change in [
            json!({"forced_inputs":[]}),
            json!({"forced_inputs":[1,2,3,4]}),
            json!({"expected_sequence_start":3}),
            json!({"label":"../escape"}),
        ] {
            let mut p = base.clone();
            for (k, v) in change.as_object().unwrap() {
                p["trials"][0][k] = v.clone();
            }
            assert!(parse(&serde_json::to_vec(&p).unwrap()).is_err());
        }
        let mut p = base.clone();
        p["trials"]
            .as_array_mut()
            .unwrap()
            .push(base["trials"][0].clone());
        assert!(parse(&serde_json::to_vec(&p).unwrap()).is_err());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn forced_inputs_and_greedy_choices_are_separate_and_evidence_is_immutable() {
        let dir = std::env::temp_dir().join(format!(
            "teacher-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let mut s = Session {
            trial: trial(),
            directory: dir.clone(),
            next_step: 0,
            first_logits_len: None,
        };
        assert!(s.admit_decode(5, &[], 4).is_err());
        assert!(s.admit_decode(10, &[5], 4).is_err());
        assert_eq!(s.admit_decode(10, &[], 6).unwrap(), 4);
        assert!(s.validate_start(2, 4, 2).is_ok());
        assert!(s.validate_start(2, 6, 2).is_err());
        assert!(s.raw_step(0, 6, &[1.0, 2.0]).is_err());
        s.raw_step(0, 4, &[1.0, 2.0]).unwrap();
        assert!(s.raw_step(0, 4, &[1.0, 2.0]).is_err());
        assert_eq!(s.select(0, 4, 8).unwrap(), 5);
        let proof: Value =
            serde_json::from_slice(&fs::read(dir.join("step-0.json")).unwrap()).unwrap();
        assert_eq!(proof["ordinary_greedy_under_forced_prefix"], 8);
        assert_eq!(proof["actually_fed_next"], 5);
        s.raw_step(1, 5, &[2.0, 1.0]).unwrap();
        assert_eq!(s.select(1, 5, 7).unwrap(), 7);
        assert!(s.raw_step(2, 7, &[f32::NAN]).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
