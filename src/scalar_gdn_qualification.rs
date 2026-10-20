//! Private integration evaluation only. This is not serving/adoption admission.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Receipt {
    fn read(&self) -> Result<Vec<u8>, String> {
        if !self.path.is_absolute() {
            return Err("qualification receipt requires an absolute path".into());
        }
        let bytes = std::fs::read(&self.path)
            .map_err(|e| format!("qualification receipt {}:{e}", self.path.display()))?;
        if bytes.len() != self.bytes || format!("{:x}", Sha256::digest(&bytes)) != self.sha256 {
            return Err(format!(
                "qualification receipt identity changed:{}",
                self.path.display()
            ));
        }
        Ok(bytes)
    }
    fn passed(&self) -> Result<serde_json::Value, String> {
        let value: serde_json::Value =
            serde_json::from_slice(&self.read()?).map_err(|e| e.to_string())?;
        if value["passed"] != true {
            return Err("qualification prerequisite did not pass".into());
        }
        Ok(value)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema_version: u32,
    purpose: String,
    activation_approved: bool,
    source_files: Vec<Receipt>,
    linked_core: Receipt,
    standalone_capture: Receipt,
    actual_store_capture: Receipt,
    output_bridge: Receipt,
    native_phase: Receipt,
    correctness: Option<Receipt>,
}
fn validate(bytes: &[u8]) -> Result<(), String> {
    let plan: Plan = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if plan.schema_version != 1
        || plan.purpose != "private-native-integration-qualification"
        || plan.activation_approved
        || plan.source_files.is_empty()
    {
        return Err("native qualification plan cannot approve serving activation".into());
    }
    for file in &plan.source_files {
        file.read()?;
    }
    plan.linked_core.read()?;
    plan.standalone_capture.passed()?;
    let store = plan.actual_store_capture.passed()?;
    let rows = store["rows"]
        .as_array()
        .ok_or("actual Store receipt has no physical runs")?;
    if rows.len() < 2
        || rows.iter().any(|r| {
            r["actualStoreCases"] != 5
                || r["unexpectedSanitizerErrors"] != 0
                || r["leakedBytes"] != 0
                || r["completionCheckedBeforeAnySynchronizingRead"] != true
                || r["streamAndGraphOwnershipRestored"] != true
                || r["subsequentCaptureAndReturnedTrajectoryRecovery"] != true
        })
    {
        return Err("actual Store prerequisites are incomplete".into());
    }
    let bridge = plan.output_bridge.passed()?;
    if bridge["reductionOnlyAttributionAccepted"] != false
        || bridge["devices"].as_array().is_none_or(|v| v.len() < 2)
    {
        return Err("output proof lacks actual device operands/coefficient attribution".into());
    }
    plan.native_phase.passed()?;
    Ok(())
}
pub(super) fn requested() -> Result<bool, String> {
    let benchmark = std::env::args_os().any(|v| v == "--benchmark-only")
        && std::env::var("KRASIS_RUN_TYPE").as_deref() == Ok("dev-benchmark");
    if !cfg!(feature = "scalar-gdn-test")
        || (!benchmark && !std::env::args_os().any(|v| v == "--test-endpoints"))
    {
        return Err("private native qualification requires scalar-gdn-test and explicit test endpoints or the built benchmark; serving activation remains rejected".into());
    }
    let path = std::env::var_os("KRASIS_SCALAR_GDN_QUALIFICATION_PLAN")
        .map(PathBuf::from)
        .ok_or("private native qualification requires its frozen prerequisite plan")?;
    if !path.is_absolute() {
        return Err("native qualification plan path must be absolute".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if benchmark {
        let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let correctness = plan
            .correctness
            .as_ref()
            .ok_or("native speed qualification requires completed correctness evidence")?
            .passed()?;
        for gate in [
            "ordinaryDiagnosticsOff",
            "completeTokenF32",
            "nativeStoreBothDevices",
            "auxiliaryRecovery",
            "cancellationAndErrorRecovery",
        ] {
            if correctness[gate] != true {
                return Err(format!(
                    "native speed qualification missing correctness gate:{gate}"
                ));
            }
        }
    }
    validate(&bytes)?;
    log::warn!("Private native integration qualification only; serving activation is NOT approved");
    Ok(true)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incomplete_and_activation_approved_plans_reject_before_execution() {
        assert!(validate(b"{}").is_err());
        let receipt =
            serde_json::json!({"path":"/not-read-by-rejected-plan","bytes":0,"sha256":"unused"});
        let plan = serde_json::json!({"schema_version":1,"purpose":"private-native-integration-qualification","activation_approved":true,"source_files":[receipt.clone()],"linked_core":receipt.clone(),"standalone_capture":receipt.clone(),"actual_store_capture":receipt.clone(),"output_bridge":receipt.clone(),"native_phase":receipt});
        assert_eq!(
            validate(&serde_json::to_vec(&plan).unwrap()).unwrap_err(),
            "native qualification plan cannot approve serving activation"
        );
    }
    #[test]
    fn changed_receipt_is_rejected() {
        let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let bytes = std::fs::read(&file).unwrap();
        let exact = Receipt {
            path: file.clone(),
            bytes: bytes.len(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        assert_eq!(exact.read().unwrap(), bytes);
        assert!(Receipt {
            path: file,
            bytes: exact.bytes,
            sha256: "changed".into()
        }
        .read()
        .is_err());
    }
}

// Separate admission: never accepted as qualification evidence or by benchmarks.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticPlan {
    schema_version: u32,
    purpose: String,
    activation_approved: bool,
    store_qualification_pending: bool,
    source_files: Vec<Receipt>,
    linked_core: Receipt,
    standalone_capture: Receipt,
    output_bridge: Receipt,
    native_phase: Receipt,
}
pub(super) fn diagnostic_requested() -> Result<bool, String> {
    if !cfg!(feature = "scalar-gdn-test")
        || std::env::var("KRASIS_RUN_TYPE").as_deref() != Ok("dev-launcher-run")
        || !std::env::args_os().any(|v| v == "--test-endpoints")
        || std::env::args_os().any(|v| v == "--benchmark-only" || v == "--benchmark")
    {
        return Err(
            "private diagnostic requires launcher test endpoints; benchmarks and serving rejected"
                .into(),
        );
    }
    if std::env::var_os("KRASIS_SCALAR_GDN_QUALIFICATION_PLAN").is_some() {
        return Err("diagnostic and qualification admissions must remain separate".into());
    }
    let path = std::env::var_os("KRASIS_SCALAR_GDN_DIAGNOSTIC_PLAN")
        .map(PathBuf::from)
        .ok_or("missing private diagnostic plan")?;
    if !path.is_absolute() {
        return Err("diagnostic plan path must be absolute".into());
    }
    validate_diagnostic(&std::fs::read(path).map_err(|e| e.to_string())?)?;
    log::warn!("PRIVATE CONVERSATION DIAGNOSTIC: Store qualification PENDING; serving, benchmark and final qualification NOT approved");
    Ok(true)
}
fn validate_diagnostic(bytes: &[u8]) -> Result<(), String> {
    let plan: DiagnosticPlan = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if plan.schema_version != 1
        || plan.purpose != "private-conversation-counterexample-diagnostic"
        || plan.activation_approved
        || !plan.store_qualification_pending
        || plan.source_files.is_empty()
    {
        return Err(
            "diagnostic plan must explicitly retain Store pending and activation rejected".into(),
        );
    }
    for file in &plan.source_files {
        file.read()?;
    }
    plan.linked_core.read()?;
    plan.standalone_capture.passed()?;
    let bridge = plan.output_bridge.passed()?;
    if bridge["reductionOnlyAttributionAccepted"] != false
        || bridge["devices"].as_array().is_none_or(|v| v.len() < 2)
    {
        return Err("diagnostic output proof lacks actual operands/coefficient attribution".into());
    }
    plan.native_phase.passed()?;
    Ok(())
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn diagnostic_never_substitutes_for_qualification() {
        let r =
            serde_json::json!({"path":"/never-read-by-rejected-plan","bytes":0,"sha256":"unused"});
        let p = serde_json::json!({"schema_version":1,"purpose":"private-conversation-counterexample-diagnostic","activation_approved":false,"store_qualification_pending":true,"source_files":[r.clone()],"linked_core":r.clone(),"standalone_capture":r.clone(),"output_bridge":r.clone(),"native_phase":r});
        assert!(validate(&serde_json::to_vec(&p).unwrap()).is_err());
        for field in ["activation_approved", "store_qualification_pending"] {
            let mut bad = p.clone();
            bad[field] = serde_json::json!(field == "activation_approved");
            assert_eq!(
                validate_diagnostic(&serde_json::to_vec(&bad).unwrap()).unwrap_err(),
                "diagnostic plan must explicitly retain Store pending and activation rejected"
            );
        }
        let mut bad = p.clone();
        bad["purpose"] = serde_json::json!("private-native-integration-qualification");
        assert_eq!(
            validate_diagnostic(&serde_json::to_vec(&bad).unwrap()).unwrap_err(),
            "diagnostic plan must explicitly retain Store pending and activation rejected"
        );
        assert!(validate_diagnostic(&serde_json::to_vec(&p).unwrap())
            .unwrap_err()
            .contains("qualification receipt"));
    }
}
