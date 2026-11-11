//! Control-plane selection; no parsing or atomic flag reads in model loops.
use crate::optimisations::{Selection, ENTRIES};
use sha2::{Digest, Sha256};

// Load-time layouts remain frozen in this bounded harness.
pub(crate) const SWITCHABLE: &[&str] = &[
    "TENSOR_DECODE_ATTENTION", "TENSOR_ABSORPTION", "VALUE_REASSOCIATION",
    "STREAMED_COLD_NF3", "STREAMED_COLD_PROMOTIONS", "CPU_INT8_K_MAJOR", "SHARED_EXPERT_COPY_OVERLAP", "COLD_EXPERT_PRUNING_FIVE",
];

pub(crate) fn manifest(selection: Selection) -> serde_json::Value {
    serde_json::Value::Array(selection.values().into_iter().map(|(name, enabled)| {
        let e = ENTRIES.iter().find(|e| e.0 == name).unwrap();
        serde_json::json!({"name":name,"enabled":enabled,"default":e.1,"kind":e.2,"description":e.3})
    }).collect())
}
pub(crate) fn hash(selection: Selection) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(&manifest(selection)).unwrap()))
}
pub(crate) fn parse(body: &str) -> Result<Selection, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e|e.to_string())?;
    let mut next = Selection::compiled();
    let mut seen = std::collections::HashSet::new();
    for (key, enabled) in [("enable", true), ("disable", false)] {
        if let Some(values) = v.get(key) {
            for value in values.as_array().ok_or("enable/disable must be arrays")? {
                let name = value.as_str().ok_or("switch name must be a string")?;
                if !seen.insert(name) { return Err(format!("duplicate/conflicting switch: {name}")); }
                next.set(name, enabled)?;
                if !SWITCHABLE.contains(&name) { return Err(format!("{name} requires a separate load/build")); }
            }
        }
    }
    let values: std::collections::HashMap<_,_> = next.values().into_iter().collect();
    for &(name, _, _, _, requires) in ENTRIES {
        if values[name] { for dependency in requires {
            if !values[dependency] { return Err(format!("{name} requires {dependency}")); }
        }}
    }
    if values.get("STREAMED_COLD_NF3") == Some(&true) && values.get("STREAMED_COLD_INT3") == Some(&true) {
        return Err("choose one streamed cold quantizer".into());
    }
    Ok(next)
}

thread_local! { static ACTIVE: std::cell::Cell<Selection> = const {std::cell::Cell::new(Selection::compiled())}; }
pub(crate) fn active_hash()->String { ACTIVE.with(|s|hash(s.get())) }
pub(crate) fn set_active(selection:Selection) { ACTIVE.with(|s|s.set(selection)); }
