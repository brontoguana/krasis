//! Selected actual prefill observation. No reference execution or GPU allocation.
use crate::{
    matched_prefill_capture::{self, Invocation},
    matched_probe::{self, Target},
};

pub(crate) struct Observer {
    target: Target,
    vocab: usize,
    hidden: usize,
    expected_hidden_ptr: u64,
}
fn geometry(
    vocab: usize,
    hidden: usize,
    rows: usize,
    max_rows: usize,
    hidden_elements: usize,
    logit_elements: usize,
    base: u64,
) -> Result<u64, String> {
    if vocab == 0
        || hidden == 0
        || rows == 0
        || rows > max_rows
        || vocab > logit_elements
        || base == 0
    {
        return Err("invalid actual first-prefill geometry".into());
    }
    let used = rows
        .checked_mul(hidden)
        .ok_or("prefill hidden extent overflow")?;
    if used > hidden_elements {
        return Err("actual prefill hidden extent exceeds loaded allocation".into());
    }
    let offset = (rows - 1)
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(2))
        .ok_or("prefill row offset overflow")?;
    let ptr = base
        .checked_add(u64::try_from(offset).map_err(|_| "prefill offset overflow")?)
        .ok_or("prefill pointer overflow")?;
    ptr.checked_add(
        u64::try_from(hidden.checked_mul(2).ok_or("prefill row bytes overflow")?)
            .map_err(|_| "prefill row bytes overflow")?,
    )
    .ok_or("prefill pointer end overflow")?;
    vocab
        .checked_mul(4)
        .ok_or("prefill logits extent overflow")?;
    Ok(ptr)
}
fn phase_bytes(logits: &[f32], vocab: usize, suppressed: bool) -> Result<Vec<u8>, String> {
    if logits.len() != vocab
        || vocab == 0
        || !logits.iter().any(|x| x.is_finite())
        || logits
            .iter()
            .any(|x| !x.is_finite() && !(suppressed && *x == f32::NEG_INFINITY))
    {
        return Err("invalid whole first-prefill distribution".into());
    }
    let mut bytes = Vec::with_capacity(
        vocab
            .checked_mul(4)
            .ok_or("prefill logits bytes overflow")?,
    );
    for value in logits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}
fn greedy_last_tie(logits: &[f32]) -> Result<u32, String> {
    phase_bytes(logits, logits.len(), true)?;
    let mut winner = 0;
    for index in 1..logits.len() {
        if logits[index] >= logits[winner] {
            winner = index;
        }
    }
    u32::try_from(winner).map_err(|_| "first-prefill winner exceeds token ID range".into())
}
pub(crate) fn begin(
    invocation: Option<&Invocation>,
    temperature: f32,
    vocab: usize,
    hidden: usize,
    rows: usize,
    max_rows: usize,
    hidden_elements: usize,
    logit_elements: usize,
    hidden_base: u64,
) -> Result<Option<Observer>, String> {
    if !matched_probe::first_prefill_logits_enabled()? {
        return Ok(None);
    }
    let invocation = invocation.ok_or("selected first-prefill capture lacks actual invocation")?;
    let Some(target) = matched_prefill_capture::target(
        Some(invocation),
        "prefill_first_logits",
        0,
        invocation.logical_tokens(),
        Some(&format!("prompt-{}", invocation.logical_tokens())),
    )?
    else {
        return Ok(None);
    };
    if !temperature.is_finite() || temperature >= 1e-6 {
        return Err("first-prefill distribution observer requires actual greedy sampling".into());
    }
    let expected_hidden_ptr = geometry(
        vocab,
        hidden,
        rows,
        max_rows,
        hidden_elements,
        logit_elements,
        hidden_base,
    )?;
    target.write("geometry.json",&serde_json::to_vec_pretty(&serde_json::json!({"vocab_size":vocab,"hidden_size":hidden,"actual_batch_rows":rows,"scratch_max_rows":max_rows,"actual_hidden_elements":hidden_elements,"actual_logit_elements":logit_elements,"final_hidden_ptr":format!("0x{expected_hidden_ptr:x}"),"temperature":temperature,"logical_tokens":invocation.logical_tokens(),"device_diagnostic_allocation_bytes":0,"diagnostic_only":true,"qualifies_candidate":false})).map_err(|e|e.to_string())?)?;
    Ok(Some(Observer {
        target,
        vocab,
        hidden,
        expected_hidden_ptr,
    }))
}
impl Observer {
    pub(crate) fn raw(
        &self,
        logits: &[f32],
        hidden_ptr: u64,
        stream: cudarc::driver::sys::CUstream,
        softcap: f32,
        lm_head_kind: &str,
    ) -> Result<(), String> {
        if hidden_ptr != self.expected_hidden_ptr || !softcap.is_finite() {
            return Err("first-prefill LM-head operand identity mismatch".into());
        }
        self.target.write(
            "raw-lm-head-logits.f32",
            &phase_bytes(logits, self.vocab, false)?,
        )?;
        self.target.device(
            "actual-final-hidden.bf16",
            hidden_ptr,
            self.hidden
                .checked_mul(2)
                .ok_or("prefill hidden bytes overflow")?,
            stream,
        )?;
        self.target.write("lm-head.json",&serde_json::to_vec_pretty(&serde_json::json!({"actual_lm_head_kind":lm_head_kind,"softcap":softcap,"raw_phase":"actual completed GPU download before softcap/suppression/sampling","hidden_phase":"actual normalized LM-head input retained after GEMM; no re-execution","all_vocab_coordinates":self.vocab})).map_err(|e|e.to_string())?)
    }
    pub(crate) fn softcap(&self, logits: &[f32]) -> Result<(), String> {
        self.target.write(
            "after-softcap-logits.f32",
            &phase_bytes(logits, self.vocab, false)?,
        )
    }
    pub(crate) fn policy(&self, logits: &[f32], suppress: &[u32]) -> Result<(), String> {
        self.target.write(
            "after-suppression-logits.f32",
            &phase_bytes(logits, self.vocab, true)?,
        )?;
        self.target.write("actual-suppressions.json",&serde_json::to_vec_pretty(&serde_json::json!({"actual_suppress_tokens":suppress,"out_of_vocab_suppressions":suppress.iter().filter(|t|**t as usize>=self.vocab).collect::<Vec<_>>(),"policy_phase":"actual first-prefill static suppressions; separate from decode EOS policy"})).map_err(|e|e.to_string())?)
    }
    pub(crate) fn selected(&self, logits: &[f32], selected: u32) -> Result<(), String> {
        let expected = greedy_last_tie(logits)?;
        if selected != expected {
            return Err(
                "actual first-prefill selection differs from retained policy distribution".into(),
            );
        }
        self.target.write("actual-selection.json",&serde_json::to_vec_pretty(&serde_json::json!({"ordinary_greedy_first":selected,"complete":true,"tie_policy":"existing prefill max_by chooses LAST equal maximum","before_teacher_first_override":true,"ordinary_conversation_equality_evidence":false})).map_err(|e|e.to_string())?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_prefill_observer_retains_raw_softcap_and_policy_values() {
        let raw = [3.0f32, 5.0, -1.0];
        let bytes = phase_bytes(&raw, 3, false).unwrap();
        assert_eq!(bytes.len(), 12);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 5.0);
        let policy = [3.0, f32::NEG_INFINITY, 3.0];
        assert!(phase_bytes(&policy, 3, false).is_err());
        assert!(phase_bytes(&policy, 3, true).is_ok());
        assert_eq!(greedy_last_tie(&policy).unwrap(), 2);
        for x in [
            &[f32::NAN, 1.0][..],
            &[f32::INFINITY, 1.0],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY],
        ] {
            assert!(phase_bytes(x, 2, true).is_err());
        }
        assert!(phase_bytes(&raw, 2, false).is_err());
        assert!(phase_bytes(&[], 0, false).is_err());
    }
    #[test]
    fn first_prefill_observer_checks_actual_row_and_allocation_extents() {
        assert_eq!(geometry(7, 5, 13, 64, 320, 7, 100).unwrap(), 220);
        assert!(geometry(7, 5, 0, 64, 320, 7, 100).is_err());
        assert!(geometry(7, 5, 13, 12, 320, 7, 100).is_err());
        assert!(geometry(7, 5, 13, 64, 64, 7, 100).is_err());
        assert!(geometry(7, 5, 13, 64, 320, 6, 100).is_err());
        assert!(geometry(7, 5, 13, 64, 320, 7, u64::MAX).is_err());
        assert!(geometry(7, usize::MAX, 2, 64, usize::MAX, 7, 100).is_err());
    }
}
