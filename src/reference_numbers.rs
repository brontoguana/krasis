//! Lossless source-F32 numerical observations. Never substitute a missing value.
use serde_json::{json, Value};

pub(crate) struct CapturedLogprobs {
    pub(crate) selected: Option<f64>,
    pub(crate) selected_bits: Option<u32>,
    pub(crate) top_k: Vec<Value>,
    pub(crate) warning: Option<&'static str>,
}

fn top_entries(top: &[(u32, f32)]) -> Result<Vec<Value>, String> {
    top.iter()
        .map(|&(id, value)| {
            if !value.is_finite() {
                return Err(format!(
                    "non-finite captured log probability for token {id}"
                ));
            }
            Ok(json!({"token_id":id,"log_prob":value as f64,"log_prob_bits":value.to_bits()}))
        })
        .collect()
}

pub(crate) fn capture(token: usize, top: &[(u32, f32)]) -> Result<CapturedLogprobs, String> {
    let top_k = top_entries(top)?;
    let selected = top
        .iter()
        .find(|&&(id, _)| id as usize == token)
        .map(|&(_, value)| value);
    Ok(CapturedLogprobs {
        selected: selected.map(|value|value as f64),
        selected_bits: selected.map(f32::to_bits),
        top_k,
        warning: selected.is_none().then_some("WARNING: selected-token log probability was not captured; no numerical value is available"),
    })
}

pub(crate) fn reference_rows(
    output: &[(usize, Vec<(u32, f32)>)],
    first: &[(u32, f32)],
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut rows = Vec::with_capacity(output.len());
    for (token, top) in output {
        let value = capture(*token, top)?;
        rows.push(
            json!({
                "token_id": token,
                "log_prob": value.selected,
                "log_prob_bits": value.selected_bits,
                "log_prob_dtype": "f32",
                "log_prob_captured": value.selected.is_some(),
                "warning": value.warning,
                "top_k": value.top_k,
            })
            .to_string(),
        );
    }
    let first = top_entries(first)?
        .into_iter()
        .map(|v| v.to_string())
        .collect();
    Ok((rows, first))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn roundtrip(value: f32) {
        let (rows, first) = reference_rows(&[(40, vec![(40, value)])], &[(40, value)]).unwrap();
        let row: Value = serde_json::from_str(&rows[0]).unwrap();
        let first: Value = serde_json::from_str(&first[0]).unwrap();
        for entry in [&row, &row["top_k"][0], &first] {
            assert_eq!(
                (entry["log_prob"].as_f64().unwrap() as f32).to_bits(),
                value.to_bits()
            );
            assert_eq!(entry["log_prob_bits"], value.to_bits());
        }
        assert_eq!(row["log_prob_captured"], true);
        assert!(row["warning"].is_null());
    }
    #[test]
    fn measured_logprob_survives_where_six_decimals_lose_bits() {
        let value = -0.0005028694868087769_f32;
        let old: f32 = format!("{value:.6}").parse().unwrap();
        assert_ne!(old.to_bits(), value.to_bits());
        roundtrip(value);
        roundtrip(-0.2556321532432122_f64 as f32);
    }
    #[test]
    fn finite_edges_and_signed_zero_keep_source_bits() {
        for value in [
            0.0,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
        ] {
            roundtrip(value);
        }
    }
    #[test]
    fn missing_selected_probability_is_null_and_explicit() {
        for top in [vec![], vec![(4, -1.0)]] {
            let (rows, _) = reference_rows(&[(40, top)], &[]).unwrap();
            let row: Value = serde_json::from_str(&rows[0]).unwrap();
            assert!(row["log_prob"].is_null());
            assert!(row["log_prob_bits"].is_null());
            assert_eq!(row["log_prob_captured"], false);
            assert!(row["warning"].as_str().unwrap().starts_with("WARNING:"));
        }
    }
    #[test]
    fn selected_identity_and_top_order_are_preserved() {
        let top = vec![(4, -0.25), (40, -2.0)];
        let capture = capture(40, &top).unwrap();
        assert_eq!(capture.selected, Some(-2.0));
        assert_eq!(capture.top_k[0]["token_id"], 4);
        assert_eq!(capture.top_k[1]["token_id"], 40);
        assert_eq!(top, vec![(4, -0.25), (40, -2.0)]);
    }
    #[test]
    fn nonfinite_capture_is_an_error() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(capture(4, &[(4, value)]).is_err());
            assert!(reference_rows(&[(4, vec![(4, value)])], &[]).is_err());
        }
    }
}
