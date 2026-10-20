use cudarc::driver::sys as cuda;
use serde::Deserialize;
use std::{path::PathBuf, sync::OnceLock};

// BEGIN bounded measured HQQ dispatch diagnostic
#[derive(Clone, Copy, Deserialize, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
enum DiagnosticHqqFamily { Scalar, Vec4 }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HqqDispatchControls {
    selections: Vec<HqqDispatchSelection>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HqqDispatchSelection {
    rows: usize,
    cols: usize,
    group_size: usize,
    family: DiagnosticHqqFamily,
    warps: usize,
}

impl HqqDispatchControls {
    fn parse(raw: &str) -> Result<Self, String> {
        let parsed: Self = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if parsed.selections.is_empty() { return Err("empty HQQ diagnostic selections".into()); }
        let mut unique = std::collections::HashSet::new();
        for s in &parsed.selections {
            if s.rows == 0 || s.cols == 0 || s.group_size == 0 || !unique.insert((s.rows, s.cols, s.group_size)) {
                return Err("invalid/duplicate HQQ diagnostic shape".into());
            }
            match s.family {
                DiagnosticHqqFamily::Scalar if s.warps == 0 => {},
                DiagnosticHqqFamily::Vec4 if s.warps > 0 && s.cols % 4 == 0 && s.group_size % 4 == 0 => {},
                _ => return Err("invalid HQQ diagnostic family/geometry".into()),
            }
        }
        Ok(parsed)
    }

    pub(crate) fn load(device: &std::sync::Arc<cudarc::driver::CudaDevice>, experiment: Option<&crate::matched_hqq_rounding::Experiment>) -> Result<Option<Self>, String> {
        let file = match std::env::var("KRASIS_MATCHED_HQQ_DISPATCH_CONFIG") {
            Ok(file) => file,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        if let Some(context) = experiment { context.authorize_dispatch(device, std::path::Path::new(&file))?; }
        match std::env::var("KRASIS_MATCHED_PROBE_CONFIG") {
            Ok(probe) => { parse(&std::fs::read_to_string(probe).map_err(|e| e.to_string())?)?; },
            Err(std::env::VarError::NotPresent) if experiment.is_some() => {},
            Err(std::env::VarError::NotPresent) => return Err("HQQ dispatch requires a verified context-owned experiment or explicit matched-probe diagnostics".into()),
            Err(e) => return Err(e.to_string()),
        }
        Self::parse(&std::fs::read_to_string(file).map_err(|e| e.to_string())?).map(Some)
    }

    pub(crate) fn require_shapes(&self, available: &[(usize, usize, usize)]) -> Result<(), String> {
        for s in &self.selections {
            if available.iter().filter(|&&key| key == (s.rows, s.cols, s.group_size)).count() != 1 {
                return Err(format!("HQQ diagnostic shape missing/ambiguous: {}x{} group {}",s.rows,s.cols,s.group_size));
            }
        }
        Ok(())
    }

    // None means this loaded shape was not requested. Some(None) explicitly
    // requests the measured scalar kernel; Some(Some(w)) requests measured vec4.
    pub(crate) fn measured_selection(&self, shape: (usize, usize, usize), scalar_ms: f64,
                                    measured: &[(usize, f64)], force_vec4: bool) -> Result<Option<Option<usize>>, String> {
        let Some(s) = self.selections.iter().find(|s| (s.rows,s.cols,s.group_size) == shape) else { return Ok(None); };
        if !scalar_ms.is_finite() || scalar_ms <= 0. { return Err("HQQ scalar diagnostic timing is invalid".into()); }
        match s.family {
            DiagnosticHqqFamily::Scalar => {
                if force_vec4 { return Err("HQQ diagnostic scalar violates required quartet policy".into()); }
                Ok(Some(None))
            },
            DiagnosticHqqFamily::Vec4 => {
                if !measured.iter().any(|&(w,ms)| w == s.warps && ms.is_finite() && ms > 0.) {
                    return Err(format!("HQQ diagnostic geometry {} was not measured for {}x{}",s.warps,s.rows,s.cols));
                }
                Ok(Some(Some(s.warps)))
            },
        }
    }
}

#[cfg(test)]
#[test]
fn test_hqq_diagnostic_requires_loaded_shapes_and_real_measurements() {
    let c = HqqDispatchControls::parse(r#"{"selections":[{"rows":8,"cols":16,"group_size":8,"family":"vec4","warps":2}]}"#).unwrap();
    assert!(c.require_shapes(&[]).is_err());
    assert!(c.require_shapes(&[(8,16,8),(8,16,8)]).is_err());
    c.require_shapes(&[(8,16,8)]).unwrap();
    assert!(c.measured_selection((8,16,8),1.,&[(1,0.5)],false).is_err());
    assert!(c.measured_selection((8,16,8),1.,&[(2,f64::NAN)],false).is_err());
    assert_eq!(c.measured_selection((8,16,8),1.,&[(2,0.5)],false).unwrap(),Some(Some(2)));
    assert_eq!(c.measured_selection((9,16,8),1.,&[],false).unwrap(),None);
    let scalar = HqqDispatchControls::parse(r#"{"selections":[{"rows":8,"cols":16,"group_size":8,"family":"scalar","warps":0}]}"#).unwrap();
    assert!(scalar.measured_selection((8,16,8),1.,&[],true).is_err());
    assert_eq!(scalar.measured_selection((8,16,8),1.,&[],false).unwrap(),Some(None));
    for invalid in [r#"{"selections":[]}"#,r#"{"selections":[{"rows":8,"cols":16,"group_size":8,"family":"vec4","warps":0}]}"#,
                    r#"{"selections":[{"rows":8,"cols":16,"group_size":8,"family":"scalar","warps":1}]}"#,
                    r#"{"selections":[{"rows":8,"cols":15,"group_size":8,"family":"vec4","warps":2}]}"#] {
        assert!(HqqDispatchControls::parse(invalid).is_err());
    }
}
// END bounded measured HQQ dispatch diagnostic

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    directory: PathBuf,
    targets: Vec<TargetConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    operation: String,
    layer: usize,
    row: usize,
    heights: Vec<usize>,
    #[serde(default)]
    request_labels: Vec<String>,
    window_rows: Option<usize>,
    window_max_bytes: Option<usize>,
    maximum_bytes: Option<usize>,
}

pub(crate) struct Target {
    pub row: usize,
    pub directory: PathBuf,
}

// BEGIN bounded model KDA probe admission
pub(crate) fn require_model_diagnostic_probe() -> Result<(), String> {
    let path = std::env::var("KRASIS_MATCHED_PROBE_CONFIG").map_err(|_| "KDA model intervention requires explicit matched-probe diagnostics".to_string())?;
    parse(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)?;
    Ok(())
}
// END bounded model KDA probe admission

fn parse(raw: &str) -> Result<Config, String> {
    let config: Config = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    if !config.directory.is_absolute() || config.targets.is_empty() {
        return Err("matched probe requires an absolute directory and targets".into());
    }
    for t in &config.targets {
        let validation_operation = match t.operation.strip_suffix("_sha256") {
            Some(base) if base.starts_with("glm5_window_") || base=="selected_prepared_window" => base,
            Some(_) => return Err("SHA256 observation requires a supported prefill window".into()),
            None => t.operation.as_str(),
        };
        if t.operation=="selected_prepared_window_sha256" || t.operation=="mla_common_stages" || t.operation=="indexer_common_stages" || t.operation.starts_with("decode_graph_index_") || t.operation.starts_with("decode_graph_mla_"){if t.maximum_bytes.map_or(true,|n|n==0){return Err("bounded tensor observation requires explicit positive maximum_bytes".into());}}else if t.maximum_bytes.is_some(){return Err("maximum_bytes belongs only to supported bounded tensor observations".into());}
        if t.operation.starts_with("glm5_window_") {
            let count=t.window_rows.ok_or("window capture requires explicit row count")?;
            let max=t.window_max_bytes.ok_or("window capture requires explicit byte bound")?;
            let end=t.row.checked_add(count).ok_or("window row extent overflow")?;
            if count==0 || max==0 || count.checked_mul(2).map_or(true,|n|n>max) || !t.request_labels.is_empty() || (!t.operation.ends_with("_sha256") && t.heights.iter().any(|&h|end>h)) {
                return Err("invalid bounded prefill window".into());
            }
        } else if t.window_rows.is_some() || t.window_max_bytes.is_some() {
            return Err("window parameters require an explicit window operation".into());
        }
        if t.operation=="decode_current_decision" && (t.layer!=0 || t.row!=0){return Err("current decision capture requires the actual frame selector layer0/row0".into());}
        if !matches!(
            validation_operation,
            "mla_common_stages"
                | "indexer_common_stages"
                | "decode_graph_index_query"
                | "decode_graph_index_query_input"
                | "decode_graph_index_query_input_f32"
                | "decode_graph_index_query_weights"
                | "decode_graph_index_query_raw"
                | "decode_graph_index_hidden"
                | "decode_graph_index_head_weights"
                | "decode_graph_index_keys"
                | "decode_graph_index_scores"
                | "decode_graph_index_position"
                | "decode_graph_index_sequence"
                | "decode_graph_mla_wvc_weights"
                | "decode_graph_mla_selected"
                | "decode_graph_mla_positions"
                | "decode_graph_mla_gathered"
                | "decode_graph_mla_query"
                | "decode_graph_mla_initial_scores"
                | "decode_graph_mla_scores"
                | "decode_graph_mla_weights"
                | "decode_graph_mla_attention"
                | "decode_graph_mla_projected"
                | "decode_graph_mla_output"
                | "selected_prepared_window"
                | "kda_transition_row"
                | "hc_post_operands"
                | "dense_prefill_operands"
                | "dense_decode_operands"
                | "hc_prepare_operands"
                | "decode_eager_hc_prepare"
                | "decode_eager_hc_post"
                | "decode_graph_kda_selected_prepared_q"
                | "decode_graph_kda_selected_prepared_k"
                | "decode_graph_kda_selected_prepared_v"
                | "decode_graph_kda_selected_prepared_gate"
                | "decode_graph_kda_selected_prepared_beta"
                | "q_projection"
                | "q_projection_full"
                | "beta_projection"
                | "shared_expert"
                | "sequence_state"
                | "snapshot"
                | "kda_inputs"
                | "kda_full_batch"
                | "decode_boundary"
                | "decode_current_decision"
                | "decode_eager_kda"
                | "decode_eager_shared_w13"
                | "decode_moe_input"
                | "decode_router_operands"
                | "prefill_first_logits"
                | "decode_moe_output"
                | "decode_graph_hc_entry"
                | "decode_graph_attn_prepared"
                | "decode_graph_attn_norm"
                | "decode_graph_attn_output"
                | "decode_graph_hc_post_attn"
                | "decode_graph_ffn_prepared"
                | "decode_graph_ffn_norm"
                | "decode_graph_ffn_output"
                | "decode_graph_dense_gate"
                | "decode_graph_dense_up"
                | "decode_graph_dense_output"
                | "decode_graph_hc_post_ffn"
                | "decode_graph_shared_w13_input"
                | "decode_graph_shared_w13_output"
                | "decode_graph_kda_q_projected"
                | "decode_graph_kda_k_projected"
                | "decode_graph_kda_v_projected"
                | "decode_graph_kda_conv_incoming"
                | "decode_graph_kda_q_conv"
                | "decode_graph_kda_k_conv"
                | "decode_graph_kda_v_conv"
                | "decode_graph_kda_forget"
                | "decode_graph_kda_beta"
                | "decode_graph_kda_recurrent_incoming"
                | "decode_graph_kda_recurrent_outgoing"
                | "decode_graph_kda_a_log"
                | "decode_graph_kda_dt_bias"
                | "decode_graph_kda_norm_weight"
                | "decode_graph_kda_recurrent_output"
                | "decode_graph_kda_output_gate"
                | "decode_graph_kda_rms_gate"
                | "mla_prefix_operands"
                | "mla_prefill_operands"
                | "mla_decode_operands"
                | "glm5_window_kda_q_proj"
                | "glm5_window_kda_k_proj"
                | "glm5_window_kda_v_proj"
                | "glm5_window_kda_q_conv"
                | "glm5_window_kda_k_conv"
                | "glm5_window_kda_v_conv"
                | "glm5_window_kda_forget_gate"
                | "glm5_window_kda_beta"
                | "glm5_window_kda_recurrent"
                | "glm5_window_kda_output_gate"
                | "glm5_window_kda_rms_gate"
                | "glm5_window_kda_o_proj"
                | "glm5_window_hc_entry"
                | "glm5_window_attn_prepared"
                | "glm5_window_attn_norm"
                | "glm5_window_attn_output"
                | "glm5_window_post_attn_state"
                | "glm5_window_ffn_prepared"
                | "glm5_window_ffn_norm"
                | "glm5_window_ffn_output"
                | "glm5_window_output_state"
                | "glm5_row_hc_entry"
                | "glm5_row_attn_prepared"
                | "glm5_row_attn_norm"
                | "glm5_row_attn_output"
                | "glm5_row_post_attn_state"
                | "glm5_row_ffn_prepared"
                | "glm5_row_ffn_norm"
                | "glm5_row_ffn_output"
                | "glm5_row_output_state"
                | "prefill_moe_row"
                // BEGIN bounded routed W1 lane admission
                | "prefill_moe_lane"
                // END bounded routed W1 lane admission
                // BEGIN bounded same-operator partition admission
                | "prefill_moe_partition"
                // END bounded same-operator partition admission
                | "glm5_stage_output_gate"
                | "glm5_stage_rms_gate"
                | "glm5_stage_o_proj"
                | "glm5_stage_post_attn_state"
                | "glm5_stage_ffn_prepared"
                | "glm5_stage_ffn_norm"
                | "glm5_stage_ffn_output"
                | "glm5_stage_output_state"
                | "glm5_stage_attn_norm"
                | "glm5_stage_recurrent"
                | "glm5_index_key"
        ) || t.heights.is_empty()
            || t.heights.iter().any(|&height| height <= t.row)
        {
            return Err("invalid matched probe operation/row/heights".into());
        }
        if t.operation == "prefill_first_logits" && (t.layer != 0 || t.row != 0 || t.request_labels.is_empty() || t.request_labels.iter().any(|label|label.strip_prefix("prompt-").and_then(|n|n.parse::<usize>().ok()).map_or(true,|n|!t.heights.contains(&n)))) { return Err("first-prefill logits require layer zero, row zero and explicit request labels".into()); }
        if t.operation == "kda_full_batch" && (t.row == 0 || t.request_labels.is_empty() || t.request_labels.iter().any(|label| label.strip_prefix("prompt-").and_then(|n|n.parse::<usize>().ok()).map_or(true, |n|n==0))) {
            return Err("full-batch recurrence capture requires explicit prompt identity and interior consumed-row checkpoint".into());
        }
        if t.operation=="decode_router_operands" && (t.row!=0 || !t.heights.iter().all(|h|config.targets.iter().any(|b|matches!(b.operation.as_str(),"decode_boundary"|"decode_current_decision")&&b.layer==0&&b.heights.contains(h)&&b.request_labels==t.request_labels))) { return Err("router observation requires row zero and matching actual decision-frame scope".into()); }
        if matches!(t.operation.as_str(), "decode_eager_kda" | "decode_eager_shared_w13" | "decode_eager_hc_post" | "decode_eager_hc_prepare" | "dense_decode_operands" | "mla_decode_operands") && (t.row != 0 || !t.heights.iter().all(|h| config.targets.iter().any(|b| matches!(b.operation.as_str(),"decode_boundary"|"decode_current_decision") && b.layer == 0 && b.heights.contains(h) && b.request_labels == t.request_labels))) {
            return Err("eager observation requires row zero and identical-scope actual decision frames".into());
        }
        if matches!(t.operation.as_str(),"mla_prefill_operands"|"mla_prefix_operands") && t.heights.iter().any(|&h|t.row>=h) {return Err("MLA observation row must be inside every admitted batch".into());}
        if t.operation=="mla_prefix_operands" && t.heights.iter().any(|&h|t.row.checked_add(1)!=Some(h)) {return Err("MLA prefix observation must end at actual final row".into());}
        // BEGIN bounded prefill stage admission
        if t.operation.starts_with("glm5_stage_") && t.row != 0 {
            return Err("whole prefill stage capture requires row zero".into());
        }
        // END bounded prefill stage admission
    }
    Ok(config)
}

// BEGIN bounded prefill stage selection
pub(crate) fn prefill_stage_operation(label: &str) -> Option<&'static str> {
    match label {
        "layer_glm5_kda_output_gate_last" => Some("glm5_stage_output_gate"),
        "layer_glm5_kda_rms_gate_last" => Some("glm5_stage_rms_gate"),
        "layer_glm5_kda_o_proj_last" => Some("glm5_stage_o_proj"),
        "layer_glm5_post_attn_state_last" => Some("glm5_stage_post_attn_state"),
        "layer_glm5_ffn_prepared_hidden_last" => Some("glm5_stage_ffn_prepared"),
        "layer_glm5_ffn_norm_hidden_last" => Some("glm5_stage_ffn_norm"),
        "layer_glm5_ffn_sublayer_last" => Some("glm5_stage_ffn_output"),
        "layer_glm5_output_state_last" => Some("glm5_stage_output_state"),
        "layer_glm5_attn_norm_hidden_last" => Some("glm5_stage_attn_norm"),
        "layer_glm5_kda_recurrent_last" => Some("glm5_stage_recurrent"),
        _ => None,
    }
}
// END bounded prefill stage selection

pub(crate) fn target(
    operation: &str,
    layer: usize,
    height: usize,
) -> Result<Option<Target>, String> {
    target_for_request(operation, layer, height, None)
}

fn configuration() -> Result<Option<&'static Config>, String> {
    static CONFIG: OnceLock<Result<Option<Config>, String>> = OnceLock::new();
    let config = CONFIG.get_or_init(|| match std::env::var("KRASIS_MATCHED_PROBE_CONFIG") {
        Ok(path) => parse(&std::fs::read_to_string(path).map_err(|e| e.to_string())?).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(e.to_string()),
    });
    config.as_ref().map(|c|c.as_ref()).map_err(Clone::clone)
}

pub(crate) fn enabled() -> Result<bool, String> { Ok(configuration()?.is_some()) }

pub(crate) fn target_for_request(
    operation: &str,
    layer: usize,
    height: usize,
    label: Option<&str>,
) -> Result<Option<Target>, String> {
    let Some(config) = configuration()? else {
        return Ok(None);
    };
    let mut matches = config.targets.iter().filter(|t| {
        t.operation == operation
            && t.layer == layer
            && t.heights.contains(&height)
            && (t.request_labels.is_empty()
                || label
                    .is_some_and(|label| t.request_labels.iter().any(|allowed| allowed == label)))
    });
    let Some(t) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err("ambiguous matched probe targets".into());
    }
    Ok(Some(Target {
        row: t.row,
        directory: config.directory.join(format!(
            "{operation}-layer{layer}-height{height}-row{}",
            t.row
        )),
    }))
}

pub(crate) fn prefill_window_parameters(operation:&str,layer:usize,height:usize)->Result<(usize,usize),String>{
    let config=configuration()?.ok_or("window configuration absent")?;
    let mut matches=config.targets.iter().filter(|t|t.operation==operation && t.layer==layer && t.heights.contains(&height));
    let t=matches.next().ok_or("window target absent")?;
    if matches.next().is_some(){return Err("ambiguous window target".into());}
    Ok((t.window_rows.ok_or("window rows absent")?,t.window_max_bytes.ok_or("window byte bound absent")?))
}

#[cfg(test)]
#[test]
fn bounded_window_parser_contract(){
    let valid=serde_json::json!({"directory":"/bounded","targets":[{"operation":"glm5_window_attn_norm","layer":0,"row":10,"heights":[23],"window_rows":13,"window_max_bytes":106496}]});
    assert!(parse(&valid.to_string()).is_ok());
    for field in ["window_rows","window_max_bytes"]{
        let mut v=valid.clone();v["targets"][0].as_object_mut().unwrap().remove(field);assert!(parse(&v.to_string()).is_err());
        let mut v=valid.clone();v["targets"][0][field]=serde_json::json!(0);assert!(parse(&v.to_string()).is_err());
    }
    for (field,value) in [("window_rows",serde_json::json!(14)),("window_rows",serde_json::json!(usize::MAX)),("window_max_bytes",serde_json::json!(25)),("operation",serde_json::json!("glm5_row_attn_norm")),("request_labels",serde_json::json!(["prompt-23"]))]{
        let mut v=valid.clone();v["targets"][0][field]=value;assert!(parse(&v.to_string()).is_err());
    }
}

pub(crate) fn capture_maximum_bytes(operation:&str,layer:usize,height:Option<usize>)->Result<usize,String>{
    let config=configuration()?.ok_or("MLA observation configuration absent")?;
    let mut rows=config.targets.iter().filter(|t|t.operation==operation&&t.layer==layer&&height.map_or(true,|h|t.heights.contains(&h)));
    let row=rows.next().ok_or("MLA observation target absent")?;
    if rows.next().is_some(){return Err("ambiguous MLA observation byte bound".into());}
    row.maximum_bytes.filter(|&n|n>0).ok_or("MLA observation byte bound absent".into())
}

impl Target {
    // Read actual merge operands while the caller's workspace owners remain borrowed.
    // Diagnostic only: no device allocation, arithmetic, or invented sequence cursor.
    pub(crate) fn hc_post_operands(&self, row: usize, rows: usize, hidden: usize, mult: usize,
        sublayer: u64, residual: u64, post: u64, comb: u64, output: u64,
        stream: cuda::CUstream, kernel: &str) -> Result<(), String> {
        if row >= rows || hidden == 0 || mult == 0 { return Err("HC operand row/shape invalid".into()); }
        let residual_cols=hidden.checked_mul(mult).ok_or("HC residual shape overflow")?;
        let comb_cols=mult.checked_mul(mult).ok_or("HC mixing shape overflow")?;
        for (name,ptr,cols,size) in [("sublayer.bf16",sublayer,hidden,2usize),("residual.bf16",residual,residual_cols,2),
            ("post.f32",post,mult,4),("comb.f32",comb,comb_cols,4)] {
            let bytes=cols.checked_mul(size).ok_or("HC operand extent overflow")?;
            let offset=row.checked_mul(bytes).ok_or("HC operand row offset overflow")?;
            self.device(name,ptr.checked_add(offset as u64).ok_or("HC operand pointer overflow")?,bytes,stream)?;
        }
        self.write("operands.json",&serde_json::to_vec_pretty(&serde_json::json!({
            "local_row":row,"rows":rows,"hidden":hidden,"mult":mult,"kernel":kernel,
            "sublayer":sublayer,"residual":residual,"post":post,"comb":comb,"output":output,
            "stream":stream as usize,"forcing":crate::matched_projection_transfer::forcing_label().is_some(),
            "device_diagnostic_allocation_bytes":0,"scope":"actual pre-launch operands; original caller owns all buffers"
        })).map_err(|e|e.to_string())?)
    }
    pub(crate) fn hc_prepare_operands(&self, row: usize, rows: usize, hidden: usize, mult: usize,
        state: u64, function: u64, base: u64, scale: u64, stream: cuda::CUstream,
        mode: &str, rms_eps: f32, hc_eps: f32, sinkhorn_iters: usize) -> Result<(),String> {
        if row>=rows || hidden==0 || mult==0 {return Err("HC prepare observed shape/row invalid".into());}
        let flat=hidden.checked_mul(mult).ok_or("HC observed flat overflow")?;
        let mix_width=mult.checked_add(2).and_then(|n|n.checked_mul(mult)).ok_or("HC observed mix overflow")?;
        self.hc_observed_row("state.bf16",state,row,flat,2,stream)?;
        self.device("function.f32",function,flat.checked_mul(mix_width).and_then(|n|n.checked_mul(4)).ok_or("HC observed weight extent overflow")?,stream)?;
        self.device("base.f32",base,mix_width.checked_mul(4).ok_or("HC observed base extent overflow")?,stream)?;
        self.device("scale.f32",scale,3*std::mem::size_of::<f32>(),stream)?;
        self.write("producer.json",&serde_json::to_vec_pretty(&serde_json::json!({"local_row":row,"rows":rows,"hidden":hidden,"mult":mult,"flat":flat,"mix_width":mix_width,
            "state":state,"function":function,"base":base,"scale":scale,"stream":stream as usize,"mode":mode,"rms_eps":rms_eps,"hc_eps":hc_eps,"sinkhorn_iters":sinkhorn_iters,
            "device_diagnostic_allocation_bytes":0,"forcing":crate::matched_projection_transfer::forcing_label().is_some(),"scope":"actual prepare input buffers; caller owners remain borrowed"})).map_err(|e|e.to_string())?)
    }
    pub(crate) fn hc_observed_row(&self,name:&str,ptr:u64,row:usize,cols:usize,size:usize,stream:cuda::CUstream)->Result<(),String>{
        let bytes=cols.checked_mul(size).ok_or("HC observed row extent overflow")?;
        let offset=row.checked_mul(bytes).ok_or("HC observed row offset overflow")?;
        self.device(name,ptr.checked_add(offset as u64).ok_or("HC observed pointer overflow")?,bytes,stream)
    }
    pub fn write(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write;
        std::fs::create_dir_all(&self.directory).map_err(|e| e.to_string())?;
        let path = self.directory.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                file.write_all(bytes).map_err(|e| e.to_string())?;
                file.sync_all().map_err(|e| e.to_string())
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if std::fs::read(&path).map_err(|e| e.to_string())? == bytes {
                    Ok(())
                } else {
                    Err(format!(
                        "matched probe refuses to overwrite changed artifact {}",
                        path.display()
                    ))
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn device(
        &self,
        name: &str,
        ptr: u64,
        bytes: usize,
        stream: cuda::CUstream,
    ) -> Result<(), String> {
        if ptr == 0 || bytes == 0 {
            return Err("empty/null matched probe tensor".into());
        }
        let rc = unsafe { cuda::lib().cuStreamSynchronize(stream) };
        if rc != cuda::CUresult::CUDA_SUCCESS {
            return Err(format!("matched probe fence: {rc:?}"));
        }
        let mut host = vec![0u8; bytes];
        let rc = unsafe { cuda::lib().cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), ptr, bytes) };
        if rc != cuda::CUresult::CUDA_SUCCESS {
            return Err(format!("matched probe download {name}: {rc:?}"));
        }
        self.write(name, &host)
    }

    pub fn row(
        &self,
        name: &str,
        ptr: u64,
        width: usize,
        stream: cuda::CUstream,
    ) -> Result<(), String> {
        let bytes = width.checked_mul(2).ok_or("matched probe width overflow")?;
        let offset = self
            .row
            .checked_mul(bytes)
            .ok_or("matched probe row overflow")?;
        self.device(
            name,
            ptr.checked_add(offset as u64)
                .ok_or("matched probe pointer overflow")?,
            bytes,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_batch_capture_requires_prompt_scope_and_interior_checkpoint() {
        let good=serde_json::json!({"directory":"/tmp/probe","targets":[{"operation":"kda_full_batch","layer":0,"row":8,"heights":[21],"request_labels":["prompt-85"]}]});
        assert!(parse(&good.to_string()).is_ok());
        for (row,labels) in [(0,vec!["prompt-85"]),(21,vec!["prompt-85"]),(8,vec![]),(8,vec!["prompt-0"]),(8,vec!["invalid"])] {
            let bad=serde_json::json!({"directory":"/tmp/probe","targets":[{"operation":"kda_full_batch","layer":0,"row":row,"heights":[21],"request_labels":labels}]});
            assert!(parse(&bad.to_string()).is_err());
        }
    }
    // BEGIN selected prefill row admission tests
    #[test]
    fn selected_row_admission_preserves_whole_stage_and_bounds_contracts() {
        for operation in ["glm5_row_ffn_output", "prefill_moe_row"] {
            let good=serde_json::json!({"directory":"/tmp/probe","targets":[{"operation":operation,"layer":3,"row":4,"heights":[10500,10564]}]});
            assert!(parse(&good.to_string()).is_ok());
            let bad=serde_json::json!({"directory":"/tmp/probe","targets":[{"operation":operation,"layer":3,"row":4,"heights":[4]}]});
            assert!(parse(&bad.to_string()).is_err());
        }
        assert!(parse(r#"{"directory":"/tmp/probe","targets":[{"operation":"glm5_stage_ffn_output","layer":3,"row":4,"heights":[10500]}]}"#).is_err());
    }
    // END selected prefill row admission tests
    // BEGIN bounded prefill stage regression
    #[test]
    fn whole_stage_capture_rejects_row_selection_and_unknown_stage() {
        assert!(parse(r#"{"directory":"/tmp/probe","targets":[{"operation":"glm5_stage_ffn_output","layer":0,"row":0,"heights":[4]}]}"#).is_ok());
        assert!(parse(r#"{"directory":"/tmp/probe","targets":[{"operation":"glm5_stage_ffn_output","layer":0,"row":1,"heights":[4]}]}"#).is_err());
        assert!(parse(r#"{"directory":"/tmp/probe","targets":[{"operation":"glm5_stage_all","layer":0,"row":0,"heights":[4]}]}"#).is_err());
        assert_eq!(prefill_stage_operation("layer_other_model_output"), None);
    }
    // END bounded prefill stage regression
    #[test]
    fn admission_rejects_unknown_empty_and_out_of_range_targets() {
        assert!(parse(r#"{"directory":"/tmp/probe","targets":[{"operation":"q_projection","layer":0,"row":3,"heights":[4]}]}"#).is_ok());
        for raw in [
            r#"{"directory":"relative","targets":[]}"#,
            r#"{"directory":"/tmp/probe","targets":[{"operation":"q_projection","layer":0,"row":4,"heights":[4]}]}"#,
            r#"{"directory":"/tmp/probe","targets":[{"operation":"unknown","layer":0,"row":0,"heights":[4]}]}"#,
        ] {
            assert!(parse(raw).is_err());
        }
    }
}

impl Target {
    pub fn child(mut self, name: &str) -> Result<Self, String> {
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return Err("invalid matched probe path component".into());
        }
        self.directory.push(name);
        Ok(self)
    }

    pub fn allocations(
        &self,
        allocations: &[crate::session_cache::SequenceStateAllocation],
        tokens: usize,
    ) -> Result<(), String> {
        let rc = unsafe { cuda::lib().cuCtxSynchronize() };
        if rc != cuda::CUresult::CUDA_SUCCESS {
            return Err(format!("matched state fence: {rc:?}"));
        }
        let mut metadata = Vec::new();
        for (index, allocation) in allocations.iter().enumerate() {
            let bytes = allocation.used_bytes(tokens);
            if bytes == 0 {
                continue;
            }
            let file = format!("allocation-{index}.bin");
            self.device(&file, allocation.ptr, bytes, std::ptr::null_mut())?;
            metadata
                .push(serde_json::json!({"file":file,"allocation":allocation,"used_bytes":bytes}));
        }
        self.write("allocations.json",serde_json::to_vec_pretty(&serde_json::json!({
            "logical_tokens":tokens,"device_diagnostic_allocation_bytes":0,"allocations":metadata
        })).map_err(|e|e.to_string())?.as_slice())
    }

    pub fn snapshot(
        &self,
        id: crate::session_cache::SnapshotId,
        snapshot: &crate::session_cache::SessionSnapshot,
    ) -> Result<(), String> {
        let mut metadata = Vec::new();
        for (index, blob) in snapshot.state_blobs.iter().enumerate() {
            let file = format!("blob-{index}.bin");
            self.write(&file, &blob.bytes)?;
            metadata.push(serde_json::json!({"file":file,"allocation_name":blob.allocation_name,"kind":blob.kind,
                "layer_idx":blob.layer_idx,"device_ordinal":blob.device_ordinal,"dtype":blob.dtype,
                "element_size":blob.element_size,"shape":blob.shape,"strides_bytes":blob.strides_bytes,"bytes":blob.bytes.len()}));
        }
        self.write("snapshot.json",serde_json::to_vec_pretty(&serde_json::json!({
            "snapshot_id":id,"compatibility":snapshot.compatibility,"consumed_token_ids":snapshot.consumed_token_ids,
            "positions":snapshot.positions,"blobs":metadata,"device_diagnostic_allocation_bytes":0
        })).map_err(|e|e.to_string())?.as_slice())
    }
}

// Return only explicitly admitted graph observation targets at startup.
pub(crate) fn graph_stage_specs() -> Result<Vec<(usize, String)>, String> {
    let path = match std::env::var("KRASIS_MATCHED_PROBE_CONFIG") {
        Ok(path) => path,
        Err(std::env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(e) => return Err(e.to_string()),
    };
    let config = parse(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)?;
    graph_stage_specs_from_config(&config)
}
fn graph_stage_specs_from_config(config: &Config) -> Result<Vec<(usize, String)>, String> {
    let mut specs = std::collections::BTreeSet::new();
    for t in config.targets.iter().filter(|t| t.operation.starts_with("decode_graph_")) {
        if t.row != 0 || !t.heights.iter().all(|h| config.targets.iter().any(|b| matches!(b.operation.as_str(),"decode_boundary"|"decode_current_decision") && b.layer == 0 && b.heights.contains(h) && b.request_labels == t.request_labels)) {
            return Err("graph observation requires row zero and identical-scope actual decision frames".into());
        }
        if !specs.insert((t.layer, t.operation.clone())) { return Err("ambiguous graph-stage capture specification".into()); }
    }
    Ok(specs.into_iter().collect())
}
#[cfg(test)]
mod graph_stage_admission_tests {
    use super::*;
    #[test]
    fn graph_stages_require_actual_boundaries_and_unique_specs() {
        let stage = r#"{"operation":"decode_graph_attn_norm","layer":0,"row":0,"heights":[101]}"#;
        let boundary = r#"{"operation":"decode_boundary","layer":0,"row":0,"heights":[101]}"#;
        let make = |body: &str| parse(&format!(r#"{{"directory":"/diagnostic","targets":[{body}]}}"#));
        assert!(graph_stage_specs_from_config(&make(stage).unwrap()).is_err());
        assert!(graph_stage_specs_from_config(&make(&format!("{stage},{stage},{boundary}")).unwrap()).is_err());
        assert_eq!(graph_stage_specs_from_config(&make(&format!("{stage},{boundary}")).unwrap()).unwrap(),vec![(0,"decode_graph_attn_norm".into())]);
        assert!(graph_stage_specs_from_config(&make(&format!("{},{boundary}",stage.replace("\"row\":0","\"row\":1"))).unwrap()).is_err());
        assert!(make(&format!("{},{boundary}",stage.replace("attn_norm","unknown"))).is_err());
    }
}

// Explicit shared-only Marlin dispatch experiment; ordinary autotune stays
// complete. This selects only a geometry actually measured on loaded tensors.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MarlinSharedDispatchControls {
    selections: Vec<MarlinSharedSelection>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MarlinSharedSelection {
    k: usize,
    n: usize,
    group_size: usize,
    bits: usize,
    ksplits: usize,
}
impl MarlinSharedDispatchControls {
    fn parse(raw: &str) -> Result<Self, String> {
        let p: Self = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if p.selections.is_empty() { return Err("empty shared Marlin diagnostic selections".into()); }
        let mut shapes = std::collections::HashSet::new();
        for s in &p.selections {
            if s.k == 0 || s.n == 0 || s.group_size == 0 || !matches!(s.bits, 4 | 8)
                || s.ksplits == 0 || !shapes.insert((s.k, s.n, s.group_size, s.bits)) {
                return Err("invalid/duplicate shared Marlin diagnostic selection".into());
            }
        }
        Ok(p)
    }
    pub(crate) fn load() -> Result<Option<Self>, String> {
        let file = match std::env::var("KRASIS_MATCHED_SHARED_MARLIN_DISPATCH_CONFIG") {
            Ok(file) => file,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        require_model_diagnostic_probe()?;
        if !std::path::Path::new(&file).is_absolute() {
            return Err("shared Marlin diagnostic requires an absolute plan path".into());
        }
        Self::parse(&std::fs::read_to_string(file).map_err(|e| e.to_string())?).map(Some)
    }
    pub(crate) fn admit(
        &self, measured: &[(usize, usize, usize, usize, Vec<(usize, f64)>)],
    ) -> Result<std::collections::HashMap<(usize, usize, bool), usize>, String> {
        let mut selected = std::collections::HashMap::new();
        for s in &self.selections {
            let matches: Vec<_> = measured.iter().filter(|m| (m.0,m.1,m.2,m.3) == (s.k,s.n,s.group_size,s.bits)).collect();
            if matches.len() != 1 { return Err("shared Marlin diagnostic shape missing/ambiguous in complete measurements".into()); }
            let results = &matches[0].4;
            if results.is_empty() || results.iter().any(|&(k,ms)| k == 0 || !ms.is_finite() || ms <= 0.)
                || results.iter().filter(|&&(k,_)| k == s.ksplits).count() != 1 {
                return Err("shared Marlin diagnostic split lacks valid actual measurement".into());
            }
            selected.insert((s.k,s.n,s.bits==8),s.ksplits);
        }
        Ok(selected)
    }
}
#[cfg(test)]
mod shared_marlin_dispatch_tests {
    use super::*;
    fn control() -> MarlinSharedDispatchControls {
        MarlinSharedDispatchControls::parse(r#"{"selections":[{"k":32,"n":16,"group_size":8,"bits":8,"ksplits":2}]}"#).unwrap()
    }
    #[test]
    fn rejects_missing_ambiguous_unmeasured_nonfinite_and_wrong_precision() {
        let c = control();
        assert!(c.admit(&[]).is_err());
        let m = (32,16,8,8,vec![(1,1.),(2,0.5)]);
        assert!(c.admit(&[m.clone(),m.clone()]).is_err());
        assert!(c.admit(&[(32,16,8,8,vec![(1,1.)])]).is_err());
        assert!(c.admit(&[(32,16,8,8,vec![(1,1.),(2,f64::NAN)])]).is_err());
        assert!(c.admit(&[(32,16,8,8,vec![(1,0.),(2,0.5)])]).is_err());
        assert!(c.admit(&[(32,16,8,4,vec![(1,1.),(2,0.5)])]).is_err());
        assert!(c.admit(&[(32,16,16,8,vec![(1,1.),(2,0.5)])]).is_err());
        assert_eq!(c.admit(&[m]).unwrap().get(&(32,16,true)),Some(&2));
    }
    #[test]
    fn rejects_invalid_and_duplicate_selections() {
        for raw in [r#"{"selections":[]}"#,
            r#"{"selections":[{"k":32,"n":16,"group_size":8,"bits":6,"ksplits":2}]}"#,
            r#"{"selections":[{"k":32,"n":16,"group_size":8,"bits":8,"ksplits":0}]}"#,
            r#"{"selections":[{"k":32,"n":16,"group_size":8,"bits":8,"ksplits":2},{"k":32,"n":16,"group_size":8,"bits":8,"ksplits":1}]}"#] {
            assert!(MarlinSharedDispatchControls::parse(raw).is_err());
        }
    }
}

// Native recurrent pointers are unused by the selected eleven-plane family.
pub(crate) fn validate_kda_state_observation(operation:&str,selected:bool)->Result<(),String>{
    if selected && matches!(operation,"decode_graph_kda_recurrent_incoming"|"decode_graph_kda_recurrent_outgoing"){
        return Err("selected KDA state observation requires its actual typed registry; native recurrent pointer is unused".into());
    }
    Ok(())
}
#[cfg(test)]
#[test]
fn kda_observation_rejects_inactive_native_state(){
    for op in ["decode_graph_kda_recurrent_incoming","decode_graph_kda_recurrent_outgoing"]{assert!(validate_kda_state_observation(op,true).is_err());validate_kda_state_observation(op,false).unwrap();}
    for op in ["decode_graph_kda_conv_incoming","decode_graph_kda_recurrent_output","decode_graph_kda_selected_prepared_q"]{validate_kda_state_observation(op,true).unwrap();validate_kda_state_observation(op,false).unwrap();}
}

// Diagnostic buffer extents use actual registered KDA geometry and operand types.
pub(crate) fn kda_graph_stage_layout(operation: &str, heads: usize, dim: usize, kernel: usize) -> Result<(usize, &'static str), String> {
 if heads==0 || dim==0 || kernel<2 { return Err("invalid registered KDA graph geometry".into()); }
 let hd=heads.checked_mul(dim).ok_or("KDA graph width overflow")?;
 let result=match operation {
 "decode_graph_kda_recurrent_incoming"|"decode_graph_kda_recurrent_outgoing" => (hd.checked_mul(dim).ok_or("KDA graph state overflow")?,"float32"),
 "decode_graph_kda_conv_incoming" => (hd.checked_mul(kernel-1).and_then(|n|n.checked_mul(3)).ok_or("KDA graph convolution extent overflow")?,"float32"),
 "decode_graph_kda_a_log" => (heads,"float32"),
 "decode_graph_kda_dt_bias" => (hd,"float32"),
 "decode_graph_kda_norm_weight" => (dim,"float32"),
 "decode_graph_kda_beta" => (heads,"bfloat16"),
 "decode_graph_kda_selected_prepared_beta" => (heads,"float32"),
 "decode_graph_kda_selected_prepared_q"|"decode_graph_kda_selected_prepared_k"|"decode_graph_kda_selected_prepared_v"|"decode_graph_kda_selected_prepared_gate" => (hd,"float32"),
 "decode_graph_kda_q_projected"|"decode_graph_kda_k_projected"|"decode_graph_kda_v_projected"|"decode_graph_kda_q_conv"|"decode_graph_kda_k_conv"|"decode_graph_kda_v_conv"|"decode_graph_kda_forget"|"decode_graph_kda_recurrent_output"|"decode_graph_kda_output_gate"|"decode_graph_kda_rms_gate" => (hd,"bfloat16"),
 _=>return Err("unknown KDA graph stage".into()),
 }; Ok(result)
}
#[cfg(test)] mod kda_graph_layout_tests { use super::*;
 #[test] fn extents_follow_registered_geometry_and_types() { assert_eq!(kda_graph_stage_layout("decode_graph_kda_recurrent_incoming",3,31,4).unwrap(),(2883,"float32")); assert_eq!(kda_graph_stage_layout("decode_graph_kda_conv_incoming",3,31,4).unwrap(),(837,"float32")); assert_eq!(kda_graph_stage_layout("decode_graph_kda_q_conv",3,31,4).unwrap(),(93,"bfloat16")); assert_eq!(kda_graph_stage_layout("decode_graph_kda_beta",3,31,4).unwrap(),(3,"bfloat16")); assert_eq!(kda_graph_stage_layout("decode_graph_kda_norm_weight",3,31,4).unwrap(),(31,"float32")); }
 #[test] fn reject_invalid_unknown_and_overflow() { assert!(kda_graph_stage_layout("decode_graph_kda_q_conv",0,31,4).is_err()); assert!(kda_graph_stage_layout("decode_graph_kda_q_conv",3,31,1).is_err()); assert!(kda_graph_stage_layout("decode_graph_kda_unknown",3,31,4).is_err()); assert!(kda_graph_stage_layout("decode_graph_kda_recurrent_incoming",usize::MAX,31,4).is_err()); }
}

// Loaded geometry only, for explicitly admitted shared Marlin diagnostics.
pub(crate) fn shared_w13_graph_stage_layout(operation: &str, hidden: usize, intermediate: usize, gated: bool, bits: usize, group: usize) -> Result<(usize, &'static str), String> {
    if hidden == 0 || intermediate == 0 || !matches!(bits,4|8) || group == 0 || hidden % group != 0 || hidden % 16 != 0 {
        return Err("unsupported loaded shared Marlin diagnostic geometry".into());
    }
    let n=intermediate.checked_mul(if gated {2} else {1}).ok_or("shared diagnostic extent overflow")?;
    if n%16 != 0 { return Err("unsupported shared Marlin output geometry".into()); }
    let elements=match operation { "decode_graph_shared_w13_input"=>hidden, "decode_graph_shared_w13_output"=>n,_=>return Err("unknown shared graph stage".into()) };
    elements.checked_mul(2).ok_or("shared diagnostic byte overflow")?;
    Ok((elements,"bfloat16"))
}
#[cfg(test)] mod shared_graph_stage_admission_tests {
 use super::*;
 #[test] fn dimensions_and_format_follow_loaded_model() {
  assert_eq!(shared_w13_graph_stage_layout("decode_graph_shared_w13_input",2048,512,true,8,128).unwrap(),(2048,"bfloat16"));
  assert_eq!(shared_w13_graph_stage_layout("decode_graph_shared_w13_output",4096,1536,true,4,64).unwrap(),(3072,"bfloat16"));
  assert_eq!(shared_w13_graph_stage_layout("decode_graph_shared_w13_output",1024,256,false,8,128).unwrap(),(256,"bfloat16"));
 }
 #[test] fn reject_unsupported_layout_and_overflow() {
  for (h,i,b,g) in [(0,512,8,128),(2048,0,8,128),(2048,512,16,128),(2048,512,8,0),(2048,512,8,31),(2048,511,8,128),(2048,usize::MAX,8,128)] {
   assert!(shared_w13_graph_stage_layout("decode_graph_shared_w13_output",h,i,true,b,g).is_err());
  }
  assert!(shared_w13_graph_stage_layout("decode_graph_shared_unknown",2048,512,true,8,128).is_err());
 }
}

// An eager-step-only startup switch; no graph buffer or replay substitution.
pub(crate) fn hqq_projection_capture_extents(rows: usize, packed_stride: usize, scales_stride: usize, zeros_stride: usize) -> Result<(usize, usize, usize), String> {
    if [rows, packed_stride, scales_stride, zeros_stride].contains(&0) {
        return Err("invalid loaded HQQ capture strides/rows".into());
    }
    let packed=rows.checked_mul(packed_stride).ok_or("HQQ packed capture extent overflow")?;
    let scales=rows.checked_mul(scales_stride).ok_or("HQQ scales capture extent overflow")?;
    let zeros=rows.checked_mul(zeros_stride).ok_or("HQQ zeros capture extent overflow")?;
    Ok((packed,scales,zeros))
}
#[cfg(test)] mod hqq_capture_extent_tests {
    use super::*;
    #[test] fn separates_loaded_payload_and_parameter_extents() {
        assert_eq!(hqq_projection_capture_extents(8192,3072,128,128).unwrap(),(25165824,1048576,1048576));
        assert_eq!(hqq_projection_capture_extents(64,3072,128,128).unwrap(),(196608,8192,8192));
        assert_eq!(hqq_projection_capture_extents(3,24,12,12).unwrap(),(72,36,36));
    }
    #[test] fn rejects_zero_and_overflow() {
        for x in [(0,3,4,4),(2,0,4,4),(2,3,0,4),(2,3,4,0),(usize::MAX,2,4,4),(2,1,usize::MAX,4),(2,1,4,usize::MAX)] {
            assert!(hqq_projection_capture_extents(x.0,x.1,x.2,x.3).is_err());
        }
    }
}

pub(crate) fn eager_kda_capture_enabled() -> Result<bool, String> {
    eager_capture_enabled("decode_eager_kda")
}

pub(crate) fn moe_input_capture_enabled() -> Result<bool, String> {
    Ok(configuration()?.is_some_and(|c| c.targets.iter().any(|t|t.operation=="decode_moe_input")))
}

pub(crate) fn router_capture_enabled() -> Result<bool, String> {
    Ok(configuration()?.is_some_and(|c| c.targets.iter().any(|t|t.operation=="decode_router_operands")))
}

pub(crate) fn eager_shared_capture_enabled() -> Result<bool, String> {
    eager_capture_enabled("decode_eager_shared_w13")
}

fn eager_capture_enabled(operation: &str) -> Result<bool, String> {
    let path = match std::env::var("KRASIS_MATCHED_PROBE_CONFIG") {
        Ok(path) => path, Err(std::env::VarError::NotPresent) => return Ok(false), Err(e) => return Err(e.to_string()),
    };
    let config = parse(&std::fs::read_to_string(path).map_err(|e|e.to_string())?)?;
    Ok(config.targets.iter().any(|t|t.operation == operation))
}

#[cfg(test)]
#[test]
fn eager_kda_capture_requires_actual_boundary_scope() {
    let make = |targets:&str| parse(&format!(r#"{{"directory":"/diagnostic","targets":[{targets}]}}"#));
    let eager=r#"{"operation":"decode_eager_kda","layer":0,"row":0,"heights":[10501],"request_labels":["seed"]}"#;
    let boundary=r#"{"operation":"decode_boundary","layer":0,"row":0,"heights":[10501],"request_labels":["seed"]}"#;
    assert!(make(eager).is_err());
    assert!(make(&format!("{eager},{boundary}")).is_ok());
    assert!(make(&format!("{eager},{}",boundary.replace("seed","other"))).is_err());
    assert!(make(&format!("{},{boundary}",eager.replace("\"row\":0","\"row\":1"))).is_err());
    let shared=eager.replace("decode_eager_kda", "decode_eager_shared_w13");
    assert!(make(&shared).is_err());
    assert!(make(&format!("{shared},{boundary}")).is_ok());
    assert!(make(&format!("{shared},{}",boundary.replace("seed","other"))).is_err());
}

#[cfg(test)]
#[test]
fn router_capture_requires_exact_request_boundary_scope() {
 let make=|targets:&str|parse(&format!(r#"{{"directory":"/diagnostic","targets":[{targets}]}}"#));
 let op=r#"{"operation":"decode_router_operands","layer":3,"row":0,"heights":[25],"request_labels":["sample"]}"#;
 let boundary=r#"{"operation":"decode_boundary","layer":0,"row":0,"heights":[25],"request_labels":["sample"]}"#;
 assert!(make(op).is_err());assert!(make(&format!("{op},{boundary}")).is_ok());assert!(make(&format!("{op},{}",boundary.replace("sample","different"))).is_err());assert!(make(&format!("{},{}",op.replace("\"row\":0","\"row\":1"),boundary)).is_err());
}

pub(crate) fn first_prefill_logits_enabled() -> Result<bool,String> { Ok(configuration()?.is_some_and(|c|c.targets.iter().any(|t|t.operation=="prefill_first_logits"))) }
#[cfg(test)] #[test] fn first_prefill_observer_requires_bounded_logical_label() {
 let make=|target:&str|parse(&format!(r#"{{"directory":"/diagnostic","targets":[{target}]}}"#));
 let op=r#"{"operation":"prefill_first_logits","layer":0,"row":0,"heights":[9627],"request_labels":["prompt-9627"]}"#;
 assert!(make(op).is_ok());assert!(make(&op.replace("prompt-9627","prompt-9626")).is_err());assert!(make(&op.replace("prompt-9627","request-unknown")).is_err());assert!(make(&op.replace("\"layer\":0","\"layer\":1")).is_err());assert!(make(&op.replace("\"row\":0","\"row\":1")).is_err());assert!(make(&op.replace("[\"prompt-9627\"]","[]")).is_err());
}

#[cfg(test)]
#[test]
fn mla_observation_requires_bounded_prefix_and_current_decode_frame(){
 let directory=serde_json::to_string(&std::env::temp_dir().join("krasis-mla-observation")).unwrap();
 let parse_targets=|targets:&str|parse(&format!(r#"{{"directory":{directory},"targets":[{targets}]}}"#));
 let current=r#"{"operation":"mla_prefill_operands","layer":3,"row":8,"heights":[9]}"#;
 assert!(parse_targets(current).is_ok());assert!(parse_targets(&current.replace("\"row\":8","\"row\":9")).is_err());
 assert!(parse_targets(&current.replace("[9]","[9,8]")).is_err());
 let prefix=current.replace("mla_prefill_operands","mla_prefix_operands");assert!(parse_targets(&prefix).is_ok());assert!(parse_targets(&prefix.replace("\"row\":8","\"row\":7")).is_err());
 let decode=r#"{"operation":"mla_decode_operands","layer":3,"row":0,"heights":[9],"request_labels":["current"]}"#;
 let boundary=r#"{"operation":"decode_boundary","layer":0,"row":0,"heights":[9],"request_labels":["current"]}"#;
 assert!(parse_targets(decode).is_err());assert!(parse_targets(&format!("{decode},{boundary}")).is_ok());
 assert!(parse_targets(&format!("{decode},{}",boundary.replace("current","other"))).is_err());
}

#[cfg(test)]mod sparse_decision_tests {
 use super::*;
 #[test]fn current_decision_admits_only_actual_frame_selector(){
  let config=|layer,row|format!(r#"{{"directory":"/tmp/krasis-sparse-decision-control","targets":[{{"operation":"decode_current_decision","layer":{layer},"row":{row},"heights":[10501]}}]}}"#);
  assert!(parse(&config(0,0)).is_ok());
  assert!(parse(&config(1,0)).is_err());assert!(parse(&config(0,1)).is_err());
 }
 #[test]fn sparse_decision_preserves_frame_scope_and_stage_uniqueness(){
  let frame=r#"{"operation":"decode_current_decision","layer":0,"row":0,"heights":[101],"request_labels":["seed"]}"#;
  let stage=r#"{"operation":"decode_graph_attn_norm","layer":0,"row":0,"heights":[101],"request_labels":["seed"]}"#;
  let eager=r#"{"operation":"mla_decode_operands","layer":3,"row":0,"heights":[101],"request_labels":["seed"]}"#;
  let make=|body:&str|parse(&format!(r#"{{"directory":"/tmp/sparse-frame-control","targets":[{body}]}}"#));
  let config=make(&format!("{stage},{frame},{eager}")).unwrap();
  assert_eq!(graph_stage_specs_from_config(&config).unwrap(),vec![(0,"decode_graph_attn_norm".into())]);
  assert!(make(&format!("{eager}")).is_err());
  assert!(make(&format!("{eager},{}",frame.replace("seed","other"))).is_err());
  assert!(graph_stage_specs_from_config(&make(&format!("{stage},{frame},{stage}")).unwrap()).is_err());
  assert!(graph_stage_specs_from_config(&make(&format!("{stage},{}",frame.replace("seed","other"))).unwrap()).is_err());
 }

}

#[cfg(test)]
#[test]
fn mla_stage_bounds_are_explicit_and_not_accepted_by_other_modes(){
    let valid=serde_json::json!({"directory":"/bounded","targets":[{"operation":"mla_common_stages","layer":3,"row":20,"heights":[21],"maximum_bytes":64}]});
    assert!(parse(&valid.to_string()).is_ok());
    let mut missing=valid.clone();missing["targets"][0].as_object_mut().unwrap().remove("maximum_bytes");assert!(parse(&missing.to_string()).is_err());
    let mut zero=valid.clone();zero["targets"][0]["maximum_bytes"]=serde_json::json!(0);assert!(parse(&zero.to_string()).is_err());
    let mut other=valid.clone();other["targets"][0]["operation"]=serde_json::json!("glm5_row_attn_norm");assert!(parse(&other.to_string()).is_err());
    let mut graph=valid.clone();graph["targets"][0]["operation"]=serde_json::json!("decode_graph_mla_scores");graph["targets"][0]["row"]=serde_json::json!(0);graph["targets"].as_array_mut().unwrap().push(serde_json::json!({"operation":"decode_current_decision","layer":0,"row":0,"heights":[21]}));assert_eq!(graph_stage_specs_from_config(&parse(&graph.to_string()).unwrap()).unwrap(),vec![(3,"decode_graph_mla_scores".into())]);
}

#[cfg(test)]
#[test]
fn indexer_stage_bounds_and_current_frame_are_required(){
 for op in ["indexer_common_stages","decode_graph_index_query_input","decode_graph_index_query_input_f32","decode_graph_index_query_weights","decode_graph_index_query_raw","decode_graph_index_hidden","decode_graph_index_query","decode_graph_index_head_weights","decode_graph_index_keys","decode_graph_index_scores","decode_graph_index_position","decode_graph_index_sequence"] {
  let mut v=serde_json::json!({"directory":"/bounded","targets":[{"operation":op,"layer":3,"row":0,"heights":[10548],"maximum_bytes":4096},{"operation":"decode_current_decision","layer":0,"row":0,"heights":[10548]}]});
  let config=parse(&v.to_string()).unwrap();
  if op.starts_with("decode_graph_"){assert_eq!(graph_stage_specs_from_config(&config).unwrap(),vec![(3,op.into())]);}
  v["targets"][0].as_object_mut().unwrap().remove("maximum_bytes");assert!(parse(&v.to_string()).is_err());
  v["targets"][0]["maximum_bytes"]=serde_json::json!(0);assert!(parse(&v.to_string()).is_err());
 }
}

// Explicit digest observations; ordinary raw observation semantics remain unchanged.
include!("matched_probe_digest.rs");
#[cfg(test)]
mod prefix_digest_tests { use super::*; include!("prefix_digest_tests.rs"); }
