// Bounded passive observations of the actual scalar-GDN producer. Included in
// gpu_decode to reuse the qualified captured-D2H/current-eager-frame machinery.
#[derive(Clone, Copy)]
enum GdnProducerFrame {
    Captured(usize),
    Eager(usize),
}
#[derive(Clone, Copy)]
enum GdnProducerPoint {
    BeforeNorm(bool),
    Projection,
    BeforeConv,
    AfterConv,
    GateBeta,
    AfterRecurrence,
}
fn gdn_producer_stage_layout(
    graph: &GpuDecodeGraph,
    layer: usize,
    operation: &str,
) -> Result<(usize, &'static str, serde_json::Value), String> {
    for flag in ["KRASIS_LA_DECODE_UNFUSED", "KRASIS_LA_FP32_OUT_PROJ"] {
        if std::env::var(flag).is_ok_and(|v| v != "0") {
            return Err(format!("GDN producer observation requires ordinary fused/BF16-output route; {flag} changes this source contract"));
        }
    }
    let loaded = graph
        .layers
        .get(layer)
        .ok_or("GDN observed actual layer absent")?;
    let GpuAttnConfig::LinearAttention {
        nk,
        nv,
        dk,
        dv,
        hr,
        kernel_dim,
        conv_dim,
        scale,
        ..
    } = &loaded.attn
    else {
        return Err(
            "GDN producer observation requires an actual scalar linear-attention layer".into(),
        );
    };
    let mul = |values: &[usize]| {
        values.iter().try_fold(1usize, |n, &v| {
            if v == 0 {
                Err("empty GDN producer extent")
            } else {
                n.checked_mul(v).ok_or("GDN producer extent overflow")
            }
        })
    };
    let keys = mul(&[*nk, *dk])?;
    let values = mul(&[*nv, *dv])?;
    let projections = keys
        .checked_add(values)
        .and_then(|x| x.checked_mul(2))
        .ok_or("GDN projection extent overflow")?;
    let ba = mul(&[*nv, 2])?;
    let conv = mul(&[*conv_dim, *kernel_dim])?;
    let state = mul(&[*nv, *dk, *dv])?;
    let role = operation
        .strip_prefix("decode_graph_gdn_")
        .ok_or("GDN producer operation namespace invalid")?;
    let mut weight_descriptor = None;
    let (elements, dtype) = match role {
        "hidden_before_norm" | "residual_before_norm" | "projection_input" => {
            (graph.hidden_size, "bfloat16")
        }
        "input_norm_weight" => (loaded.input_norm_size, "bfloat16"),
        "qkvz_projected" => (projections, "float32"),
        "ba_projected" => (ba, "float32"),
        "conv_input" | "conv_output" => (*conv_dim, "float32"),
        "conv_incoming" | "conv_outgoing" | "conv_weight" => (conv, "float32"),
        "recurrent_incoming" | "recurrent_outgoing" => (state, "float32"),
        "gate_beta" => (ba, "float32"),
        "a_log" | "dt_bias" => (*nv, "float32"),
        "norm_weight" => (*dv, "float32"),
        "gated_output" => (values, "bfloat16"),
        _ => {
            let (name, part) = role.split_once('_').ok_or("unknown GDN producer stage")?;
            let Some(HqqExecutionDescriptor::LinearAttention(hqq)) = &loaded.hqq_exec else {
                return Err("GDN quantized weight capture requires actual HQQ descriptor".into());
            };
            let desc = match name {
                "qkvz" => &hqq.in_proj_qkvz,
                "ba" => &hqq.in_proj_ba,
                "out" => &hqq.out_proj,
                _ => return Err("unknown GDN observed weight".into()),
            };
            if desc.scales_dtype != "float32" || desc.zeros_dtype != "float32" {
                return Err("GDN observed weight scale/zero dtype is not actual FP32".into());
            }
            let (packed, scales, zeros) = crate::matched_probe::hqq_projection_capture_extents(
                desc.rows,
                desc.packed_row_stride_bytes,
                desc.scales_row_stride_bytes,
                desc.zeros_row_stride_bytes,
            )?;
            weight_descriptor = Some(hqq_tensor_exec_json(desc, name));
            match part {
                "packed" => (packed, "uint8"),
                "scales" => (scales / std::mem::size_of::<f32>(), "float32"),
                "zeros" => (zeros / std::mem::size_of::<f32>(), "float32"),
                _ => return Err("unknown GDN observed weight plane".into()),
            }
        }
    };
    let bytes = elements
        .checked_mul(match dtype {
            "bfloat16" => 2,
            "float32" => 4,
            "uint8" => 1,
            _ => return Err("GDN observed dtype invalid".into()),
        })
        .ok_or("GDN producer byte extent overflow")?;
    if bytes > crate::matched_probe::capture_maximum_bytes(operation, layer, None)? {
        return Err("GDN producer observation exceeds explicit actual stage byte bound".into());
    }
    Ok((
        elements,
        dtype,
        serde_json::json!({"num_k_heads":nk,"num_v_heads":nv,"key_dim":dk,"value_dim":dv,"head_ratio":hr,"conv_kernel_dim":kernel_dim,"conv_dim":conv_dim,"q_scale":scale,"norm_eps":graph.eps,"hidden_size":graph.hidden_size,"weight_descriptor":weight_descriptor,"gate_beta_layout":"gate[heads], beta[heads]; gate is exponentiated decay, not raw log gate","scope":"Actual producer boundary; matching inputs must be established before operation attribution. No arithmetic change or serving qualification."}),
    ))
}
impl GpuDecodeStore {
    #[inline]
    fn observe_gdn_producer(
        &self,
        graph: &GpuDecodeGraph,
        layer: usize,
        point: GdnProducerPoint,
        frame: GdnProducerFrame,
        stream: cuda_sys::CUstream,
    ) -> Result<(), String> {
        if !graph
            .matched_graph_stages
            .iter()
            .any(|s| s.layer == layer && s.operation.starts_with("decode_graph_gdn_"))
        {
            return Ok(());
        }
        let loaded = graph
            .layers
            .get(layer)
            .ok_or("GDN actual observed layer absent")?;
        let GpuAttnConfig::LinearAttention {
            conv_state_ptr,
            recur_state_ptr,
            conv_weight_ptr,
            a_log_ptr,
            dt_bias_ptr,
            norm_weight_ptr,
            ..
        } = &loaded.attn
        else {
            return Err("GDN observed source is not a scalar layer".into());
        };
        if let GdnProducerPoint::BeforeNorm(has_residual) = point {
            for stage in graph
                .matched_graph_stages
                .iter()
                .filter(|s| s.layer == layer && s.operation.starts_with("decode_graph_gdn_"))
            {
                if !has_residual && stage.operation == "decode_graph_gdn_residual_before_norm" {
                    return Err(
                        "GDN requested unused residual storage; it is not a valid producer operand"
                            .into(),
                    );
                }
                let flag = match frame {
                    GdnProducerFrame::Captured(_) => &stage.gdn_graph_has_residual,
                    GdnProducerFrame::Eager(_) => &stage.gdn_eager_has_residual,
                };
                flag.store(has_residual as usize, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let roles: &[&str] = match point {
            GdnProducerPoint::BeforeNorm(true) => &[
                "hidden_before_norm",
                "residual_before_norm",
                "input_norm_weight",
            ],
            GdnProducerPoint::BeforeNorm(false) => &["hidden_before_norm", "input_norm_weight"],
            GdnProducerPoint::Projection => &[
                "projection_input",
                "qkvz_projected",
                "ba_projected",
                "qkvz_packed",
                "qkvz_scales",
                "qkvz_zeros",
                "ba_packed",
                "ba_scales",
                "ba_zeros",
            ],
            GdnProducerPoint::BeforeConv => &[
                "conv_input",
                "conv_incoming",
                "recurrent_incoming",
                "conv_weight",
                "a_log",
                "dt_bias",
                "norm_weight",
            ],
            GdnProducerPoint::AfterConv => &["conv_output", "conv_outgoing"],
            GdnProducerPoint::GateBeta => &["gate_beta"],
            GdnProducerPoint::AfterRecurrence => &[
                "gated_output",
                "recurrent_outgoing",
                "out_packed",
                "out_scales",
                "out_zeros",
            ],
        };
        for &role in roles {
            let operation = format!("decode_graph_gdn_{role}");
            let Some(stage) = graph
                .matched_graph_stages
                .iter()
                .find(|s| s.layer == layer && s.operation == operation)
            else {
                continue;
            };
            let pointer = match role {
                "hidden_before_norm" | "projection_input" => *graph.d_hidden.device_ptr(),
                "residual_before_norm" => *graph.d_residual.device_ptr(),
                "input_norm_weight" => loaded.input_norm_ptr,
                "qkvz_projected" | "conv_output" => *graph.d_la_qkvz.device_ptr(),
                "ba_projected" => *graph.d_la_ba.device_ptr(),
                "conv_input" | "gate_beta" => *graph.d_la_conv_out.device_ptr(),
                "conv_incoming" | "conv_outgoing" => *conv_state_ptr,
                "recurrent_incoming" | "recurrent_outgoing" => *recur_state_ptr,
                "conv_weight" => *conv_weight_ptr,
                "a_log" => *a_log_ptr,
                "dt_bias" => *dt_bias_ptr,
                "norm_weight" => *norm_weight_ptr,
                "gated_output" => *graph.d_scratch.device_ptr(),
                _ => {
                    let (name, part) = role.split_once('_').ok_or("GDN weight role invalid")?;
                    let Some(HqqExecutionDescriptor::LinearAttention(hqq)) = &loaded.hqq_exec
                    else {
                        return Err("GDN actual quantized projection descriptor absent".into());
                    };
                    let weight = match name {
                        "qkvz" => &hqq.in_proj_qkvz,
                        "ba" => &hqq.in_proj_ba,
                        "out" => &hqq.out_proj,
                        _ => return Err("GDN actual projection role invalid".into()),
                    };
                    match part {
                        "packed" => weight.packed_ptr,
                        "scales" => weight.scales_ptr,
                        "zeros" => weight.zeros_ptr,
                        _ => return Err("GDN actual weight plane invalid".into()),
                    }
                }
            };
            match frame {
                GdnProducerFrame::Captured(segment) => MatchedGraphStageBuffer::record(
                    graph,
                    layer,
                    &operation,
                    pointer,
                    stage.buffer.bytes,
                    segment,
                    stream,
                )?,
                GdnProducerFrame::Eager(height) => MatchedGraphStageBuffer::record_eager(
                    graph,
                    layer,
                    &operation,
                    pointer,
                    stage.buffer.bytes,
                    height,
                    stream,
                )?,
            }
        }
        Ok(())
    }
}
