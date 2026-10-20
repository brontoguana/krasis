// Bounded actual producer windows. No device allocation or changed arithmetic.
enum GdnPrefillPoint {
    BeforeNorm(bool),
    ConvBefore,
    ConvAfter,
    Prepared,
}
impl PrefillEngine {
    fn write_gdn_prefill_window(
        &self,
        target: crate::matched_probe::Target,
        operation: &str,
        layer: usize,
        m: usize,
        start: usize,
        planes: &[(&str, u64, usize, usize)],
        constants: &[(&str, u64, usize)],
        semantics: serde_json::Value,
    ) -> Result<(), String> {
        let (count, maximum) =
            crate::matched_probe::prefill_window_parameters(operation, layer, m)?;
        if count == 0 || target.row.checked_add(count).is_none_or(|end| end > m) {
            return Err("GDN actual prefill window exceeds real chunk".into());
        }
        let mut total = 0usize;
        let mut captures = Vec::new();
        for &(name, pointer, width, element_bytes) in planes {
            let stride = width
                .checked_mul(element_bytes)
                .ok_or("GDN actual producer stride overflow")?;
            let offset = target
                .row
                .checked_mul(stride)
                .ok_or("GDN actual producer offset overflow")?;
            let bytes = count
                .checked_mul(stride)
                .ok_or("GDN actual producer window overflow")?;
            let pointer = pointer
                .checked_add(offset as u64)
                .ok_or("GDN producer window address overflow")?;
            captures.push((name, pointer, bytes));
        }
        captures.extend_from_slice(constants);
        for &(_, pointer, bytes) in &captures {
            if pointer == 0 || bytes == 0 || pointer.checked_add(bytes as u64).is_none() {
                return Err("invalid actual GDN producer source extent".into());
            }
            total = total
                .checked_add(bytes)
                .ok_or("GDN actual producer bundle overflow")?;
        }
        if total > maximum {
            return Err(format!("GDN actual producer window requires {total} bytes, exceeds explicit bound {maximum}"));
        }
        for &(name, pointer, bytes) in &captures {
            target.device(name, pointer, bytes, self.stream)?;
        }
        target.write("window.json",serde_json::to_vec_pretty(&serde_json::json!({
            "operation":operation,"layer":layer,"chunk_rows":m,"absolute_chunk_start":start,"local_first_row":target.row,
            "absolute_first_row":start.checked_add(target.row).ok_or("GDN absolute row overflow")?,"window_rows":count,
            "actual_payload_bytes":total,"maximum_bytes":maximum,"semantics":semantics,
            "planes":planes.iter().map(|&(name,_,width,element_bytes)|serde_json::json!({"name":name,"row_elements":width,"element_bytes":element_bytes})).collect::<Vec<_>>(),
            "constants":constants.iter().map(|&(name,_,bytes)|serde_json::json!({"name":name,"actual_bytes":bytes})).collect::<Vec<_>>(),
            "device_diagnostic_allocation_bytes":0,"arithmetic_changed":false,
            "scope":"Actual live producer windows; compare real consumed IDs, hidden operands and weights before attributing first different operation. Unused state is not causal evidence."
        })).map_err(|e|e.to_string())?.as_slice())
    }
    fn observe_gdn_prefill_projection(
        &self,
        layer: usize,
        name: &str,
        m: usize,
        start: usize,
        input: u64,
        output: u64,
        hqq: Option<&HqqTensorExecDescriptor>,
    ) -> Result<(), String> {
        let operation = match name {
            "in_proj_qkvz" => "gdn_prefill_qkvz",
            "in_proj_ba" => "gdn_prefill_ba",
            "out_proj" => "gdn_prefill_out",
            _ => return Err("unknown GDN observed projection".into()),
        };
        let Some(target) = crate::matched_probe::target(operation, layer, m)? else {
            return Ok(());
        };
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = &self.selected_scalar_gdn {
            let producer = storage.producer()?;
            let role = match name {
                "in_proj_qkvz" => crate::scalar_gdn_producer::Role::Qkvz,
                "in_proj_ba" => crate::scalar_gdn_producer::Role::Ba,
                "out_proj" => crate::scalar_gdn_producer::Role::Out,
                _ => return Err("unknown scalar observed projection".into()),
            };
            let actual = producer.projection_layout(layer, role)?;
            let packed = actual
                .rows
                .checked_mul(actual.packed_stride)
                .ok_or("actual canonical packed observer extent overflow")?;
            let scales = actual
                .rows
                .checked_mul(actual.scales_stride)
                .ok_or("actual canonical scale observer extent overflow")?;
            let zeros = actual
                .rows
                .checked_mul(actual.zeros_stride)
                .ok_or("actual canonical zero observer extent overflow")?;
            return self.write_gdn_prefill_window(target,operation,layer,m,start,&[("input.bf16",input,actual.cols,2),("projected.bf16",output,actual.rows,2)],&[("packed.bin",actual.packed,packed),("scales.f32",actual.scales,scales),("zeros.f32",actual.zeros,zeros)],
                serde_json::json!({"name":name,"rows":actual.rows,"cols":actual.cols,"nbits":4,"group_size":actual.group,"actual_arithmetic_backend":producer.identity,"actual_execution_weight_payloads":{"format":"canonical_row_aligned_u4_f32","packed_bytes":packed,"scales_bytes":scales,"zeros_bytes":zeros,"packed_stride_bytes":actual.packed_stride,"scales_stride_bytes":actual.scales_stride,"zeros_stride_bytes":actual.zeros_stride},"input_dtype":"bfloat16","output_dtype":"bfloat16","hqq_prefill_materialize_bf16":self.config.hqq_prefill_materialize_bf16}));
        }
        let desc =
            hqq.ok_or("bounded GDN projection packet requires actual loaded HQQ descriptors")?;
        let bits = self.validate_hqq_quantized_prefill_tensor_desc(name, desc)?;
        let payload = crate::hqq_observation_layout::describe(
            desc.rows,
            desc.cols,
            desc.group_size,
            bits,
            &desc.layout,
            &desc.packed_dtype,
            &desc.scales_dtype,
            &desc.zeros_dtype,
            desc.tensor_bytes,
        )?;
        let constants = [
            ("packed.bin", desc.packed_ptr, payload.packed),
            (payload.scales_file, desc.scales_ptr, payload.scales),
            (payload.zeros_file, desc.zeros_ptr, payload.zeros),
        ];
        self.write_gdn_prefill_window(target,operation,layer,m,start,&[("input.bf16",input,desc.cols,2),("projected.bf16",output,desc.rows,2)],&constants,
            serde_json::json!({"name":name,"rows":desc.rows,"cols":desc.cols,"nbits":bits,"group_size":desc.group_size,"axis":desc.axis,"layout":desc.layout,"packed_dtype":desc.packed_dtype,"scales_dtype":desc.scales_dtype,"zeros_dtype":desc.zeros_dtype,"original_dtype":desc.original_dtype,"tensor_bytes":desc.tensor_bytes,"actual_execution_weight_payloads":payload.metadata,"input_dtype":"bfloat16","output_dtype":"bfloat16","hqq_prefill_materialize_bf16":self.config.hqq_prefill_materialize_bf16}))
    }
    fn observe_gdn_prefill_point(
        &self,
        layer: usize,
        m: usize,
        start: usize,
        point: GdnPrefillPoint,
    ) -> Result<(), String> {
        let operation = match point {
            GdnPrefillPoint::BeforeNorm(_) => "gdn_prefill_before_norm",
            GdnPrefillPoint::ConvBefore => "gdn_prefill_conv_before",
            GdnPrefillPoint::ConvAfter => "gdn_prefill_conv_after",
            GdnPrefillPoint::Prepared => "gdn_prefill_prepared",
        };
        let Some(target) = crate::matched_probe::target(operation, layer, m)? else {
            return Ok(());
        };
        let lw = self
            .layer_weights
            .get(layer)
            .ok_or("GDN observed prefill layer absent")?;
        if self.config.layer_types.get(layer) != Some(&3) {
            return Err(
                "GDN producer observation requires actual scalar linear-attention layer".into(),
            );
        }
        let cfg = &self.config;
        let mul = |v: &[usize]| {
            v.iter().try_fold(1usize, |n, &x| {
                if x == 0 {
                    Err("empty GDN observed extent")
                } else {
                    n.checked_mul(x).ok_or("GDN observed extent overflow")
                }
            })
        };
        let keys = mul(&[cfg.la_num_k_heads, cfg.la_k_head_dim])?;
        let normalized_keys = mul(&[cfg.la_num_v_heads, cfg.la_k_head_dim])?;
        let values = mul(&[cfg.la_num_v_heads, cfg.la_v_head_dim])?;
        let conv = mul(&[cfg.la_conv_dim, cfg.la_conv_kernel_dim, 4])?;
        let pointer = |name: &str| -> Result<u64, String> {
            Ok(match name {
                "q.raw.bf16" | "q.normalized.bf16" => *self.scratch.d_scratch1.device_ptr(),
                "k.raw.bf16" | "k.normalized.bf16" => *self.scratch.d_scratch2.device_ptr(),
                "v.raw.bf16" => *self
                    .scratch
                    .d_la_k_beta
                    .as_ref()
                    .ok_or("GDN actual V input absent")?
                    .device_ptr(),
                "q.conv.bf16" => *self
                    .scratch
                    .d_la_q
                    .as_ref()
                    .ok_or("GDN actual Q conv absent")?
                    .device_ptr(),
                "k.conv.bf16" => *self
                    .scratch
                    .d_la_k
                    .as_ref()
                    .ok_or("GDN actual K conv absent")?
                    .device_ptr(),
                "v.conv.bf16" => *self
                    .scratch
                    .d_la_v
                    .as_ref()
                    .ok_or("GDN actual V conv absent")?
                    .device_ptr(),
                "z.bf16" => *self
                    .scratch
                    .d_la_z
                    .as_ref()
                    .ok_or("GDN actual Z absent")?
                    .device_ptr(),
                "a.raw.bf16" => *self
                    .scratch
                    .d_la_a
                    .as_ref()
                    .ok_or("GDN actual A projection absent")?
                    .device_ptr(),
                "b.raw.bf16" => *self
                    .scratch
                    .d_la_b
                    .as_ref()
                    .ok_or("GDN actual B projection absent")?
                    .device_ptr(),
                "beta.bf16" => *self
                    .scratch
                    .d_la_beta
                    .as_ref()
                    .ok_or("GDN actual beta absent")?
                    .device_ptr(),
                "gate.raw.f32" => *self.scratch.d_fp32_scratch.device_ptr(),
                _ => return Err("unknown GDN producer plane".into()),
            })
        };
        let (planes, constants, has_residual) = match point {
            GdnPrefillPoint::BeforeNorm(has) => {
                let mut planes = vec![(
                    "hidden.before_norm.bf16",
                    *self.scratch.d_hidden.device_ptr(),
                    cfg.hidden_size,
                    2,
                )];
                if has {
                    planes.push((
                        "residual.before_norm.bf16",
                        *self.scratch.d_residual.device_ptr(),
                        cfg.hidden_size,
                        2,
                    ));
                }
                (
                    planes,
                    vec![(
                        "input-norm-weight.bf16",
                        lw.input_norm,
                        mul(&[cfg.hidden_size, 2])?,
                    )],
                    Some(has),
                )
            }
            GdnPrefillPoint::ConvBefore => (
                vec![
                    ("q.raw.bf16", pointer("q.raw.bf16")?, keys, 2),
                    ("k.raw.bf16", pointer("k.raw.bf16")?, keys, 2),
                    ("v.raw.bf16", pointer("v.raw.bf16")?, values, 2),
                    ("z.bf16", pointer("z.bf16")?, values, 2),
                    ("a.raw.bf16", pointer("a.raw.bf16")?, cfg.la_num_v_heads, 2),
                    ("b.raw.bf16", pointer("b.raw.bf16")?, cfg.la_num_v_heads, 2),
                ],
                vec![
                    ("conv.incoming.f32", lw.la_conv_state_ptr, conv),
                    ("conv-weight.f32", lw.la_conv_weight_ptr, conv),
                    ("a-log.f32", lw.la_a_log_ptr, mul(&[cfg.la_num_v_heads, 4])?),
                    (
                        "dt-bias.f32",
                        lw.la_dt_bias_ptr,
                        mul(&[cfg.la_num_v_heads, 4])?,
                    ),
                    (
                        "output-norm-weight.f32",
                        lw.la_norm_weight_ptr,
                        mul(&[cfg.la_v_head_dim, 4])?,
                    ),
                    (
                        "recurrent.incoming.f32",
                        lw.la_recur_state_ptr,
                        mul(&[cfg.la_num_v_heads, cfg.la_k_head_dim, cfg.la_v_head_dim, 4])?,
                    ),
                ],
                None,
            ),
            GdnPrefillPoint::ConvAfter => (
                vec![
                    ("q.conv.bf16", pointer("q.conv.bf16")?, keys, 2),
                    ("k.conv.bf16", pointer("k.conv.bf16")?, keys, 2),
                    ("v.conv.bf16", pointer("v.conv.bf16")?, values, 2),
                ],
                vec![("conv.outgoing.f32", lw.la_conv_state_ptr, conv)],
                None,
            ),
            GdnPrefillPoint::Prepared => (
                vec![
                    (
                        "q.normalized.bf16",
                        pointer("q.normalized.bf16")?,
                        normalized_keys,
                        2,
                    ),
                    (
                        "k.normalized.bf16",
                        pointer("k.normalized.bf16")?,
                        normalized_keys,
                        2,
                    ),
                    ("v.conv.bf16", pointer("v.conv.bf16")?, values, 2),
                    ("beta.bf16", pointer("beta.bf16")?, cfg.la_num_v_heads, 2),
                    (
                        "gate.raw.f32",
                        pointer("gate.raw.f32")?,
                        cfg.la_num_v_heads,
                        4,
                    ),
                ],
                vec![],
                None,
            ),
        };
        self.write_gdn_prefill_window(target,operation,layer,m,start,&planes,&constants,serde_json::json!({"num_k_heads":cfg.la_num_k_heads,"num_v_heads":cfg.la_num_v_heads,"key_dim":cfg.la_k_head_dim,"value_dim":cfg.la_v_head_dim,"head_ratio":cfg.la_head_ratio,"conv_kernel_dim":cfg.la_conv_kernel_dim,"conv_dim":cfg.la_conv_dim,"q_scale":cfg.la_scale,"rms_norm_eps":cfg.rms_norm_eps,"has_residual":has_residual,"input_dtype":"bfloat16","raw_gate_dtype":"float32 before cumsum/exponentiation","q_normalized_includes_scale":true}))
    }
}
