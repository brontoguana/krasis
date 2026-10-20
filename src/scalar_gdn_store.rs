// Included in gpu_decode, so all geometry comes from the actually loaded graph.
fn scalar_gdn_projection_layout(
    name: &str,
    desc: &HqqTensorExecDescriptor,
) -> Result<crate::scalar_gdn_projection::Layout, String> {
    validate_hqq_tensor_desc_for_layout(name, desc, 4, hqq_expected_layout(4, "decode")?, true)?;
    if !desc.decode_sidecars.is_empty() {
        return Err(format!(
            "canonical scalar producer does not admit unqualified exception sidecars for {name}"
        ));
    }
    let layout = crate::scalar_gdn_projection::Layout {
        packed: desc.packed_ptr,
        scales: desc.scales_ptr,
        zeros: desc.zeros_ptr,
        rows: desc.rows,
        cols: desc.cols,
        group: desc.group_size,
        packed_stride: desc.packed_row_stride_bytes,
        scales_stride: desc.scales_row_stride_bytes,
        zeros_stride: desc.zeros_row_stride_bytes,
    };
    layout.validate()?;
    Ok(layout)
}
fn scalar_gdn_loaded_geometry(layer: &GpuDecodeLayer) -> Option<(usize, usize, usize, u64)> {
    if let Some(HqqExecutionDescriptor::LinearAttention(loaded)) = &layer.hqq_exec {
        return Some((
            loaded.num_v_heads,
            loaded.k_head_dim,
            loaded.v_head_dim,
            loaded.recur_state_ptr,
        ));
    }
    match &layer.attn {
        GpuAttnConfig::LinearAttention {
            nv,
            dk,
            dv,
            recur_state_ptr,
            ..
        } => Some((*nv, *dk, *dv, *recur_state_ptr)),
        _ => None,
    }
}
fn scalar_gdn_loaded_conv(layer: &GpuDecodeLayer) -> Result<(u64, usize), String> {
    let (pointer, dim, kernel) =
        if let Some(HqqExecutionDescriptor::LinearAttention(x)) = &layer.hqq_exec {
            (x.conv_state_ptr, x.conv_dim, x.kernel_dim)
        } else if let GpuAttnConfig::LinearAttention {
            conv_state_ptr,
            conv_dim,
            kernel_dim,
            ..
        } = &layer.attn
        {
            (*conv_state_ptr, *conv_dim, *kernel_dim)
        } else {
            return Err("scalar actual convolution descriptor absent".into());
        };
    let bytes = dim
        .checked_mul(kernel)
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .ok_or("scalar actual convolution extent overflow")?;
    if pointer == 0 || bytes == 0 {
        return Err("scalar actual convolution allocation absent".into());
    }
    Ok((pointer, bytes))
}
impl GpuDecodeStore {
    fn initialize_scalar_gdn_storage(&mut self) -> Result<(), String> {
        let graph = self.graph.as_ref().ok_or("scalar actual graph absent")?;
        if graph
            .layers
            .iter()
            .any(|layer| matches!(layer.attn, GpuAttnConfig::KimiDeltaAttention(_)))
        {
            return Err("private native64 scalar candidate has not qualified mixed scalar/vector KDA execution".into());
        }
        if self.scalar_gdn_initialized || self.selected_scalar_gdn.is_some() {
            return Err("scalar storage already initialized".into());
        }
        let mut layers = Vec::new();
        let mut producers = std::collections::BTreeMap::new();
        for (index, layer) in graph.layers.iter().take(graph.num_layers).enumerate() {
            let Some((heads, key, value, pointer)) = scalar_gdn_loaded_geometry(layer) else {
                continue;
            };
            // Auxiliary layer placeholders have no actual state allocation and
            // never execute. They must not acquire synthetic owners or budgets.
            if pointer == 0 {
                continue;
            }
            let layout = crate::scalar_gdn_layout::Layout::checked(heads, key, value, 1)?;
            let actual = self.scalar_gdn_latest.get(&index).ok_or_else(|| {
                format!("scalar layer {index} has no retained actual latest tensor")
            })?;
            let registered = self
                .sequence_state_registry
                .allocations()
                .iter()
                .find(|region| {
                    region.layer_idx == Some(index)
                        && region.ptr == pointer
                        && region.kind == "linear_attention_recurrent_state"
                })
                .ok_or_else(|| {
                    format!("scalar layer {index} latest missing from actual sequence inventory")
                })?;
            if registered.storage_bytes != layout.persistent["boundary-state.f32"]
                || registered.dtype != "float32"
                || registered.element_size != std::mem::size_of::<f32>()
                || registered.device_ordinal != self.device.ordinal()
            {
                return Err(format!(
                    "scalar layer {index} actual latest registry metadata differs"
                ));
            }
            let conv = self
                .scalar_gdn_conv
                .get(&index)
                .ok_or("scalar actual convolution tensor owner absent")?;
            let (conv_pointer, conv_bytes) = scalar_gdn_loaded_conv(layer)?;
            let registered_conv = self
                .sequence_state_registry
                .allocations()
                .iter()
                .find(|region| {
                    region.layer_idx == Some(index)
                        && region.ptr == conv_pointer
                        && region.kind == "linear_attention_conv_state"
                })
                .ok_or("scalar actual convolution missing from sequence inventory")?;
            if registered_conv.storage_bytes != conv_bytes
                || registered_conv.element_size != std::mem::size_of::<f32>()
                || registered_conv.dtype != "float32"
                || registered_conv.device_ordinal != self.device.ordinal()
            {
                return Err("scalar actual convolution registry differs".into());
            }
            layers.push(crate::scalar_gdn_runtime::ColdLayer {
                layer: index,
                layout,
                latest: crate::scalar_gdn_owner::LatestState::model(actual.clone())?,
                conv: conv.clone(),
            });
            let Some(HqqExecutionDescriptor::LinearAttention(hqq)) = &layer.hqq_exec else {
                return Err(
                    "native scalar producer requires canonical INT4 attention descriptors".into(),
                );
            };
            if hqq.nbits != 4 || !hqq_linear_attention_decode_ready(hqq) {
                return Err(
                    "native scalar producer requires admitted canonical INT4 attention".into(),
                );
            }
            producers.insert(
                index,
                crate::scalar_gdn_producer::Layer {
                    geometry: crate::scalar_gdn_producer::Geometry {
                        nk: hqq.num_k_heads,
                        nv: hqq.num_v_heads,
                        dk: hqq.k_head_dim,
                        dv: hqq.v_head_dim,
                        hr: hqq.head_ratio,
                        kernel: hqq.kernel_dim,
                        scale: hqq.scale,
                        eps: graph.eps,
                    },
                    qkvz: scalar_gdn_projection_layout("in_proj_qkvz", &hqq.in_proj_qkvz)?,
                    ba: scalar_gdn_projection_layout("in_proj_ba", &hqq.in_proj_ba)?,
                    out: scalar_gdn_projection_layout("out_proj", &hqq.out_proj)?,
                    conv: conv.clone(),
                    conv_weight: hqq.conv_weight_ptr,
                    a_log: hqq.a_log_ptr,
                    dt_bias: hqq.dt_bias_ptr,
                    norm_weight: hqq.norm_weight_ptr,
                },
            );
        }
        if layers.len() != self.scalar_gdn_latest.len()
            || layers.len() != self.scalar_gdn_conv.len()
        {
            return Err(
                "scalar retained tensor inventory contains an unexecuted/unregistered layer".into(),
            );
        }
        if layers.is_empty() {
            // A real auxiliary split can own no scalar layers. It needs no
            // synthetic state, FLA context, workspace or allocation budget.
            self.scalar_gdn_initialized = true;
            return Ok(());
        }
        // Both native phases use the same canonical projection program. Its
        // retained pointers must address canonical coefficients in BOTH stages.
        let native_layers = layers.iter().map(|layer| layer.layer).collect();
        self.select_scalar_canonical_prefill_formats(&native_layers)?;
        let sidecar = crate::scalar_gdn_select::sidecar(&self.device)?;
        let mut storage = crate::scalar_gdn_runtime::Storage::new(
            self.device.clone(),
            None,
            layers,
            &sidecar,
            include_str!(concat!(env!("OUT_DIR"), "/scalar_gdn_transport.ptx")),
        )?;
        let producer = crate::scalar_gdn_producer::Producer::new(self.device.clone(), producers)?;
        Arc::get_mut(&mut storage)
            .ok_or("scalar cold storage unexpectedly shared during producer attachment")?
            .attach_producer(producer)?;
        let additions = storage.registry().allocations();
        self.sequence_state_registry = self
            .sequence_state_registry
            .with_validated_inventory(additions, additions)?;
        self.selected_scalar_gdn = Some(storage);
        self.scalar_gdn_latest.clear();
        self.scalar_gdn_conv.clear();
        self.scalar_gdn_initialized = true;
        Ok(())
    }
}

impl GpuDecodeStore {
    /// Cold selected-layer policy, never an alternate runtime path. Ordinary
    /// tensors keep their production formats. The already-qualified shared
    /// native projector requires the exact canonical coefficients in both
    /// phases, in the SAME stable VMM slots; no duplicate GPU bank is allocated.
    fn select_scalar_canonical_prefill_formats(
        &mut self,
        layers: &std::collections::BTreeSet<usize>,
    ) -> Result<(), String> {
        if layers.is_empty() {
            return Err("canonical scalar prefill policy requires actual executed layers".into());
        }
        let graph = self
            .graph
            .as_ref()
            .ok_or("canonical scalar prefill policy requires configure")?;
        if graph.graphs_ever_captured
            || !graph.per_layer_graphs.is_empty()
            || graph.hot_hcs_full_graph.is_some()
        {
            return Err(
                "canonical scalar prefill policy must be finalized before any graph capture".into(),
            );
        }
        let mut indices = Vec::new();
        for &layer in layers {
            for name in ["in_proj_qkvz", "in_proj_ba", "out_proj"] {
                let found = self
                    .hqq_runtime_slots
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.layer_idx == layer && slot.tensor_name == name)
                    .collect::<Vec<_>>();
                if found.len() != 1 {
                    return Err(format!("canonical scalar prefill requires one actual slot:layer={layer},tensor={name},found={}",found.len()));
                }
                let (index, slot) = found[0];
                if slot.nbits != 4
                    || !matches!(slot.decode.kind, HqqRuntimeFormatKind::DecodeRowAlignedV1)
                    || !slot.decode.kernel_ready
                    || slot.decode.layout != hqq_expected_layout(4, "decode")?
                {
                    return Err(format!("canonical scalar prefill requires the qualified INT4 canonical descriptor:layer={layer},tensor={name}"));
                }
                indices.push(index);
            }
        }
        // Stage swaps are a whole-Store contract. A selected descriptor must
        // not be relabelled while mixed physical stages are active.
        let stages = self
            .hqq_runtime_slots
            .iter()
            .filter(|slot| slot.is_registered())
            .map(|slot| slot.current_stage.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if stages.len() > 1 {
            return Err(
                "canonical scalar prefill selection encountered mixed actual slot stages".into(),
            );
        }
        let reload_prefill = stages.contains("prefill");
        drop(stages);
        self.device
            .bind_to_thread()
            .map_err(|e| format!("canonical scalar prefill selection context:{e}"))?;
        unsafe { cuda_sys::lib().cuCtxSynchronize() }
            .result()
            .map_err(|e| format!("canonical scalar prefill old-stage completion:{e}"))?;
        for index in indices {
            let started = std::time::Instant::now();
            let slot = &mut self.hqq_runtime_slots[index];
            let canonical = &slot.decode;
            let mut selected = HqqRuntimeFormatDescriptor {
                stage: "prefill",
                kind: HqqRuntimeFormatKind::PrefillCanonicalRowAlignedV1,
                layout: canonical.layout.clone(),
                packed_dtype: canonical.packed_dtype.clone(),
                scales_dtype: canonical.scales_dtype.clone(),
                zeros_dtype: canonical.zeros_dtype.clone(),
                packed_host: canonical.packed_host.clone(),
                scales_host: canonical.scales_host.clone(),
                zeros_host: canonical.zeros_host.clone(),
                packed_host_pin: None,
                scales_host_pin: None,
                zeros_host_pin: None,
                packed_component_bytes: canonical.packed_component_bytes,
                scales_component_bytes: canonical.scales_component_bytes,
                zeros_component_bytes: canonical.zeros_component_bytes,
                row_blocks: canonical.row_blocks,
                rows_per_tile: canonical.rows_per_tile,
                packed_row_stride_bytes: canonical.packed_row_stride_bytes,
                scales_row_stride_bytes: canonical.scales_row_stride_bytes,
                zeros_row_stride_bytes: canonical.zeros_row_stride_bytes,
                build_ms: slot.prefill.build_ms,
                kernel_ready: canonical.kernel_ready,
            };
            if self.hqq_stage_pin_host {
                selected.pin_host_components(&format!(
                    "native scalar canonical prefill layer={} tensor={}",
                    slot.layer_idx, slot.tensor_name
                ))?;
            }
            selected.build_ms += started.elapsed().as_secs_f64() * 1000.;
            slot.prefill = selected;
            if slot.current_stage == "prefill" {
                slot.current_stage = "registered_unloaded".into();
            }
        }
        self.recompute_hqq_runtime_staging_stats();
        // Preserve the caller's actual physical phase. The production swap
        // computes all component/remap extents from the new real descriptors.
        if reload_prefill {
            self.swap_hqq_runtime_stage_rust("prefill")?;
        }
        Ok(())
    }
}
