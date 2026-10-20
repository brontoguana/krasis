// Included in gpu_prefill. Cold setup is Python; every operation here is native.
impl PrefillEngine {
    // Validate the executor selected by la_gemm. Canonical bytes are never
    // admitted to the legacy Marlin executor.
    fn validate_linear_attention_projection_bindings(
        &self,
        layer: usize,
        hqq: &HqqLinearAttentionPrefillDescriptor,
    ) -> Result<(), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = &self.selected_scalar_gdn {
            use crate::{scalar_gdn_producer::Role, scalar_gdn_projection::Binding};
            if hqq.nbits != 4
                || self
                    .layer_weights
                    .get(layer)
                    .ok_or("native projection layer absent")?
                    .la_recur_state_ptr
                    == 0
            {
                return Err(
                    "native projection requires its actual INT4 linear-attention layer".into(),
                );
            }
            let producer = storage.producer()?;
            for (name, role, desc) in [
                ("in_proj_qkvz", Role::Qkvz, &hqq.in_proj_qkvz),
                ("in_proj_ba", Role::Ba, &hqq.in_proj_ba),
                ("out_proj", Role::Out, &hqq.out_proj),
            ] {
                let retained = producer.projection_layout(layer, role)?;
                retained
                    .validate_live_binding(&Binding {
                        layout: &desc.layout,
                        packed_dtype: &desc.packed_dtype,
                        scales_dtype: &desc.scales_dtype,
                        zeros_dtype: &desc.zeros_dtype,
                        pointers: [desc.packed_ptr, desc.scales_ptr, desc.zeros_ptr],
                        shape: [desc.rows, desc.cols, desc.group_size],
                        axis: desc.axis,
                        tensor_bytes: desc.tensor_bytes,
                    })
                    .map_err(|error| {
                        format!("native projection layer {layer} tensor {name}: {error}")
                    })?;
            }
            return Ok(());
        }
        self.validate_hqq_quantized_prefill_tensor_desc("in_proj_qkvz", &hqq.in_proj_qkvz)?;
        self.validate_hqq_quantized_prefill_tensor_desc("in_proj_ba", &hqq.in_proj_ba)?;
        self.validate_hqq_quantized_prefill_tensor_desc("out_proj", &hqq.out_proj)?;
        Ok(())
    }
    #[cfg(feature = "scalar-gdn-test")]
    fn scalar_native_projection(
        &self,
        layer: usize,
        name: &str,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<(), String> {
        use crate::{cuda_region::CudaRegion, scalar_gdn_owner::Plane, scalar_gdn_producer::Role};
        let producer = self
            .selected_scalar_gdn
            .as_ref()
            .ok_or("scalar native prefill storage absent")?
            .producer()?;
        let role = match name {
            "in_proj_qkvz" => Role::Qkvz,
            "in_proj_ba" => Role::Ba,
            "out_proj" => Role::Out,
            _ => return Err("unknown scalar native projection role".into()),
        };
        let layout = producer.projection_layout(layer, role)?;
        let (input_owner, output_owner): (&dyn CudaRegion, &dyn CudaRegion) = match role {
            Role::Qkvz | Role::Ba => (
                &self.scratch.d_hidden,
                self.scratch
                    .d_la_proj_buf
                    .as_ref()
                    .ok_or("scalar actual BF16 projection buffer absent")?,
            ),
            Role::Out => (
                self.scratch
                    .d_la_v
                    .as_ref()
                    .ok_or("scalar actual gated-normalization buffer absent")?,
                &self.scratch.d_scratch1,
            ),
        };
        fn view<'a>(
            owner: &'a dyn CudaRegion,
            ptr: u64,
            elements: usize,
        ) -> Result<Plane<'a, u16>, String> {
            let (base, _) = CudaRegion::extent(owner)?;
            let bytes = ptr
                .checked_sub(base)
                .ok_or("scalar projection view precedes actual allocation")?;
            if bytes % std::mem::size_of::<u16>() as u64 != 0 {
                return Err("scalar projection BF16 view misaligned".into());
            }
            // The existing hidden/projection/postnorm CUDA writers produce
            // BF16 here, including postnorm's larger real F32 backing buffer.
            unsafe {
                Plane::from_region(
                    owner,
                    usize::try_from(bytes / std::mem::size_of::<u16>() as u64)
                        .map_err(|e| e.to_string())?,
                    elements,
                )
            }
        }
        let input = view(
            input_owner,
            input,
            rows.checked_mul(layout.cols)
                .ok_or("scalar prefill input extent overflow")?,
        )?;
        let output = view(
            output_owner,
            output,
            rows.checked_mul(layout.rows)
                .ok_or("scalar prefill output extent overflow")?,
        )?;
        unsafe { producer.project(layer, role, &input, &output, rows, self.stream) }
    }
    fn scalar_native_selected(&self) -> bool {
        #[cfg(feature = "scalar-gdn-test")]
        {
            self.selected_scalar_gdn.is_some()
        }
        #[cfg(not(feature = "scalar-gdn-test"))]
        {
            false
        }
    }
    #[cfg(feature = "scalar-gdn-test")]
    fn scalar_native_prefill_output(
        &self,
        layer: usize,
        rows: usize,
        start: usize,
    ) -> Result<u64, String> {
        use crate::scalar_gdn_owner::{Inputs, Plane};
        let storage = self
            .selected_scalar_gdn
            .as_ref()
            .ok_or("scalar native prefill storage absent")?;
        let workspace = unsafe { storage.prepare_prefill(layer, rows)? };
        let lw = &self.layer_weights[layer];
        let heads = self.config.la_num_v_heads;
        let key = self.config.la_k_head_dim;
        let value = self.config.la_v_head_dim;
        let gates = rows
            .checked_mul(heads)
            .ok_or("scalar actual gate extent overflow")?;
        let keys = gates
            .checked_mul(key)
            .ok_or("scalar actual key extent overflow")?;
        let values = gates
            .checked_mul(value)
            .ok_or("scalar actual value extent overflow")?;
        let raw = &self.scratch.d_fp32_scratch;
        if raw.len() < gates {
            return Err("scalar actual raw gate scratch too short".into());
        }
        // Same actual FP32 gate producer as the retained native169 operands.
        // Its unused FP32 beta output uses the real A arena before native KKT
        // clears/reuses it. This preserves raw projection operands for capture.
        let mut args = [
            workspace.buffer("A.f32")?.device_ptr().to_owned(),
            *raw.device_ptr(),
            *self
                .scratch
                .d_la_b
                .as_ref()
                .ok_or("scalar actual b projection absent")?
                .device_ptr(),
            *self
                .scratch
                .d_la_a
                .as_ref()
                .ok_or("scalar actual a projection absent")?
                .device_ptr(),
            lw.la_a_log_ptr,
            lw.la_dt_bias_ptr,
        ];
        let mut h = i32::try_from(heads).map_err(|e| e.to_string())?;
        let threads = std::cmp::max(32, heads.min(1024).div_ceil(32) * 32) as u32;
        let mut addresses = args
            .iter_mut()
            .map(|p| (p as *mut u64).cast::<std::ffi::c_void>())
            .collect::<Vec<_>>();
        addresses.push((&mut h as *mut i32).cast());
        unsafe {
            cuda_sys::lib()
                .cuLaunchKernel(
                    self.kernels.la_compute_gate_beta.0,
                    rows as u32,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    addresses.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
                .result()
                .map_err(|e| format!("scalar actual raw gate producer:{e}"))?;
        }
        // These concrete GpuBuf owners supply actual bounds/context. The
        // producer above and the existing conv/normalization kernels prove
        // BF16 representation in the reused F32 backing for V and beta.
        let inputs = unsafe {
            Inputs {
                q: Plane::from_region(&self.scratch.d_scratch1, 0, keys)?,
                k: Plane::from_region(self.scratch.d_scratch2.as_ref(), 0, keys)?,
                v: Plane::from_region(
                    self.scratch
                        .d_la_v
                        .as_ref()
                        .ok_or("scalar actual V absent")?,
                    0,
                    values,
                )?,
                beta: Plane::from_region(
                    self.scratch
                        .d_la_beta
                        .as_ref()
                        .ok_or("scalar actual beta absent")?,
                    0,
                    gates,
                )?,
                gate: Plane::from_region(raw, 0, gates)?,
            }
        };
        self.observe_gdn_prefill_point(layer, rows, start, GdnPrefillPoint::Prepared)?;
        unsafe {
            storage.prefill_output(
                layer,
                u64::try_from(start).map_err(|e| e.to_string())?,
                rows,
                &inputs,
            )
        }
    }
}
