// Native Gemma full-history WMMA continuation, using the existing cache contract.
impl PrefillEngine {
    pub(crate) fn gemma4_prepare_attention_replay(&mut self, total_tokens: usize) -> Result<String, String> {
        self.gemma4_ring_replay_completion()?;
        let before = self.scratch.d_fp32_scratch.len * 4;
        let required = total_tokens * 608;
        if self.scratch.d_fp32_scratch.len < required {
            self.scratch.d_fp32_scratch = GpuBuf::<f32>::alloc_zeroed(required)?;
        }
        Ok(serde_json::json!({"before_bytes":before,"workspace_bytes":self.scratch.d_fp32_scratch.len*4,"full_gemm_required_bytes":total_tokens*(512*4+64*6),"window_gemm_required_bytes":total_tokens.min(1024+64+280)*(256*4+64*6)}).to_string())
    }

    pub(crate) fn gemma4_full_history_attention(
        &self,
        layer: usize,
        q: u64,
        k: u64,
        v: u64,
        ck: u64,
        cv: u64,
        out: u64,
        rows: usize,
        prefix: usize,
        cache_dtype: i32,
        scale: f32,
    ) -> Result<(), String> {
        if rows == 0
            || rows > 262144
            || prefix > 262144
            || rows + prefix > 262144
            || q == 0
            || k == 0
            || v == 0
            || out == 0
            || !scale.is_finite()
            || !(0..=1).contains(&cache_dtype)
            || (prefix > 0 && (ck == 0 || cv == 0))
        {
            return Err("Gemma full-history attention contract".into());
        }
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        if std::env::var("KRASIS_GEMMA_FULL_HISTORY_GEMM").as_deref() == Ok("1")
            && self.gemma4_full_gemm_attention(
                q,
                k,
                v,
                ck,
                cv,
                out,
                rows,
                prefix,
                cache_dtype,
                scale,
            )?
        {
            return Ok(());
        }
        let raw = self.gemma4_attention_kernel("gemma4_full_history_wmma")?;
        let smem = 86016usize;
        unsafe {
            cuda_sys::lib().cuFuncSetAttribute(raw.0,cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,smem as i32).result().map_err(|e|format!("Gemma WMMA shared memory {e:?}"))?;
        }
        // Explicit diagnostic readback on real ordinary operands. No allocations
        // or readbacks occur when the capture variable is absent.
        if layer != usize::MAX && prefix > 0 {
            if let Ok(directory) = std::env::var("KRASIS_GEMMA_ATTN_CAPTURE_DIR") {
                use std::io::Write;
                self.stream_sync()?;
                let path = std::path::Path::new(&directory)
                    .join(format!("layer{layer}-prefix{prefix}-rows{rows}"));
                std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
                for (name, ptr, bytes) in [
                    ("q.bf16", q, rows * 16 * 512 * 2),
                    ("k.bf16", k, rows * 2 * 512 * 2),
                    ("v.bf16", v, rows * 2 * 512 * 2),
                    (
                        "ck.raw",
                        ck,
                        prefix * 2 * 512 * if cache_dtype == 1 { 1 } else { 2 },
                    ),
                    (
                        "cv.raw",
                        cv,
                        prefix * 2 * 512 * if cache_dtype == 1 { 1 } else { 2 },
                    ),
                ] {
                    if bytes == 0 {
                        continue;
                    }
                    let mut data = vec![0u8; bytes];
                    unsafe {
                        cuda_sys::lib()
                            .cuMemcpyDtoH_v2(data.as_mut_ptr() as _, ptr, bytes)
                            .result()
                            .map_err(|e| e.to_string())?;
                    }
                    let mut file = std::fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(path.join(name))
                        .map_err(|e| e.to_string())?;
                    file.write_all(&data).map_err(|e| e.to_string())?;
                }
                std::fs::write(path.join("contract.json"),serde_json::to_vec_pretty(&serde_json::json!({"rows":rows,"prefix":prefix,"heads":16,"kv_heads":2,"dim":512,"cache_dtype":cache_dtype,"scale":scale,"sliding_window":0})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            }
        }
        let mut a = (
            out,
            q,
            k,
            v,
            rows as i32,
            16i32,
            2i32,
            scale,
            ck,
            cv,
            prefix as i32,
            cache_dtype,
        );
        let result = unsafe {
            launch(
                raw,
                (((rows + 15) / 16) as u32, 16, 1),
                (32, 1, 1),
                smem as u32,
                self.stream,
                &mut [
                    &mut a.0 as *mut _ as _,
                    &mut a.1 as *mut _ as _,
                    &mut a.2 as *mut _ as _,
                    &mut a.3 as *mut _ as _,
                    &mut a.4 as *mut _ as _,
                    &mut a.5 as *mut _ as _,
                    &mut a.6 as *mut _ as _,
                    &mut a.7 as *mut _ as _,
                    &mut a.8 as *mut _ as _,
                    &mut a.9 as *mut _ as _,
                    &mut a.10 as *mut _ as _,
                    &mut a.11 as *mut _ as _,
                ],
            )
        };
        if layer == usize::MAX {
            let completion = self.stream_sync();
            result.and(completion)
        } else {
            result
        }
    }
}

impl PrefillEngine {
    fn gemma4_full_gemm_attention(
        &self,
        q: u64,
        k: u64,
        v: u64,
        ck: u64,
        cv: u64,
        out: u64,
        rows: usize,
        prefix: usize,
        dtype: i32,
        scale: f32,
    ) -> Result<bool, String> {
        let total = prefix + rows;
        let band = 64usize;
        let needed = total * (512 * 4 + band * 6);
        if needed > self.scratch.d_fp32_scratch.len * 4 {
            return Ok(false);
        }
        let kp = *self.scratch.d_fp32_scratch.device_ptr();
        let vp = kp + (total * 512 * 2) as u64;
        let scores = vp + (total * 512 * 2) as u64;
        let probability = scores + (total * band * 4) as u64;
        let pack = self.gemma4_attention_kernel("gemma4_full_kv_pack")?;
        let softmax = self.gemma4_attention_kernel("gemma4_full_softmax")?;
        unsafe {
            cublas_result::set_stream(self.cublas_handle, self.stream as cublas_sys::cudaStream_t)
                .map_err(|e| e.to_string())?;
            for head in 0..2i32 {
                let mut a = (
                    kp,
                    vp,
                    ck,
                    cv,
                    k,
                    v,
                    total as i32,
                    prefix as i32,
                    head,
                    dtype,
                );
                launch(
                    pack,
                    (((total * 512 + 255) / 256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    self.stream,
                    &mut [
                        &mut a.0 as *mut _ as _,
                        &mut a.1 as *mut _ as _,
                        &mut a.2 as *mut _ as _,
                        &mut a.3 as *mut _ as _,
                        &mut a.4 as *mut _ as _,
                        &mut a.5 as *mut _ as _,
                        &mut a.6 as *mut _ as _,
                        &mut a.7 as *mut _ as _,
                        &mut a.8 as *mut _ as _,
                        &mut a.9 as *mut _ as _,
                    ],
                )?;
                for qh in head as usize * 8..(head as usize + 1) * 8 {
                    for first in (0..rows).step_by(band) {
                        let count = band.min(rows - first);
                        let zero = 0f32;
                        let one = 1f32;
                        cublas_result::gemm_ex(
                            self.cublas_handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            total as i32,
                            count as i32,
                            512,
                            &scale as *const _ as _,
                            kp as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            512,
                            (q + ((first * 16 + qh) * 512 * 2) as u64) as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            8192,
                            &zero as *const _ as _,
                            scores as _,
                            cublas_sys::cudaDataType::CUDA_R_32F,
                            total as i32,
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                        )
                        .map_err(|e| format!("Gemma QK GEMM: {e:?}"))?;
                        let mut a = (scores, probability, total as i32, (prefix + first) as i32);
                        launch(
                            softmax,
                            (count as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            self.stream,
                            &mut [
                                &mut a.0 as *mut _ as _,
                                &mut a.1 as *mut _ as _,
                                &mut a.2 as *mut _ as _,
                                &mut a.3 as *mut _ as _,
                            ],
                        )?;
                        cublas_result::gemm_ex(
                            self.cublas_handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            512,
                            count as i32,
                            total as i32,
                            &one as *const _ as _,
                            vp as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            512,
                            probability as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            total as i32,
                            &zero as *const _ as _,
                            (out + ((first * 16 + qh) * 512 * 2) as u64) as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            8192,
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                        )
                        .map_err(|e| format!("Gemma PV GEMM: {e:?}"))?;
                    }
                }
            }
        }
        Ok(true)
    }
    /// Export a chronological window into physical k6v6 slots. BF16 append
    /// kernels accept physical offsets, whereas FP8 conversion accepts an
    /// absolute source position and has a different final argument order.
    pub(crate) fn gemma4_stage_k6_export(
        &self,
        ks: u64,
        ki: u64,
        vs: u64,
        vi: u64,
        k: u64,
        v: u64,
        rows: usize,
        stride: usize,
        capacity: usize,
        first: usize,
        dtype: u32,
        norm: i32,
    ) -> Result<(), String> {
        if rows == 0
            || rows > capacity
            || capacity == 0
            || stride == 0
            || stride % 16 != 0
            || dtype > 1
            || [ks, ki, vs, vi, k, v].contains(&0)
        {
            return Err("invalid Gemma k6 export contract".into());
        }
        let kernel = if dtype == 0 {
            self.kernels.kv_cache_append_k6v6
        } else {
            self.kernels.kv_convert_fp8_to_k6v6
        };
        let threads = 256u32;
        let mut done = 0;
        while done < rows {
            let physical = (first + done) % capacity;
            let count = if dtype == 0 {
                (rows - done).min(capacity - physical)
            } else {
                rows - done
            };
            let mut a = (
                ks,
                ki,
                vs,
                vi,
                k + (done * stride * (if dtype == 0 { 2 } else { 1 })) as u64,
                v + (done * stride * (if dtype == 0 { 2 } else { 1 })) as u64,
                count as i32,
                stride as i32,
                capacity as i32,
                if dtype == 0 { physical as i32 } else { norm },
                if dtype == 0 {
                    norm
                } else {
                    (first + done) as i32
                },
            );
            unsafe {
                launch(
                    kernel,
                    (count as u32, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream,
                    &mut [
                        &mut a.0 as *mut _ as _,
                        &mut a.1 as *mut _ as _,
                        &mut a.2 as *mut _ as _,
                        &mut a.3 as *mut _ as _,
                        &mut a.4 as *mut _ as _,
                        &mut a.5 as *mut _ as _,
                        &mut a.6 as *mut _ as _,
                        &mut a.7 as *mut _ as _,
                        &mut a.8 as *mut _ as _,
                        &mut a.9 as *mut _ as _,
                        &mut a.10 as *mut _ as _,
                    ],
                )?;
            }
            done += count;
        }
        Ok(())
    }
    fn gemma4_window_gemm_attention(
        &self,
        q: u64,
        k: u64,
        v: u64,
        ck: u64,
        cv: u64,
        out: u64,
        rows: usize,
        prefix: usize,
        capacity: usize,
        dtype: i32,
        vision: u64,
    ) -> Result<bool, String> {
        let band = 64usize;
        let maximum_keys = (1024 + band + if vision != 0 { 280 } else { 0 }).min(prefix + rows);
        if maximum_keys * (256 * 4 + band * 6) > self.scratch.d_fp32_scratch.len * 4 {
            return Ok(false);
        }
        let pack = self.gemma4_attention_kernel("gemma4_window_kv_pack")?;
        let softmax = self.gemma4_attention_kernel("gemma4_window_softmax")?;
        unsafe {
            cublas_result::set_stream(self.cublas_handle, self.stream as cublas_sys::cudaStream_t)
                .map_err(|e| e.to_string())?;
            for first in (0..rows).step_by(band) {
                let count = band.min(rows - first);
                let key_first = (prefix + first).saturating_sub(1023);
                let key_end =
                    (prefix + first + count + if vision != 0 { 280 } else { 0 }).min(prefix + rows);
                let keys = key_end - key_first;
                let kp = *self.scratch.d_fp32_scratch.device_ptr();
                let vp = kp + (keys * 256 * 2) as u64;
                let scores = vp + (keys * 256 * 2) as u64;
                let probability = scores + (keys * band * 4) as u64;
                for head in 0..8i32 {
                    let mut a = (
                        kp,
                        vp,
                        ck,
                        cv,
                        k,
                        v,
                        keys as i32,
                        key_first as i32,
                        prefix as i32,
                        head,
                        dtype,
                        capacity as i32,
                    );
                    launch(
                        pack,
                        (((keys * 256 + 255) / 256) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        self.stream,
                        &mut [
                            &mut a.0 as *mut _ as _,
                            &mut a.1 as *mut _ as _,
                            &mut a.2 as *mut _ as _,
                            &mut a.3 as *mut _ as _,
                            &mut a.4 as *mut _ as _,
                            &mut a.5 as *mut _ as _,
                            &mut a.6 as *mut _ as _,
                            &mut a.7 as *mut _ as _,
                            &mut a.8 as *mut _ as _,
                            &mut a.9 as *mut _ as _,
                            &mut a.10 as *mut _ as _,
                            &mut a.11 as *mut _ as _,
                        ],
                    )?;
                    for qh in head as usize * 2..(head as usize + 1) * 2 {
                        let one = 1f32;
                        let zero = 0f32;
                        cublas_result::gemm_ex(
                            self.cublas_handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            keys as i32,
                            count as i32,
                            256,
                            &one as *const _ as _,
                            kp as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            256,
                            (q + ((first * 16 + qh) * 256 * 2) as u64) as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            4096,
                            &zero as *const _ as _,
                            scores as _,
                            cublas_sys::cudaDataType::CUDA_R_32F,
                            keys as i32,
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                        )
                        .map_err(|e| format!("Gemma sliding QK GEMM: {e:?}"))?;
                        let mut a = (
                            scores,
                            probability,
                            keys as i32,
                            key_first as i32,
                            (prefix + first) as i32,
                            vision,
                        );
                        launch(
                            softmax,
                            (count as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            self.stream,
                            &mut [
                                &mut a.0 as *mut _ as _,
                                &mut a.1 as *mut _ as _,
                                &mut a.2 as *mut _ as _,
                                &mut a.3 as *mut _ as _,
                                &mut a.4 as *mut _ as _,
                                &mut a.5 as *mut _ as _,
                            ],
                        )?;
                        cublas_result::gemm_ex(
                            self.cublas_handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            256,
                            count as i32,
                            keys as i32,
                            &one as *const _ as _,
                            vp as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            256,
                            probability as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            keys as i32,
                            &zero as *const _ as _,
                            (out + ((first * 16 + qh) * 256 * 2) as u64) as _,
                            cublas_sys::cudaDataType::CUDA_R_16BF,
                            4096,
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                        )
                        .map_err(|e| format!("Gemma sliding PV GEMM: {e:?}"))?;
                    }
                }
            }
        }
        Ok(true)
    }
    pub(crate) fn gemma4_ring_replay_completion(&self) -> Result<(), String> {
        self.stream_sync()
    }
    fn gemma4_attention_kernel(&self, name: &str) -> Result<RawCuFunc, String> {
        if self.device.get_func("gemma4_attention", name).is_none() {
            self.device
                .load_ptx(
                    cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
                        env!("OUT_DIR"),
                        "/gemma4_attention.ptx"
                    ))),
                    "gemma4_attention",
                    &[
                        "gemma4_full_history_wmma",
                        "gemma4_window_ring",
                        "gemma4_ring_append",
                        "gemma4_ring_gather",
                        "gemma4_full_kv_pack",
                        "gemma4_full_softmax",
                        "gemma4_window_kv_pack",
                        "gemma4_window_softmax",
                    ],
                )
                .map_err(|e| e.to_string())?;
        }
        self.device
            .get_func("gemma4_attention", name)
            .map(|f| extract_cu_func(&f))
            .ok_or_else(|| format!("missing {name}"))
    }
    fn stage_ring_capacity(&self, layer: usize) -> usize {
        self.prefill_kv_ring_capacity
            .get(layer)
            .copied()
            .unwrap_or(0)
    }
    fn stage_layer_capacity(&self, layer: usize) -> usize {
        let ring = self.stage_ring_capacity(layer);
        if ring > 0 {
            ring
        } else {
            self.prefill_kv_temp_seq
        }
    }
    pub(crate) fn gemma4_window_ring_attention(
        &self,
        layer: usize,
        q: u64,
        k: u64,
        v: u64,
        ck: u64,
        cv: u64,
        out: u64,
        rows: usize,
        prefix: usize,
        capacity: usize,
        dtype: i32,
        vision: u64,
    ) -> Result<(), String> {
        if rows == 0
            || ![1024, 1056].contains(&capacity)
            || !(0..=1).contains(&dtype)
            || q == 0
            || k == 0
            || v == 0
            || out == 0
            || (prefix > 0 && (ck == 0 || cv == 0))
        {
            return Err("Gemma sliding ring contract".into());
        }
        if layer == 0 {
            if let Ok(directory) = std::env::var("KRASIS_GEMMA_RING_CAPTURE_DIR") {
                use std::io::Write;
                self.stream_sync()?;
                let path =
                    std::path::Path::new(&directory).join(format!("prefix{prefix}-rows{rows}"));
                std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
                let row_bytes = 2048 * if dtype == 0 { 2 } else { 1 };
                for (name, ptr, bytes) in [
                    ("q.bf16", q, rows * 16 * 256 * 2),
                    ("k.bf16", k, rows * 2048 * 2),
                    ("v.bf16", v, rows * 2048 * 2),
                    ("ck.raw", ck, capacity * row_bytes),
                    ("cv.raw", cv, capacity * row_bytes),
                    (
                        "blocks.i32",
                        vision,
                        if vision != 0 { (prefix + rows) * 4 } else { 0 },
                    ),
                ] {
                    if bytes == 0 {
                        continue;
                    }
                    let mut data = vec![0u8; bytes];
                    unsafe {
                        cuda_sys::lib()
                            .cuMemcpyDtoH_v2(data.as_mut_ptr() as _, ptr, bytes)
                            .result()
                            .map_err(|e| e.to_string())?;
                    }
                    let mut file = std::fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(path.join(name))
                        .map_err(|e| e.to_string())?;
                    file.write_all(&data).map_err(|e| e.to_string())?;
                }
                std::fs::write(path.join("contract.json"),serde_json::to_vec_pretty(&serde_json::json!({"rows":rows,"prefix":prefix,"capacity":capacity,"dtype":dtype,"has_image":vision!=0})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            }
        }
        if std::env::var("KRASIS_GEMMA_WINDOW_GEMM").as_deref() == Ok("1")
            && self.gemma4_window_gemm_attention(
                q, k, v, ck, cv, out, rows, prefix, capacity, dtype, vision,
            )?
        {
            return Ok(());
        }
        let raw = self.gemma4_attention_kernel("gemma4_window_ring")?;
        let mut a = (
            out,
            q,
            ck,
            cv,
            k,
            v,
            rows as i32,
            16i32,
            8i32,
            256i32,
            1f32,
            prefix as i32,
            2048i32,
            1024i32,
            dtype,
            vision,
            capacity as i32,
        );
        unsafe {
            launch(
                raw,
                (((rows + 15) / 16) as u32, 16, 1),
                (32, 1, 1),
                27136,
                self.stream,
                &mut [
                    &mut a.0 as *mut _ as _,
                    &mut a.1 as *mut _ as _,
                    &mut a.2 as *mut _ as _,
                    &mut a.3 as *mut _ as _,
                    &mut a.4 as *mut _ as _,
                    &mut a.5 as *mut _ as _,
                    &mut a.6 as *mut _ as _,
                    &mut a.7 as *mut _ as _,
                    &mut a.8 as *mut _ as _,
                    &mut a.9 as *mut _ as _,
                    &mut a.10 as *mut _ as _,
                    &mut a.11 as *mut _ as _,
                    &mut a.12 as *mut _ as _,
                    &mut a.13 as *mut _ as _,
                    &mut a.14 as *mut _ as _,
                    &mut a.15 as *mut _ as _,
                    &mut a.16 as *mut _ as _,
                ],
            )
        }
    }
    fn gemma4_stage_ring_append(
        &self,
        layer: usize,
        k: u64,
        v: u64,
        ck: u64,
        cv: u64,
        rows: usize,
        prefix: usize,
        stride: usize,
        cap: usize,
        dtype: u32,
    ) -> Result<(), String> {
        if cap == 0 || rows == 0 || !matches!(dtype, 0 | 1) {
            return Err("invalid stage ring append".into());
        }
        let raw = self.gemma4_attention_kernel("gemma4_ring_append")?;
        let mut a = (
            ck,
            cv,
            k,
            v,
            rows as i32,
            stride as i32,
            prefix as i32,
            cap as i32,
            dtype as i32,
        );
        let result = unsafe {
            launch(
                raw,
                (rows.min(cap) as u32, 1, 1),
                (256, 1, 1),
                0,
                self.stream,
                &mut [
                    &mut a.0 as *mut _ as _,
                    &mut a.1 as *mut _ as _,
                    &mut a.2 as *mut _ as _,
                    &mut a.3 as *mut _ as _,
                    &mut a.4 as *mut _ as _,
                    &mut a.5 as *mut _ as _,
                    &mut a.6 as *mut _ as _,
                    &mut a.7 as *mut _ as _,
                    &mut a.8 as *mut _ as _,
                ],
            )
        };
        result?;
        if layer == 0 {
            if let Ok(directory) = std::env::var("KRASIS_GEMMA_RING_CAPTURE_DIR") {
                use std::io::Write;
                self.stream_sync()?;
                let path =
                    std::path::Path::new(&directory).join(format!("prefix{prefix}-rows{rows}"));
                for (name, ptr) in [("ck-after.raw", ck), ("cv-after.raw", cv)] {
                    let mut data = vec![0u8; cap * stride * if dtype == 0 { 2 } else { 1 }];
                    unsafe {
                        cuda_sys::lib()
                            .cuMemcpyDtoH_v2(data.as_mut_ptr() as _, ptr, data.len())
                            .result()
                            .map_err(|e| e.to_string())?;
                    }
                    let mut file = std::fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(path.join(name))
                        .map_err(|e| e.to_string())?;
                    file.write_all(&data).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
    fn gemma4_stage_ring_gather(
        &self,
        source: u64,
        output: u64,
        rows: usize,
        row_bytes: usize,
        first: usize,
        cap: usize,
    ) -> Result<(), String> {
        if cap == 0 || rows > cap {
            return Err("invalid stage ring gather".into());
        }
        let raw = self.gemma4_attention_kernel("gemma4_ring_gather")?;
        let mut a = (
            source,
            output,
            rows as i32,
            row_bytes as i32,
            first as i32,
            cap as i32,
        );
        unsafe {
            launch(
                raw,
                (((rows * row_bytes + 255) / 256) as u32, 1, 1),
                (256, 1, 1),
                0,
                self.stream,
                &mut [
                    &mut a.0 as *mut _ as _,
                    &mut a.1 as *mut _ as _,
                    &mut a.2 as *mut _ as _,
                    &mut a.3 as *mut _ as _,
                    &mut a.4 as *mut _ as _,
                    &mut a.5 as *mut _ as _,
                ],
            )
        }
    }
}

// Image blocks are indivisible. Only the native multimodal route uses this
// planner; ordinary text keeps its existing measured chunk geometry.
fn build_gemma_image_chunk_plan(blocks: &[i32], maximum: usize) -> Result<Vec<usize>, String> {
    if maximum == 0 {
        return Err("Gemma image chunk capacity is zero".into());
    }
    let mut plan = Vec::new();
    let mut start = 0;
    while start < blocks.len() {
        let mut end = (start + maximum).min(blocks.len());
        if end < blocks.len() && blocks[end] >= 0 && blocks[end] == blocks[end - 1] {
            let id = blocks[end];
            while end > start && blocks[end - 1] == id {
                end -= 1;
            }
            if end == start {
                return Err(format!(
                    "Gemma image soft-token block exceeds measured chunk capacity {maximum}"
                ));
            }
        }
        plan.push(end - start);
        start = end;
    }
    Ok(plan)
}
#[cfg(test)]
mod gemma_bounded_chunk_tests {
    #[test]
    fn image_block_crossing_boundary_stays_whole() {
        let mut blocks = vec![-1; 1600];
        blocks[500..760].fill(0);
        let p = super::build_gemma_image_chunk_plan(&blocks, 577).unwrap();
        assert_eq!(p, vec![500, 577, 523]);
        let mut n = 0;
        for size in p {
            n += size;
            if n < blocks.len() {
                assert!(blocks[n] < 0 || blocks[n] != blocks[n - 1]);
            }
        }
    }
    #[test]
    fn image_larger_than_capacity_rejects() {
        assert!(super::build_gemma_image_chunk_plan(&vec![0; 280], 128).is_err());
    }
}
