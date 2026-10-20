use std::path::Path;
// BEGIN bounded matching KDA block diagnostics
fn matched_kda_capture_window(
    tokens: usize,
    boundary: usize,
    block: usize,
) -> Option<(usize, usize)> {
    assert!(tokens > 0 && boundary > 0 && block > 0);
    let start = ((boundary - 1) / block) * block;
    (start < tokens).then(|| (start, (tokens - start).min(block)))
}
// END bounded matching KDA block diagnostics
fn matched_kda_fixture(
    input: &Path,
    tokens: usize,
    artifacts: &Path,
    anchor_subchunk: Option<usize>,
    intra_variant: Option<&serde_json::Value>,
) -> [Vec<u8>; 4] {
    let ctx = GpuTestCtx::new();
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input.join("descriptor.json")).unwrap()).unwrap();
    let value = |name: &str| {
        descriptor[name]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .expect(name)
    };
    let heads = value("heads");
    let dim = value("dim");
    let source_tokens = value("height");
    assert!(
        tokens <= source_tokens && dim == 128,
        "unsupported captured KDA geometry"
    );
    pyo3::prepare_freethreaded_python();
    let fla = load_fla(true, heads)
        .unwrap()
        .expect("required production FLA sidecar");
    // BEGIN bounded causal-reference KDA fixture selection
    assert!(anchor_subchunk.is_none() || intra_variant.is_none(), "separate diagnostic interventions required");
    let causal_intra = intra_variant.map(|entry| MatchedKdaDiagnosticIntra::load(entry, heads, &descriptor, &ctx));
    let intra_function = match &causal_intra { Some(owner) => owner.function, None => fla.kda_intra };
    // END bounded causal-reference KDA fixture selection
    let prepare = ctx.get_decode_kernel("kimi_delta_chunk_prepare_bf16");
    let recurrent = ctx.get_decode_kernel("kimi_delta_recurrent_scan_tiled_bf16");
    let block_tokens = value("block_tokens");
    assert_eq!(
        block_tokens, FLA_PREFILL_BLOCK_TOKENS,
        "captured sidecar block size"
    );
    let chunks = tokens.div_ceil(block_tokens);
    let lower = descriptor["gate_lower_bound"]
        .as_f64()
        .expect("gate lower bound") as f32;
    let q_scale = descriptor["q_scale"].as_f64().expect("captured Q scale") as f32;
    let eps = descriptor["normalization_epsilon"]
        .as_f64()
        .expect("captured epsilon") as f32;
    assert!(q_scale.is_finite() && q_scale > 0. && eps.is_finite() && eps > 0.);
    let captured = |name: &str, full_bytes: usize, selected_bytes: usize| {
        let mut bytes = std::fs::read(input.join(name)).unwrap();
        assert_eq!(bytes.len(), full_bytes, "capture tensor {name}");
        bytes.truncate(selected_bytes);
        bytes
    };
    let q = captured(
        "q.bf16",
        source_tokens * heads * dim * 2,
        tokens * heads * dim * 2,
    );
    let k = captured(
        "k.bf16",
        source_tokens * heads * dim * 2,
        tokens * heads * dim * 2,
    );
    let v = captured(
        "v.bf16",
        source_tokens * heads * dim * 2,
        tokens * heads * dim * 2,
    );
    let gates = captured(
        "gates.bf16",
        source_tokens * heads * dim * 2,
        tokens * heads * dim * 2,
    );
    let betas = captured("beta.bf16", source_tokens * heads * 2, tokens * heads * 2);
    let a_log = captured("a-log.f32", heads * 4, heads * 4);
    let dt_bias = captured("dt-bias.f32", heads * dim * 4, heads * dim * 4);
    let initial_state = captured(
        "initial-state.f32",
        heads * dim * dim * 4,
        heads * dim * dim * 4,
    );
    // The tiled recurrent kernel has an independent CPU-parity test above;
    // retain raw inputs for it because the chunk preparation kernel
    // deliberately normalizes Q/K and applies sigmoid(beta) in place.
    let d_oracle_q = ctx.upload_bf16(&q);
    let d_oracle_k = ctx.upload_bf16(&k);
    let d_oracle_v = ctx.upload_bf16(&v);
    let d_oracle_gates = ctx.upload_bf16(&gates);
    let d_oracle_betas = ctx.upload_bf16(&betas);
    let d_a_log = ctx.upload_f32(&a_log);
    let d_dt_bias = ctx.upload_f32(&dt_bias);
    let d_oracle_state = ctx.upload_f32(&initial_state);
    let d_oracle_output = ctx.alloc_bf16(tokens * heads * dim);
    let mut tokens_i32 = tokens as i32;
    let mut heads_i32 = heads as i32;
    let mut dim_i32 = dim as i32;
    let mut lower_f32 = lower;
    let mut scale_f32 = q_scale;
    let mut eps_f32 = eps;
    unsafe {
        let mut state_ptr = *d_oracle_state.device_ptr() as u64;
        let mut q_ptr = *d_oracle_q.device_ptr() as u64;
        let mut k_ptr = *d_oracle_k.device_ptr() as u64;
        let mut v_ptr = *d_oracle_v.device_ptr() as u64;
        let mut gates_ptr = *d_oracle_gates.device_ptr() as u64;
        let mut betas_ptr = *d_oracle_betas.device_ptr() as u64;
        let mut a_log_ptr = *d_a_log.device_ptr() as u64;
        let mut dt_bias_ptr = *d_dt_bias.device_ptr() as u64;
        let mut output_ptr = *d_oracle_output.device_ptr() as u64;
        launch(
            recurrent,
            (heads as u32, dim.div_ceil(32) as u32, 1),
            (32, 1, 1),
            ((3 * dim + 2) * std::mem::size_of::<f32>()) as u32,
            ctx.stream(),
            &mut [
                &mut state_ptr as *mut _ as *mut _,
                &mut q_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut v_ptr as *mut _ as *mut _,
                &mut gates_ptr as *mut _ as *mut _,
                &mut betas_ptr as *mut _ as *mut _,
                &mut a_log_ptr as *mut _ as *mut _,
                &mut dt_bias_ptr as *mut _ as *mut _,
                &mut output_ptr as *mut _ as *mut _,
                &mut tokens_i32 as *mut _ as *mut _,
                &mut heads_i32 as *mut _ as *mut _,
                &mut dim_i32 as *mut _ as *mut _,
                &mut lower_f32 as *mut _ as *mut _,
                &mut scale_f32 as *mut _ as *mut _,
                &mut eps_f32 as *mut _ as *mut _,
            ],
        )
        .unwrap();
    }

    let d_q = ctx.upload_bf16(&q);
    let d_k = ctx.upload_bf16(&k);
    let d_v = ctx.upload_bf16(&v);
    let d_gates = ctx.upload_bf16(&gates);
    let d_betas = ctx.upload_bf16(&betas);
    let d_state = ctx.upload_f32(&initial_state);
    let d_output = ctx.alloc_bf16(tokens * heads * dim);
    let d_g = ctx.alloc_f32(tokens * heads * dim);
    let d_aqk = ctx.alloc_f32(tokens * heads * block_tokens);
    let d_akkd = ctx.alloc_f32(tokens * heads * block_tokens);
    let d_akk = ctx.alloc_bf16(tokens * heads * block_tokens);
    let d_w = ctx.alloc_bf16(tokens * heads * dim);
    let d_u = ctx.alloc_bf16(tokens * heads * dim);
    let d_kg = ctx.alloc_bf16(tokens * heads * dim);
    let d_h = ctx.alloc_f32(chunks * heads * dim * dim);
    let d_final_state = ctx.alloc_f32(heads * dim * dim);
    let d_v_new = ctx.alloc_bf16(tokens * heads * dim);

    std::fs::create_dir(artifacts).unwrap();
    let first_rows = tokens.min(block_tokens);
    let dump = |name: &str, ptr: u64, bytes: usize| {
        let mut host = vec![0u8; bytes];
        unsafe {
            assert_eq!(
                cuda_sys::lib().cuStreamSynchronize(ctx.stream()),
                cuda_sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                cuda_sys::lib().cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), ptr, bytes),
                cuda_sys::CUresult::CUDA_SUCCESS
            );
        }
        std::fs::write(artifacts.join(name), host).unwrap();
    };
    // BEGIN bounded matching KDA window capture
    // Capture the block containing the authoritative prefix boundary, rather
    // than another broad tensor trace. Native chunk state is stored BF16;
    // its F32-sized backing does not make it a true F32 register carry.
    let boundary = descriptor["row"].as_u64().unwrap() as usize + 1;
    let window = matched_kda_capture_window(tokens, boundary, block_tokens);
    if let Some((start, rows)) = window {
        std::fs::write(artifacts.join("matching-window.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "start":start,"rows":rows,"tokens":tokens,"boundary":boundary,
            "heads":heads,"dim":dim,"block_tokens":block_tokens,
            "state_dtype":"bf16","state_scope":"native stored chunk-boundary state, not F32 register carry",
            "qualifies_candidate":false
        })).unwrap()).unwrap();
    }
    let dump_window = |name: &str, ptr: u64, row_elements: usize, element_size: usize| {
        if let Some((start, rows)) = window {
            let offset = start
                .checked_mul(row_elements)
                .unwrap()
                .checked_mul(element_size)
                .unwrap();
            let bytes = rows
                .checked_mul(row_elements)
                .unwrap()
                .checked_mul(element_size)
                .unwrap();
            dump(
                &format!("matching-window-{name}"),
                ptr.checked_add(offset as u64).unwrap(),
                bytes,
            );
        }
    };
    // END bounded matching KDA window capture

    let mut q_ptr = *d_q.device_ptr() as u64;
    let mut k_ptr = *d_k.device_ptr() as u64;
    let mut beta_ptr = *d_betas.device_ptr() as u64;
    let mut prep_tokens = tokens as i32;
    let mut prep_heads = heads as i32;
    let mut prep_dim = dim as i32;
    let mut prep_scale = 1.0f32;
    let mut prep_eps = eps;
    unsafe {
        launch(
            prepare,
            ((tokens * heads) as u32, 1, 1),
            (dim as u32, 1, 1),
            (2 * dim * std::mem::size_of::<f32>()) as u32,
            ctx.stream(),
            &mut [
                &mut q_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut beta_ptr as *mut _ as *mut _,
                &mut prep_tokens as *mut _ as *mut _,
                &mut prep_heads as *mut _ as *mut _,
                &mut prep_dim as *mut _ as *mut _,
                &mut prep_scale as *mut _ as *mut _,
                &mut prep_eps as *mut _ as *mut _,
            ],
        )
        .unwrap();
    }

    dump(
        "prepared-q.bf16",
        *d_q.device_ptr() as u64,
        first_rows * heads * dim * 2,
    );
    dump(
        "prepared-k.bf16",
        *d_k.device_ptr() as u64,
        first_rows * heads * dim * 2,
    );
    dump(
        "prepared-beta.bf16",
        *d_betas.device_ptr() as u64,
        first_rows * heads * 2,
    );
    // BEGIN bounded matching KDA prepared inputs
    dump_window("prepared-q.bf16", *d_q.device_ptr() as u64, heads * dim, 2);
    dump_window("prepared-k.bf16", *d_k.device_ptr() as u64, heads * dim, 2);
    dump_window("prepared-beta.bf16", *d_betas.device_ptr() as u64, heads, 2);
    // END bounded matching KDA prepared inputs
    let stream_ptr = ctx.stream() as *mut std::ffi::c_void;
    let t_arg = tokens as i32;
    let call = |label: &str, rc: u32| assert_eq!(rc, 0, "{label} returned rc={rc}");
    unsafe {
        call(
            "KDA gate cumsum",
            (fla.kda_gate_cumsum)(
                *d_gates.device_ptr() as u64,
                *d_a_log.device_ptr() as u64,
                *d_dt_bias.device_ptr() as u64,
                *d_g.device_ptr() as u64,
                std::f32::consts::LOG2_E,
                0,
                0,
                lower,
                t_arg,
                dim.div_ceil(32) as u32,
                chunks as u32,
                heads as u32,
                stream_ptr,
            ),
        );
        dump(
            "gate-cumsum.f32",
            *d_g.device_ptr() as u64,
            first_rows * heads * dim * 4,
        );
        // BEGIN bounded matching KDA gate inputs
        dump_window("gate-cumsum.f32", *d_g.device_ptr() as u64, heads * dim, 4);
        // END bounded matching KDA gate inputs
        // BEGIN bounded KDA chain anchor intervention
        // Only the standalone diagnostic can supply this treatment. Real
        // operands stay intact; restore the original gate buffer immediately
        // after intra, before every remaining production-chain operation.
        let anchor_restore = anchor_subchunk.map(|subchunk| {
            assert!(subchunk > 0 && block_tokens % subchunk == 0);
            assert!(tokens > boundary, "anchor intervention needs future rows");
            let first = ((boundary - 1) / subchunk) * subchunk;
            let prefix_rows = boundary - first;
            let available_rows = (tokens - first).min(subchunk);
            let short_anchor = (subchunk / 2).min(prefix_rows - 1);
            let long_anchor = (subchunk / 2).min(available_rows - 1);
            assert!(long_anchor >= prefix_rows && short_anchor < prefix_rows);
            let bytes = heads * dim * 4;
            let from = (*d_g.device_ptr() as u64) + ((first + short_anchor) * bytes) as u64;
            let to = (*d_g.device_ptr() as u64) + ((first + long_anchor) * bytes) as u64;
            let mut saved = vec![0u8; bytes];
            assert_eq!(cuda_sys::lib().cuStreamSynchronize(ctx.stream()), cuda_sys::CUresult::CUDA_SUCCESS);
            assert_eq!(cuda_sys::lib().cuMemcpyDtoH_v2(saved.as_mut_ptr().cast(), to, bytes), cuda_sys::CUresult::CUDA_SUCCESS);
            assert_eq!(cuda_sys::lib().cuMemcpyDtoDAsync_v2(to, from, bytes, ctx.stream()), cuda_sys::CUresult::CUDA_SUCCESS);
            std::fs::write(artifacts.join("anchor-intervention.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "subchunk":subchunk,"first":first,"boundary":boundary,"from_row":first+short_anchor,"to_row":first+long_anchor,
                "modified_bytes":bytes,"scope":"intra-only future rescaling reference; raw model operands unchanged; future output rows uninterpreted",
                "qualifies_candidate":false
            })).unwrap()).unwrap();
            (to, saved)
        });
        // END bounded KDA chain anchor intervention
        call(
            "KDA intra",
            (intra_function)(
                *d_q.device_ptr() as u64,
                *d_k.device_ptr() as u64,
                *d_g.device_ptr() as u64,
                *d_betas.device_ptr() as u64,
                *d_aqk.device_ptr() as u64,
                *d_akkd.device_ptr() as u64,
                q_scale,
                0,
                0,
                t_arg,
                chunks as u32,
                dim.div_ceil(32) as u32,
                heads as u32,
                stream_ptr,
            ),
        );
        // BEGIN bounded KDA chain original gates restoration
        if let Some((ptr, saved)) = anchor_restore {
            assert_eq!(
                cuda_sys::lib().cuStreamSynchronize(ctx.stream()),
                cuda_sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                cuda_sys::lib().cuMemcpyHtoD_v2(ptr, saved.as_ptr().cast(), saved.len()),
                cuda_sys::CUresult::CUDA_SUCCESS
            );
            let mut restored = vec![0u8; saved.len()];
            assert_eq!(
                cuda_sys::lib().cuMemcpyDtoH_v2(restored.as_mut_ptr().cast(), ptr, restored.len()),
                cuda_sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                restored, saved,
                "original gates must be restored before inter/solve/WY/state/output"
            );
            std::fs::write(artifacts.join("anchor-restored-original.f32"), restored).unwrap();
        }
        // END bounded KDA chain original gates restoration
        // Aqk's compiled sidecar signature is BF16 despite the existing
        // fixture's oversized F32 backing allocation. Preserve its actual dtype.
        dump(
            "intra-aqk.bf16",
            *d_aqk.device_ptr() as u64,
            first_rows * heads * block_tokens * 2,
        );
        // BEGIN bounded matching KDA intra outputs
        dump_window(
            "intra-aqk.bf16",
            *d_aqk.device_ptr() as u64,
            heads * block_tokens,
            2,
        );
        // END bounded matching KDA intra outputs
        call(
            "KDA inter solve",
            (fla.kda_inter_solve)(
                *d_q.device_ptr() as u64,
                *d_k.device_ptr() as u64,
                *d_g.device_ptr() as u64,
                *d_betas.device_ptr() as u64,
                *d_aqk.device_ptr() as u64,
                *d_akkd.device_ptr() as u64,
                *d_akk.device_ptr() as u64,
                q_scale,
                0,
                0,
                t_arg,
                chunks as u32,
                heads as u32,
                1,
                stream_ptr,
            ),
        );
        dump(
            "solved-aqk.bf16",
            *d_aqk.device_ptr() as u64,
            first_rows * heads * block_tokens * 2,
        );
        dump(
            "solved-akk.bf16",
            *d_akk.device_ptr() as u64,
            first_rows * heads * block_tokens * 2,
        );
        // BEGIN bounded matching KDA solved outputs
        dump_window(
            "solved-aqk.bf16",
            *d_aqk.device_ptr() as u64,
            heads * block_tokens,
            2,
        );
        dump_window(
            "solved-akk.bf16",
            *d_akk.device_ptr() as u64,
            heads * block_tokens,
            2,
        );
        // END bounded matching KDA solved outputs
        call(
            "KDA WY",
            (fla.kda_wy_repr)(
                0,
                *d_k.device_ptr() as u64,
                0,
                *d_kg.device_ptr() as u64,
                *d_v.device_ptr() as u64,
                *d_betas.device_ptr() as u64,
                *d_w.device_ptr() as u64,
                *d_u.device_ptr() as u64,
                *d_akk.device_ptr() as u64,
                *d_g.device_ptr() as u64,
                0,
                0,
                t_arg,
                chunks as u32,
                heads as u32,
                1,
                stream_ptr,
            ),
        );
        dump(
            "wy-w.bf16",
            *d_w.device_ptr() as u64,
            first_rows * heads * dim * 2,
        );
        dump(
            "wy-u.bf16",
            *d_u.device_ptr() as u64,
            first_rows * heads * dim * 2,
        );
        // BEGIN bounded matching KDA WY outputs
        dump_window("wy-w.bf16", *d_w.device_ptr() as u64, heads * dim, 2);
        dump_window("wy-u.bf16", *d_u.device_ptr() as u64, heads * dim, 2);
        // END bounded matching KDA WY outputs
        call(
            "KDA state recurrence",
            (fla.kda_state_recurrence)(
                *d_kg.device_ptr() as u64,
                *d_u.device_ptr() as u64,
                *d_w.device_ptr() as u64,
                *d_v_new.device_ptr() as u64,
                0,
                *d_g.device_ptr() as u64,
                *d_h.device_ptr() as u64,
                *d_state.device_ptr() as u64,
                *d_final_state.device_ptr() as u64,
                0,
                0,
                t_arg,
                dim.div_ceil(32) as u32,
                heads as u32,
                1,
                stream_ptr,
            ),
        );
        dump(
            "state-v-new.bf16",
            *d_v_new.device_ptr() as u64,
            first_rows * heads * dim * 2,
        );
        dump(
            "initial-chunk-state.bf16",
            *d_h.device_ptr() as u64,
            heads * dim * dim * 2,
        );
        // BEGIN bounded matching KDA stored state
        dump_window(
            "state-v-new.bf16",
            *d_v_new.device_ptr() as u64,
            heads * dim,
            2,
        );
        if let Some((start, _)) = window {
            let chunk = start / block_tokens;
            let offset = chunk.checked_mul(heads * dim * dim * 2).unwrap();
            dump(
                "matching-window-start-state.bf16",
                (*d_h.device_ptr() as u64)
                    .checked_add(offset as u64)
                    .unwrap(),
                heads * dim * dim * 2,
            );
        }
        // END bounded matching KDA stored state
        call(
            "KDA output",
            (fla.kda_output)(
                *d_q.device_ptr() as u64,
                *d_v_new.device_ptr() as u64,
                *d_g.device_ptr() as u64,
                *d_h.device_ptr() as u64,
                *d_output.device_ptr() as u64,
                *d_aqk.device_ptr() as u64,
                0,
                0,
                q_scale,
                t_arg,
                dim.div_ceil(64) as u32,
                chunks as u32,
                heads as u32,
                stream_ptr,
            ),
        );
    }
    self::cuda_sync();
    dump(
        "output.bf16",
        *d_output.device_ptr() as u64,
        first_rows * heads * dim * 2,
    );
    // BEGIN bounded matching KDA final output
    dump_window("output.bf16", *d_output.device_ptr() as u64, heads * dim, 2);
    // END bounded matching KDA final output

    [
        ctx.download_bf16(&d_output),
        ctx.download_bf16(&d_oracle_output),
        ctx.download_f32(&d_final_state),
        ctx.download_f32(&d_oracle_state),
    ]
}

// BEGIN bounded matching KDA window regression
#[test]
fn test_matched_kda_capture_window_tracks_authoritative_boundary() {
    assert_eq!(
        matched_kda_capture_window(10500, 10500, 64),
        Some((10496, 4))
    );
    assert_eq!(
        matched_kda_capture_window(10564, 10500, 64),
        Some((10496, 64))
    );
    assert_eq!(matched_kda_capture_window(10496, 10500, 64), None);
    for block in [1, 16, 64, 128] {
        for boundary in [1, block, block + 1, block * 3 - 1] {
            let (start, rows) = matched_kda_capture_window(boundary, boundary, block).unwrap();
            assert_eq!(start % block, 0);
            assert_eq!(start + rows, boundary);
            assert!(rows > 0 && rows <= block);
        }
    }
}
// END bounded matching KDA window regression

// BEGIN bounded native KDA anchor counterfactual
#[test]
#[ignore]
fn test_matched_kda_intra_anchor_from_real_window() {
    use sha2::{Digest, Sha256};
    assert_eq!(std::env::var("KRASIS_DEV_SCRIPT").as_deref(), Ok("1"));
    let input = std::path::PathBuf::from(std::env::var("KRASIS_MATCHED_KDA_ANCHOR_PLAN").unwrap());
    let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    let out = std::path::PathBuf::from(plan["output"].as_str().unwrap());
    assert!(!out.exists(), "fresh diagnostic directory required");
    let directory = |key: &str| std::path::PathBuf::from(plan[key].as_str().unwrap());
    let short = directory("short_window");
    let long = directory("long_window");
    let metadata = |p: &Path| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(p.join("matching-window.json")).unwrap()).unwrap()
    };
    let sm = metadata(&short);
    let lm = metadata(&long);
    for key in ["start", "heads", "dim", "block_tokens"] {
        assert_eq!(sm[key], lm[key]);
    }
    let heads = lm["heads"].as_u64().unwrap() as usize;
    let dim = lm["dim"].as_u64().unwrap() as usize;
    let block = lm["block_tokens"].as_u64().unwrap() as usize;
    let rows = lm["rows"].as_u64().unwrap() as usize;
    let prefix = sm["rows"].as_u64().unwrap() as usize;
    let subchunk = plan["intra_subchunk_tokens"].as_u64().unwrap() as usize;
    assert_eq!(block, FLA_PREFILL_BLOCK_TOKENS);
    assert!(dim == 128 && prefix > 0 && prefix < subchunk && subchunk <= rows && rows <= block);
    let short_anchor = (subchunk / 2).min(prefix - 1);
    let long_anchor = (subchunk / 2).min(rows - 1);
    assert!(short_anchor < prefix && long_anchor >= prefix);
    let scale = plan["q_scale"].as_f64().unwrap() as f32;
    assert!(scale.is_finite() && scale > 0.);
    let mut identities = serde_json::Map::new();
    let read = |name: &str,
                elements: usize,
                size: usize,
                identities: &mut serde_json::Map<String, serde_json::Value>| {
        let file = long.join(format!("matching-window-{name}"));
        let b = std::fs::read(&file).unwrap();
        assert_eq!(b.len(), elements * size, "{name}");
        identities.insert(name.into(), serde_json::json!({"path":file,"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b))}));
        b
    };
    let q = read("prepared-q.bf16", rows * heads * dim, 2, &mut identities);
    let k = read("prepared-k.bf16", rows * heads * dim, 2, &mut identities);
    let beta = read("prepared-beta.bf16", rows * heads, 2, &mut identities);
    let gates = read("gate-cumsum.f32", rows * heads * dim, 4, &mut identities);
    let expected_long = read("intra-aqk.bf16", rows * heads * block, 2, &mut identities);
    let expected_short = std::fs::read(short.join("matching-window-intra-aqk.bf16")).unwrap();
    assert_eq!(expected_short.len(), prefix * heads * block * 2);
    for (name, full, width, size) in [
        ("prepared-q.bf16", &q, heads * dim, 2),
        ("prepared-k.bf16", &k, heads * dim, 2),
        ("prepared-beta.bf16", &beta, heads, 2),
        ("gate-cumsum.f32", &gates, heads * dim, 4),
    ] {
        assert_eq!(
            std::fs::read(short.join(format!("matching-window-{name}"))).unwrap(),
            full[..prefix * width * size]
        );
    }
    let mut anchor_gates = gates.clone();
    let bytes = heads * dim * 4;
    anchor_gates[long_anchor * bytes..(long_anchor + 1) * bytes]
        .copy_from_slice(&gates[short_anchor * bytes..(short_anchor + 1) * bytes]);
    assert_eq!(anchor_gates[..prefix * bytes], gates[..prefix * bytes]);
    std::fs::create_dir(&out).unwrap();
    let ctx = GpuTestCtx::new();
    pyo3::prepare_freethreaded_python();
    let fla = load_fla(true, heads)
        .unwrap()
        .expect("required loaded native KDA sidecar");
    let mut results = serde_json::Map::new();
    let mut outputs = Vec::new();
    for (label, tokens, g) in [
        ("full-control", rows, &gates),
        ("prefix-control", prefix, &gates),
        ("prefix-anchor", rows, &anchor_gates),
    ] {
        let dq = ctx.upload_bf16(&q);
        let dk = ctx.upload_bf16(&k);
        let db = ctx.upload_bf16(&beta);
        let dg = ctx.upload_f32(g);
        let daqk = ctx.alloc_bf16(rows * heads * block);
        let dakk = ctx.alloc_f32(rows * heads * subchunk);
        let rc = unsafe {
            (fla.kda_intra)(
                *dq.device_ptr() as u64,
                *dk.device_ptr() as u64,
                *dg.device_ptr() as u64,
                *db.device_ptr() as u64,
                *daqk.device_ptr() as u64,
                *dakk.device_ptr() as u64,
                scale,
                0,
                0,
                tokens as i32,
                tokens.div_ceil(block) as u32,
                dim.div_ceil(32) as u32,
                heads as u32,
                ctx.stream() as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, 0, "native intra {label}");
        self::cuda_sync();
        let aqk = ctx.download_bf16(&daqk);
        let akk = ctx.download_f32(&dakk);
        let selected = prefix * heads * block * 2;
        if label == "full-control" {
            assert_eq!(aqk, expected_long, "actual full native matrix control");
        }
        if label == "prefix-control" {
            assert_eq!(
                &aqk[..selected],
                &expected_short,
                "actual short native matrix control"
            );
        }
        std::fs::write(out.join(format!("{label}-aqk.bf16")), &aqk).unwrap();
        std::fs::write(out.join(format!("{label}-akkd.f32")), &akk).unwrap();
        results.insert(label.into(), serde_json::json!({"tokens":tokens,"causal_rows":prefix,"full_inputs_sha256_unchanged":label!="prefix-anchor","gate_input_sha256":format!("{:x}",Sha256::digest(g)),"causal_gate_rows_unchanged":g[..prefix*bytes]==gates[..prefix*bytes],"scope":if label=="prefix-anchor" {"counterfactual future rescaling reference remap for intra only; future noncausal rows are not interpreted"} else {"exact native control"}}));
        outputs.push((aqk, akk));
    }
    let selected = prefix * heads * block * 2;
    let diagonal = prefix * heads * subchunk * 4;
    let report = serde_json::json!({"diagnostic_only":true,"qualifies_candidate":false,"inputs":identities,"plan":plan,"short_anchor":short_anchor,"long_anchor":long_anchor,"results":results,
        "ordinary_prefix":matched_binary_metrics(&outputs[0].0[..selected], &outputs[1].0[..selected], "bf16", &out.join("ordinary-prefix-mismatches.u64")),
        "anchor_vs_prefix":matched_binary_metrics(&outputs[2].0[..selected], &outputs[1].0[..selected], "bf16", &out.join("anchor-prefix-mismatches.u64")),
        "anchor_diagonal_vs_prefix":matched_binary_metrics(&outputs[2].1[..diagonal], &outputs[1].1[..diagonal], "f32", &out.join("anchor-diagonal-prefix-mismatches.u64")),
        "limitations":["Counterfactual changes an internal future gate reference for intra only, not raw model inputs or production computation","No full model generation, precision adoption or candidate qualification","Mathematical/instruction-order/implementation contracts remain distinct"]});
    std::fs::write(
        out.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
// END bounded native KDA anchor counterfactual

// BEGIN bounded native KDA anchor chain counterfactual
#[test]
#[ignore]
fn test_matched_kda_anchor_chain_from_real_capture() {
    assert_eq!(std::env::var("KRASIS_DEV_SCRIPT").as_deref(), Ok("1"));
    let plan: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("KRASIS_MATCHED_KDA_CHAIN_PLAN").unwrap()).unwrap(),
    )
    .unwrap();
    let input = std::path::PathBuf::from(plan["capture"].as_str().unwrap());
    let original = std::path::PathBuf::from(plan["original_replay"].as_str().unwrap());
    let out = std::path::PathBuf::from(plan["output"].as_str().unwrap());
    assert!(!out.exists(), "fresh counterfactual directory required");
    std::fs::create_dir(&out).unwrap();
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input.join("descriptor.json")).unwrap()).unwrap();
    let prefix = descriptor["row"].as_u64().unwrap() as usize + 1;
    let height = descriptor["height"].as_u64().unwrap() as usize;
    let heads = descriptor["heads"].as_u64().unwrap() as usize;
    let dim = descriptor["dim"].as_u64().unwrap() as usize;
    let subchunk = plan["intra_subchunk_tokens"].as_u64().unwrap() as usize;
    let bytes = prefix * heads * dim * 2;
    let mut controls = Vec::new();
    let mut outputs = Vec::new();
    for (label, tokens, anchor) in [
        ("full-control", height, None),
        ("prefix-control", prefix, None),
        ("prefix-anchor", height, Some(subchunk)),
    ] {
        let values = matched_kda_fixture(&input, tokens, &out.join(label), anchor, None);
        if anchor.is_none() {
            for (i, name) in [
                "chunk.bf16",
                "oracle.bf16",
                "chunk-state.f32",
                "oracle-state.f32",
            ]
            .iter()
            .enumerate()
            {
                assert_eq!(
                    values[i],
                    std::fs::read(original.join(format!("height{tokens}-{name}"))).unwrap(),
                    "untouched native/serial control {label}/{name}"
                );
            }
            controls.push(label);
        }
        // Future counterfactual outputs are retained as raw diagnostics but
        // never interpreted as valid model output or candidate quality.
        std::fs::write(out.join(format!("{label}-output.bf16")), &values[0]).unwrap();
        outputs.push(values[0].clone());
    }
    let start = ((prefix - 1) / subchunk) * subchunk;
    let before = start * heads * dim * 2;
    assert_eq!(
        &outputs[0][..before],
        &outputs[2][..before],
        "intervention changed earlier history"
    );
    let report = serde_json::json!({"diagnostic_only":true,"qualifies_candidate":false,"plan":plan,"descriptor":descriptor,"controls":controls,
        "ordinary_shared_prefix":matched_binary_metrics(&outputs[0][..bytes], &outputs[1], "bf16", &out.join("ordinary-prefix-mismatches.u64")),
        "anchor_shared_prefix":matched_binary_metrics(&outputs[2][..bytes], &outputs[1], "bf16", &out.join("anchor-prefix-mismatches.u64")),
        "earlier_history_unchanged":true,"limitations":["Only original prefix output is interpreted; future counterfactual rows are not model output","No full-model decision reversal repair or candidate qualification"]});
    std::fs::write(
        out.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
// END bounded native KDA anchor chain counterfactual

// BEGIN bounded isolated KDA precision replay
#[test]
#[ignore]
fn test_matched_kda_precision_from_real_window() {
    use sha2::{Digest, Sha256};
    assert_eq!(std::env::var("KRASIS_DEV_SCRIPT").as_deref(), Ok("1"));
    let plan: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("KRASIS_MATCHED_KDA_PRECISION_PLAN").unwrap()).unwrap(),
    ).unwrap();
    let out = std::path::PathBuf::from(plan["output"].as_str().unwrap());
    assert!(!out.exists(), "fresh precision diagnostic directory required");
    let short = std::path::PathBuf::from(plan["short_window"].as_str().unwrap());
    let long = std::path::PathBuf::from(plan["long_window"].as_str().unwrap());
    let metadata = |p: &Path| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(p.join("matching-window.json")).unwrap()).unwrap()
    };
    let sm = metadata(&short);
    let lm = metadata(&long);
    for key in ["start", "heads", "dim", "block_tokens"] { assert_eq!(sm[key], lm[key]); }
    let heads = lm["heads"].as_u64().unwrap() as usize;
    let dim = lm["dim"].as_u64().unwrap() as usize;
    let block = lm["block_tokens"].as_u64().unwrap() as usize;
    let rows = lm["rows"].as_u64().unwrap() as usize;
    let prefix = sm["rows"].as_u64().unwrap() as usize;
    let subchunk = plan["intra_subchunk_tokens"].as_u64().unwrap() as usize;
    assert_eq!(block, FLA_PREFILL_BLOCK_TOKENS);
    assert!(dim == 128 && prefix > 0 && prefix < subchunk && subchunk <= rows && rows <= block);
    let scale = plan["q_scale"].as_f64().unwrap() as f32;
    assert!(scale.is_finite() && scale > 0.);
    let repeats = plan["timing_repeats"].as_u64().unwrap() as usize;
    assert!(repeats > 0);
    let mut identities = serde_json::Map::new();
    let read = |name: &str, elements: usize, size: usize, identities: &mut serde_json::Map<String, serde_json::Value>| {
        let file = long.join(format!("matching-window-{name}"));
        let b = std::fs::read(&file).unwrap();
        assert_eq!(b.len(), elements * size, "{name}");
        identities.insert(name.into(), serde_json::json!({"path":file,"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b))}));
        b
    };
    let q = read("prepared-q.bf16", rows * heads * dim, 2, &mut identities);
    let k = read("prepared-k.bf16", rows * heads * dim, 2, &mut identities);
    let beta = read("prepared-beta.bf16", rows * heads, 2, &mut identities);
    let gates = read("gate-cumsum.f32", rows * heads * dim, 4, &mut identities);
    for (name, full, width, size) in [
        ("prepared-q.bf16", &q, heads * dim, 2), ("prepared-k.bf16", &k, heads * dim, 2),
        ("prepared-beta.bf16", &beta, heads, 2), ("gate-cumsum.f32", &gates, heads * dim, 4),
    ] { assert_eq!(std::fs::read(short.join(format!("matching-window-{name}"))).unwrap(), full[..prefix * width * size]); }
    std::fs::create_dir(&out).unwrap();
    let ctx = GpuTestCtx::new();
    pyo3::prepare_freethreaded_python();
    let fla = load_fla(true, heads).unwrap().expect("required loaded native KDA sidecar");
    let dq = ctx.upload_bf16(&q);
    let dk = ctx.upload_bf16(&k);
    let db = ctx.upload_bf16(&beta);
    let dg = ctx.upload_f32(&gates);
    let daqk = ctx.alloc_bf16(rows * heads * block);
    let dakk = ctx.alloc_f32(rows * heads * subchunk);
    let launch_intra = |function: FlaKdaIntraFn, tokens: usize| {
        let rc = unsafe { function(
            *dq.device_ptr() as u64, *dk.device_ptr() as u64, *dg.device_ptr() as u64,
            *db.device_ptr() as u64, *daqk.device_ptr() as u64, *dakk.device_ptr() as u64,
            scale, 0, 0, tokens as i32, tokens.div_ceil(block) as u32,
            dim.div_ceil(32) as u32, heads as u32, ctx.stream() as *mut std::ffi::c_void,
        ) };
        assert_eq!(rc, 0, "isolated KDA precision kernel");
    };
    let selected = prefix * heads * block * 2;
    let diagonal = prefix * heads * subchunk * 4;
    let mut native = Vec::new();
    for (label, tokens, dir) in [("full", rows, &long), ("prefix", prefix, &short)] {
        launch_intra(fla.kda_intra, tokens);
        self::cuda_sync();
        let aqk = ctx.download_bf16(&daqk);
        let akk = ctx.download_f32(&dakk);
        if plan["require_archived_control"].as_bool().unwrap() {
            let expected = std::fs::read(dir.join("matching-window-intra-aqk.bf16")).unwrap();
            assert_eq!(&aqk[..tokens * heads * block * 2], &expected, "archived native {label}");
        }
        std::fs::write(out.join(format!("native-{label}-aqk.bf16")), &aqk).unwrap();
        std::fs::write(out.join(format!("native-{label}-akkd.f32")), &akk).unwrap();
        native.push((aqk, akk));
    }
    let mut variants = Vec::new();
    for entry in plan["variants"].as_array().unwrap() {
        let precision = entry["precision"].as_str().unwrap();
        assert!(["tf32", "tf32x3", "ieee"].contains(&precision));
        let library_path = entry["library"].as_str().unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(std::fs::read(library_path).unwrap())), entry["sha256"].as_str().unwrap());
        let compiled: serde_json::Value = serde_json::from_slice(&std::fs::read(entry["metadata"].as_str().unwrap()).unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(std::fs::read(entry["metadata"].as_str().unwrap()).unwrap())), entry["metadata_sha256"].as_str().unwrap());
        assert_eq!(compiled["heads"].as_u64().unwrap() as usize, heads);
        assert!(!compiled["effective_dot_precisions"].as_array().unwrap().is_empty());
        assert!(compiled["effective_dot_precisions"].as_array().unwrap().iter().all(|v| v.as_str() == Some(precision)));
        assert_eq!(compiled["requested_dot_precision"].as_str(), Some(precision));
        let anchor = compiled["metadata"]["rescaling_anchor"].as_str().unwrap();
        assert!(["native", "first", "causal-center"].contains(&anchor));
        // BEGIN bounded configured-gate diagnostic admission
        if anchor == "causal-center" {
            let file = plan["gate_bound_descriptor"].as_str().unwrap();
            let bytes = std::fs::read(file).unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), plan["gate_bound_descriptor_sha256"].as_str().unwrap());
            let source: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(source["heads"].as_u64().unwrap() as usize, heads);
            assert_eq!(source["dim"].as_u64().unwrap() as usize, dim);
            let bound = source["gate_lower_bound"].as_f64().unwrap();
            assert!(bound.is_finite() && bound < 0.);
            assert_eq!(compiled["metadata"]["configured_gate_lower_bound"].as_f64(), Some(bound));
        }
        // END bounded configured-gate diagnostic admission
        let tag = format!("{precision}-{anchor}");
        let library = unsafe { libloading::Library::new(library_path).unwrap() };
        let init = unsafe { *library.get::<unsafe extern "C" fn() -> i32>(b"krasis_fla_init\0").unwrap() };
        let cleanup = unsafe { *library.get::<unsafe extern "C" fn()>(b"krasis_fla_cleanup\0").unwrap() };
        let function = unsafe { *library.get::<FlaKdaIntraFn>(format!("krasis_fla_chunk_kda_fwd_kernel_intra_sub_chunk_h{heads}\0").as_bytes()).unwrap() };
        assert_eq!(unsafe { init() }, 0, "isolated diagnostic sidecar initialization");
        let mut values = Vec::new();
        let mut timings = Vec::new();
        for (index, (label, tokens)) in [("full", rows), ("prefix", prefix)].into_iter().enumerate() {
            launch_intra(function, tokens);
            self::cuda_sync();
            let aqk = ctx.download_bf16(&daqk);
            let akk = ctx.download_f32(&dakk);
            if precision == "tf32" && anchor == "native" {
                assert_eq!(&aqk[..tokens * heads * block * 2], &native[index].0[..tokens * heads * block * 2], "local native TF32 {label}");
                assert_eq!(&akk[..tokens * heads * subchunk * 4], &native[index].1[..tokens * heads * subchunk * 4], "local native TF32 diagonal {label}");
            }
            for bytes in aqk.chunks_exact(2) {
                assert!(half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32().is_finite(), "nonfinite {tag}/{label} matrix");
            }
            for bytes in akk.chunks_exact(4) {
                assert!(f32::from_le_bytes(bytes.try_into().unwrap()).is_finite(), "nonfinite {tag}/{label} diagonal");
            }
            for _ in 0..plan["timing_warmup"].as_u64().unwrap() { launch_intra(function, tokens); }
            let events = DsaMlaGatherAutotuneEvents::new().unwrap();
            let mut samples = Vec::new();
            for _ in 0..repeats {
                unsafe { assert_eq!(cuda_sys::lib().cuEventRecord(events.start, ctx.stream()), cuda_sys::CUresult::CUDA_SUCCESS); }
                launch_intra(function, tokens);
                unsafe {
                    assert_eq!(cuda_sys::lib().cuEventRecord(events.stop, ctx.stream()), cuda_sys::CUresult::CUDA_SUCCESS);
                    assert_eq!(cuda_sys::lib().cuEventSynchronize(events.stop), cuda_sys::CUresult::CUDA_SUCCESS);
                    let mut ms = 0.;
                    assert_eq!(cuda_sys::lib().cuEventElapsedTime(&mut ms, events.start, events.stop), cuda_sys::CUresult::CUDA_SUCCESS);
                    assert!(ms > 0. && ms.is_finite());
                    samples.push(ms);
                }
            }
            assert_eq!(ctx.download_bf16(&daqk), aqk, "timing launches changed matrix");
            assert_eq!(ctx.download_f32(&dakk), akk, "timing launches changed diagonal");
            std::fs::write(out.join(format!("{tag}-{label}-aqk.bf16")), &aqk).unwrap();
            std::fs::write(out.join(format!("{tag}-{label}-akkd.f32")), &akk).unwrap();
            timings.push(serde_json::json!({"tokens":tokens,"cuda_event_kernel_ms":samples,"scope":"isolated single captured block, not model speed"}));
            values.push((aqk, akk));
        }
        unsafe { cleanup(); }
        variants.push(serde_json::json!({"precision":precision,"rescaling_anchor":anchor,"tag":tag,"library":entry,"compiled":compiled,"timings":timings,
            "shared_prefix":matched_binary_metrics(&values[0].0[..selected], &values[1].0[..selected], "bf16", &out.join(format!("{tag}-prefix-mismatches.u64"))),
            "shared_diagonal":matched_binary_metrics(&values[0].1[..diagonal], &values[1].1[..diagonal], "f32", &out.join(format!("{tag}-diagonal-mismatches.u64")))}));
    }
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&serde_json::json!({"diagnostic_only":true,"qualifies_candidate":false,"plan":plan,"inputs":identities,"short":sm,"long":lm,"variants":variants,"local_native_tf32_controls_exact":true,
        "limitations":["Only intra stage and same captured block; no complete model or prefix-invariant redesign","CUDA event samples measure isolated block, not prefill throughput","No package substitution or precision adoption"]})).unwrap()).unwrap();
}
// END bounded isolated KDA precision replay

fn matched_binary_metrics(
    left: &[u8],
    right: &[u8],
    dtype: &str,
    coordinates: &Path,
) -> serde_json::Value {
    use std::io::Write;
    assert_eq!(left.len(), right.len());
    let element = match dtype {
        "bf16" => 2,
        "f32" => 4,
        _ => panic!("unsupported metric dtype"),
    };
    assert_eq!(left.len() % element, 0);
    let mut file = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(coordinates)
            .unwrap(),
    );
    let decode = |b: &[u8]| -> f64 {
        if element == 2 {
            half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64()
        } else {
            f32::from_le_bytes(b.try_into().unwrap()) as f64
        }
    };
    let (mut diff2, mut ref2, mut max, mut max_at, mut count, mut first) =
        (0f64, 0f64, 0f64, 0usize, 0u64, None);
    for (i, (a, b)) in left
        .chunks_exact(element)
        .zip(right.chunks_exact(element))
        .enumerate()
    {
        let (av, bv) = (decode(a), decode(b));
        assert!(
            av.is_finite() && bv.is_finite(),
            "nonfinite metric coordinate {i}"
        );
        let delta = av - bv;
        diff2 += delta * delta;
        ref2 += bv * bv;
        if delta.abs() > max {
            max = delta.abs();
            max_at = i;
        }
        if a != b {
            file.write_all(&(i as u64).to_le_bytes()).unwrap();
            count += 1;
            first.get_or_insert(i);
        }
    }
    file.flush().unwrap();
    assert!(
        ref2 > 0.,
        "relative-L2 is undefined for zero reference norm"
    );
    serde_json::json!({"elements":left.len()/element,"dtype":dtype,"bit_mismatches":count,"first_mismatch":first,
            "mismatch_coordinates":coordinates.file_name().unwrap().to_string_lossy(),"coordinate_dtype":"u64-le",
            "max_abs":max,"max_abs_coordinate":max_at,"difference_l2":diff2.sqrt(),"reference_l2":ref2.sqrt(),"relative_l2":(diff2/ref2).sqrt()})
}

#[cfg(has_decode_kernels)]
#[test]
#[ignore = "requires complete binary captures and an idle selected GPU through ./dev matched-kda"]
fn test_matched_kda_prefix_from_identical_real_capture() {
    use sha2::{Digest, Sha256};
    assert_eq!(std::env::var("KRASIS_DEV_SCRIPT").as_deref(), Ok("1"));
    assert!(
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok(),
        "select the captured GPU UUID explicitly"
    );
    let input =
        PathBuf::from(std::env::var("KRASIS_MATCHED_KDA_INPUT").expect("capture directory"));
    let out =
        PathBuf::from(std::env::var("KRASIS_MATCHED_KDA_OUTPUT").expect("fresh output directory"));
    std::fs::create_dir(&out).unwrap();
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input.join("descriptor.json")).unwrap()).unwrap();
    let height = descriptor["height"].as_u64().unwrap() as usize;
    let prefix = descriptor["row"].as_u64().unwrap() as usize + 1;
    let heads = descriptor["heads"].as_u64().unwrap() as usize;
    let dim = descriptor["dim"].as_u64().unwrap() as usize;
    assert!(prefix > 0 && prefix <= height);
    let mut identities = serde_json::Map::new();
    for name in [
        "descriptor.json",
        "q.bf16",
        "k.bf16",
        "v.bf16",
        "gates.bf16",
        "beta.bf16",
        "a-log.f32",
        "dt-bias.f32",
        "initial-state.f32",
    ] {
        let bytes = std::fs::read(input.join(name)).unwrap();
        identities.insert(name.into(),serde_json::json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    std::fs::write(
        out.join("INPUT_IDENTITIES.json"),
        serde_json::to_vec_pretty(&identities).unwrap(),
    )
    .unwrap();
    let mut heights = vec![prefix, height];
    heights.dedup();
    for tokens in heights {
        let [chunk, oracle, state, oracle_state] = matched_kda_fixture(
            &input,
            tokens,
            &out.join(format!("height{tokens}-first-chunk")),
            None,
            None,
        );
        for (name, bytes) in [
            ("chunk.bf16", &chunk),
            ("oracle.bf16", &oracle),
            ("chunk-state.f32", &state),
            ("oracle-state.f32", &oracle_state),
        ] {
            std::fs::write(out.join(format!("height{tokens}-{name}")), bytes).unwrap();
        }
        let report = serde_json::json!({"height":tokens,"heads":heads,"dim":dim,"descriptor":descriptor,
                "output":matched_binary_metrics(&chunk,&oracle,"bf16",&out.join(format!("height{tokens}-output-mismatches.u64"))),
                "final_state":matched_binary_metrics(&state,&oracle_state,"f32",&out.join(format!("height{tokens}-state-mismatches.u64"))),
                "scope":"quantified production chunk chain versus independent serial recurrence; collection completion does not relax model equality"});
        std::fs::write(
            out.join(format!("height{tokens}-REPORT.json")),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    let bytes = prefix * heads * dim * 2;
    let short = std::fs::read(out.join(format!("height{prefix}-chunk.bf16"))).unwrap();
    let long = std::fs::read(out.join(format!("height{height}-chunk.bf16"))).unwrap();
    let short_oracle = std::fs::read(out.join(format!("height{prefix}-oracle.bf16"))).unwrap();
    let long_oracle = std::fs::read(out.join(format!("height{height}-oracle.bf16"))).unwrap();
    let report = serde_json::json!({"prefix":prefix,"height":height,"height_control_performed":prefix < height,"inputs_and_initial_state_identical_by_construction":true,
            "chunk_prefix":matched_binary_metrics(&long[..bytes],&short,"bf16",&out.join("chunk-prefix-mismatches.u64")),
            "serial_prefix":matched_binary_metrics(&long_oracle[..bytes],&short_oracle,"bf16",&out.join("serial-prefix-mismatches.u64")),
            "scope":"same real captured Q/K/V/gates/beta/initial state; no upstream projection differences"});
    std::fs::write(
        out.join("PREFIX_REPORT.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

// BEGIN bounded causal-reference KDA chain diagnostic
// Test-only owner; initialization and teardown use the fixture's live context.
struct MatchedKdaDiagnosticIntra {
    _library: libloading::Library,
    function: FlaKdaIntraFn,
    cleanup: unsafe extern "C" fn(),
}
impl MatchedKdaDiagnosticIntra {
    fn load(entry: &serde_json::Value, heads: usize, descriptor: &serde_json::Value, ctx: &GpuTestCtx) -> Self {
        use sha2::{Digest, Sha256};
        let library_path = entry["library"].as_str().unwrap();
        let metadata_path = entry["metadata"].as_str().unwrap();
        let library_bytes = std::fs::read(library_path).unwrap();
        let metadata_bytes = std::fs::read(metadata_path).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&library_bytes)), entry["sha256"].as_str().unwrap());
        assert_eq!(format!("{:x}", Sha256::digest(&metadata_bytes)), entry["metadata_sha256"].as_str().unwrap());
        let compiled: serde_json::Value = serde_json::from_slice(&metadata_bytes).unwrap();
        let precision = entry["precision"].as_str().unwrap();
        assert!(["tf32", "tf32x3", "ieee"].contains(&precision));
        assert_eq!(compiled["artifact_scope"].as_str(), Some("diagnostic-only-single-intra"));
        assert_eq!(compiled["qualifies_candidate"].as_bool(), Some(false));
        assert_eq!(compiled["heads"].as_u64().unwrap() as usize, heads);
        let anchor = entry["rescaling_anchor"].as_str().unwrap();
        assert!(["first", "causal-center"].contains(&anchor));
        assert_eq!(compiled["metadata"]["rescaling_anchor"].as_str(), Some(anchor));
        if anchor == "causal-center" {
            let bound = descriptor["gate_lower_bound"].as_f64().unwrap();
            assert!(bound.is_finite() && bound < 0.);
            assert_eq!(compiled["metadata"]["configured_gate_lower_bound"].as_f64(), Some(bound));
        }
        assert_eq!(compiled["requested_dot_precision"].as_str(), Some(precision));
        let dots = compiled["effective_dot_precisions"].as_array().unwrap();
        assert!(!dots.is_empty() && dots.iter().all(|v| v.as_str() == Some(precision)));
        let mut device = 0;
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            assert_eq!(cuda_sys::lib().cuDeviceGet(&mut device, ctx.dev.ordinal() as i32), cuda_sys::CUresult::CUDA_SUCCESS);
            assert_eq!(cuda_sys::lib().cuDeviceGetAttribute(&mut major, cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), cuda_sys::CUresult::CUDA_SUCCESS);
            assert_eq!(cuda_sys::lib().cuDeviceGetAttribute(&mut minor, cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), cuda_sys::CUresult::CUDA_SUCCESS);
        }
        let arch = major * 10 + minor;
        let compiled_arch = compiled["arch"].as_i64().unwrap() as i32;
        // Explicit diagnostic admission for exact/same-generation base cubins.
        // There is no missing-library/unsupported-architecture fallback.
        assert!(compiled_arch == arch || (compiled_arch == major * 10 && minor > 0));
        let library = unsafe { libloading::Library::new(library_path).unwrap() };
        let init = unsafe { *library.get::<unsafe extern "C" fn() -> i32>(b"krasis_fla_init\0").unwrap() };
        let cleanup = unsafe { *library.get::<unsafe extern "C" fn()>(b"krasis_fla_cleanup\0").unwrap() };
        let function = unsafe { *library.get::<FlaKdaIntraFn>(format!("krasis_fla_chunk_kda_fwd_kernel_intra_sub_chunk_h{heads}\0").as_bytes()).unwrap() };
        assert_eq!(unsafe { init() }, 0, "diagnostic-only intra initialization");
        Self { _library: library, function, cleanup }
    }
}
impl Drop for MatchedKdaDiagnosticIntra {
    fn drop(&mut self) { unsafe { (self.cleanup)(); } }
}

#[test]
#[ignore]
fn test_matched_kda_causal_chain_from_real_capture() {
    assert_eq!(std::env::var("KRASIS_DEV_SCRIPT").as_deref(), Ok("1"));
    let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(std::env::var("KRASIS_MATCHED_KDA_CAUSAL_CHAIN_PLAN").unwrap()).unwrap()).unwrap();
    let input = std::path::PathBuf::from(plan["capture"].as_str().unwrap());
    let original = std::path::PathBuf::from(plan["original_replay"].as_str().unwrap());
    let out = std::path::PathBuf::from(plan["output"].as_str().unwrap());
    assert!(!out.exists(), "fresh complete-chain directory required");
    std::fs::create_dir(&out).unwrap();
    let descriptor: serde_json::Value = serde_json::from_slice(&std::fs::read(input.join("descriptor.json")).unwrap()).unwrap();
    let prefix = descriptor["row"].as_u64().unwrap() as usize + 1;
    let height = descriptor["height"].as_u64().unwrap() as usize;
    let heads = descriptor["heads"].as_u64().unwrap() as usize;
    let dim = descriptor["dim"].as_u64().unwrap() as usize;
    assert!(prefix < height);
    let bytes = prefix * heads * dim * 2;
    let names = ["chunk.bf16", "oracle.bf16", "chunk-state.f32", "oracle-state.f32"];
    let mut native = Vec::new();
    for tokens in [height, prefix] {
        let label = format!("native-height{tokens}");
        let values = matched_kda_fixture(&input, tokens, &out.join(&label), None, None);
        for (index, name) in names.iter().enumerate() {
            assert_eq!(values[index], std::fs::read(original.join(format!("height{tokens}-{name}"))).unwrap(), "untouched native/serial control {label}/{name}");
        }
        native.push(values);
    }
    let mut variants = Vec::new();
    for entry in plan["variants"].as_array().unwrap() {
        let precision = entry["precision"].as_str().unwrap();
        let mut values = Vec::new();
        let anchor = entry["rescaling_anchor"].as_str().unwrap();
        let tag = format!("{precision}-{anchor}");
        for (index, tokens) in [height, prefix].into_iter().enumerate() {
            let label = format!("{tag}-height{tokens}");
            let v = matched_kda_fixture(&input, tokens, &out.join(&label), None, Some(entry));
            assert_eq!(v[1], native[index][1], "raw serial output changed under intra-only diagnostic");
            assert_eq!(v[3], native[index][3], "raw serial final state changed under intra-only diagnostic");
            for b in v[0].chunks_exact(2) {
                assert!(half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32().is_finite(), "nonfinite native diagnostic output");
            }
            for b in v[2].chunks_exact(4) {
                assert!(f32::from_le_bytes(b.try_into().unwrap()).is_finite(), "nonfinite native diagnostic final state");
            }
            std::fs::write(out.join(format!("{label}-output.bf16")), &v[0]).unwrap();
            std::fs::write(out.join(format!("{label}-final-state.f32")), &v[2]).unwrap();
            values.push(v);
        }
        variants.push(serde_json::json!({"precision":precision,"rescaling_anchor":entry["rescaling_anchor"],"library":entry,
            "shared_prefix":matched_binary_metrics(&values[0][0][..bytes], &values[1][0], "bf16", &out.join(format!("{tag}-prefix-mismatches.u64"))),
            "full_vs_native":matched_binary_metrics(&values[0][0], &native[0][0], "bf16", &out.join(format!("{tag}-full-native-mismatches.u64"))),
            "prefix_vs_native":matched_binary_metrics(&values[1][0], &native[1][0], "bf16", &out.join(format!("{tag}-short-native-mismatches.u64"))),
            "raw_serial_controls_exact":true}));
    }
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&serde_json::json!({"diagnostic_only":true,"qualifies_candidate":false,"plan":plan,"descriptor":descriptor,
        "original_full_prefix_native_and_serial_controls_exact":true,"variants":variants,
        "limitations":["First-layer standalone native chain with identical captured raw operands; not full-model word-reversal repair","Independent recurrence must use each treatment own BF16 matching-window starting state","Full10564 and prefix10500 final states represent different boundaries and must not be compared as equality","No model speed or dynamic-range qualification"]})).unwrap()).unwrap();
}
// END bounded causal-reference KDA chain diagnostic

// BEGIN bounded context-owned KDA model module test
#[test]
#[ignore]
fn test_model_kda_context_owned_cubin_from_real_capture() {
    assert_eq!(std::env::var("KRASIS_DEV_SCRIPT").as_deref(), Ok("1"));
    let plan:serde_json::Value=serde_json::from_slice(&std::fs::read(std::env::var("KRASIS_MATCHED_KDA_MODULE_PLAN").unwrap()).unwrap()).unwrap();
    let ctx=GpuTestCtx::new();
    let desc:serde_json::Value=serde_json::from_slice(&std::fs::read(plan["descriptor"].as_str().unwrap()).unwrap()).unwrap();
    let (heads,dim,lower)=(desc["heads"].as_u64().unwrap() as usize,desc["dim"].as_u64().unwrap() as usize,desc["gate_lower_bound"].as_f64().unwrap() as f32);
    std::env::set_var("KRASIS_MATCHED_KDA_INTRA_CONFIG",plan["module_plan"].as_str().unwrap());
    std::env::set_var("KRASIS_MATCHED_PROBE_CONFIG",plan["probe_plan"].as_str().unwrap());
    let module=crate::matched_kda_model::ModelKdaDiagnostic::load(ctx.dev.clone(),&[(heads,dim,lower)]).unwrap().unwrap();
    let input=Path::new(plan["window"].as_str().unwrap());
    let metadata:serde_json::Value=serde_json::from_slice(&std::fs::read(input.join("matching-window.json")).unwrap()).unwrap();
    let rows=metadata["rows"].as_u64().unwrap() as usize;
    assert_eq!(metadata["heads"].as_u64().unwrap() as usize,heads);
    assert_eq!(metadata["dim"].as_u64().unwrap() as usize,dim);
    let dq=ctx.upload_bf16(&std::fs::read(input.join("matching-window-prepared-q.bf16")).unwrap());
    let dk=ctx.upload_bf16(&std::fs::read(input.join("matching-window-prepared-k.bf16")).unwrap());
    let dg=ctx.upload_f32(&std::fs::read(input.join("matching-window-gate-cumsum.f32")).unwrap());
    let db=ctx.upload_bf16(&std::fs::read(input.join("matching-window-prepared-beta.bf16")).unwrap());
    let da=ctx.alloc_bf16(rows*heads*FLA_PREFILL_BLOCK_TOKENS);
    let dd=ctx.alloc_f32(rows*heads*16);
    for (label,tokens) in [("full",rows),("prefix",plan["prefix_rows"].as_u64().unwrap() as usize)] {
        unsafe {module.launch((heads,dim,lower),*dq.device_ptr(),*dk.device_ptr(),*dg.device_ptr(),*db.device_ptr(),*da.device_ptr(),*dd.device_ptr(),desc["q_scale"].as_f64().unwrap() as f32,tokens as i32,ctx.stream()).unwrap();}
        self::cuda_sync();
        let expected=Path::new(plan["expected"].as_str().unwrap());
        assert_eq!(ctx.download_bf16(&da),std::fs::read(expected.join(format!("tf32-causal-center-{label}-aqk.bf16"))).unwrap());
        assert_eq!(ctx.download_f32(&dd),std::fs::read(expected.join(format!("tf32-causal-center-{label}-akkd.f32"))).unwrap());
    }
    drop(module);
    std::env::remove_var("KRASIS_MATCHED_KDA_INTRA_CONFIG");
    std::env::remove_var("KRASIS_MATCHED_PROBE_CONFIG");
}
// END bounded context-owned KDA model module test
