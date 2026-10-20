// Private matched operator measurement; no serving dispatch or arithmetic changes.
mod operator {
    include!("row-projection-probe.rs");
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ProfilePlan {
        operator: ProbePlan,
        materialized: Identity,
        cublas_contract: Identity,
        production_output: Identity,
        native_output: Identity,
        warmups: usize,
        pairs: usize,
        require_original_baseline: bool,
    }
    fn samples(xs: &[f64]) -> Result<Value, String> {
        if xs.is_empty() || xs.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err("invalid measured samples".into());
        }
        let mut s = xs.to_vec();
        s.sort_by(f64::total_cmp);
        Ok(
            json!({"seconds":xs,"count":s.len(),"minimum":s[0],"median":s[s.len()/2],"maximum":s[s.len()-1]}),
        )
    }
    pub(super) fn paired(plan: &Path, out: &Path) -> Result<Value, String> {
        let profile: ProfilePlan =
            serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if profile.warmups == 0 || profile.pairs == 0 {
            return Err("measurement requires explicit positive warmup/pairs".into());
        }
        let p = profile.operator;
        if p.schema != 1 || p.safety_margin_bytes != 600 * 1024 * 1024 {
            return Err("explicit schema and user 600 MiB margin required".into());
        }
        p.source.read()?;
        let built: Value =
            serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
        if built["plan"]["source"] != serde_json::to_value(&p.source).map_err(|e| e.to_string())?
            || built["ptx"] != serde_json::to_value(&p.ptx).map_err(|e| e.to_string())?
        {
            return Err("PTX/source/build provenance mismatch".into());
        }
        let dispatch: Value =
            serde_json::from_slice(&p.dispatch.read()?).map_err(|e| e.to_string())?;
        if dispatch["hqq6_selected_vec4_warps"].as_u64() != Some(p.measured_decode_warps as u64)
            || dispatch["runtime_stage"] != "decode"
            || dispatch["kernel_ready"] != true
        {
            return Err("measured dispatch does not match explicit warp admission".into());
        }
        let ptx = p.ptx.read()?;
        let read = |k: &str| {
            p.capture
                .get(k)
                .ok_or(format!("missing identity {k}"))?
                .read()
        };
        let desc: Value =
            serde_json::from_slice(&read("descriptor")?).map_err(|e| e.to_string())?;
        let (r, k, g, ps, ss, zs, h, row) = dims(&desc)?;
        validate_cases(&p.cases, h)?;
        for (key, value) in [
            ("rows", r),
            ("cols", k),
            ("group_size", g),
            ("packed_row_stride_bytes", ps),
            ("scales_row_stride_bytes", ss),
            ("zeros_row_stride_bytes", zs),
        ] {
            if dispatch[key].as_u64() != Some(value as u64) {
                return Err(format!("dispatch geometry differs: {key}"));
            }
        }
        let packed = read("packed")?;
        let sb = read("scales")?;
        let zb = read("zeros")?;
        let xb = read("full_input")?;
        let selected = read("selected_input")?;
        let nb = read("native_output")?;
        for (actual, expected) in [
            (packed.len(), mul(r, ps)?),
            (sb.len(), mul(r, ss)?),
            (zb.len(), mul(r, zs)?),
            (xb.len(), mul(mul(h, k)?, 2)?),
            (selected.len(), mul(k, 2)?),
            (nb.len(), mul(r, 2)?),
        ] {
            if actual != expected {
                return Err("actual captured allocation extent mismatch".into());
            }
        }
        let start = mul(mul(row, k)?, 2)?;
        if xb[start..start + mul(k, 2)?] != selected {
            return Err("selected row differs from authentic controlled full input".into());
        }
        let scales = f32s(&sb)?;
        let zeros = f32s(&zb)?;
        let x = u16s(&xb)?;
        let native = u16s(&nb)?;
        let contract: Value =
            serde_json::from_slice(&profile.cublas_contract.read()?).map_err(|e| e.to_string())?;
        if contract["m"] != r
            || contract["n"] != h
            || contract["k"] != k
            || contract["transpose_a"] != "T"
            || contract["transpose_b"] != "N"
            || contract["lda"] != k
            || contract["ldb"] != k
            || contract["ldc"] != r
            || contract["alpha"] != 1.0
            || contract["beta"] != 0.0
            || contract["compute"] != "CUBLAS_COMPUTE_32F"
            || contract["algorithm"] != "CUBLAS_GEMM_DEFAULT"
            || contract["pointer_mode"] != 0
            || contract["stream_matches_production"] != true
            || contract["input_dtype"] != "bf16"
            || contract["weights_dtype"] != "bf16"
            || contract["output_dtype"] != "bf16"
        {
            return Err("actual Q baseline contract differs".into());
        }
        let weights = u16s(&profile.materialized.read()?)?;
        let captured = u16s(&profile.production_output.read()?)?;
        let expected = u16s(&profile.native_output.read()?)?;
        if weights.len() != mul(r, k)?
            || captured.len() != mul(r, h)?
            || expected.len() != mul(r, h)?
            || expected[row * r..(row + 1) * r] != native
        {
            return Err("complete reference extents/selected native differ".into());
        }
        let dev = match contract["stream_is_null"].as_bool() {
            Some(true) => CudaDevice::new(p.gpu_ordinal),
            Some(false) => CudaDevice::new_with_stream(p.gpu_ordinal),
            None => return Err("actual stream kind absent".into()),
        }
        .map_err(|e| e.to_string())?;
        let mut uuid = unsafe { std::mem::zeroed::<cudarc::driver::sys::CUuuid>() };
        unsafe {
            cudarc::driver::sys::lib()
                .cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device())
                .result()
                .map_err(|e| e.to_string())?;
        }
        let hex = uuid
            .bytes
            .iter()
            .map(|&b| format!("{:02x}", b as u8))
            .collect::<String>();
        let actual_uuid = format!(
            "GPU-{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        );
        if actual_uuid != p.gpu_uuid {
            return Err("actual GPU UUID differs from frozen admission".into());
        }
        use cudarc::driver::sys::CUdevice_attribute as A;
        let attr = |a| dev.attribute(a).map_err(|e| e.to_string());
        let cc = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
            + attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        if cc < 80
            || cc as u32 != p.compute_capability
            || dev.name().map_err(|e| e.to_string())? != p.gpu_name
        {
            return Err("actual device does not match explicit admission".into());
        }
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ]
        .map(|v| v as usize);
        let config = grid(r, h, p.measured_decode_warps, limits, mul(k, 2)?)?;
        let before = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
        dev.load_ptx(
            cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx).map_err(|e| e.to_string())?),
            "row_operator",
            &["hqq6_decode_gemv_bf16_vec4", "hqq6_batch_gemv_bf16_vec4"],
        )
        .map_err(|e| e.to_string())?;

        let batch = dev
            .get_func("row_operator", "hqq6_batch_gemv_bf16_vec4")
            .ok_or("batch kernel missing")?;
        let common_bytes = add(xb.len(), mul(mul(h, r)?, 2)?)?;
        let native_weight_bytes = add(packed.len(), add(sb.len(), zb.len())?)?;
        let baseline_weight_bytes = mul(weights.len(), 2)?;
        let combined_tensor_bytes = add(
            common_bytes,
            add(native_weight_bytes, baseline_weight_bytes)?,
        )?;
        let after_module = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if after_module < add(combined_tensor_bytes, p.safety_margin_bytes)? {
            return Err("actual combined tensors breach600MiB".into());
        }
        let dp = dev.htod_copy(packed).map_err(|e| e.to_string())?;
        let ds = dev.htod_copy(scales).map_err(|e| e.to_string())?;
        let dz = dev.htod_copy(zeros).map_err(|e| e.to_string())?;
        let dx = dev.htod_copy(x).map_err(|e| e.to_string())?;
        let dw = dev.htod_copy(weights).map_err(|e| e.to_string())?;
        let dy = dev
            .alloc_zeros::<u16>(mul(h, r)?)
            .map_err(|e| e.to_string())?;
        let _completion = Completion(&dev);
        use cudarc::cublas::{result as r, sys as c};
        let handle = r::create_handle().map_err(|e| e.to_string())?;
        let result = (|| -> Result<Value, String> {
            unsafe {
                r::set_stream(handle, *dev.cu_stream() as c::cudaStream_t)
                    .map_err(|e| e.to_string())?;
                let check = |name: &str, status: c::cublasStatus_t| -> Result<(), String> {
                    if status == c::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        Ok(())
                    } else {
                        Err(format!("paired actual Q cuBLAS {name}: {status:?}"))
                    }
                };
                let mut version = 0i32;
                check(
                    "version",
                    c::lib().cublasGetVersion_v2(handle, &mut version),
                )?;
                if contract["version"] != version {
                    return Err("cuBLAS version does not match capture".into());
                }
                let math = match contract["math_mode"].as_u64() {
                    Some(0) => c::cublasMath_t::CUBLAS_DEFAULT_MATH,
                    Some(1) => c::cublasMath_t::CUBLAS_TENSOR_OP_MATH,
                    Some(2) => c::cublasMath_t::CUBLAS_PEDANTIC_MATH,
                    Some(3) => c::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                    Some(16) => c::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
                    _ => {
                        return Err(
                            "unsupported captured cuBLAS math flags; no substitute contract".into(),
                        )
                    }
                };
                let atomics = match contract["atomics_mode"].as_u64() {
                    Some(0) => c::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED,
                    Some(1) => c::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED,
                    _ => return Err("invalid captured cuBLAS atomics mode".into()),
                };
                let sm = contract["sm_count_target"]
                    .as_i64()
                    .and_then(|n| i32::try_from(n).ok())
                    .filter(|&n| n >= 0)
                    .ok_or("invalid captured cuBLAS SM target")?;
                if std::env::var("CUBLAS_WORKSPACE_CONFIG").ok()
                    != contract["workspace_environment"]
                        .as_str()
                        .map(str::to_owned)
                {
                    return Err("workspace environment differs from captured contract".into());
                }
                check("math", c::lib().cublasSetMathMode(handle, math))?;
                check("atomics", c::lib().cublasSetAtomicsMode(handle, atomics))?;
                check(
                    "pointer",
                    c::lib().cublasSetPointerMode_v2(
                        handle,
                        c::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                    ),
                )?;
                check("SM target", c::lib().cublasSetSmCountTarget(handle, sm))?;
            }
            let actual_contract = unsafe {
                let mut math = 0u32;
                let mut atomics = c::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED;
                let mut pointer = c::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
                let mut sm = 0i32;
                let mut stream = std::ptr::null_mut();
                let statuses = [
                    c::lib().cublasGetMathMode(
                        handle,
                        (&mut math as *mut u32).cast::<c::cublasMath_t>(),
                    ),
                    c::lib().cublasGetAtomicsMode(handle, &mut atomics),
                    c::lib().cublasGetPointerMode_v2(handle, &mut pointer),
                    c::lib().cublasGetSmCountTarget(handle, &mut sm),
                    c::lib().cublasGetStream_v2(handle, &mut stream),
                ];
                if statuses
                    .iter()
                    .any(|s| *s != c::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                    || contract["math_mode"] != math
                    || contract["atomics_mode"] != atomics as u32
                    || contract["pointer_mode"] != pointer as u32
                    || contract["sm_count_target"] != sm
                    || stream as usize != *dev.cu_stream() as usize
                    || contract["stream_is_null"] != stream.is_null()
                {
                    return Err("actual cuBLAS readback/stream differs".into());
                }
                json!({"math_mode":math,"atomics_mode":atomics as u32,"pointer_mode":pointer as u32,"sm_count_target":sm,"stream_matches_device":true,"stream_is_null":stream.is_null()})
            };
            let baseline = || -> Result<(), String> {
                unsafe {
                    let alpha = 1f32;
                    let beta = 0f32;
                    r::gemm_ex(
                        handle,
                        c::cublasOperation_t::CUBLAS_OP_T,
                        c::cublasOperation_t::CUBLAS_OP_N,
                        r as i32,
                        h as i32,
                        k as i32,
                        (&alpha as *const f32).cast(),
                        (*dw.device_ptr() as *const std::ffi::c_void),
                        c::cudaDataType::CUDA_R_16BF,
                        k as i32,
                        (*dx.device_ptr() as *const std::ffi::c_void),
                        c::cudaDataType::CUDA_R_16BF,
                        k as i32,
                        (&beta as *const f32).cast(),
                        (*dy.device_ptr() as *mut std::ffi::c_void),
                        c::cudaDataType::CUDA_R_16BF,
                        r as i32,
                        c::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        c::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .map_err(|e| e.to_string())?;
                }
                Ok(())
            };
            let native_launch = || -> Result<(), String> {
                unsafe {
                    batch
                        .clone()
                        .launch(
                            config,
                            (
                                *dp.device_ptr(),
                                *ds.device_ptr(),
                                *dz.device_ptr(),
                                *dx.device_ptr(),
                                *dy.device_ptr(),
                                r as i32,
                                k as i32,
                                g as i32,
                                ps as i32,
                                ss as i32,
                                zs as i32,
                                h as u64,
                            ),
                        )
                        .map_err(|e| e.to_string())?
                }
                Ok(())
            };
            let sample = |native: bool| -> Result<f64, String> {
                dev.synchronize().map_err(|e| e.to_string())?;
                let start = Instant::now();
                if native {
                    native_launch()?
                } else {
                    baseline()?
                }
                dev.synchronize().map_err(|e| e.to_string())?;
                Ok(start.elapsed().as_secs_f64())
            };
            baseline()?;
            dev.synchronize().map_err(|e| e.to_string())?;
            let baseline_first = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
            let baseline_id = save(out, "baseline-first.bf16", &baseline_first)?;
            let baseline_diff = baseline_first
                .iter()
                .zip(&captured)
                .filter(|(a, b)| a != b)
                .count();
            if profile.require_original_baseline && baseline_diff != 0 {
                return Err(format!(
                    "original-device full baseline differs at {baseline_diff} coordinates"
                ));
            }
            native_launch()?;
            dev.synchronize().map_err(|e| e.to_string())?;
            let native_first = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
            let native_id = save(out, "native-first.bf16", &native_first)?;
            if native_first != expected {
                return Err("full native differs from completed485 reference".into());
            }
            for _ in 0..profile.warmups {
                baseline()?;
                native_launch()?;
                dev.synchronize().map_err(|e| e.to_string())?
            }
            let mut minimum_free = cudarc::driver::result::mem_get_info()
                .map_err(|e| e.to_string())?
                .0;
            let (mut base_samples, mut native_samples) = (Vec::new(), Vec::new());
            for i in 0..profile.pairs {
                let free = cudarc::driver::result::mem_get_info()
                    .map_err(|e| e.to_string())?
                    .0;
                minimum_free = minimum_free.min(free);
                if free < p.safety_margin_bytes {
                    return Err("measurement runtime reserve violated".into());
                }
                if i % 2 == 0 {
                    base_samples.push(sample(false)?);
                    native_samples.push(sample(true)?)
                } else {
                    native_samples.push(sample(true)?);
                    base_samples.push(sample(false)?)
                }
            }
            baseline()?;
            dev.synchronize().map_err(|e| e.to_string())?;
            let base_last = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
            let base_last_id = save(out, "baseline-final.bf16", &base_last)?;
            native_launch()?;
            dev.synchronize().map_err(|e| e.to_string())?;
            let native_last = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
            let native_last_id = save(out, "native-final.bf16", &native_last)?;
            if base_last != baseline_first || native_last != expected {
                return Err("whole output changed across paired windows".into());
            }
            minimum_free = minimum_free.min(
                cudarc::driver::result::mem_get_info()
                    .map_err(|e| e.to_string())?
                    .0,
            );
            if minimum_free < p.safety_margin_bytes {
                return Err("final runtime reserve violated".into());
            }
            let refs = vec![
                profile.materialized,
                profile.cublas_contract,
                profile.production_output,
                profile.native_output,
            ];
            for x in refs {
                x.read()?;
            }
            for x in p.capture.values() {
                x.read()?;
            }
            p.source.read()?;
            p.ptx.read()?;
            p.build_report.read()?;
            p.dispatch.read()?;
            Ok(
                json!({"passed":true,"adopted":false,"device":{"uuid":actual_uuid,"cc":cc,"name":p.gpu_name},"shape":{"rows":r,"cols":k,"height":h},"warmups":profile.warmups,"pairs":profile.pairs,"cublas_contract":contract,"actual_handle_readback":actual_contract,"baseline_captured_difference_coordinates":baseline_diff,"native_complete_reference_exact":true,"repeat_outputs_exact":true,"baseline_host_launch_completion":samples(&base_samples)?,"unrounded_batch_host_launch_completion":samples(&native_samples)?,"memory":{"common_tensor_bytes":common_bytes,"baseline_weight_bytes":baseline_weight_bytes,"native_weight_bytes":native_weight_bytes,"combined_allocated_tensor_bytes":combined_tensor_bytes,"before_module_free":before.0,"after_module_free":after_module,"observed_minimum_free":minimum_free,"safety_margin_bytes":p.safety_margin_bytes},"outputs":[baseline_id,native_id,base_last_id,native_last_id],"limitations":["Standalone Rust host launch plus completion, not engine or HTTP throughput","Baseline excludes GPU materialization/sidecars; weights preloaded","No serving math/cache/ordinary acceptance","Free VRAM sample at completed boundaries is not CUDA low-water monitoring; wholemodel calibration remains separate"]}),
            )
        })();
        let completion = dev.synchronize().map_err(|e| e.to_string());
        let destroy = unsafe { r::destroy_handle(handle).map_err(|e| e.to_string()) };
        completion?;
        destroy?;
        result
    }
    #[cfg(test)]
    mod profile_tests {
        use super::*;
        #[test]
        fn measured_summary_preserves_samples_and_rejects_nonfinite() {
            let r = samples(&[3., 1., 2.]).unwrap();
            assert_eq!(r["seconds"], json!([3., 1., 2.]));
            assert_eq!(r["median"], 2.);
            assert!(samples(&[f64::NAN]).is_err());
            assert!(samples(&[0.]).is_err());
        }
    }
}
fn main() {
    let result = (|| -> Result<(), String> {
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
            return Err("Use ./dev row-projection-profile".into());
        }
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args.len() != 2 {
            return Err("frozen plan and fresh output required".into());
        }
        let out = std::path::Path::new(&args[1]);
        std::fs::create_dir(out).map_err(|e| e.to_string())?;
        match operator::paired(std::path::Path::new(&args[0]), out) {
            Ok(r) => {
                std::fs::write(
                    out.join("REPORT.json"),
                    serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                println!("Evidence saved {}", out.display());
                Ok(())
            }
            Err(e) => {
                std::fs::write(out.join("FAILURE.txt"), &e).map_err(|e| e.to_string())?;
                Err(e)
            }
        }
    })();
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
