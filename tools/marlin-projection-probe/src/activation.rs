use super::*;
use common_marlin::Program;
pub fn check(
    dev: &Arc<CudaDevice>,
    program: &Program,
    name: &str,
    values: &[u16],
    stride: usize,
    rows: usize,
    inter: usize,
    out: &Path,
) -> Result<(Value, usize), String> {
    let input = dev.htod_copy(values.to_vec()).map_err(|e| e.to_string())?;
    let inplace = dev.htod_copy(values.to_vec()).map_err(|e| e.to_string())?;
    let ystride = inter + 16;
    let output = dev
        .htod_copy(vec![0x7fc1u16; rows * ystride + 32])
        .map_err(|e| e.to_string())?;
    let _completion = Completion(dev.clone());
    let mut records = Vec::new();
    let mut raw = 0;
    for (kind, limit, raw_limit) in [
        (0u8, 0f32, false),
        (0, 0.125, false),
        (0, 0.125, true),
        (1, 0., false),
        (2, 0., false),
    ] {
        let key = format!("kind{kind}-limit{limit}-raw{raw_limit}");
        let launch = || unsafe {
            program.enqueue_activation(
                *input.device_ptr(),
                *output.device_ptr(),
                0,
                inter,
                kind,
                limit,
                raw_limit,
                stride,
                ystride,
                rows,
                *dev.cu_stream(),
            )
        };
        let mut baseline = None;
        for mode in ["full", "singletons", "graph", "error"] {
            unsafe {
                sys::lib().cuMemsetD16Async(
                    *output.device_ptr(),
                    0x7fc1,
                    rows * ystride + 32,
                    *dev.cu_stream(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            match mode {
                "full" => launch()?,
                "singletons" => {
                    for m in 0..rows {
                        unsafe {
                            program.enqueue_activation(
                                *input.device_ptr() + (m * stride * 2) as u64,
                                *output.device_ptr() + (m * ystride * 2) as u64,
                                0,
                                inter,
                                kind,
                                limit,
                                raw_limit,
                                stride,
                                ystride,
                                1,
                                *dev.cu_stream(),
                            )
                        }?;
                    }
                }
                "graph" => graph_owner::completed_graph_launch(dev, launch)?,
                _ => {
                    if graph_owner::completed_graph_launch(dev, || {
                        launch()?;
                        Err("intentional activation error".into())
                    })
                    .is_ok()
                    {
                        return Err("activation error accepted".into());
                    }
                    launch()?;
                }
            }
            let y = dev.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
            for m in 0..rows {
                if y[m * ystride + inter..(m + 1) * ystride]
                    .iter()
                    .any(|&v| v != 0x7fc1)
                {
                    return Err("activation row padding changed".into());
                }
            }
            if y[rows * ystride..].iter().any(|&v| v != 0x7fc1) {
                return Err("activation suffix changed".into());
            }
            if let Some(b) = &baseline {
                if *b != y {
                    return Err("activation placement/graph differs".into());
                }
            } else {
                baseline = Some(y);
            }
        }
        unsafe {
            sys::lib().cuMemcpyDtoDAsync_v2(
                *inplace.device_ptr(),
                *input.device_ptr(),
                values.len() * 2,
                *dev.cu_stream(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        unsafe {
            program.enqueue_activation(
                *inplace.device_ptr(),
                *inplace.device_ptr(),
                0,
                inter,
                kind,
                limit,
                raw_limit,
                stride,
                stride,
                rows,
                *dev.cu_stream(),
            )
        }?;
        let in_y = dev.dtoh_sync_copy(&inplace).map_err(|e| e.to_string())?;
        let y = baseline.unwrap();
        let mut nearest = 0;
        let mut max_steps = 0;
        let ordered = |b: u16| {
            if b & 0x8000 != 0 {
                0x8000i32 - (b & 0x7fff) as i32
            } else {
                0x8000 + (b as i32)
            }
        };
        for m in 0..rows {
            for i in 0..inter {
                if in_y[m * stride + i] != y[m * ystride + i] {
                    return Err("activation in-place differs".into());
                }
                let mut gate = f32::from_bits((values[m * stride + i] as u32) << 16) as f64;
                let reference = if kind == 1 {
                    gate = gate.max(0.);
                    gate * gate
                } else {
                    let mut up =
                        f32::from_bits((values[m * stride + inter + i] as u32) << 16) as f64;
                    if raw_limit {
                        gate = gate.min(limit as f64);
                        up = up.clamp(-(limit as f64), limit as f64);
                    }
                    let mut act = if kind == 2 {
                        0.5 * gate
                            * (1.
                                + (0.7978845608028654 * (gate + 0.044715 * gate * gate * gate))
                                    .tanh())
                    } else {
                        gate / (1. + (-gate).exp())
                    };
                    if !raw_limit && limit > 0. {
                        act = act.min(limit as f64);
                        up = up.clamp(-(limit as f64), limit as f64);
                    }
                    act * up
                };
                if !reference.is_finite() {
                    return Err("nonfinite activation F64 reference".into());
                }
                let target = reference_rounding::bf16_from_f64_bits(reference);
                let steps = (ordered(y[m * ystride + i]) - ordered(target)).abs();
                max_steps = max_steps.max(steps);
                nearest += usize::from(steps != 0);
                if steps > if kind == 1 { 0 } else { 1 } {
                    return Err(format!("activation exceeds admitted F64 BF16-ulp bound {name} {key} coordinate{i} steps{steps}"));
                }
            }
            if in_y[m * stride + inter..(m + 1) * stride]
                != values[m * stride + inter..(m + 1) * stride]
            {
                return Err("activation in-place changed up/padding".into());
            }
        }
        // Hot/cold precision passes leave inactive gate-up rows stale, including NaNs.
        // The actual masked activation must preserve them without reading/mutating them.
        for all_zero in [false,true] {
            let mask:Vec<f32>=(0..rows).map(|m|if all_zero||m%2==0{0.}else{1.}).collect();
            let mut masked_values=values.to_vec();for m in 0..rows{if mask[m]==0.{masked_values[m*stride..m*stride+2*inter].fill(0x7fc1);}}
            let masked_input=dev.htod_copy(masked_values.clone()).map_err(|e|e.to_string())?;
            let mask_gpu=dev.htod_copy(mask.clone()).map_err(|e|e.to_string())?;
            let _mask_completion=Completion(dev.clone());
            unsafe{sys::lib().cuMemsetD16Async(*output.device_ptr(),0x7fc1,rows*ystride+32,*dev.cu_stream())}.result().map_err(|e|e.to_string())?;
            graph_owner::completed_graph_launch(dev,||unsafe{program.enqueue_activation(*masked_input.device_ptr(),*output.device_ptr(),*mask_gpu.device_ptr(),inter,kind,limit,raw_limit,stride,ystride,rows,*dev.cu_stream())})?;
            let masked_output=dev.dtoh_sync_copy(&output).map_err(|e|e.to_string())?;
            for m in 0..rows{for i in 0..ystride{let expected=if mask[m]==0.||i>=inter{0x7fc1}else{y[m*ystride+i]};if masked_output[m*ystride+i]!=expected{return Err("masked stale-NaN activation graph differs".into());}}}
            if masked_output[rows*ystride..].iter().any(|&v|v!=0x7fc1){return Err("masked activation suffix changed".into());}
            unsafe{program.enqueue_activation(*masked_input.device_ptr(),*masked_input.device_ptr(),*mask_gpu.device_ptr(),inter,kind,limit,raw_limit,stride,stride,rows,*dev.cu_stream())}?;
            let got=dev.dtoh_sync_copy(&masked_input).map_err(|e|e.to_string())?;
            for m in 0..rows{for i in 0..stride{let expected=if mask[m]!=0.&&i<inter{y[m*ystride+i]}else{masked_values[m*stride+i]};if got[m*stride+i]!=expected{return Err("masked activation in-place changed inactive/up/padding".into());}}}
            if got[rows*stride..]!=masked_values[rows*stride..]||dev.dtoh_sync_copy(&mask_gpu).map_err(|e|e.to_string())?!=mask{return Err("masked activation readonly/suffix changed".into());}
        }
        let bytes: Vec<_> = (0..rows)
            .flat_map(|m| {
                y[m * ystride..m * ystride + inter]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
            })
            .collect();
        raw += bytes.len();
        fs::write(out.join(format!("{name}-activation-{key}.bf16")), bytes)
            .map_err(|e| e.to_string())?;
        records.push(json!({"kind":kind,"limit":limit,"rawLimit":raw_limit,"fullSingletonGraphErrorAndInplaceExact":true,"maskedStaleNaNEagerAndGraphExact":true,"nearestF64Mismatch":nearest,"maxF64BF16Steps":max_steps}));
    }
    if dev.dtoh_sync_copy(&input).map_err(|e| e.to_string())? != values {
        return Err("activation readonly source changed".into());
    }
    Ok((json!({"activationControls":records}), raw))
}
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("FATAL activation completion unknown:{e}");
            std::process::abort()
        }
    }
}
