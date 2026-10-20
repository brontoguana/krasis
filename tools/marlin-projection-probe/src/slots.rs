//! Real operand pointer-slot and common activation layout controls.
use super::*;
use common_marlin::{Geometry, Program};
pub fn check(
    dev: &Arc<CudaDevice>,
    program: &Program,
    case: &Case,
    wp: u64,
    sp: u64,
    packed_stride: u64,
    scale_stride: u64,
    inputs: &[Vec<u16>],
    out: &Path,
) -> Result<(Value, usize), String> {
    let count = 67usize;
    let n = case.n;
    let k = case.k;
    let stride = k + 16;
    let input = dev
        .htod_copy(
            (0..count)
                .flat_map(|i| {
                    inputs[i % 2]
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(0x7fc1).take(16))
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
    let pt = dev
        .htod_copy(
            (0..count)
                .map(|i| wp + (i % 2) as u64 * packed_stride)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
    let st = dev
        .htod_copy(
            (0..count)
                .map(|i| sp + (i % 2) as u64 * scale_stride)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
    let masks = [
        (0..count)
            .map(|i| if i % 3 == 0 { 0f32 } else { 1f32 })
            .collect::<Vec<_>>(),
        (0..count)
            .map(|i| if i % 3 == 0 { 1f32 } else { 0f32 })
            .collect::<Vec<_>>(),
    ];
    let masks = masks
        .iter()
        .map(|v| dev.htod_copy(v.clone()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let output = dev
        .htod_copy(vec![0x7fc1u16; count * (n + 16) + 32])
        .map_err(|e| e.to_string())?;
    let oracle = dev
        .htod_copy(vec![0x7fc1u16; count * (n + 16) + 32])
        .map_err(|e| e.to_string())?;
    let _complete = Completion(dev.clone());
    let g = Geometry {
        rows: n,
        cols: k,
        group: case.group,
        bits: case.bits,
    };
    let mut records = Vec::new();
    let mut raw = 0;
    for broadcast in [false, true] {
        unsafe {
            sys::lib().cuMemsetD16Async(
                *output.device_ptr(),
                0x7fc1,
                count * (n + 16) + 32,
                *dev.cu_stream(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        let input_stride = if broadcast { 0 } else { stride };
        for i in 0..count {
            unsafe {
                program.enqueue(
                    g,
                    wp + (i % 2) as u64 * packed_stride,
                    sp + (i % 2) as u64 * scale_stride,
                    *input.device_ptr() + (i * input_stride * 2) as u64,
                    *oracle.device_ptr() + (i * (n + 16) * 2) as u64,
                    0,
                    1,
                    *dev.cu_stream(),
                )
            }?;
        }
        let expected = dev.dtoh_sync_copy(&oracle).map_err(|e| e.to_string())?;
        let launch = || -> Result<(), String> {
            for mask in &masks {
                unsafe {
                    program.enqueue_slots(
                        g,
                        *pt.device_ptr(),
                        *st.device_ptr(),
                        *input.device_ptr(),
                        *output.device_ptr(),
                        *mask.device_ptr(),
                        input_stride,
                        n + 16,
                        count,
                        *dev.cu_stream(),
                    )
                }?;
            }
            Ok(())
        };
        for mode in ["eager", "graph", "post-enqueue-error"] {
            unsafe {
                sys::lib().cuMemsetD16Async(
                    *output.device_ptr(),
                    0x7fc1,
                    count * (n + 16) + 32,
                    *dev.cu_stream(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            match mode {
                "eager" => launch()?,
                "graph" => graph_owner::completed_graph_launch(dev, launch)?,
                _ => {
                    if graph_owner::completed_graph_launch(dev, || {
                        launch()?;
                        Err("intentional slot enqueue rejection".into())
                    })
                    .is_ok()
                    {
                        return Err("slot error accepted".into());
                    }
                    launch()?;
                }
            }
            if dev.dtoh_sync_copy(&output).map_err(|e| e.to_string())? != expected {
                return Err(format!(
                    "pointer-slot/rowpadding differs {} broadcast{broadcast} {mode}",
                    case.name
                ));
            }
        }
        // First masked pass must leave all inactive live and padding cells intact.
        unsafe {
            sys::lib().cuMemsetD16Async(
                *output.device_ptr(),
                0x7fc1,
                count * (n + 16) + 32,
                *dev.cu_stream(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        unsafe {
            program.enqueue_slots(
                g,
                *pt.device_ptr(),
                *st.device_ptr(),
                *input.device_ptr(),
                *output.device_ptr(),
                *masks[0].device_ptr(),
                input_stride,
                n + 16,
                count,
                *dev.cu_stream(),
            )
        }?;
        let partial = dev.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
        for i in 0..count {
            for j in 0..n + 16 {
                let target = if i % 3 == 0 || j >= n {
                    0x7fc1
                } else {
                    expected[i * (n + 16) + j]
                };
                if partial[i * (n + 16) + j] != target {
                    return Err("zero slot mask wrote data/guard".into());
                }
            }
        }
        let bytes: Vec<_> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        raw += bytes.len();
        fs::write(
            out.join(format!("{}-slots-broadcast{broadcast}.bf16", case.name)),
            bytes,
        )
        .map_err(|e| e.to_string())?;
        records.push(json!({"broadcast":broadcast,"allEagerGraphErrorOutputsExactSingleton":true,"inputStride":input_stride,"outputStride":n+16,"maskedSlotsAndPaddingUntouched":true}));
    }
    if case.name.contains("w1") {
        let values=dev.dtoh_sync_copy(&oracle).map_err(|e|e.to_string())?;
        let (control,bytes)=activation::check(dev,program,&case.name,&values,n+16,count,n/2,out)?;
        raw+=bytes;save(&out.join(format!("{}-activation.json",case.name)),&control)?;
    }
    if dev.dtoh_sync_copy(&pt).map_err(|e| e.to_string())?
        != (0..count)
            .map(|i| wp + (i % 2) as u64 * packed_stride)
            .collect::<Vec<_>>()
        || dev.dtoh_sync_copy(&st).map_err(|e| e.to_string())?
            != (0..count)
                .map(|i| sp + (i % 2) as u64 * scale_stride)
                .collect::<Vec<_>>()
    {
        return Err("slot pointer table mutated".into());
    }
    Ok((json!({"pointerSlots":records}), raw))
}
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("FATAL slots completion unknown:{e}");
            std::process::abort()
        }
    }
}
