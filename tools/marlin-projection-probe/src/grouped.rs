//! Explicit sorted-slot ABI controls with real retained quantized operands.
use super::*;
use common_marlin::{Geometry, Program};
pub fn execute(path: &Path, out: &Path) -> Result<Value, String> {
    let plan: Plan = read_json(path)?;
    plan.source.read()?;
    let ptx = String::from_utf8(plan.ptx.read()?).map_err(|e| e.to_string())?;
    let build: Value = serde_json::from_slice(&plan.build.read()?).map_err(|e| e.to_string())?;
    if build["passed"] != true
        || build["source"] != serde_json::to_value(&plan.source).unwrap()
        || build["ptx"] != serde_json::to_value(&plan.ptx).unwrap()
    {
        return Err("grouped build identity mismatch".into());
    }
    let dev = CudaDevice::new_with_stream(plan.gpu_ordinal).map_err(|e| e.to_string())?;
    let mut uuid = std::mem::MaybeUninit::uninit();
    unsafe { sys::lib().cuDeviceGetUuid_v2(uuid.as_mut_ptr(), *dev.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let uuid = unsafe { uuid.assume_init() };
    let h = uuid
        .bytes
        .iter()
        .map(|v| format!("{:02x}", *v as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    );
    if actual != plan.gpu_uuid {
        return Err("grouped GPU identity mismatch".into());
    }
    let program = Program::load(dev.clone(), &ptx)?;
    let mut records = Vec::new();
    let mut raw = 0usize;
    for case in &plan.cases {
        let geometry = Geometry {
            rows: case.n,
            cols: case.k,
            group: case.group,
            bits: case.bits,
        };
        geometry.extents()?;
        let peer = if case.name == "routed-w1-expert226" {
            plan.cases
                .iter()
                .find(|c| c.name == "routed-w1-expert113")
                .ok_or("missing distinct routed W1 peer")?
        } else if case.name == "routed-w2-expert226" {
            plan.cases
                .iter()
                .find(|c| c.name == "routed-w2-expert113")
                .ok_or("missing distinct routed W2 peer")?
        } else {
            case
        };
        if (peer.n, peer.k, peer.group, peer.bits) != (case.n, case.k, case.group, case.bits) {
            return Err("peer geometry mismatch".into());
        }
        let packed = [case.packed.read()?, peer.packed.read()?];
        let scales = [u16s(&case.scales.read()?)?, u16s(&peer.scales.read()?)?];
        let inputs = [case, peer]
            .iter()
            .map(|c| -> Result<Vec<u16>, String> {
                let b = c.input.read()?;
                let end = c
                    .input_offset
                    .checked_add(c.k * 2)
                    .ok_or("input overflow")?;
                if end > b.len() {
                    return Err("source input range invalid".into());
                }
                u16s(&b[c.input_offset..end])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weight = dev
            .htod_copy(
                packed
                    .iter()
                    .flat_map(|b| {
                        b.chunks_exact(4)
                            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())?;
        let scale = dev.htod_copy(scales.concat()).map_err(|e| e.to_string())?;
        let batch = 67usize;
        let guard = 32;
        let count = batch * case.n;
        let input = dev
            .htod_copy(
                (0..batch)
                    .flat_map(|i| inputs[i % 2].iter().copied())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())?;
        let output = dev
            .htod_copy(vec![0x7fc1u16; count + guard])
            .map_err(|e| e.to_string())?;
        let observation = dev
            .htod_copy(vec![f32::from_bits(0x7fc12345); count + guard])
            .map_err(|e| e.to_string())?;
        // Scope guard owns the device and completes before all borrowed allocations drop,
        // including a failed enqueue/readback. The tool does no production lifetime fabrication.
        let _completion = Completion(dev.clone());
        let wp = *weight.device_ptr();
        let sp = *scale.device_ptr();
        let packed_stride = packed[0].len() as u64;
        let scale_stride = (scales[0].len() * 2) as u64;
        for i in 0..batch {
            unsafe {
                program.enqueue(
                    geometry,
                    wp + (i % 2) as u64 * packed_stride,
                    sp + (i % 2) as u64 * scale_stride,
                    *input.device_ptr() + (i * case.k * 2) as u64,
                    *output.device_ptr() + (i * case.n * 2) as u64,
                    0,
                    1,
                    *dev.cu_stream(),
                )
            }?;
        }
        let expected = dev.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
        let before_input = dev.dtoh_sync_copy(&input).map_err(|e| e.to_string())?;
        let before_weight = dev.dtoh_sync_copy(&weight).map_err(|e| e.to_string())?;
        let before_scale = dev.dtoh_sync_copy(&scale).map_err(|e| e.to_string())?;
        let mut comparisons = Vec::new();
        for block in [16usize, 32, 48, 64] {
            let mut sorted = Vec::new();
            let mut experts = Vec::new();
            for e in 0..2 {
                let selected: Vec<_> = (0..batch)
                    .filter(|i| i % 2 == e)
                    .map(|i| i as i32)
                    .rev()
                    .collect();
                for chunk in selected.chunks(block) {
                    experts.push(e as i32);
                    sorted.extend_from_slice(chunk);
                    sorted.resize(sorted.len() + block - chunk.len(), batch as i32);
                }
            }
            // Genuine empty filtered block inside post_count, plus an unreachable valid
            // trailing block beyond post_count. Neither may touch an unrelated slot.
            experts.push(-1);
            sorted.resize(sorted.len() + block, 0);
            let post = sorted.len() as i32;
            experts.push(0);
            sorted.resize(sorted.len() + block, 0);
            let sorted_gpu = dev.htod_copy(sorted.clone()).map_err(|e| e.to_string())?;
            let post_gpu = dev.htod_copy(vec![post]).map_err(|e| e.to_string())?;
            let _sort_completion = Completion(dev.clone());
            for table in [false, true] {
                for split in [false, true] {
                    let pt = dev
                        .htod_copy(vec![wp + packed_stride, wp])
                        .map_err(|e| e.to_string())?;
                    let st = dev
                        .htod_copy(vec![sp + scale_stride, sp])
                        .map_err(|e| e.to_string())?;
                    let lists: Vec<Vec<i32>> = if split {
                        (0..2)
                            .map(|e| {
                                experts
                                    .iter()
                                    .map(|&id| if id == e { id } else { -1 })
                                    .collect()
                            })
                            .collect()
                    } else {
                        vec![experts.clone()]
                    };
                    let ids = lists
                        .iter()
                        .map(|list| {
                            dev.htod_copy(
                                list.iter()
                                    .map(|&id| if table && id >= 0 { 1 - id } else { id })
                                    .collect::<Vec<_>>(),
                            )
                            .map_err(|e| e.to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let _list_completion = Completion(dev.clone());
                    let launch = || -> Result<(), String> {
                        for list in &ids {
                            unsafe {
                                program.enqueue_grouped(
                                    geometry,
                                    wp,
                                    sp,
                                    if table { *pt.device_ptr() } else { 0 },
                                    if table { *st.device_ptr() } else { 0 },
                                    *input.device_ptr(),
                                    *output.device_ptr(),
                                    *sorted_gpu.device_ptr(),
                                    *list.device_ptr(),
                                    *post_gpu.device_ptr(),
                                    block,
                                    2,
                                    batch,
                                    sorted.len(),
                                    *dev.cu_stream(),
                                )
                            }?;
                        }
                        Ok(())
                    };
                    for mode in ["eager", "graph", "post-enqueue-error"] {
                        unsafe { sys::lib().cuMemsetD16Async(*output.device_ptr(), 0x7fc1, count + guard, *dev.cu_stream()) }.result().map_err(|e|e.to_string())?;
                        match mode {
                            "eager" => launch()?,
                            "graph" => graph_owner::completed_graph_launch(&dev, launch)?,
                            _ => {
                                if graph_owner::completed_graph_launch(&dev, || {
                                    launch()?;
                                    Err("intentional grouped post-enqueue error".into())
                                })
                                .is_ok()
                                {
                                    return Err("grouped graph error accepted".into());
                                }
                                launch()?;
                            }
                        }
                        let got = dev.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
                        if got != expected {
                            return Err(format!("grouped output/guard differs {} block{block} table{table} split{split} {mode}",case.name));
                        }
                        comparisons.push(json!({"block":block,"pointerTables":table,"precisionFilteredPasses":split,"mode":mode,"allOutputAndGuardBytesExact":true}));
                    }
                }
            }
        }
        let (slot_controls,slot_bytes)=slots::check(&dev,&program,case,wp,sp,packed_stride,scale_stride,&inputs,out)?;
        raw=raw.checked_add(slot_bytes).filter(|&v|v<=plan.raw_limit).ok_or("slot output cap exceeded")?;
        save(&out.join(format!("{}-slots.json",case.name)),&slot_controls)?;
        if before_input != dev.dtoh_sync_copy(&input).map_err(|e| e.to_string())?
            || before_weight != dev.dtoh_sync_copy(&weight).map_err(|e| e.to_string())?
            || before_scale != dev.dtoh_sync_copy(&scale).map_err(|e| e.to_string())?
            || dev
                .dtoh_sync_copy(&observation)
                .map_err(|e| e.to_string())?
                .iter()
                .any(|v| v.to_bits() != 0x7fc12345)
        {
            return Err("grouped readonly/absent-observation violation".into());
        }
        let bytes: Vec<_> = expected[..count]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        raw += bytes.len();
        if raw > plan.raw_limit {
            return Err("grouped raw limit exceeded".into());
        }
        fs::write(out.join(format!("{}.bf16", case.name)), bytes).map_err(|e| e.to_string())?;
        records.push(json!({"name":case.name,"peer":peer.name,"comparisons":comparisons,"optionalObservationAbsent":true,"readOnlyExact":true}));
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < plan.safety_margin_bytes {
            return Err("grouped actual VRAM below margin".into());
        }
    }
    plan.source.read()?;
    plan.ptx.read()?;
    Ok(
        json!({"passed":true,"servingQualified":false,"gpuUuid":actual,"program":program.identity,"rawBytes":raw,"records":records,"scope":"Sortedslot/padding/filter/pointertable/contiguous ABI, optionalF32absence andrealallocations retainedthroughgraph/errorcompletion; no installedmodelqualification."}),
    )
}
struct Completion(Arc<CudaDevice>);
impl Drop for Completion {
    fn drop(&mut self) {
        if let Err(e) = self.0.synchronize() {
            eprintln!("FATAL grouped completion unknown:{e}");
            std::process::abort()
        }
    }
}
