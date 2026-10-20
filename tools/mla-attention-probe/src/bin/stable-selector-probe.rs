// Complete real common execution object, explicit operator fixture only.
#[path = "../../../../src/cuda_region.rs"]
mod cuda_region;
#[path = "../../../../src/stable_mla_gemm.rs"]
mod stable_mla_gemm;
#[path = "../../../../src/stable_mla_ops.rs"]
mod stable_mla_ops;
#[path = "../../../../src/stable_mla_path.rs"]
mod stable_mla_path;
#[path = "../../../../src/stable_mla_select.rs"]
mod stable_mla_select;
// The exact production completion guard is compiled here. This standalone
// operator has only a fail-closed fatal policy, not a model VRAM monitor.
#[path = "../../../../src/cuda_failure_code.rs"]
mod cuda_failure_code;
mod vram_monitor {
    pub use crate::cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE;
    pub fn fatal_cuda_context_error(context: &str, error: &str) -> ! {
        eprintln!("FATAL operator CUDA completion {context}: {error}");
        #[cfg(unix)]
        unsafe {
            libc::_exit(CUDA_CONTEXT_POISONED_EXIT_CODE)
        }
        #[cfg(not(unix))]
        std::process::exit(CUDA_CONTEXT_POISONED_EXIT_CODE)
    }
}
#[path = "../../../bf16-projection-probe/src/graph_owner.rs"]
mod graph_owner;
#[path = "../../../../src/sequence_copy_completion.rs"]
mod sequence_copy_completion;
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use stable_mla_gemm::{Candidates, Span};
use stable_mla_ops::{Codec, Gather, Ops};
use stable_mla_path::{Geometry, Inputs, Path as Program, Work};
use std::{fs, path::PathBuf, sync::Arc};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Identity {
    fn read(&self) -> Result<Vec<u8>, String> {
        let b = fs::read(&self.path).map_err(|e| e.to_string())?;
        if b.len() != self.bytes || hash(&b) != self.sha256 {
            return Err(format!("identity {}", self.path.display()));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    gpu_uuid: String,
    ptx: Identity,
    provenance: Identity,
    rows: usize,
    heads: usize,
    ccd: usize,
    rope: usize,
    nope: usize,
    context: usize,
    slots: usize,
    codec: usize,
    partitions: Vec<usize>,
    query: Identity,
    absorb: Identity,
    ckv: Identity,
    kpe: Identity,
    positions: Identity,
    selected: Identity,
    bf16: Identity,
    values: usize,
    expected_gather: Identity,
    expected_query: Identity,
    expected_mask: Identity,
    native_ptx: Identity,
}
fn floats(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err("F32 extent".into());
    }
    Ok(b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect())
}
fn serial(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn copy<T: cudarc::driver::DeviceRepr + Clone>(
    d: &Arc<CudaDevice>,
    v: &[T],
) -> Result<CudaSlice<T>, String> {
    d.htod_sync_copy(v).map_err(|e| e.to_string())
}
fn bf(x: f32) -> u16 {
    let v = x.to_bits();
    ((v.wrapping_add(0x7fff + ((v >> 16) & 1))) >> 16) as u16
}
fn checked_equal(a: &[f32], b: &[f32], label: &str) -> Result<(), String> {
    if serial(a) != serial(b) {
        return Err(format!(
            "complete {label} bytes differ: lengths {} {}; first coordinate {:?}",
            a.len(),
            b.len(),
            a.iter()
                .zip(b)
                .enumerate()
                .find(|(_, (x, y))| x.to_bits() != y.to_bits())
                .map(|(i, (x, y))| (i, x.to_bits(), y.to_bits()))
        ));
    }
    Ok(())
}
fn get(d: &Arc<CudaDevice>, v: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
    d.synchronize().map_err(|e| e.to_string())?;
    d.dtoh_sync_copy(v).map_err(|e| e.to_string())
}
fn reset(d: &Arc<CudaDevice>, v: &CudaSlice<f32>, bits: u32) -> Result<(), String> {
    unsafe { sys::lib().cuMemsetD32Async(*v.device_ptr(), bits, v.len(), *d.cu_stream()) }
        .result()
        .map_err(|e| e.to_string())
}
fn run(plan_path: &str, out: &str) -> Result<(), String> {
    let raw = fs::read(plan_path).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let root = PathBuf::from(out);
    if root.exists() {
        return Err("fresh complete-path output required".into());
    }
    p.provenance.read()?;
    let d = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *d.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let h = uuid
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    );
    if actual != p.gpu_uuid {
        return Err("physical GPU mismatch".into());
    }
    let geom = Geometry {
        heads: p.heads,
        ccd: p.ccd,
        rope: p.rope,
        values: p.values,
        slots: p.slots,
        codec: match p.codec {
            4 => Codec::K4,
            6 => Codec::K6,
            _ => return Err("actual codec not admitted".into()),
        },
    };
    let elements = geom.work_elements(p.rows)?;
    let abs = floats(&p.absorb.read()?)?;
    let query = floats(&p.query.read()?)?;
    let ckv = p.ckv.read()?;
    let kpe = p.kpe.read()?;
    let selected: Vec<_> = p
        .selected
        .read()?
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let positions: Vec<_> = p
        .positions
        .read()?
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let wb = p.bf16.read()?;
    let wb: Vec<_> = wb
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if wb.len() != p.heads * p.values * p.ccd
        || abs.len() != p.rows * p.heads * p.ccd
        || p.partitions.iter().sum::<usize>() != p.rows
    {
        return Err("real path input extents".into());
    }
    let guard = usize::try_from(
        d.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let live = abs.len() * 4
        + query.len() * 4
        + ckv.len()
        + kpe.len()
        + selected.len() * 4
        + positions.len() * 4
        + wb.len() * 6
        + (elements[0] + elements[1].max(elements[5]) + elements[2] * 3 + elements[4]) * 4
        + elements[6] * 2
        + p.heads * p.ccd * 4
        + guard
            * (2 * [
                elements[0],
                elements[1].max(elements[5]),
                elements[2],
                elements[3],
                elements[4],
            ]
            .len()
                * std::mem::size_of::<f32>()
                + 2 * std::mem::size_of::<u16>());
    let ops = Ops::load(
        d.clone(),
        &String::from_utf8(p.ptx.read()?).map_err(|e| e.to_string())?,
    )?;
    if cudarc::driver::result::mem_get_info()
        .map_err(|e| e.to_string())?
        .0
        < live
            .checked_add(600 * 1024 * 1024)
            .ok_or("actual memory extent overflow")?
    {
        return Err(
            "actual free VRAM below actual path allocation extent plus default margin".into(),
        );
    }
    let da = copy(&d, &abs)?;
    let dq = copy(&d, &query)?;
    let dc = copy(&d, &ckv)?;
    let dk = if kpe.is_empty() {
        None
    } else {
        Some(copy(&d, &kpe)?)
    };
    let di = copy(&d, &selected)?;
    let dp = copy(&d, &positions)?;
    let db = copy(&d, &wb)?;
    let wf = d.alloc_zeros::<f32>(wb.len()).map_err(|e| e.to_string())?;
    graph_owner::completed_graph_launch(&d, || unsafe {
        ops.widen(
            &Span::bf16(&db, 0, wb.len())?,
            &Span::f32(&wf, 0, wb.len())?,
            wb.len(),
            *d.cu_stream(),
        )
    })?;
    let inputs = Inputs {
        q_abs: Span::f32(&da, 0, abs.len())?,
        q_rope: if p.rope > 0 {
            Some(Span::f32(&dq, p.nope, query.len() - p.nope)?)
        } else {
            None
        },
        ckv: Span::bytes(&dc, 0, ckv.len())?,
        kpe: if let Some(k) = &dk {
            Some(Span::bytes(k, 0, kpe.len())?)
        } else {
            None
        },
        selected: Span::i32(&di, 0, selected.len())?,
        positions: Span::i32(&dp, 0, positions.len())?,
        wvc: Span::f32(&wf, 0, wb.len())?,
    };
    let nan = f32::from_bits(0x7fc12345);
    let gathered = copy(&d, &vec![nan; elements[0] + 2 * guard])?;
    let qp = copy(&d, &vec![nan; elements[1].max(elements[5]) + 2 * guard])?;
    let scores = copy(&d, &vec![nan; elements[2] + 2 * guard])?;
    let weights = copy(&d, &vec![nan; elements[3] + 2 * guard])?;
    let attention = copy(&d, &vec![nan; elements[4] + 2 * guard])?;
    let output = copy(&d, &vec![0xa55au16; elements[6] + 2 * guard])?;
    let counts: Vec<usize> = std::iter::once(p.rows)
        .chain(p.partitions.iter().copied())
        .chain(std::iter::once(1))
        .collect();
    let gather_param = |start, rows| Gather {
        row_start: start,
        rows,
        heads: p.heads,
        ccd: p.ccd,
        rope: p.rope,
        slots: p.slots,
        context: p.context,
        q_abs_row_stride: p.heads * p.ccd,
        q_abs_head_stride: p.ccd,
        q_rope_row_stride: p.heads * (p.nope + p.rope),
        q_rope_head_stride: p.nope + p.rope,
        codec: if p.codec == 6 { Codec::K6 } else { Codec::K4 },
    };
    let work = |start: usize, rows: usize| -> Result<Work<'_>, String> {
        let n = geom.work_elements(rows)?;
        Ok(Work {
            gathered: Span::f32(&gathered, guard, n[0])?,
            query: Span::f32(&qp, guard, n[1])?,
            scores: Span::f32(&scores, guard, n[2])?,
            weights: Span::f32(&weights, guard, n[3])?,
            attention: Span::f32(&attention, guard, n[4])?,
            projected: Span::f32(&qp, guard, n[5])?,
            output: Span::bf16(&output, guard + start * p.heads * p.values, n[6])?,
        })
    };
    let scale = ((p.nope + p.rope) as f32).sqrt().recip();

    let workspace_bytes = p
        .heads
        .checked_mul(p.ccd)
        .and_then(|n| n.checked_mul(4))
        .ok_or("actual calibration workspace extent overflow")?;
    let workspace_owner = d
        .alloc_zeros::<u8>(workspace_bytes)
        .map_err(|e| e.to_string())?;
    let workspace = Span::bytes(&workspace_owner, 0, workspace_bytes)?;
    let plane = work(0, p.rows)?;
    let selected_families = unsafe {
        sequence_copy_completion::with_copy_completion(
            &d,
            *d.cu_stream(),
            &(&inputs, &plane, &workspace, &ops),
            |_| {
                ops.gather(
                    gather_param(0, p.rows),
                    &plane.gathered,
                    &plane.query,
                    &plane.weights,
                    &inputs.q_abs,
                    inputs.q_rope.as_ref(),
                    &inputs.ckv,
                    inputs.kpe.as_ref(),
                    &inputs.selected,
                    &inputs.positions,
                    *d.cu_stream(),
                )?;
                let score = stable_mla_select::select_stage(
                    d.clone(),
                    geom,
                    stable_mla_select::Stage::Score,
                    p.rows,
                    &inputs,
                    &plane,
                    Some(&workspace),
                    workspace_bytes,
                    scale,
                    8,
                    32,
                    *d.cu_stream(),
                )?;
                let prepared = score
                    .candidate
                    .prepare(geom.shapes(p.rows)?[0], workspace_bytes)?;
                sequence_copy_completion::with_copy_completion(
                    &d,
                    *d.cu_stream(),
                    &(&prepared, &inputs, &plane, &workspace),
                    |_| {
                        prepared.enqueue(
                            &plane.gathered,
                            &plane.query,
                            &plane.weights,
                            &plane.scores,
                            Some(&workspace),
                            scale,
                            1.0,
                            *d.cu_stream(),
                        )
                    },
                )?;
                ops.softmax(
                    &plane.scores,
                    &plane.weights,
                    p.rows,
                    p.heads,
                    p.slots,
                    *d.cu_stream(),
                )?;
                let latent = stable_mla_select::select_stage(
                    d.clone(),
                    geom,
                    stable_mla_select::Stage::Latent,
                    p.rows,
                    &inputs,
                    &plane,
                    Some(&workspace),
                    workspace_bytes,
                    1.0,
                    8,
                    32,
                    *d.cu_stream(),
                )?;
                let prepared = latent
                    .candidate
                    .prepare(geom.shapes(p.rows)?[1], workspace_bytes)?;
                sequence_copy_completion::with_copy_completion(
                    &d,
                    *d.cu_stream(),
                    &(&prepared, &inputs, &plane, &workspace),
                    |_| {
                        prepared.enqueue(
                            &plane.gathered,
                            &plane.weights,
                            &plane.attention,
                            &plane.attention,
                            Some(&workspace),
                            1.0,
                            0.0,
                            *d.cu_stream(),
                        )
                    },
                )?;
                let wvc = stable_mla_select::select_stage(
                    d.clone(),
                    geom,
                    stable_mla_select::Stage::Wvc,
                    p.rows,
                    &inputs,
                    &plane,
                    Some(&workspace),
                    workspace_bytes,
                    1.0,
                    8,
                    32,
                    *d.cu_stream(),
                )?;
                Ok([score, latent, wvc])
            },
        )
    }?;
    let choices: Vec<_> = selected_families
        .iter()
        .map(|s| {
            s.report["selected_index"]
                .as_u64()
                .ok_or("missing actual measured choice")
        })
        .collect::<Result<_, _>>()?;
    let eligibility: Vec<_> = selected_families.iter().map(|s| s.report.clone()).collect();
    let algorithms = [
        &selected_families[0].candidate,
        &selected_families[1].candidate,
        &selected_families[2].candidate,
    ];
    let mut programs = std::collections::BTreeMap::new();
    for &rows in &counts {
        if !programs.contains_key(&rows) {
            programs.insert(
                rows,
                Program::prepare(ops.clone(), geom, rows, algorithms, workspace_bytes)?,
            );
        }
    }
    eprintln!(
        "{}",
        json!({"calibration":eligibility,"workspace_bytes":workspace_bytes})
    );
    let enqueue_raw = |start, rows| -> Result<(), String> {
        unsafe {
            programs.get(&rows).ok_or("prepared count absent")?.enqueue(
                gather_param(start, rows),
                &inputs,
                &work(start, rows)?,
                Some(&workspace),
                scale,
                *d.cu_stream(),
            )
        }
    };
    let enqueue = |start, rows| -> Result<(), String> {
        let prepared = programs.get(&rows).ok_or("prepared count absent")?;
        let plane = work(start, rows)?;
        let retained = (prepared, &inputs, &plane);
        graph_owner::completed_eager(&d, &retained, |(prepared, inputs, plane)| unsafe {
            prepared.enqueue(
                gather_param(start, rows),
                inputs,
                plane,
                Some(&workspace),
                scale,
                *d.cu_stream(),
            )
        })
    };
    enqueue(0, p.rows)?;
    let baseline = d.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
    let latent = get(&d, &attention)?;
    let score = get(&d, &scores)?;
    let probabilities = get(&d, &weights)?;
    let projected = get(&d, &qp)?;
    let cache = get(&d, &gathered)?;
    checked_equal(
        &cache[guard..guard + elements[0]],
        &floats(&p.expected_gather.read()?)?,
        "coupled gather",
    )?;
    // Native softmax parity on the exact common score plane, with unchanged native PTX.
    let native_ptx = String::from_utf8(p.native_ptx.read()?).map_err(|e| e.to_string())?;
    d.load_ptx(
        native_ptx.into(),
        "native_softmax",
        &["mla_prefill_softmax_weights_warp_f32_kernel"],
    )
    .map_err(|e| e.to_string())?;
    let native = d
        .get_func(
            "native_softmax",
            "mla_prefill_softmax_weights_warp_f32_kernel",
        )
        .ok_or("native softmax symbol")?;
    let native_weights = d
        .alloc_zeros::<f32>(elements[3])
        .map_err(|e| e.to_string())?;
    let (mut x, mut y, mut n, mut heads, mut slots) = (
        *native_weights.device_ptr(),
        *scores.device_ptr() + guard as u64 * 4,
        p.rows as i32,
        p.heads as i32,
        p.slots as i32,
    );
    unsafe {
        sys::lib().cuLaunchKernel(
            native.cu_function(),
            (p.rows * p.heads).div_ceil(8) as u32,
            1,
            1,
            256,
            1,
            1,
            0,
            *d.cu_stream(),
            [
                &mut x as *mut _ as *mut std::ffi::c_void,
                &mut y as *mut _ as _,
                &mut n as *mut _ as _,
                &mut heads as *mut _ as _,
                &mut slots as *mut _ as _,
            ]
            .as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
    .map_err(|e| e.to_string())?;
    checked_equal(
        &get(&d, &native_weights)?,
        &probabilities[guard..guard + elements[3]],
        "native/common softmax",
    )?;
    let check_tile = |start: usize, rows: usize| -> Result<(), String> {
        let now = d.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
        let begin = guard + start * p.heads * p.values;
        let end = begin + rows * p.heads * p.values;
        if now[begin..end] != baseline[begin..end] {
            return Err(format!("complete coupled final output differs by partition: start={start} rows={rows} first={:?}",now[begin..end].iter().zip(&baseline[begin..end]).enumerate().find(|(_, (a,b))|a!=b)));
        }
        let local = get(&d, &attention)?;
        checked_equal(
            &local[guard..guard + rows * p.heads * p.ccd],
            &latent[guard + start * p.heads * p.ccd..guard + (start + rows) * p.heads * p.ccd],
            "coupled latent rows",
        )?;
        let local = get(&d, &scores)?;
        checked_equal(
            &local[guard..guard + rows * p.heads * p.slots],
            &score[guard + start * p.heads * p.slots..guard + (start + rows) * p.heads * p.slots],
            "coupled score rows",
        )?;
        Ok(())
    };
    for parts in [p.partitions.clone(), vec![1; p.rows]] {
        unsafe {
            sys::lib().cuMemsetD16Async(*output.device_ptr(), 0xa55a, output.len(), *d.cu_stream())
        }
        .result()
        .map_err(|e| e.to_string())?;
        let mut start = 0;
        for rows in parts {
            reset(&d, &qp, nan.to_bits())?;
            reset(&d, &attention, nan.to_bits())?;
            reset(&d, &scores, nan.to_bits())?;
            enqueue(start, rows)?;
            check_tile(start, rows)?;
            start += rows
        }
        let all = d.dtoh_sync_copy(&output).map_err(|e| e.to_string())?;
        if all != baseline {
            return Err("complete coupled partition/output guards differ".into());
        }
    }
    graph_owner::completed_graph_launch(&d, || enqueue_raw(0, p.rows))?;
    check_tile(0, p.rows)?;
    let failed = graph_owner::completed_graph_launch(&d, || {
        enqueue_raw(0, p.rows)?;
        Err("injected complete path captured callback error".into())
    })
    .err()
    .ok_or("complete path callback error absent")?;
    if !failed.contains("injected complete path") {
        return Err(failed);
    }
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = graph_owner::completed_graph_launch(&d, || -> Result<(), String> {
            enqueue_raw(0, p.rows)?;
            panic!("injected complete path callback unwind")
        });
    }));
    if unwind.is_ok() {
        return Err("complete path callback did not unwind".into());
    }
    graph_owner::completed_graph_launch(&d, || enqueue_raw(0, p.rows))?;
    check_tile(0, p.rows)?;
    let mut bad = work(0, 1)?;
    bad.attention = Span::f32(&wf, 0, geom.work_elements(1)?[4])?;
    let rejection = unsafe {
        programs[&1].enqueue(
            gather_param(0, 1),
            &inputs,
            &bad,
            Some(&workspace),
            1.0,
            *d.cu_stream(),
        )
    }
    .err()
    .ok_or("retained weight alias admitted")?;
    if !rejection.contains("aliases retained input") {
        return Err(format!("wrong alias rejection {rejection}"));
    }
    // Every final/projected coordinate agrees after the one BF16 output rounding.
    for r in 0..p.rows {
        for h in 0..p.heads {
            for v in 0..p.values {
                if baseline[guard + (r * p.heads + h) * p.values + v]
                    != bf(projected[guard + (h * p.rows + r) * p.values + v])
                {
                    return Err("coupled output rounding differs".into());
                }
            }
        }
    }
    for (v, n) in [
        (&gathered, elements[0]),
        (&qp, elements[1].max(elements[5])),
        (&scores, elements[2]),
        (&weights, elements[3]),
        (&attention, elements[4]),
    ] {
        let all = get(&d, v)?;
        if all[..guard]
            .iter()
            .chain(&all[guard + n..])
            .any(|x| x.to_bits() != nan.to_bits())
        {
            return Err("common path allocation guard changed".into());
        }
    }
    checked_equal(&get(&d, &da)?, &abs, "readonly absorbed input")?;
    checked_equal(&get(&d, &dq)?, &query, "readonly query input")?;
    let expected_w: Vec<_> = wb
        .iter()
        .map(|&x| f32::from_bits((x as u32) << 16))
        .collect();
    checked_equal(&get(&d, &wf)?, &expected_w, "readonly widened weights")?;
    if d.dtoh_sync_copy(&dc).map_err(|e| e.to_string())? != ckv
        || d.dtoh_sync_copy(&di).map_err(|e| e.to_string())? != selected
        || d.dtoh_sync_copy(&dp).map_err(|e| e.to_string())? != positions
    {
        return Err("retained cache/indices/positions mutated".into());
    }
    if let Some(k) = &dk {
        if d.dtoh_sync_copy(k).map_err(|e| e.to_string())? != kpe {
            return Err("retained rotary cache mutated".into());
        }
    }
    p.expected_query.read()?;
    p.expected_mask.read()?;
    fs::create_dir(&root).map_err(|e| e.to_string())?;
    let float_count = [elements[2], elements[3], elements[4], elements[5]]
        .into_iter()
        .try_fold(0usize, |a, b| {
            a.checked_add(b).ok_or("capture count overflow")
        })?;
    let raw_limit = float_count
        .checked_mul(4)
        .and_then(|n| elements[6].checked_mul(2).and_then(|v| n.checked_add(v)))
        .ok_or("capture bytes overflow")?;
    let mut written = 0usize;
    let mut write_raw = |name: &str, bytes: &[u8]| -> Result<(), String> {
        let next = written
            .checked_add(bytes.len())
            .ok_or("capture written bytes overflow")?;
        if next > raw_limit {
            return Err("raw write exceeds dimension-derived preregistered case bound".into());
        }
        fs::write(root.join(name), bytes).map_err(|e| e.to_string())?;
        written = next;
        Ok(())
    };
    for (name, values) in [
        ("scores", &score[guard..guard + elements[2]]),
        ("weights", &probabilities[guard..guard + elements[3]]),
        ("attention", &latent[guard..guard + elements[4]]),
        ("projected", &projected[guard..guard + elements[5]]),
    ] {
        write_raw(&format!("{name}.f32"), &serial(values))?
    }
    write_raw(
        "output.bf16",
        &baseline[guard..guard + elements[6]]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    drop(write_raw);
    if written != raw_limit {
        return Err("complete raw output collection missing bytes".into());
    }
    let families: Vec<_> = programs
        .values()
        .map(|v| v.family_identity.clone())
        .collect();
    if families.iter().any(|x| x != &families[0]) {
        return Err("global family identity depends on request rows".into());
    }
    let report = json!({"passed":true,"raw_bytes":written,"raw_bound_bytes":raw_limit,"gpu_uuid":actual,"plan_sha256":hash(&raw),"geometry":format!("{geom:?}"),"family_identity":families[0],"runtime_fixture_choice":choices,"eligibility":eligibility,"full_partition_singleton_graph_exact":true,"native_softmax_exact":true,"captured_error_unwind_recovery_exact":true,"retained_weight_alias_rejected":rejection,"allocation_guards_exact":true,"readonly_inputs_exact":true,"scope":"Combined production candidate and actual measured runtime selector. Actual WKC operand reused as arbitrary output projection. Ordinary/sanitized selection may choose different eligible families; instrumented timing is not used as production performance evidence. No serving/model fidelity/throughput acceptance."});
    fs::write(
        root.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") || args.len() != 3 {
        eprintln!("Use ./dev stable-selector-probe <plan> <fresh-output>");
        std::process::exit(2)
    }
    if let Err(e) = run(&args[1], &args[2]) {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
