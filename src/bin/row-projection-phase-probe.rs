// Private instrumented phase-cost control; never a serving or speed benchmark.
#[path = "row_projection_resident_graph/artifact.rs"]
mod artifact;
mod operator {
    include!("row-projection-compensated-probe.rs");
    use super::artifact;
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct PhasePlan {
        operator: Identity,
        cuda_source: Identity,
        ptx: Identity,
        build_report: Identity,
        complete_bf16_reference: Identity,
        complete_f32_reference: Identity,
        repeats: usize,
        inject_post_enqueue_failure: bool,
    }
    struct PhaseCompletion<'a> {
        device: &'a std::sync::Arc<CudaDevice>,
        failure_queued: std::cell::Cell<bool>,
    }
    impl Drop for PhaseCompletion<'_> {
        fn drop(&mut self) {
            match self.device.synchronize() {
                Ok(()) => {
                    if self.failure_queued.get() {
                        eprintln!("PHASE_TEARDOWN_COMPLETED_WITH_ALLOCATIONS_RETAINED");
                    }
                }
                Err(e) => eprintln!("ERROR: phase teardown CUDA completion failed: {e}"),
            }
        }
    }
    const FIELDS: &[&str] = &[
        "staging_reconstruction_and_barrier",
        "wmma_fragment_loads",
        "protected_twosum_carry",
        "post_carry_barrier",
        "whole_k_loop",
        "adjacent_clock_read_control",
        "packed_weight_reconstruction",
        "bf16_input_staging",
        "initial_staging_barrier",
    ];
    pub fn compile(plan: &Path, out: &Path) -> Result<Value, String> {
        build(plan, out)
    }
    pub fn phase(plan: &Path, out: &Path) -> Result<Value, String> {
        let specification = fs::read(plan).map_err(|e| e.to_string())?;
        let t: PhasePlan = serde_json::from_slice(&specification).map_err(|e| e.to_string())?;
        if t.repeats == 0 {
            return Err("phase control requires admitted repeats".into());
        }
        let p: ProbePlan =
            serde_json::from_slice(&t.operator.read()?).map_err(|e| e.to_string())?;
        if p.schema != 1
            || p.kernel_contract != "tf32_three_component_fixed_k8_twosum_v1"
            || p.safety_margin_bytes != 600 * 1024 * 1024
        {
            return Err("phase source contract/schema/reserve mismatch".into());
        }
        let original = p.source.read()?;
        let source = t.cuda_source.read()?;
        if !source.starts_with(&original) {
            return Err("original complete CUDA source is not exact prefix".into());
        }
        let built: Value =
            serde_json::from_slice(&t.build_report.read()?).map_err(|e| e.to_string())?;
        if built["plan"]["source"]
            != serde_json::to_value(&t.cuda_source).map_err(|e| e.to_string())?
            || built["ptx"] != serde_json::to_value(&t.ptx).map_err(|e| e.to_string())?
        {
            return Err("phase PTX/source build identity mismatch".into());
        }
        let read = |key: &str| {
            p.capture
                .get(key)
                .ok_or(format!("missing actual capture {key}"))?
                .read()
        };
        let descriptor: Value =
            serde_json::from_slice(&read("descriptor")?).map_err(|e| e.to_string())?;
        let (r, k, g, ps, ss, zs, h, row) = dims(&descriptor)?;
        let dispatch: Value =
            serde_json::from_slice(&p.dispatch.read()?).map_err(|e| e.to_string())?;
        for (key, v) in [
            ("rows", r),
            ("cols", k),
            ("group_size", g),
            ("packed_row_stride_bytes", ps),
            ("scales_row_stride_bytes", ss),
            ("zeros_row_stride_bytes", zs),
        ] {
            if dispatch[key].as_u64() != Some(v as u64) {
                return Err(format!("actual dispatch extent mismatch {key}"));
            }
        }
        let packed = read("packed")?;
        let scales = read("scales")?;
        let zeros = read("zeros")?;
        let input = read("full_input")?;
        let selected = read("selected_input")?;
        for (actual, expected) in [
            (packed.len(), mul(r, ps)?),
            (scales.len(), mul(r, ss)?),
            (zeros.len(), mul(r, zs)?),
            (input.len(), mul(mul(h, k)?, 2)?),
            (selected.len(), mul(k, 2)?),
        ] {
            if actual != expected {
                return Err("captured phase operand extent mismatch".into());
            }
        }
        if input[row * k * 2..(row + 1) * k * 2] != selected {
            return Err("actual selected row mismatch".into());
        }
        let bf16_ref = t.complete_bf16_reference.read()?;
        let f32_ref = t.complete_f32_reference.read()?;
        if bf16_ref.len() != mul(mul(h, r)?, 2)? || f32_ref.len() != mul(mul(h, r)?, 4)? {
            return Err("complete reference extent mismatch".into());
        }
        let dev = CudaDevice::new_with_stream(p.gpu_ordinal).map_err(|e| e.to_string())?;
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
        use cudarc::driver::sys::CUdevice_attribute as A;
        let attr = |a| dev.attribute(a).map_err(|e| e.to_string());
        let cc = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
            + attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        if actual_uuid != p.gpu_uuid
            || dev.name().map_err(|e| e.to_string())? != p.gpu_name
            || cc as u32 != p.compute_capability
            || cc < 80
        {
            return Err("phase actual GPU identity/capability mismatch".into());
        }
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ]
        .map(|v| v as usize);
        let grid = tensor_grid(r, h, limits)?;
        let warps = grid.block_dim.0 as usize / 32;
        let blocks = mul(
            mul(grid.grid_dim.0 as usize, grid.grid_dim.1 as usize)?,
            grid.grid_dim.2 as usize,
        )?;
        let records = mul(blocks, warps)?;
        let words = mul(records, FIELDS.len())?;
        let ptx = t.ptx.read()?;
        dev.load_ptx(
            cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx).map_err(|e| e.to_string())?),
            "phase",
            &[
                "hqq6_batch_compensated_tf32x3",
                "hqq6_batch_compensated_phase",
            ],
        )
        .map_err(|e| e.to_string())?;
        let ordinary = dev
            .get_func("phase", "hqq6_batch_compensated_tf32x3")
            .ok_or("original phase control kernel missing")?;
        let observed = dev
            .get_func("phase", "hqq6_batch_compensated_phase")
            .ok_or("phase instrumented kernel missing")?;
        let original_max_blocks = ordinary
            .occupancy_max_active_blocks_per_multiprocessor(
                grid.block_dim.0,
                grid.shared_mem_bytes as usize,
                None,
            )
            .map_err(|e| e.to_string())?;
        let instrumented_max_blocks = observed
            .occupancy_max_active_blocks_per_multiprocessor(
                grid.block_dim.0,
                grid.shared_mem_bytes as usize,
                None,
            )
            .map_err(|e| e.to_string())?;
        let tensor_bytes = [
            packed.len(),
            scales.len(),
            zeros.len(),
            input.len(),
            mul(mul(h, r)?, 6)?,
            mul(add(words, 32)?, 8)?,
        ]
        .iter()
        .try_fold(0, |a, &b| add(a, b))?;
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < add(tensor_bytes, p.safety_margin_bytes)? {
            return Err("phase tensors plus user reserve cannot fit measured free VRAM".into());
        }
        let dp = dev.htod_copy(packed).map_err(|e| e.to_string())?;
        let ds = dev.htod_copy(f32s(&scales)?).map_err(|e| e.to_string())?;
        let dz = dev.htod_copy(f32s(&zeros)?).map_err(|e| e.to_string())?;
        let dx = dev.htod_copy(u16s(&input)?).map_err(|e| e.to_string())?;
        let mut dy = dev
            .alloc_zeros::<u16>(mul(h, r)?)
            .map_err(|e| e.to_string())?;
        let mut df = dev
            .alloc_zeros::<f32>(mul(h, r)?)
            .map_err(|e| e.to_string())?;
        let mut dc = dev
            .htod_copy(vec![u64::MAX; add(words, 32)?])
            .map_err(|e| e.to_string())?;
        // Rust drops this guard before the allocations declared above, including on error.
        let _completion = PhaseCompletion {
            device: &dev,
            failure_queued: std::cell::Cell::new(false),
        };
        let mut minimum_free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if minimum_free < p.safety_margin_bytes {
            return Err("phase allocation breached user reserve".into());
        }
        let mut results = Vec::new();
        for repetition in 0..t.repeats {
            for instrumented in [false, true] {
                dev.htod_sync_copy_into(&vec![u64::MAX; add(words, 32)?], &mut dc)
                    .map_err(|e| e.to_string())?;
                let mut pointers = [
                    *dp.device_ptr(),
                    *ds.device_ptr(),
                    *dz.device_ptr(),
                    *dx.device_ptr(),
                    *dy.device_ptr(),
                    *df.device_ptr(),
                    *dc.device_ptr(),
                ];
                let mut integers = [r as i32, k as i32, g as i32, ps as i32, ss as i32];
                let mut tokens = h as u64;
                let mut arguments = pointers[..6]
                    .iter_mut()
                    .map(|v| (v as *mut u64).cast::<std::ffi::c_void>())
                    .collect::<Vec<_>>();
                arguments.extend(
                    integers
                        .iter_mut()
                        .map(|v| (v as *mut i32).cast::<std::ffi::c_void>()),
                );
                arguments.push((&mut tokens as *mut u64).cast());
                if instrumented {
                    arguments.push((&mut pointers[6] as *mut u64).cast());
                }
                let started = Instant::now();
                unsafe {
                    if instrumented {
                        observed.clone().launch(grid, &mut arguments)
                    } else {
                        ordinary.clone().launch(grid, &mut arguments)
                    }
                }
                .map_err(|e| e.to_string())?;
                if t.inject_post_enqueue_failure {
                    _completion.failure_queued.set(true);
                    return Err("explicit injected post-enqueue phase failure".into());
                }
                dev.synchronize().map_err(|e| e.to_string())?;
                let seconds = started.elapsed().as_secs_f64();
                let bf16_bytes = dev
                    .dtoh_sync_copy(&dy)
                    .map_err(|e| e.to_string())?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                let f32_bytes = dev
                    .dtoh_sync_copy(&df)
                    .map_err(|e| e.to_string())?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect::<Vec<_>>();
                if bf16_bytes != bf16_ref || f32_bytes != f32_ref {
                    artifact::write_sparse_new(
                        &out.join(format!("failure-{repetition}-{instrumented}.bf16")),
                        &bf16_bytes,
                    )
                    .map_err(|e| e.to_string())?;
                    artifact::write_sparse_new(
                        &out.join(format!("failure-{repetition}-{instrumented}.f32")),
                        &f32_bytes,
                    )
                    .map_err(|e| e.to_string())?;
                    return Err("complete phase output differs frozen509 BF16/F32 reference".into());
                }
                let raw = dev.dtoh_sync_copy(&dc).map_err(|e| e.to_string())?;
                if raw[words..].iter().any(|&v| v != u64::MAX)
                    || (!instrumented && raw[..words].iter().any(|&v| v != u64::MAX))
                {
                    return Err("phase counter guard/control modified".into());
                }
                let mut phases = serde_json::Map::new();
                if instrumented {
                    for record in raw[..words].chunks_exact(FIELDS.len()) {
                        if record.iter().any(|&v| v == 0 || v == u64::MAX)
                            || record[..4].iter().sum::<u64>() > record[4]
                            || record[6..9].iter().sum::<u64>() > record[0]
                        {
                            return Err("invalid raw phase intervals".into());
                        }
                    }
                    let bytes = raw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                    let path = out.join(format!("phase-{repetition}.u64"));
                    artifact::write_sparse_new(&path, &bytes).map_err(|e| e.to_string())?;
                    for (index, key) in FIELDS.iter().enumerate() {
                        let mut values = raw[..words]
                            .chunks_exact(FIELDS.len())
                            .map(|v| v[index])
                            .collect::<Vec<_>>();
                        values.sort_unstable();
                        phases.insert((*key).into(),json!({"min":values[0],"median":values[values.len()/2],"max":values[values.len()-1],"records":values.len()}));
                    }
                    phases.insert(
                        "raw".into(),
                        serde_json::to_value(Identity {
                            path,
                            bytes: bytes.len(),
                            sha256: format!("{:x}", Sha256::digest(&bytes)),
                        })
                        .map_err(|e| e.to_string())?,
                    );
                }
                minimum_free = minimum_free.min(
                    cudarc::driver::result::mem_get_info()
                        .map_err(|e| e.to_string())?
                        .0,
                );
                if minimum_free < p.safety_margin_bytes {
                    return Err("phase measured free VRAM breached reserve".into());
                }
                results.push(json!({"repetition":repetition,"instrumented":instrumented,"diagnostic_host_launch_plus_completion_seconds":seconds,"complete_measured_bf16_sha256":format!("{:x}",Sha256::digest(&bf16_bytes)),"complete_measured_f32_sha256":format!("{:x}",Sha256::digest(&f32_bytes)),"complete_reference_exact":true,"counter_guard_exact":true,"phases":phases}));
            }
        }
        t.operator.read()?;
        t.cuda_source.read()?;
        t.ptx.read()?;
        t.build_report.read()?;
        t.complete_bf16_reference.read()?;
        t.complete_f32_reference.read()?;
        for v in p.capture.values() {
            v.read()?;
        }
        Ok(
            json!({"passed":true,"adopted":false,"instrumentation_experiment":true,"plan":t,"device_uuid":actual_uuid,"cc":cc,"fields":FIELDS,"warps":warps,"grid":grid.grid_dim,"records":records,"shape":{"rows":r,"cols":k,"height":h},"occupancy_api_max_blocks_per_sm":{"original":original_max_blocks,"instrumented":instrumented_max_blocks,"actual_runtime_utilization_measured":false},"actual_tensor_bytes":tensor_bytes,"observed_free_minimum":minimum_free,"safety_margin_bytes":p.safety_margin_bytes,"results":results,"meaning":"Per-warp lane0 clock intervals include scheduling and synchronization; not exclusive utilization, occupancy, bandwidth or speed benchmarks; original source/arithmetic matched, read clocks perturb scheduling/register allocation"}),
        )
    }
    pub fn entry() -> Result<(), String> {
        let args = std::env::args().collect::<Vec<_>>();
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") || args.len() != 4 {
            return Err(
                "use ./dev row-projection-phase-probe build|probe PLAN FRESH_OUTPUT".into(),
            );
        }
        let out = Path::new(&args[3]);
        fs::create_dir(out).map_err(|e| e.to_string())?;
        let result = match args[1].as_str() {
            "build" => compile(Path::new(&args[2]), out),
            "probe" => phase(Path::new(&args[2]), out),
            _ => Err("unknown phase operation".into()),
        };
        match result {
            Ok(r) => {
                fs::write(
                    out.join("REPORT.json"),
                    serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                Ok(())
            }
            Err(e) => {
                fs::write(out.join("FAILURE.txt"), &e).map_err(|e| e.to_string())?;
                Err(e)
            }
        }
    }
}
fn main() {
    if let Err(e) = operator::entry() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
