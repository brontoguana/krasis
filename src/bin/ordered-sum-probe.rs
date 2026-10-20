// Private primitive qualification. No model dispatch or speed benchmark.
mod operator {
    include!("row-projection-compensated-probe.rs");
    use cudarc::driver::CudaSlice;
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct SumPlan {
        config: Identity,
        pairs: Identity,
        expected: Identity,
        cuda_source: Identity,
        ptx: Identity,
        build_report: Identity,
        gpu_uuid: String,
        gpu_name: String,
        compute_capability: u32,
        safety_margin_bytes: usize,
    }
    struct SumBuffers {
        device: std::sync::Arc<CudaDevice>,
        pairs: Option<CudaSlice<u32>>,
        output: Option<CudaSlice<u32>>,
        flags: Option<CudaSlice<u32>>,
    }
    impl Drop for SumBuffers {
        fn drop(&mut self) {
            // Partial allocation failures also drain before every owned slice is freed.
            if let Err(e) = self.device.synchronize() {
                eprintln!("ERROR: ordered sum owner completion failed: {e}");
            }
        }
    }
    fn u32s(raw: &[u8]) -> Result<Vec<u32>, String> {
        if raw.len() % 4 != 0 {
            return Err("u32 extent mismatch".into());
        }
        Ok(raw
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().expect("exact4")))
            .collect())
    }
    fn qualify(plan: &Path, out: &Path) -> Result<Value, String> {
        let p: SumPlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if p.safety_margin_bytes != 600 * 1024 * 1024 {
            return Err("user 600 MiB reserve required".into());
        }
        let config: Value = serde_json::from_slice(&p.config.read()?).map_err(|e| e.to_string())?;
        if config["contract"]
            != "protected_ieee_f32_magnitude_ordered_fasttwosum_positive_zero_error_v1"
            || config["passed"] != true
            || config["input"] != serde_json::to_value(&p.pairs).map_err(|e| e.to_string())?
            || config["reference"]
                != serde_json::to_value(&p.expected).map_err(|e| e.to_string())?
        {
            return Err("independent config/input/reference contract mismatch".into());
        }
        let count = usize::try_from(config["count"].as_u64().ok_or("missing actual count")?)
            .map_err(|e| e.to_string())?;
        let pairs = u32s(&p.pairs.read()?)?;
        let expected = u32s(&p.expected.read()?)?;
        if count == 0 || pairs.len() != mul(count, 2)? || expected.len() != mul(count, 3)? {
            return Err("primitive input/reference extents mismatch".into());
        }
        let built: Value =
            serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
        if built["plan"]["source"]
            != serde_json::to_value(&p.cuda_source).map_err(|e| e.to_string())?
            || built["ptx"] != serde_json::to_value(&p.ptx).map_err(|e| e.to_string())?
        {
            return Err("primitive CUDA/PTX/build identity mismatch".into());
        }
        p.cuda_source.read()?;
        let ptx = p.ptx.read()?;
        let dev = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
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
            return Err("actual device identity/capability mismatch".into());
        }
        let block = usize::min(
            256,
            usize::try_from(attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?)
                .map_err(|e| e.to_string())?,
        );
        if block == 0 {
            return Err("device reports no thread capacity".into());
        }
        let blocks = count.div_ceil(block);
        if blocks
            > usize::try_from(attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?)
                .map_err(|e| e.to_string())?
        {
            return Err("actual grid unsupported".into());
        }
        dev.load_ptx(
            cudarc::nvrtc::Ptx::from_src(String::from_utf8(ptx).map_err(|e| e.to_string())?),
            "ordered_sum_primitive",
            &["ordered_sum_pair_probe"],
        )
        .map_err(|e| e.to_string())?;
        let kernel = dev
            .get_func("ordered_sum_primitive", "ordered_sum_pair_probe")
            .ok_or("missing primitive kernel")?;
        let out_words = add(mul(count, 4)?, 32)?;
        let flag_words = add(count, 32)?;
        let tensor_bytes = mul(add(add(pairs.len(), out_words)?, flag_words)?, 4)?;
        let baseline_free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if baseline_free < add(tensor_bytes, p.safety_margin_bytes)? {
            return Err("primitive tensors exceed actual measured reserve".into());
        }
        let mut buffers = SumBuffers {
            device: dev.clone(),
            pairs: None,
            output: None,
            flags: None,
        };
        buffers.pairs = Some(dev.htod_copy(pairs).map_err(|e| e.to_string())?);
        buffers.output = Some(
            dev.htod_copy(vec![0xa5a5a5a5u32; out_words])
                .map_err(|e| e.to_string())?,
        );
        buffers.flags = Some(
            dev.htod_copy(vec![u32::MAX; flag_words])
                .map_err(|e| e.to_string())?,
        );
        let free = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if free < p.safety_margin_bytes {
            return Err("actual primitive allocation breached reserve".into());
        }
        let launch = cudarc::driver::LaunchConfig {
            grid_dim: (u32::try_from(blocks).map_err(|e| e.to_string())?, 1, 1),
            block_dim: (u32::try_from(block).map_err(|e| e.to_string())?, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernel.launch(
                launch,
                (
                    buffers.pairs.as_ref().ok_or("pair owner absent")?,
                    buffers.output.as_mut().ok_or("output owner absent")?,
                    buffers.flags.as_mut().ok_or("flag owner absent")?,
                    count as u64,
                ),
            )
        }
        .map_err(|e| e.to_string())?;
        dev.synchronize().map_err(|e| e.to_string())?;
        let output = dev
            .dtoh_sync_copy(buffers.output.as_ref().ok_or("output owner absent")?)
            .map_err(|e| e.to_string())?;
        let flags = dev
            .dtoh_sync_copy(buffers.flags.as_ref().ok_or("flag owner absent")?)
            .map_err(|e| e.to_string())?;
        let raw = output
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let raw_flags = flags
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(out.join("actual-results.u32"), &raw).map_err(|e| e.to_string())?;
        fs::write(out.join("actual-flags.u32"), &raw_flags).map_err(|e| e.to_string())?;
        if output[count * 4..].iter().any(|&v| v != 0xa5a5a5a5)
            || flags[count..].iter().any(|&v| v != u32::MAX)
        {
            return Err("primitive output/status guard modified".into());
        }
        let mut counts = [0usize; 3];
        let mut failures = Vec::new();
        let mut mismatches = 0usize;
        for row in 0..count {
            let flag = expected[row * 3];
            if flag > 2 {
                return Err("unsupported reference rejection flag".into());
            }
            counts[flag as usize] += 1;
            let actual = &output[row * 4..row * 4 + 4];
            let wanted = if flag == 0 {
                [
                    expected[row * 3 + 1],
                    expected[row * 3 + 2],
                    expected[row * 3 + 1],
                    expected[row * 3 + 2],
                ]
            } else {
                [0xa5a5a5a5; 4]
            };
            if flags[row] != flag || actual != wanted {
                mismatches += 1;
                if failures.len() < 32 {
                    failures.push(json!({"row":row,"expected_status":flag,"actual_status":flags[row],"expected":wanted,"actual":actual}));
                }
            }
        }
        let result = json!({"passed":mismatches==0,"plan":p,"count":count,"finite":counts[0],"overflow_rejected":counts[1],"nonfinite_rejected":counts[2],"mismatches":mismatches,"failures":failures,"guards_exact":true,"complete_general_ordered_and_independent_dyadic_words_exact":mismatches==0,"device_uuid":actual_uuid,"cc":cc,"tensor_bytes":tensor_bytes,"free_before":baseline_free,"free_after_allocation":free,"scope":"primitive only; not complete model/operator/conversation/performance acceptance","adopted":false});
        fs::write(
            out.join("AUDIT.json"),
            serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if mismatches != 0 {
            return Err(format!("primitive exact mismatches: {mismatches}"));
        }
        Ok(result)
    }
    pub fn entry() -> Result<(), String> {
        let args = std::env::args().collect::<Vec<_>>();
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") || args.len() != 4 {
            return Err("use ./dev ordered-sum-probe build|probe PLAN FRESH_OUTPUT".into());
        }
        let out = Path::new(&args[3]);
        fs::create_dir(out).map_err(|e| e.to_string())?;
        let result = match args[1].as_str() {
            "build" => build(Path::new(&args[2]), out),
            "probe" => qualify(Path::new(&args[2]), out),
            _ => Err("unsupported primitive operation".into()),
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
