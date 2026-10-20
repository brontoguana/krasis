#[cfg(has_decode_kernels)]
mod diagnostic {
    use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
    use half::bf16;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs,
        io::Read,
        path::{Path, PathBuf},
        sync::Arc,
    };

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct FileIdentity {
        name: String,
        path: PathBuf,
        bytes: usize,
        sha256: String,
    }
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Plan {
        schema_version: u32,
        gpu_ordinal: usize,
        files: Vec<FileIdentity>,
        lengths: Vec<usize>,
        short_prefix: usize,
        segments: Vec<usize>,
        samples: usize,
    }
    fn checked_product(factors: &[usize]) -> Result<usize, String> {
        factors.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(b).ok_or("capture geometry overflow".into())
        })
    }
    fn validate(plan: &Plan, height: usize, heads: usize, dim: usize) -> Result<(), String> {
        if plan.schema_version != 1
            || heads == 0
            || dim == 0
            || dim
                .checked_next_power_of_two()
                .is_none_or(|n| n.max(32) > 1024)
            || height == 0
            || height > i32::MAX as usize
            || heads > i32::MAX as usize
            || dim > i32::MAX as usize
            || plan.samples == 0
            || plan.samples > 1024
            || plan.short_prefix == 0
            || plan.lengths.is_empty()
            || plan.lengths.iter().any(|&n| n == 0 || n > height)
            || plan.segments.is_empty()
            || plan.segments.contains(&0)
        {
            return Err("invalid bounded raw recurrence plan/ABI".into());
        }
        let total = plan.segments.iter().try_fold(0usize, |a, &b| {
            a.checked_add(b).ok_or("segment sum overflow")
        })?;
        if total != height
            || !plan.lengths.contains(&height)
            || !plan.lengths.contains(&plan.short_prefix)
            || !plan.lengths.iter().any(|&n| n > plan.short_prefix)
        {
            return Err("plan does not bind the complete segmentation and short prefix".into());
        }
        checked_product(&[height, heads, dim, 10])?;
        checked_product(&[heads, dim, dim, 4])?;
        Ok(())
    }
    fn identified_prefix(file: &FileIdentity, take: usize) -> Result<Vec<u8>, String> {
        if take > file.bytes
            || fs::metadata(&file.path).map_err(|e| e.to_string())?.len() != file.bytes as u64
        {
            return Err(format!("input size mismatch: {}", file.path.display()));
        }
        let mut input = fs::File::open(&file.path).map_err(|e| e.to_string())?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; file.bytes.min(1024 * 1024).max(1)];
        let mut retained = Vec::with_capacity(take);
        loop {
            let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
            let keep = count.min(take - retained.len());
            retained.extend_from_slice(&buffer[..keep]);
        }
        if format!("{:x}", digest.finalize()) != file.sha256 || retained.len() != take {
            return Err(format!("input checksum mismatch: {}", file.path.display()));
        }
        Ok(retained)
    }
    fn f32_values(raw: &[u8]) -> Result<Vec<f32>, String> {
        if raw.len() % 4 != 0 {
            return Err("unaligned captured F32".into());
        }
        let values = raw
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite captured F32".into());
        }
        Ok(values)
    }
    fn bf16_values(raw: &[u8]) -> Result<Vec<u16>, String> {
        if raw.len() % 2 != 0 {
            return Err("unaligned captured BF16".into());
        }
        let values = raw
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values
            .iter()
            .any(|&v| !bf16::from_bits(v).to_f32().is_finite())
        {
            return Err("nonfinite captured BF16".into());
        }
        Ok(values)
    }
    fn save16(path: &Path, values: &[u16]) -> Result<(), String> {
        fs::write(
            path,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())
    }
    fn save32(path: &Path, values: &[f32]) -> Result<(), String> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite recurrence state".into());
        }
        fs::write(
            path,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())
    }
    fn configured_reserve(raw: &[u8]) -> Result<usize, String> {
        let text = std::str::from_utf8(raw).map_err(|e| e.to_string())?;
        let entries = text
            .lines()
            .filter_map(|line| line.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
            .collect::<Vec<_>>();
        if entries.len() != 1 {
            return Err("explicit unique runtime safety margin required".into());
        }
        let quoted = entries[0]
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .ok_or("runtime margin must be quoted integer MiB")?;
        let mib = quoted
            .parse::<usize>()
            .map_err(|_| "invalid runtime safety margin")?;
        if mib == 0 {
            return Err("runtime safety margin cannot be zero".into());
        }
        mib.checked_mul(1024 * 1024)
            .ok_or("runtime safety margin overflow".into())
    }
    fn require_free(observation: &Value, required: usize) -> Result<(), String> {
        let free = observation["free"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("invalid measured free memory")?;
        if free < required {
            return Err(format!("raw operator requires {required} bytes including configured safety margin; {free} bytes actually available"));
        }
        Ok(())
    }
    fn memory() -> Result<Value, String> {
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) };
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("memory observation failed: {rc:?}"));
        }
        Ok(json!({"free":free,"total":total}))
    }
    struct Events {
        start: sys::CUevent,
        stop: sys::CUevent,
    }
    impl Events {
        fn new() -> Result<Self, String> {
            let mut e = Self {
                start: std::ptr::null_mut(),
                stop: std::ptr::null_mut(),
            };
            for p in [&mut e.start, &mut e.stop] {
                let rc = unsafe { sys::lib().cuEventCreate(p, 0) };
                if rc != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("event create: {rc:?}"));
                }
            }
            Ok(e)
        }
        fn record(&self, event: sys::CUevent, stream: sys::CUstream) -> Result<(), String> {
            let rc = unsafe { sys::lib().cuEventRecord(event, stream) };
            if rc != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("event record: {rc:?}"));
            }
            Ok(())
        }
        fn elapsed(&self) -> Result<f32, String> {
            let mut ms = 0f32;
            let rc = unsafe { sys::lib().cuEventElapsedTime(&mut ms, self.start, self.stop) };
            if rc != sys::CUresult::CUDA_SUCCESS || !ms.is_finite() || ms <= 0.0 {
                return Err(format!("event elapsed: {rc:?}, {ms}"));
            }
            Ok(ms)
        }
    }
    impl Drop for Events {
        fn drop(&mut self) {
            for e in [self.start, self.stop] {
                if !e.is_null() {
                    unsafe {
                        let _ = sys::lib().cuEventDestroy_v2(e);
                    }
                }
            }
        }
    }
    struct DeviceInputs {
        device: Arc<CudaDevice>,
        q: CudaSlice<u16>,
        k: CudaSlice<u16>,
        v: CudaSlice<u16>,
        gate: CudaSlice<u16>,
        beta: CudaSlice<u16>,
        a: CudaSlice<f32>,
        dt: CudaSlice<f32>,
        state: CudaSlice<f32>,
        output: CudaSlice<u16>,
        heads: usize,
        dim: usize,
        lower: f32,
        scale: f32,
        eps: f32,
    }
    impl DeviceInputs {
        fn reset(&self, initial: &[f32]) -> Result<(), String> {
            self.device.synchronize().map_err(|e| e.to_string())?;
            let rc = unsafe {
                sys::lib().cuMemcpyHtoD_v2(
                    *self.state.device_ptr(),
                    initial.as_ptr().cast(),
                    initial.len() * 4,
                )
            };
            if rc != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("state reset failed: {rc:?}"));
            }
            Ok(())
        }
        fn execute(
            &self,
            variant: &str,
            start: usize,
            tokens: usize,
            timer: Option<&Events>,
        ) -> Result<Option<f32>, String> {
            let tiled = match variant {
                "shared-tree" => false,
                "warp-tree" => true,
                _ => return Err("unknown raw recurrence contract".into()),
            };
            let name = if tiled {
                "kimi_delta_recurrent_scan_tiled_bf16"
            } else {
                "kimi_delta_recurrent_scan_bf16"
            };
            let mut state = *self.state.device_ptr();
            let offset = checked_product(&[start, self.heads, self.dim, 2])? as u64;
            let mut q = *self.q.device_ptr() + offset;
            let mut k = *self.k.device_ptr() + offset;
            let mut v = *self.v.device_ptr() + offset;
            let mut gate = *self.gate.device_ptr() + offset;
            let mut beta =
                *self.beta.device_ptr() + checked_product(&[start, self.heads, 2])? as u64;
            let mut a = *self.a.device_ptr();
            let mut dt = *self.dt.device_ptr();
            let mut output = *self.output.device_ptr() + offset;
            let (mut t, mut h, mut d) = (
                i32::try_from(tokens).map_err(|_| "token ABI overflow")?,
                self.heads as i32,
                self.dim as i32,
            );
            let (mut lower, mut scale, mut eps) = (self.lower, self.scale, self.eps);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut state as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut k as *mut u64).cast(),
                (&mut v as *mut u64).cast(),
                (&mut gate as *mut u64).cast(),
                (&mut beta as *mut u64).cast(),
                (&mut a as *mut u64).cast(),
                (&mut dt as *mut u64).cast(),
                (&mut output as *mut u64).cast(),
                (&mut t as *mut i32).cast(),
                (&mut h as *mut i32).cast(),
                (&mut d as *mut i32).cast(),
                (&mut lower as *mut f32).cast(),
                (&mut scale as *mut f32).cast(),
                (&mut eps as *mut f32).cast(),
            ];
            let threads = if tiled {
                32
            } else {
                self.dim.next_power_of_two().max(32)
            };
            let shared = if tiled {
                3 * self.dim + 2
            } else {
                3 * self.dim + threads
            };
            let cfg = LaunchConfig {
                grid_dim: (
                    self.heads as u32,
                    if tiled {
                        self.dim.div_ceil(32) as u32
                    } else {
                        1
                    },
                    1,
                ),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: (shared * 4) as u32,
            };
            if let Some(e) = timer {
                e.record(e.start, *self.device.cu_stream())?;
            }
            let function = self
                .device
                .get_func("raw_recurrence", name)
                .ok_or("actual production raw kernel missing")?;
            unsafe {
                function
                    .launch(cfg, args.as_mut_slice())
                    .map_err(|e| e.to_string())?;
            }
            if let Some(e) = timer {
                e.record(e.stop, *self.device.cu_stream())?;
            }
            self.device.synchronize().map_err(|e| e.to_string())?;
            timer.map(Events::elapsed).transpose()
        }
        fn result(&self, tokens: usize) -> Result<(Vec<u16>, Vec<f32>), String> {
            let mut output = self
                .device
                .dtoh_sync_copy(&self.output)
                .map_err(|e| e.to_string())?;
            output.truncate(checked_product(&[tokens, self.heads, self.dim])?);
            if output
                .iter()
                .any(|&v| !bf16::from_bits(v).to_f32().is_finite())
            {
                return Err("nonfinite recurrence output".into());
            }
            let state = self
                .device
                .dtoh_sync_copy(&self.state)
                .map_err(|e| e.to_string())?;
            if state.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite recurrence state".into());
            }
            Ok((output, state))
        }
    }
    fn same_state(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
    fn number(v: &Value, key: &str) -> Result<usize, String> {
        v[key]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or(format!("invalid descriptor {key}"))
    }
    fn probe(plan: &Plan, out: &Path) -> Result<Value, String> {
        let mut files = BTreeMap::new();
        for file in &plan.files {
            if files.insert(file.name.as_str(), file).is_some() {
                return Err("duplicate capture identity".into());
            }
        }
        let source = |name: &str| {
            files
                .get(name)
                .copied()
                .ok_or(format!("missing captured {name}"))
        };
        let descriptor_file = source("descriptor")?;
        let descriptor: Value =
            serde_json::from_slice(&identified_prefix(descriptor_file, descriptor_file.bytes)?)
                .map_err(|e| e.to_string())?;
        let (height, heads, dim) = (
            number(&descriptor, "height")?,
            number(&descriptor, "heads")?,
            number(&descriptor, "dim")?,
        );
        validate(plan, height, heads, dim)?;
        let parameter = |key: &str| {
            descriptor[key]
                .as_f64()
                .map(|f| f as f32)
                .filter(|f| f.is_finite())
                .ok_or(format!("invalid {key}"))
        };
        let (lower, scale, eps) = (
            parameter("gate_lower_bound")?,
            parameter("q_scale")?,
            parameter("normalization_epsilon")?,
        );
        if lower >= 0.0 || scale <= 0.0 || eps <= 0.0 {
            return Err("invalid recurrence math parameters".into());
        }
        let runtime = source("runtime-config")?;
        let reserve_bytes = configured_reserve(&identified_prefix(runtime, runtime.bytes)?)?;
        let source_code = source("cuda-source")?;
        identified_prefix(source_code, 0)?;
        let count = checked_product(&[height, heads, dim])?;
        let states = checked_product(&[heads, dim, dim])?;
        let captured = |name: &str, bytes: usize| -> Result<Vec<u8>, String> {
            let f = source(name)?;
            if f.bytes != bytes {
                return Err(format!("capture descriptor size mismatch {name}"));
            }
            identified_prefix(f, bytes)
        };
        let q = bf16_values(&captured("q", count * 2)?)?;
        let k = bf16_values(&captured("k", count * 2)?)?;
        let v = bf16_values(&captured("v", count * 2)?)?;
        let gate = bf16_values(&captured("gates", count * 2)?)?;
        let beta = bf16_values(&captured("beta", height * heads * 2)?)?;
        let a = f32_values(&captured("a-log", heads * 4)?)?;
        let dt = f32_values(&captured("dt-bias", heads * dim * 4)?)?;
        let initial = f32_values(&captured("initial-state", states * 4)?)?;
        let device = CudaDevice::new(plan.gpu_ordinal).map_err(|e| e.to_string())?;
        let before_module = memory()?;
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/decode_kernels.ptx"));
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(ptx),
                "raw_recurrence",
                &[
                    "kimi_delta_recurrent_scan_bf16",
                    "kimi_delta_recurrent_scan_tiled_bf16",
                ],
            )
            .map_err(|e| e.to_string())?;
        let after_module = memory()?;
        let tensor_bytes = [
            checked_product(&[count, 10])?,
            checked_product(&[height, heads, 2])?,
            checked_product(&[heads, 4])?,
            checked_product(&[heads, dim, 4])?,
            checked_product(&[states, 4])?,
        ]
        .iter()
        .try_fold(0usize, |a, &b| {
            a.checked_add(b).ok_or("tensor byte sum overflow")
        })?;
        require_free(
            &after_module,
            tensor_bytes
                .checked_add(reserve_bytes)
                .ok_or("operator plus margin overflow")?,
        )?;
        let upload16 = |values: Vec<u16>| device.htod_copy(values).map_err(|e| e.to_string());
        let upload32 = |values: Vec<f32>| device.htod_copy(values).map_err(|e| e.to_string());
        let inputs = DeviceInputs {
            device: device.clone(),
            q: upload16(q)?,
            k: upload16(k)?,
            v: upload16(v)?,
            gate: upload16(gate)?,
            beta: upload16(beta)?,
            a: upload32(a)?,
            dt: upload32(dt)?,
            state: upload32(initial.clone())?,
            output: device
                .alloc_zeros::<u16>(count)
                .map_err(|e| e.to_string())?,
            heads,
            dim,
            lower,
            scale,
            eps,
        };
        let after_buffers = memory()?;
        let events = Events::new()?;
        let after_events = memory()?;
        require_free(&after_events, reserve_bytes)?;
        let mut results = Vec::new();
        for variant in ["shared-tree", "warp-tree"] {
            let dir = out.join(variant);
            fs::create_dir(&dir).map_err(|e| e.to_string())?;
            let mut full_result = None;
            for &tokens in &plan.lengths {
                inputs.reset(&initial)?;
                inputs.execute(variant, 0, tokens, None)?;
                let (output, state) = inputs.result(tokens)?;
                save16(&dir.join(format!("tokens{tokens}-output.bf16")), &output)?;
                save32(&dir.join(format!("tokens{tokens}-state.f32")), &state)?;
                let mut times = Vec::with_capacity(plan.samples);
                for _ in 0..plan.samples {
                    inputs.reset(&initial)?;
                    times.push(
                        inputs
                            .execute(variant, 0, tokens, Some(&events))?
                            .ok_or("missing explicit timing")?,
                    );
                }
                let warmed = inputs.result(tokens)?;
                if warmed.0 != output || !same_state(&warmed.1, &state) {
                    return Err("raw operator repetitions changed output/state".into());
                }
                results.push(json!({"variant":variant,"tokens":tokens,"event_milliseconds":times,"instrumented_operator_measurement":true,"production_speed_benchmark":false,"repetition_output_and_state_exact":true}));
                if tokens == height {
                    full_result = Some((output, state));
                }
            }
            inputs.reset(&initial)?;
            let mut consumed = 0usize;
            for &segment in &plan.segments {
                inputs.execute(variant, consumed, segment, None)?;
                consumed += segment;
                let state = inputs
                    .device
                    .dtoh_sync_copy(&inputs.state)
                    .map_err(|e| e.to_string())?;
                save32(
                    &dir.join(format!("segmented-boundary{consumed}-state.f32")),
                    &state,
                )?;
            }
            let segmented = inputs.result(height)?;
            let full = full_result.ok_or("full-length case missing")?;
            save16(&dir.join("segmented-output.bf16"), &segmented.0)?;
            if full.0 != segmented.0 || !same_state(&full.1, &segmented.1) {
                return Err(
                    "same raw kernel/full-versus-segmented output or final state differs".into(),
                );
            }
            results.push(json!({"variant":variant,"segmentation":plan.segments,"identical_raw_inputs":true,"full_segmented_outputs_exact":true,"full_segmented_final_state_exact":true,"prefix_states":"actual standalone segmented recurrence states; not unavailable native chunk register carry"}));
        }
        let after_runs = memory()?;
        require_free(&after_runs, reserve_bytes)?;
        drop(events);
        let after_event_destroy = memory()?;
        Ok(
            json!({"diagnostic_only":true,"qualifies_candidate":false,"descriptor":descriptor,"plan":plan,"embedded_ptx_sha256":format!("{:x}",Sha256::digest(ptx.as_bytes())),"visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),"memory":{"before_module":before_module,"after_module":after_module,"after_buffers":after_buffers,"after_events":after_events,"after_runs":after_runs,"after_event_destroy":after_event_destroy},"actual_tensor_buffer_bytes":tensor_bytes,"configured_safety_margin_bytes":reserve_bytes,"results":results,"limits":["Both actual unchanged CUDA kernels on identical captured raw operands; no full model generation or consistency qualification","Prepared-input/normalization/exp and final BF16 rounding contracts remain explicit in independent Witness comparisons","Operator timings have diagnostics enabled; no production speed claim","Model graphs/caching/calibration/defaults are untouched; standalone operator launches do not stand in for ordinary cache execution"]}),
        )
    }
    fn run() -> Result<(), String> {
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
            return Err("Run through ./dev raw-recurrence".into());
        }
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args == ["--identity"] {
            println!(
                "{}",
                json!({"diagnostic_only":true,"qualifies_candidate":false,"embedded_decode_ptx_sha256":format!("{:x}",Sha256::digest(include_str!(concat!(env!("OUT_DIR"),"/decode_kernels.ptx")).as_bytes()))})
            );
            return Ok(());
        }
        if args == ["--help"] {
            println!("./dev raw-recurrence <plan.json> <fresh-output>; --identity reports the embedded actual CUDA module");
            return Ok(());
        }
        if args.len() != 2 {
            return Err("usage: ./dev raw-recurrence <plan.json> <fresh-output>".into());
        }
        let raw = fs::read(&args[0]).map_err(|e| e.to_string())?;
        let plan: Plan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        let out = PathBuf::from(&args[1]);
        fs::create_dir(&out).map_err(|e| format!("fresh evidence directory required: {e}"))?;
        let report = probe(&plan, &out)?;
        fs::write(
            out.join("REPORT.json"),
            serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "Raw recurrence diagnostics saved: {} (not candidate qualification)",
            out.display()
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn plan() -> Plan {
            Plan {
                schema_version: 1,
                gpu_ordinal: 0,
                files: vec![],
                lengths: vec![4, 64, 100],
                short_prefix: 4,
                segments: vec![80, 7, 13],
                samples: 7,
            }
        }
        #[test]
        fn segmentation_admission_rejects_incomplete_history_and_overflow_before_allocation() {
            assert!(validate(&plan(), 100, 3, 128).is_ok());
            let mut p = plan();
            p.segments = vec![80, 7, 12];
            assert!(validate(&p, 100, 3, 128).is_err());
            let mut p = plan();
            p.segments = vec![80, 0, 20];
            assert!(validate(&p, 100, 3, 128).is_err());
            assert!(validate(&plan(), 100, 3, usize::MAX).is_err());
            assert!(validate(&plan(), 100, 0, 128).is_err());
            assert!(checked_product(&[usize::MAX, 2]).is_err());
        }
        #[test]
        fn runtime_memory_contract_has_no_default_or_underallocation() {
            assert_eq!(
                configured_reserve(b"CFG_VRAM_SAFETY_MARGIN=\"17\"\n").unwrap(),
                17 * 1024 * 1024
            );
            for raw in [
                &b""[..],
                &b"CFG_VRAM_SAFETY_MARGIN=\"0\"\n"[..],
                &b"CFG_VRAM_SAFETY_MARGIN=oops\n"[..],
            ] {
                assert!(configured_reserve(raw).is_err());
            }
            assert!(require_free(&json!({"free":99}), 100).is_err());
            assert!(require_free(&json!({"free":100}), 100).is_ok());
        }
        #[test]
        fn persistent_state_equality_distinguishes_signed_zero() {
            assert!(!same_state(&[0.0], &[-0.0]));
            assert!(same_state(&[1.0, -0.0], &[1.0, -0.0]));
        }
        #[test]
        fn capture_hash_checks_bytes_beyond_retained_prefix() {
            let path = std::env::temp_dir().join(format!(
                "raw-recurrence-{}-{}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::write(&path, b"abcd").unwrap();
            let file = FileIdentity {
                name: "q".into(),
                path: path.clone(),
                bytes: 4,
                sha256: format!("{:x}", Sha256::digest(b"abcd")),
            };
            assert_eq!(identified_prefix(&file, 2).unwrap(), b"ab");
            fs::write(&path, b"abce").unwrap();
            assert!(identified_prefix(&file, 2).is_err());
            fs::remove_file(path).unwrap();
        }
    }
    pub fn main() {
        if let Err(e) = run() {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
#[cfg(has_decode_kernels)]
fn main() {
    diagnostic::main()
}
#[cfg(not(has_decode_kernels))]
fn main() {
    eprintln!("raw-recurrence requires built CUDA decode kernels; unavailable in this build");
    std::process::exit(1)
}
