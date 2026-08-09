//! Real-data correctness and throughput gate for the GPU expert entropy codec.

#[cfg(not(has_expert_codec_kernels))]
compile_error!("expert-compression-gpu requires nvcc-built expert codec kernels");

#[cfg(has_expert_codec_kernels)]
mod probe {
    use cudarc::driver::sys;
    use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use krasis::expert_codec::{
        encode_expert, plan_expert_chunks, CodecHistogram, ComponentKind, EncodedExpert,
        ExpertComponent, CODEC_LANES,
    };
    use memmap2::{Mmap, MmapOptions};
    use serde::{Deserialize, Serialize};
    use std::fs::File;
    use std::path::PathBuf;
    use std::ptr;
    use std::sync::Arc;

    const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/expert_codec_kernels.ptx"));
    const CACHE_HEADER_BYTES: usize = 64;
    const MARLIN_CACHE_VERSION: u32 = 7;
    type ProbeResult<T> = Result<T, String>;

    #[derive(Debug)]
    struct Args {
        cache: PathBuf,
        model_config: PathBuf,
        gpu: usize,
        expert_ordinals: Vec<usize>,
        lane_bytes: Vec<usize>,
        warmup: usize,
        samples: usize,
        pcie_gbps: f64,
        minimum_decoder_gbps: f64,
        registration_sidecar: Option<PathBuf>,
        output: Option<PathBuf>,
    }

    #[derive(Deserialize)]
    struct ModelConfig {
        #[serde(default = "default_experts_gated")]
        experts_gated: bool,
    }

    fn default_experts_gated() -> bool {
        true
    }

    struct CacheLayout {
        hidden_size: usize,
        intermediate_size: usize,
        experts_per_layer: usize,
        moe_layers: usize,
        group_size: usize,
        w13_packed: usize,
        w13_scales: usize,
        w2_packed: usize,
        w2_scales: usize,
        expert_bytes: usize,
    }

    #[derive(Serialize)]
    struct Distribution {
        min_us: f64,
        p05_us: f64,
        p50_us: f64,
        p95_us: f64,
        p99_us: f64,
        max_us: f64,
        mean_us: f64,
    }

    #[derive(Serialize)]
    struct CandidateReport {
        lane_bytes: usize,
        original_bytes: u64,
        encoded_bytes: u64,
        encoded_ratio: f64,
        saving_pct: f64,
        task_count: u64,
        kernel_time: Distribution,
        p50_decoder_gbps: f64,
        p95_time_decoder_gbps: f64,
        predicted_p95_pipeline_us: f64,
        predicted_raw_copy_us: f64,
        predicted_pipeline_speedup: f64,
        every_cpu_round_trip_bit_exact: bool,
        every_gpu_round_trip_bit_exact: bool,
    }

    #[derive(Serialize)]
    struct H2dBackingReport {
        transfer_bytes: usize,
        file_backed_time: Distribution,
        file_backed_gbps_at_p95_time: f64,
        anonymous_time: Distribution,
        anonymous_gbps_at_p95_time: f64,
        anonymous_hugepage_advice_succeeded: bool,
        payload_bit_exact: bool,
    }

    #[derive(Serialize)]
    struct Report {
        schema_version: u32,
        source_cache: String,
        source_cache_bytes: u64,
        model_config: String,
        gpu_ordinal: usize,
        gpu_name: String,
        expert_ordinals: Vec<usize>,
        expert_bytes: usize,
        warmup_samples_per_expert: usize,
        measured_samples_per_expert: usize,
        pcie_gbps: f64,
        minimum_decoder_gbps: f64,
        registration_sidecar: Option<String>,
        sidecar_registration_bytes: Option<usize>,
        sidecar_registration_passed: Option<bool>,
        sidecar_h2d_backing: Option<H2dBackingReport>,
        candidates: Vec<CandidateReport>,
        selected_lane_bytes: usize,
        selected_p95_time_decoder_gbps: f64,
        gate_passed: bool,
    }

    struct CudaEvent(sys::CUevent);

    impl CudaEvent {
        fn new() -> ProbeResult<Self> {
            let mut event = ptr::null_mut();
            cuda(
                unsafe { sys::lib().cuEventCreate(&mut event, 0) },
                "cuEventCreate",
            )?;
            Ok(Self(event))
        }
    }

    impl Drop for CudaEvent {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let result = unsafe { sys::lib().cuEventDestroy_v2(self.0) };
                assert_eq!(
                    result,
                    sys::CUresult::CUDA_SUCCESS,
                    "cuEventDestroy_v2 failed"
                );
            }
        }
    }

    pub fn main() -> ProbeResult<()> {
        let args = parse_args()?;
        let config: ModelConfig =
            serde_json::from_reader(File::open(&args.model_config).map_err(|error| {
                format!("failed to open {}: {error}", args.model_config.display())
            })?)
            .map_err(|error| format!("failed to parse model config: {error}"))?;
        let cache_file = File::open(&args.cache)
            .map_err(|error| format!("failed to open {}: {error}", args.cache.display()))?;
        let source_cache_bytes = cache_file
            .metadata()
            .map_err(|error| error.to_string())?
            .len();
        let mmap = unsafe { Mmap::map(&cache_file) }.map_err(|error| error.to_string())?;
        let layout = read_layout(&mmap, config.experts_gated)?;
        let total_experts = layout
            .moe_layers
            .checked_mul(layout.experts_per_layer)
            .ok_or_else(|| "cache expert count overflow".to_string())?;
        if args
            .expert_ordinals
            .iter()
            .any(|&ordinal| ordinal >= total_experts)
        {
            return Err(format!(
                "expert ordinal exceeds cache population {total_experts}"
            ));
        }

        let sources: Vec<&[u8]> = args
            .expert_ordinals
            .iter()
            .map(|&ordinal| expert_bytes(&mmap, &layout, ordinal))
            .collect::<ProbeResult<_>>()?;
        let component_sets: Vec<[ExpertComponent<'_>; 4]> = sources
            .iter()
            .map(|source| components(source, &layout))
            .collect();
        let mut histogram = CodecHistogram::default();
        for component_set in &component_sets {
            for &component in component_set {
                histogram.observe(component);
            }
        }
        let tables = histogram.build_tables()?;

        let device = CudaDevice::new_with_stream(args.gpu).map_err(driver_error)?;
        device.bind_to_thread().map_err(driver_error)?;
        let sidecar_registration = if let Some(path) = args.registration_sidecar.as_ref() {
            let registration_bytes = [
                layout.w13_packed,
                layout.w13_scales,
                layout.w2_packed,
                layout.w2_scales,
            ]
            .into_iter()
            .max()
            .unwrap_or(0)
            .checked_mul(layout.experts_per_layer)
            .ok_or_else(|| "sidecar registration geometry overflow".to_string())?;
            if registration_bytes == 0 {
                return Err("sidecar registration probe requires non-empty data".to_string());
            }
            let mmap = krasis::expert_sidecar::private_payload_mapping(path, registration_bytes)?;
            let sidecar_file = File::open(path)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
            let sidecar_file_bytes = usize::try_from(
                sidecar_file
                    .metadata()
                    .map_err(|error| format!("failed to stat {}: {error}", path.display()))?
                    .len(),
            )
            .map_err(|_| "sidecar file length exceeds usize".to_string())?;
            let mut encoded_header = [0_u8; krasis::expert_sidecar::SIDECAR_HEADER_BYTES];
            krasis::expert_sidecar::read_file_exact_at(
                &sidecar_file,
                &mut encoded_header,
                0,
            )
                .map_err(|error| format!("failed to read sidecar header: {error}"))?;
            let sidecar_header =
                krasis::expert_sidecar::parse_header_for_file(&encoded_header, sidecar_file_bytes)?;
            let mapping_granularity =
                krasis::expert_sidecar::system_mapping_granularity()?;
            let payload_prefix = sidecar_header.payload_offset % mapping_granularity;
            let transfer_bytes = layout.expert_bytes.min(
                mmap.len()
                    .checked_sub(payload_prefix)
                    .ok_or_else(|| "sidecar payload prefix exceeds mapping".to_string())?,
            );
            if transfer_bytes != layout.expert_bytes {
                return Err(format!(
                    "sidecar mapping exposes {transfer_bytes} payload bytes, expected one full {}-byte expert",
                    layout.expert_bytes,
                ));
            }
            cuda(
                unsafe {
                    sys::lib().cuMemHostRegister_v2(
                        mmap.as_ptr() as *mut std::ffi::c_void,
                        mmap.len(),
                        0,
                    )
                },
                "real sidecar private-mapping cuMemHostRegister",
            )?;
            let file_source = unsafe { mmap.as_ptr().add(payload_prefix) };
            let (file_backed_time, file_exact) = measure_registered_h2d(
                &device,
                file_source,
                transfer_bytes,
                args.warmup,
                args.samples,
            )?;
            cuda(
                unsafe { sys::lib().cuMemHostUnregister(mmap.as_ptr() as *mut std::ffi::c_void) },
                "real sidecar private-mapping cuMemHostUnregister",
            )?;

            let mut anonymous = MmapOptions::new()
                .len(mmap.len())
                .map_anon()
                .map_err(|error| format!("failed to allocate anonymous sidecar probe: {error}"))?;
            let hugepage_result = unsafe {
                libc::madvise(
                    anonymous.as_mut_ptr().cast(),
                    anonymous.len(),
                    libc::MADV_HUGEPAGE,
                )
            };
            anonymous[payload_prefix..payload_prefix + transfer_bytes]
                .copy_from_slice(&mmap[payload_prefix..payload_prefix + transfer_bytes]);
            cuda(
                unsafe {
                    sys::lib().cuMemHostRegister_v2(
                        anonymous.as_mut_ptr().cast(),
                        anonymous.len(),
                        0,
                    )
                },
                "anonymous sidecar-probe cuMemHostRegister",
            )?;
            let anonymous_source = unsafe { anonymous.as_ptr().add(payload_prefix) };
            let (anonymous_time, anonymous_exact) = measure_registered_h2d(
                &device,
                anonymous_source,
                transfer_bytes,
                args.warmup,
                args.samples,
            )?;
            cuda(
                unsafe { sys::lib().cuMemHostUnregister(anonymous.as_mut_ptr().cast()) },
                "anonymous sidecar-probe cuMemHostUnregister",
            )?;
            let report = H2dBackingReport {
                transfer_bytes,
                file_backed_gbps_at_p95_time: transfer_bytes as f64
                    / (file_backed_time.p95_us / 1e6)
                    / 1e9,
                anonymous_gbps_at_p95_time: transfer_bytes as f64
                    / (anonymous_time.p95_us / 1e6)
                    / 1e9,
                file_backed_time,
                anonymous_time,
                anonymous_hugepage_advice_succeeded: hugepage_result == 0,
                payload_bit_exact: file_exact && anonymous_exact,
            };
            if !report.payload_bit_exact {
                return Err("sidecar H2D backing probe changed payload bytes".to_string());
            }
            Some((path.display().to_string(), mmap.len(), true, report))
        } else {
            None
        };
        device
            .load_ptx(Ptx::from_src(PTX), "expert_codec", &["decode_expert_rans"])
            .map_err(driver_error)?;
        let function = device
            .get_func("expert_codec", "decode_expert_rans")
            .ok_or_else(|| "decode_expert_rans kernel was not loaded".to_string())?;
        let d_decode_symbols = device
            .htod_sync_copy(&tables.gpu_decode_symbols())
            .map_err(driver_error)?;
        let d_frequencies = device
            .htod_sync_copy(&tables.gpu_frequencies())
            .map_err(driver_error)?;
        let d_starts = device
            .htod_sync_copy(&tables.gpu_starts())
            .map_err(driver_error)?;
        let decode_symbols_ptr = *d_decode_symbols.device_ptr();
        let frequencies_ptr = *d_frequencies.device_ptr();
        let starts_ptr = *d_starts.device_ptr();

        let mut candidates = Vec::new();
        for &lane_bytes in &args.lane_bytes {
            let encoded: Vec<EncodedExpert> = component_sets
                .iter()
                .map(|component_set| encode_expert(component_set, &tables, lane_bytes))
                .collect::<ProbeResult<_>>()?;
            let every_cpu_round_trip_bit_exact = encoded
                .iter()
                .zip(sources.iter())
                .map(|(encoded, source)| {
                    encoded
                        .decode_cpu(&tables)
                        .map(|decoded| decoded == **source)
                })
                .collect::<ProbeResult<Vec<_>>>()?
                .into_iter()
                .all(|exact| exact);
            if !every_cpu_round_trip_bit_exact {
                return Err(format!("lane_bytes={lane_bytes} failed CPU byte identity"));
            }

            let mut elapsed_us = Vec::with_capacity(args.samples * encoded.len());
            let mut every_gpu_round_trip_bit_exact = true;
            for (encoded_expert, source) in encoded.iter().zip(sources.iter()) {
                let d_blob = device
                    .htod_sync_copy(&encoded_expert.blob)
                    .map_err(driver_error)?;
                let d_output = unsafe {
                    device
                        .alloc::<u8>(encoded_expert.original_bytes)
                        .map_err(driver_error)?
                };
                let blob_ptr = *d_blob.device_ptr();
                let output_ptr = *d_output.device_ptr();
                let chunk_plan = plan_expert_chunks(
                    &encoded_expert.blob,
                    [
                        layout.w13_packed,
                        layout.w13_scales,
                        layout.w2_packed,
                        layout.w2_scales,
                    ],
                )?;
                let full_task_count = u32::try_from(encoded_expert.task_count)
                    .map_err(|_| "task count exceeds CUDA grid".to_string())?;
                let full_launch_config = LaunchConfig {
                    grid_dim: (full_task_count, 1, 1),
                    block_dim: (CODEC_LANES as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let start = CudaEvent::new()?;
                let end = CudaEvent::new()?;
                for sample in 0..args.warmup + args.samples {
                    cuda(
                        unsafe { sys::lib().cuEventRecord(start.0, *device.cu_stream()) },
                        "cuEventRecord(start)",
                    )?;
                    unsafe {
                        function
                            .clone()
                            .launch(
                                full_launch_config,
                                (
                                    blob_ptr,
                                    output_ptr,
                                    decode_symbols_ptr,
                                    frequencies_ptr,
                                    starts_ptr,
                                    0_u32,
                                    full_task_count,
                                ),
                            )
                            .map_err(driver_error)?;
                    }
                    cuda(
                        unsafe { sys::lib().cuEventRecord(end.0, *device.cu_stream()) },
                        "cuEventRecord(end)",
                    )?;
                    cuda(
                        unsafe { sys::lib().cuEventSynchronize(end.0) },
                        "cuEventSynchronize",
                    )?;
                    let mut elapsed_ms = 0_f32;
                    cuda(
                        unsafe { sys::lib().cuEventElapsedTime(&mut elapsed_ms, start.0, end.0) },
                        "cuEventElapsedTime",
                    )?;
                    if sample >= args.warmup {
                        elapsed_us.push(f64::from(elapsed_ms) * 1_000.0);
                    }
                }
                let restored = device.dtoh_sync_copy(&d_output).map_err(driver_error)?;
                every_gpu_round_trip_bit_exact &= restored == **source;

                // Exercise the production partial-task ABI independently of
                // the throughput measurement. Start from zeroed output so a
                // missing range cannot inherit bytes from the full launch.
                let d_partial_output = device
                    .htod_sync_copy(&vec![0_u8; encoded_expert.original_bytes])
                    .map_err(driver_error)?;
                let partial_output_ptr = *d_partial_output.device_ptr();
                for chunk in chunk_plan {
                    let task_start = u32::try_from(chunk.task_start)
                        .map_err(|_| "task start exceeds u32".to_string())?;
                    let task_count = u32::try_from(chunk.task_count)
                        .map_err(|_| "task count exceeds CUDA grid".to_string())?;
                    unsafe {
                        function
                            .clone()
                            .launch(
                                LaunchConfig {
                                    grid_dim: (task_count, 1, 1),
                                    block_dim: (CODEC_LANES as u32, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (
                                    blob_ptr,
                                    partial_output_ptr,
                                    decode_symbols_ptr,
                                    frequencies_ptr,
                                    starts_ptr,
                                    task_start,
                                    task_count,
                                ),
                            )
                            .map_err(driver_error)?;
                    }
                }
                let partial_restored = device
                    .dtoh_sync_copy(&d_partial_output)
                    .map_err(driver_error)?;
                every_gpu_round_trip_bit_exact &= partial_restored == **source;
            }
            if !every_gpu_round_trip_bit_exact {
                return Err(format!("lane_bytes={lane_bytes} failed GPU byte identity"));
            }

            let timing = distribution(&elapsed_us);
            let original_bytes = encoded
                .iter()
                .map(|expert| expert.original_bytes as u64)
                .sum::<u64>();
            let encoded_bytes = encoded
                .iter()
                .map(|expert| expert.blob.len() as u64)
                .sum::<u64>();
            let bytes_per_sample = original_bytes as f64 / encoded.len() as f64;
            let encoded_per_sample = encoded_bytes as f64 / encoded.len() as f64;
            let p50_decoder_gbps = bytes_per_sample / (timing.p50_us / 1e6) / 1e9;
            let p95_decoder_gbps = bytes_per_sample / (timing.p95_us / 1e6) / 1e9;
            let compressed_copy_us = encoded_per_sample / (args.pcie_gbps * 1e9) * 1e6;
            let raw_copy_us = bytes_per_sample / (args.pcie_gbps * 1e9) * 1e6;
            let pipeline_us = compressed_copy_us.max(timing.p95_us);
            candidates.push(CandidateReport {
                lane_bytes,
                original_bytes,
                encoded_bytes,
                encoded_ratio: encoded_bytes as f64 / original_bytes as f64,
                saving_pct: (1.0 - encoded_bytes as f64 / original_bytes as f64) * 100.0,
                task_count: encoded.iter().map(|expert| expert.task_count as u64).sum(),
                kernel_time: timing,
                p50_decoder_gbps,
                p95_time_decoder_gbps: p95_decoder_gbps,
                predicted_p95_pipeline_us: pipeline_us,
                predicted_raw_copy_us: raw_copy_us,
                predicted_pipeline_speedup: raw_copy_us / pipeline_us,
                every_cpu_round_trip_bit_exact,
                every_gpu_round_trip_bit_exact,
            });
        }

        let selected = candidates
            .iter()
            .filter(|candidate| {
                candidate.p95_time_decoder_gbps >= args.minimum_decoder_gbps
            })
            .min_by(|left, right| {
                left.predicted_p95_pipeline_us
                    .total_cmp(&right.predicted_p95_pipeline_us)
                    .then_with(|| left.lane_bytes.cmp(&right.lane_bytes))
            })
            .ok_or_else(|| {
                let fastest = candidates
                    .iter()
                    .map(|candidate| candidate.p95_time_decoder_gbps)
                    .fold(0.0_f64, f64::max);
                format!(
                    "no codec candidate satisfies the p95-time decoder gate: fastest={fastest:.3} GB/s required={:.3} GB/s",
                    args.minimum_decoder_gbps,
                )
            })?;
        let gate_passed = true;
        let report = Report {
            schema_version: 1,
            source_cache: args.cache.display().to_string(),
            source_cache_bytes,
            model_config: args.model_config.display().to_string(),
            gpu_ordinal: args.gpu,
            gpu_name: device.name().map_err(driver_error)?,
            expert_ordinals: args.expert_ordinals,
            expert_bytes: layout.expert_bytes,
            warmup_samples_per_expert: args.warmup,
            measured_samples_per_expert: args.samples,
            pcie_gbps: args.pcie_gbps,
            minimum_decoder_gbps: args.minimum_decoder_gbps,
            registration_sidecar: sidecar_registration
                .as_ref()
                .map(|(path, _, _, _)| path.clone()),
            sidecar_registration_bytes: sidecar_registration
                .as_ref()
                .map(|(_, bytes, _, _)| *bytes),
            sidecar_registration_passed: sidecar_registration
                .as_ref()
                .map(|(_, _, passed, _)| *passed),
            sidecar_h2d_backing: sidecar_registration.map(|(_, _, _, report)| report),
            selected_lane_bytes: selected.lane_bytes,
            selected_p95_time_decoder_gbps: selected.p95_time_decoder_gbps,
            gate_passed,
            candidates,
        };
        let json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
        println!("{json}");
        if let Some(path) = &args.output {
            std::fs::write(path, format!("{json}\n"))
                .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        }
        if !gate_passed {
            return Err(format!(
                "GPU expert codec gate failed: selected p95-time throughput {:.3} GB/s is below {:.3} GB/s",
                report.selected_p95_time_decoder_gbps, report.minimum_decoder_gbps
            ));
        }
        Ok(())
    }

    fn components<'a>(source: &'a [u8], layout: &CacheLayout) -> [ExpertComponent<'a>; 4] {
        let w13_scales = layout.w13_packed;
        let w2_packed = w13_scales + layout.w13_scales;
        let w2_scales = w2_packed + layout.w2_packed;
        [
            ExpertComponent {
                bytes: &source[..w13_scales],
                kind: ComponentKind::PackedNibbles,
            },
            ExpertComponent {
                bytes: &source[w13_scales..w2_packed],
                kind: ComponentKind::Bf16Scales,
            },
            ExpertComponent {
                bytes: &source[w2_packed..w2_scales],
                kind: ComponentKind::PackedNibbles,
            },
            ExpertComponent {
                bytes: &source[w2_scales..],
                kind: ComponentKind::Bf16Scales,
            },
        ]
    }

    fn read_layout(bytes: &[u8], gated: bool) -> ProbeResult<CacheLayout> {
        if bytes.len() < CACHE_HEADER_BYTES || &bytes[0..4] != b"KRAS" {
            return Err("invalid Marlin cache header".to_string());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != MARLIN_CACHE_VERSION {
            return Err(format!(
                "Marlin cache version {version}, expected {MARLIN_CACHE_VERSION}"
            ));
        }
        let hidden_size = read_header_usize(bytes, 8)?;
        let intermediate_size = read_header_usize(bytes, 16)?;
        let experts_per_layer = read_header_usize(bytes, 24)?;
        let moe_layers = read_header_usize(bytes, 32)?;
        let group_size = read_header_usize(bytes, 40)?;
        if hidden_size % 8 != 0 || hidden_size % group_size != 0 || intermediate_size % 8 != 0 {
            return Err("cache dimensions do not satisfy Marlin INT4 packing".to_string());
        }
        let padded_hidden = if hidden_size == intermediate_size && hidden_size % 256 != 0 {
            hidden_size + 64
        } else {
            hidden_size
        };
        let w13_width = intermediate_size * if gated { 2 } else { 1 };
        let w13_packed = (hidden_size / 8) * w13_width * 4;
        let w13_scales = (hidden_size / group_size) * w13_width * 2;
        let w2_packed = (intermediate_size / 8) * padded_hidden * 4;
        let w2_scales = intermediate_size.div_ceil(group_size) * padded_hidden * 2;
        let expert_bytes = w13_packed + w13_scales + w2_packed + w2_scales;
        Ok(CacheLayout {
            hidden_size,
            intermediate_size,
            experts_per_layer,
            moe_layers,
            group_size,
            w13_packed,
            w13_scales,
            w2_packed,
            w2_scales,
            expert_bytes,
        })
    }

    fn expert_bytes<'a>(
        cache: &'a [u8],
        layout: &CacheLayout,
        ordinal: usize,
    ) -> ProbeResult<&'a [u8]> {
        let start = CACHE_HEADER_BYTES
            .checked_add(ordinal.saturating_mul(layout.expert_bytes))
            .ok_or_else(|| "expert cache offset overflow".to_string())?;
        cache
            .get(start..start + layout.expert_bytes)
            .ok_or_else(|| format!("cache has no complete expert at ordinal {ordinal}"))
    }

    fn read_header_usize(bytes: &[u8], offset: usize) -> ProbeResult<usize> {
        usize::try_from(u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap(),
        ))
        .map_err(|_| "cache dimension does not fit usize".to_string())
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut raw = std::env::args().skip(1);
        let cache = required_path(raw.next(), "cache path")?;
        let model_config = required_path(raw.next(), "model config path")?;
        let mut gpu = None;
        let mut expert_ordinals = None;
        let mut lane_bytes = None;
        let mut warmup = 10;
        let mut samples = 50;
        let mut pcie_gbps = None;
        let mut minimum_decoder_gbps = None;
        let mut registration_sidecar = None;
        let mut output = None;
        while let Some(flag) = raw.next() {
            match flag.as_str() {
                "--gpu" => gpu = Some(parse_value(raw.next(), &flag)?),
                "--expert-ordinals" => expert_ordinals = Some(parse_csv(raw.next(), &flag)?),
                "--lane-bytes" => lane_bytes = Some(parse_csv(raw.next(), &flag)?),
                "--warmup" => warmup = parse_value(raw.next(), &flag)?,
                "--samples" => samples = parse_value(raw.next(), &flag)?,
                "--pcie-gbps" => pcie_gbps = Some(parse_value(raw.next(), &flag)?),
                "--minimum-decoder-gbps" => {
                    minimum_decoder_gbps = Some(parse_value(raw.next(), &flag)?)
                }
                "--registration-sidecar" => {
                    registration_sidecar = Some(required_path(raw.next(), &flag)?)
                }
                "--output" => output = Some(required_path(raw.next(), &flag)?),
                "--help" | "-h" => return Err(usage()),
                _ => return Err(format!("unknown argument {flag}\n{}", usage())),
            }
        }
        let args = Args {
            cache,
            model_config,
            gpu: gpu.ok_or_else(usage)?,
            expert_ordinals: expert_ordinals.ok_or_else(usage)?,
            lane_bytes: lane_bytes.ok_or_else(usage)?,
            warmup,
            samples,
            pcie_gbps: pcie_gbps.ok_or_else(usage)?,
            minimum_decoder_gbps: minimum_decoder_gbps.ok_or_else(usage)?,
            registration_sidecar,
            output,
        };
        if args.expert_ordinals.is_empty()
            || args.lane_bytes.is_empty()
            || args.lane_bytes.iter().any(|&bytes| bytes == 0)
            || args.samples == 0
            || args.pcie_gbps <= 0.0
            || args.minimum_decoder_gbps <= 0.0
        {
            return Err(format!(
                "all codec gate inputs must be positive and explicit\n{}",
                usage()
            ));
        }
        Ok(args)
    }

    fn parse_csv<T: std::str::FromStr>(raw: Option<String>, name: &str) -> ProbeResult<Vec<T>>
    where
        T::Err: std::fmt::Display,
    {
        raw.ok_or_else(|| format!("{name} requires a comma-separated value"))?
            .split(',')
            .map(|part| {
                part.parse::<T>()
                    .map_err(|error| format!("invalid {name} entry {part}: {error}"))
            })
            .collect()
    }

    fn parse_value<T: std::str::FromStr>(raw: Option<String>, name: &str) -> ProbeResult<T>
    where
        T::Err: std::fmt::Display,
    {
        raw.ok_or_else(|| format!("{name} requires a value"))?
            .parse::<T>()
            .map_err(|error| format!("invalid {name}: {error}"))
    }

    fn required_path(raw: Option<String>, name: &str) -> ProbeResult<PathBuf> {
        Ok(PathBuf::from(
            raw.ok_or_else(|| format!("missing {name}\n{}", usage()))?,
        ))
    }

    fn usage() -> String {
        "Usage: ./dev expert-compression-gpu <cache.bin> <config.json> --gpu N --expert-ordinals N,N,... --lane-bytes N,N,... --pcie-gbps F --minimum-decoder-gbps F [--registration-sidecar PATH] [--warmup N] [--samples N] [--output PATH]".to_string()
    }

    fn distribution(values: &[f64]) -> Distribution {
        let mut ordered = values.to_vec();
        ordered.sort_by(f64::total_cmp);
        Distribution {
            min_us: ordered[0],
            p05_us: percentile(&ordered, 0.05),
            p50_us: percentile(&ordered, 0.50),
            p95_us: percentile(&ordered, 0.95),
            p99_us: percentile(&ordered, 0.99),
            max_us: *ordered.last().unwrap(),
            mean_us: ordered.iter().sum::<f64>() / ordered.len() as f64,
        }
    }

    fn measure_registered_h2d(
        device: &Arc<CudaDevice>,
        source: *const u8,
        bytes: usize,
        warmup: usize,
        samples: usize,
    ) -> ProbeResult<(Distribution, bool)> {
        let destination = unsafe { device.alloc::<u8>(bytes).map_err(driver_error)? };
        let destination_ptr = *destination.device_ptr();
        let start = CudaEvent::new()?;
        let end = CudaEvent::new()?;
        let stream = *device.cu_stream();
        let mut elapsed_us = Vec::with_capacity(samples);
        for sample in 0..warmup.saturating_add(samples) {
            cuda(
                unsafe { sys::lib().cuEventRecord(start.0, stream) },
                "sidecar H2D cuEventRecord(start)",
            )?;
            cuda(
                unsafe {
                    sys::lib().cuMemcpyHtoDAsync_v2(destination_ptr, source.cast(), bytes, stream)
                },
                "sidecar H2D cuMemcpyHtoDAsync",
            )?;
            cuda(
                unsafe { sys::lib().cuEventRecord(end.0, stream) },
                "sidecar H2D cuEventRecord(end)",
            )?;
            cuda(
                unsafe { sys::lib().cuEventSynchronize(end.0) },
                "sidecar H2D cuEventSynchronize",
            )?;
            let mut elapsed_ms = 0_f32;
            cuda(
                unsafe { sys::lib().cuEventElapsedTime(&mut elapsed_ms, start.0, end.0) },
                "sidecar H2D cuEventElapsedTime",
            )?;
            if sample >= warmup {
                elapsed_us.push(f64::from(elapsed_ms) * 1_000.0);
            }
        }
        let mut restored = vec![0_u8; bytes];
        cuda(
            unsafe {
                sys::lib().cuMemcpyDtoH_v2(restored.as_mut_ptr().cast(), destination_ptr, bytes)
            },
            "sidecar H2D verification download",
        )?;
        let expected = unsafe { std::slice::from_raw_parts(source, bytes) };
        Ok((distribution(&elapsed_us), restored == expected))
    }

    fn percentile(ordered: &[f64], quantile: f64) -> f64 {
        let rank = quantile * (ordered.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        ordered[lower] * (1.0 - rank.fract()) + ordered[upper] * rank.fract()
    }

    fn cuda(result: sys::CUresult, operation: &str) -> ProbeResult<()> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation} failed: {result:?}"))
        }
    }

    fn driver_error(error: impl std::fmt::Debug) -> String {
        format!("CUDA driver error: {error:?}")
    }
}

#[cfg(has_expert_codec_kernels)]
fn main() {
    if let Err(error) = probe::main() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}
