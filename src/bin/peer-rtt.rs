//! Standalone Phase-0 feasibility probe for host-bounced peer expert serving.
//!
//! This deliberately bypasses Python and Torch.  A persistent CUDA kernel on
//! the peer polls a portable mapped-host mailbox.  The primary GPU publishes a
//! 12 KiB D2H request and consumes the peer's 12 KiB H2D response on one CUDA
//! stream, so every sample covers the actual request/response critical path.

#[cfg(not(has_peer_rtt_kernels))]
compile_error!("peer-rtt requires nvcc-built peer RTT kernels");

#[cfg(has_peer_rtt_kernels)]
mod probe {
    use cudarc::driver::sys;
    use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use serde::Serialize;
    use std::ffi::c_void;
    use std::path::PathBuf;
    use std::ptr;
    use std::time::{Duration, Instant};

    const PEER_RTT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/peer_rtt_kernels.ptx"));
    const CONTROL_BYTES: usize = 4096;

    type ProbeResult<T> = Result<T, String>;

    #[derive(Debug)]
    struct Args {
        primary: usize,
        peer: usize,
        message_bytes: usize,
        warmup: usize,
        samples: usize,
        gate_us: f64,
        timeout_ms: u64,
        output: Option<PathBuf>,
    }

    #[derive(Serialize)]
    struct DeviceReport {
        ordinal: usize,
        name: String,
        free_bytes: usize,
        total_bytes: usize,
        can_map_host_memory: bool,
        unified_addressing: bool,
        stream_memops_v1: bool,
    }

    #[derive(Serialize)]
    struct Distribution {
        min_us: f64,
        p01_us: f64,
        p05_us: f64,
        p50_us: f64,
        p90_us: f64,
        p95_us: f64,
        p99_us: f64,
        max_us: f64,
        mean_us: f64,
        stddev_us: f64,
    }

    #[derive(Serialize)]
    struct Report {
        schema_version: u32,
        transport: &'static str,
        message_bytes_each_direction: usize,
        round_trip_payload_bytes: usize,
        warmup_samples: usize,
        measured_samples: usize,
        timeout_ms: u64,
        primary: DeviceReport,
        peer: DeviceReport,
        primary_can_access_peer: bool,
        peer_can_access_primary: bool,
        distribution: Distribution,
        gate_threshold_us: f64,
        gate_passed: bool,
        payload_bit_exact: bool,
    }

    struct MappedMailbox {
        host: *mut u8,
        bytes: usize,
    }

    impl MappedMailbox {
        fn allocate(bytes: usize) -> ProbeResult<Self> {
            let mut host: *mut c_void = ptr::null_mut();
            // PORTABLE | DEVICEMAP.  The allocation is visible from both
            // primary contexts but remains canonical host memory.
            let result = unsafe { sys::lib().cuMemHostAlloc(&mut host, bytes, 1 | 2) };
            cuda(result, format!("cuMemHostAlloc({bytes})"))?;
            unsafe { ptr::write_bytes(host, 0, bytes) };
            Ok(Self {
                host: host.cast(),
                bytes,
            })
        }

        fn device_pointer(&self, device: &CudaDevice) -> ProbeResult<u64> {
            device.bind_to_thread().map_err(driver_error)?;
            let mut mapped = 0_u64;
            let result = unsafe {
                sys::lib().cuMemHostGetDevicePointer_v2(&mut mapped, self.host.cast::<c_void>(), 0)
            };
            cuda(result, "cuMemHostGetDevicePointer_v2")?;
            Ok(mapped)
        }
    }

    impl Drop for MappedMailbox {
        fn drop(&mut self) {
            if !self.host.is_null() {
                let result = unsafe { sys::lib().cuMemFreeHost(self.host.cast::<c_void>()) };
                assert_eq!(
                    result,
                    sys::CUresult::CUDA_SUCCESS,
                    "cuMemFreeHost({} bytes) failed: {result:?}",
                    self.bytes
                );
            }
        }
    }

    pub fn main() -> ProbeResult<()> {
        let args = parse_args()?;
        if args.primary == args.peer {
            return Err("primary and peer GPU ordinals must differ".to_string());
        }
        if args.message_bytes == 0 || args.message_bytes % std::mem::size_of::<u32>() != 0 {
            return Err("--message-bytes must be a non-zero multiple of four".to_string());
        }
        let iterations = args
            .warmup
            .checked_add(args.samples)
            .ok_or_else(|| "sample count overflow".to_string())?;
        let iterations_u32 = u32::try_from(iterations)
            .map_err(|_| "warmup + samples exceeds the kernel sequence range".to_string())?;
        if args.samples == 0 {
            return Err("--samples must be non-zero".to_string());
        }

        let primary = CudaDevice::new_with_stream(args.primary).map_err(driver_error)?;
        let peer = CudaDevice::new_with_stream(args.peer).map_err(driver_error)?;
        let primary_report = device_report(&primary)?;
        let peer_report = device_report(&peer)?;
        for (role, report) in [("primary", &primary_report), ("peer", &peer_report)] {
            if !report.can_map_host_memory || !report.unified_addressing {
                return Err(format!(
                    "{role} GPU does not support the required mapped-host-memory contract"
                ));
            }
        }
        let primary_can_access_peer = can_access_peer(&primary, &peer)?;
        let peer_can_access_primary = can_access_peer(&peer, &primary)?;

        let mailbox_bytes = CONTROL_BYTES
            .checked_add(args.message_bytes.saturating_mul(2))
            .ok_or_else(|| "mailbox size overflow".to_string())?;
        let mailbox = MappedMailbox::allocate(mailbox_bytes)?;
        let primary_mapping = mailbox.device_pointer(&primary)?;
        let peer_mapping = mailbox.device_pointer(&peer)?;

        let host_request = unsafe { mailbox.host.add(CONTROL_BYTES) };
        let host_response = unsafe { host_request.add(args.message_bytes) };
        let primary_control = primary_mapping;
        let peer_control = peer_mapping;
        let peer_request = peer_mapping + CONTROL_BYTES as u64;
        let peer_response = peer_request + args.message_bytes as u64;

        let source: Vec<u8> = (0..args.message_bytes)
            .map(|index| ((index.wrapping_mul(131) ^ (index >> 3)) & 0xff) as u8)
            .collect();
        primary.bind_to_thread().map_err(driver_error)?;
        let device_source = primary.htod_sync_copy(&source).map_err(driver_error)?;
        let device_result = unsafe {
            primary
                .alloc::<u8>(args.message_bytes)
                .map_err(driver_error)?
        };
        let source_ptr = *device_source.device_ptr();
        let result_ptr = *device_result.device_ptr();

        primary.bind_to_thread().map_err(driver_error)?;
        primary
            .load_ptx(
                Ptx::from_src(PEER_RTT_PTX),
                "peer_rtt_primary",
                &["primary_mailbox_publish_and_wait"],
            )
            .map_err(driver_error)?;
        let primary_publish_wait = primary
            .get_func("peer_rtt_primary", "primary_mailbox_publish_and_wait")
            .ok_or_else(|| "primary_mailbox_publish_and_wait kernel was not loaded".to_string())?;

        peer.bind_to_thread().map_err(driver_error)?;
        peer.load_ptx(
            Ptx::from_src(PEER_RTT_PTX),
            "peer_rtt_peer",
            &["peer_mailbox_round_trip"],
        )
        .map_err(driver_error)?;
        let function = peer
            .get_func("peer_rtt_peer", "peer_mailbox_round_trip")
            .ok_or_else(|| "peer_mailbox_round_trip kernel was not loaded".to_string())?;
        let message_words = u32::try_from(args.message_bytes / std::mem::size_of::<u32>())
            .map_err(|_| "message is too large for the probe kernel".to_string())?;
        unsafe {
            function
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        peer_control,
                        peer_request,
                        peer_response,
                        message_words,
                        iterations_u32,
                    ),
                )
                .map_err(driver_error)?;
        }

        let timeout = Duration::from_millis(args.timeout_ms);
        let mut samples_us = Vec::with_capacity(args.samples);
        for index in 0..iterations {
            let sequence = u32::try_from(index + 1).unwrap();
            primary.bind_to_thread().map_err(driver_error)?;
            let start = Instant::now();
            cuda(
                unsafe {
                    sys::lib().cuMemcpyDtoHAsync_v2(
                        host_request.cast::<c_void>(),
                        source_ptr,
                        args.message_bytes,
                        *primary.cu_stream(),
                    )
                },
                "primary request D2H",
            )?;
            unsafe {
                primary_publish_wait
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (primary_control, sequence),
                    )
                    .map_err(driver_error)?;
            }
            cuda(
                unsafe {
                    sys::lib().cuMemcpyHtoDAsync_v2(
                        result_ptr,
                        host_response.cast::<c_void>(),
                        args.message_bytes,
                        *primary.cu_stream(),
                    )
                },
                "primary response H2D",
            )?;
            loop {
                let status = unsafe { sys::lib().cuStreamQuery(*primary.cu_stream()) };
                if status == sys::CUresult::CUDA_SUCCESS {
                    break;
                }
                if status != sys::CUresult::CUDA_ERROR_NOT_READY {
                    return Err(format!("primary stream query failed: {status:?}"));
                }
                if start.elapsed() > timeout {
                    return Err(format!(
                        "peer response timed out at sequence {sequence} after {} ms",
                        args.timeout_ms
                    ));
                }
                std::hint::spin_loop();
            }
            if index >= args.warmup {
                samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
            }
        }

        peer.bind_to_thread().map_err(driver_error)?;
        peer.synchronize().map_err(driver_error)?;
        primary.bind_to_thread().map_err(driver_error)?;
        let returned = primary
            .dtoh_sync_copy(&device_result)
            .map_err(driver_error)?;
        let payload_bit_exact = returned == source;
        if !payload_bit_exact {
            return Err("peer response payload was not bit-exact".to_string());
        }

        let distribution = distribution(&samples_us);
        let report = Report {
            schema_version: 1,
            transport: "portable_mapped_pinned_host_mailbox_gpu_ordered",
            message_bytes_each_direction: args.message_bytes,
            round_trip_payload_bytes: args.message_bytes * 2,
            warmup_samples: args.warmup,
            measured_samples: args.samples,
            timeout_ms: args.timeout_ms,
            primary: primary_report,
            peer: peer_report,
            primary_can_access_peer,
            peer_can_access_primary,
            gate_threshold_us: args.gate_us,
            gate_passed: distribution.p95_us <= args.gate_us,
            payload_bit_exact,
            distribution,
        };
        let json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
        println!("{json}");
        if let Some(path) = &args.output {
            std::fs::write(path, format!("{json}\n"))
                .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        }
        if !report.gate_passed {
            return Err(format!(
                "peer RTT gate failed: p95 {:.3} us exceeds {:.3} us",
                report.distribution.p95_us, report.gate_threshold_us
            ));
        }
        Ok(())
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut raw = std::env::args().skip(1);
        let primary = parse_required::<usize>(raw.next(), "primary GPU ordinal")?;
        let peer = parse_required::<usize>(raw.next(), "peer GPU ordinal")?;
        let mut args = Args {
            primary,
            peer,
            message_bytes: 12 * 1024,
            warmup: 1_000,
            samples: 10_000,
            gate_us: 30.0,
            timeout_ms: 1_000,
            output: None,
        };
        while let Some(flag) = raw.next() {
            match flag.as_str() {
                "--message-bytes" => {
                    args.message_bytes = parse_required(raw.next(), "--message-bytes")?
                }
                "--warmup" => args.warmup = parse_required(raw.next(), "--warmup")?,
                "--samples" => args.samples = parse_required(raw.next(), "--samples")?,
                "--gate-us" => args.gate_us = parse_required(raw.next(), "--gate-us")?,
                "--timeout-ms" => args.timeout_ms = parse_required(raw.next(), "--timeout-ms")?,
                "--output" => {
                    args.output = Some(PathBuf::from(
                        raw.next()
                            .ok_or_else(|| "--output requires a path".to_string())?,
                    ))
                }
                "--help" | "-h" => return Err(usage()),
                _ => return Err(format!("unknown argument {flag}\n{}", usage())),
            }
        }
        Ok(args)
    }

    fn parse_required<T: std::str::FromStr>(raw: Option<String>, name: &str) -> ProbeResult<T>
    where
        T::Err: std::fmt::Display,
    {
        raw.ok_or_else(|| format!("missing {name}\n{}", usage()))?
            .parse::<T>()
            .map_err(|error| format!("invalid {name}: {error}"))
    }

    fn usage() -> String {
        "Usage: ./dev peer-rtt <primary-gpu> <peer-gpu> [--message-bytes N] [--warmup N] [--samples N] [--gate-us F] [--timeout-ms N] [--output PATH]".to_string()
    }

    fn device_report(device: &CudaDevice) -> ProbeResult<DeviceReport> {
        device.bind_to_thread().map_err(driver_error)?;
        let (free_bytes, total_bytes) =
            cudarc::driver::result::mem_get_info().map_err(driver_error)?;
        Ok(DeviceReport {
            ordinal: device.ordinal(),
            name: device.name().map_err(driver_error)?,
            free_bytes,
            total_bytes,
            can_map_host_memory: attribute_bool(
                device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
            )?,
            unified_addressing: attribute_bool(
                device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            )?,
            stream_memops_v1: attribute_bool(
                device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
            )?,
        })
    }

    fn attribute_bool(
        device: &CudaDevice,
        attribute: sys::CUdevice_attribute,
    ) -> ProbeResult<bool> {
        Ok(device.attribute(attribute).map_err(driver_error)? != 0)
    }

    fn can_access_peer(source: &CudaDevice, destination: &CudaDevice) -> ProbeResult<bool> {
        let mut accessible = 0;
        cuda(
            unsafe {
                sys::lib().cuDeviceCanAccessPeer(
                    &mut accessible,
                    *source.cu_device(),
                    *destination.cu_device(),
                )
            },
            "cuDeviceCanAccessPeer",
        )?;
        Ok(accessible != 0)
    }

    fn distribution(samples: &[f64]) -> Distribution {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let variance = sorted
            .iter()
            .map(|sample| {
                let difference = sample - mean;
                difference * difference
            })
            .sum::<f64>()
            / sorted.len() as f64;
        Distribution {
            min_us: sorted[0],
            p01_us: percentile(&sorted, 0.01),
            p05_us: percentile(&sorted, 0.05),
            p50_us: percentile(&sorted, 0.50),
            p90_us: percentile(&sorted, 0.90),
            p95_us: percentile(&sorted, 0.95),
            p99_us: percentile(&sorted, 0.99),
            max_us: *sorted.last().unwrap(),
            mean_us: mean,
            stddev_us: variance.sqrt(),
        }
    }

    fn percentile(sorted: &[f64], quantile: f64) -> f64 {
        let rank = quantile * (sorted.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        let fraction = rank - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }

    fn cuda(result: sys::CUresult, operation: impl Into<String>) -> ProbeResult<()> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{} failed: {result:?}", operation.into()))
        }
    }

    fn driver_error(error: impl std::fmt::Display) -> String {
        error.to_string()
    }
}

#[cfg(has_peer_rtt_kernels)]
fn main() {
    if let Err(error) = probe::main() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}
