use cudarc::driver::sys as cuda_sys;
use krasis::pcie_batch::CudaBatchCopy;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    gpu: usize,
    host_mode: HostMode,
    component_bytes: Vec<usize>,
    cold_experts: Vec<usize>,
    warmup: usize,
    samples: usize,
}

#[derive(Serialize)]
struct Row {
    cold_experts: usize,
    bytes_per_iteration: usize,
    individual_calls: usize,
    individual_gbps: f64,
    individual_submit_us: f64,
    component_major_calls: usize,
    component_major_gbps: f64,
    component_major_submit_us: f64,
    component_major_speedup: f64,
    batch_api_calls: usize,
    batch_entries: usize,
    batch_gbps: f64,
    batch_submit_us: f64,
    speedup: f64,
}

#[derive(Serialize)]
struct Report {
    gpu_ordinal: usize,
    host_mode: &'static str,
    component_bytes: Vec<usize>,
    warmup: usize,
    samples: usize,
    rows: Vec<Row>,
}

struct HostAllocation {
    pointer: *mut std::ffi::c_void,
    _storage: Option<Vec<u8>>,
    host_allocated: bool,
}

impl HostAllocation {
    fn new(bytes: usize, seed: u8, mode: HostMode) -> Result<Self, String> {
        if mode == HostMode::Alloc {
            let mut pointer = std::ptr::null_mut();
            let result = unsafe { cuda_sys::lib().cuMemHostAlloc(&mut pointer, bytes, 0) };
            if result != cuda_sys::CUresult::CUDA_SUCCESS || pointer.is_null() {
                return Err(format!("cuMemHostAlloc({bytes}) failed: {result:?}"));
            }
            unsafe { std::ptr::write_bytes(pointer, seed, bytes) };
            return Ok(Self {
                pointer,
                _storage: None,
                host_allocated: true,
            });
        }

        let mut storage = vec![seed; bytes];
        let pointer = storage.as_mut_ptr().cast();
        let flags = if mode == HostMode::RegisterMapped {
            0x01 | 0x02
        } else {
            0
        };
        let result = unsafe { cuda_sys::lib().cuMemHostRegister_v2(pointer, bytes, flags) };
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemHostRegister({bytes}, flags={flags:#x}) failed: {result:?}"
            ));
        }
        Ok(Self {
            pointer,
            _storage: Some(storage),
            host_allocated: false,
        })
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe {
                if self.host_allocated {
                    let _ = cuda_sys::lib().cuMemFreeHost(self.pointer);
                } else {
                    let _ = cuda_sys::lib().cuMemHostUnregister(self.pointer);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HostMode {
    Alloc,
    Register,
    RegisterMapped,
}

impl HostMode {
    fn label(self) -> &'static str {
        match self {
            Self::Alloc => "alloc",
            Self::Register => "register",
            Self::RegisterMapped => "register-mapped",
        }
    }
}

struct DeviceAllocation(cuda_sys::CUdeviceptr);

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe {
                let _ = cuda_sys::lib().cuMemFree_v2(self.0);
            }
        }
    }
}

struct Stream(cuda_sys::CUstream);

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = cuda_sys::lib().cuStreamSynchronize(self.0);
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.0);
            }
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let device_handle = cudarc::driver::CudaDevice::new(args.gpu)
        .map_err(|error| format!("open CUDA device {}: {error:?}", args.gpu))?;
    device_handle
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA device {}: {error:?}", args.gpu))?;
    let batch = CudaBatchCopy::load()?;
    let maximum_cold = *args
        .cold_experts
        .iter()
        .max()
        .ok_or_else(|| "at least one cold-expert count is required".to_string())?;
    let expert_bytes = args
        .component_bytes
        .iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| "expert byte count overflow".to_string())
        })?;
    let total_device_bytes = expert_bytes
        .checked_mul(maximum_cold)
        .ok_or_else(|| "device allocation size overflow".to_string())?;

    let mut host = Vec::with_capacity(args.component_bytes.len());
    for (index, bytes) in args.component_bytes.iter().enumerate() {
        let bank_bytes = bytes
            .checked_mul(maximum_cold)
            .ok_or_else(|| "host bank size overflow".to_string())?;
        host.push(HostAllocation::new(
            bank_bytes,
            (index + 1) as u8,
            args.host_mode,
        )?);
    }
    let mut device_pointer = 0;
    check(
        unsafe { cuda_sys::lib().cuMemAlloc_v2(&mut device_pointer, total_device_bytes) },
        "cuMemAlloc",
    )?;
    let device = DeviceAllocation(device_pointer);
    let mut stream = std::ptr::null_mut();
    check(
        unsafe {
            cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        },
        "cuStreamCreate",
    )?;
    let stream = Stream(stream);

    let mut rows = Vec::new();
    for &cold in &args.cold_experts {
        let (mut destinations, mut sources, mut sizes) =
            build_copy_plan(device.0, &host, &args.component_bytes, cold, expert_bytes)?;
        let (component_destinations, component_sources, component_sizes) =
            build_component_major_copy_plan(
                device.0,
                &host,
                &args.component_bytes,
                cold,
                expert_bytes,
            )?;
        for _ in 0..args.warmup {
            enqueue_individual(&destinations, &sources, &sizes, stream.0)?;
            check(
                unsafe { cuda_sys::lib().cuStreamSynchronize(stream.0) },
                "individual warmup sync",
            )?;
            enqueue_individual(
                &component_destinations,
                &component_sources,
                &component_sizes,
                stream.0,
            )?;
            check(
                unsafe { cuda_sys::lib().cuStreamSynchronize(stream.0) },
                "component-major warmup sync",
            )?;
            batch.enqueue(&mut destinations, &mut sources, &mut sizes, stream.0, false)?;
            check(
                unsafe { cuda_sys::lib().cuStreamSynchronize(stream.0) },
                "batch warmup sync",
            )?;
        }

        let (individual_seconds, individual_submit) = time_samples(
            args.samples,
            || enqueue_individual(&destinations, &sources, &sizes, stream.0),
            stream.0,
        )?;
        let (component_seconds, component_submit) = time_samples(
            args.samples,
            || {
                enqueue_individual(
                    &component_destinations,
                    &component_sources,
                    &component_sizes,
                    stream.0,
                )
            },
            stream.0,
        )?;
        let (batch_seconds, batch_submit) = time_samples(
            args.samples,
            || batch.enqueue(&mut destinations, &mut sources, &mut sizes, stream.0, false),
            stream.0,
        )?;

        verify_last_copy(device.0, &host, &args.component_bytes, cold, expert_bytes)?;
        let bytes_per_iteration = expert_bytes * cold;
        let individual_gbps =
            bytes_per_iteration as f64 * args.samples as f64 / individual_seconds / 1e9;
        let component_major_gbps =
            bytes_per_iteration as f64 * args.samples as f64 / component_seconds / 1e9;
        let batch_gbps = bytes_per_iteration as f64 * args.samples as f64 / batch_seconds / 1e9;
        rows.push(Row {
            cold_experts: cold,
            bytes_per_iteration,
            individual_calls: sizes.len(),
            individual_gbps,
            individual_submit_us: individual_submit / args.samples as f64 * 1e6,
            component_major_calls: component_sizes.len(),
            component_major_gbps,
            component_major_submit_us: component_submit / args.samples as f64 * 1e6,
            component_major_speedup: component_major_gbps / individual_gbps,
            batch_api_calls: 1,
            batch_entries: sizes.len(),
            batch_gbps,
            batch_submit_us: batch_submit / args.samples as f64 * 1e6,
            speedup: batch_gbps / individual_gbps,
        });
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            gpu_ordinal: args.gpu,
            host_mode: args.host_mode.label(),
            component_bytes: args.component_bytes,
            warmup: args.warmup,
            samples: args.samples,
            rows,
        })
        .map_err(|error| format!("serialize report: {error}"))?
    );
    Ok(())
}

fn time_samples<F>(
    samples: usize,
    mut enqueue: F,
    stream: cuda_sys::CUstream,
) -> Result<(f64, f64), String>
where
    F: FnMut() -> Result<(), String>,
{
    let total_start = Instant::now();
    let mut submit_seconds = 0.0;
    for _ in 0..samples {
        let submit_start = Instant::now();
        enqueue()?;
        submit_seconds += submit_start.elapsed().as_secs_f64();
        check(
            unsafe { cuda_sys::lib().cuStreamSynchronize(stream) },
            "sample sync",
        )?;
    }
    Ok((total_start.elapsed().as_secs_f64(), submit_seconds))
}

fn enqueue_individual(
    destinations: &[cuda_sys::CUdeviceptr],
    sources: &[cuda_sys::CUdeviceptr],
    sizes: &[usize],
    stream: cuda_sys::CUstream,
) -> Result<(), String> {
    for index in 0..sizes.len() {
        check(
            unsafe {
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    destinations[index],
                    sources[index] as usize as *const std::ffi::c_void,
                    sizes[index],
                    stream,
                )
            },
            "cuMemcpyHtoDAsync",
        )?;
    }
    Ok(())
}

fn build_copy_plan(
    device_base: cuda_sys::CUdeviceptr,
    host: &[HostAllocation],
    component_bytes: &[usize],
    cold: usize,
    expert_bytes: usize,
) -> Result<(Vec<u64>, Vec<u64>, Vec<usize>), String> {
    let mut destinations = Vec::with_capacity(cold * component_bytes.len());
    let mut sources = Vec::with_capacity(cold * component_bytes.len());
    let mut sizes = Vec::with_capacity(cold * component_bytes.len());
    for expert in 0..cold {
        let mut destination_offset = expert * expert_bytes;
        for (component, &bytes) in component_bytes.iter().enumerate() {
            destinations.push(device_base + destination_offset as u64);
            sources.push(host[component].pointer as usize as u64 + (expert * bytes) as u64);
            sizes.push(bytes);
            destination_offset += bytes;
        }
    }
    Ok((destinations, sources, sizes))
}

fn build_component_major_copy_plan(
    device_base: cuda_sys::CUdeviceptr,
    host: &[HostAllocation],
    component_bytes: &[usize],
    cold: usize,
    expert_bytes: usize,
) -> Result<(Vec<u64>, Vec<u64>, Vec<usize>), String> {
    let mut destinations = Vec::with_capacity(cold * component_bytes.len());
    let mut sources = Vec::with_capacity(cold * component_bytes.len());
    let mut sizes = Vec::with_capacity(cold * component_bytes.len());
    let mut component_destination_offsets = Vec::with_capacity(component_bytes.len());
    let mut offset = 0usize;
    for &bytes in component_bytes {
        component_destination_offsets.push(offset);
        offset = offset
            .checked_add(bytes)
            .ok_or_else(|| "component destination offset overflow".to_string())?;
    }
    if offset != expert_bytes {
        return Err(format!(
            "component byte sum {offset} differs from expert bytes {expert_bytes}"
        ));
    }
    for (component, &bytes) in component_bytes.iter().enumerate() {
        for expert in 0..cold {
            destinations.push(
                device_base
                    + (expert * expert_bytes + component_destination_offsets[component]) as u64,
            );
            sources.push(host[component].pointer as usize as u64 + (expert * bytes) as u64);
            sizes.push(bytes);
        }
    }
    Ok((destinations, sources, sizes))
}

fn verify_last_copy(
    device_base: cuda_sys::CUdeviceptr,
    host: &[HostAllocation],
    component_bytes: &[usize],
    cold: usize,
    expert_bytes: usize,
) -> Result<(), String> {
    for expert in 0..cold {
        let mut destination_offset = expert * expert_bytes;
        for (component, &bytes) in component_bytes.iter().enumerate() {
            let mut observed = [0u8; 1];
            check(
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        observed.as_mut_ptr() as *mut std::ffi::c_void,
                        device_base + destination_offset as u64,
                        1,
                    )
                },
                "verification D2H",
            )?;
            let expected = unsafe { *((host[component].pointer as *const u8).add(expert * bytes)) };
            if observed[0] != expected {
                return Err(format!(
                    "copy verification failed expert={expert} component={component}: observed={} expected={expected}",
                    observed[0]
                ));
            }
            destination_offset += bytes;
        }
    }
    Ok(())
}

fn check(result: cuda_sys::CUresult, operation: &str) -> Result<(), String> {
    if result == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation}: {result:?}"))
    }
}

fn parse_args() -> Result<Args, String> {
    let mut gpu = None;
    let mut host_mode = None;
    let mut component_bytes = None;
    let mut cold_experts = None;
    let mut warmup = 3usize;
    let mut samples = 20usize;
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let value = raw
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--gpu" => gpu = Some(parse_number(&value, &flag)?),
            "--host-mode" => {
                host_mode = Some(match value.as_str() {
                    "alloc" => HostMode::Alloc,
                    "register" => HostMode::Register,
                    "register-mapped" => HostMode::RegisterMapped,
                    _ => return Err(format!("invalid --host-mode value {value:?}\n{}", usage())),
                })
            }
            "--component-bytes" => component_bytes = Some(parse_list(&value, &flag)?),
            "--cold-experts" => cold_experts = Some(parse_list(&value, &flag)?),
            "--warmup" => warmup = parse_number(&value, &flag)?,
            "--samples" => samples = parse_number(&value, &flag)?,
            _ => return Err(format!("unknown argument {flag}\n{}", usage())),
        }
    }
    let component_bytes = component_bytes.ok_or_else(usage)?;
    let cold_experts = cold_experts.ok_or_else(usage)?;
    if component_bytes.is_empty()
        || component_bytes.iter().any(|value| *value == 0)
        || cold_experts.is_empty()
        || cold_experts.iter().any(|value| *value == 0)
        || samples == 0
    {
        return Err(usage());
    }
    Ok(Args {
        gpu: gpu.ok_or_else(usage)?,
        host_mode: host_mode.ok_or_else(usage)?,
        component_bytes,
        cold_experts,
        warmup,
        samples,
    })
}

fn parse_list(value: &str, flag: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|part| parse_number(part, flag))
        .collect()
}

fn parse_number<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))
}

fn usage() -> String {
    "Usage: ./dev pcie-batch-probe --gpu N --host-mode alloc|register|register-mapped --component-bytes N,N,... --cold-experts N,N,... [--warmup N] [--samples N]".to_string()
}
