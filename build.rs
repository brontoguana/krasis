fn main() {
    let total_timer = BuildTimer::start("build.rs total");
    let sidecar_abi = std::fs::read_to_string("sidecar_abi_version.txt")
        .expect("sidecar_abi_version.txt is required")
        .trim()
        .to_string();
    sidecar_abi
        .parse::<u32>()
        .expect("sidecar_abi_version.txt must contain a u32 ABI version");
    println!("cargo:rerun-if-changed=sidecar_abi_version.txt");
    println!("cargo:rustc-env=KRASIS_SIDECAR_ABI_VERSION={sidecar_abi}");
    println!("cargo::rustc-check-cfg=cfg(no_numa)");
    println!("cargo::rustc-check-cfg=cfg(has_decode_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_prefill_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_hqq_search_kernels)");

    // Force rerun when env changes (e.g. CUDA_HOME)
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Probe for libnuma — link only if the library is found.
    // The runtime code (numa.rs) checks numa_available() and falls back
    // gracefully, but the linker needs -lnuma at build time if we use
    // extern "C" FFI declarations.
    //
    // When libnuma is NOT found (e.g. CI manylinux containers), we set
    // cfg(no_numa) so numa.rs can stub out the FFI calls.
    let has_numa = timed_value("probe libnuma", || probe_lib("numa"));
    if has_numa {
        println!("cargo:rustc-link-lib=numa");
    } else {
        println!("cargo:rustc-cfg=no_numa");
        println!("cargo:warning=libnuma not found — NUMA support disabled (will use fallback)");
    }

    // Compile CUDA decode kernels to PTX if nvcc is available.
    // The PTX is embedded as a string constant via include_str!.
    timed_phase("decode PTX", compile_cuda_kernels);

    // Compile CUDA prefill kernels to PTX (Rust prefill path).
    timed_phase("prefill PTX", compile_prefill_kernels);

    // Compile diagnostic HQQ search kernels to PTX.
    timed_phase("HQQ search PTX", compile_hqq_search_kernels);

    total_timer.finish();
}

struct BuildTimer {
    label: &'static str,
    start: std::time::Instant,
}

impl BuildTimer {
    fn start(label: &'static str) -> Self {
        Self {
            label,
            start: std::time::Instant::now(),
        }
    }

    fn finish(self) {
        log_build_timing(self.label, self.start.elapsed());
    }
}

fn timed_phase<F>(label: &'static str, f: F)
where
    F: FnOnce(),
{
    let timer = BuildTimer::start(label);
    f();
    timer.finish();
}

fn timed_value<T, F>(label: &'static str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let timer = BuildTimer::start(label);
    let value = f();
    timer.finish();
    value
}

fn log_build_timing(label: &str, elapsed: std::time::Duration) {
    let safe_label = label.replace('"', "'");
    println!(
        "cargo:warning=KRASIS_BUILD_TIMING phase=\"{}\" duration_ms={} duration_s={:.3}",
        safe_label,
        elapsed.as_millis(),
        elapsed.as_secs_f64()
    );
}

fn is_output_fresh(inputs: &[&str], outputs: &[&str]) -> bool {
    if outputs.is_empty()
        || outputs
            .iter()
            .any(|path| !std::path::Path::new(path).exists())
    {
        return false;
    }

    let newest_input = inputs
        .iter()
        .filter_map(|path| file_mtime(path))
        .max()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let oldest_output = outputs
        .iter()
        .filter_map(|path| file_mtime(path))
        .min()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    oldest_output >= newest_input
}

fn file_mtime(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

fn run_status_timed(
    mut cmd: std::process::Command,
    label: &str,
) -> Result<std::process::ExitStatus, std::io::Error> {
    let start = std::time::Instant::now();
    let status = cmd.status();
    log_build_timing(label, start.elapsed());
    status
}

fn nvcc_host_compiler_args() -> Vec<String> {
    match std::env::var("KRASIS_NVCC_CCBIN") {
        Ok(path) if !path.trim().is_empty() => {
            vec!["-ccbin".to_string(), path]
        }
        _ => Vec::new(),
    }
}

fn compile_cuda_kernels() {
    let cu_src = "src/cuda/decode_kernels.cu";
    println!("cargo:rerun-if-changed={cu_src}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=decode_kernels.cu not found — GPU decode kernels disabled");
        return;
    }

    // Find nvcc
    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — GPU decode kernels disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/decode_kernels.ptx");

    if is_output_fresh(&[cu_src], &[&ptx_path]) {
        println!("cargo:rustc-cfg=has_decode_kernels");
        println!("cargo:warning=Reusing cached GPU decode kernels at {ptx_path}");
        return;
    }

    // Compile .cu to .ptx targeting sm_80 (works on Ampere, Ada, Hopper)
    let mut cmd = std::process::Command::new(&nvcc);
    cmd.args([
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-o",
        &ptx_path,
        cu_src,
    ])
    .args(nvcc_host_compiler_args());
    let status = run_status_timed(cmd, "nvcc decode PTX compile");

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_decode_kernels");
            println!("cargo:warning=Compiled GPU decode kernels to PTX ({ptx_path})");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed with status {s} — GPU decode kernels disabled");
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution error: {e} — GPU decode kernels disabled");
        }
    }
}

fn compile_prefill_kernels() {
    let cu_src = "src/cuda/prefill_kernels.cu";
    let shim_header = "src/cuda/prefill_shim.h";
    println!("cargo:rerun-if-changed={cu_src}");
    println!("cargo:rerun-if-changed={shim_header}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=prefill_kernels.cu not found — GPU prefill kernels disabled");
        return;
    }

    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — GPU prefill kernels disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/prefill_kernels.ptx");

    if is_output_fresh(&[cu_src, shim_header], &[&ptx_path]) {
        println!("cargo:rustc-cfg=has_prefill_kernels");
        println!("cargo:warning=Reusing cached GPU prefill kernels at {ptx_path}");
        return;
    }

    let mut cmd = std::process::Command::new(&nvcc);
    cmd.args([
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-o",
        &ptx_path,
        cu_src,
    ])
    .args(nvcc_host_compiler_args());
    let status = run_status_timed(cmd, "nvcc prefill PTX compile");

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_prefill_kernels");
            println!("cargo:warning=Compiled GPU prefill kernels to PTX ({ptx_path})");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed with status {s} — GPU prefill kernels disabled");
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution error: {e} — GPU prefill kernels disabled");
        }
    }
}

fn compile_hqq_search_kernels() {
    let cu_src = "src/cuda/hqq_search_kernels.cu";
    println!("cargo:rerun-if-changed={cu_src}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=hqq_search_kernels.cu not found — HQQ CUDA search disabled");
        return;
    }

    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — HQQ CUDA search disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/hqq_search_kernels.ptx");

    if is_output_fresh(&[cu_src], &[&ptx_path]) {
        println!("cargo:rustc-cfg=has_hqq_search_kernels");
        println!("cargo:warning=Reusing cached HQQ CUDA search kernels at {ptx_path}");
        return;
    }

    let mut cmd = std::process::Command::new(&nvcc);
    cmd.args([
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-o",
        &ptx_path,
        cu_src,
    ])
    .args(nvcc_host_compiler_args());
    let status = run_status_timed(cmd, "nvcc HQQ search PTX compile");

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_hqq_search_kernels");
            println!("cargo:warning=Compiled HQQ CUDA search kernels to PTX ({ptx_path})");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed with status {s} — HQQ CUDA search disabled");
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution error: {e} — HQQ CUDA search disabled");
        }
    }
}

fn find_nvcc() -> Option<String> {
    // Check CUDA_HOME / CUDA_PATH
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(cuda_dir) = std::env::var(var) {
            let nvcc = format!("{cuda_dir}/bin/nvcc");
            if std::path::Path::new(&nvcc).exists() {
                return Some(nvcc);
            }
        }
    }
    // Check common paths
    for path in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ] {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    // Try PATH
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        return Some("nvcc".to_string());
    }
    None
}

/// Try to find a shared library by compiling a minimal C program that links it.
fn probe_lib(name: &str) -> bool {
    // Quick check: see if the lib exists in common paths
    for dir in &["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"] {
        let so = format!("{dir}/lib{name}.so");
        if std::path::Path::new(&so).exists() {
            return true;
        }
    }
    // Try pkg-config as fallback
    std::process::Command::new("pkg-config")
        .args(["--exists", name])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
