fn main() {
    let total_timer = BuildTimer::start("build.rs total");
    compile_windows_launcher_resources();
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
    println!("cargo::rustc-check-cfg=cfg(has_peer_rtt_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_expert_codec_kernels)");

    // Force rerun when env changes (e.g. CUDA_HOME)
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=KRASIS_BUILD_PEER_RTT_PROBE");
    println!("cargo:rerun-if-env-changed=KRASIS_BUILD_EXPERT_CODEC_PROBE");

    // Probe for libnuma — link only if the library is found.
    // The runtime code (numa.rs) checks numa_available() and falls back
    // gracefully, but the linker needs -lnuma at build time if we use
    // extern "C" FFI declarations.
    //
    // When libnuma is NOT found (e.g. CI manylinux containers), we set
    // cfg(no_numa) so numa.rs can stub out the FFI calls.
    let has_numa = timed_value("probe libnuma", || {
        if cfg!(windows) {
            false
        } else {
            probe_lib("numa")
        }
    });
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

    // Compile the standalone peer-link feasibility probe. This kernel is not
    // loaded by model execution and adds no production hot-path work.
    timed_phase("peer RTT PTX", compile_peer_rtt_kernels);

    // Compile the real-data GPU entropy-codec gate independently before the
    // codec is admitted into production decode builds.
    timed_phase("expert codec PTX", compile_expert_codec_kernels);

    total_timer.finish();
}

fn compile_windows_launcher_resources() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let icon_path = std::path::Path::new("assets/windows/krasis.ico");
    println!("cargo:rerun-if-changed={}", icon_path.display());
    let absolute_icon = icon_path
        .canonicalize()
        .unwrap_or_else(|error| panic!("Windows launcher icon is missing: {error}"));
    let out_dir = std::path::PathBuf::from(
        std::env::var_os("OUT_DIR").expect("OUT_DIR is required for Windows resources"),
    );
    let rc_path = out_dir.join("krasis-windows-launcher.rc");
    let res_path = out_dir.join("krasis-windows-launcher.res");
    let icon_resource_path = absolute_icon.to_string_lossy().replace('\\', "/");
    std::fs::write(&rc_path, format!("1 ICON \"{icon_resource_path}\"\r\n"))
        .expect("failed to write Windows launcher resource script");

    let status = std::process::Command::new("rc.exe")
        .arg("/nologo")
        .arg("/fo")
        .arg(&res_path)
        .arg(&rc_path)
        .status()
        .unwrap_or_else(|error| {
            panic!("failed to start rc.exe for Windows launcher icon: {error}")
        });
    assert!(
        status.success(),
        "rc.exe failed to compile the Windows launcher icon: {status}"
    );
    println!(
        "cargo:rustc-link-arg-bin=krasis-windows-launcher={}",
        res_path.display()
    );
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
    let mut args = match std::env::var("KRASIS_NVCC_CCBIN") {
        Ok(path) if !path.trim().is_empty() => {
            vec!["-ccbin".to_string(), path]
        }
        _ => Vec::new(),
    };
    // glibc 2.43 exposes rsqrt/rsqrtf under _GNU_SOURCE with a noexcept
    // declaration that conflicts with CUDA 13.0/13.1's math_functions.h.
    // Keep the standard/default interfaces while hiding those GNU additions.
    if cfg!(target_os = "linux") {
        args.extend([
            "-U_GNU_SOURCE".to_string(),
            "-D_DEFAULT_SOURCE".to_string(),
            "-include".to_string(),
            "src/cuda/cuda_glibc_compat.h".to_string(),
        ]);
    }
    args
}

fn compile_peer_rtt_kernels() {
    if std::env::var("KRASIS_BUILD_PEER_RTT_PROBE").as_deref() != Ok("1") {
        return;
    }
    let cu_src = "src/cuda/peer_rtt_kernels.cu";
    println!("cargo:rerun-if-changed={cu_src}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=peer_rtt_kernels.cu not found — peer RTT probe disabled");
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — peer RTT probe disabled");
        return;
    };
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/peer_rtt_kernels.ptx");
    if is_output_fresh(&[cu_src], &[&ptx_path]) {
        println!("cargo:rustc-cfg=has_peer_rtt_kernels");
        println!("cargo:warning=Reusing cached peer RTT kernels at {ptx_path}");
        return;
    }

    let mut cmd = std::process::Command::new(&nvcc);
    cmd.args([
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "-o",
        &ptx_path,
        cu_src,
    ])
    .args(nvcc_host_compiler_args());
    match cmd.output() {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-cfg=has_peer_rtt_kernels");
            println!("cargo:warning=Compiled peer RTT kernels to PTX ({ptx_path})");
        }
        Ok(output) => {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                println!("cargo:warning=nvcc peer RTT: {line}");
            }
            println!(
                "cargo:warning=nvcc peer RTT failed with status {} — probe disabled",
                output.status
            );
        }
        Err(error) => {
            println!("cargo:warning=nvcc peer RTT execution error: {error} — probe disabled");
        }
    }
}

fn compile_expert_codec_kernels() {
    if std::env::var("KRASIS_BUILD_EXPERT_CODEC_PROBE").as_deref() != Ok("1") {
        return;
    }
    let cu_src = "src/cuda/expert_codec_kernels.cu";
    println!("cargo:rerun-if-changed={cu_src}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=expert_codec_kernels.cu not found — expert codec probe disabled");
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — expert codec probe disabled");
        return;
    };
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/expert_codec_kernels.ptx");
    if is_output_fresh(&[cu_src], &[&ptx_path]) {
        println!("cargo:rustc-cfg=has_expert_codec_kernels");
        println!("cargo:warning=Reusing cached expert codec kernels at {ptx_path}");
        return;
    }

    let mut cmd = std::process::Command::new(&nvcc);
    cmd.args([
        "-ptx",
        "-allow-unsupported-compiler",
        "-arch=sm_80",
        "-O3",
        "-o",
        &ptx_path,
        cu_src,
    ])
    .args(nvcc_host_compiler_args());
    match cmd.output() {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-cfg=has_expert_codec_kernels");
            println!("cargo:warning=Compiled expert codec kernels to PTX ({ptx_path})");
        }
        Ok(output) => {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                println!("cargo:warning=nvcc expert codec: {line}");
            }
            println!(
                "cargo:warning=nvcc expert codec failed with status {} — probe disabled",
                output.status
            );
        }
        Err(error) => {
            println!("cargo:warning=nvcc expert codec execution error: {error} — probe disabled");
        }
    }
}

fn compile_cuda_kernels() {
    let cu_src = "src/cuda/decode_kernels.cu";
    let expert_codec_src = "src/cuda/expert_codec_kernels.cu";
    let deepseek_v4_hc_header = "src/cuda/deepseek_v4_hc.cuh";
    let deepseek_v4_attention_header = "src/cuda/deepseek_v4_attention.cuh";
    let deepseek_v4_compressor_header = "src/cuda/deepseek_v4_compressor.cuh";
    println!("cargo:rerun-if-changed={cu_src}");
    // decode_kernels.cu includes the production entropy decoder directly.
    // Track the included source independently so a codec-only edit can never
    // reuse a stale decode cubin.
    println!("cargo:rerun-if-changed={expert_codec_src}");
    println!("cargo:rerun-if-changed={deepseek_v4_hc_header}");
    println!("cargo:rerun-if-changed={deepseek_v4_attention_header}");
    println!("cargo:rerun-if-changed={deepseek_v4_compressor_header}");
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

    if is_output_fresh(
        &[
            cu_src,
            expert_codec_src,
            deepseek_v4_hc_header,
            deepseek_v4_attention_header,
            deepseek_v4_compressor_header,
        ],
        &[&ptx_path],
    ) {
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
    let start = std::time::Instant::now();
    let output = cmd.output();
    log_build_timing("nvcc decode PTX compile", start.elapsed());

    match output {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-cfg=has_decode_kernels");
            println!("cargo:warning=Compiled GPU decode kernels to PTX ({ptx_path})");
        }
        Ok(output) => {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                println!("cargo:warning=nvcc decode: {line}");
            }
            panic!(
                "nvcc failed to compile required GPU decode kernels with status {}",
                output.status
            );
        }
        Err(e) => {
            panic!("failed to execute nvcc for required GPU decode kernels: {e}");
        }
    }
}

fn compile_prefill_kernels() {
    let cu_src = "src/cuda/prefill_kernels.cu";
    let shim_header = "src/cuda/prefill_shim.h";
    let deepseek_v4_hc_header = "src/cuda/deepseek_v4_hc.cuh";
    let deepseek_v4_attention_header = "src/cuda/deepseek_v4_attention.cuh";
    let deepseek_v4_compressor_header = "src/cuda/deepseek_v4_compressor.cuh";
    println!("cargo:rerun-if-changed={cu_src}");
    println!("cargo:rerun-if-changed={shim_header}");
    println!("cargo:rerun-if-changed={deepseek_v4_hc_header}");
    println!("cargo:rerun-if-changed={deepseek_v4_attention_header}");
    println!("cargo:rerun-if-changed={deepseek_v4_compressor_header}");
    if !std::path::Path::new(cu_src).exists() {
        panic!("required GPU prefill source is missing: {cu_src}");
    }

    let nvcc = find_nvcc()
        .unwrap_or_else(|| panic!("nvcc is required to build Krasis GPU prefill kernels"));

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/prefill_kernels.ptx");

    if is_output_fresh(
        &[
            cu_src,
            shim_header,
            deepseek_v4_hc_header,
            deepseek_v4_attention_header,
            deepseek_v4_compressor_header,
        ],
        &[&ptx_path],
    ) {
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
            panic!("nvcc failed to compile required GPU prefill kernels with status {s}");
        }
        Err(e) => {
            panic!("failed to execute nvcc for required GPU prefill kernels: {e}");
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
            for exe in ["nvcc", "nvcc.exe"] {
                let nvcc = std::path::Path::new(&cuda_dir).join("bin").join(exe);
                if nvcc.exists() {
                    return Some(nvcc.to_string_lossy().to_string());
                }
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
