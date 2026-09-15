//! Diagnostic Cargo wrapper. It only instruments the Krasis library test build.
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::process::{Command, ExitCode, Stdio};

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let compiler = arguments.next().ok_or("missing rustc executable")?;
    let arguments: Vec<OsString> = arguments.collect();
    let profile = arguments
        .windows(2)
        .any(|pair| pair[0] == "--crate-name" && pair[1] == "krasis")
        && arguments.iter().any(|arg| arg == "--test");
    let mut command = Command::new(&compiler);
    command.args(&arguments);
    if profile {
        let directory = std::path::PathBuf::from(
            env::var_os("KRASIS_COMPILER_TRACE_DIR").ok_or("missing trace directory")?,
        );
        fs::create_dir_all(&directory)?;
        let prefix = directory.join(format!("rustc-{}", std::process::id()));
        fs::write(
            prefix.with_extension("command.txt"),
            format!("{command:?}\n"),
        )?;
        // Scope unstable diagnostics to this crate and this child process.
        // Optimization, panic handling, test inventory and execution are unchanged.
        command.env("RUSTC_BOOTSTRAP", "krasis");
        // Stream diagnostics instead of retaining LLVM time-trace events in RAM.
        command.args(["-Ztime-passes", "-Csave-temps"]);
        // -1 runs every optimization and prints pass/function progress. Also
        // trace the legacy machine-code pass manager, which runs after IR passes.
        command.arg("-Cllvm-args=-opt-bisect-limit=-1");
        command.arg("-Cllvm-args=-debug-pass=Executions");
        command.stderr(Stdio::from(File::create(
            prefix.with_extension("stderr.log"),
        )?));
        eprintln!(
            "Compiler trace: {}",
            prefix.with_extension("stderr.log").display()
        );
    }
    let status = command.status()?;
    // Windows exception status codes cannot be represented in ExitCode. Preserve
    // their full value in the log and return failure to Cargo.
    if !status.success() {
        eprintln!("Wrapped compiler exited with {status}");
    }
    Ok(if status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("Compiler profiling failed: {error}");
            ExitCode::FAILURE
        }
    }
}
