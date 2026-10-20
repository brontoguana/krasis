#[path = "../../bf16-projection-probe/src/graph_owner.rs"]
mod graph_owner;
mod probe;
// Explicit F32 MLA primitive build/validation entry point, separate from serving.
#[allow(dead_code)]
#[path = "../../../src/mla_f32.rs"]
mod mla_f32;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
#[derive(Clone, Deserialize)]
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
            return Err(format!("identity mismatch {}", self.path.display()));
        }
        Ok(b)
    }
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn identity(path: &Path) -> Result<Value, String> {
    let b = fs::read(path).map_err(|e| e.to_string())?;
    Ok(json!({"path":path,"bytes":b.len(),"sha256":hash(&b)}))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildPlan {
    source: Identity,
    source_admission: Identity,
    nvcc: Identity,
    flags: Vec<String>,
}
fn build(plan: &str, out: &str) -> Result<(), String> {
    let raw = fs::read(plan).map_err(|e| e.to_string())?;
    let p: BuildPlan = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let source = p.source.read()?;
    p.nvcc.read()?;
    let admission: Value =
        serde_json::from_slice(&p.source_admission.read()?).map_err(|e| e.to_string())?;
    if admission["source"]["sha256"] != p.source.sha256 {
        return Err("primitive source admission differs".into());
    }
    if p.flags
        != [
            "-ptx",
            "-allow-unsupported-compiler",
            "-arch=sm_80",
            "-O3",
            "--use_fast_math",
            "-std=c++17",
        ]
    {
        return Err(
            "F32 MLA compilation flags differ from explicit native arithmetic contract".into(),
        );
    }
    let version = Command::new(&p.nvcc.path)
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !version.status.success() {
        return Err("nvcc version admission failed".into());
    }
    let root = PathBuf::from(out);
    fs::create_dir(&root).map_err(|e| e.to_string())?;
    let ptx = root.join("operator.ptx");
    let result = Command::new(&p.nvcc.path)
        .args(&p.flags)
        .arg(&p.source.path)
        .arg("-o")
        .arg(&ptx)
        .output()
        .map_err(|e| e.to_string())?;
    fs::write(root.join("compiler.stdout"), result.stdout).map_err(|e| e.to_string())?;
    fs::write(root.join("compiler.stderr"), result.stderr).map_err(|e| e.to_string())?;
    fs::write(root.join("compiler.version"), version.stdout).map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!("F32 MLA nvcc failed: {}", result.status));
    }
    if p.source.read()? != source {
        return Err("primitive source changed during compilation".into());
    }
    p.nvcc.read()?;
    let b = fs::read(&ptx).map_err(|e| e.to_string())?;
    let text = std::str::from_utf8(&b).map_err(|e| e.to_string())?;
    for name in [
        "mla_f32_rmsnorm",
        "mla_f32_prepare_query",
        "mla_f32_cache_k4",
        "mla_f32_cache_k6",
        "mla_f32_absorb_wkc",
    ] {
        if !text.contains(&format!(".entry {name}(")) {
            return Err(format!("compiled primitive absent: {name}"));
        }
    }
    let report = json!({"passed":true,"source":identity(&p.source.path)?,"ptx":identity(&ptx)?,"nvcc":identity(&p.nvcc.path)?,"flags":p.flags,"planSha256":hash(&raw),"binary":identity(&std::env::current_exe().map_err(|e|e.to_string())?)?,"scope":"Compilation and explicit ABI inventory only; no GPU arithmetic, native-parity, serving, fidelity, speed or ADQ qualification."});
    fs::write(
        root.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
fn run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use ./dev mla-primitives-probe".into());
    }
    let a: Vec<_> = std::env::args().collect();
    if a.len() == 4 && a[1] == "build" {
        build(&a[2], &a[3])
    } else if a.len() == 4 && a[1] == "sanitize" {
        probe::sanitize(&a[2], &a[3])
    } else if a.len() == 4 && a[1] == "probe" {
        probe::run(&a[2], &a[3])
    } else {
        Err("implemented command: build <bound-plan> <fresh-output>; primitive GPU controls are still under construction".into())
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
