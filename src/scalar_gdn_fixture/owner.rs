use crate::{cuda_failure_code,sequence_copy_completion,scalar_gdn_native,scalar_gdn_layout,scalar_gdn_owner,shared_gdn_workspace};
use shared_gdn_workspace::{Program, Workspace};
use cudarc::driver::{sys, CudaDevice, CudaSlice};
use scalar_gdn_layout::Layout;
use scalar_gdn_native::Native;
use scalar_gdn_owner::{Inputs, Owner, Plane};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
static OUTPUT_ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static RECORD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn record_number() -> usize {
    RECORD.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileId {
    pub(super) path: PathBuf,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema_version: u32,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    transport: FileId,
    native_plan: FileId,
    boundary_reference: FileId,
    admission_files: Vec<FileId>,
}
fn fail<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
pub(super) fn read(f: &FileId) -> Result<Vec<u8>, String> {
    let b = fs::read(&f.path).map_err(fail)?;
    if b.len() != f.bytes || hash(&b) != f.sha256 {
        return Err(format!("identity changed:{}", f.path.display()));
    }
    Ok(b)
}
pub(super) fn file(v: &Value) -> Result<FileId, String> {
    serde_json::from_value(v.clone()).map_err(fail)
}
pub(super) fn number(v: &Value, k: &str) -> Result<usize, String> {
    v[k].as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| format!("invalid actual GDN geometry:{k}"))
}
fn bytes32(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
pub(super) struct InputStorage {
    q: CudaSlice<u16>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    beta: CudaSlice<u16>,
    gate: CudaSlice<f32>,
    h: usize,
    kd: usize,
    vd: usize,
}
impl InputStorage {
    pub(super) fn new(
        d: &Arc<CudaDevice>,
        data: &BTreeMap<String, Vec<u8>>,
        h: usize,
        kd: usize,
        vd: usize,
    ) -> Result<Self, String> {
        let u16s = |n: &str| {
            data[n]
                .chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let gates = data["gate.f32"]
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        Ok(Self {
            q: d.htod_sync_copy(&u16s("q.bf16")).map_err(fail)?,
            k: d.htod_sync_copy(&u16s("k.bf16")).map_err(fail)?,
            v: d.htod_sync_copy(&u16s("v.bf16")).map_err(fail)?,
            beta: d.htod_sync_copy(&u16s("beta.bf16")).map_err(fail)?,
            gate: d.htod_sync_copy(&gates).map_err(fail)?,
            h,
            kd,
            vd,
        })
    }
    pub(super) fn verify_readonly(
        &self,
        d: &Arc<CudaDevice>,
        data: &BTreeMap<String, Vec<u8>>,
    ) -> Result<(), String> {
        for (n, buffer) in [
            ("q.bf16", &self.q),
            ("k.bf16", &self.k),
            ("v.bf16", &self.v),
            ("beta.bf16", &self.beta),
        ] {
            let actual = d
                .dtoh_sync_copy(buffer)
                .map_err(fail)?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            compare(&format!("readonly original input {n}"), &actual, &data[n])?;
        }
        compare(
            "readonly original raw gate",
            &bytes32(&d.dtoh_sync_copy(&self.gate).map_err(fail)?),
            &data["gate.f32"],
        )
    }
    pub(super) fn view(&self, start: usize, rows: usize) -> Result<Inputs<'_>, String> {
        Ok(Inputs {
            q: Plane::checked(&self.q, start * self.h * self.kd, rows * self.h * self.kd)?,
            k: Plane::checked(&self.k, start * self.h * self.kd, rows * self.h * self.kd)?,
            v: Plane::checked(&self.v, start * self.h * self.vd, rows * self.h * self.vd)?,
            beta: Plane::checked(&self.beta, start * self.h, rows * self.h)?,
            gate: Plane::checked(&self.gate, start * self.h, rows * self.h)?,
        })
    }
}
fn memory(margin: usize, minimum: &mut usize) -> Result<usize, String> {
    let (free, _) = cudarc::driver::result::mem_get_info().map_err(fail)?;
    *minimum = (*minimum).min(free);
    if free < margin {
        return Err(format!(
            "actual scalar GDN owner free VRAM{free} below configured margin{margin}"
        ));
    }
    Ok(free)
}
fn metadata(d: &Arc<CudaDevice>, o: &Owner, cursor: usize) -> Result<Vec<u64>, String> {
    if o.host_poisoned() {
        return Err("owner host error poison present".into());
    }
    let b = d
        .dtoh_sync_copy(&o.persistent()["metadata.u64"])
        .map_err(fail)?;
    if let Some(root) = OUTPUT_ROOT.get() {
        fs::write(
            root.join(format!(
                "metadata-{:04}-cursor{cursor}.bin",
                record_number()
            )),
            &b,
        )
        .map_err(fail)?;
    }
    let m = b
        .chunks_exact(8)
        .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    if m.len() != 8
        || m[0] != cursor as u64
        || m[1] != (cursor % 64) as u64
        || m[3] != 0
        || m[4] != 0
    {
        return Err(format!(
            "scalar GDN cursor/pending/poison/inflight invalid:{m:?}"
        ));
    }
    Ok(m)
}
fn emitted(
    d: &Arc<CudaDevice>,
    workspace: &Workspace,
    o: &Owner,
    rows: usize,
) -> Result<Vec<u8>, String> {
    let bytes = rows * o.layout.heads * o.layout.value_dim * 2;
    let view = workspace.buffer("emitted.bf16")?.slice(..bytes);
    d.dtoh_sync_copy(&view).map_err(fail)
}
fn latest(d:&Arc<CudaDevice>,o:&Owner)->Result<Vec<u8>,String>{super::read_latest(d,o)}
fn snapshot(d: &Arc<CudaDevice>, o: &Owner) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut r = BTreeMap::new();
    for (&n, b) in o.persistent() {
        r.insert(n.into(), d.dtoh_sync_copy(b).map_err(fail)?);
    }
    r.insert("latest-state.f32".into(), latest(d, o)?);
    Ok(r)
}
fn retain_state(root: &Path, label: &str, data: &BTreeMap<String, Vec<u8>>) -> Result<(), String> {
    let dir = root.join(label);
    fs::create_dir(&dir).map_err(fail)?;
    for (name, bytes) in data {
        fs::write(dir.join(name), bytes).map_err(fail)?;
    }
    Ok(())
}
fn make_owner(program:&Arc<Program>,layout:&Layout,initial:&[u8])->Result<Owner,String>{super::take_latest(program,layout,initial)}

pub(super) fn compare(label: &str, actual: &[u8], expected: &[u8]) -> Result<(), String> {
    if actual != expected {
        let first = actual.iter().zip(expected).position(|(a, b)| a != b);
        if let Some(root) = OUTPUT_ROOT.get() {
            let seq = record_number();
            let prefix = format!("failure-{seq:04}");
            fs::write(root.join(format!("{prefix}-actual.bin")), actual).map_err(fail)?;
            fs::write(root.join(format!("{prefix}-expected.bin")), expected).map_err(fail)?;
            fs::write(root.join("FAILURE.json"),serde_json::to_vec_pretty(&json!({"passed":false,"label":label,"firstDifferingByte":first,"actualBytes":actual.len(),"expectedBytes":expected.len(),"actualSha256":hash(actual),"expectedSha256":hash(expected),"prefix":prefix,"scope":"Actual negative observation retained before error return; no failed comparison waived."})).map_err(fail)?).map_err(fail)?;
        }

        return Err(format!(
            "{label} differs; actualbytes{} expectedbytes{} firstbyte{first:?}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(())
}
struct SecondaryContext {
    raw: sys::CUcontext,
    primary: sys::CUcontext,
}
impl Drop for SecondaryContext {
    fn drop(&mut self) {
        let finish = (|| -> Result<(), String> {
            unsafe { sys::lib().cuCtxSetCurrent(self.raw) }
                .result()
                .map_err(fail)?;
            unsafe { sys::lib().cuCtxSynchronize() }
                .result()
                .map_err(fail)?;
            unsafe { sys::lib().cuCtxSetCurrent(self.primary) }
                .result()
                .map_err(fail)?;
            unsafe { sys::lib().cuCtxDestroy_v2(self.raw) }
                .result()
                .map_err(fail)?;
            unsafe { sys::lib().cuCtxSetCurrent(self.primary) }
                .result()
                .map_err(fail)?;
            Ok(())
        })();
        if let Err(e) = finish {
            eprintln!("KRASIS FATAL: secondary context completion/release unknown:{e}");
            std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
        }
    }
}
pub(super) fn run(path: &Path, root: &Path, check: bool) -> Result<(), String> {
    let p: Plan = serde_json::from_slice(&fs::read(path).map_err(fail)?).map_err(fail)?;
    if p.schema_version != 1 || p.safety_margin_bytes == 0 || !p.gpu_uuid.starts_with("GPU-") {
        return Err("invalid explicit scalar owner plan".into());
    }
    sequence_copy_completion::error_contract_controls()?;
    for f in &p.admission_files {
        read(f)?;
    }
    let transport = String::from_utf8(read(&p.transport)?).map_err(fail)?;
    let original: Value = serde_json::from_slice(&read(&p.native_plan)?).map_err(fail)?;
    if original["gpu_uuid"] != p.gpu_uuid
        || original["safety_margin_bytes"] != p.safety_margin_bytes
    {
        return Err("first owner replay must use actual predecessor GPU/margin".into());
    }
    let cases = original["cases"]
        .as_array()
        .ok_or("actual scalar cases absent")?;
    let full = cases
        .iter()
        .find(|x| x["name"] == "full")
        .ok_or("full case absent")?;
    let base = cases
        .iter()
        .find(|x| x["name"] == "base")
        .ok_or("base case absent")?;
    let g = &full["geometry"];
    let h = number(g, "heads")?;
    let kd = number(g, "key_dim")?;
    let vd = number(g, "value_dim")?;
    let rows = number(g, "real_rows")?;
    let padded = number(g, "padded_rows")?;
    let base_rows = number(&base["geometry"], "real_rows")?;
    if number(g, "block_rows")? != 64
        || base_rows >= rows
        || base_rows % 64 == 0
        || rows - base_rows >= 64
    {
        return Err("actual retained pending-block discriminator absent".into());
    }
    let layout = Layout::checked(h, kd, vd, rows)?;
    let sidecar = file(&original["sidecar"])?;
    read(&sidecar)?;
    let mut data: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for name in [
        "q.bf16",
        "k.bf16",
        "v.bf16",
        "beta.bf16",
        "gate.f32",
        "initial-state.f32",
        "actual-output.bf16",
        "actual-final-state.f32",
    ] {
        data.insert(name.into(), read(&file(&full["files"][name])?)?);
    }
    for (name, width, bytes) in [
        ("q.bf16", h * kd, 2),
        ("k.bf16", h * kd, 2),
        ("v.bf16", h * vd, 2),
        ("beta.bf16", h, 2),
        ("gate.f32", h, 4),
    ] {
        if data[name].len() != padded * width * bytes {
            return Err(format!("actual input extent differs:{name}"));
        }
    }
    let state_bytes = h * kd * vd * 4;
    let boundary_reference = read(&p.boundary_reference)?;
    if boundary_reference.len() != state_bytes {
        return Err("native boundary reference extent differs".into());
    }
    if data["initial-state.f32"].len() != state_bytes
        || data["actual-final-state.f32"].len() != state_bytes
        || data["actual-output.bf16"].len() != rows * h * vd * 2
    {
        return Err("actual output/state extents differ".into());
    }
    for (n, b) in &data {
        let finite = if n.ends_with(".bf16") {
            b.chunks_exact(2)
                .all(|x| u16::from_le_bytes(x.try_into().unwrap()) & 0x7f80 != 0x7f80)
        } else {
            b.chunks_exact(4)
                .all(|x| f32::from_le_bytes(x.try_into().unwrap()).is_finite())
        };
        if !finite {
            return Err(format!("nonfinite actual captured operand/reference:{n}"));
        }
    }
    let base_output = read(&file(&base["files"]["actual-output.bf16"])?)?;
    let base_state = read(&file(&base["files"]["actual-final-state.f32"])?)?;
    if base_output.len() != base_rows * h * vd * 2 || base_state.len() != state_bytes {
        return Err("actual base output/state extents differ".into());
    }
    let input_bytes = ["q.bf16", "k.bf16", "v.bf16", "beta.bf16", "gate.f32"]
        .iter()
        .try_fold(0usize, |n, k| {
            n.checked_add(data[*k].len())
                .ok_or("input allocation overflow")
        })?;
    // Peak fixture cardinalities: two independent live layer owners and two
    // real position slots. These are control counts, never memory reserves.
    let decode_shape = Layout::checked(h, kd, vd, 1)?;
    let decode_workspace_bytes = decode_shape.workspace_bytes()?;
    let state_live = layout
        .persistent_bytes()?
        .checked_add(state_bytes)
        .and_then(|n| n.checked_mul(2))
        .ok_or("two state owners overflow")?;
    let actual_required = input_bytes
        .checked_add(layout.workspace_bytes()?)
        .and_then(|n| n.checked_add(decode_workspace_bytes))
        .and_then(|n| n.checked_add(state_live))
        .and_then(|n| n.checked_add(2 * std::mem::size_of::<i32>()))
        .ok_or("shared owner admission overflow")?;
    if check {
        println!(
            "{}",
            json!({"passed":true,"gpu_initialized":false,"outputs_created":false,"heads":h,"key_dim":kd,"value_dim":vd,"real_rows":rows,"base_rows":base_rows,"actual_required_cuda_bytes":actual_required,"new_persistent_bytes_excluding_external_latest":layout.persistent_bytes()?,"workspace_bytes":layout.workspace_bytes()?})
        );
        return Ok(());
    }
    if root.exists() {
        return Err("fresh scalar owner output required".into());
    }
    std::env::set_var("CUDA_VISIBLE_DEVICES", &p.gpu_uuid);
    let d = CudaDevice::new_with_stream(0).map_err(fail)?;
    let mut u = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut u, *d.cu_device()) }
        .result()
        .map_err(fail)?;
    let u = u
        .bytes
        .iter()
        .map(|x| format!("{:02x}", *x as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &u[..8],
        &u[8..12],
        &u[12..16],
        &u[16..20],
        &u[20..]
    );
    if actual != p.gpu_uuid {
        return Err("actual scalar owner CUDA UUID differs".into());
    }
    let mut minimum = usize::MAX;
    let free = memory(p.safety_margin_bytes, &mut minimum)?;
    if free
        < actual_required
            .checked_add(p.safety_margin_bytes)
            .ok_or("margin overflow")?
    {
        return Err(
            "actual GPU free VRAM cannot admit scalar owner test allocations and configured margin"
                .into(),
        );
    }
    fs::create_dir(root).map_err(fail)?;
    OUTPUT_ROOT
        .set(root.to_path_buf())
        .map_err(|_| "duplicate observation root")?;
    let outcome = (|| -> Result<Value, String> {
        // Inputs precede owners and remain alive through every queued operation and
        // failure path. Owner Drop drains its stream before input allocations drop.
        let inputs = InputStorage::new(&d, &data, h, kd, vd)?;
        let program = Program::new(d.clone(), &[layout.clone()], &sidecar.path, &transport)?;
        let workspace = Workspace::new(program.clone(), &[layout.clone()])?;
        let lease = workspace.lease()?;
        let full_snapshot = {
            let mut owner = make_owner(&program, &layout, &data["initial-state.f32"])?;
            memory(p.safety_margin_bytes, &mut minimum)?;
            unsafe {
                owner.enqueue_prefill(&lease, 0, rows, &inputs.view(0, rows)?)?;
                super::after_first_enqueue()?;
            }
            d.synchronize().map_err(fail)?;
            memory(p.safety_margin_bytes, &mut minimum)?;
            metadata(&d, &owner, rows)?;
            let output = emitted(&d, &workspace, &owner, rows)?;
            compare(
                "complete full owner output",
                &output,
                &data["actual-output.bf16"],
            )?;
            compare(
                "complete full owner state",
                &latest(&d, &owner)?,
                &data["actual-final-state.f32"],
            )?;
            fs::write(root.join("full-output.bf16"), &output).map_err(fail)?;
            let captured = snapshot(&d, &owner)?;
            retain_state(root, "state-full", &captured)?;
            captured
        };
        let mut owner = make_owner(&program, &layout, &data["initial-state.f32"])?;
        unsafe {
            owner.enqueue_prefill(&lease, 0, base_rows, &inputs.view(0, base_rows)?)?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &owner, base_rows)?;
        compare(
            "complete base owner output",
            &emitted(&d, &workspace, &owner, base_rows)?,
            &base_output,
        )?;
        compare(
            "complete base owner state",
            &latest(&d, &owner)?,
            &base_state,
        )?;
        let checkpoint = snapshot(&d, &owner)?;
        retain_state(root, "state-base", &checkpoint)?;
        unsafe {
            owner.enqueue_prefill(
                &lease,
                base_rows as u64,
                rows - base_rows,
                &inputs.view(base_rows, rows - base_rows)?,
            )?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &owner, rows)?;
        let expected = &data["actual-output.bf16"][base_rows * h * vd * 2..];
        let resumed = emitted(&d, &workspace, &owner, rows - base_rows)?;
        compare("pending plus suffix output", &resumed, expected)?;
        compare(
            "pending plus suffix final state",
            &latest(&d, &owner)?,
            &data["actual-final-state.f32"],
        )?;
        let resumed_snapshot = snapshot(&d, &owner)?;
        retain_state(root, "state-resumed", &resumed_snapshot)?;
        for (n, b) in &full_snapshot {
            if n != "metadata.u64" {
                compare(
                    &format!("full versus split persistent {n}"),
                    &resumed_snapshot[n],
                    b,
                )?;
            }
        }
        fs::write(root.join("resumed-output.bf16"), &resumed).map_err(fail)?;
        // Restore all real persistent planes and actual latest tensor; host vectors
        // survive through the completion barrier. This is an operator control, not
        // installed session-cache restore qualification.
        unsafe {
            sequence_copy_completion::with_copy_completion(
                &d,
                *d.cu_stream(),
                &(&owner, &checkpoint),
                |(owner, checkpoint)| {
                    for (n, b) in checkpoint.iter() {
                        use cudarc::driver::DevicePtr;
                        let ptr = if n == "latest-state.f32" {
                            owner.latest_pointer()
                        } else {
                            *owner.persistent()[n.as_str()].device_ptr()
                        };
                        sys::lib()
                            .cuMemcpyHtoDAsync_v2(ptr, b.as_ptr().cast(), b.len(), *d.cu_stream())
                            .result()
                            .map_err(fail)?;
                    }
                    Ok(())
                },
            )?;
        }
        metadata(&d, &owner, base_rows)?;
        let mut decoded = Vec::new();
        for position in base_rows..rows {
            let position_tensor = d.htod_sync_copy(&[position as i32]).map_err(fail)?;
            // Explicit completion while both the real position tensor and input owner
            // remain borrowed, including any enqueue error.
            let r = unsafe {
                owner.enqueue_decode(&lease, &inputs.view(position, 1)?, &position_tensor)
            };
            let completion = d.synchronize().map_err(fail);
            if completion.is_err() {
                eprintln!("KRASIS FATAL: owner decode completion unknown");
                std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
            }
            r?;
            completion?;
            metadata(&d, &owner, position + 1)?;
            let one = emitted(&d, &workspace, &owner, 1)?;
            compare(
                "eager owner row",
                &one,
                &data["actual-output.bf16"][position * h * vd * 2..(position + 1) * h * vd * 2],
            )?;
            decoded.extend(one);
        }
        compare("all eager rows", &decoded, expected)?;
        compare(
            "eager final state",
            &latest(&d, &owner)?,
            &data["actual-final-state.f32"],
        )?;
        let eager_snapshot = snapshot(&d, &owner)?;
        retain_state(root, "state-eager", &eager_snapshot)?;
        for (n, b) in &full_snapshot {
            if n != "metadata.u64" {
                compare(
                    &format!("full versus eager persistent {n}"),
                    &eager_snapshot[n],
                    b,
                )?;
            }
        }
        memory(p.safety_margin_bytes, &mut minimum)?;
        fs::write(root.join("eager-output.bf16"), &decoded).map_err(fail)?;
        drop(owner);
        // Cover exact completion and the eager63+1 rollover; the actual169
        // native aligned-prefix FP32 carry is an independent retained reference.
        let anchor = base_rows / 64 * 64;
        let mut exact = make_owner(&program, &layout, &data["initial-state.f32"])?;
        unsafe {
            exact.enqueue_prefill(&lease, 0, anchor, &inputs.view(0, anchor)?)?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &exact, anchor)?;
        compare(
            "exact-block prefix output",
            &emitted(&d, &workspace, &exact, anchor)?,
            &data["actual-output.bf16"][..anchor * h * vd * 2],
        )?;
        compare(
            "exact-block prefix FP32state",
            &latest(&d, &exact)?,
            &boundary_reference,
        )?;
        compare(
            "exact-block boundary carry",
            &d.dtoh_sync_copy(&exact.persistent()["boundary-state.f32"])
                .map_err(fail)?,
            &boundary_reference,
        )?;
        retain_state(root, "state-exact-prefix", &snapshot(&d, &exact)?)?;
        unsafe {
            exact.enqueue_prefill(
                &lease,
                anchor as u64,
                rows - anchor,
                &inputs.view(anchor, rows - anchor)?,
            )?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &exact, rows)?;
        compare(
            "exact-block suffix output",
            &emitted(&d, &workspace, &exact, rows - anchor)?,
            &data["actual-output.bf16"][anchor * h * vd * 2..],
        )?;
        compare(
            "exact-block suffix state",
            &latest(&d, &exact)?,
            &data["actual-final-state.f32"],
        )?;
        drop(exact);
        let crossing_start = anchor - 1;
        let mut crossing = make_owner(&program, &layout, &data["initial-state.f32"])?;
        unsafe {
            crossing.enqueue_prefill(
                &lease,
                0,
                crossing_start,
                &inputs.view(0, crossing_start)?,
            )?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &crossing, crossing_start)?;
        compare(
            "crossing prefix output",
            &emitted(&d, &workspace, &crossing, crossing_start)?,
            &data["actual-output.bf16"][..crossing_start * h * vd * 2],
        )?;
        for position in crossing_start..rows {
            let position_tensor = d.htod_sync_copy(&[position as i32]).map_err(fail)?;
            let r = unsafe {
                crossing.enqueue_decode(&lease, &inputs.view(position, 1)?, &position_tensor)
            };
            let completion = d.synchronize().map_err(fail);
            if completion.is_err() {
                eprintln!("KRASIS FATAL: crossing decode completion unknown");
                std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
            }
            r?;
            completion?;
            metadata(&d, &crossing, position + 1)?;
            compare(
                "crossing eager output",
                &emitted(&d, &workspace, &crossing, 1)?,
                &data["actual-output.bf16"][position * h * vd * 2..(position + 1) * h * vd * 2],
            )?;
            if position + 1 == anchor {
                retain_state(
                    root,
                    "state-eager-exact-boundary",
                    &snapshot(&d, &crossing)?,
                )?;
                compare(
                    "eager exact64 lateststate",
                    &latest(&d, &crossing)?,
                    &boundary_reference,
                )?;
                compare(
                    "eager exact64 boundarycarry",
                    &d.dtoh_sync_copy(&crossing.persistent()["boundary-state.f32"])
                        .map_err(fail)?,
                    &boundary_reference,
                )?;
            }
        }
        compare(
            "crossing final state",
            &latest(&d, &crossing)?,
            &data["actual-final-state.f32"],
        )?;
        let crossing_snapshot = snapshot(&d, &crossing)?;
        retain_state(root, "state-crossing-final", &crossing_snapshot)?;
        for (n, b) in &full_snapshot {
            if n != "metadata.u64" {
                compare(
                    &format!("full versus crossing persistent {n}"),
                    &crossing_snapshot[n],
                    b,
                )?;
            }
        }
        memory(p.safety_margin_bytes, &mut minimum)?;
        inputs.verify_readonly(&d, &data)?;
        // Named adoption uncertainty: generated FLA C wrappers store static
        // module/function globals. Two real same-device contexts exercise the
        // same-architecture isolation needed by arbitrary-GPU decode owners.
        let small_layout = Layout::checked(h, kd, vd, 64)?;
        let mut existing = make_owner(&program, &small_layout, &data["initial-state.f32"])?;
        unsafe {
            existing.enqueue_prefill(&lease, 0, 64, &inputs.view(0, 64)?)?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &existing, 64)?;
        compare(
            "pre-secondary-context complete64output",
            &emitted(&d, &workspace, &existing, 64)?,
            &data["actual-output.bf16"][..64 * h * vd * 2],
        )?;
        let mut secondary = std::ptr::null_mut();
        unsafe {
            sys::lib().cuCtxCreate_v2(
                &mut secondary,
                sys::CUctx_flags::CU_CTX_SCHED_AUTO as u32,
                *d.cu_device(),
            )
        }
        .result()
        .map_err(fail)?;
        let context = SecondaryContext {
            raw: secondary,
            primary: *d.cu_primary_ctx(),
        };
        let second = unsafe { Native::load(&sidecar.path, h, kd, vd)? };
        let secondary_free = memory(p.safety_margin_bytes, &mut minimum)?;
        d.bind_to_thread().map_err(fail)?;
        let enqueue = unsafe { existing.enqueue_prefill(&lease, 64, 1, &inputs.view(64, 1)?) };
        let mut observation = json!({"same_physical_gpu_two_real_contexts":true,"gpu_uuid":actual,"primary_context":format!("{:p}",context.primary),"secondary_context":format!("{:p}",context.raw),"pre_secondary_64row_output_exact":true,"same_native_library":true,"secondary_context_free_bytes":secondary_free,"old_owner_enqueue_after_other_context_init_ok":enqueue.is_ok(),"actual_error":enqueue.as_ref().err(),"scope":"Real explicit-context wrapper isolation control. All108numerical cubins byte-identical to qualified originals; actual create/destroy/launch ABI and native bindings changed. No model/serving change; failed comparison remains failure."});
        fs::write(
            root.join("CONTEXT_OBSERVATION.json"),
            serde_json::to_vec_pretty(&observation).map_err(fail)?,
        )
        .map_err(fail)?;
        let completion = d.synchronize();
        observation["primary_completion_ok"] = json!(completion.is_ok());
        fs::write(
            root.join("CONTEXT_OBSERVATION.json"),
            serde_json::to_vec_pretty(&observation).map_err(fail)?,
        )
        .map_err(fail)?;
        if let Err(e) = completion {
            eprintln!("KRASIS FATAL: primary context completion unknown:{e}");
            std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
        }
        drop(second);
        drop(context);
        d.bind_to_thread().map_err(fail)?;
        if let Err(e) = enqueue {
            return Err(format!("native context isolation FAILED after actual secondary-context init:{e}; observation retained"));
        }
        metadata(&d, &existing, 65)?;
        compare(
            "post-secondary-context output",
            &emitted(&d, &workspace, &existing, 1)?,
            &data["actual-output.bf16"][64 * h * vd * 2..65 * h * vd * 2],
        )?;
        drop(existing);
        drop(crossing);
        if workspace.lease().is_ok() {
            return Err("workspace allowed a second concurrent host phase".into());
        }
        let decode_layout = Layout::checked(h, kd, vd, 1)?;
        let decode_workspace = Workspace::new(program.clone(), &[decode_layout])?;
        let decode_lease = decode_workspace.lease()?;
        let mut a_state = make_owner(&program, &layout, &data["initial-state.f32"])?;
        let mut b_state = make_owner(&program, &layout, &data["initial-state.f32"])?;
        for (name, a_buffer) in a_state.persistent() {
            use cudarc::driver::DevicePtr;
            if a_buffer.device_ptr() == b_state.persistent()[name].device_ptr() {
                return Err("independent layer states alias".into());
            }
        }
        unsafe {
            a_state.enqueue_prefill(&lease, 0, base_rows, &inputs.view(0, base_rows)?)?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &a_state, base_rows)?;
        compare(
            "interleaved A base",
            &emitted(&d, &workspace, &a_state, base_rows)?,
            &base_output,
        )?;
        unsafe {
            b_state.enqueue_prefill(&lease, 0, anchor, &inputs.view(0, anchor)?)?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &b_state, anchor)?;
        compare(
            "interleaved B prefix",
            &emitted(&d, &workspace, &b_state, anchor)?,
            &data["actual-output.bf16"][..anchor * h * vd * 2],
        )?;
        let a_before = snapshot(&d, &a_state)?;
        let invalid = unsafe {
            a_state.enqueue_prefill(
                &decode_lease,
                base_rows as u64,
                rows - base_rows,
                &inputs.view(base_rows, rows - base_rows)?,
            )
        };
        if invalid.is_ok() {
            return Err("decode-only workspace admitted an oversized prefill phase".into());
        }
        compare(
            "rejected workspace leaves metadata unchanged",
            &snapshot(&d, &a_state)?["metadata.u64"],
            &a_before["metadata.u64"],
        )?;
        if a_state.host_poisoned() {
            return Err("pre-enqueue capacity rejection poisoned valid state".into());
        }
        let position = d.htod_sync_copy(&[base_rows as i32]).map_err(fail)?;
        let queued = unsafe {
            a_state.enqueue_decode(&decode_lease, &inputs.view(base_rows, 1)?, &position)
        };
        let completed = d.synchronize().map_err(fail);
        if completed.is_err() {
            eprintln!("KRASIS FATAL: shared decode completion unknown");
            std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
        }
        queued?;
        completed?;
        metadata(&d, &a_state, base_rows + 1)?;
        compare(
            "small shared decode output",
            &emitted(&d, &decode_workspace, &a_state, 1)?,
            &data["actual-output.bf16"][base_rows * h * vd * 2..(base_rows + 1) * h * vd * 2],
        )?;
        unsafe {
            a_state.enqueue_prefill(
                &lease,
                (base_rows + 1) as u64,
                rows - base_rows - 1,
                &inputs.view(base_rows + 1, rows - base_rows - 1)?,
            )?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &a_state, rows)?;
        compare(
            "interleaved A suffix",
            &emitted(&d, &workspace, &a_state, rows - base_rows - 1)?,
            &data["actual-output.bf16"][(base_rows + 1) * h * vd * 2..],
        )?;
        unsafe {
            b_state.enqueue_prefill(
                &lease,
                anchor as u64,
                rows - anchor,
                &inputs.view(anchor, rows - anchor)?,
            )?;
        }
        d.synchronize().map_err(fail)?;
        metadata(&d, &b_state, rows)?;
        compare(
            "interleaved B suffix",
            &emitted(&d, &workspace, &b_state, rows - anchor)?,
            &data["actual-output.bf16"][anchor * h * vd * 2..],
        )?;
        for (label, state) in [
            ("state-interleaved-A", &a_state),
            ("state-interleaved-B", &b_state),
        ] {
            let captured = snapshot(&d, state)?;
            retain_state(root, label, &captured)?;
            for (name, want) in &full_snapshot {
                if name != "metadata.u64" {
                    compare(&format!("{label} {name}"), &captured[name], want)?;
                }
            }
        }
        let shared_prefill_bytes = workspace.bytes()?;
        let shared_decode_bytes = decode_workspace.bytes()?;
        drop(a_state);
        drop(b_state);
        drop(decode_lease);
        drop(lease);
        d.synchronize().map_err(fail)?;
        if Arc::strong_count(&workspace) != 1 {
            return Err("persistent states retained the transient prefill workspace".into());
        }
        let before_release = memory(p.safety_margin_bytes, &mut minimum)?;
        drop(workspace);
        d.synchronize().map_err(fail)?;
        let after_release = memory(p.safety_margin_bytes, &mut minimum)?;
        let mut after = make_owner(&program, &layout, &data["initial-state.f32"])?;
        let position = d.htod_sync_copy(&[0i32]).map_err(fail)?;
        let decode_lease = decode_workspace.lease()?;
        let queued = unsafe { after.enqueue_decode(&decode_lease, &inputs.view(0, 1)?, &position) };
        let completed = d.synchronize().map_err(fail);
        if completed.is_err() {
            eprintln!("KRASIS FATAL: postrelease decode completion unknown");
            std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
        }
        queued?;
        completed?;
        metadata(&d, &after, 1)?;
        compare(
            "decode after prefill arena release",
            &emitted(&d, &decode_workspace, &after, 1)?,
            &data["actual-output.bf16"][..h * vd * 2],
        )?;
        retain_state(
            root,
            "state-decode-after-prefill-release",
            &snapshot(&d, &after)?,
        )?;
        inputs.verify_readonly(&d, &data)?;
        Ok(
            json!({"passed":true,"actual_gpu_uuid":actual,"complete_full_and_base_output_state_exact":true,"partial_plus_suffix_output_state_exact":true,"all_eager_rows_and_final_state_exact":true,"all_real_persistent_planes_match_except_execution_metadata":true,"operator_checkpoint_restore":true,"successful_raw_states_retained":true,"two_independent_states_share_one_prefill_workspace":true,"separate_small_decode_workspace_passed":true,"decode_after_prefill_workspace_release_exact":true,"pre_enqueue_capacity_rejection_preserves_state":true,"exclusive_host_phase_enforced":true,"shared_prefill_workspace_bytes":shared_prefill_bytes,"shared_decode_workspace_bytes":shared_decode_bytes,"actual_free_before_prefill_release":before_release,"actual_free_after_prefill_release":after_release,"all_original_gpu_input_bytes_readonly_exact":true,"exact_completed_block_and_eager_rollover_native_carry_exact":true,"crossing_eager_rows":rows-crossing_start,"eager_rows":rows-base_rows,"full_rows":rows,"base_rows":base_rows,"minimum_completed_stage_free_bytes":minimum,"safety_margin_bytes":p.safety_margin_bytes,"new_persistent_bytes_excluding_external_latest":layout.persistent_bytes()?,"workspace_bytes":layout.workspace_bytes()?,"graph_replay_qualified":false,"installed_snapshot_error_cancel_paths_qualified":false,"serving_qualified":false,"actual_two_context_isolation_passed":true,"scope":"Actual scalar native64 owner and retained real inputs. Complete native169 references, prefillsplit and eagerdecode through same owner. Operator rawrestore only. No model load, ordinary conversation, source-fidelity/Witness or performance claim."}),
        )
    })();
    // Input owners dropped inside closure; drain their async frees on every path.
    if let Err(e) = d.synchronize() {
        eprintln!("KRASIS FATAL: owner-probe final cleanup completion unknown:{e}");
        std::process::exit(cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
    }
    let report = outcome?;
    fs::write(
        root.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(fail)?,
    )
    .map_err(fail)?;
    println!("{report}");
    Ok(())
}

impl InputStorage {
    /// Actual static row slots for capture; caller retains both real allocations
    /// through queued completion. This is diagnostic transport, not sampling.
    pub(super) unsafe fn stage_row(&self,device:&Arc<CudaDevice>,slots:&Self,row:usize,position:&CudaSlice<i32>)->Result<(),String>{
        use cudarc::driver::{DevicePtr,DeviceSlice};
        if (self.h,self.kd,self.vd)!=(slots.h,slots.kd,slots.vd){return Err("actual graph slot geometry differs".into());}
        device.bind_to_thread().map_err(fail)?;
        for (destination,source,width,bytes,capacity) in [
            (*slots.q.device_ptr(),*self.q.device_ptr(),self.h*self.kd,2,self.q.len()),
            (*slots.k.device_ptr(),*self.k.device_ptr(),self.h*self.kd,2,self.k.len()),
            (*slots.v.device_ptr(),*self.v.device_ptr(),self.h*self.vd,2,self.v.len()),
            (*slots.beta.device_ptr(),*self.beta.device_ptr(),self.h,2,self.beta.len()),
            (*slots.gate.device_ptr(),*self.gate.device_ptr(),self.h,4,self.gate.len()),
        ]{
            if row.checked_add(1).and_then(|n|n.checked_mul(width)).is_none_or(|n|n>capacity){return Err("actual source row exceeds input extent".into());}
            sys::lib().cuMemcpyDtoDAsync_v2(destination,source+(row*width*bytes) as u64,width*bytes,*device.cu_stream()).result().map_err(fail)?;
        }
        sys::lib().cuMemsetD32Async(*position.device_ptr(),u32::try_from(row).map_err(fail)?,1,*device.cu_stream()).result().map_err(fail)
    }
}
