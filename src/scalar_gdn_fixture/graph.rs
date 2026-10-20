#[path="graph_adapter.rs"]
mod scalar_gdn_graph;
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
struct FileId {
    path: PathBuf,
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
fn read(f: &FileId) -> Result<Vec<u8>, String> {
    let b = fs::read(&f.path).map_err(fail)?;
    if b.len() != f.bytes || hash(&b) != f.sha256 {
        return Err(format!("identity changed:{}", f.path.display()));
    }
    Ok(b)
}
fn file(v: &Value) -> Result<FileId, String> {
    serde_json::from_value(v.clone()).map_err(fail)
}
fn number(v: &Value, k: &str) -> Result<usize, String> {
    v[k].as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| format!("invalid actual GDN geometry:{k}"))
}
fn bytes32(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
struct InputStorage {
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
    fn new(
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
    fn verify_readonly(
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
    fn view(&self, start: usize, rows: usize) -> Result<Inputs<'_>, String> {
        Ok(Inputs {
            q: Plane::checked(&self.q, start * self.h * self.kd, rows * self.h * self.kd)?,
            k: Plane::checked(&self.k, start * self.h * self.kd, rows * self.h * self.kd)?,
            v: Plane::checked(&self.v, start * self.h * self.vd, rows * self.h * self.vd)?,
            beta: Plane::checked(&self.beta, start * self.h, rows * self.h)?,
            gate: Plane::checked(&self.gate, start * self.h, rows * self.h)?,
        })
    }
}
fn fill_slots(
    d: &Arc<CudaDevice>,
    slots: &InputStorage,
    source: &InputStorage,
    row: usize,
    position: &CudaSlice<i32>,
) -> Result<(), String> {
    use cudarc::driver::DevicePtr;
    let p = &source;
    for (dst, src, width, bytes) in [
        (*slots.q.device_ptr(), *p.q.device_ptr(), p.h * p.kd, 2),
        (*slots.k.device_ptr(), *p.k.device_ptr(), p.h * p.kd, 2),
        (*slots.v.device_ptr(), *p.v.device_ptr(), p.h * p.vd, 2),
        (*slots.beta.device_ptr(), *p.beta.device_ptr(), p.h, 2),
        (*slots.gate.device_ptr(), *p.gate.device_ptr(), p.h, 4),
    ] {
        unsafe {
            sys::lib().cuMemcpyDtoDAsync_v2(
                dst,
                src + (row * width * bytes) as u64,
                width * bytes,
                *d.cu_stream(),
            )
        }
        .result()
        .map_err(fail)?;
    }
    unsafe {
        sys::lib().cuMemsetD32Async(
            *position.device_ptr(),
            u32::try_from(row).map_err(fail)?,
            1,
            *d.cu_stream(),
        )
    }
    .result()
    .map_err(fail)?;
    Ok(())
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

fn compare(label: &str, actual: &[u8], expected: &[u8]) -> Result<(), String> {
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
    let prefix_workspace_bytes =
        Layout::checked(h, kd, vd, scalar_gdn_layout::NATIVE_BLOCK_ROWS - 1)?.workspace_bytes()?;
    let one_input_bytes = h * (kd * 4 + vd * 2 + 2 + 4);
    let actual_required = input_bytes
        .checked_add(prefix_workspace_bytes)
        .and_then(|n| n.checked_add(decode_workspace_bytes))
        .and_then(|n| n.checked_add(state_live))
        .and_then(|n| n.checked_add(one_input_bytes))
        .and_then(|n| n.checked_add(2 * std::mem::size_of::<i32>()))
        .ok_or("shared owner admission overflow")?;
    if check {
        println!(
            "{}",
            json!({"passed":true,"gpu_initialized":false,"outputs_created":false,"heads":h,"key_dim":kd,"value_dim":vd,"real_rows":rows,"base_rows":base_rows,"actual_required_cuda_bytes":actual_required,"new_persistent_bytes_excluding_external_latest":layout.persistent_bytes()?,"prefix_workspace_bytes":prefix_workspace_bytes,"decode_workspace_bytes":decode_workspace_bytes,"one_row_operand_bytes":one_input_bytes})
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
        let inputs = InputStorage::new(&d, &data, h, kd, vd)?;
        let program = Program::new(d.clone(), &[layout.clone()], &sidecar.path, &transport)?;
        // Fixture uses only the first complete block and four rows beyond it.
        // This is not a serving chunk or a runtime budget constant.
        let prefix = scalar_gdn_layout::NATIVE_BLOCK_ROWS - 1;
        let end = scalar_gdn_layout::NATIVE_BLOCK_ROWS + 4;
        if rows < end {
            return Err("retained fixture lacks boundary rows".into());
        }
        let prefix_shape = Layout::checked(h, kd, vd, prefix)?;
        let prefix_workspace = Workspace::new(program.clone(), &[prefix_shape])?;
        let workspace = Workspace::new(program.clone(), &[decode_shape.clone()])?;
        let mut state = make_owner(&program, &layout, &data["initial-state.f32"])?;
        let mut eager = make_owner(&program, &layout, &data["initial-state.f32"])?;
        let mut slots_data = BTreeMap::new();
        for (name, width, bytes) in [
            ("q.bf16", h * kd, 2),
            ("k.bf16", h * kd, 2),
            ("v.bf16", h * vd, 2),
            ("beta.bf16", h, 2),
            ("gate.f32", h, 4),
        ] {
            slots_data.insert(name.into(), data[name][..width * bytes].to_vec());
        }
        let slots = InputStorage::new(&d, &slots_data, h, kd, vd)?;
        let position = d.htod_sync_copy(&[prefix as i32]).map_err(fail)?;
        let eager_position = d.htod_sync_copy(&[prefix as i32]).map_err(fail)?;
        for o in [&mut state, &mut eager] {
            let lease = prefix_workspace.lease()?;
            let queued = unsafe { o.enqueue_prefill(&lease, 0, prefix, &inputs.view(0, prefix)?) };
            d.synchronize().map_err(fail)?;
            queued?;
            metadata(&d, o, prefix)?;
            compare(
                "actual native prefix outputs",
                &emitted(&d, &prefix_workspace, o, prefix)?,
                &data["actual-output.bf16"][..prefix * h * vd * 2],
            )?;
        }
        drop(prefix_workspace);
        let before = snapshot(&d, &state)?;
        retain_state(root, "before-failed-capture", &before)?;
        fill_slots(&d, &slots, &inputs, prefix, &position)?;
        d.synchronize().map_err(fail)?;
        {
            let injected =
                scalar_gdn_graph::Graph::capture(&mut state, &slots, &position, &workspace, true);
            match injected {
                Err(e) if e == "explicit CPU injection after successful captured enqueue" => {}
                Err(e) => return Err(format!("unexpected capture failure:{e}")),
                Ok(_) => return Err("CPU post-enqueue capture injection not observed".into()),
            }
        }
        d.synchronize().map_err(fail)?;
        let after = snapshot(&d, &state)?;
        for (n, b) in &before {
            compare(
                &format!("discarded capture preserves full {n}"),
                &after[n],
                b,
            )?;
        }
        if state.host_poisoned() {
            return Err("discarded unexecuted capture poisoned owner".into());
        }
        retain_state(root, "after-failed-capture", &after)?;
        let mut graph =
            scalar_gdn_graph::Graph::capture(&mut state, &slots, &position, &workspace, false)?;
        let captured = snapshot(&d, graph.owner())?;
        for (n, b) in &before {
            compare(
                &format!("successful capture before replay preserves full {n}"),
                &captured[n],
                b,
            )?;
        }
        retain_state(root, "after-successful-capture-before-replay", &captured)?;
        let node_count = graph.nodes();
        for row in prefix..end {
            fill_slots(&d, &slots, &inputs, row, &position)?;
            graph.replay()?;
            d.synchronize().map_err(fail)?;
            metadata(&d, graph.owner(), row + 1)?;
            let graph_output = emitted(&d, &workspace, graph.owner(), 1)?;
            compare(
                "graph actual native row output",
                &graph_output,
                &data["actual-output.bf16"][row * h * vd * 2..(row + 1) * h * vd * 2],
            )?;
            fs::write(
                root.join(format!("graph-output-row{row}.bf16")),
                &graph_output,
            )
            .map_err(fail)?;
            let actual = snapshot(&d, graph.owner())?;
            retain_state(root, &format!("graph-state-cursor{}", row + 1), &actual)?;
            use cudarc::driver::DevicePtr;
            unsafe {
                sys::lib().cuMemsetD32Async(
                    *eager_position.device_ptr(),
                    u32::try_from(row).map_err(fail)?,
                    1,
                    *d.cu_stream(),
                )
            }
            .result()
            .map_err(fail)?;
            let lease = workspace.lease()?;
            let queued =
                unsafe { eager.enqueue_decode(&lease, &inputs.view(row, 1)?, &eager_position) };
            d.synchronize().map_err(fail)?;
            queued?;
            metadata(&d, &eager, row + 1)?;
            let eager_output = emitted(&d, &workspace, &eager, 1)?;
            compare(
                "complete graph/eager row output",
                &eager_output,
                &graph_output,
            )?;
            fs::write(
                root.join(format!("eager-output-row{row}.bf16")),
                &eager_output,
            )
            .map_err(fail)?;
            let expected = snapshot(&d, &eager)?;
            for (n, b) in &expected {
                compare(
                    &format!("graph/eager complete {n} at{}", row + 1),
                    &actual[n],
                    b,
                )?;
            }
            retain_state(root, &format!("eager-state-cursor{}", row + 1), &expected)?;
            memory(p.safety_margin_bytes, &mut minimum)?;
        }
        drop(graph);
        // Owners and arena remain usable after graph execution resources release.
        use cudarc::driver::DevicePtr;
        unsafe {
            sys::lib().cuMemsetD32Async(
                *position.device_ptr(),
                u32::try_from(end).map_err(fail)?,
                1,
                *d.cu_stream(),
            )
        }
        .result()
        .map_err(fail)?;
        let lease = workspace.lease()?;
        let queued = unsafe { state.enqueue_decode(&lease, &inputs.view(end, 1)?, &position) };
        d.synchronize().map_err(fail)?;
        queued?;
        metadata(&d, &state, end + 1)?;
        let output = emitted(&d, &workspace, &state, 1)?;
        compare(
            "eager after graph teardown actual native output",
            &output,
            &data["actual-output.bf16"][end * h * vd * 2..(end + 1) * h * vd * 2],
        )?;
        fs::write(root.join("output-after-graph-release.bf16"), output).map_err(fail)?;
        retain_state(root, "after-graph-release-eager", &snapshot(&d, &state)?)?;
        inputs.verify_readonly(&d, &data)?;
        Ok(
            json!({"passed":true,"actual_gpu_uuid":actual,"actual_graph_nodes":node_count,"replay_first_row":prefix,"replay_last_row":end-1,"graph_replays":end-prefix,"complete_graph_outputs_match_independent_native_reference":true,"complete_state_planes_match_eager_each_replay":true,"post_enqueue_cpu_capture_failure_full_state_unchanged":true,"successful_capture_does_not_execute":true,"recapture_after_failure_passed":true,"eager_after_graph_release_passed":true,"all_original_gpu_input_bytes_readonly_exact":true,"minimum_completed_stage_free_bytes":minimum,"safety_margin_bytes":p.safety_margin_bytes,"hardware_failure_injected":false,"other_architecture_qualified":false,"installed_snapshot_error_cancel_paths_qualified":false,"serving_qualified":false,"scope":"Actual native64 graph replay with real static one-row input allocations and dynamically updated device position; captures are never substituted by eager calls. CPU failure after successful captured enqueue, not a CUDA API or hardware error. All graph/eager persistent bytes retained. No model run, Witness or performance claim."}),
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
