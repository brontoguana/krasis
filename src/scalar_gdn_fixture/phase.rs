//! Actual native stream/thread/complete-state phase gate, never a model score.
use super::owner::{file, number, read, FileId, InputStorage};
use crate::{
    cuda_region::CudaRegion,
    gpu_decode::SequenceStatePinnedBuffer,
    gpu_prefill::PrefillComputeStream,
    model_tensor_owner::ModelTensorOwner,
    scalar_gdn_layout::Layout,
    scalar_gdn_owner::LatestState,
    scalar_gdn_runtime::{ColdLayer, Storage},
    sequence_copy_completion::with_copy_completion,
    session_cache::{SequenceStateAllocation, SequenceStateGrowth, SequenceStateRegistry},
};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
fn fail(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn check(code: sys::CUresult, label: &str) -> Result<(), String> {
    code.result().map_err(|e| format!("{label}:{e}"))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema_version: u32,
    gpu_uuid: String,
    owner_plan: FileId,
    model_config: FileId,
    transport: FileId,
    sidecar: FileId,
    admission_files: Vec<FileId>,
}
pub(super) struct Prepared {
    plan: Plan,
    sidecar: PathBuf,
    transport: String,
    data: BTreeMap<String, Vec<u8>>,
    h: usize,
    k: usize,
    v: usize,
    rows: usize,
    base: usize,
    pub(super) conv_shape: Vec<usize>,
    margin: usize,
}
impl Prepared {
    pub(super) fn load(path: &Path) -> Result<Self, String> {
        let plan: Plan = serde_json::from_slice(&fs::read(path).map_err(fail)?).map_err(fail)?;
        if plan.schema_version != 1 || !plan.gpu_uuid.starts_with("GPU-") {
            return Err("invalid actual phase plan".into());
        }
        for identity in &plan.admission_files {
            read(identity)?;
        }
        let parent: Value = serde_json::from_slice(&read(&plan.owner_plan)?).map_err(fail)?;
        if parent["gpu_uuid"] != plan.gpu_uuid {
            return Err("phase/owner real GPU identity differs".into());
        }
        // Reuse the complete unchanged CPU-only input/guard/reference admission.
        super::owner::run(&plan.owner_plan.path, Path::new("."), true)?;
        let native: Value =
            serde_json::from_slice(&read(&file(&parent["native_plan"])?)?).map_err(fail)?;
        let cases = native["cases"]
            .as_array()
            .ok_or("actual native cases absent")?;
        let full = cases
            .iter()
            .find(|x| x["name"] == "full")
            .ok_or("actual full case absent")?;
        let base = cases
            .iter()
            .find(|x| x["name"] == "base")
            .ok_or("actual base case absent")?;
        let geometry = &full["geometry"];
        let (h, k, v, rows, base) = (
            number(geometry, "heads")?,
            number(geometry, "key_dim")?,
            number(geometry, "value_dim")?,
            number(geometry, "real_rows")?,
            number(&base["geometry"], "real_rows")?,
        );
        let mut data = BTreeMap::new();
        for role in [
            "q.bf16",
            "k.bf16",
            "v.bf16",
            "beta.bf16",
            "gate.f32",
            "initial-state.f32",
            "actual-output.bf16",
            "actual-final-state.f32",
        ] {
            data.insert(role.into(), read(&file(&full["files"][role])?)?);
        }
        if data["initial-state.f32"].iter().any(|&x| x != 0) {
            return Err(
                "reset gate requires actual captured zero initial state; no seed substitution"
                    .into(),
            );
        }
        let config: Value = serde_json::from_slice(&read(&plan.model_config)?).map_err(fail)?;
        let config = config.get("text_config").unwrap_or(&config);
        if number(config, "linear_num_value_heads")? != h
            || number(config, "linear_key_head_dim")? != k
            || number(config, "linear_value_head_dim")? != v
        {
            return Err("actual loaded architecture differs from retained native geometry".into());
        }
        let key_heads = number(config, "linear_num_key_heads")?;
        let conv_dim = key_heads
            .checked_mul(k)
            .and_then(|x| x.checked_mul(2))
            .and_then(|x| h.checked_mul(v).and_then(|y| x.checked_add(y)))
            .ok_or("actual convolution geometry overflow")?;
        let conv_shape = vec![conv_dim, number(config, "linear_conv_kernel_dim")?];
        let sidecar = file(&native["sidecar"])?;
        read(&sidecar)?;
        let actual_transport = read(&plan.transport)?;
        if actual_transport != include_bytes!(concat!(env!("OUT_DIR"), "/scalar_gdn_transport.ptx"))
        {
            return Err("phase transport is not the actual compiled production transport".into());
        }
        let transport = String::from_utf8(actual_transport).map_err(fail)?;
        let actual_sidecar = read(&plan.sidecar)?;
        if actual_sidecar != read(&sidecar)? {
            return Err("installed actual sidecar differs from qualified own-GPU library".into());
        }
        let sidecar_path = plan.sidecar.path.clone();
        let margin = number(&parent, "safety_margin_bytes")?;
        Ok(Self {
            plan,
            sidecar: sidecar_path,
            transport,
            data,
            h,
            k,
            v,
            rows,
            base,
            conv_shape,
            margin,
        })
    }
    pub(super) fn initial(&self) -> &[u8] {
        &self.data["initial-state.f32"]
    }
    pub(super) fn uuid(&self) -> &str {
        &self.plan.gpu_uuid
    }
    pub(super) fn admission(&self) -> Value {
        json!({"passed":true,"gpu_initialized":false,"model_loaded":false,"real_rows":self.rows,"base_rows":self.base,"heads":self.h,"key_dim":self.k,"value_dim":self.v,"actual_conv_shape":self.conv_shape,"scope":"CPU-only retained input/source/geometry admission; not a GPU result"})
    }
}
fn actual_tensor_region(
    owner: &ModelTensorOwner,
    name: &str,
    kind: &str,
    device: &Arc<CudaDevice>,
) -> Result<SequenceStateAllocation, String> {
    let (pointer, end) = owner.extent()?;
    let mut bytes = std::mem::size_of::<f32>();
    let mut strides = vec![0; owner.shape.len()];
    for i in (0..owner.shape.len()).rev() {
        strides[i] = bytes;
        bytes = bytes
            .checked_mul(owner.shape[i])
            .ok_or("actual typed tensor extent overflow")?;
    }
    if bytes as u64 != end - pointer {
        return Err("actual typed state shape/extent differs".into());
    }
    Ok(SequenceStateAllocation {
        name: name.into(),
        kind: kind.into(),
        layer_idx: Some(0),
        device_ordinal: device.ordinal(),
        ptr: pointer,
        storage_bytes: bytes,
        dtype: "float32".into(),
        element_size: 4,
        shape: owner.shape.clone(),
        strides_bytes: strides,
        growth: SequenceStateGrowth::Fixed,
    })
}
fn make_storage(
    prepared: &Prepared,
    device: &Arc<CudaDevice>,
    latest: Arc<ModelTensorOwner>,
    conv: Arc<ModelTensorOwner>,
) -> Result<(Arc<Storage>, SequenceStateRegistry), String> {
    let mut registry = SequenceStateRegistry::default();
    registry.register(actual_tensor_region(
        &latest,
        "latest-state.f32",
        "linear_attention_recurrent_state",
        device,
    )?)?;
    registry.register(actual_tensor_region(
        &conv,
        "conv-state.f32",
        "linear_attention_conv_state",
        device,
    )?)?;
    if conv.shape != prepared.conv_shape {
        return Err("actual convolution tensor shape differs from loaded configuration".into());
    }
    let stream = PrefillComputeStream::create(device.clone())?;
    let layout = Layout::checked(prepared.h, prepared.k, prepared.v, 1)?;
    let storage = Storage::new(
        device.clone(),
        Some(stream),
        vec![ColdLayer {
            layer: 0,
            layout,
            latest: LatestState::model(latest)?,
            conv,
        }],
        &prepared.sidecar,
        &prepared.transport,
    )?;
    for region in storage.registry().allocations() {
        registry.register(region.clone())?;
    }
    Ok((storage, registry))
}
struct Snapshot {
    // Real pinned host storage used by the same production copy guard.
    host: SequenceStatePinnedBuffer,
    regions: Vec<(SequenceStateAllocation, usize)>,
    cursor: usize,
    device: Arc<CudaDevice>,
}
impl Snapshot {
    fn capture(
        storage: &Storage,
        registry: &SequenceStateRegistry,
        cursor: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Self, String> {
        let mut bytes = 0usize;
        let mut regions = Vec::new();
        for region in registry.allocations() {
            let offset = bytes;
            bytes = bytes
                .checked_add(region.storage_bytes)
                .ok_or("actual full state snapshot overflow")?;
            regions.push((region.clone(), offset));
        }
        let snapshot = Self {
            host: SequenceStatePinnedBuffer::new(bytes)?,
            regions,
            cursor,
            device: device.clone(),
        };
        unsafe {
            storage.quiescent(cursor, || {
                with_copy_completion(device, *device.cu_stream(), &(storage, &snapshot), |_| {
                    for (region, offset) in &snapshot.regions {
                        check(
                            sys::lib().cuMemcpyDtoHAsync_v2(
                                snapshot.host.ptr.add(*offset).cast(),
                                region.ptr,
                                region.storage_bytes,
                                *device.cu_stream(),
                            ),
                            "phase snapshot actual D2H",
                        )?;
                    }
                    Ok(())
                })
            })?;
        }
        Ok(snapshot)
    }
    fn plane(&self, name: &str) -> Result<&[u8], String> {
        let (region, offset) = self
            .regions
            .iter()
            .find(|(r, _)| r.name == name)
            .ok_or_else(|| format!("actual snapshot role missing:{name}"))?;
        Ok(unsafe { std::slice::from_raw_parts(self.host.ptr.add(*offset), region.storage_bytes) })
    }
    fn restore(
        &self,
        storage: &Storage,
        registry: &SequenceStateRegistry,
        fail_after: Option<usize>,
    ) -> Result<(), String> {
        if self.regions.len() != registry.allocations().len()
            || self
                .regions
                .iter()
                .any(|(a, _)| !registry.allocations().contains(a))
        {
            return Err("restore actual region inventory/source identity differs".into());
        }
        unsafe {
            storage.restore(self.cursor, || {
                with_copy_completion(
                    &self.device,
                    *self.device.cu_stream(),
                    &(self, storage, registry),
                    |_| {
                        for (index, (region, offset)) in self.regions.iter().enumerate() {
                            check(
                                sys::lib().cuMemcpyHtoDAsync_v2(
                                    region.ptr,
                                    self.host.ptr.add(*offset).cast(),
                                    region.storage_bytes,
                                    *self.device.cu_stream(),
                                ),
                                "phase restore actual H2D",
                            )?;
                            if fail_after == Some(index + 1) {
                                return Err("explicit error after actual restore enqueue".into());
                            }
                        }
                        Ok(())
                    },
                )
            })
        }
    }
    fn save(&self, root: &Path) -> Result<(), String> {
        fs::create_dir(root).map_err(fail)?;
        let mut rows = Vec::new();
        for (index, (region, _)) in self.regions.iter().enumerate() {
            let raw = self.plane(&region.name)?;
            let path = root.join(format!("{index:03}.bin"));
            fs::write(&path, raw).map_err(fail)?;
            rows.push(json!({"allocation":region,"path":path,"bytes":raw.len(),"sha256":format!("{:x}",Sha256::digest(raw))}));
        }
        fs::write(
            root.join("STATE.json"),
            serde_json::to_vec_pretty(&json!({"cursor":self.cursor,"regions":rows}))
                .map_err(fail)?,
        )
        .map_err(fail)
    }
}
fn compare(label: &str, actual: &[u8], expected: &[u8]) -> Result<(), String> {
    if actual != expected {
        let first = actual.iter().zip(expected).position(|(a, b)| a != b);
        return Err(format!(
            "phase {label} bytes differ: first={first:?} actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(())
}
fn same_state(actual: &Snapshot, expected: &Snapshot) -> Result<(), String> {
    if actual.cursor != expected.cursor {
        return Err("phase state cursor differs".into());
    }
    for (region, _) in &actual.regions {
        let (a, b) = (actual.plane(&region.name)?, expected.plane(&region.name)?);
        if region.name.ends_with("metadata.u64") {
            // Epoch and completed-work geometry describe different histories;
            // live cursor/pending/poison/inflight/reserved are state contracts.
            for index in [0usize, 1, 3, 4, 7] {
                compare(
                    "live metadata",
                    &a[index * 8..index * 8 + 8],
                    &b[index * 8..index * 8 + 8],
                )?;
            }
        } else {
            compare(&region.name, a, b)?;
        }
    }
    Ok(())
}
fn completed_output(
    storage: &Storage,
    device: &Arc<CudaDevice>,
    cursor: usize,
    pointer: u64,
    bytes: usize,
) -> Result<Vec<u8>, String> {
    unsafe {
        storage.quiescent(cursor, || {
            let mut output = vec![0u8; bytes];
            check(
                sys::lib().cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), pointer, bytes),
                "phase output D2H",
            )?;
            Ok(output)
        })
    }
}
fn require_poison(storage: &Storage, cursor: usize) -> Result<(), String> {
    let mut entered = false;
    if unsafe {
        storage.with_decode(|| {
            entered = true;
            Ok(())
        })
    }
    .is_ok()
        || entered
    {
        return Err("poisoned execution entered callback".into());
    }
    if unsafe { storage.quiescent(cursor, || Ok(())) }.is_ok() {
        return Err("poisoned state was snapshot eligible".into());
    }
    Ok(())
}

struct CaptureEnd<'a> {
    graph: &'a mut sys::CUgraph,
    device: Arc<CudaDevice>,
    active: bool,
}
impl CaptureEnd<'_> {
    fn finish(&mut self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(fail)?;
        let result = check(
            unsafe { sys::lib().cuStreamEndCapture(*self.device.cu_stream(), self.graph) },
            "phase graph capture end",
        );
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        if let Err(e) = check(
            unsafe { sys::lib().cuStreamIsCapturing(*self.device.cu_stream(), &mut status) },
            "phase capture cleanup status",
        ) {
            crate::vram_monitor::fatal_cuda_context_error("phase capture completion unknown", &e);
        }
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            crate::vram_monitor::fatal_cuda_context_error(
                "phase capture remains active",
                "cannot release captured owners",
            );
        }
        self.active = false;
        result
    }
}
impl Drop for CaptureEnd<'_> {
    fn drop(&mut self) {
        if self.active {
            if let Err(e) = self.finish() {
                eprintln!("phase failed callback capture ended with error:{e}");
            }
        }
    }
}
struct Captured {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    output: u64,
    nodes: usize,
    // These actual owners outlive graph execution/destruction, including errors.
    storage: Arc<Storage>,
    slots: Arc<InputStorage>,
    position: Arc<CudaSlice<i32>>,
    device: Arc<CudaDevice>,
}
impl Captured {
    fn new(
        storage: Arc<Storage>,
        slots: Arc<InputStorage>,
        position: Arc<CudaSlice<i32>>,
        device: Arc<CudaDevice>,
        inject: bool,
    ) -> Result<Self, String> {
        let mut graph = Self {
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
            output: 0,
            nodes: 0,
            storage: storage.clone(),
            slots,
            position,
            device: device.clone(),
        };
        unsafe {
            storage.with_decode(|| {
                check(
                    sys::lib().cuStreamBeginCapture_v2(
                        *device.cu_stream(),
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    ),
                    "phase graph capture begin",
                )?;
                let mut end = CaptureEnd {
                    graph: &mut graph.graph,
                    device: device.clone(),
                    active: true,
                };
                let result = graph
                    .storage
                    .decode_output(0, &graph.slots.view(0, 1)?, &graph.position)
                    .map(|pointer| {
                        graph.output = pointer;
                    });
                let completed = end.finish();
                result?;
                completed?;
                if inject {
                    return Err("explicit CPU error after successful phase capture enqueue".into());
                }
                Ok(())
            })?;
        }
        check(
            unsafe {
                sys::lib().cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut graph.nodes)
            },
            "phase graph real nodes",
        )?;
        if graph.nodes == 0 {
            return Err("phase capture contains no actual nodes".into());
        }
        check(
            unsafe { sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec, graph.graph, 0) },
            "phase graph instantiate",
        )?;
        Ok(graph)
    }
    fn replay(&self) -> Result<u64, String> {
        unsafe {
            self.storage.with_decode(|| {
                check(
                    sys::lib().cuGraphLaunch(self.exec, *self.device.cu_stream()),
                    "phase actual graph replay",
                )
            })?;
        }
        Ok(self.output)
    }
}
impl Drop for Captured {
    fn drop(&mut self) {
        if let Err(e) = self
            .device
            .bind_to_thread()
            .and_then(|_| self.device.synchronize())
        {
            crate::vram_monitor::fatal_cuda_context_error(
                "phase graph queued completion",
                &e.to_string(),
            );
        }
        for (pointer, exec) in [
            (self.exec as *mut std::ffi::c_void, true),
            (self.graph as *mut std::ffi::c_void, false),
        ] {
            if !pointer.is_null() {
                let code = unsafe {
                    if exec {
                        sys::lib().cuGraphExecDestroy(pointer.cast())
                    } else {
                        sys::lib().cuGraphDestroy(pointer.cast())
                    }
                };
                if let Err(e) = code.result() {
                    crate::vram_monitor::fatal_cuda_context_error(
                        "phase graph resource release",
                        &e.to_string(),
                    );
                }
            }
        }
    }
}
fn memory(prepared: &Prepared, minimum: &mut usize) -> Result<(), String> {
    let (free, _) = cudarc::driver::result::mem_get_info().map_err(fail)?;
    *minimum = (*minimum).min(free);
    if free < prepared.margin {
        return Err("phase actual free VRAM below admitted safety margin".into());
    }
    Ok(())
}
fn record_output(root: &Path, name: &str, actual: &[u8], expected: &[u8]) -> Result<(), String> {
    fs::write(root.join(name), actual).map_err(fail)?;
    compare(name, actual, expected)
}
pub(super) fn run(
    prepared: Prepared,
    root: &Path,
    device: Arc<CudaDevice>,
    pairs: Vec<(Arc<ModelTensorOwner>, Arc<ModelTensorOwner>)>,
) -> Result<Value, String> {
    if root.exists() {
        return Err("fresh actual phase output directory required".into());
    }
    fs::create_dir(root).map_err(fail)?;
    device.bind_to_thread().map_err(fail)?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *device.cu_device()) },
        "phase actual GPU UUID",
    )?;
    let hex = uuid
        .bytes
        .iter()
        .map(|&b| format!("{:02x}", b as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    if actual != prepared.plan.gpu_uuid {
        return Err("phase physical device differs from CPU admission".into());
    }
    if pairs.len() != 2 {
        return Err("actual two-family phase control tensor cardinality differs".into());
    }
    let mut pairs = pairs.into_iter();
    let (latest, conv) = pairs.next().ok_or("phase first real state absent")?;
    let (storage, registry) = make_storage(&prepared, &device, latest, conv)?;
    let inputs = Arc::new(InputStorage::new(
        &device,
        &prepared.data,
        prepared.h,
        prepared.k,
        prepared.v,
    )?);
    let row_bytes = prepared
        .h
        .checked_mul(prepared.v)
        .and_then(|x| x.checked_mul(2))
        .ok_or("actual output row extent overflow")?;
    let mut minimum = usize::MAX;
    memory(&prepared, &mut minimum)?;
    storage.reset()?;
    let pointer = unsafe {
        storage.with_prefill(|| {
            storage.prefill_output(0, 0, prepared.rows, &inputs.view(0, prepared.rows)?)
        })?
    };
    let output = completed_output(
        &storage,
        &device,
        prepared.rows,
        pointer,
        prepared.rows * row_bytes,
    )?;
    record_output(
        root,
        "full-output.bf16",
        &output,
        &prepared.data["actual-output.bf16"],
    )?;
    let full = Snapshot::capture(&storage, &registry, prepared.rows, &device)?;
    full.save(&root.join("full-state"))?;
    compare(
        "actual native full latest",
        full.plane("latest-state.f32")?,
        &prepared.data["actual-final-state.f32"],
    )?;
    storage.reset()?;
    unsafe {
        storage.with_prefill(|| {
            storage.prefill_output(0, 0, prepared.base, &inputs.view(0, prepared.base)?)
        })?;
    }
    let mut base = Snapshot::capture(&storage, &registry, prepared.base, &device)?;
    base.save(&root.join("base-state"))?;
    // Requeue actual prefill and transfer the retained native program/real
    // stream/state/input owners to another OS thread without host completion.
    storage.reset()?;
    unsafe {
        storage.with_prefill(|| {
            storage.prefill_output(0, 0, prepared.base, &inputs.view(0, prepared.base)?)
        })?;
    }
    let prepared = Arc::new(prepared);
    let thread_prepared = prepared.clone();
    let thread_device = device.clone();
    let thread_inputs = inputs.clone();
    let thread_root = root.to_path_buf();
    let (storage, graph_nodes, thread_minimum) =
        std::thread::spawn(move || -> Result<_, String> {
            thread_device.bind_to_thread().map_err(fail)?;
            let p = thread_prepared;
            let d = thread_device;
            let source = thread_inputs;
            let root = thread_root;
            let mut minimum = usize::MAX;
            let position = d.htod_sync_copy(&[p.base as i32]).map_err(fail)?;
            let mut eager_output = Vec::new();
            for row in p.base..p.rows {
                unsafe {
                    check(
                        sys::lib().cuMemsetD32Async(
                            *position.device_ptr(),
                            u32::try_from(row).map_err(fail)?,
                            1,
                            *d.cu_stream(),
                        ),
                        "phase actual decode cursor",
                    )?;
                }
                let pointer = unsafe {
                    storage.with_decode(|| {
                        storage.decode_output(0, &source.view(row, 1)?, &position)
                    })?
                };
                eager_output.extend(completed_output(&storage, &d, row + 1, pointer, row_bytes)?);
            }
            record_output(
                &root,
                "cross-thread-eager-suffix.bf16",
                &eager_output,
                &p.data["actual-output.bf16"][p.base * row_bytes..],
            )?;
            let actual = Snapshot::capture(&storage, &registry, p.rows, &d)?;
            actual.save(&root.join("cross-thread-eager-state"))?;
            same_state(&actual, &full)?;
            // Restore EVERY real fixed region, then cross decode -> prefill while
            // the actual first decode row is still queued on its distinct stream.
            base.restore(&storage, &registry, None)?;
            unsafe {
                check(
                    sys::lib().cuMemsetD32Async(
                        *position.device_ptr(),
                        p.base as u32,
                        1,
                        *d.cu_stream(),
                    ),
                    "phase restored actual cursor",
                )?;
            }
            let first = unsafe {
                storage
                    .with_decode(|| storage.decode_output(0, &source.view(p.base, 1)?, &position))?
            };
            let suffix = unsafe {
                storage.with_prefill(|| {
                    storage.prefill_output(
                        0,
                        (p.base + 1) as u64,
                        p.rows - p.base - 1,
                        &source.view(p.base + 1, p.rows - p.base - 1)?,
                    )
                })?
            };
            let mut mixed = completed_output(&storage, &d, p.rows, first, row_bytes)?;
            mixed.extend(completed_output(
                &storage,
                &d,
                p.rows,
                suffix,
                (p.rows - p.base - 1) * row_bytes,
            )?);
            record_output(
                &root,
                "decode-then-prefill-suffix.bf16",
                &mixed,
                &p.data["actual-output.bf16"][p.base * row_bytes..],
            )?;
            let actual = Snapshot::capture(&storage, &registry, p.rows, &d)?;
            actual.save(&root.join("mixed-phase-state"))?;
            same_state(&actual, &full)?;
            if base.restore(&storage, &registry, Some(1)).unwrap_err()
                != "explicit error after actual restore enqueue"
            {
                return Err("partial actual restore injection changed".into());
            }
            require_poison(&storage, p.base)?;
            base.restore(&storage, &registry, None)?;
            let actual = Snapshot::capture(&storage, &registry, p.base, &d)?;
            for (region, _) in &base.regions {
                compare(
                    "complete restore",
                    actual.plane(&region.name)?,
                    base.plane(&region.name)?,
                )?;
            }
            // Device-side poison/inflight are invalid even after a complete real
            // copy. Alter only explicit negative-control metadata, then restore the
            // retained bytes before the next valid recovery; no golden is changed.
            let meta_offset = base
                .regions
                .iter()
                .find(|(r, _)| r.name.ends_with("metadata.u64"))
                .ok_or("actual metadata region absent")?
                .1;
            for index in [3usize, 4] {
                let mut original = [0u8; 8];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        base.host.ptr.add(meta_offset + index * 8),
                        original.as_mut_ptr(),
                        8,
                    );
                    std::ptr::copy_nonoverlapping(
                        1u64.to_le_bytes().as_ptr(),
                        base.host.ptr.add(meta_offset + index * 8),
                        8,
                    );
                }
                let rejected = base
                    .restore(&storage, &registry, None)
                    .err()
                    .ok_or("invalid actual restored metadata accepted")?;
                if !rejected.contains(
                    "restored scalar state cursor/pending/poison/inflight/reserved invalid",
                ) {
                    return Err(rejected);
                }
                require_poison(&storage, p.base)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        original.as_ptr(),
                        base.host.ptr.add(meta_offset + index * 8),
                        8,
                    );
                }
                base.restore(&storage, &registry, None)?;
            }
            base.cursor += 1;
            let rejected = base
                .restore(&storage, &registry, None)
                .err()
                .ok_or("wrong restored consumed cursor accepted")?;
            if !rejected
                .contains("restored scalar state cursor/pending/poison/inflight/reserved invalid")
            {
                return Err(rejected);
            }
            base.cursor -= 1;
            require_poison(&storage, p.base)?;
            base.restore(&storage, &registry, None)?;
            // Actual post-enqueue CPU failure must drain and reject subsequent
            // computation/snapshot; it is not a hardware fault or successful token.
            let error = unsafe {
                storage.with_decode(|| {
                    storage.decode_output(0, &source.view(p.base, 1)?, &position)?;
                    Err::<(), String>("explicit CPU error after actual phase kernel enqueue".into())
                })
            }
            .unwrap_err();
            if error != "explicit CPU error after actual phase kernel enqueue" {
                return Err(error);
            }
            require_poison(&storage, p.base)?;
            base.restore(&storage, &registry, None)?;
            let bt = crate::scalar_gdn_layout::NATIVE_BLOCK_ROWS;
            let prefix = bt - 1;
            let replay_count = 5;
            let end = prefix + replay_count;
            storage.reset()?;
            unsafe {
                storage.with_prefill(|| {
                    storage.prefill_output(0, 0, prefix, &source.view(0, prefix)?)
                })?;
            }
            let before = Snapshot::capture(&storage, &registry, prefix, &d)?;
            let slot_data = BTreeMap::from([
                ("q.bf16".into(), p.data["q.bf16"][..p.h * p.k * 2].to_vec()),
                ("k.bf16".into(), p.data["k.bf16"][..p.h * p.k * 2].to_vec()),
                ("v.bf16".into(), p.data["v.bf16"][..p.h * p.v * 2].to_vec()),
                ("beta.bf16".into(), p.data["beta.bf16"][..p.h * 2].to_vec()),
                ("gate.f32".into(), p.data["gate.f32"][..p.h * 4].to_vec()),
            ]);
            let slots = Arc::new(InputStorage::new(&d, &slot_data, p.h, p.k, p.v)?);
            let graph_position = Arc::new(d.htod_sync_copy(&[prefix as i32]).map_err(fail)?);
            unsafe {
                source.stage_row(&d, &slots, prefix, &graph_position)?;
            }
            let failed = Captured::new(
                storage.clone(),
                slots.clone(),
                graph_position.clone(),
                d.clone(),
                true,
            );
            if failed.err().as_deref()
                != Some("explicit CPU error after successful phase capture enqueue")
            {
                return Err("post-enqueue captured CPU failure missing".into());
            }
            require_poison(&storage, prefix)?;
            before.restore(&storage, &registry, None)?;
            let graph = Captured::new(
                storage.clone(),
                slots.clone(),
                graph_position.clone(),
                d.clone(),
                false,
            )?;
            let untouched = Snapshot::capture(&storage, &registry, prefix, &d)?;
            for (region, _) in &before.regions {
                compare(
                    "capture alone must not execute",
                    untouched.plane(&region.name)?,
                    before.plane(&region.name)?,
                )?;
            }
            let mut graph_output = Vec::new();
            for row in prefix..end {
                unsafe {
                    source.stage_row(&d, &slots, row, &graph_position)?;
                }
                let pointer = graph.replay()?;
                graph_output.extend(completed_output(&storage, &d, row + 1, pointer, row_bytes)?);
            }
            record_output(
                &root,
                "actual-phase-graph-output.bf16",
                &graph_output,
                &p.data["actual-output.bf16"][prefix * row_bytes..end * row_bytes],
            )?;
            let graph_state = Snapshot::capture(&storage, &registry, end, &d)?;
            graph_state.save(&root.join("graph-state"))?;
            let nodes = graph.nodes;
            drop(graph);
            storage.reset()?;
            unsafe {
                storage
                    .with_prefill(|| storage.prefill_output(0, 0, end, &source.view(0, end)?))?;
            }
            let direct = Snapshot::capture(&storage, &registry, end, &d)?;
            direct.save(&root.join("direct-graph-reference-state"))?;
            same_state(&graph_state, &direct)?;
            revision205_controls(&p,&storage,&registry,&source,&base,&full,&d,&root)?;
            source.verify_readonly(&d, &p.data)?;
            memory(&p, &mut minimum)?;
            Ok((storage, nodes, minimum))
        })
        .join()
        .map_err(|_| "phase worker unexpectedly panicked")??;
    minimum = minimum.min(thread_minimum);
    drop(storage);
    // Separate whole-family unwind control. The actual queued work must finish
    // before Rust resumes the panic, while all real tensor owners remain alive.
    let (latest, conv) = pairs.next().ok_or("phase unwind real state absent")?;
    let (unwind_storage, unwind_registry) = make_storage(&prepared, &device, latest, conv)?;
    unwind_storage.reset()?;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        unwind_storage.with_prefill::<()>(|| {
            unwind_storage.prefill_output(
                0,
                0,
                crate::scalar_gdn_layout::NATIVE_BLOCK_ROWS - 1,
                &inputs.view(0, crate::scalar_gdn_layout::NATIVE_BLOCK_ROWS - 1)?,
            )?;
            panic!("explicit CPU unwind after actual phase enqueue");
        })
    }));
    if panic.is_ok() {
        return Err("actual post-enqueue unwind control did not panic".into());
    }
    require_poison(&unwind_storage, 0)?;
    unwind_storage.reset()?;
    let prefix = crate::scalar_gdn_layout::NATIVE_BLOCK_ROWS - 1;
    let pointer = unsafe {
        unwind_storage.with_prefill(|| {
            unwind_storage.prefill_output(0, 0, prefix, &inputs.view(0, prefix)?)
        })?
    };
    let recovered = completed_output(
        &unwind_storage,
        &device,
        prefix,
        pointer,
        prefix * row_bytes,
    )?;
    record_output(
        root,
        "after-unwind-reset-output.bf16",
        &recovered,
        &prepared.data["actual-output.bf16"][..prefix * row_bytes],
    )?;
    Snapshot::capture(&unwind_storage, &unwind_registry, prefix, &device)?
        .save(&root.join("after-unwind-reset-state"))?;
    drop(unwind_storage);
    inputs.verify_readonly(&device, &prepared.data)?;
    device.synchronize().map_err(fail)?;
    let report = json!({"passed":true,"actual_gpu_uuid":actual,"actual_prefill_decode_distinct_streams":true,"actual_native_owner_cross_thread_transfer":true,"full_and_split_outputs_exact":true,"complete_registered_state_restored":true,"post_enqueue_error_and_partial_restore_poison_rejected":true,"post_enqueue_unwind_completed_and_reset_recovered":true,"successful_capture_does_not_execute":true,"capture_failure_rejected_then_restored":true,"actual_graph_nodes":graph_nodes,"actual_graph_replays":5,"graph_output_matches_retained_reference":true,"complete_graph_state_matches_direct_prefill":true,"all_original_inputs_readonly_exact":true,"minimum_completed_stage_free_bytes":minimum,"safety_margin_bytes":prepared.margin,"model_loaded":false,"ordinary_serving_qualified":false,"scope":"Actual193Storage/Execution with typed Torch state, owned distinct compute streams, receiving-thread binding, complete pinned-host transfers and original copy-completion guard, error/unwind/restore/reset and real graph replay. Retained prepared operands only; does not qualify actual model producers, Store/session lifecycle, multiGPU, Windows, Witness, performance or ADQ."});
    fs::write(
        root.join("REPORT.json"),
        serde_json::to_vec_pretty(&report).map_err(fail)?,
    )
    .map_err(fail)?;
    Ok(report)
}

include!("phase205.rs");
