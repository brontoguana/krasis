//! Explicit diagnostic route substitution and consumer evidence.
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, io::Read, path::PathBuf, sync::Arc};
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedFile {
    path: PathBuf,
    sha256: String,
}
impl CheckedFile {
    fn bytes(&self) -> Result<Vec<u8>, String> {
        self.verify()?;
        std::fs::read(&self.path).map_err(|e| e.to_string())
    }
    fn verify(&self) -> Result<(), String> {
        if !self.path.is_absolute()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid route file identity".into());
        }
        let mut f = std::fs::File::open(&self.path).map_err(|e| e.to_string())?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = f.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        if format!("{:x}", h.finalize()) != self.sha256 {
            return Err(format!("route checksum mismatch {}", self.path.display()));
        }
        Ok(())
    }
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    layer: usize,
    position: usize,
    num_experts: usize,
    topk: usize,
    mask: u32,
    ids: CheckedFile,
    weights: CheckedFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Trial {
    label: String,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: u32,
    directory: PathBuf,
    loaded_native: CheckedFile,
    identities: Vec<CheckedFile>,
    ptx: CheckedFile,
    trials: Vec<Trial>,
}
struct LoadedRow {
    descriptor: Row,
    ids: Vec<i32>,
    weights: Vec<f32>,
}
struct LoadedTrial {
    label: String,
    rows: Vec<LoadedRow>,
}
pub(crate) struct Experiment {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
    active: CudaSlice<u32>,
    entries: CudaSlice<u32>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    layers: usize,
    trials: Vec<LoadedTrial>,
    directory: PathBuf,
    active_trial: Option<usize>,
    armed_label: Option<String>,
}
fn checked_row(r: Row, layers: usize) -> Result<LoadedRow, String> {
    if r.layer >= layers
        || r.topk == 0
        || r.num_experts < r.topk
        || r.topk > i32::MAX as usize
        || r.num_experts > i32::MAX as usize
        || r.position > i32::MAX as usize
        || !(1..=3).contains(&r.mask)
    {
        return Err("invalid route target geometry/mask".into());
    }
    let ids = r.ids.bytes()?;
    let weights = r.weights.bytes()?;
    let bytes = r.topk.checked_mul(4).ok_or("route size overflow")?;
    if ids.len() != bytes || weights.len() != bytes {
        return Err("route tensor sizes do not match descriptor".into());
    }
    let ids: Vec<i32> = ids
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let weights: Vec<f32> = weights
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let mut seen = HashSet::new();
    if ids
        .iter()
        .any(|&i| i < 0 || i as usize >= r.num_experts || !seen.insert(i))
        || weights.iter().any(|w| !w.is_finite() || *w < 0.)
    {
        return Err("invalid substituted route IDs/weights".into());
    }
    Ok(LoadedRow {
        descriptor: r,
        ids,
        weights,
    })
}
impl Experiment {
    pub(crate) fn load(device: Arc<CudaDevice>, layers: usize) -> Result<Option<Self>, String> {
        let path = match std::env::var_os("KRASIS_MATCHED_ROUTES_CONFIG") {
            None => return Ok(None),
            Some(p) => p,
        };
        let p: Plan = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if p.schema != 1
            || !p.directory.is_absolute()
            || p.trials.is_empty()
            || p.identities.is_empty()
            || layers == 0
            || layers > u32::MAX as usize
            || p.trials.len() > u32::MAX as usize
        {
            return Err("invalid route plan contract".into());
        }
        p.loaded_native.verify()?;
        let maps = std::fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
        let native = p
            .loaded_native
            .path
            .to_str()
            .ok_or("non-UTF8 native path")?;
        if !maps
            .lines()
            .any(|l| l.split_whitespace().last() == Some(native))
        {
            return Err("route native identity is not mapped".into());
        }
        for f in &p.identities {
            f.verify()?
        }
        let ptx = String::from_utf8(p.ptx.bytes()?).map_err(|e| e.to_string())?;
        let mut labels = HashSet::new();
        let mut trials = Vec::new();
        for t in p.trials {
            if t.label.is_empty()
                || t.label.len() + "matched_replay_".len() > 48
                || !t
                    .label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                || !labels.insert(t.label.clone())
                || t.rows.is_empty()
            {
                return Err("invalid/duplicate routing trial".into());
            }
            let mut seen = HashSet::new();
            let mut rows = Vec::new();
            for r in t.rows {
                if !seen.insert(r.layer) {
                    return Err(
                        "bounded route diagnostic admits one position per layer per trial".into(),
                    );
                }
                rows.push(checked_row(r, layers)?)
            }
            trials.push(LoadedTrial {
                label: t.label,
                rows,
            });
        }
        let count = trials
            .len()
            .checked_mul(layers)
            .and_then(|x| x.checked_mul(4))
            .ok_or("route entry overflow")?;
        if count > u32::MAX as usize {
            return Err("route entries exceed kernel index ABI".into());
        }
        let mut entries = vec![0u32; count];
        let mut ids = Vec::new();
        let mut weights = Vec::new();
        for (i, t) in trials.iter().enumerate() {
            for r in &t.rows {
                let offset = u32::try_from(ids.len()).map_err(|_| "route offset exceeds u32")?;
                ids.extend_from_slice(&r.ids);
                weights.extend_from_slice(&r.weights);
                if ids.len() > u32::MAX as usize {
                    return Err("route payload exceeds kernel index ABI".into());
                }
                let at = (i * layers + r.descriptor.layer) * 4;
                entries[at..at + 4].copy_from_slice(&[
                    r.descriptor.position as u32,
                    offset,
                    r.descriptor.topk as u32,
                    r.descriptor.mask,
                ]);
            }
        }
        device
            .load_ptx(ptx.into(), "matched_routes", &["matched_route_override"])
            .map_err(|e| format!("load route module: {e:?}"))?;
        let kernel = device
            .get_func("matched_routes", "matched_route_override")
            .ok_or("missing route kernel")?;
        let active = device
            .htod_sync_copy(&[0u32])
            .map_err(|e| format!("route selector: {e:?}"))?;
        let entries = device
            .htod_sync_copy(&entries)
            .map_err(|e| format!("route entries: {e:?}"))?;
        let ids = device
            .htod_sync_copy(&ids)
            .map_err(|e| format!("route IDs: {e:?}"))?;
        let weights = device
            .htod_sync_copy(&weights)
            .map_err(|e| format!("route weights: {e:?}"))?;
        let bytes = active.len() * 4 + entries.len() * 4 + ids.len() * 4 + weights.len() * 4;
        log::info!("Explicit route diagnostic staging: bytes={} layers={} trials={}; allocated before calibration",bytes,layers,trials.len());
        Ok(Some(Self {
            device,
            kernel,
            active,
            entries,
            ids,
            weights,
            layers,
            trials,
            directory: p.directory,
            active_trial: None,
            armed_label: None,
        }))
    }
    pub(crate) fn validate_layer(
        &self,
        layer: usize,
        experts: usize,
        topk: usize,
    ) -> Result<(), String> {
        for t in &self.trials {
            for r in t.rows.iter().filter(|r| r.descriptor.layer == layer) {
                if r.descriptor.num_experts != experts || r.descriptor.topk != topk {
                    return Err(format!("route loaded descriptor mismatch layer{layer}"));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn target_layers(&self) -> HashSet<usize> {
        self.trials
            .iter()
            .flat_map(|t| t.rows.iter().map(|r| r.descriptor.layer))
            .collect()
    }
    pub(crate) fn arm(&mut self, label: &str) -> Result<(), String> {
        if self.armed_label.as_deref() == Some(label) {
            return Ok(());
        }
        let trial = label
            .strip_prefix("matched_replay_")
            .and_then(|l| self.trials.iter().position(|t| t.label == l));
        let selector = trial.map_or(0, |i| i + 1);
        let value = u32::try_from(selector).map_err(|_| "route selector overflow")?;
        let err = unsafe {
            cudarc::driver::sys::lib().cuMemcpyHtoD_v2(
                *self.active.device_ptr(),
                &value as *const u32 as *const std::ffi::c_void,
                4,
            )
        };
        if err != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("route arm: {err:?}"));
        }
        self.active_trial = trial;
        self.armed_label = Some(label.to_owned());
        Ok(())
    }
    pub(crate) fn capture_node(
        &self,
        layer: usize,
        ids: u64,
        weights: u64,
        position: u64,
    ) -> Result<(), String> {
        if layer >= self.layers || ids == 0 || weights == 0 || position == 0 {
            return Err("invalid captured route destination".into());
        }
        unsafe {
            self.kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        ids,
                        weights,
                        position,
                        *self.active.device_ptr(),
                        *self.entries.device_ptr(),
                        *self.ids.device_ptr(),
                        *self.weights.device_ptr(),
                        layer as u32,
                        self.layers as u32,
                    ),
                )
                .map_err(|e| format!("capture route override: {e:?}"))?
        }
        Ok(())
    }
    /// Native host classification plus split hot/cold execution. No execution mode changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn legacy_consumers(
        &self,
        layer: usize,
        position: usize,
        phase: &str,
        stream: cudarc::driver::sys::CUstream,
        route_ids: u64,
        route_weights: u64,
        upload_ptr: u64,
        max_ept: usize,
        hot_weights: u64,
        full_weights: u64,
        hot_flags: &[u8],
        expected_ptrs: &[[u64; 4]],
        host_experts: &[[(usize, usize); 4]],
    ) -> Result<(), String> {
        let Some(row) = self.row(layer, position) else {
            return Ok(());
        };
        if !["pre_hot", "pre_cold"].contains(&phase) {
            return Err("invalid consumer capture phase".into());
        }
        self.device
            .bind_to_thread()
            .map_err(|e| format!("route capture context: {e:?}"))?;
        let topk = row.descriptor.topk;
        if topk > max_ept
            || hot_flags.len() != topk
            || expected_ptrs.len() != topk
            || host_experts.len() != row.descriptor.num_experts
        {
            return Err("legacy route consumer geometry mismatch".into());
        }
        let rc = unsafe { cudarc::driver::sys::lib().cuStreamSynchronize(stream) };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("legacy consumer fence: {rc:?}"));
        }
        let read = |ptr: u64, bytes: usize| -> Result<Vec<u8>, String> {
            if ptr == 0 || bytes == 0 || ptr.checked_add(bytes as u64).is_none() {
                return Err("invalid legacy consumer range".into());
            }
            let mut raw = vec![0u8; bytes];
            let rc = unsafe {
                cudarc::driver::sys::lib().cuMemcpyDtoH_v2(raw.as_mut_ptr().cast(), ptr, bytes)
            };
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!("legacy consumer download: {rc:?}"));
            }
            Ok(raw)
        };
        let ids_raw = read(route_ids, topk * 4)?;
        let weights_raw = read(route_weights, topk * 4)?;
        let upload = read(
            upload_ptr,
            max_ept.checked_mul(48).ok_or("upload overflow")?,
        )?;
        let hot = read(hot_weights, topk * 4)?;
        let full = read(full_weights, topk * 4)?;
        let ids: Vec<i32> = ids_raw
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let weights: Vec<u32> = weights_raw
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let mut unique = HashSet::new();
        if ids
            .iter()
            .any(|&id| id < 0 || id as usize >= host_experts.len() || !unique.insert(id))
            || weights
                .iter()
                .any(|&v| !f32::from_bits(v).is_finite() || f32::from_bits(v) < 0.)
        {
            return Err("invalid actual legacy routing".into());
        }
        if row.descriptor.mask & 1 != 0 && ids != row.ids {
            return Err("legacy IDs did not consume substitution".into());
        }
        if row.descriptor.mask & 2 != 0
            && weights
                .iter()
                .copied()
                .ne(row.weights.iter().map(|w| w.to_bits()))
        {
            return Err("legacy weights did not consume substitution".into());
        }
        let directory = self
            .evidence_directory(layer, position)?
            .ok_or("missing legacy evidence target")?
            .join(phase);
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let write = |name: &str, raw: &[u8]| -> Result<(), String> {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(name))
                .map_err(|e| e.to_string())?;
            file.write_all(raw)
                .and_then(|_| file.sync_all())
                .map_err(|e| e.to_string())
        };
        let mut slots = Vec::new();
        for slot in 0..topk {
            if hot_flags[slot] > 1 {
                return Err("invalid legacy hot flag".into());
            }
            let hot_slot = hot_flags[slot] != 0;
            let classified_id = i32::from_le_bytes(
                upload[max_ept * 44 + slot * 4..max_ept * 44 + slot * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let ordinary = u32::from_le_bytes(
                upload[max_ept * 32 + slot * 4..max_ept * 32 + slot * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let hot_bits = u32::from_le_bytes(hot[slot * 4..slot * 4 + 4].try_into().unwrap());
            let full_bits = u32::from_le_bytes(full[slot * 4..slot * 4 + 4].try_into().unwrap());
            if classified_id != ids[slot]
                || ordinary != if hot_slot { 0 } else { weights[slot] }
                || hot_bits != if hot_slot { weights[slot] } else { 0 }
                || full_bits != weights[slot]
            {
                return Err(
                    "native host classification/split steering did not consume route".into(),
                );
            }
            let actual: Vec<u64> = (0..4)
                .map(|kind| {
                    let off = (kind * max_ept + slot) * 8;
                    u64::from_le_bytes(upload[off..off + 8].try_into().unwrap())
                })
                .collect();
            if actual.as_slice() != expected_ptrs[slot] || actual.contains(&0) {
                return Err("native expert pointer plan mismatch".into());
            }
            let mut tensors = Vec::new();
            if phase == "pre_cold" || hot_slot {
                for kind in 0..4 {
                    let (host, bytes) = host_experts[ids[slot] as usize][kind];
                    if host == 0 || bytes == 0 || host.checked_add(bytes).is_none() {
                        return Err("invalid loaded expert source descriptor".into());
                    }
                    let raw = read(actual[kind], bytes)?;
                    // Descriptors come directly from the admitted, loaded expert bank; no invented host address.
                    let source = unsafe { std::slice::from_raw_parts(host as *const u8, bytes) };
                    if raw != source {
                        return Err(format!(
                            "executed weights differ from loaded bank L{layer} E{} component{kind}",
                            ids[slot]
                        ));
                    }
                    write(&format!("slot{slot}-component{kind}.bin"), &raw)?;
                    tensors.push(serde_json::json!({"component":kind,"bytes":bytes,"host_source_pointer":host,"device_pointer":actual[kind],"sha256":format!("{:x}",Sha256::digest(&raw)),"exact_loaded_weight_match":true}));
                }
            }
            slots.push(serde_json::json!({"slot":slot,"expert":ids[slot],"weight_bits":weights[slot],"hot":hot_slot,"actual_pointers":actual,"weights":tensors}));
        }
        for (name, raw) in [
            ("route-ids.i32", &ids_raw),
            ("route-weights.f32", &weights_raw),
            ("batch-upload.bin", &upload),
            ("hot-weights.f32", &hot),
            ("full-weights.f32", &full),
        ] {
            write(name, raw)?;
        }
        write("consumer.json",&serde_json::to_vec_pretty(&serde_json::json!({"layer":layer,"position":position,"phase":phase,"mask":row.descriptor.mask,"topk":topk,"num_experts":row.descriptor.num_experts,"slots":slots,"classification":"native host classification","execution":"unchanged split hot/cold","device_diagnostic_allocation_bytes":0})).map_err(|e|e.to_string())?)
    }

    pub(crate) fn selected(&self, layer: usize, position: usize) -> bool {
        self.row(layer, position).is_some()
    }
    fn row(&self, layer: usize, position: usize) -> Option<&LoadedRow> {
        self.active_trial
            .and_then(|i| self.trials.get(i))
            .and_then(|t| {
                t.rows
                    .iter()
                    .find(|r| r.descriptor.layer == layer && r.descriptor.position == position)
            })
    }
    pub(crate) fn evidence_directory(
        &self,
        layer: usize,
        position: usize,
    ) -> Result<Option<PathBuf>, String> {
        let Some(i) = self.active_trial else {
            return Ok(None);
        };
        if self.row(layer, position).is_none() {
            return Ok(None);
        };
        Ok(Some(
            self.directory
                .join(&self.trials[i].label)
                .join(format!("layer{layer}-position{position}")),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(ids: Vec<i32>, weights: Vec<f32>) -> (PathBuf, Row) {
        let d = std::env::temp_dir().join(format!(
            "route-row-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&d).unwrap();
        let idsfile = d.join("ids");
        let weightfile = d.join("weights");
        std::fs::write(
            &idsfile,
            ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            &weightfile,
            weights
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let id = |f: PathBuf| CheckedFile {
            sha256: format!("{:x}", Sha256::digest(std::fs::read(&f).unwrap())),
            path: f,
        };
        (
            d,
            Row {
                layer: 0,
                position: 9,
                num_experts: 7,
                topk: 2,
                mask: 3,
                ids: id(idsfile),
                weights: id(weightfile),
            },
        )
    }
    #[test]
    fn valid_route_uses_actual_exact_bytes() {
        let (d, r) = row(vec![6, 0], vec![0.25, 0.75]);
        let loaded = checked_row(r, 1).unwrap();
        assert_eq!(loaded.ids, vec![6, 0]);
        assert_eq!(loaded.weights, vec![0.25, 0.75]);
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn invalid_route_ids_never_admitted() {
        for ids in [vec![1, 1], vec![-1, 1], vec![7, 0]] {
            let (d, r) = row(ids, vec![0.25, 0.75]);
            assert!(checked_row(r, 1).is_err());
            std::fs::remove_dir_all(d).unwrap();
        }
    }
    #[test]
    fn nonfinite_or_negative_weights_never_admitted() {
        for weights in [vec![f32::NAN, 1.], vec![f32::INFINITY, 0.], vec![-0.1, 1.1]] {
            let (d, r) = row(vec![0, 1], weights);
            assert!(checked_row(r, 1).is_err());
            std::fs::remove_dir_all(d).unwrap();
        }
    }
    #[test]
    fn descriptor_size_ownership_and_checksum_are_exact() {
        let (d, r) = row(vec![0, 1], vec![0.25, 0.75]);
        let mut bad = r.clone();
        bad.layer = 1;
        assert!(checked_row(bad, 1).is_err());
        let mut bad = r.clone();
        bad.topk = 3;
        assert!(checked_row(bad, 1).is_err());
        let mut bad = r.clone();
        bad.mask = 0;
        assert!(checked_row(bad, 1).is_err());
        std::fs::write(&r.ids.path, [0u8; 8]).unwrap();
        assert!(checked_row(r, 1).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    #[test]
    #[ignore]
    fn captured_route_node_preserves_inactive_and_applies_all_masks() {
        assert_eq!(
            std::env::var("KRASIS_ROUTE_KERNEL_TEST").as_deref(),
            Ok("1")
        );
        let device = CudaDevice::new_with_stream(0).unwrap();
        let ptx = std::fs::read_to_string("route_override.ptx").unwrap();
        device
            .load_ptx(ptx.into(), "route_graph_test", &["matched_route_override"])
            .unwrap();
        let kernel = device
            .get_func("route_graph_test", "matched_route_override")
            .unwrap();
        for (layers, topk) in [(1usize, 1usize), (3, 8), (7, 17), (9, 48)] {
            for mask in 1u32..=3 {
                let layer = layers - 1;
                let original_ids = (0..topk as i32).collect::<Vec<_>>();
                let original_weights = vec![1.0f32 / topk as f32; topk];
                let replacement_ids = original_ids.iter().rev().copied().collect::<Vec<_>>();
                let replacement_weights = (0..topk)
                    .map(|i| (i + 1) as f32 / (topk * (topk + 1) / 2) as f32)
                    .collect::<Vec<_>>();
                let mut entries = vec![0u32; layers * 4];
                entries[layer * 4..layer * 4 + 4].copy_from_slice(&[29, 0, topk as u32, mask]);
                let experiment = Experiment {
                    device: device.clone(),
                    kernel: kernel.clone(),
                    active: device.htod_sync_copy(&[0u32]).unwrap(),
                    entries: device.htod_sync_copy(&entries).unwrap(),
                    ids: device.htod_sync_copy(&replacement_ids).unwrap(),
                    weights: device.htod_sync_copy(&replacement_weights).unwrap(),
                    layers,
                    trials: Vec::new(),
                    directory: PathBuf::new(),
                    active_trial: None,
                    armed_label: None,
                };
                let ids = device.htod_sync_copy(&original_ids).unwrap();
                let weights = device.htod_sync_copy(&original_weights).unwrap();
                let position = device.htod_sync_copy(&[29i32]).unwrap();
                let stream = *device.cu_stream();
                unsafe {
                    assert_eq!(
                        cudarc::driver::sys::lib().cuStreamBeginCapture_v2(
                            stream,
                            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL
                        ),
                        cudarc::driver::sys::CUresult::CUDA_SUCCESS
                    );
                    experiment
                        .capture_node(
                            layer,
                            *ids.device_ptr(),
                            *weights.device_ptr(),
                            *position.device_ptr(),
                        )
                        .unwrap();
                    let mut graph = std::ptr::null_mut();
                    assert_eq!(
                        cudarc::driver::sys::lib().cuStreamEndCapture(stream, &mut graph),
                        cudarc::driver::sys::CUresult::CUDA_SUCCESS
                    );
                    let mut exec = std::ptr::null_mut();
                    assert_eq!(
                        cudarc::driver::sys::lib().cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                        cudarc::driver::sys::CUresult::CUDA_SUCCESS
                    );
                    for (active, pos, applies) in
                        [(0u32, 29i32, false), (1, 28, false), (1, 29, true)]
                    {
                        assert_eq!(
                            cudarc::driver::sys::lib().cuMemcpyHtoD_v2(
                                *experiment.active.device_ptr(),
                                (&active as *const u32).cast(),
                                4
                            ),
                            cudarc::driver::sys::CUresult::CUDA_SUCCESS
                        );
                        assert_eq!(
                            cudarc::driver::sys::lib().cuMemcpyHtoD_v2(
                                *position.device_ptr(),
                                (&pos as *const i32).cast(),
                                4
                            ),
                            cudarc::driver::sys::CUresult::CUDA_SUCCESS
                        );
                        assert_eq!(
                            cudarc::driver::sys::lib().cuGraphLaunch(exec, stream),
                            cudarc::driver::sys::CUresult::CUDA_SUCCESS
                        );
                        device.synchronize().unwrap();
                        assert_eq!(
                            device.dtoh_sync_copy(&ids).unwrap(),
                            if applies && mask & 1 != 0 {
                                replacement_ids.clone()
                            } else {
                                original_ids.clone()
                            }
                        );
                        assert_eq!(
                            device
                                .dtoh_sync_copy(&weights)
                                .unwrap()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            if applies && mask & 2 != 0 {
                                replacement_weights.clone()
                            } else {
                                original_weights.clone()
                            }
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                        );
                    }
                    assert_eq!(
                        cudarc::driver::sys::lib().cuGraphExecDestroy(exec),
                        cudarc::driver::sys::CUresult::CUDA_SUCCESS
                    );
                    assert_eq!(
                        cudarc::driver::sys::lib().cuGraphDestroy(graph),
                        cudarc::driver::sys::CUresult::CUDA_SUCCESS
                    );
                }
            }
        }
    }
}
