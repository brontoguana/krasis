use cudarc::driver::{sys as cuda, CudaDevice};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{ffi::CString, fs, io::Read, path::PathBuf, sync::Arc};

const ABI: &str = "hqq6-packed3-f32-scales-zeros-bf16-input-scalar-or-vec4-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}
impl Identity {
    fn verify(&self) -> Result<(), String> {
        if !self.path.is_absolute()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid HQQ experiment artifact identity".into());
        }
        let mut f = fs::File::open(&self.path).map_err(|e| e.to_string())?;
        if f.metadata().map_err(|e| e.to_string())?.len() != self.bytes {
            return Err("HQQ artifact length mismatch".into());
        }
        let mut h = Sha256::new();
        let mut buf = vec![0; 1024 * 1024];
        loop {
            let n = f.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        if format!("{:x}", h.finalize()) != self.sha256 {
            return Err(format!(
                "HQQ artifact checksum mismatch: {}",
                self.path.display()
            ));
        }
        Ok(())
    }
    fn read(&self) -> Result<Vec<u8>, String> {
        self.verify()?;
        let b = fs::read(&self.path).map_err(|e| e.to_string())?;
        if b.len() as u64 != self.bytes || format!("{:x}", Sha256::digest(&b)) != self.sha256 {
            return Err("HQQ artifact changed during admission".into());
        }
        Ok(b)
    }
}
#[derive(Clone, Copy, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Mode { NativeControl, Bf16Rounding }
fn validate_command(mode: Mode, rounded: bool, native: bool) -> Result<(), String> {
    if rounded == native || (mode == Mode::Bf16Rounding) != rounded {
        return Err("HQQ experiment mode/command mismatch".into());
    }
    Ok(())
}
fn validate_route_replay(mode: Mode, replay: bool, routes: bool) -> Result<(), String> {
    if routes != replay || (replay && mode != Mode::NativeControl) {
        return Err("HQQ composition requires native-control arithmetic and paired explicit routing/replay plans".into());
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: u32,
    mode: Mode,
    diagnostic_only: bool,
    qualifies_candidate: bool,
    abi: String,
    source: Identity,
    ptx: Identity,
    build_report: Identity,
    native: Identity,
    identities: Vec<Identity>,
    dispatch: Identity,
    descriptors: Vec<[usize; 6]>,
    devices: Vec<Device>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Device {
    uuid_hex: String,
    compute_capability: i32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dispatch {
    selections: Vec<Selection>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    rows: usize,
    cols: usize,
    group_size: usize,
    family: String,
    warps: usize,
}
fn validate_selection(s: &Selection) -> Result<(), String> {
    let valid_family = match s.family.as_str() {
        "scalar" => s.warps == 0,
        "vec4" => [1, 2, 4, 8, 16, 32].contains(&s.warps),
        _ => false,
    };
    if !valid_family
        || s.rows == 0
        || s.cols == 0
        || s.group_size == 0
        || s.cols % 4 != 0
        || s.group_size % 4 != 0
        || s.cols % s.group_size != 0
        || [s.rows, s.cols, s.group_size]
            .iter()
            .any(|&n| n > i32::MAX as usize)
    {
        return Err("unsupported HQQ experiment geometry/family".into());
    }
    Ok(())
}
fn validate_layout(shape: [usize; 6]) -> Result<(), String> {
    let [r,c,g,p,s,z] = shape;
    if r == 0 || c == 0 || g == 0 || c % 4 != 0 || g % 4 != 0 || c % g != 0
        || shape.iter().any(|&n| n > i32::MAX as usize)
        || p < c.checked_mul(3).ok_or("HQQ packed overflow")? / 4
        || s < c / g * 4 || z < c / g * 4 || s % 4 != 0 || z % 4 != 0
    { return Err(format!("unsupported HQQ packed/scales/zeros layout: {shape:?}")); }
    Ok(())
}
fn storage_extent(rows: usize, stride: usize, logical: usize) -> Result<usize, String> {
    if rows == 0 || logical == 0 || stride < logical { return Err("invalid HQQ storage extent".into()); }
    (rows-1).checked_mul(stride).and_then(|n| n.checked_add(logical)).ok_or("HQQ storage extent overflow".into())
}
fn checked(rc: cuda::CUresult, op: &str) -> Result<(), String> {
    if rc == cuda::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{op}: {rc:?}"))
    }
}
pub(crate) struct Experiment {
    device: Arc<CudaDevice>,
    module: cuda::CUmodule,
    functions: [cuda::CUfunction; 3],
    descriptors: Vec<[usize; 6]>,
    mode: Mode,
    dispatch: Identity,
}
// All handles are owned by the retained context, including error-path unload.
unsafe impl Send for Experiment {}
unsafe impl Sync for Experiment {}
impl Drop for Experiment {
    fn drop(&mut self) {
        if self.device.bind_to_thread().is_ok() && !self.module.is_null() {
            unsafe {
                cuda::lib().cuModuleUnload(self.module);
            }
        }
    }
}
impl Experiment {
    pub(crate) fn load(device: Arc<CudaDevice>) -> Result<Option<Self>, String> {
        let rounded = std::env::var_os("KRASIS_MATCHED_HQQ_ROUNDING_CONFIG");
        let native = std::env::var_os("KRASIS_MATCHED_HQQ_NATIVE_CONFIG");
        let path = match (&rounded, &native) {
            (None, None) => {
                if std::env::var_os("KRASIS_HQQ_ROUNDING_COMMAND").is_some()
                    || std::env::var_os("KRASIS_HQQ_NATIVE_COMMAND").is_some() {
                    return Err("HQQ experiment command requires its explicit plan".into());
                }
                return Ok(None);
            },
            (Some(p), None) => p,
            (None, Some(p)) => p,
            _ => return Err("conflicting HQQ experimental plans".into()),
        };
        let rounded_command = std::env::var_os("KRASIS_HQQ_ROUNDING_COMMAND");
        let native_command = std::env::var_os("KRASIS_HQQ_NATIVE_COMMAND");
        if rounded_command.is_some() && rounded_command != Some("1".into())
            || native_command.is_some() && native_command != Some("1".into()) {
            return Err("invalid HQQ experimental command marker".into());
        }
        if !cfg!(target_os = "linux") {
            return Err("HQQ experiment requires Linux mapped-native admission".into());
        }
        let plan: Plan = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        validate_route_replay(
            plan.mode,
            std::env::var_os("KRASIS_MATCHED_REPLAY_CONFIG").is_some(),
            std::env::var_os("KRASIS_MATCHED_ROUTES_CONFIG").is_some(),
        )?;
        for key in [
            "KRASIS_MATCHED_KDA_INTRA_CONFIG",
            "KRASIS_MATCHED_MOE_LANE_CONFIG",
            "KRASIS_MATCHED_MOE_PARTITION_CONFIG",
            "KRASIS_GLM53_KDA_RECURRENT_ORACLE",
            "KRASIS_GLM53_KDA_LEGACY_RECURRENT_ORACLE",
            "KRASIS_GLM53_KDA_SERIAL_CONV_ORACLE",
        ] {
            if std::env::var_os(key).is_some() {
                return Err(format!("HQQ single-factor experiment forbids {key}"));
            }
        }
        validate_command(plan.mode, rounded_command.is_some(), native_command.is_some())?;
        if (plan.mode == Mode::Bf16Rounding) != rounded.is_some() {
            return Err("HQQ experiment plan namespace mismatch".into());
        }
        if plan.schema != 3
            || !plan.diagnostic_only
            || plan.qualifies_candidate
            || plan.abi != ABI
            || plan.identities.is_empty()
            || plan.devices.is_empty()
            || plan.descriptors.is_empty()
        {
            return Err("unsupported HQQ experiment plan/ABI".into());
        }
        plan.native.verify()?;
        let maps = fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
        if !maps.lines().any(|l| {
            l.split_whitespace().last().map(PathBuf::from).as_ref() == Some(&plan.native.path)
        }) {
            return Err("HQQ experiment native is not mapped".into());
        }
        for id in &plan.identities {
            id.verify()?;
        }
        let source = plan.source.read()?;
        let ptx = plan.ptx.read()?;
        let build: serde_json::Value =
            serde_json::from_slice(&plan.build_report.read()?).map_err(|e| e.to_string())?;
        if build["diagnostic_only"] != true
            || build["qualifies_candidate"] != false
            || build["plan"]["abi"] != ABI
            || build["plan"]["target_compute"] != 80
            || build["plan"]["source"]["sha256"] != format!("{:x}", Sha256::digest(&source))
            || build["ptx_sha256"] != plan.ptx.sha256
            || build["ptx_bytes"].as_u64() != Some(plan.ptx.bytes)
        {
            return Err("HQQ source/PTX/compiler provenance mismatch".into());
        }
        let nvcc: Identity =
            serde_json::from_value(build["plan"]["nvcc"].clone()).map_err(|e| e.to_string())?;
        nvcc.verify()?;
        let dispatch: Dispatch =
            serde_json::from_slice(&plan.dispatch.read()?).map_err(|e| e.to_string())?;
        if dispatch.selections.is_empty() {
            return Err("HQQ experiment requires measured dispatch".into());
        }
        let configured = std::env::var_os("KRASIS_MATCHED_HQQ_DISPATCH_CONFIG")
            .ok_or("HQQ experiment requires original measured dispatch config")?;
        if fs::canonicalize(configured).map_err(|e| e.to_string())?
            != fs::canonicalize(&plan.dispatch.path).map_err(|e| e.to_string())?
        {
            return Err("HQQ runtime dispatch identity mismatch".into());
        }
        let mut unique = std::collections::HashSet::new();
        for s in &dispatch.selections {
            validate_selection(s)?;
            if !unique.insert((s.rows, s.cols, s.group_size)) {
                return Err("duplicate HQQ geometry".into());
            }
        }
        let mut layouts = std::collections::HashSet::new();
        for &d in &plan.descriptors {
            validate_layout(d)?;
            if !layouts.insert(d) || !unique.contains(&(d[0],d[1],d[2])) {
                return Err("duplicate or unadmitted HQQ descriptor inventory".into());
            }
        }
        if layouts.iter().map(|d| (d[0],d[1],d[2])).collect::<std::collections::HashSet<_>>() != unique {
            return Err("HQQ descriptor inventory does not cover measured dispatch".into());
        }
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut active = 0;
        checked(
            unsafe { cuda::lib().cuCtxGetDevice(&mut active) },
            "HQQ context device",
        )?;
        let attr = |kind| -> Result<i32, String> {
            let mut x = 0;
            checked(
                unsafe { cuda::lib().cuDeviceGetAttribute(&mut x, kind, active) },
                "HQQ device limits",
            )?;
            Ok(x)
        };
        let arch = 10
            * attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
            + attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let mut uuid = cuda::CUuuid { bytes: [0; 16] };
        checked(
            unsafe { cuda::lib().cuDeviceGetUuid_v2(&mut uuid, active) },
            "HQQ device UUID",
        )?;
        let uuid_hex = uuid
            .bytes
            .iter()
            .map(|&b| format!("{:02x}", b as u8))
            .collect::<String>();
        if arch < 80
            || plan
                .devices
                .iter()
                .filter(|d| d.uuid_hex == uuid_hex && d.compute_capability == arch)
                .count()
                != 1
        {
            return Err("HQQ artifact device/ISA mismatch".into());
        }
        let max_threads =
            attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as usize;
        let max_shared =
            attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?
                as usize;
        if attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_WARP_SIZE)? != 32
            || dispatch.selections.iter().any(|s| {
                s.warps.max(8) * 32 > max_threads
                    || s.cols.checked_mul(2).map_or(true, |b| b > max_shared)
            })
        {
            return Err("HQQ experiment exceeds device limits".into());
        }
        let text = CString::new(ptx).map_err(|e| e.to_string())?;
        let mut owner = Self {
            device,
            module: std::ptr::null_mut(),
            functions: [std::ptr::null_mut(); 3],
            descriptors: plan.descriptors,
            mode: plan.mode,
            dispatch: plan.dispatch,
        };
        checked(
            unsafe { cuda::lib().cuModuleLoadData(&mut owner.module, text.as_ptr().cast()) },
            "load HQQ experiment before memory planning",
        )?;
        for (i, name) in [
            "hqq6_bf16_weight_gemv_f32",
            "hqq6_bf16_weight_gemv_bf16",
            "hqq6_bf16_weight_gemv_bf16_vec4",
        ]
        .iter()
        .enumerate()
        {
            let symbol = CString::new(*name).unwrap();
            checked(
                unsafe {
                    cuda::lib().cuModuleGetFunction(
                        &mut owner.functions[i],
                        owner.module,
                        symbol.as_ptr(),
                    )
                },
                "HQQ experiment ABI symbol",
            )?;
            for j in 0..11 {
                let mut off = 0;
                let mut bytes = 0;
                checked(
                    unsafe {
                        cuda::lib().cuFuncGetParamInfo(owner.functions[i], j, &mut off, &mut bytes)
                    },
                    "HQQ experiment ABI parameter",
                )?;
                if off != if j < 5 { j * 8 } else { 40 + (j - 5) * 4 }
                    || bytes != if j < 5 { 8 } else { 4 }
                {
                    return Err("HQQ experiment ABI parameter layout mismatch".into());
                }
            }
            checked(
                unsafe { cuda::lib().cuFuncLoad(owner.functions[i]) },
                "eager HQQ function load before memory planning",
            )?;
            let mut offset = 0;
            let mut size = 0;
            if unsafe {
                cuda::lib().cuFuncGetParamInfo(owner.functions[i], 11, &mut offset, &mut size)
            } != cuda::CUresult::CUDA_ERROR_INVALID_VALUE
            {
                return Err("HQQ experiment ABI has unexpected trailing arguments".into());
            }
        }
        eprintln!("[matched-hqq-rounding] device={} arch={} ptx={} source={} native={} diagnostic_only=true buffers=0",uuid_hex,arch,plan.ptx.sha256,plan.source.sha256,plan.native.sha256);
        eprintln!("[matched-hqq-rounding] mode={:?} context_owned_dispatch=true", owner.mode);
        Ok(Some(owner))
    }
    pub(crate) fn is_rounded(&self) -> bool { self.mode == Mode::Bf16Rounding }
    pub(crate) fn authorize_dispatch(&self, device: &Arc<CudaDevice>, path: &std::path::Path) -> Result<(), String> {
        if !Arc::ptr_eq(device, &self.device) { return Err("HQQ dispatch context owner mismatch".into()); }
        if fs::canonicalize(path).map_err(|e| e.to_string())?
            != fs::canonicalize(&self.dispatch.path).map_err(|e| e.to_string())? {
            return Err("HQQ dispatch admission identity mismatch".into());
        }
        self.dispatch.verify()
    }
    pub(crate) fn validate_storage(&self, shape: [usize; 6], ptrs: [u64; 3]) -> Result<(), String> {
        self.validate_descriptor(shape, ptrs)?;
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let logical = [shape[1].checked_mul(3).ok_or("HQQ packed overflow")? / 4, shape[1] / shape[2] * 4, shape[1] / shape[2] * 4];
        for i in 0..3 {
            let extent = storage_extent(shape[0], shape[i+3], logical[i])?;
            let mut ordinal = -1i32; let mut start = 0u64; let mut size = 0usize;
            unsafe {
                checked(cuda::lib().cuPointerGetAttribute((&mut ordinal as *mut i32).cast(), cuda::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptrs[i]), "HQQ pointer device")?;
                checked(cuda::lib().cuPointerGetAttribute((&mut start as *mut u64).cast(), cuda::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ptrs[i]), "HQQ pointer range start")?;
                checked(cuda::lib().cuPointerGetAttribute((&mut size as *mut usize).cast(), cuda::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_RANGE_SIZE, ptrs[i]), "HQQ pointer range size")?;
            }
            if ordinal < 0 || ordinal as usize != self.device.ordinal() || ptrs[i] < start
                || ptrs[i].checked_add(extent as u64).ok_or("HQQ pointer extent overflow")?
                    > start.checked_add(size as u64).ok_or("HQQ allocation extent overflow")? {
                return Err("HQQ loaded tensor device/allocation extent mismatch".into());
            }
        }
        Ok(())
    }
    pub(crate) fn validate_inventory(
        &self,
        shapes: &[[usize; 6]],
    ) -> Result<(), String> {
        let actual = shapes
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let expected = self
            .descriptors
            .iter()
            .copied()
            .collect();
        if actual != expected {
            return Err(
                "HQQ experiment loaded descriptor inventory differs from frozen dispatch".into(),
            );
        }
        Ok(())
    }
    pub(crate) fn validate_descriptor(
        &self,
        shape: [usize; 6],
        ptrs: [u64; 3],
    ) -> Result<(), String> {
        validate_layout(shape)?;
        if ptrs.iter().any(|&p| p == 0) || !self.descriptors.contains(&shape)
        {
            return Err(format!("unadmitted HQQ experiment descriptor {shape:?}, nonnull={:?}",ptrs.map(|p|p!=0)));
        }
        Ok(())
    }
    pub(crate) fn launch(
        &self,
        name: &str,
        shape: [usize; 6],
        ptrs: [u64; 5],
        threads: u32,
        stream: cuda::CUstream,
    ) -> Result<(), String> {
        if !self.is_rounded() { return Err("native control cannot launch rounded kernel".into()); }
        self.validate_descriptor(shape, [ptrs[0], ptrs[1], ptrs[2]])?;
        let index = match name {
            "hqq6_decode_gemv_f32" => 0,
            "hqq6_decode_gemv_bf16" => 1,
            "hqq6_decode_gemv_bf16_vec4" => 2,
            _ => return Err("unsupported HQQ experimental kernel".into()),
        };
        // Autotune measures every candidate before applying the frozen geometry.
        // Caller retains native family/geometry, including these calibration calls.
        if threads == 0
            || threads % 32 != 0
            || threads > 1024
            || (index < 2 && threads != 256)
            || ptrs[3] == 0
            || ptrs[4] == 0
        {
            return Err("HQQ experimental launch geometry/pointers invalid".into());
        }
        let mut ptrs = ptrs;
        let mut dims = shape.map(|d| i32::try_from(d).map_err(|_| "HQQ descriptor exceeds ABI"));
        let mut values = [0i32; 6];
        for (v, d) in values.iter_mut().zip(dims.iter_mut()) {
            *v = d.clone()?;
        }
        let mut args: [*mut std::ffi::c_void; 11] = [std::ptr::null_mut(); 11];
        for i in 0..5 {
            args[i] = (&mut ptrs[i] as *mut u64).cast();
        }
        for i in 0..6 {
            args[5 + i] = (&mut values[i] as *mut i32).cast();
        }
        let blocks = u32::try_from(shape[0].div_ceil(threads as usize / 32))
            .map_err(|_| "HQQ grid overflow")?;
        checked(
            unsafe {
                cuda::lib().cuLaunchKernel(
                    self.functions[index],
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    u32::try_from(shape[1] * 2).map_err(|_| "HQQ shared overflow")?,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "HQQ reconstructed-weight experiment launch",
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routing_replay_keeps_original_arithmetic_and_requires_both_plans() {
        assert!(validate_route_replay(Mode::NativeControl, true, true).is_ok());
        for mode in [Mode::NativeControl, Mode::Bf16Rounding] {
            assert!(validate_route_replay(mode, false, false).is_ok());
            assert!(validate_route_replay(mode, true, false).is_err());
            assert!(validate_route_replay(mode, false, true).is_err());
        }
        assert!(validate_route_replay(Mode::Bf16Rounding, true, true).is_err());
    }
    #[test]
    fn mode_requires_one_matching_built_command() {
        for mode in [Mode::NativeControl, Mode::Bf16Rounding] {
            for (r,n) in [(false,false),(true,true)] { assert!(validate_command(mode,r,n).is_err()); }
        }
        assert!(validate_command(Mode::NativeControl,false,true).is_ok());
        assert!(validate_command(Mode::Bf16Rounding,true,false).is_ok());
        assert!(validate_command(Mode::NativeControl,true,false).is_err());
        assert!(validate_command(Mode::Bf16Rounding,false,true).is_err());
        assert!(serde_json::from_str::<Mode>("\"unknown\"").is_err());
    }
    #[test]
    fn extent_includes_logical_tail_without_inventing_padding() {
        assert_eq!(storage_extent(2,32,4).unwrap(),36);
        assert_eq!(storage_extent(1,32,4).unwrap(),4);
        assert!(storage_extent(0,32,4).is_err());
        assert!(storage_extent(2,3,4).is_err());
        assert!(storage_extent(usize::MAX,32,4).is_err());
    }
    #[test]
    fn padded_layouts_require_sufficient_aligned_storage() {
        assert!(validate_layout([8192,128,128,96,32,32]).is_ok());
        assert!(validate_layout([16384,1536,128,1152,64,64]).is_ok());
        assert!(validate_layout([8,256,128,197,12,16]).is_ok());
        for shape in [[8,256,128,191,8,8],[8,256,128,192,4,8],[8,256,128,192,8,4],[8,256,128,192,9,8],[8,256,128,192,8,9],[8,257,128,196,8,8],[8,256,0,192,8,8]] {
            assert!(validate_layout(shape).is_err(),"{shape:?}");
        }
    }
    #[test]
    fn rejects_unadmitted_layouts_and_geometry() {
        for (family, warps) in [("scalar", 0), ("vec4", 1), ("vec4", 32)] {
            assert!(validate_selection(&Selection {
                rows: 8,
                cols: 256,
                group_size: 128,
                family: family.into(),
                warps
            })
            .is_ok());
        }
        for (cols, group, family, warps) in [
            (0, 128, "vec4", 8),
            (257, 128, "vec4", 8),
            (256, 0, "vec4", 8),
            (256, 128, "vec4", 3),
            (256, 128, "scalar", 8),
            (256, 128, "unknown", 0),
        ] {
            assert!(validate_selection(&Selection {
                rows: 8,
                cols,
                group_size: group,
                family: family.into(),
                warps
            })
            .is_err());
        }
    }
    #[test]
    fn rejects_relative_and_corrupted_identity() {
        assert!(Identity {
            path: "relative".into(),
            bytes: 0,
            sha256: "0".repeat(64)
        }
        .verify()
        .is_err());
        let p = std::env::temp_dir().join(format!("krasis-hqq-identity-{}", std::process::id()));
        fs::write(&p, b"bound").unwrap();
        let i = Identity {
            path: p.clone(),
            bytes: 5,
            sha256: format!("{:x}", Sha256::digest(b"bound")),
        };
        assert!(i.verify().is_ok());
        fs::write(&p, b"other").unwrap();
        assert!(i.verify().is_err());
        fs::remove_file(p).unwrap();
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ParityPlan {
        experiment: PathBuf,
        native_control: Identity,
        inputs: std::collections::BTreeMap<String, Identity>,
        references: std::collections::BTreeMap<String, Identity>,
        output: PathBuf,
        shape: [usize; 6],
        warps: usize,
    }
    #[test]
    #[ignore]
    fn context_owned_hqq_rounding_matches_same_order_reference() {
        let file = std::env::var_os("KRASIS_HQQ_ROUNDING_PARITY_PLAN")
            .expect("built parity command requires frozen plan");
        let p: ParityPlan = serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
        assert_eq!(
            std::env::var_os("KRASIS_MATCHED_HQQ_ROUNDING_CONFIG"),
            Some(p.experiment.into_os_string())
        );
        assert!(p.output.is_absolute());
        fs::create_dir(&p.output).expect("fresh parity output required");
        assert_eq!(p.inputs.len(), 4);
        assert_eq!(p.references.len(), 6);
        assert!([1, 2, 4, 8, 16, 32].contains(&p.warps));
        let dev = CudaDevice::new(0).unwrap();
        let experiment = Experiment::load(dev.clone()).unwrap().unwrap();
        let raw = |name: &str| p.inputs.get(name).unwrap().read().unwrap();
        let packed = raw("packed");
        let scales = raw("scales");
        let zeros = raw("zeros");
        let input = raw("input");
        let [rows, cols, group, ps, ss, zs] = p.shape;
        assert!(rows > 0 && cols > 0 && group > 0 && cols % group == 0);
        assert_eq!(packed.len(), rows * ps);
        assert_eq!(scales.len(), rows * ss);
        assert_eq!(zeros.len(), rows * zs);
        assert_eq!(input.len(), cols * 2);
        let pv = dev.htod_copy(packed).unwrap();
        let f32s = |b: Vec<u8>| {
            b.chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let sv = dev.htod_copy(f32s(scales)).unwrap();
        let zv = dev.htod_copy(f32s(zeros)).unwrap();
        let xv = dev
            .htod_copy(
                input
                    .chunks_exact(2)
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let tensor_ptrs = [*pv.device_ptr(), *sv.device_ptr(), *zv.device_ptr()];
        experiment.validate_storage(p.shape, tensor_ptrs).unwrap();
        let mut short = tensor_ptrs;
        short[0] += (rows * ps - 1) as u64;
        assert!(experiment.validate_storage(p.shape, short).is_err());
        let other_owner = CudaDevice::new(0).unwrap();
        assert!(experiment.authorize_dispatch(&other_owner, &experiment.dispatch.path).is_err());
        assert!(experiment.authorize_dispatch(&dev, &experiment.dispatch.path).is_ok());
        p.native_control.verify().unwrap();
        let rounded_path = std::env::var_os("KRASIS_MATCHED_HQQ_ROUNDING_CONFIG").unwrap();
        std::env::remove_var("KRASIS_MATCHED_HQQ_ROUNDING_CONFIG");
        std::env::remove_var("KRASIS_HQQ_ROUNDING_COMMAND");
        std::env::set_var("KRASIS_MATCHED_HQQ_NATIVE_CONFIG", &p.native_control.path);
        std::env::set_var("KRASIS_HQQ_NATIVE_COMMAND", "1");
        let control = Experiment::load(dev.clone()).unwrap().unwrap();
        assert!(!control.is_rounded());
        control.authorize_dispatch(&dev, &control.dispatch.path).unwrap();
        control.validate_storage(p.shape, tensor_ptrs).unwrap();
        assert!(control.launch("hqq6_decode_gemv_bf16", p.shape, [tensor_ptrs[0],tensor_ptrs[1],tensor_ptrs[2],*xv.device_ptr(),*xv.device_ptr()], 256, *dev.cu_stream()).is_err());
        std::env::remove_var("KRASIS_MATCHED_HQQ_NATIVE_CONFIG");
        std::env::remove_var("KRASIS_HQQ_NATIVE_COMMAND");
        std::env::set_var("KRASIS_MATCHED_HQQ_ROUNDING_CONFIG", rounded_path);
        std::env::set_var("KRASIS_HQQ_ROUNDING_COMMAND", "1");
        let mut out32 = dev.alloc_zeros::<f32>(rows).unwrap();
        let mut out16 = dev.alloc_zeros::<u16>(rows).unwrap();
        let embedded = include_str!(concat!(env!("OUT_DIR"), "/decode_kernels.ptx"));
        dev.load_ptx(
            cudarc::nvrtc::Ptx::from_src(embedded),
            "hqq_native_parity",
            &[
                "hqq6_decode_gemv_f32",
                "hqq6_decode_gemv_bf16",
                "hqq6_decode_gemv_bf16_vec4",
            ],
        )
        .unwrap();
        let mut reports = Vec::new();
        for rounded in [false, true] {
            for name in [
                "hqq6_decode_gemv_f32",
                "hqq6_decode_gemv_bf16",
                "hqq6_decode_gemv_bf16_vec4",
            ] {
                let fp32 = name.ends_with("f32");
                let threads = if name.ends_with("vec4") {
                    (p.warps * 32) as u32
                } else {
                    256
                };
                let out = if fp32 {
                    *out32.device_ptr()
                } else {
                    *out16.device_ptr()
                };
                let ptrs = [
                    *pv.device_ptr(),
                    *sv.device_ptr(),
                    *zv.device_ptr(),
                    *xv.device_ptr(),
                    out,
                ];
                if rounded {
                    experiment
                        .launch(name, p.shape, ptrs, threads, *dev.cu_stream())
                        .unwrap();
                } else {
                    let f = dev.get_func("hqq_native_parity", name).unwrap();
                    unsafe {
                        f.launch(
                            LaunchConfig {
                                grid_dim: (rows.div_ceil(threads as usize / 32) as u32, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes: (cols * 2) as u32,
                            },
                            (
                                ptrs[0],
                                ptrs[1],
                                ptrs[2],
                                ptrs[3],
                                ptrs[4],
                                rows as i32,
                                cols as i32,
                                group as i32,
                                ps as i32,
                                ss as i32,
                                zs as i32,
                            ),
                        )
                        .unwrap();
                    }
                }
                dev.synchronize().unwrap();
                let actual: Vec<u8> = if fp32 {
                    dev.dtoh_sync_copy(&out32)
                        .unwrap()
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect()
                } else {
                    dev.dtoh_sync_copy(&out16)
                        .unwrap()
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect()
                };
                let key = format!("{}-{}", if rounded { "rounded" } else { "native" }, name);
                let expected = p.references.get(&key).unwrap().read().unwrap();
                fs::write(p.output.join(&key), &actual).unwrap();
                reports.push(serde_json::json!({"path":key,"exact":actual==expected,"sha256":format!("{:x}",Sha256::digest(&actual))}));
            }
        }
        let all = reports.iter().all(|r| r["exact"] == true);
        fs::write(p.output.join("PARITY.json"),serde_json::to_vec_pretty(&serde_json::json!({"diagnostic_only":true,"qualifies_candidate":false,"context_owned_dispatch":true,"all_six_exact":all,"paths":reports})).unwrap()).unwrap();
        assert!(
            all,
            "context-owned dispatch differs from independent same-order references"
        );
        // Runtime layout, pointer and geometry admission must reject unsupported inputs.
        assert!(experiment
            .validate_descriptor(p.shape, [0, *sv.device_ptr(), *zv.device_ptr()])
            .is_err());
        let mut bad = p.shape;
        bad[3] += 1;
        assert!(experiment
            .validate_descriptor(bad, [*pv.device_ptr(), *sv.device_ptr(), *zv.device_ptr()])
            .is_err());
        assert!(experiment
            .launch(
                "unsupported",
                p.shape,
                ptrs_for_test(&pv, &sv, &zv, &xv, &out16),
                threads_for_test(),
                *dev.cu_stream()
            )
            .is_err());
        let _ = (&mut out32, &mut out16);
    }
    fn ptrs_for_test(
        p: &cudarc::driver::CudaSlice<u8>,
        s: &cudarc::driver::CudaSlice<f32>,
        z: &cudarc::driver::CudaSlice<f32>,
        x: &cudarc::driver::CudaSlice<u16>,
        y: &cudarc::driver::CudaSlice<u16>,
    ) -> [u64; 5] {
        [
            *p.device_ptr(),
            *s.device_ptr(),
            *z.device_ptr(),
            *x.device_ptr(),
            *y.device_ptr(),
        ]
    }
    fn threads_for_test() -> u32 {
        256
    }
}
