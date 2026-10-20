//! Explicit diagnostic-only, context-owned KDA intra intervention.
//! Complete packaged FLA remains loaded; this does not qualify a candidate.
use cudarc::driver::{sys as cuda, CudaDevice};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{ffi::CString, path::PathBuf, sync::Arc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    entries: Vec<Entry>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    device_arch: i32,
    cubin: PathBuf,
    cubin_sha256: String,
    metadata: PathBuf,
    metadata_sha256: String,
}

fn identity(bytes: &[u8], expected: &str) -> Result<(), String> {
    if expected.len() != 64 || format!("{:x}", Sha256::digest(bytes)) != expected {
        return Err("KDA model diagnostic artifact checksum mismatch".into());
    }
    Ok(())
}

fn contract(m: &Value, arch: i32, geometry: (usize, usize, f32)) -> Result<(), String> {
    let (heads, dim, lower) = geometry;
    let compiled_arch = m["arch"].as_i64().ok_or("missing cubin architecture")?;
    // Explicitly requested base-architecture cubins use CUDA's same-major
    // binary compatibility. This is admission, never an automatic selection.
    if compiled_arch != i64::from(arch) && compiled_arch != i64::from(arch / 10 * 10) {
        return Err("KDA diagnostic cubin architecture incompatible with active context".into());
    }
    if heads == 0
        || dim == 0
        || !lower.is_finite()
        || lower >= 0.0
        || m["artifact_scope"] != "diagnostic-only-single-intra"
        || m["qualifies_candidate"] != false
        || m["heads"].as_u64() != Some(heads as u64)
        || m["metadata"]["rescaling_anchor"] != "causal-center"
        || m["metadata"]["configured_gate_lower_bound"].as_f64() != Some(f64::from(lower))
        || m["metadata"]["name"] != "chunk_kda_fwd_kernel_intra_sub_chunk"
        || m["metadata"]["compiled_constants"]
            != json!({"H":heads,"HV":heads,"K":dim,"BT":64,"BC":16,"BK":dim,"IS_VARLEN":false,"USE_GATHER":false})
        || m["metadata"]["compiled_signature"]
            != json!({"q":"*bf16","k":"*bf16","g":"*fp32","beta":"*bf16","Aqk":"*bf16","Akk":"*fp32","scale":"fp32","cu_seqlens":"*i32","chunk_indices":"*i32","T":"i32"})
        || m["arguments"]
            != json!([
                ["q", "CUdeviceptr"],
                ["k", "CUdeviceptr"],
                ["g", "CUdeviceptr"],
                ["beta", "CUdeviceptr"],
                ["Aqk", "CUdeviceptr"],
                ["Akk", "CUdeviceptr"],
                ["scale", "float"],
                ["cu_seqlens", "CUdeviceptr"],
                ["chunk_indices", "CUdeviceptr"],
                ["T", "int32_t"]
            ])
    {
        return Err("KDA model diagnostic geometry/arithmetic/ABI contract mismatch".into());
    }
    if m["metadata"]["driver_parameter_types"]
        != json!([
            "u64", "u64", "u64", "u64", "u64", "u64", "f32", "u64", "u64", "u32", "u64", "u64"
        ])
        || m["metadata"]["global_scratch_size"] != 0
        || m["metadata"]["profile_scratch_size"] != 0
    {
        return Err("KDA diagnostic driver ABI or scratch requirement mismatch".into());
    }
    let precision = m["requested_dot_precision"]
        .as_str()
        .ok_or("missing diagnostic precision")?;
    let effective = m["effective_dot_precisions"]
        .as_array()
        .ok_or("missing measured dot precision")?;
    if !["tf32", "tf32x3", "ieee"].contains(&precision)
        || effective.is_empty()
        || effective.iter().any(|v| v.as_str() != Some(precision))
        || m["metadata"]["effective_dot_input_precisions"] != m["effective_dot_precisions"]
        || m["metadata"]["requested_dot_input_precision"].as_str() != Some(precision)
    {
        return Err("KDA model diagnostic effective precision mismatch".into());
    }
    Ok(())
}

struct Kernel {
    geometry: (usize, usize, f32),
    module: cuda::CUmodule,
    function: cuda::CUfunction,
    threads: u32,
    shared: u32,
}
pub(crate) struct ModelKdaDiagnostic {
    device: Arc<CudaDevice>,
    kernels: Vec<Kernel>,
}
// Handles belong to the retained context. Every load/launch/drop binds that
// context; no shared wrapper-global module is used across GPUs.
unsafe impl Send for ModelKdaDiagnostic {}
unsafe impl Sync for ModelKdaDiagnostic {}

fn checked(rc: cuda::CUresult, operation: &str) -> Result<(), String> {
    if rc == cuda::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation}: {rc:?}"))
    }
}
impl ModelKdaDiagnostic {
    pub(crate) fn load(
        device: Arc<CudaDevice>,
        geometries: &[(usize, usize, f32)],
    ) -> Result<Option<Self>, String> {
        let file = match std::env::var("KRASIS_MATCHED_KDA_INTRA_CONFIG") {
            Ok(file) => file,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        crate::matched_probe::require_model_diagnostic_probe()?;
        let plan: Plan = serde_json::from_slice(&std::fs::read(file).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if plan.entries.is_empty() || geometries.is_empty() {
            return Err("KDA model diagnostic has no entries or loaded KDA layers".into());
        }
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut active = 0;
        checked(
            unsafe { cuda::lib().cuCtxGetDevice(&mut active) },
            "query diagnostic context",
        )?;
        let attr = |kind| -> Result<i32, String> {
            let mut v = 0;
            checked(
                unsafe { cuda::lib().cuDeviceGetAttribute(&mut v, kind, active) },
                "query diagnostic device",
            )?;
            Ok(v)
        };
        let arch =
            attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
                + attr(
                    cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )?;
        let warp = attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
        let max_threads =
            attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
        let max_shared =
            attr(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        let mut owner = Self {
            device,
            kernels: Vec::new(),
        };
        for entry in plan.entries.iter().filter(|e| e.device_arch == arch) {
            if !entry.cubin.is_absolute() || !entry.metadata.is_absolute() {
                return Err("KDA diagnostic paths must be absolute".into());
            }
            let bytes = std::fs::read(&entry.cubin).map_err(|e| e.to_string())?;
            let metadata = std::fs::read(&entry.metadata).map_err(|e| e.to_string())?;
            identity(&bytes, &entry.cubin_sha256)?;
            identity(&metadata, &entry.metadata_sha256)?;
            let m: Value = serde_json::from_slice(&metadata).map_err(|e| e.to_string())?;
            identity(
                &bytes,
                m["cubin_sha256"]
                    .as_str()
                    .ok_or("missing compiled cubin identity")?,
            )?;
            let geometry = geometries
                .iter()
                .copied()
                .find(|g| contract(&m, arch, *g).is_ok())
                .ok_or("KDA diagnostic does not match any loaded layer geometry")?;
            contract(&m, arch, geometry)?;
            if owner.kernels.iter().any(|k| k.geometry == geometry) {
                return Err("duplicate KDA diagnostic geometry".into());
            }
            let warps = m["metadata"]["num_warps"]
                .as_u64()
                .ok_or("missing compiled warps")?;
            let threads = warps
                .checked_mul(u64::try_from(warp).map_err(|_| "invalid warp size")?)
                .ok_or("diagnostic threads overflow")?;
            let shared = m["metadata"]["shared"]
                .as_u64()
                .ok_or("missing compiled shared bytes")?;
            if threads == 0 || threads > max_threads as u64 || shared > max_shared as u64 {
                return Err("diagnostic kernel exceeds measured device limits".into());
            }
            let symbol =
                CString::new(m["metadata"]["name"].as_str().unwrap()).map_err(|e| e.to_string())?;
            let mut module = std::ptr::null_mut();
            checked(
                unsafe { cuda::lib().cuModuleLoadData(&mut module, bytes.as_ptr().cast()) },
                "load diagnostic cubin",
            )?;
            // Install ownership immediately so every subsequent error unloads it.
            owner.kernels.push(Kernel {
                geometry,
                module,
                function: std::ptr::null_mut(),
                threads: threads as u32,
                shared: shared as u32,
            });
            let kernel = owner.kernels.last_mut().unwrap();
            checked(
                unsafe {
                    cuda::lib().cuModuleGetFunction(&mut kernel.function, module, symbol.as_ptr())
                },
                "resolve diagnostic intra",
            )?;
            log::warn!("DIAGNOSTIC ONLY KDA intra context={arch} geometry={geometry:?} cubin={} metadata={} precision={} anchor=causal-center qualifies_candidate=false",entry.cubin_sha256,entry.metadata_sha256,m["requested_dot_precision"]);
        }
        if geometries
            .iter()
            .any(|g| !owner.kernels.iter().any(|k| k.geometry == *g))
        {
            return Err("KDA diagnostic missing a loaded context/geometry".into());
        }
        Ok(Some(owner))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch(
        &self,
        geometry: (usize, usize, f32),
        q: u64,
        k: u64,
        g: u64,
        beta: u64,
        aqk: u64,
        akk: u64,
        scale: f32,
        t: i32,
        stream: cuda::CUstream,
    ) -> Result<(), String> {
        if t <= 0 {
            return Err("invalid diagnostic KDA height".into());
        }
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let kernel = self
            .kernels
            .iter()
            .find(|k| k.geometry == geometry)
            .ok_or("unadmitted KDA diagnostic layer")?;
        let (
            mut q,
            mut k,
            mut g,
            mut beta,
            mut aqk,
            mut akk,
            mut scale,
            mut null_sequence,
            mut null_chunks,
            mut t,
        ) = (q, k, g, beta, aqk, akk, scale, 0u64, 0u64, t);
        let (mut global_scratch, mut profile_scratch) = (0u64, 0u64);
        let mut args = [
            (&mut q as *mut u64).cast(),
            (&mut k as *mut u64).cast(),
            (&mut g as *mut u64).cast(),
            (&mut beta as *mut u64).cast(),
            (&mut aqk as *mut u64).cast(),
            (&mut akk as *mut u64).cast(),
            (&mut scale as *mut f32).cast(),
            (&mut null_sequence as *mut u64).cast(),
            (&mut null_chunks as *mut u64).cast(),
            (&mut t as *mut i32).cast(),
            (&mut global_scratch as *mut u64).cast(),
            (&mut profile_scratch as *mut u64).cast(),
        ];
        checked(
            cuda::lib().cuLaunchKernel(
                kernel.function,
                (t as u32).div_ceil(64),
                4,
                u32::try_from(geometry.0).map_err(|_| "diagnostic heads exceed u32")?,
                kernel.threads,
                1,
                1,
                kernel.shared,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            "launch diagnostic KDA intra",
        )
    }
}
impl Drop for ModelKdaDiagnostic {
    fn drop(&mut self) {
        if let Err(e) = self.device.bind_to_thread() {
            log::error!("diagnostic context cleanup failed: {e}");
            return;
        }
        let sync = unsafe { cuda::lib().cuCtxSynchronize() };
        if sync != cuda::CUresult::CUDA_SUCCESS {
            log::error!("diagnostic context synchronization failed: {sync:?}");
        }
        for k in &self.kernels {
            let rc = unsafe { cuda::lib().cuModuleUnload(k.module) };
            if rc != cuda::CUresult::CUDA_SUCCESS {
                log::error!("diagnostic module unload failed: {rc:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_kda_diagnostic_contract_rejects_unmeasured_or_wrong_abi() {
        let valid = json!({"arch":80,"heads":16,"artifact_scope":"diagnostic-only-single-intra","qualifies_candidate":false,
            "requested_dot_precision":"tf32","effective_dot_precisions":["tf32","tf32"],
            "arguments":[["q","CUdeviceptr"],["k","CUdeviceptr"],["g","CUdeviceptr"],["beta","CUdeviceptr"],["Aqk","CUdeviceptr"],["Akk","CUdeviceptr"],["scale","float"],["cu_seqlens","CUdeviceptr"],["chunk_indices","CUdeviceptr"],["T","int32_t"]],
            "metadata":{"rescaling_anchor":"causal-center","configured_gate_lower_bound":-4.0,"name":"chunk_kda_fwd_kernel_intra_sub_chunk",
                "requested_dot_input_precision":"tf32","effective_dot_input_precisions":["tf32","tf32"],
                "compiled_constants":{"H":16,"HV":16,"K":128,"BT":64,"BC":16,"BK":128,"IS_VARLEN":false,"USE_GATHER":false},
                "driver_parameter_types":["u64","u64","u64","u64","u64","u64","f32","u64","u64","u32","u64","u64"],"global_scratch_size":0,"profile_scratch_size":0,
                "compiled_signature":{"q":"*bf16","k":"*bf16","g":"*fp32","beta":"*bf16","Aqk":"*bf16","Akk":"*fp32","scale":"fp32","cu_seqlens":"*i32","chunk_indices":"*i32","T":"i32"}}});
        let arch = valid["arch"].as_i64().unwrap() as i32;
        let geometry = (
            valid["heads"].as_u64().unwrap() as usize,
            valid["metadata"]["compiled_constants"]["K"]
                .as_u64()
                .unwrap() as usize,
            valid["metadata"]["configured_gate_lower_bound"]
                .as_f64()
                .unwrap() as f32,
        );
        contract(&valid, arch, geometry).unwrap();
        for pointer in [
            "/qualifies_candidate",
            "/metadata/rescaling_anchor",
            "/metadata/configured_gate_lower_bound",
            "/metadata/compiled_constants/USE_GATHER",
            "/arguments/0/1",
            "/effective_dot_precisions/0",
            "/metadata/driver_parameter_types/10",
            "/metadata/global_scratch_size",
            "/metadata/profile_scratch_size",
        ] {
            let mut bad = valid.clone();
            *bad.pointer_mut(pointer).unwrap() = json!("invalid");
            assert!(contract(&bad, arch, geometry).is_err(), "{pointer}");
        }
        assert!(contract(&valid, arch, (geometry.0 + 1, geometry.1, geometry.2)).is_err());
        assert!(contract(&valid, arch, (geometry.0, geometry.1 + 1, geometry.2)).is_err());
        assert!(contract(&valid, arch + 10, geometry).is_err());
        identity(b"test", &format!("{:x}", Sha256::digest(b"test"))).unwrap();
        assert!(identity(b"changed", &format!("{:x}", Sha256::digest(b"test"))).is_err());
    }
}
