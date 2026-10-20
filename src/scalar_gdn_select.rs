//! Cold explicit admission for the private native64 scalar-GDN candidate.
#[path = "scalar_gdn_qualification.rs"]
mod qualification;
pub(crate) fn requested() -> Result<bool, String> {
    match std::env::var("KRASIS_SCALAR_GDN_NATIVE64") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "1" => {
            if !cfg!(feature = "scalar-gdn-test") {
                return Err(
                    "native64 scalar-GDN requires the explicitly built scalar-gdn-test candidate"
                        .into(),
                );
            }
            if std::env::var_os("KRASIS_NO_FLA").is_some() {
                return Err("native64 scalar execution requires its owned FLA sidecar; KRASIS_NO_FLA is incompatible".into());
            }
            Err("This diagnostic build provides native64 resource controls only; serving admission requires complete decode, session and device-handoff integration".into())
        }
        Ok(value) if value == "qualification" => {
            if std::env::var_os("KRASIS_NO_FLA").is_some() { return Err("native qualification requires its owned FLA sidecar".into()); }
            qualification::requested()
        }
        Ok(value) if value == "diagnostic" => {
            if std::env::var_os("KRASIS_NO_FLA").is_some() { return Err("native diagnostic requires its owned FLA sidecar".into()); }
            qualification::diagnostic_requested()
        }
        _ => Err("KRASIS_SCALAR_GDN_NATIVE64 must be absent, 1 (activation rejected), or qualification (private test only)".into()),
    }
}
#[cfg(feature = "scalar-gdn-test")]
pub(crate) fn sidecar(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<std::path::PathBuf, String> {
    use cudarc::driver::sys::CUdevice_attribute as A;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let major = device
        .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| e.to_string())?;
    let minor = device
        .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| e.to_string())?;
    let arch = major
        .checked_mul(10)
        .and_then(|x| x.checked_add(minor))
        .ok_or("scalar actual architecture overflow")?;
    let contract: serde_json::Value =
        serde_json::from_str(include_str!("../python/krasis/fla_sidecar_contract.json"))
            .map_err(|e| e.to_string())?;
    let supported = contract["architectures"]
        .as_array()
        .ok_or("scalar sidecar architecture contract absent")?;
    // Same-generation cubin compatibility is the package's existing selection
    // contract. No different-generation or generic numerical fallback exists.
    let selected = supported
        .iter()
        .filter_map(|v| v.as_i64())
        .filter(|&v| v / 10 == major as i64 && v <= arch as i64)
        .max()
        .ok_or_else(|| format!("no packaged scalar FLA cubin supports actual sm_{arch}"))?;
    let name = format!("libkrasis_fla_sm{selected}.so");
    let path = crate::gpu_prefill::find_vendor_so(&name)
        .ok_or_else(|| format!("required scalar native sidecar absent:{name}"))?;
    log::info!(
        "Scalar native64 actual device {} sm_{} selects {} (required owned context ABI2)",
        device.ordinal(),
        arch,
        path
    );
    Ok(path.into())
}
