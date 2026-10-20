pub(crate) fn shared_kda_requested() -> Result<bool, String> {
    match std::env::var("KRASIS_SHARED_KDA") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "1" => {
            if !cfg!(feature = "shared-kda-test") {
                return Err(
                    "shared KDA requires explicitly built shared-kda-test candidate".into(),
                );
            }
            Ok(true)
        }
        _ => Err("KRASIS_SHARED_KDA must be absent or exactly 1".into()),
    }
}
#[cfg(feature = "shared-kda-test")]
pub(crate) fn serving_kernel_bank(device: Arc<CudaDevice>) -> Result<KdaModelKernelBank, String> {
    KdaModelKernelBank::load(
        device,
        include_str!(concat!(env!("OUT_DIR"), "/kda_owner_transport.ptx")),
        include_str!(concat!(env!("OUT_DIR"), "/kda_owner_math.ptx")),
        include_str!(concat!(env!("OUT_DIR"), "/kda_raw_producer.ptx")),
    )
}
#[cfg(not(feature = "shared-kda-test"))]
pub(crate) fn serving_kernel_bank(_device: Arc<CudaDevice>) -> Result<KdaModelKernelBank, String> {
    Err("selected KDA kernels were not built".into())
}
