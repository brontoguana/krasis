// Selected-owner logits failure ABI; explicit experimental admission only.
pub const PREFIX_FAILURE_MARKER: u32 = 0x7fc04b52;
pub fn check_prefix_logits(logits: &[f32]) -> Result<(), String> {
    let first = logits.first().ok_or("fixed16 logits buffer is empty")?;
    if first.to_bits() == PREFIX_FAILURE_MARKER {
        return Err("fixed16 recurrence owner failed before logits consumption".into());
    }
    Ok(())
}
#[cfg(test)]
mod prefix_guard_tests {
    use super::*;
    #[test]
    fn marker_rejects_before_consumption_without_changing_valid_logits() {
        assert!(check_prefix_logits(&[]).is_err());
        assert!(check_prefix_logits(&[f32::from_bits(PREFIX_FAILURE_MARKER), 1.0]).is_err());
        for bits in [0u32, 1, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234] {
            let values = [f32::from_bits(bits), f32::from_bits(PREFIX_FAILURE_MARKER)];
            assert!(check_prefix_logits(&values).is_ok());
            assert_eq!(values[0].to_bits(), bits);
            assert_eq!(values[1].to_bits(), PREFIX_FAILURE_MARKER);
        }
    }
}
pub struct PrefixLogitGuard {
    owners: Vec<Arc<PrefixRecurrence>>,
    table: CudaSlice<u64>,
    pub allocation_bytes: usize,
}
impl std::fmt::Debug for PrefixLogitGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixLogitGuard")
            .field("owners", &self.owners.len())
            .field("allocation_bytes", &self.allocation_bytes)
            .finish()
    }
}
impl PrefixLogitGuard {
    pub fn new(owners: Vec<Arc<PrefixRecurrence>>, reserve: usize) -> Result<Self, String> {
        let first = owners
            .first()
            .ok_or("fixed16 error guard requires selected owners")?;
        i32::try_from(owners.len()).map_err(|_| "fixed16 error-table count exceeds CUDA i32")?;
        let context = first.device.cu_primary_ctx();
        let mut addresses = Vec::with_capacity(owners.len());
        for owner in &owners {
            if owner.device.cu_primary_ctx() != context {
                return Err("fixed16 error guard cannot span CUDA contexts".into());
            }
            if owner.arithmetic_sha256 != first.arithmetic_sha256 {
                return Err("fixed16 error guard owner arithmetic contracts differ".into());
            }
            let address = ptr(&owner.error);
            if address == 0 || addresses.contains(&address) {
                return Err("fixed16 error guard owner pointer is null or duplicated".into());
            }
            addresses.push(address);
        }
        let allocation_bytes = addresses
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or("fixed16 error-table bytes overflow")?;
        first.device.bind_to_thread().map_err(|e| e.to_string())?;
        let (mut free, mut total) = (0, 0);
        status(
            unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) },
            "error-table memory",
        )?;
        if free
            < allocation_bytes
                .checked_add(reserve)
                .ok_or("fixed16 error-table reserve overflow")?
        {
            return Err("fixed16 error-table measured allocation admission failed".into());
        }
        let table = first
            .device
            .htod_copy(addresses)
            .map_err(|e| e.to_string())?;
        first.device.synchronize().map_err(|e| e.to_string())?;
        Ok(Self {
            owners,
            table,
            allocation_bytes,
        })
    }
    pub fn required_pointers(&self) -> Vec<u64> {
        std::iter::once(ptr(&self.table))
            .chain(self.owners.iter().map(|x| ptr(&x.error)))
            .collect()
    }
    pub fn validate_owners(&self, owners: &[Arc<PrefixRecurrence>]) -> Result<(), String> {
        if owners.len() != self.owners.len()
            || owners
                .iter()
                .zip(&self.owners)
                .any(|(a, b)| !Arc::ptr_eq(a, b))
        {
            return Err(
                "fixed16 captured error-table owners changed; create a fresh diagnostic store"
                    .into(),
            );
        }
        Ok(())
    }
    /// The caller owns the stream and schedules this after the complete final
    /// head/softcap, before the existing logits transfer. No host error read.
    pub unsafe fn launch(
        &self,
        logits: u64,
        elements: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if logits == 0 || elements == 0 {
            return Err("fixed16 error guard logits descriptor is empty".into());
        }
        self.owners[0].launch(
            "prefix_logits_guard",
            (1, 1, 1),
            1,
            0,
            stream,
            &mut arguments(
                &mut [ptr(&self.table), logits],
                &mut [self.owners.len() as i32],
                &mut [],
            ),
        )
    }
}
impl Drop for PrefixLogitGuard {
    fn drop(&mut self) {
        crate::prefix_lifecycle_observer::record("actual_guard_drop_enter",ptr(&self.table) as usize,None);
        // Prefill may consume this table on a caller-owned stream. This is cold
        // destruction, never token execution or graph replay.
        match self.owners[0].device.bind_to_thread() {
            Ok(()) => unsafe {
                let rc = sys::lib().cuCtxSynchronize();
                if rc != sys::CUresult::CUDA_SUCCESS {
                    log::error!("fixed16 error-table cleanup synchronization: {rc:?}");
                }
            },
            Err(e) => log::error!("fixed16 error-table cleanup context bind: {e:?}"),
        }
    }
}
