//! Canonical sealed borrow of a real device allocation, shared by transfer and compute.
use cudarc::driver::{sys, CudaSlice, DevicePtr, DeviceSlice};
pub(crate) mod sealed {
    pub trait Sealed {}
    impl<T> Sealed for cudarc::driver::CudaSlice<T> {}
}
pub trait CudaRegion: sealed::Sealed + Send + Sync {
    fn extent(&self) -> Result<(u64, u64), String>;
    fn context(&self) -> sys::CUcontext;
}
impl<T: Send + Sync> CudaRegion for CudaSlice<T> {
    fn extent(&self) -> Result<(u64, u64), String> {
        let bytes = self
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("external owner byte count overflow")?;
        let start = *self.device_ptr();
        let end = start
            .checked_add(
                u64::try_from(bytes)
                    .map_err(|_| "external owner extent exceeds CUDA address space")?,
            )
            .ok_or("external owner address overflow")?;
        Ok((start, end))
    }
    fn context(&self) -> sys::CUcontext {
        *self.device().cu_primary_ctx()
    }
}

// The prefill buffer itself owns allocation/context metadata. This borrow does
// not invent an allocation owner, extend its lifetime, or query CUDA per token.
impl<T: Copy> sealed::Sealed for crate::gpu_prefill::GpuBuf<T> {}
impl<T: Copy + Send + Sync> CudaRegion for crate::gpu_prefill::GpuBuf<T> {
    fn extent(&self) -> Result<(u64, u64), String> {
        let bytes = self
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("prefill region byte extent overflow")?;
        let start = *self.device_ptr();
        let end = start
            .checked_add(u64::try_from(bytes).map_err(|e| e.to_string())?)
            .ok_or("prefill region address extent overflow")?;
        Ok((start, end))
    }
    fn context(&self) -> sys::CUcontext {
        self.allocation_context()
    }
}
