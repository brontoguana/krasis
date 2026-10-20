//! Borrow actual device allocation owners, rather than pointer descriptors.
use crate::session_cache::SequenceStateAllocation;
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::sync::Arc;
use crate::cuda_region::sealed;
pub use crate::cuda_region::CudaRegion;
pub(crate) fn validate(
    device: &Arc<CudaDevice>,
    allocations: &[SequenceStateAllocation],
    owners: &[&dyn CudaRegion],
) -> Result<(), String> {
    let mut extents = Vec::with_capacity(owners.len());
    for owner in owners {
        if owner.context() != *device.cu_primary_ctx() {
            return Err(
                "external sequence allocation owner belongs to a different CUDA context".into(),
            );
        }
        extents.push(owner.extent()?);
    }
    for allocation in allocations {
        allocation.validate()?;
        if allocation.device_ordinal != device.ordinal() {
            return Err(format!(
                "external sequence allocation {} has a different CUDA ordinal",
                allocation.name
            ));
        }
        let end = allocation
            .ptr
            .checked_add(
                u64::try_from(allocation.storage_bytes)
                    .map_err(|_| "external allocation extent exceeds CUDA address space")?,
            )
            .ok_or("external allocation address overflow")?;
        if !extents
            .iter()
            .any(|&(start, stop)| start <= allocation.ptr && stop >= end)
        {
            return Err(format!(
                "external sequence allocation {} is not covered by a retained device owner",
                allocation.name
            ));
        }
    }
    Ok(())
}

// Own the *borrow* of a real legacy GpuBuf and its retained CUDA device; an
// allocation descriptor alone cannot create one of these admission objects.
trait RealGpuBuffer: Send + Sync {
    fn region_extent(&self) -> Result<(u64, u64), String>;
}
impl<T: Copy + Send + Sync> RealGpuBuffer for crate::gpu_prefill::GpuBuf<T> {
    fn region_extent(&self) -> Result<(u64, u64), String> {
        let start = *self.device_ptr();
        let count = self
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("native owner byte overflow")?;
        let end = start
            .checked_add(
                u64::try_from(count)
                    .map_err(|_| "native owner extent exceeds CUDA address space")?,
            )
            .ok_or("native owner address overflow")?;
        if start == 0 || count == 0 {
            return Err("native owner has no live allocation".into());
        }
        Ok((start, end))
    }
}
pub(crate) struct BorrowedGpuRegion<'a> {
    owner: &'a dyn RealGpuBuffer,
    _device: &'a Arc<CudaDevice>,
    context: sys::CUcontext,
}
impl sealed::Sealed for BorrowedGpuRegion<'_> {}
// Immutable driver identity, with the actual allocation and context owner
// borrowed throughout use. Every receiving phase validates/binds its context.
unsafe impl Send for BorrowedGpuRegion<'_> {}
unsafe impl Sync for BorrowedGpuRegion<'_> {}
impl CudaRegion for BorrowedGpuRegion<'_> {
    fn extent(&self) -> Result<(u64, u64), String> {
        self.owner.region_extent()
    }
    fn context(&self) -> sys::CUcontext {
        self.context
    }
}
impl<'a> BorrowedGpuRegion<'a> {
    /// Pure native hot-path admission of the real allocation and retained device.
    /// Context was measured at allocation, not inferred from a raw pointer.
    pub(crate) fn retained<T:Copy+Send+Sync>(device:&'a Arc<CudaDevice>,owner:&'a crate::gpu_prefill::GpuBuf<T>)->Result<Self,String>{
        owner.region_extent()?;let context=owner.allocation_context();
        if context.is_null()||context!=*device.cu_primary_ctx(){return Err("native allocation metadata differs from retained device context".into())}
        Ok(Self{owner,_device:device,context})
    }

    pub(crate) fn checked<T: Copy + Send + Sync>(
        device: &'a Arc<CudaDevice>,
        owner: &'a crate::gpu_prefill::GpuBuf<T>,
    ) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let (start, _) = owner.region_extent()?;
        let mut context = std::ptr::null_mut();
        unsafe {
            sys::lib().cuPointerGetAttribute(
                (&mut context as *mut sys::CUcontext).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                start,
            )
        }
        .result()
        .map_err(|e| format!("native owned allocation context: {e}"))?;
        // GpuBuf uses cuMemAlloc, not a pool allocation: a missing context is not
        // admissible here. This makes no assumption about ordinary CudaSlice pools.
        if context != *device.cu_primary_ctx() {
            return Err("native buffer owner has a different CUDA context".into());
        }
        Ok(Self {
            owner,
            _device: device,
            context,
        })
    }
}
