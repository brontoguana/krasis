//! Runtime binding for CUDA's pointer-to-pointer batched asynchronous copy.
//!
//! Krasis builds against CUDA 12.6 bindings for wheel portability, while
//! `cuMemcpyBatchAsync` was added in CUDA 12.8. Resolve the optional driver
//! entry point at runtime so callers can measure support without raising the
//! minimum build-toolkit version.

use cudarc::driver::sys as cuda_sys;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CuMemLocation {
    location_type: i32,
    id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CuMemcpyAttributes {
    src_access_order: i32,
    src_location_hint: CuMemLocation,
    dst_location_hint: CuMemLocation,
    flags: u32,
}

type CuMemcpyBatchAsyncFn = unsafe extern "C" fn(
    *mut cuda_sys::CUdeviceptr,
    *mut cuda_sys::CUdeviceptr,
    *mut usize,
    usize,
    *mut CuMemcpyAttributes,
    *mut usize,
    usize,
    *mut usize,
    cuda_sys::CUstream,
) -> cuda_sys::CUresult;

pub struct CudaBatchCopy {
    _library: libloading::Library,
    copy: CuMemcpyBatchAsyncFn,
}

impl CudaBatchCopy {
    pub fn load() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        let candidates = ["nvcuda.dll"];
        #[cfg(not(target_os = "windows"))]
        let candidates = ["libcuda.so.1", "libcuda.so"];

        let mut errors = Vec::new();
        for candidate in candidates {
            let library = match unsafe { libloading::Library::new(candidate) } {
                Ok(library) => library,
                Err(error) => {
                    errors.push(format!("{candidate}: {error}"));
                    continue;
                }
            };
            let copy = unsafe {
                library
                    .get::<CuMemcpyBatchAsyncFn>(b"cuMemcpyBatchAsync_ptsz\0")
                    .or_else(|_| library.get::<CuMemcpyBatchAsyncFn>(b"cuMemcpyBatchAsync\0"))
                    .map(|symbol| *symbol)
            };
            match copy {
                Ok(copy) => {
                    return Ok(Self {
                        _library: library,
                        copy,
                    })
                }
                Err(error) => errors.push(format!("{candidate}: {error}")),
            }
        }
        Err(format!(
            "CUDA driver does not expose cuMemcpyBatchAsync (requires CUDA 12.8+ driver API): {}",
            errors.join("; ")
        ))
    }

    /// Queue independent pointer-to-pointer copies as one CUDA driver batch.
    ///
    /// Host sources must remain valid until the stream reaches the copy. The
    /// caller owns registration/pinning and all stream ordering.
    pub fn enqueue(
        &self,
        destinations: &mut [cuda_sys::CUdeviceptr],
        sources: &mut [cuda_sys::CUdeviceptr],
        sizes: &mut [usize],
        stream: cuda_sys::CUstream,
        prefer_compute_overlap: bool,
    ) -> Result<(), String> {
        if destinations.is_empty()
            || destinations.len() != sources.len()
            || destinations.len() != sizes.len()
            || destinations.iter().any(|pointer| *pointer == 0)
            || sources.iter().any(|pointer| *pointer == 0)
            || sizes.contains(&0)
            || stream.is_null()
        {
            return Err(format!(
                "invalid CUDA batch-copy contract: destinations={} sources={} sizes={} stream={stream:p}",
                destinations.len(),
                sources.len(),
                sizes.len(),
            ));
        }
        // CU_MEMCPY_SRC_ACCESS_ORDER_ANY is valid for long-lived registered
        // host sources and permits the driver to optimize their scheduling.
        let mut attributes = CuMemcpyAttributes {
            src_access_order: 3,
            src_location_hint: CuMemLocation::default(),
            dst_location_hint: CuMemLocation::default(),
            flags: u32::from(prefer_compute_overlap),
        };
        let mut attribute_index = 0usize;
        let mut failure_index = usize::MAX;
        let result = unsafe {
            (self.copy)(
                destinations.as_mut_ptr(),
                sources.as_mut_ptr(),
                sizes.as_mut_ptr(),
                destinations.len(),
                &mut attributes,
                &mut attribute_index,
                1,
                &mut failure_index,
                stream,
            )
        };
        if result == cuda_sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!(
                "cuMemcpyBatchAsync failed at copy index {failure_index}: {result:?}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_layout_matches_cuda_12_8_abi() {
        assert_eq!(std::mem::size_of::<CuMemLocation>(), 8);
        assert_eq!(std::mem::size_of::<CuMemcpyAttributes>(), 24);
        assert_eq!(std::mem::align_of::<CuMemcpyAttributes>(), 4);
    }
}
