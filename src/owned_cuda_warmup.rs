//! One-time INT8 capability warmup with explicit CUDA/cuBLAS ownership.
//! Shapes come from loaded tensors; no workspace budget or GPU assumptions.
use cudarc::cublas::{sys as blas, CudaBlas};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use pyo3::prelude::*;
use std::sync::Arc;

struct CompleteBeforeRelease(Arc<CudaDevice>);
impl Drop for CompleteBeforeRelease {
    fn drop(&mut self) {
        if let Err(error) = self.0.bind_to_thread().and_then(|_| self.0.synchronize()) {
            crate::vram_monitor::fatal_cuda_context_error(
                "owned INT8 warmup completion",
                &error.to_string(),
            );
        }
    }
}
fn product(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .filter(|&n| n > 0)
        .ok_or_else(|| "INT8 warmup has empty/overflowing geometry".into())
}
fn run(device_index: usize, k: usize, n: usize, batches: &[usize]) -> Result<Vec<usize>, String> {
    let ki = i32::try_from(k).map_err(|_| "INT8 warmup K exceeds CUDA ABI")?;
    let ni = i32::try_from(n).map_err(|_| "INT8 warmup N exceeds CUDA ABI")?;
    if ki <= 0 || ni <= 0 || batches.is_empty() {
        return Err("INT8 warmup requires positive geometry and batch probes".into());
    }
    let device = CudaDevice::new(device_index).map_err(|e| e.to_string())?;
    let handle = CudaBlas::new(device.clone()).map_err(|e| e.to_string())?;
    for &m in batches {
        let mi = i32::try_from(m).map_err(|_| "INT8 warmup M exceeds CUDA ABI")?;
        let x_len = product(m, k)?;
        let w_len = product(n, k)?;
        let y_len = product(m, n)?;
        // Declare owners before the guard: on every exit its completion runs
        // before allocations, host upload data, or the cuBLAS handle release.
        let mut x: Option<CudaSlice<i8>> = None;
        let mut w: Option<CudaSlice<i8>> = None;
        let mut y: Option<CudaSlice<i32>> = None;
        let completion = CompleteBeforeRelease(device.clone());
        x = Some(
            device
                .htod_copy(vec![1i8; x_len])
                .map_err(|e| e.to_string())?,
        );
        w = Some(
            device
                .htod_copy(vec![1i8; w_len])
                .map_err(|e| e.to_string())?,
        );
        y = Some(
            device
                .alloc_zeros::<i32>(y_len)
                .map_err(|e| e.to_string())?,
        );
        let alpha = 1i32;
        let beta = 0i32;
        let status = unsafe {
            // Row-major X[M,K] * W[N,K]^T, expressed as column-major
            // W^T[N,K] * X[K,M]. Output bytes are row-major [M,N].
            blas::lib().cublasGemmEx(
                *handle.handle(),
                blas::cublasOperation_t::CUBLAS_OP_T,
                blas::cublasOperation_t::CUBLAS_OP_N,
                ni,
                mi,
                ki,
                (&alpha as *const i32).cast(),
                (*w.as_ref().unwrap().device_ptr() as *const std::ffi::c_void),
                blas::cudaDataType::CUDA_R_8I,
                ki,
                (*x.as_ref().unwrap().device_ptr() as *const std::ffi::c_void),
                blas::cudaDataType::CUDA_R_8I,
                ki,
                (&beta as *const i32).cast(),
                (*y.as_ref().unwrap().device_ptr() as *mut std::ffi::c_void),
                blas::cudaDataType::CUDA_R_32I,
                ni,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if status != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!(
                "owned INT8 warmup [{m},{k}]x[{n},{k}] failed:{status:?}"
            ));
        }
        let actual = device
            .dtoh_sync_copy(y.as_ref().unwrap())
            .map_err(|e| e.to_string())?;
        if actual.iter().any(|&value| value != ki) {
            return Err("owned INT8 warmup returned incorrect integer products".into());
        }
        drop(completion);
    }
    Ok(batches.to_vec())
}

#[pyfunction]
pub(crate) fn warmup_int8_cuda(
    py: Python<'_>,
    device_index: usize,
    k: usize,
    n: usize,
    batch_sizes: Vec<usize>,
) -> PyResult<Vec<usize>> {
    py.allow_threads(|| run(device_index, k, n, &batch_sizes))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
