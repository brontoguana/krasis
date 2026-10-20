// Exact frozen120 native helper; only visibility changes for this standalone probe.
use cudarc::driver::sys as cuda_sys;
#[allow(clippy::too_many_arguments)]
pub(super) fn run_cublas_bf16_row_independent_gemm(
    handle: cudarc::cublas::sys::cublasHandle_t,
    stream: cuda_sys::CUstream,
    input: u64,
    weight: u64,
    output: u64,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    if rows == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if input == 0 || weight == 0 || output == 0 {
        return Err(format!(
            "row-independent BF16 GEMM received a null pointer: input=0x{input:x} weight=0x{weight:x} output=0x{output:x}"
        ));
    }
    let rows_i32 = i32::try_from(rows)
        .map_err(|_| format!("row-independent BF16 GEMM rows exceed i32: {rows}"))?;
    let n_i32 =
        i32::try_from(n).map_err(|_| format!("row-independent BF16 GEMM N exceeds i32: {n}"))?;
    let k_i32 =
        i32::try_from(k).map_err(|_| format!("row-independent BF16 GEMM K exceeds i32: {k}"))?;
    let k_stride = i64::try_from(k)
        .map_err(|_| format!("row-independent BF16 GEMM input stride exceeds i64: {k}"))?;
    let n_stride = i64::try_from(n)
        .map_err(|_| format!("row-independent BF16 GEMM output stride exceeds i64: {n}"))?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        use cudarc::cublas::result as cublas_result;
        use cudarc::cublas::sys as cublas_sys;

        cublas_result::set_stream(handle, stream as cublas_sys::cudaStream_t)
            .map_err(|error| format!("row-independent BF16 GEMM set stream: {error:?}"))?;
        cublas_result::gemm_strided_batched_ex(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n_i32,
            1,
            k_i32,
            &alpha as *const f32 as *const std::ffi::c_void,
            weight as *const std::ffi::c_void,
            cublas_sys::cudaDataType::CUDA_R_16BF,
            k_i32,
            0,
            input as *const std::ffi::c_void,
            cublas_sys::cudaDataType::CUDA_R_16BF,
            k_i32,
            k_stride,
            &beta as *const f32 as *const std::ffi::c_void,
            output as *mut std::ffi::c_void,
            cublas_sys::cudaDataType::CUDA_R_16BF,
            n_i32,
            n_stride,
            rows_i32,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|error| format!("row-independent BF16 GEMM: {error:?}"))?;
    }
    Ok(())
}
