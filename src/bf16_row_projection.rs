//! One owned, batch-invariant BF16-to-F32 row projection program.
//! Uses the unchanged qualified router PTX without selecting the broader KDA bank.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, sync::Arc};
pub(crate) struct Program {
    _device: Arc<CudaDevice>,
    function: CudaFunction,
    limits: [usize; 2],
    pub(crate) identity: String,
}
impl Program {
    pub(crate) fn load(device: Arc<CudaDevice>) -> Result<Arc<Self>, String> {
        #[cfg(not(feature = "scalar-gdn-test"))]
        {
            let _ = device;
            Err("native BF16 row projection requires the explicitly built scalar candidate".into())
        }
        #[cfg(feature = "scalar-gdn-test")]
        {
            use sys::CUdevice_attribute as A;
            device.bind_to_thread().map_err(|e| e.to_string())?;
            let read = |a| {
                device
                    .attribute(a)
                    .map_err(|e| e.to_string())
                    .and_then(|v| usize::try_from(v).map_err(|e| e.to_string()))
            };
            if read(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8 {
                return Err("native BF16 row projection requires Ampere or later".into());
            }
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/ordered_router_f32.ptx"));
            device
                .load_ptx(
                    ptx.into(),
                    "native_bf16_row_projection_v1",
                    &["bf16_projection_tensorcomp_f32"],
                )
                .map_err(|e| e.to_string())?;
            let function = device
                .get_func(
                    "native_bf16_row_projection_v1",
                    "bf16_projection_tensorcomp_f32",
                )
                .ok_or("native BF16 row projection entry missing")?;
            let mut threads = 0;
            let mut active = 0;
            unsafe {
                sys::lib().cuFuncGetAttribute(
                    &mut threads,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    function.cu_function(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            // 512 threads / 64-row tile are the qualified kernel ABI, not a
            // model, GPU budget, measurement replacement or dispatch heuristic.
            unsafe {
                sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut active,
                    function.cu_function(),
                    512,
                    0,
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            if threads < 512 || active <= 0 {
                return Err(
                    "actual GPU cannot admit qualified BF16 row projection resources".into(),
                );
            }
            let limits = [
                read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            ];
            Ok(Arc::new(Self {
                _device: device,
                function,
                limits,
                identity: format!(
                    "bf16-k16-compensated-f32-v1:{:x}",
                    Sha256::digest(ptx.as_bytes())
                ),
            }))
        }
    }
    /// # Safety
    /// Actual immutable row-major BF16 weights/input and disjoint F32 output
    /// have the specified extents in this program's CUDA context. Their model,
    /// graph and stream owners survive enclosing completion/error/unwind.
    /// Stream belongs to this context, including its legacy default stream when
    /// the existing caller explicitly uses it. No new stream or allocation.
    pub(crate) unsafe fn enqueue(
        &self,
        weight: u64,
        input: u64,
        output: u64,
        rows: usize,
        cols: usize,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if rows == 0 || cols == 0 || tokens == 0 {
            return Err("native BF16 row projection requires real nonempty geometry".into());
        }
        let mut n = i32::try_from(rows).map_err(|_| "BF16 projection rows exceed CUDA ABI")?;
        let mut k = i32::try_from(cols).map_err(|_| "BF16 projection cols exceed CUDA ABI")?;
        let mut m = u64::try_from(tokens).map_err(|_| "BF16 projection tokens exceed CUDA ABI")?;
        for (ptr, count, bytes) in [
            (weight, rows.checked_mul(cols), 2usize),
            (input, tokens.checked_mul(cols), 2),
            (output, tokens.checked_mul(rows), 4),
        ] {
            if ptr == 0 || ptr % bytes as u64 != 0 {
                return Err("native BF16 projection pointer/alignment invalid".into());
            }
            let extent = count
                .and_then(|x| x.checked_mul(bytes))
                .ok_or("native BF16 projection extent overflow")?;
            ptr.checked_add(
                u64::try_from(extent)
                    .map_err(|_| "native BF16 projection address extent exceeds CUDA ABI")?,
            )
            .ok_or("native BF16 projection address overflow")?;
        }
        let gx = tokens.div_ceil(64);
        let gy = rows.div_ceil(64);
        if gx > self.limits[0] || gy > self.limits[1] {
            return Err("native BF16 projection grid exceeds actual device limits".into());
        }
        let gx = u32::try_from(gx).map_err(|_| "native BF16 projection X grid exceeds CUDA ABI")?;
        let gy = u32::try_from(gy).map_err(|_| "native BF16 projection Y grid exceeds CUDA ABI")?;
        let mut weight = weight;
        let mut input = input;
        let mut output = output;
        let mut args: [*mut c_void; 6] = [
            (&mut input as *mut u64).cast(),
            (&mut weight as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
            (&mut m as *mut u64).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.function.cu_function(),
                gx,
                gy,
                1,
                512,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| format!("native BF16 row projection:{e}"))
    }
}
