//! Owned fixed-contraction F32 index scorer. No allocation or driver query on enqueue.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use std::sync::Arc;

pub(crate) struct Program {
    _device: Arc<CudaDevice>,
    function: CudaFunction,
    limits: [usize; 3],
}

/// Only valid while its Program owner and source/output owners remain alive
/// through stream completion. The selected engine retains that Program.
#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    function: sys::CUfunction,
    limits: [usize; 3],
}
// Immutable CUDA function/context identity; enqueue does not mutate host state.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Program {
    pub(crate) fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        use sys::CUdevice_attribute as A;
        let attr = |key| -> Result<usize, String> {
            usize::try_from(device.attribute(key).map_err(|e| e.to_string())?)
                .map_err(|_| "negative score device attribute".into())
        };
        if attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8
            || attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 512
        {
            return Err("device cannot execute qualified BF16 score tile".into());
        }
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        device
            .load_ptx(ptx.into(), "fixed_dsa_score", &["dsa_score_tensorcomp_f32"])
            .map_err(|e| e.to_string())?;
        let function = device
            .get_func("fixed_dsa_score", "dsa_score_tensorcomp_f32")
            .ok_or("qualified score symbol absent")?;
        let mut threads = 0;
        let mut occupancy = 0;
        unsafe {
            sys::lib().cuFuncGetAttribute(
                &mut threads,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                function.cu_function(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        unsafe {
            sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut occupancy,
                function.cu_function(),
                512,
                0,
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        if threads < 512 || occupancy <= 0 || limits.contains(&0) {
            return Err("score tile has no admitted launch resources".into());
        }
        Ok(Self {
            _device: device,
            function,
            limits,
        })
    }
    pub(crate) fn kernel(&self) -> Kernel {
        Kernel {
            function: self.function.cu_function(),
            limits: self.limits,
        }
    }
}

fn product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .ok_or("score extent multiplication overflow".into())
    })
}
fn span(ptr: u64, elements: usize, bytes: usize) -> Result<(u64, u64), String> {
    if ptr == 0 || ptr % bytes as u64 != 0 {
        return Err("score pointer/alignment invalid".into());
    }
    let count = u64::try_from(product(&[elements, bytes])?)
        .map_err(|_| "score extent exceeds CUDA address space")?;
    Ok((ptr, ptr.checked_add(count).ok_or("score address overflow")?))
}
fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

impl Kernel {
    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        key: u64,
        query: u64,
        output: u64,
        rows: usize,
        context: usize,
        heads: usize,
        dim: usize,
        first: usize,
        count: usize,
        scale: f32,
    ) -> Result<[u32; 3], String> {
        if rows == 0
            || context == 0
            || heads == 0
            || dim == 0
            || count == 0
            || first.checked_add(count).is_none_or(|end| end > heads)
            || [context, heads, dim, first, count]
                .iter()
                .any(|&n| n > i32::MAX as usize)
            || !scale.is_finite()
            || scale <= 0.0
            || self.limits.contains(&0)
        {
            return Err("invalid fixed score geometry or scale".into());
        }
        let inputs = [
            span(key, product(&[context, dim])?, 2)?,
            span(query, product(&[rows, heads, dim])?, 2)?,
        ];
        let destination = span(output, product(&[count, rows, context])?, 4)?;
        if inputs.iter().any(|&source| overlap(source, destination)) {
            return Err("fixed score output aliases retained input".into());
        }
        let tokens = product(&[rows, count])?;
        let gx = tokens.div_ceil(64);
        let tiles = context.div_ceil(64);
        let gy = tiles.min(self.limits[1]);
        let gz = tiles.div_ceil(gy);
        if gx > self.limits[0]
            || gz > self.limits[2]
            || [gx, gy, gz].iter().any(|&v| v > u32::MAX as usize)
        {
            return Err("fixed score grid exceeds device limits".into());
        }
        Ok([gx as u32, gy as u32, gz as u32])
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &self,
        key: u64,
        query: u64,
        output: u64,
        rows: usize,
        context: usize,
        heads: usize,
        dim: usize,
        first: usize,
        count: usize,
        scale: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let grid = self.validate(
            key, query, output, rows, context, heads, dim, first, count, scale,
        )?;
        if self.function.is_null() {
            return Err("fixed score function absent".into());
        }
        let (mut x, mut w, mut y) = (query, key, output);
        let (mut n, mut k, mut tokens, mut h, mut start, mut group, mut alpha) = (
            context as i32,
            dim as i32,
            product(&[rows, count])? as u64,
            heads as i32,
            first as i32,
            count as i32,
            scale,
        );
        macro_rules! arg {
            ($v:ident) => {
                &mut $v as *mut _ as *mut std::ffi::c_void
            };
        }
        sys::lib()
            .cuLaunchKernel(
                self.function,
                grid[0],
                grid[1],
                grid[2],
                512,
                1,
                1,
                0,
                stream,
                [
                    arg!(x),
                    arg!(w),
                    arg!(y),
                    arg!(n),
                    arg!(k),
                    arg!(tokens),
                    arg!(h),
                    arg!(start),
                    arg!(group),
                    arg!(alpha),
                ]
                .as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| format!("fixed index score launch: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kernel() -> Kernel {
        Kernel {
            function: std::ptr::null_mut(),
            limits: [i32::MAX as usize, 65535, 65535],
        }
    }
    #[test]
    fn head_tail_and_odd_dimensions_use_actual_strides() {
        assert_eq!(
            kernel()
                .validate(0x1000, 0x10000, 0x100000, 67, 19, 3, 111, 2, 1, 0.5)
                .unwrap(),
            [2, 1, 1]
        );
        assert!(kernel()
            .validate(0x1000, 0x10000, 0x100000, 67, 19, 3, 111, 2, 2, 0.5)
            .is_err());
    }
    #[test]
    fn aliased_and_wrapped_extents_are_rejected() {
        assert!(kernel()
            .validate(0x1000, 0x10000, 0x1000, 67, 19, 3, 111, 0, 2, 0.5)
            .is_err());
        assert!(kernel()
            .validate(u64::MAX - 1, 0x10000, 0x100000, 67, 19, 3, 111, 0, 2, 0.5)
            .is_err());
        assert!(kernel()
            .validate(0x1000, 0x10000, 0x100000, usize::MAX, 19, 3, 111, 0, 2, 0.5)
            .is_err());
    }
    #[test]
    fn oversized_key_grid_folds_over_actual_y_limit() {
        let mut k = kernel();
        k.limits[1] = 2;
        assert_eq!(
            k.validate(0x1000, 0x10000, 0x100000, 1, 193, 3, 7, 0, 1, 0.5)
                .unwrap(),
            [1, 2, 2]
        );
        k.limits[2] = 1;
        assert!(k
            .validate(0x1000, 0x10000, 0x100000, 1, 193, 3, 7, 0, 1, 0.5)
            .is_err());
    }
    #[test]
    fn invalid_scale_and_alignment_fail_before_launch() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(kernel()
                .validate(0x1000, 0x10000, 0x100000, 1, 17, 3, 111, 0, 1, scale)
                .is_err());
        }
        assert!(kernel()
            .validate(0x1001, 0x10000, 0x100000, 1, 17, 3, 111, 0, 1, 0.5)
            .is_err());
    }
}
