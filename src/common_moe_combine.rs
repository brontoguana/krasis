//! Actual F32 routed accumulator and BF16 shared/output views, borrowed by callers.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::sync::Arc;
pub(crate) struct Program {
    _device: Arc<CudaDevice>,
    reduce: CudaFunction,
    combine: CudaFunction,
    limits: [usize; 3],
    pub identity: String,
}
fn region(ptr: u64, count: Option<usize>, bytes: usize) -> Result<(), String> {
    if ptr == 0 || ptr % bytes as u64 != 0 {
        return Err("invalid common MoE region".into());
    }
    let n = count
        .and_then(|v| v.checked_mul(bytes))
        .filter(|&v| v > 0)
        .ok_or("common MoE extent overflow")?;
    ptr.checked_add(n as u64)
        .ok_or("common MoE pointer overflow")?;
    Ok(())
}
impl Program {
    pub(crate) fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        use sys::CUdevice_attribute as A;
        let attr = |a| -> Result<usize, String> {
            usize::try_from(device.attribute(a).map_err(|e| e.to_string())?)
                .map_err(|_| "negative CUDA limit".into())
        };
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        device
            .load_ptx(
                ptx.into(),
                "common_moe_combine",
                &["common_moe_reduce", "common_moe_combine"],
            )
            .map_err(|e| e.to_string())?;
        let reduce = device
            .get_func("common_moe_combine", "common_moe_reduce")
            .ok_or("common MoE reduce symbol missing")?;
        let combine = device
            .get_func("common_moe_combine", "common_moe_combine")
            .ok_or("common MoE combine symbol missing")?;
        for f in [&reduce, &combine] {
            let mut active = 0;
            unsafe {
                sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut active,
                    f.cu_function(),
                    256,
                    0,
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            if active <= 0 {
                return Err("common MoE function cannot admit tile".into());
            }
        }
        Ok(Self {
            _device: device,
            reduce,
            combine,
            limits,
            identity: format!(
                "router_slot_f32_fma_then_shared_bf16_rne_v1:{:x}",
                Sha256::digest(ptx.as_bytes())
            ),
        })
    }
    /// SAFETY: actual same-context expert outputs, route weights and unique F32
    /// accumulator/BF16 mirror owners are held through completion on this stream.
    pub(crate) unsafe fn enqueue_reduce(
        &self,
        experts: u64,
        weights: u64,
        routed: u64,
        mirror: u64,
        width: usize,
        topk: usize,
        tokens: usize,
        scale: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if width == 0
            || width > i32::MAX as usize
            || topk == 0
            || topk > i32::MAX as usize
            || tokens == 0
            || !scale.is_finite()
        {
            return Err("invalid common MoE reduction geometry/scale".into());
        }
        let count = tokens.checked_mul(width);
        region(experts, count.and_then(|v| v.checked_mul(topk)), 2)?;
        region(weights, tokens.checked_mul(topk), 4)?;
        region(routed, count, 4)?;
        if mirror != 0 {
            region(mirror, count, 2)?;
        }
        let gx = width.div_ceil(256);
        let (gy,gz)=crate::common_marlin::row_grid(tokens,self.limits[1],self.limits[2])?;
        if gx > self.limits[0] || gz as usize > self.limits[2] {
            return Err("common MoE reduction grid limit".into());
        }
        let mut e = experts;
        let mut w = weights;
        let mut r = routed;
        let mut m = mirror;
        let mut n = width as i32;
        let mut k = topk as i32;
        let mut rows = tokens as u64;
        let mut scale = scale;
        let mut args = [
            (&mut e as *mut u64).cast(),
            (&mut w as *mut u64).cast(),
            (&mut r as *mut u64).cast(),
            (&mut m as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
            (&mut rows as *mut u64).cast(),
            (&mut scale as *mut f32).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.reduce.cu_function(),
                gx as u32,
                gy as u32,
                gz as u32,
                256,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| e.to_string())
    }
    /// SAFETY: actual F32 routed and optional BF16 shared/F32 gate owners are held;
    /// BF16 output is unique and disjoint through completion on the actual stream.
    pub(crate) unsafe fn enqueue_combine(
        &self,
        routed: u64,
        shared: u64,
        gate: u64,
        output: u64,
        combined: u64,
        width: usize,
        tokens: usize,
        shared_weight: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if width == 0
            || width > i32::MAX as usize
            || tokens == 0
            || !shared_weight.is_finite()
            || (shared == 0 && gate != 0)
            || (output == 0 && combined == 0)
        {
            return Err("invalid common MoE combination".into());
        }
        let count = tokens.checked_mul(width);
        region(routed, count, 4)?;
        if output != 0 {
            region(output, count, 2)?;
        }
        if combined != 0 {
            region(combined, count, 4)?;
        }
        if shared != 0 {
            region(shared, count, 2)?;
        }
        if gate != 0 {
            region(gate, Some(tokens), 4)?;
        }
        let gx = width.div_ceil(256);
        let (gy,gz)=crate::common_marlin::row_grid(tokens,self.limits[1],self.limits[2])?;
        if gx > self.limits[0] || gz as usize > self.limits[2] {
            return Err("common MoE combine grid limit".into());
        }
        let mut r = routed;
        let mut s = shared;
        let mut g = gate;
        let mut out = output;
        let mut combined = combined;
        let mut n = width as i32;
        let mut rows = tokens as u64;
        let mut weight = shared_weight;
        let mut args = [
            (&mut r as *mut u64).cast(),
            (&mut s as *mut u64).cast(),
            (&mut g as *mut u64).cast(),
            (&mut out as *mut u64).cast(),
            (&mut combined as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut rows as *mut u64).cast(),
            (&mut weight as *mut f32).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.combine.cu_function(),
                gx as u32,
                gy as u32,
                gz as u32,
                256,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| e.to_string())
    }
}
