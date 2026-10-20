//! Shared canonical HQQ4 projection under operational qualification.
//! No dispatch/default change. Loaded weights retain their actual model owners;
//! the whole scalar execution domain must enclose every use and error path.
use crate::scalar_gdn_owner::Plane;
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, sync::Arc};

/// Cold-validated actual canonical row-aligned U4 / FP32 scale-zero weights.
/// Layout metadata does not invent an allocation owner: callers retain the real
/// configured model weights until the enclosing owned invocation completes.
pub(crate) struct Layout {
    pub packed: u64,
    pub scales: u64,
    pub zeros: u64,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub packed_stride: usize,
    pub scales_stride: usize,
    pub zeros_stride: usize,
}
/// Live prefill metadata must identify the same cold-admitted coefficients.
pub(crate) struct Binding<'a> {
    pub layout: &'a str,
    pub packed_dtype: &'a str,
    pub scales_dtype: &'a str,
    pub zeros_dtype: &'a str,
    pub pointers: [u64; 3],
    pub shape: [usize; 3],
    pub axis: usize,
    pub tensor_bytes: usize,
}
impl Layout {
    pub(crate) fn validate_live_binding(&self, binding: &Binding<'_>) -> Result<(), String> {
        self.validate()?;
        if binding.layout != "row_aligned_grouped_uint4_decode_v1"
            || binding.packed_dtype != "uint8"
            || binding.scales_dtype != "float32"
            || binding.zeros_dtype != "float32"
            || binding.axis != 1
        {
            return Err(
                "selected native projector requires canonical row-aligned UINT4/F32 descriptors"
                    .into(),
            );
        }
        if binding.pointers != [self.packed, self.scales, self.zeros]
            || binding.shape != [self.rows, self.cols, self.group]
        {
            return Err(
                "live prefill weights/geometry differ from the cold-admitted native producer"
                    .into(),
            );
        }
        let bytes = [self.packed_stride, self.scales_stride, self.zeros_stride]
            .into_iter()
            .try_fold(0usize, |sum, stride| {
                self.rows
                    .checked_mul(stride)
                    .and_then(|n| sum.checked_add(n))
                    .ok_or("canonical live binding extent overflow")
            })?;
        if binding.tensor_bytes < bytes {
            return Err(
                "live canonical tensor extent is smaller than its retained actual strides".into(),
            );
        }
        Ok(())
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.rows == 0
            || self.cols == 0
            || self.group == 0
            || self.cols % 4 != 0
            || self.group % 4 != 0
            || self.cols % self.group != 0
            || self.packed_stride < self.cols / 2
            || self.scales_stride % 4 != 0
            || self.scales_stride
                < (self.cols / self.group)
                    .checked_mul(4)
                    .ok_or("canonical parameter stride overflow")?
            || self.zeros_stride != self.scales_stride
            || self.packed == 0
            || self.scales == 0
            || self.zeros == 0
            || self.scales % 4 != 0
            || self.zeros % 4 != 0
        {
            return Err("canonical scalar projection requires complete row-aligned U4/F32 scale-zero geometry".into());
        }
        for n in [
            self.rows,
            self.cols,
            self.group,
            self.packed_stride,
            self.scales_stride,
        ] {
            i32::try_from(n).map_err(|_| "canonical scalar projection exceeds signed CUDA ABI")?;
        }
        for (ptr, stride) in [
            (self.packed, self.packed_stride),
            (self.scales, self.scales_stride),
            (self.zeros, self.zeros_stride),
        ] {
            ptr.checked_add(
                u64::try_from(
                    self.rows
                        .checked_mul(stride)
                        .ok_or("canonical weight extent overflow")?,
                )
                .map_err(|e| e.to_string())?,
            )
            .ok_or("canonical weight address overflow")?;
        }
        Ok(())
    }
}
pub(crate) struct Program {
    device: Arc<CudaDevice>,
    function: CudaFunction,
    grid: [usize; 3],
    pub(crate) identity: String,
}
impl Program {
    pub(crate) fn load(device: Arc<CudaDevice>) -> Result<Arc<Self>, String> {
        use sys::CUdevice_attribute as A;
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let read = |a| {
            device
                .attribute(a)
                .map_err(|e| e.to_string())
                .and_then(|v| usize::try_from(v).map_err(|e| e.to_string()))
        };
        if read(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8 {
            return Err("canonical scalar tensor-core projection requires Ampere or later".into());
        }
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/scalar_gdn_projection.ptx"));
        device
            .load_ptx(
                ptx.into(),
                "scalar_gdn_projection_v1",
                &["hqq4_canonical_ordered_tf32x3"],
            )
            .map_err(|e| e.to_string())?;
        let function = device
            .get_func("scalar_gdn_projection_v1", "hqq4_canonical_ordered_tf32x3")
            .ok_or("canonical scalar projection symbol missing")?;
        let mut shared = 0;
        unsafe {
            sys::lib()
                .cuFuncGetAttribute(
                    &mut shared,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                    function.cu_function(),
                )
                .result()
                .map_err(|e| e.to_string())?;
        }
        // Kernel dimensions are the compiled algorithm contract, not a memory
        // reserve. Actual static shared storage is queried from the loaded code.
        if read(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 256
            || read(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?
                < usize::try_from(shared).map_err(|e| e.to_string())?
        {
            return Err(
                "actual device cannot execute canonical scalar projection resources".into(),
            );
        }
        let grid = [
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        Ok(Arc::new(Self {
            device,
            function,
            grid,
            identity: format!(
                "u4-canonical-f32-k1-tf32x3-ordered-pair-bf16-v1:{:x}",
                Sha256::digest(ptx.as_bytes())
            ),
        }))
    }
    /// # Safety
    /// The current context and stream belong to this device. Immutable weight
    /// owners and every real plane remain held through whole-phase completion,
    /// including failures/unwinds. Output planes are exclusive and all input,
    /// output and weight ranges are disjoint. No buffer is reused prematurely.
    pub(crate) unsafe fn enqueue(
        &self,
        weight: &Layout,
        input: &Plane<'_, u16>,
        output: &Plane<'_, u16>,
        output_f32: Option<&Plane<'_, f32>>,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        weight.validate()?;
        if tokens == 0 || stream.is_null() {
            return Err(
                "canonical scalar projection needs a live owned stream and nonempty rows".into(),
            );
        }
        let context = *self.device.cu_primary_ctx();
        input.require_context_extent(
            context,
            tokens
                .checked_mul(weight.cols)
                .ok_or("canonical input size overflow")?,
        )?;
        let elements = tokens
            .checked_mul(weight.rows)
            .ok_or("canonical output size overflow")?;
        output.require_context_extent(context, elements)?;
        if let Some(p) = output_f32 {
            p.require_context_extent(context, elements)?;
        }
        let gx = tokens.div_ceil(32);
        let gy = weight.rows.div_ceil(64);
        if gx > self.grid[0] || gy > self.grid[1] {
            return Err("canonical scalar projection exceeds actual device grid".into());
        }
        let mut pointers = [
            weight.packed,
            weight.scales,
            weight.zeros,
            input.pointer(),
            output.pointer(),
            output_f32.map(Plane::pointer).unwrap_or(0),
        ];
        let mut dimensions = [
            weight.rows as i32,
            weight.cols as i32,
            weight.group as i32,
            weight.packed_stride as i32,
            weight.scales_stride as i32,
        ];
        let mut rows = u64::try_from(tokens).map_err(|e| e.to_string())?;
        let mut args = [std::ptr::null_mut::<c_void>(); 12];
        for (slot, pointer) in args[..6].iter_mut().zip(&mut pointers) {
            *slot = (pointer as *mut u64).cast();
        }
        for (slot, dimension) in args[6..11].iter_mut().zip(&mut dimensions) {
            *slot = (dimension as *mut i32).cast();
        }
        args[11] = (&mut rows as *mut u64).cast();
        sys::lib()
            .cuLaunchKernel(
                self.function.cu_function(),
                gx as u32,
                gy as u32,
                1,
                256,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| format!("canonical scalar projection:{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_stride_contract_accepts_padding_but_rejects_different_parameter_layout() {
        let mut w = Layout {
            packed: 0x1000,
            scales: 0x4000,
            zeros: 0x8000,
            rows: 64,
            cols: 256,
            group: 64,
            packed_stride: 144,
            scales_stride: 32,
            zeros_stride: 32,
        };
        w.validate().unwrap();
        w.zeros_stride = 16;
        assert!(w.validate().is_err());
    }
    #[test]
    fn live_prefill_binding_requires_canonical_weights_and_geometry() {
        let w = Layout {
            packed: 0x1000,
            scales: 0x4000,
            zeros: 0x8000,
            rows: 37,
            cols: 256,
            group: 64,
            packed_stride: 144,
            scales_stride: 32,
            zeros_stride: 32,
        };
        let mut b = Binding {
            layout: "row_aligned_grouped_uint4_decode_v1",
            packed_dtype: "uint8",
            scales_dtype: "float32",
            zeros_dtype: "float32",
            pointers: [w.packed, w.scales, w.zeros],
            shape: [w.rows, w.cols, w.group],
            axis: 1,
            tensor_bytes: w.rows * (w.packed_stride + w.scales_stride + w.zeros_stride),
        };
        w.validate_live_binding(&b).unwrap();
        b.layout = "row_major_axis1_grouped_uint4_packed";
        assert!(w.validate_live_binding(&b).is_err());
        b.layout = "row_aligned_grouped_uint4_decode_v1";
        b.pointers[1] += 4;
        assert!(w.validate_live_binding(&b).is_err());
        b.pointers[1] = w.scales;
        b.shape[0] -= 1;
        assert!(w.validate_live_binding(&b).is_err());
        b.shape[0] = w.rows;
        b.axis = 0;
        assert!(w.validate_live_binding(&b).is_err());
        b.axis = 1;
        b.scales_dtype = "bfloat16";
        assert!(w.validate_live_binding(&b).is_err());
        b.scales_dtype = "float32";
        b.tensor_bytes -= 1;
        assert!(w.validate_live_binding(&b).is_err());
    }
    #[test]
    fn weight_addresses_cannot_wrap() {
        let w = Layout {
            packed: u64::MAX - 1,
            scales: 0x4000,
            zeros: 0x8000,
            rows: 64,
            cols: 256,
            group: 64,
            packed_stride: 128,
            scales_stride: 16,
            zeros_stride: 16,
        };
        assert!(w.validate().is_err());
    }
}
