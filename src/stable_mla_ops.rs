//! Common selected F32 gather/codec and boundary conversions, without allocation or synchronization.
use crate::stable_mla_gemm::{Element, Span};
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("stable MLA extent overflow".into())
}
fn sum(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b).ok_or("stable MLA extent overflow".into())
}
fn integer(v: usize) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| "stable MLA dimension exceeds i32".into())
}
fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Codec {
    K4,
    K6,
}
impl Codec {
    fn bytes(self) -> usize {
        match self {
            Self::K4 => 10,
            Self::K6 => 14,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gather {
    pub row_start: usize,
    pub rows: usize,
    pub heads: usize,
    pub ccd: usize,
    pub rope: usize,
    pub slots: usize,
    pub context: usize,
    pub q_abs_row_stride: usize,
    pub q_abs_head_stride: usize,
    pub q_rope_row_stride: usize,
    pub q_rope_head_stride: usize,
    pub codec: Codec,
}
fn strided(
    rows: usize,
    heads: usize,
    width: usize,
    row_stride: usize,
    head_stride: usize,
) -> Result<usize, String> {
    if rows == 0 || heads == 0 || width == 0 {
        return Err("empty stable MLA strided view".into());
    }
    sum(
        sum(mul(rows - 1, row_stride)?, mul(heads - 1, head_stride)?)?,
        width,
    )
}
impl Gather {
    pub(crate) fn validate(self) -> Result<(), String> {
        if self.rows == 0
            || self.heads == 0
            || self.ccd == 0
            || self.slots == 0
            || self.context == 0
            || self.ccd % 16 != 0
            || self.rope % 16 != 0
        {
            return Err("invalid stable MLA gather geometry/codec blocks".into());
        }
        for v in [
            sum(self.row_start, self.rows)?,
            self.heads,
            self.ccd,
            self.rope,
            self.slots,
            self.context,
            sum(self.ccd, self.rope)?,
        ] {
            integer(v)?;
        }
        self.outputs()?;
        self.inputs()?;
        Ok(())
    }
    pub(crate) fn outputs(self) -> Result<[usize; 3], String> {
        let width = sum(self.ccd, self.rope)?;
        Ok([
            mul(mul(self.rows, self.slots)?, width)?,
            mul(mul(self.rows, self.heads)?, width)?,
            mul(mul(self.rows, self.heads)?, self.slots)?,
        ])
    }
    fn inputs(self) -> Result<[usize; 6], String> {
        let end = sum(self.row_start, self.rows)?;
        Ok([
            strided(
                end,
                self.heads,
                self.ccd,
                self.q_abs_row_stride,
                self.q_abs_head_stride,
            )?,
            if self.rope == 0 {
                0
            } else {
                strided(
                    end,
                    self.heads,
                    self.rope,
                    self.q_rope_row_stride,
                    self.q_rope_head_stride,
                )?
            },
            mul(mul(self.context, self.ccd / 16)?, self.codec.bytes())?,
            mul(mul(self.context, self.rope / 16)?, self.codec.bytes())?,
            mul(end, self.slots)?,
            end,
        ])
    }
}
pub(crate) struct Ops {
    device: Arc<CudaDevice>,
    functions: [CudaFunction; 5],
    grid_limits: [usize; 3],
    max_pitch: usize,
    pub(crate) identity: String,
}
impl Ops {
    pub(crate) fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Arc<Self>, String> {
        let mut current = std::ptr::null_mut();
        unsafe { sys::lib().cuCtxGetCurrent(&mut current) }
            .result()
            .map_err(|e| e.to_string())?;
        if current != *device.cu_primary_ctx() {
            return Err("stable MLA kernel load requires actual owning context".into());
        }
        use sys::CUdevice_attribute as A;
        let attr = |v| {
            device
                .attribute(v)
                .map_err(|e| e.to_string())
                .and_then(|n| usize::try_from(n).map_err(|e| e.to_string()))
        };
        if attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8
            || attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 256
        {
            return Err(
                "stable MLA ABI requires admitted Ampere-or-newer 256-thread execution".into(),
            );
        }
        let names = [
            "stable_mla_gather_k4",
            "stable_mla_gather_k6",
            "stable_mla_widen_bf16",
            "stable_mla_scatter_bf16",
            "stable_mla_softmax",
        ];
        device
            .load_ptx(ptx.to_owned().into(), "stable_mla_ops", &names)
            .map_err(|e| e.to_string())?;
        let functions = names
            .map(|name| {
                device
                    .get_func("stable_mla_ops", name)
                    .ok_or_else(|| format!("stable MLA symbol absent: {name}"))
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        for f in &functions {
            let mut threads = 0;
            unsafe {
                sys::lib().cuFuncGetAttribute(
                    &mut threads,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    f.cu_function(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
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
            if threads < 256 || active <= 0 {
                return Err("loaded stable MLA function cannot admit its thread ABI".into());
            }
        }
        let grid_limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        if grid_limits.iter().any(|&n| n == 0) {
            return Err("invalid actual CUDA grid limits".into());
        }
        let max_pitch = attr(A::CU_DEVICE_ATTRIBUTE_MAX_PITCH)?;
        if max_pitch == 0 {
            return Err("invalid actual CUDA maximum copy pitch".into());
        }
        Ok(Arc::new(Self {
            device,
            max_pitch,
            functions: functions
                .try_into()
                .map_err(|_| "stable MLA symbol count")?,
            grid_limits,
            identity: format!(
                "stable_mla_f32_gather_v1:{:x}",
                Sha256::digest(ptx.as_bytes())
            ),
        }))
    }
    fn current(&self, stream: sys::CUstream) -> Result<(), String> {
        let mut current = std::ptr::null_mut();
        unsafe { sys::lib().cuCtxGetCurrent(&mut current) }
            .result()
            .map_err(|e| e.to_string())?;
        if current != *self.device.cu_primary_ctx() {
            return Err("stable MLA current context mismatch".into());
        }
        if !stream.is_null() {
            let mut ctx = std::ptr::null_mut();
            unsafe { sys::lib().cuStreamGetCtx(stream, &mut ctx) }
                .result()
                .map_err(|e| e.to_string())?;
            if ctx != current {
                return Err("stable MLA foreign stream".into());
            }
        }
        Ok(())
    }
    fn range(
        &self,
        v: &Span<'_>,
        elements: usize,
        bytes: usize,
        element: Element,
    ) -> Result<(u64, u64), String> {
        if v.element() != element {
            return Err("stable MLA operand dtype mismatch".into());
        }
        if v.context() != *self.device.cu_primary_ctx() {
            return Err("stable MLA tensor has foreign owner context".into());
        }
        if v.pointer() % bytes as u64 != 0 {
            return Err("stable MLA tensor alignment mismatch".into());
        }
        v.range(mul(elements, bytes)?)
    }
    fn grid(&self, blocks: usize) -> Result<[u32; 3], String> {
        if blocks == 0 {
            return Err("stable MLA empty launch".into());
        }
        let x = blocks.min(self.grid_limits[0]);
        let remaining = blocks.div_ceil(x);
        let y = remaining.min(self.grid_limits[1]);
        let z = remaining.div_ceil(y);
        if z > self.grid_limits[2] {
            return Err("stable MLA launch exceeds actual grid limits".into());
        }
        Ok([
            u32::try_from(x).map_err(|e| e.to_string())?,
            u32::try_from(y).map_err(|e| e.to_string())?,
            u32::try_from(z).map_err(|e| e.to_string())?,
        ])
    }
    unsafe fn launch(
        &self,
        index: usize,
        blocks: usize,
        args: &mut [*mut std::ffi::c_void],
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let g = self.grid(blocks)?;
        sys::lib()
            .cuLaunchKernel(
                self.functions[index].cu_function(),
                g[0],
                g[1],
                g[2],
                256,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| format!("stable MLA enqueue failed: {e:?}"))
    }
    /// The caller retains all borrowed allocations and this bank through actual stream/graph completion, including errors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gather(
        &self,
        p: Gather,
        gathered: &Span<'_>,
        query: &Span<'_>,
        initial: &Span<'_>,
        q_abs: &Span<'_>,
        q_rope: Option<&Span<'_>>,
        ckv: &Span<'_>,
        kpe: Option<&Span<'_>>,
        selected: &Span<'_>,
        positions: &Span<'_>,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        p.validate()?;
        self.current(stream)?;
        let out = p.outputs()?;
        let input = p.inputs()?;
        let outputs = [
            self.range(gathered, out[0], 4, Element::F32)?,
            self.range(query, out[1], 4, Element::F32)?,
            self.range(initial, out[2], 4, Element::F32)?,
        ];
        if ckv.pointer() % 2 != 0 {
            return Err("stable MLA compact cache alignment invalid".into());
        }
        let mut sources = [
            self.range(q_abs, input[0], 4, Element::F32)?,
            self.range(ckv, input[2], 1, Element::Bytes)?,
            self.range(selected, input[4], 4, Element::I32)?,
            self.range(positions, input[5], 4, Element::I32)?,
            (0, 0),
            (0, 0),
        ];
        let (qr, kp) = if p.rope > 0 {
            let q = q_rope.ok_or("F32 rotary query owner absent")?;
            let k = kpe.ok_or("rotary cache owner absent")?;
            if k.pointer() % 2 != 0 {
                return Err("stable MLA rotary cache alignment invalid".into());
            }
            sources[4] = self.range(q, input[1], 4, Element::F32)?;
            sources[5] = self.range(k, input[3], 1, Element::Bytes)?;
            (q.pointer(), k.pointer())
        } else {
            (0, 0)
        };
        for (i, &a) in outputs.iter().enumerate() {
            if sources.iter().any(|&b| overlap(a, b))
                || outputs.iter().skip(i + 1).any(|&b| overlap(a, b))
            {
                return Err("stable MLA gather output aliases live input/output".into());
            }
        }
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
            gathered.pointer(),
            query.pointer(),
            initial.pointer(),
            q_abs.pointer(),
            qr,
            ckv.pointer(),
            kp,
            selected.pointer(),
            positions.pointer(),
        );
        let (mut a9, mut a10, mut a11, mut a12, mut a13, mut a14, mut a15) = (
            integer(p.row_start)?,
            integer(p.rows)?,
            integer(p.heads)?,
            integer(p.ccd)?,
            integer(p.rope)?,
            integer(p.slots)?,
            integer(p.context)?,
        );
        let (mut a16, mut a17, mut a18, mut a19) = (
            p.q_abs_row_stride as u64,
            p.q_abs_head_stride as u64,
            p.q_rope_row_stride as u64,
            p.q_rope_head_stride as u64,
        );
        let blocks = mul(
            p.rows,
            sum(sum(p.slots.div_ceil(8), p.heads.div_ceil(8))?, 1)?,
        )?;
        self.launch(
            match p.codec {
                Codec::K4 => 0,
                Codec::K6 => 1,
            },
            blocks,
            &mut [
                &mut a0 as *mut _ as _,
                &mut a1 as *mut _ as _,
                &mut a2 as *mut _ as _,
                &mut a3 as *mut _ as _,
                &mut a4 as *mut _ as _,
                &mut a5 as *mut _ as _,
                &mut a6 as *mut _ as _,
                &mut a7 as *mut _ as _,
                &mut a8 as *mut _ as _,
                &mut a9 as *mut _ as _,
                &mut a10 as *mut _ as _,
                &mut a11 as *mut _ as _,
                &mut a12 as *mut _ as _,
                &mut a13 as *mut _ as _,
                &mut a14 as *mut _ as _,
                &mut a15 as *mut _ as _,
                &mut a16 as *mut _ as _,
                &mut a17 as *mut _ as _,
                &mut a18 as *mut _ as _,
                &mut a19 as *mut _ as _,
            ],
            stream,
        )
    }
    /// Preserve the actual F32 rotary query before its source plane becomes output.
    /// This enqueues a device-only pitched copy, never a CPU readback/conversion.
    pub(crate) unsafe fn pack_rope(
        &self,
        input: &Span<'_>,
        output: &Span<'_>,
        rows: usize,
        heads: usize,
        nope: usize,
        rope: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if rows == 0 || heads == 0 || rope == 0 {
            return Err("empty optional rotary copy must not be enqueued".into());
        }
        self.current(stream)?;
        let width = sum(nope, rope)?;
        let height = mul(rows, heads)?;
        let source_pitch = mul(width, 4)?;
        let target_pitch = mul(rope, 4)?;
        if source_pitch > self.max_pitch || target_pitch > self.max_pitch {
            return Err("actual CUDA device cannot admit rotary copy pitch".into());
        }
        let a = self.range(input, mul(height, width)?, 4, Element::F32)?;
        let b = self.range(output, mul(height, rope)?, 4, Element::F32)?;
        if overlap(a, b) {
            return Err("rotary preservation aliases live source".into());
        }
        let copy = sys::CUDA_MEMCPY2D {
            srcXInBytes: mul(nope, 4)?,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: input.pointer(),
            srcArray: std::ptr::null_mut(),
            srcPitch: source_pitch,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: output.pointer(),
            dstArray: std::ptr::null_mut(),
            dstPitch: target_pitch,
            WidthInBytes: target_pitch,
            Height: height,
        };
        sys::lib()
            .cuMemcpy2DAsync_v2(&copy, stream)
            .result()
            .map_err(|e| format!("F32 rotary preservation enqueue failed: {e:?}"))
    }
    pub(crate) unsafe fn widen(
        &self,
        input: &Span<'_>,
        output: &Span<'_>,
        elements: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if elements == 0 {
            return Err("empty WVC weight conversion".into());
        }
        self.current(stream)?;
        let a = self.range(input, elements, 2, Element::Bf16)?;
        let b = self.range(output, elements, 4, Element::F32)?;
        if overlap(a, b) {
            return Err("WVC widening aliases source".into());
        }
        let (mut x, mut y, mut n) = (output.pointer(), input.pointer(), elements as u64);
        self.launch(
            2,
            elements.div_ceil(256),
            &mut [
                &mut x as *mut _ as _,
                &mut y as *mut _ as _,
                &mut n as *mut _ as _,
            ],
            stream,
        )
    }
    pub(crate) unsafe fn scatter(
        &self,
        input: &Span<'_>,
        output: &Span<'_>,
        rows: usize,
        heads: usize,
        width: usize,
        head_stride_rows: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if rows == 0 || heads == 0 || width == 0 || head_stride_rows < rows {
            return Err("invalid WVC physical head stride".into());
        }
        self.current(stream)?;
        let n = mul(mul(rows, heads)?, width)?;
        let source = mul(sum(mul(heads - 1, head_stride_rows)?, rows)?, width)?;
        let a = self.range(input, source, 4, Element::F32)?;
        let b = self.range(output, n, 2, Element::Bf16)?;
        if overlap(a, b) {
            return Err("WVC scatter aliases source".into());
        }
        let (mut x, mut y, mut r, mut h, mut w, mut stride) = (
            output.pointer(),
            input.pointer(),
            integer(rows)?,
            integer(heads)?,
            integer(width)?,
            head_stride_rows as u64,
        );
        self.launch(
            3,
            n.div_ceil(256),
            &mut [
                &mut x as *mut _ as _,
                &mut y as *mut _ as _,
                &mut r as *mut _ as _,
                &mut h as *mut _ as _,
                &mut w as *mut _ as _,
                &mut stride as *mut _ as _,
            ],
            stream,
        )
    }
    pub(crate) unsafe fn softmax(
        &self,
        input: &Span<'_>,
        output: &Span<'_>,
        rows: usize,
        heads: usize,
        slots: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if rows == 0 || heads == 0 || slots == 0 {
            return Err("empty stable MLA softmax".into());
        }
        self.current(stream)?;
        let n = mul(mul(rows, heads)?, slots)?;
        let a = self.range(input, n, 4, Element::F32)?;
        let b = self.range(output, n, 4, Element::F32)?;
        if overlap(a, b) {
            return Err("stable MLA softmax aliases scores".into());
        }
        let (mut x, mut y, mut r, mut h, mut s) = (
            output.pointer(),
            input.pointer(),
            integer(rows)?,
            integer(heads)?,
            integer(slots)?,
        );
        self.launch(
            4,
            mul(rows, heads)?.div_ceil(8),
            &mut [
                &mut x as *mut _ as _,
                &mut y as *mut _ as _,
                &mut r as *mut _ as _,
                &mut h as *mut _ as _,
                &mut s as *mut _ as _,
            ],
            stream,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotary_stride_and_broadcast_views_have_real_extents() {
        assert_eq!(strided(3, 2, 4, 24, 8).unwrap(), 60);
        assert_eq!(strided(3, 2, 4, 0, 0).unwrap(), 4);
        assert!(strided(2, 2, 4, usize::MAX, usize::MAX).is_err());
    }
    #[test]
    fn gather_block_count_and_capacity_are_checked() {
        let p = Gather {
            row_start: 3,
            rows: 2,
            heads: 2,
            ccd: 16,
            rope: 16,
            slots: 5,
            context: 7,
            q_abs_row_stride: 32,
            q_abs_head_stride: 16,
            q_rope_row_stride: 48,
            q_rope_head_stride: 24,
            codec: Codec::K6,
        };
        p.validate().unwrap();
        assert_eq!(p.outputs().unwrap(), [320, 128, 20]);
        assert_eq!(p.inputs().unwrap(), [160, 232, 98, 98, 25, 5]);
        assert!(Gather { ccd: 17, ..p }.validate().is_err());
        assert!(Gather {
            row_start: usize::MAX,
            ..p
        }
        .validate()
        .is_err());
    }
}

#[cfg(feature = "shared-kda-test")]
pub(crate) fn load_serving(device: Arc<CudaDevice>) -> Result<Arc<Ops>, String> {
    Ops::load(
        device,
        include_str!(concat!(env!("OUT_DIR"), "/stable_mla_ops.ptx")),
    )
}

#[cfg(not(feature = "shared-kda-test"))]
pub(crate) fn load_serving(_device: Arc<CudaDevice>) -> Result<Arc<Ops>, String> {
    Err("common MLA diagnostic backend was not compiled; build with shared-kda-test before requesting it".into())
}
