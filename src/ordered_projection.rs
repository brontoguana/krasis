//! Shared direct-packed projection; fixed ordered F32 coefficient arithmetic.
//! Tensor-layout constants are the qualified CTA/WMMA contract, not budgets.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::sync::Arc;

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
impl Layout {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.rows == 0
            || self.cols == 0
            || self.group == 0
            || self.cols % 4 != 0
            || self.group % 4 != 0
            || self.cols % self.group != 0
            || self.packed_stride
                != (self.cols / 4)
                    .checked_mul(3)
                    .ok_or("ordered packed stride overflow")?
            || self.scales_stride % 4 != 0
            || self.scales_stride
                < (self.cols / self.group)
                    .checked_mul(4)
                    .ok_or("ordered scale stride overflow")?
            || self.zeros_stride != self.scales_stride
            || [
                self.rows,
                self.cols,
                self.group,
                self.packed_stride,
                self.scales_stride,
            ]
            .iter()
            .any(|&v| v > i32::MAX as usize)
            || [self.packed, self.scales, self.zeros].contains(&0)
            || self.scales % 4 != 0
            || self.zeros % 4 != 0
        {
            return Err(
                "ordered projection requires complete row-major HQQ6/F32 scale-zero geometry"
                    .into(),
            );
        }
        for (ptr, stride) in [
            (self.packed, self.packed_stride),
            (self.scales, self.scales_stride),
            (self.zeros, self.zeros_stride),
        ] {
            ptr.checked_add(
                self.rows
                    .checked_mul(stride)
                    .ok_or("ordered weight extent overflow")? as u64,
            )
            .ok_or("ordered weight pointer overflow")?;
        }
        Ok(())
    }
}
pub(crate) struct OrderedProjection {
    _device: Arc<CudaDevice>,
    function: CudaFunction,
    limits: [usize; 3],
    hc_function: Option<CudaFunction>,
    hc_width_limit: usize,
    dense_function: Option<CudaFunction>,
    router_function: Option<CudaFunction>,
    index_score: Option<crate::dsa_score::Program>,
    pub(crate) marlin: Option<crate::common_marlin::Program>,
    pub(crate) moe_combine: Option<crate::common_moe_combine::Program>,
    pub(crate) norm: Option<crate::common_rmsnorm::Program>,
    pub(crate) identity: String,
}
impl OrderedProjection {
    pub(crate) fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Arc<Self>, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        use sys::CUdevice_attribute as A;
        let read = |attr| {
            device
                .attribute(attr)
                .map_err(|e| e.to_string())
                .and_then(|n| {
                    usize::try_from(n).map_err(|_| "negative ordered device attribute".into())
                })
        };
        if read(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8
            || read(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 8 * 32
            || read(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? < (3 * 64 + 32) * 40 * 4
        {
            return Err("actual CUDA device cannot execute qualified ordered projection".into());
        }
        let limits = [
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        device
            .load_ptx(
                ptx.into(),
                "shared_ordered_projection",
                &["hqq6_batch_ordered_tf32x3"],
            )
            .map_err(|e| e.to_string())?;
        let function = device
            .get_func("shared_ordered_projection", "hqq6_batch_ordered_tf32x3")
            .ok_or("ordered projection symbol absent")?;
        let identity = format!(
            "tf32_three_component_fixed_k8_ordered_twosum_v1:bf16_only:{:x}",
            Sha256::digest(ptx.as_bytes())
        );
        Ok(Arc::new(Self {
            _device: device,
            function,
            limits,
            hc_function: None,
            hc_width_limit: 0,
            dense_function: None,
            router_function: None,
            index_score: None,
            marlin: None,
            moe_combine: None,
            norm: None,
            identity,
        }))
    }
    pub(crate) fn score_kernel(&self) -> Result<crate::dsa_score::Kernel,String> {
        self.index_score.as_ref().map(|p|p.kernel()).ok_or("selected F32 index scorer not admitted".into())
    }
    fn attach_hc(&mut self, ptx: &str) -> Result<(), String> {
        use sys::CUdevice_attribute as A;
        let threads = usize::try_from(
            self._device
                .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|_| "invalid HC thread limit")?;
        let shared = usize::try_from(
            self._device
                .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|_| "invalid HC shared limit")?;
        let bound = threads.min(shared / std::mem::size_of::<f64>());
        if bound < 32 {
            return Err("device cannot execute HC block reduction".into());
        }
        self.hc_width_limit = 1usize << (usize::BITS - 1 - bound.leading_zeros());
        self._device
            .load_ptx(
                ptx.into(),
                "shared_hc_projection",
                &["ordered_hc_f64_block"],
            )
            .map_err(|e| e.to_string())?;
        self.hc_function = Some(
            self._device
                .get_func("shared_hc_projection", "ordered_hc_f64_block")
                .ok_or("HC projection symbol absent")?,
        );
        self.identity.push_str(&format!(
            ":hc_normalized_f32_compensated_v1:{:x}",
            Sha256::digest(ptx.as_bytes())
        ));
        Ok(())
    }
    fn attach_experts(
        &mut self,
        marlin_ptx: &str,
        router_ptx: &str,
        combine_ptx: &str,
        norm_ptx: &str,
    ) -> Result<(), String> {
        let marlin = crate::common_marlin::Program::load(self._device.clone(), marlin_ptx)?;
        self._device
            .load_ptx(
                router_ptx.into(),
                "common_router_f32",
                &["bf16_projection_tensorcomp_f32"],
            )
            .map_err(|e| e.to_string())?;
        let function = self
            ._device
            .get_func("common_router_f32", "bf16_projection_tensorcomp_f32")
            .ok_or("common router symbol absent")?;
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
            return Err("loaded router cannot admit qualified tile".into());
        }
        self.identity.push_str(&format!(
            ":{}:router_bf16_k16_compensated_f32_v1:{:x}",
            marlin.identity,
            Sha256::digest(router_ptx.as_bytes())
        ));
        let combine = crate::common_moe_combine::Program::load(self._device.clone(), combine_ptx)?;
        self.identity.push_str(&format!(":{}", combine.identity));
        let norm=crate::common_rmsnorm::Program::load(self._device.clone(),norm_ptx)?;
        self.identity.push_str(&format!(":{}",norm.identity));
        self.norm=Some(norm);
        self.moe_combine = Some(combine);
        self.marlin = Some(marlin);
        self.router_function = Some(function);
        Ok(())
    }
    /// Actual BF16 model-gate and scratch owners remain held by the engine.
    /// The caller provides disjoint same-context extents through stream completion.
    pub(crate) unsafe fn enqueue_router(
        &self,
        weight: u64,
        input: u64,
        output: u64,
        rows: usize,
        cols: usize,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let function = self
            .router_function
            .as_ref()
            .ok_or("common F32 router not admitted")?;
        if rows == 0
            || cols == 0
            || tokens == 0
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return Err("invalid common router geometry".into());
        }
        for (ptr, count, bytes) in [
            (weight, rows.checked_mul(cols), 2usize),
            (input, tokens.checked_mul(cols), 2),
            (output, tokens.checked_mul(rows), 4),
        ] {
            if ptr == 0 || ptr % bytes as u64 != 0 {
                return Err("invalid common router alignment".into());
            }
            ptr.checked_add(
                count
                    .and_then(|v| v.checked_mul(bytes))
                    .ok_or("common router extent overflow")? as u64,
            )
            .ok_or("common router pointer overflow")?;
        }
        let gx = tokens.div_ceil(64);
        let gy = rows.div_ceil(64);
        if gx > self.limits[0] || gy > self.limits[1] {
            return Err("common router grid exceeds device limits".into());
        }
        let mut input = input;
        let mut weight = weight;
        let mut output = output;
        let mut n = rows as i32;
        let mut k = cols as i32;
        let mut m = tokens as u64;
        let mut args = [
            (&mut input as *mut u64).cast(),
            (&mut weight as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
            (&mut m as *mut u64).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                function.cu_function(),
                gx as u32,
                gy as u32,
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
            .map_err(|e| e.to_string())
    }
    /// SAFETY: actual shared weights, gate-up scratch, BF16 shared output,
    /// F32 routed owner and BF16 final output remain held through this stream's
    /// completion. No temporary allocation or workspace aliases another stream.
    pub(crate) unsafe fn enqueue_shared_finish(
        &self,
        w: crate::common_marlin::Geometry,
        packed: u64,
        scales: u64,
        gate_up: u64,
        shared_output: u64,
        routed: u64,
        output: u64,
        gate: u64,
        weight: f32,
        kind: u8,
        limit: f32,
        raw_limit: bool,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let program = self
            .marlin
            .as_ref()
            .ok_or("common shared expert program absent")?;
        let stride = w
            .cols
            .checked_mul(if kind == 1 { 1 } else { 2 })
            .ok_or("shared activation extent overflow")?;
        program.enqueue_activation(
            gate_up, gate_up, 0, w.cols, kind, limit, raw_limit, stride, stride, 1, stream,
        )?;
        program.enqueue(w, packed, scales, gate_up, shared_output, 0, 1, stream)?;
        self.moe_combine
            .as_ref()
            .ok_or("common shared combination program absent")?
            .enqueue_combine(
                routed,
                shared_output,
                gate,
                output,
                0,
                w.rows,
                1,
                weight,
                stream,
            )
    }
    /// Actual scratch/weight owners stay held by the calling engine. No new
    /// allocation or synchronization; arithmetic/tree independent of token count.
    pub(crate) unsafe fn enqueue_hc(
        &self,
        state: u64,
        inverse: u64,
        weight: u64,
        output: u64,
        rows: usize,
        cols: usize,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let function = self.hc_function.as_ref().ok_or("HC kernel not admitted")?;
        if rows == 0
            || cols == 0
            || tokens == 0
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return Err("invalid HC geometry".into());
        }
        for (ptr, count, bytes) in [
            (state, tokens.checked_mul(cols), 2usize),
            (inverse, Some(tokens), 4),
            (weight, rows.checked_mul(cols), 4),
            (output, tokens.checked_mul(rows), 4),
        ] {
            if ptr == 0 || ptr % bytes as u64 != 0 {
                return Err("invalid HC pointer alignment".into());
            }
            ptr.checked_add(
                count
                    .and_then(|n| n.checked_mul(bytes))
                    .ok_or("HC extent overflow")? as u64,
            )
            .ok_or("HC pointer overflow")?;
        }
        let width = cols.min(self.hc_width_limit).next_power_of_two().max(32);
        let blocks = tokens.div_ceil(4);
        let x = blocks.min(self.limits[0]);
        if x == 0 || rows > self.limits[1] {
            return Err("HC x/y grid exceeds device".into());
        }
        let z = blocks.div_ceil(x);
        if z > self.limits[2] {
            return Err("HC z grid exceeds device".into());
        }
        let mut state = state;
        let mut inverse = inverse;
        let mut weight = weight;
        let mut output = output;
        let mut rows = rows as i32;
        let mut cols = cols as i32;
        let mut tokens = tokens as u64;
        let mut args = [
            &mut state as *mut _ as *mut std::ffi::c_void,
            &mut inverse as *mut _ as _,
            &mut weight as *mut _ as _,
            &mut output as *mut _ as _,
            &mut rows as *mut _ as _,
            &mut cols as *mut _ as _,
            &mut tokens as *mut _ as _,
        ];
        let rc = sys::lib().cuLaunchKernel(
            function.cu_function(),
            x as u32,
            rows as u32,
            z as u32,
            width as u32,
            1,
            1,
            (width * std::mem::size_of::<f64>()) as u32,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("HC projection launch: {rc:?}"));
        }
        Ok(())
    }
    fn attach_dense(&mut self, ptx: &str) -> Result<(), String> {
        self._device.bind_to_thread().map_err(|e| e.to_string())?;
        self._device
            .load_ptx(
                ptx.into(),
                "shared_dense_bf16",
                &["bf16_projection_tensorcomp"],
            )
            .map_err(|e| e.to_string())?;
        let function = self
            ._device
            .get_func("shared_dense_bf16", "bf16_projection_tensorcomp")
            .ok_or("shared dense BF16 symbol absent")?;
        let mut max_threads = 0i32;
        unsafe {
            sys::lib().cuFuncGetAttribute(
                &mut max_threads,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                function.cu_function(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        let mut active = 0i32;
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
        if max_threads < 512 || active <= 0 {
            return Err("loaded dense BF16 function cannot admit qualified 512-thread tile".into());
        }
        self.dense_function = Some(function);
        self.identity.push_str(&format!(
            ":dense_bf16_tensor_compensated_v1:{:x}",
            Sha256::digest(ptx.as_bytes())
        ));
        Ok(())
    }
    /// Model weight/scratch owners and the bank outlive work on their engine stream.
    /// No allocation or synchronization in either prefill or continuation.
    pub(crate) unsafe fn enqueue_dense_bf16(
        &self,
        weight: u64,
        input: u64,
        output: u64,
        rows: usize,
        cols: usize,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let function = self
            .dense_function
            .as_ref()
            .ok_or("dense BF16 bank not admitted")?;
        if rows == 0
            || cols == 0
            || tokens == 0
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return Err("invalid dense BF16 geometry".into());
        }
        let mut ends = [0u64; 3];
        for (i, (ptr, extent)) in [
            (weight, rows.checked_mul(cols)),
            (input, tokens.checked_mul(cols)),
            (output, tokens.checked_mul(rows)),
        ]
        .into_iter()
        .enumerate()
        {
            if ptr == 0 || ptr % 2 != 0 {
                return Err("invalid dense BF16 alignment".into());
            }
            ends[i] = ptr
                .checked_add(
                    extent
                        .and_then(|n| n.checked_mul(2))
                        .ok_or("dense BF16 extent overflow")? as u64,
                )
                .ok_or("dense BF16 pointer overflow")?;
        }
        if (output < ends[0] && weight < ends[2]) || (output < ends[1] && input < ends[2]) {
            return Err("dense BF16 output overlaps read-only operand".into());
        }
        let x = tokens.div_ceil(64);
        let nt = rows.div_ceil(64);
        let y = nt.min(self.limits[1]);
        if x > self.limits[0] || y == 0 {
            return Err("dense BF16 grid x/y exceeds device limits".into());
        }
        let z = nt.div_ceil(y);
        if z > self.limits[2] {
            return Err("dense BF16 grid z exceeds device limits".into());
        }
        let mut weight = weight;
        let mut input = input;
        let mut output = output;
        let mut rows = rows as i32;
        let mut cols = cols as i32;
        let mut tokens = tokens as u64;
        let mut args = [
            &mut input as *mut _ as *mut std::ffi::c_void,
            &mut weight as *mut _ as _,
            &mut output as *mut _ as _,
            &mut rows as *mut _ as _,
            &mut cols as *mut _ as _,
            &mut tokens as *mut _ as _,
        ];
        let rc = sys::lib().cuLaunchKernel(
            function.cu_function(),
            x as u32,
            y as u32,
            z as u32,
            512,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("dense BF16 projection launch: {rc:?}"));
        }
        Ok(())
    }
    /// Caller retains registered weight/input/output owners through completion,
    /// and supplies their admitted model stream. No allocation or synchronization.
    /// Use the already-compiled optional F32 result without rounding it through BF16.
    /// Both output allocations and all source owners must survive stream completion.
    pub(crate) unsafe fn enqueue_f32(
        &self,
        w: &Layout,
        input: u64,
        bf16_output: u64,
        f32_output: u64,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if f32_output == 0 {
            return Err("F32 projection requires an actual F32 output allocation".into());
        }
        self.enqueue_outputs(w, input, bf16_output, f32_output, tokens, stream)
    }

    pub(crate) unsafe fn enqueue(
        &self,
        w: &Layout,
        input: u64,
        output: u64,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        self.enqueue_outputs(w, input, output, 0, tokens, stream)
    }

    unsafe fn enqueue_outputs(
        &self,
        w: &Layout,
        input: u64,
        output: u64,
        f32_output: u64,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        validate_projection_outputs(w, input, output, f32_output, tokens)?;
        if tokens == 0 || input == 0 || output == 0 || input % 2 != 0 || output % 2 != 0 {
            return Err("ordered projection requires nonempty aligned input/output".into());
        }
        for (ptr, width) in [(input, w.cols), (output, w.rows)] {
            ptr.checked_add(
                tokens
                    .checked_mul(width)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or("ordered input/output extent overflow")? as u64,
            )
            .ok_or("ordered input/output pointer overflow")?;
        }
        let blocks = tokens.div_ceil(32);
        let x = blocks.min(self.limits[0]);
        let y = w.rows.div_ceil(64);
        if x == 0 || y > self.limits[1] {
            return Err("ordered projection exceeds actual grid x/y limits".into());
        }
        let z = blocks.div_ceil(x);
        if z > self.limits[2] {
            return Err("ordered projection exceeds actual grid z limits".into());
        }
        let mut packed = w.packed;
        let mut scales = w.scales;
        let mut zeros = w.zeros;
        let mut input = input;
        let mut output = output;
        let mut diagnostic = f32_output;
        let mut rows = w.rows as i32;
        let mut cols = w.cols as i32;
        let mut group = w.group as i32;
        let mut packed_stride = w.packed_stride as i32;
        let mut scales_stride = w.scales_stride as i32;
        let mut count = tokens as u64;
        let mut args = [
            &mut packed as *mut _ as *mut std::ffi::c_void,
            &mut scales as *mut _ as _,
            &mut zeros as *mut _ as _,
            &mut input as *mut _ as _,
            &mut output as *mut _ as _,
            &mut diagnostic as *mut _ as _,
            &mut rows as *mut _ as _,
            &mut cols as *mut _ as _,
            &mut group as *mut _ as _,
            &mut packed_stride as *mut _ as _,
            &mut scales_stride as *mut _ as _,
            &mut count as *mut _ as _,
        ];
        let rc = sys::lib().cuLaunchKernel(
            self.function.cu_function(),
            x as u32,
            y as u32,
            z as u32,
            8 * 32,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("ordered projection launch: {rc:?}"));
        }
        Ok(())
    }
}
#[cfg(feature = "shared-kda-test")]
pub(crate) fn load_serving(device: Arc<CudaDevice>) -> Result<Arc<OrderedProjection>, String> {
    let mut bank = OrderedProjection::load(
        device,
        include_str!(concat!(env!("OUT_DIR"), "/ordered_projection.ptx")),
    )?;
    Arc::get_mut(&mut bank)
        .ok_or("HC bank unexpectedly shared before admission")?
        .attach_hc(include_str!(concat!(
            env!("OUT_DIR"),
            "/ordered_hc_compensated.ptx"
        )))?;
    Arc::get_mut(&mut bank)
        .ok_or("dense bank unexpectedly shared before admission")?
        .attach_dense(include_str!(concat!(
            env!("OUT_DIR"),
            "/ordered_dense_bf16.ptx"
        )))?;
    Arc::get_mut(&mut bank)
        .ok_or("expert bank unexpectedly shared before admission")?
        .attach_experts(
            include_str!(concat!(env!("OUT_DIR"), "/ordered_marlin_serving.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/ordered_router_f32.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/ordered_moe_combine.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/common_rmsnorm.ptx")),
        )?;
    let score=crate::dsa_score::Program::load(bank._device.clone(),include_str!(concat!(env!("OUT_DIR"),"/dsa_score_tensorcomp.ptx")))?;
    let owned=Arc::get_mut(&mut bank).ok_or("score bank unexpectedly shared before admission")?;
    owned.identity.push_str(&format!(":dsa_score_k16_compensated_f32_v1:{:x}",Sha256::digest(include_str!(concat!(env!("OUT_DIR"),"/dsa_score_tensorcomp.ptx")).as_bytes())));
    owned.index_score=Some(score);
    Ok(bank)
}
#[cfg(not(feature = "shared-kda-test"))]
pub(crate) fn load_serving(_device: Arc<CudaDevice>) -> Result<Arc<OrderedProjection>, String> {
    Err("ordered projection kernels were not built".into())
}

// Exact byte extents derive from actual loaded geometry; no allocation or driver query.
pub(crate) fn validate_projection_outputs(
    w: &Layout,
    input: u64,
    bf16_output: u64,
    f32_output: u64,
    tokens: usize,
) -> Result<(), String> {
    w.validate()?;
    if tokens == 0 {
        return Err("empty ordered projection output".into());
    }
    let span = |ptr: u64, elements: Option<usize>, bytes: usize| -> Result<(u64, u64), String> {
        if ptr == 0 || ptr % bytes as u64 != 0 {
            return Err("ordered projection pointer/alignment invalid".into());
        }
        let size = elements
            .and_then(|v| v.checked_mul(bytes))
            .ok_or("ordered projection byte extent overflow")?;
        Ok((
            ptr,
            ptr.checked_add(
                u64::try_from(size).map_err(|_| "ordered byte size exceeds CUDA address space")?,
            )
            .ok_or("ordered projection address overflow")?,
        ))
    };
    let sources = [
        span(input, tokens.checked_mul(w.cols), 2)?,
        span(w.packed, w.rows.checked_mul(w.packed_stride), 1)?,
        span(w.scales, w.rows.checked_mul(w.scales_stride / 4), 4)?,
        span(w.zeros, w.rows.checked_mul(w.zeros_stride / 4), 4)?,
    ];
    let bf = span(bf16_output, tokens.checked_mul(w.rows), 2)?;
    let overlap = |x: (u64, u64), y: (u64, u64)| x.0 < y.1 && y.0 < x.1;
    if sources.iter().any(|&src| overlap(src, bf)) {
        return Err("BF16 output overlaps a read-only source".into());
    }
    if f32_output != 0 {
        let fp = span(f32_output, tokens.checked_mul(w.rows), 4)?;
        if overlap(fp, bf) || sources.iter().any(|&src| overlap(src, fp)) {
            return Err("F32 output overlaps another live operand".into());
        }
    }
    Ok(())
}
