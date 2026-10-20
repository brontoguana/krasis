//! Context-owned common Marlin arithmetic. Callers retain real weight/scratch owners.
//! No allocation, host copies, synchronization or Python on the enqueue path.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::sync::Arc;
/// Cover every logical row with balanced Y/Z planes using actual device limits.
pub(crate) fn row_grid(rows: usize, max_y: usize, max_z: usize) -> Result<(u32,u32),String> {
    if rows == 0 || max_y == 0 || max_z == 0 { return Err("invalid CUDA row grid".into()); }
    let z = rows.div_ceil(max_y);
    if z > max_z { return Err("CUDA row grid exceeds device limits".into()); }
    let y = rows.div_ceil(z);
    Ok((u32::try_from(y).map_err(|_|"CUDA grid Y overflow")?,u32::try_from(z).map_err(|_|"CUDA grid Z overflow")?))
}
#[cfg(test)]
mod row_grid_tests {
    #[test]
    fn common_row_grid_covers_boundary_and_rejects_overflow() {
        for max_y in [1usize,3,17,65535] { for max_z in [1usize,2,7] {
            let cap=max_y*max_z;
            for rows in [1,max_y,max_y.saturating_add(1),cap,cap+1,usize::MAX] {
                let actual=super::row_grid(rows,max_y,max_z);
                if rows<=cap {let(y,z)=actual.unwrap();assert!(y as usize<=max_y&&z as usize<=max_z);assert!((y as usize)*(z as usize)>=rows);assert!((y as usize)*(z as usize)-rows<z as usize);}else{assert!(actual.is_err());}
            }
        }}
        assert!(super::row_grid(0,1,1).is_err());assert!(super::row_grid(1,0,1).is_err());assert!(super::row_grid(1,1,0).is_err());
    }
}
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub bits: usize,
}
impl Geometry {
    pub(crate) fn extents(self) -> Result<(usize, usize), String> {
        if self.rows == 0
            || self.cols == 0
            || self.rows % 64 != 0
            || self.cols % 16 != 0
            || self.group == 0
            || self.group > self.cols
            || ![4, 8].contains(&self.bits)
            || self.rows > i32::MAX as usize
            || self.cols > i32::MAX as usize
        {
            return Err("unsupported common Marlin geometry".into());
        }
        let packed = self
            .rows
            .checked_mul(self.cols)
            .and_then(|x| x.checked_mul(self.bits))
            .ok_or("packed extent overflow")?
            / 8;
        let scales = self
            .rows
            .checked_mul(self.cols.div_ceil(self.group))
            .and_then(|x| x.checked_mul(2))
            .ok_or("scale extent overflow")?;
        Ok((packed, scales))
    }
}
fn region(ptr: u64, count: usize, align: usize) -> Result<(), String> {
    if ptr == 0 || ptr % align as u64 != 0 || count == 0 {
        return Err("invalid common Marlin region".into());
    }
    ptr.checked_add(u64::try_from(count).map_err(|_| "region extent exceeds pointer ABI")?)
        .ok_or("common Marlin pointer overflow")?;
    Ok(())
}
pub(crate) struct Program {
    _device: Arc<CudaDevice>,
    single: CudaFunction,
    grouped: CudaFunction,
    slots: CudaFunction,
    activation: CudaFunction,
    limits: [usize; 3],
    pub identity: String,
}
impl Program {
    pub(crate) fn stream(&self)->sys::CUstream{*self._device.cu_stream()}
    pub(crate) fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Self, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        use sys::CUdevice_attribute as A;
        let attr = |a| -> Result<usize, String> {
            usize::try_from(device.attribute(a).map_err(|e| e.to_string())?)
                .map_err(|_| "negative CUDA device attribute".into())
        };
        if attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8 {
            return Err("common Marlin requires Ampere or newer".into());
        }
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        device
            .load_ptx(
                ptx.into(),
                "common_marlin_k16",
                &[
                    "marlin_projection_tensorcomp",
                    "marlin_projection_grouped",
                    "marlin_projection_slots",
                    "marlin_expert_activation",
                ],
            )
            .map_err(|e| e.to_string())?;
        let single = device
            .get_func("common_marlin_k16", "marlin_projection_tensorcomp")
            .ok_or("common Marlin singleton symbol absent")?;
        let grouped = device
            .get_func("common_marlin_k16", "marlin_projection_grouped")
            .ok_or("common Marlin grouped symbol absent")?;
        let slots = device
            .get_func("common_marlin_k16", "marlin_projection_slots")
            .ok_or("common Marlin slot symbol absent")?;
        let activation = device
            .get_func("common_marlin_k16", "marlin_expert_activation")
            .ok_or("common expert activation symbol absent")?;
        for function in [&single, &grouped, &slots, &activation] {
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
                    256,
                    0,
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            if threads < 256 || active <= 0 {
                return Err("loaded common Marlin cannot admit fixed arithmetic tile".into());
            }
        }
        Ok(Self {
            _device: device,
            single,
            grouped,
            slots,
            activation,
            limits,
            identity: format!(
                "marlin_exact_bf16_split_k16_compensated_v1:{:x}",
                Sha256::digest(ptx.as_bytes())
            ),
        })
    }
    /// SAFETY: actual same-context allocations and stream are owned and held by the
    /// engine/lease through completion. All declared extents must be allocated,
    /// inputs read-only and disjoint from output, which has unique stream access.
    pub(crate) unsafe fn enqueue(
        &self,
        g: Geometry,
        packed: u64,
        scales: u64,
        input: u64,
        output: u64,
        observation: u64,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let (p, s) = g.extents()?;
        region(packed, p, 4)?;
        region(scales, s, 2)?;
        region(
            input,
            tokens
                .checked_mul(g.cols)
                .and_then(|x| x.checked_mul(2))
                .ok_or("input extent overflow")?,
            4,
        )?;
        region(
            output,
            tokens
                .checked_mul(g.rows)
                .and_then(|x| x.checked_mul(2))
                .ok_or("output extent overflow")?,
            2,
        )?;
        if observation != 0 {
            region(
                observation,
                tokens
                    .checked_mul(g.rows)
                    .and_then(|x| x.checked_mul(4))
                    .ok_or("observation extent overflow")?,
                4,
            )?;
        }
        let tiles = tokens.div_ceil(32);
        let x = tiles.min(self.limits[0]);
        let z = tiles.div_ceil(x);
        if z > self.limits[2] || g.rows / 64 > self.limits[1] {
            return Err("common Marlin grid exceeds device limits".into());
        }
        let mut pw = packed;
        let mut ps = scales;
        let mut null = 0u64;
        let mut input = input;
        let mut output = output;
        let mut observation = observation;
        let mut n = g.rows as i32;
        let mut k = g.cols as i32;
        let mut group = g.group as i32;
        let mut perm = if g.group < g.cols { 64i32 } else { 32 };
        let mut bits = g.bits as i32;
        let mut m = tokens as u64;
        let mut args = [
            (&mut pw as *mut u64).cast(),
            (&mut ps as *mut u64).cast(),
            (&mut null as *mut u64).cast(),
            (&mut null as *mut u64).cast(),
            (&mut input as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut observation as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
            (&mut group as *mut i32).cast(),
            (&mut perm as *mut i32).cast(),
            (&mut bits as *mut i32).cast(),
            (&mut m as *mut u64).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.single.cu_function(),
                x as u32,
                (g.rows / 64) as u32,
                z as u32,
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
    /// SAFETY: enqueue's allocation/stream rules also apply to actual sorter and
    /// pointer-table owners. Every live sorted ID is an input/output slot; every
    /// nonnegative block expert ID indexes the declared expert population. Each
    /// live slot occurs once across precision-filtered launches. Table entries
    /// name actual held allocations of this geometry; no host dereference occurs.
    pub(crate) unsafe fn enqueue_grouped(
        &self,
        g: Geometry,
        packed: u64,
        scales: u64,
        packed_ptrs: u64,
        scale_ptrs: u64,
        input: u64,
        output: u64,
        sorted: u64,
        experts: u64,
        post_count: u64,
        block_size: usize,
        expert_count: usize,
        slots: usize,
        sorted_capacity: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let (p, s) = g.extents()?;
        if block_size == 0
            || block_size > i32::MAX as usize
            || expert_count == 0
            || expert_count > i32::MAX as usize
            || slots == 0
            || slots > i32::MAX as usize
            || sorted_capacity == 0
            || sorted_capacity > i32::MAX as usize
            || sorted_capacity % block_size != 0
        {
            return Err("invalid common Marlin sorter geometry".into());
        }
        if packed_ptrs == 0 && scale_ptrs == 0 {
            region(
                packed,
                p.checked_mul(expert_count)
                    .ok_or("expert packed extent overflow")?,
                4,
            )?;
            region(
                scales,
                s.checked_mul(expert_count)
                    .ok_or("expert scales extent overflow")?,
                2,
            )?;
        } else {
            region(
                packed_ptrs,
                expert_count
                    .checked_mul(8)
                    .ok_or("packed table extent overflow")?,
                8,
            )?;
            region(
                scale_ptrs,
                expert_count
                    .checked_mul(8)
                    .ok_or("scale table extent overflow")?,
                8,
            )?;
        }
        region(
            input,
            slots
                .checked_mul(g.cols)
                .and_then(|x| x.checked_mul(2))
                .ok_or("group input extent overflow")?,
            4,
        )?;
        region(
            output,
            slots
                .checked_mul(g.rows)
                .and_then(|x| x.checked_mul(2))
                .ok_or("group output extent overflow")?,
            2,
        )?;
        region(
            sorted,
            sorted_capacity
                .checked_mul(4)
                .ok_or("sort extent overflow")?,
            4,
        )?;
        let blocks = sorted_capacity / block_size;
        region(
            experts,
            blocks.checked_mul(4).ok_or("expert ID extent overflow")?,
            4,
        )?;
        region(post_count, 4, 4)?;
        let x = blocks
            .checked_mul(block_size.div_ceil(32))
            .ok_or("group grid overflow")?;
        if x > self.limits[0] || g.rows / 64 > self.limits[1] {
            return Err("common Marlin grouped grid exceeds device limits".into());
        }
        let mut pw = packed;
        let mut ps = scales;
        let mut pt = packed_ptrs;
        let mut st = scale_ptrs;
        let mut input = input;
        let mut output = output;
        let mut sorted = sorted;
        let mut experts = experts;
        let mut post = post_count;
        let mut block = block_size as i32;
        let mut ec = expert_count as i32;
        let mut n = g.rows as i32;
        let mut k = g.cols as i32;
        let mut group = g.group as i32;
        let mut bits = g.bits as i32;
        let mut m = slots as u64;
        let mut capacity = sorted_capacity as u64;
        let mut args = [
            (&mut pw as *mut u64).cast(),
            (&mut ps as *mut u64).cast(),
            (&mut pt as *mut u64).cast(),
            (&mut st as *mut u64).cast(),
            (&mut input as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut sorted as *mut u64).cast(),
            (&mut experts as *mut u64).cast(),
            (&mut post as *mut u64).cast(),
            (&mut block as *mut i32).cast(),
            (&mut ec as *mut i32).cast(),
            (&mut n as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
            (&mut group as *mut i32).cast(),
            (&mut bits as *mut i32).cast(),
            (&mut m as *mut u64).cast(),
            (&mut capacity as *mut u64).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.grouped.cu_function(),
                x as u32,
                (g.rows / 64) as u32,
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
            .map_err(|e| e.to_string())
    }
    /// SAFETY: actual same-context router pointer tables, weights, input and
    /// output allocations are held through completion. Each nonzero-weight table
    /// entry names a live allocation of this geometry, with disjoint outputs.
    pub(crate) unsafe fn enqueue_slots(
        &self,
        g: Geometry,
        packed_ptrs: u64,
        scale_ptrs: u64,
        input: u64,
        output: u64,
        weights: u64,
        input_stride: usize,
        output_stride: usize,
        slots: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        g.extents()?;
        if slots == 0
            || slots > self.limits[0]
            || g.rows / 64 > self.limits[1]
            || output_stride < g.rows
            || (input_stride != 0 && input_stride < g.cols) || input_stride % 2 != 0
        {
            return Err("invalid common expert pointer-slot geometry".into());
        }
        region(
            packed_ptrs,
            slots
                .checked_mul(8)
                .ok_or("packed pointer slots overflow")?,
            8,
        )?;
        region(
            scale_ptrs,
            slots.checked_mul(8).ok_or("scale pointer slots overflow")?,
            8,
        )?;
        region(
            weights,
            slots.checked_mul(4).ok_or("weight slots overflow")?,
            4,
        )?;
        region(
            input,
            (slots - 1)
                .checked_mul(input_stride)
                .and_then(|v| v.checked_add(g.cols))
                .and_then(|v| v.checked_mul(2))
                .ok_or("slot input extent overflow")?,
            4,
        )?;
        region(
            output,
            (slots - 1)
                .checked_mul(output_stride)
                .and_then(|v| v.checked_add(g.rows))
                .and_then(|v| v.checked_mul(2))
                .ok_or("slot output extent overflow")?,
            2,
        )?;
        let mut pw = packed_ptrs;
        let mut ps = scale_ptrs;
        let mut x = input;
        let mut y = output;
        let mut w = weights;
        let mut n = g.rows as i32;
        let mut k = g.cols as i32;
        let mut group = g.group as i32;
        let mut bits = g.bits as i32;
        let mut xs = input_stride as u64;
        let mut ys = output_stride as u64;
        let mut count = slots as u64;
        let mut args = [
            (&mut pw as *mut u64).cast(),
            (&mut ps as *mut u64).cast(),
            (&mut x as *mut u64).cast(),
            (&mut y as *mut u64).cast(),
            (&mut w as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
            (&mut group as *mut i32).cast(),
            (&mut bits as *mut i32).cast(),
            (&mut xs as *mut u64).cast(),
            (&mut ys as *mut u64).cast(),
            (&mut count as *mut u64).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.slots.cu_function(),
                slots as u32,
                (g.rows / 64) as u32,
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
            .map_err(|e| e.to_string())
    }
    /// SAFETY: actual BF16 scratch/input and optional active-weight owners remain
    /// held through completion. In-place activation uses the identical row stride;
    /// otherwise the full input/output extents are disjoint.
    pub(crate) unsafe fn enqueue_activation(
        &self,
        input: u64,
        output: u64,
        weights: u64,
        intermediate: usize,
        kind: u8,
        limit: f32,
        raw_limit: bool,
        input_stride: usize,
        output_stride: usize,
        tokens: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if intermediate == 0
            || intermediate > i32::MAX as usize
            || tokens == 0
            || kind > 2
            || !limit.is_finite()
            || (raw_limit && (limit <= 0.0 || kind != 0))
        {
            return Err("invalid common expert activation".into());
        }
        let width = intermediate
            .checked_mul(if kind == 1 { 1 } else { 2 })
            .ok_or("activation width overflow")?;
        if input_stride < width
            || output_stride < intermediate
            || (input == output && input_stride != output_stride)
        {
            return Err("activation row stride/alias contract".into());
        }
        region(
            input,
            (tokens - 1)
                .checked_mul(input_stride)
                .and_then(|v| v.checked_add(width))
                .and_then(|v| v.checked_mul(2))
                .ok_or("activation input overflow")?,
            2,
        )?;
        region(
            output,
            (tokens - 1)
                .checked_mul(output_stride)
                .and_then(|v| v.checked_add(intermediate))
                .and_then(|v| v.checked_mul(2))
                .ok_or("activation output overflow")?,
            2,
        )?;
        if weights != 0 {
            region(
                weights,
                tokens.checked_mul(4).ok_or("activation mask overflow")?,
                4,
            )?;
        }
        let gx = intermediate.div_ceil(256);
        let (gy,gz)=row_grid(tokens,self.limits[1],self.limits[2])?;
        if gx > self.limits[0] || gz as usize > self.limits[2] {
            return Err("activation grid exceeds device limits".into());
        }
        let mut x = input;
        let mut y = output;
        let mut w = weights;
        let mut n = intermediate as i32;
        let mut kind = kind as i32;
        let mut limit = limit;
        let mut raw = raw_limit as i32;
        let mut xs = input_stride as u64;
        let mut ys = output_stride as u64;
        let mut count = tokens as u64;
        let mut args = [
            (&mut x as *mut u64).cast(),
            (&mut y as *mut u64).cast(),
            (&mut w as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
            (&mut kind as *mut i32).cast(),
            (&mut limit as *mut f32).cast(),
            (&mut raw as *mut i32).cast(),
            (&mut xs as *mut u64).cast(),
            (&mut ys as *mut u64).cast(),
            (&mut count as *mut u64).cast(),
        ];
        sys::lib()
            .cuLaunchKernel(
                self.activation.cu_function(),
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
