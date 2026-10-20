//! Selected batched F32 MLA primitives; caller retains real device/stream owners.
//! All byte extents and launch limits derive from loaded geometry and the device.
use cudarc::driver::{sys, CudaDevice, CudaFunction};
use sha2::{Digest, Sha256};
use std::sync::Arc;

type Range = (u64, u64);
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or("MLA F32 extent overflow".into())
}
fn span(ptr: u64, elements: usize, bytes: usize) -> Result<Range, String> {
    if elements == 0 {
        return Ok((ptr, ptr));
    }
    if ptr == 0 || ptr % bytes as u64 != 0 {
        return Err("MLA F32 pointer/alignment invalid".into());
    }
    let size = u64::try_from(mul(elements, bytes)?).map_err(|e| e.to_string())?;
    Ok((
        ptr,
        ptr.checked_add(size).ok_or("MLA F32 address overflow")?,
    ))
}
fn overlaps(a: Range, b: Range) -> bool {
    a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1
}
fn distinct(output: Range, sources: &[Range]) -> Result<(), String> {
    if sources.iter().any(|&x| overlaps(output, x)) {
        return Err("MLA F32 output aliases a live source".into());
    }
    Ok(())
}
fn integer(value: usize) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| "MLA F32 shape exceeds kernel i32 contract".into())
}

fn head_major_elements(
    rows: usize,
    heads: usize,
    width: usize,
    head_stride_rows: usize,
) -> Result<usize, String> {
    if rows == 0 || heads == 0 || width == 0 || head_stride_rows < rows {
        return Err("invalid physical head-major MLA view".into());
    }
    mul(
        (heads - 1)
            .checked_mul(head_stride_rows)
            .and_then(|v| v.checked_add(rows))
            .ok_or("MLA head-major extent overflow")?,
        width,
    )
}

pub(crate) struct MlaF32Ops {
    _device: Arc<CudaDevice>,
    functions: [CudaFunction; 5],
    limits: [usize; 3],
    pack_shared_limit: usize,
    pub(crate) identity: String,
}
impl MlaF32Ops {
    pub(crate) fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Arc<Self>, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        use sys::CUdevice_attribute as A;
        let attr = |a| {
            device
                .attribute(a)
                .map_err(|e| e.to_string())
                .and_then(|v| usize::try_from(v).map_err(|e| e.to_string()))
        };
        if attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8
            || attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 256
        {
            return Err(
                "selected F32 MLA primitives require admitted Ampere-or-newer 256-thread execution"
                    .into(),
            );
        }
        let names = [
            "mla_f32_rmsnorm",
            "mla_f32_prepare_query",
            "mla_f32_cache_k4",
            "mla_f32_cache_k6",
            "mla_f32_absorb_wkc",
        ];
        device
            .load_ptx(ptx.into(), "shared_mla_f32", &names)
            .map_err(|e| e.to_string())?;
        let functions = names
            .map(|n| {
                device
                    .get_func("shared_mla_f32", n)
                    .ok_or_else(|| format!("F32 MLA symbol absent: {n}"))
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        for function in &functions {
            let mut max_threads = 0;
            unsafe {
                sys::lib().cuFuncGetAttribute(
                    &mut max_threads,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    function.cu_function(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            if max_threads < 256 {
                return Err(
                    "actual loaded F32 MLA function cannot admit its native-size block".into(),
                );
            }
        }
        let max_optin = attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?;
        let mut static_shared = 0;
        unsafe {
            sys::lib().cuFuncGetAttribute(
                &mut static_shared,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                functions[1].cu_function(),
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        let pack_shared_limit = max_optin
            .checked_sub(usize::try_from(static_shared).map_err(|e| e.to_string())?)
            .ok_or("F32 MLA shared-memory accounting invalid")?;
        unsafe {
            sys::lib().cuFuncSetAttribute(
                functions[1].cu_function(),
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                integer(pack_shared_limit)?,
            )
        }
        .result()
        .map_err(|e| e.to_string())?;
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        Ok(Arc::new(Self {
            _device: device,
            functions: functions
                .try_into()
                .map_err(|_| "F32 MLA function inventory")?,
            limits,
            pack_shared_limit,
            identity: format!(
                "batched_native_f32_mla_v1:{:x}",
                Sha256::digest(ptx.as_bytes())
            ),
        }))
    }
    fn grid(&self, x: usize, rows: usize) -> Result<[u32; 3], String> {
        if x == 0 || rows == 0 || x > self.limits[0] || self.limits[1] == 0 {
            return Err("F32 MLA nonempty/grid x/y admission failed".into());
        }
        let y = rows.min(self.limits[1]);
        let z = rows.div_ceil(y);
        if z > self.limits[2] {
            return Err("F32 MLA row folding exceeds actual grid z".into());
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
        grid: [u32; 3],
        shared: usize,
        stream: sys::CUstream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let shared = u32::try_from(shared).map_err(|e| e.to_string())?;
        let result = sys::lib().cuLaunchKernel(
            self.functions[index].cu_function(),
            grid[0],
            grid[1],
            grid[2],
            256,
            1,
            1,
            shared,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("F32 MLA primitive {index} launch: {result:?}"));
        }
        Ok(())
    }
    /// F32 in-place prefix RMS; an optional distinct BF16 view is co-produced.
    pub(crate) unsafe fn norm(
        &self,
        data: u64,
        weight: u64,
        bf16: u64,
        rows: usize,
        width: usize,
        stride: usize,
        bf16_stride: usize,
        eps: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if rows == 0 || width == 0 || stride < width || !eps.is_finite() || eps <= 0.0 {
            return Err("F32 MLA RMS shape/epsilon admission".into());
        }
        let input = span(data, mul(rows, stride)?, 4)?;
        let weights = if weight == 0 {
            None
        } else {
            Some(span(weight, width, 4)?)
        };
        if let Some(w) = weights {
            distinct(input, &[w])?;
        }
        if bf16 != 0 {
            if bf16_stride < width {
                return Err("F32 MLA BF16 RMS stride too small".into());
            }
            let output = span(bf16, mul(rows, bf16_stride)?, 2)?;
            distinct(output, &[input])?;
            if let Some(w) = weights {
                distinct(output, &[w])?;
            }
        }
        let grid = self.grid(1, rows)?;
        let mut data = data;
        let mut weight = weight;
        let mut eps = eps;
        let mut rows = u64::try_from(rows).map_err(|e| e.to_string())?;
        let mut width = integer(width)?;
        let mut stride = integer(stride)?;
        let mut bf16 = bf16;
        let mut bf16_stride = integer(bf16_stride)?;
        self.launch(
            0,
            grid,
            0,
            stream,
            &mut [
                &mut data as *mut _ as _,
                &mut weight as *mut _ as _,
                &mut eps as *mut _ as _,
                &mut rows as *mut _ as _,
                &mut width as *mut _ as _,
                &mut stride as *mut _ as _,
                &mut bf16 as *mut _ as _,
                &mut bf16_stride as *mut _ as _,
            ],
        )
    }
    /// Actual RoPE position array; caller has validated every referenced table row.
    pub(crate) unsafe fn prepare_query(
        &self,
        query: u64,
        kv: u64,
        query_bf16: u64,
        bf16_head_stride_rows: usize,
        cos: u64,
        sin: u64,
        positions: u64,
        rope_table_rows: usize,
        rows: usize,
        heads: usize,
        nope: usize,
        rope: usize,
        kv_rank: usize,
        kv_stride: usize,
        interleaved: bool,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let qwidth = nope.checked_add(rope).ok_or("F32 query width overflow")?;
        if rows == 0
            || heads == 0
            || qwidth == 0
            || rope % 2 != 0
            || kv_rank == 0
            || kv_stride < kv_rank.checked_add(rope).ok_or("F32 KV width overflow")?
        {
            return Err("F32 MLA query/KV geometry admission".into());
        }
        let q = span(query, mul(mul(rows, heads)?, qwidth)?, 4)?;
        let k = span(kv, mul(rows, kv_stride)?, 4)?;
        distinct(q, &[k])?;
        if query_bf16 != 0 {
            distinct(
                span(
                    query_bf16,
                    head_major_elements(rows, heads, qwidth, bf16_head_stride_rows)?,
                    2,
                )?,
                &[q, k],
            )?;
        }
        if rope > 0 {
            if rope_table_rows == 0 {
                return Err("actual RoPE table has no admitted rows".into());
            }
            let cos_span = span(cos, mul(rope_table_rows, rope / 2)?, 4)?;
            let sin_span = span(sin, mul(rope_table_rows, rope / 2)?, 4)?;
            let pos = span(positions, rows, 4)?;
            distinct(q, &[pos, cos_span, sin_span])?;
            distinct(k, &[pos, cos_span, sin_span])?;
            if query_bf16 != 0 {
                distinct(
                    span(
                        query_bf16,
                        head_major_elements(rows, heads, qwidth, bf16_head_stride_rows)?,
                        2,
                    )?,
                    &[pos, cos_span, sin_span],
                )?;
            }
        }
        let shared = mul(mul(rope, 2)?, 4)?;
        if shared > self.pack_shared_limit {
            return Err(
                "actual MLA RoPE staging exceeds admitted function/device shared memory".into(),
            );
        }
        let grid = self.grid(heads, rows)?;
        let mut query = query;
        let mut kv = kv;
        let mut bf = query_bf16;
        let mut bf_head_stride = u64::try_from(bf16_head_stride_rows).map_err(|e| e.to_string())?;
        let mut cos = cos;
        let mut sin = sin;
        let mut pos = positions;
        let mut rows = u64::try_from(rows).map_err(|e| e.to_string())?;
        let mut heads = integer(heads)?;
        let mut nope = integer(nope)?;
        let mut rope = integer(rope)?;
        let mut rank = integer(kv_rank)?;
        let mut stride = integer(kv_stride)?;
        let mut inter = i32::from(interleaved);
        self.launch(
            1,
            grid,
            shared,
            stream,
            &mut [
                &mut query as *mut _ as _,
                &mut kv as *mut _ as _,
                &mut bf as *mut _ as _,
                &mut bf_head_stride as *mut _ as _,
                &mut cos as *mut _ as _,
                &mut sin as *mut _ as _,
                &mut pos as *mut _ as _,
                &mut rows as *mut _ as _,
                &mut heads as *mut _ as _,
                &mut nope as *mut _ as _,
                &mut rope as *mut _ as _,
                &mut rank as *mut _ as _,
                &mut stride as *mut _ as _,
                &mut inter as *mut _ as _,
            ],
        )
    }
    pub(crate) unsafe fn cache(
        &self,
        ckv: u64,
        kpe: u64,
        kv: u64,
        rows: usize,
        base: usize,
        capacity: usize,
        rank: usize,
        stride: usize,
        ccd: usize,
        rope: usize,
        format: u32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if rows == 0
            || rank == 0
            || rank > ccd
            || ccd % 16 != 0
            || rope % 16 != 0
            || stride
                < rank
                    .checked_add(rope)
                    .ok_or("F32 cache KV width overflow")?
        {
            return Err("F32 MLA cache shape admission".into());
        }
        let end = base
            .checked_add(rows)
            .ok_or("F32 cache logical position overflow")?;
        if end > capacity || end > i32::MAX as usize {
            return Err("F32 MLA cache exceeds actual capacity/position contract".into());
        }
        let (index, block_bytes) = match format {
            9 => (2, 10),
            7 => (3, 14),
            _ => return Err("F32 MLA cache supports measured native K4/K6 formats only".into()),
        };
        let input = span(kv, mul(rows, stride)?, 4)?;
        let c = span(ckv, mul(mul(capacity, ccd / 16)?, block_bytes / 2)?, 2)?;
        distinct(c, &[input])?;
        if rope > 0 {
            let k = span(kpe, mul(mul(capacity, rope / 16)?, block_bytes / 2)?, 2)?;
            distinct(k, &[input, c])?;
        }
        let grid = self.grid((ccd / 16).max(rope / 16).div_ceil(256), rows)?;
        let mut ckv = ckv;
        let mut kpe = kpe;
        let mut kv = kv;
        let mut rows = u64::try_from(rows).map_err(|e| e.to_string())?;
        let mut base = integer(base)?;
        let mut rank = integer(rank)?;
        let mut stride = integer(stride)?;
        let mut ccd = integer(ccd)?;
        let mut rope = integer(rope)?;
        self.launch(
            index,
            grid,
            0,
            stream,
            &mut [
                &mut ckv as *mut _ as _,
                &mut kpe as *mut _ as _,
                &mut kv as *mut _ as _,
                &mut rows as *mut _ as _,
                &mut base as *mut _ as _,
                &mut rank as *mut _ as _,
                &mut stride as *mut _ as _,
                &mut ccd as *mut _ as _,
                &mut rope as *mut _ as _,
            ],
        )
    }
    pub(crate) unsafe fn absorb(
        &self,
        output: u64,
        query: u64,
        weight: u64,
        rows: usize,
        heads: usize,
        nope: usize,
        rope: usize,
        ccd: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let width = nope
            .checked_add(rope)
            .ok_or("F32 absorption query width overflow")?;
        if rows == 0 || heads == 0 || width == 0 || ccd == 0 {
            return Err("F32 MLA absorption shape admission".into());
        }
        let q = span(query, mul(mul(rows, heads)?, width)?, 4)?;
        let w = span(weight, mul(mul(heads, nope)?, ccd)?, 2)?;
        let out = span(output, mul(mul(rows, heads)?, ccd)?, 4)?;
        distinct(out, &[q, w])?;
        let grid = self.grid(heads, rows)?;
        let mut output = output;
        let mut query = query;
        let mut weight = weight;
        let mut rows = u64::try_from(rows).map_err(|e| e.to_string())?;
        let mut heads = integer(heads)?;
        let mut nope = integer(nope)?;
        let mut rope = integer(rope)?;
        let mut ccd = integer(ccd)?;
        self.launch(
            4,
            grid,
            0,
            stream,
            &mut [
                &mut output as *mut _ as _,
                &mut query as *mut _ as _,
                &mut weight as *mut _ as _,
                &mut rows as *mut _ as _,
                &mut heads as *mut _ as _,
                &mut nope as *mut _ as _,
                &mut rope as *mut _ as _,
                &mut ccd as *mut _ as _,
            ],
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_typed_extent_and_alias_admission() {
        assert_eq!(span(0x1000, 3, 4).unwrap(), (0x1000, 0x100c));
        assert!(span(0x1002, 3, 4).is_err());
        assert!(span(0x1001, 5, 2).is_err());
        assert!(span(0, 3, 4).is_err());
        assert!(span(u64::MAX - 3, 2, 4).is_err());
        assert!(span(0x1000, usize::MAX, 4).is_err());
        assert!(distinct((0x1000, 0x1010), &[(0x1010, 0x1020)]).is_ok());
        assert!(distinct((0x1000, 0x1010), &[(0x100c, 0x1020)]).is_err());
    }
    #[test]
    fn empty_ranges_never_alias_live_regions() {
        assert!(!overlaps((0x1004, 0x1004), (0x1000, 0x1008)));
        assert!(!overlaps((0x1000, 0x1008), (0x1004, 0x1004)));
        assert!(distinct((0x1000, 0x1008), &[span(0x1004, 0, 2).unwrap()]).is_ok());
    }
    #[test]
    fn physical_head_stride_preserves_partition_views() {
        assert_eq!(head_major_elements(3, 2, 5, 7).unwrap(), 50);
        assert_eq!(head_major_elements(7, 2, 5, 7).unwrap(), 70);
        assert!(head_major_elements(8, 2, 5, 7).is_err());
        assert!(head_major_elements(3, usize::MAX, 5, 7).is_err());
    }
}

#[cfg(feature = "shared-kda-test")]
pub(crate) fn load_serving(device: Arc<CudaDevice>) -> Result<Arc<MlaF32Ops>, String> {
    MlaF32Ops::load(
        device,
        include_str!(concat!(env!("OUT_DIR"), "/mla_f32_ops.ptx")),
    )
}
#[cfg(not(feature = "shared-kda-test"))]
pub(crate) fn load_serving(_device: Arc<CudaDevice>) -> Result<Arc<MlaF32Ops>, String> {
    Err("selected F32 MLA primitives were not built".into())
}

/// Actual simultaneously required MLA row extents; shared by allocation/budget admission.
pub(crate) fn workspace_row_elements(
    heads: usize,
    ccd: usize,
    q_rank: usize,
    kv_rank: usize,
    nope: usize,
    rope: usize,
) -> Result<(usize, usize), String> {
    let attention = mul(heads, ccd)?;
    let kv = kv_rank.checked_add(rope).ok_or("MLA KV width overflow")?;
    let query = mul(
        heads,
        nope.checked_add(rope).ok_or("MLA query width overflow")?,
    )?;
    Ok((attention.max(q_rank).max(kv), attention.max(query)))
}
#[cfg(test)]
mod workspace_tests {
    use super::*;
    #[test]
    fn all_loaded_widths_and_overflow_are_accounted() {
        assert_eq!(workspace_row_elements(3, 16, 7, 8, 4, 0).unwrap(), (48, 48));
        assert_eq!(
            workspace_row_elements(3, 16, 57, 8, 4, 0).unwrap(),
            (57, 48)
        );
        assert_eq!(
            workspace_row_elements(3, 16, 7, 61, 4, 0).unwrap(),
            (61, 48)
        );
        assert_eq!(
            workspace_row_elements(3, 16, 7, 8, 4, 16).unwrap(),
            (48, 60)
        );
        assert!(workspace_row_elements(usize::MAX, 2, 7, 8, 4, 0).is_err());
        assert!(workspace_row_elements(3, 16, 7, usize::MAX, 4, 16).is_err());
        assert!(workspace_row_elements(3, 16, 7, 8, usize::MAX, 16).is_err());
    }
}
