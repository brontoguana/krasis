use crate::cuda_region::CudaRegion;
// Explicit, prepared F32 cuBLASLt operations for the selected MLA contract.
// Preparation owns descriptors and the device/handle. Enqueue allocates nothing.
// The unsafe caller retains real tensor owners through stream/graph completion,
// including an error return; this module does not substitute a different kernel.
use cudarc::{
    cublas::sys::cublasOperation_t,
    cublaslt::{result as lt, sys},
    driver::{sys as cu, CudaDevice, CudaSlice, DevicePtr, DeviceSlice},
};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, mem::size_of, sync::Arc};

fn product(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| "F32 GEMM extent overflow".into())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub leading: usize,
    pub batch: usize,
    pub stride: usize,
}
impl Matrix {
    pub(crate) fn elements(self, output: bool) -> Result<usize, String> {
        if self.rows == 0 || self.cols == 0 || self.batch == 0 || self.leading < self.rows {
            return Err("invalid F32 GEMM matrix geometry".into());
        }
        for value in [self.rows, self.cols, self.leading, self.batch] {
            i32::try_from(value).map_err(|_| "F32 GEMM matrix exceeds admitted i32 geometry")?;
        }
        let one = product(self.cols - 1, self.leading)?
            .checked_add(self.rows)
            .ok_or("F32 GEMM matrix extent overflow")?;
        if output && self.batch > 1 && self.stride < one {
            return Err("F32 GEMM output batches overlap".into());
        }
        let count = product(self.batch - 1, self.stride)?
            .checked_add(one)
            .ok_or("F32 GEMM batch extent overflow")?;
        i64::try_from(self.stride).map_err(|_| "F32 GEMM stride exceeds i64")?;
        product(count, size_of::<f32>())?;
        Ok(count)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Shape {
    pub a: Matrix,
    pub b: Matrix,
    pub c: Matrix,
    pub transpose_a: bool,
    pub transpose_b: bool,
}
impl Shape {
    pub(crate) fn validate(self) -> Result<(), String> {
        self.a.elements(false)?;
        self.b.elements(false)?;
        self.c.elements(true)?;
        let (ar, ac) = if self.transpose_a {
            (self.a.cols, self.a.rows)
        } else {
            (self.a.rows, self.a.cols)
        };
        let (br, bc) = if self.transpose_b {
            (self.b.cols, self.b.rows)
        } else {
            (self.b.rows, self.b.cols)
        };
        if ac != br
            || self.c.rows != ar
            || self.c.cols != bc
            || self.a.batch != self.c.batch
            || self.b.batch != self.c.batch
        {
            return Err("incompatible F32 GEMM dimensions/batches".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum BorrowedBuffer<'a> {
    F32(&'a CudaSlice<f32>),
    Bytes(&'a CudaSlice<u8>),
    Bf16(&'a CudaSlice<u16>),
    I32(&'a CudaSlice<i32>),
    Region(&'a dyn CudaRegion),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Element {
    F32,
    Bf16,
    I32,
    Bytes,
}
/// Actual allocation borrow, rather than an assertion that a raw address has an owner.
pub(crate) struct Span<'a> {
    owner: BorrowedBuffer<'a>,
    ptr: u64,
    bytes: usize,
    element: Element,
}
impl<'a> Span<'a> {
    pub(crate) fn f32(
        owner: &'a CudaSlice<f32>,
        offset: usize,
        elements: usize,
    ) -> Result<Self, String> {
        if offset.checked_add(elements).ok_or("F32 view overflow")? > owner.len() {
            return Err("F32 view exceeds actual allocation".into());
        }
        let bytes = product(elements, size_of::<f32>())?;
        let ptr = (*owner.device_ptr())
            .checked_add(product(offset, size_of::<f32>())? as u64)
            .ok_or("F32 view address overflow")?;
        Ok(Self {
            owner: BorrowedBuffer::F32(owner),
            ptr,
            bytes,
            element: Element::F32,
        })
    }
    pub(crate) fn bf16(
        owner: &'a CudaSlice<u16>,
        offset: usize,
        elements: usize,
    ) -> Result<Self, String> {
        if offset.checked_add(elements).ok_or("BF16 view overflow")? > owner.len() {
            return Err("BF16 view exceeds actual allocation".into());
        }
        let bytes = product(elements, size_of::<u16>())?;
        let ptr = (*owner.device_ptr())
            .checked_add(product(offset, size_of::<u16>())? as u64)
            .ok_or("BF16 view address overflow")?;
        Ok(Self {
            owner: BorrowedBuffer::Bf16(owner),
            ptr,
            bytes,
            element: Element::Bf16,
        })
    }
    pub(crate) fn i32(
        owner: &'a CudaSlice<i32>,
        offset: usize,
        elements: usize,
    ) -> Result<Self, String> {
        if offset.checked_add(elements).ok_or("I32 view overflow")? > owner.len() {
            return Err("I32 view exceeds actual allocation".into());
        }
        let bytes = product(elements, size_of::<i32>())?;
        let ptr = (*owner.device_ptr())
            .checked_add(product(offset, size_of::<i32>())? as u64)
            .ok_or("I32 view address overflow")?;
        Ok(Self {
            owner: BorrowedBuffer::I32(owner),
            ptr,
            bytes,
            element: Element::I32,
        })
    }
    /// Explicit F32 view of an owned byte workspace, with checked alignment.
    pub(crate) fn f32_bytes(
        owner: &'a CudaSlice<u8>,
        offset: usize,
        elements: usize,
    ) -> Result<Self, String> {
        let mut span = Self::bytes(owner, offset, product(elements, size_of::<f32>())?)?;
        if span.ptr % size_of::<f32>() as u64 != 0 {
            return Err("F32 workspace alignment invalid".into());
        }
        span.element = Element::F32;
        Ok(span)
    }
    pub(crate) fn element(&self) -> Element {
        self.element
    }
    /// Explicit logical view while borrowing a canonical, sealed real allocation owner.
    /// Caller supplies the producer's real element type; this is not a numeric conversion.
    pub(crate) unsafe fn region(
        owner: &'a dyn CudaRegion,
        offset: usize,
        bytes: usize,
        element: Element,
    ) -> Result<Self, String> {
        let (start, end) = owner.extent()?;
        let ptr = start
            .checked_add(u64::try_from(offset).map_err(|e| e.to_string())?)
            .ok_or("region offset overflow")?;
        let stop = ptr
            .checked_add(u64::try_from(bytes).map_err(|e| e.to_string())?)
            .ok_or("region extent overflow")?;
        if ptr < start || stop > end {
            return Err("logical view exceeds actual retained region".into());
        }
        let alignment = match element {
            Element::F32 | Element::I32 => 4,
            Element::Bf16 => 2,
            Element::Bytes => 1,
        };
        if ptr % alignment != 0 {
            return Err("retained region element alignment invalid".into());
        }
        Ok(Self {
            owner: BorrowedBuffer::Region(owner),
            ptr,
            bytes,
            element,
        })
    }
    pub(crate) fn pointer(&self) -> u64 {
        self.ptr
    }
    pub(crate) fn bytes(
        owner: &'a CudaSlice<u8>,
        offset: usize,
        bytes: usize,
    ) -> Result<Self, String> {
        if offset.checked_add(bytes).ok_or("byte view overflow")? > owner.len() {
            return Err("byte view exceeds actual allocation".into());
        }
        let ptr = (*owner.device_ptr())
            .checked_add(offset as u64)
            .ok_or("byte view address overflow")?;
        Ok(Self {
            owner: BorrowedBuffer::Bytes(owner),
            ptr,
            bytes,
            element: Element::Bytes,
        })
    }
    pub(crate) fn context(&self) -> cu::CUcontext {
        match &self.owner {
            BorrowedBuffer::F32(b) => *b.device().cu_primary_ctx(),
            BorrowedBuffer::Bytes(b) => *b.device().cu_primary_ctx(),
            BorrowedBuffer::Bf16(b) => *b.device().cu_primary_ctx(),
            BorrowedBuffer::I32(b) => *b.device().cu_primary_ctx(),
            BorrowedBuffer::Region(b) => b.context(),
        }
    }
    pub(crate) fn alignment(&self) -> Result<usize, String> {
        if self.ptr == 0 {
            return Err("null allocation has no admitted alignment".into());
        }
        let low = self.ptr & self.ptr.wrapping_neg();
        usize::try_from(low).map_err(|_| "allocation alignment exceeds host representation".into())
    }
    pub(crate) fn subview(&self, offset: usize, bytes: usize) -> Result<Span<'_>, String> {
        self.range(
            offset
                .checked_add(bytes)
                .ok_or("retained subview extent overflow")?,
        )?;
        let ptr = self
            .ptr
            .checked_add(u64::try_from(offset).map_err(|e| e.to_string())?)
            .ok_or("retained subview address overflow")?;
        let alignment = match self.element {
            Element::F32 | Element::I32 => 4,
            Element::Bf16 => 2,
            Element::Bytes => 1,
        };
        if ptr % alignment != 0 {
            return Err("retained subview element alignment invalid".into());
        }
        Ok(Span {
            owner: self.owner,
            ptr,
            bytes,
            element: self.element,
        })
    }
    pub(crate) fn full_range(&self) -> Result<(u64, u64), String> {
        self.range(self.bytes)
    }
    pub(crate) fn range(&self, bytes: usize) -> Result<(u64, u64), String> {
        if bytes > self.bytes || (bytes > 0 && self.ptr == 0) {
            return Err("F32 GEMM view below actual required capacity".into());
        }
        Ok((
            self.ptr,
            self.ptr
                .checked_add(u64::try_from(bytes).map_err(|e| e.to_string())?)
                .ok_or("F32 GEMM pointer range overflow")?,
        ))
    }
}
fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1
}

struct Handle {
    raw: sys::cublasLtHandle_t,
    device: Arc<CudaDevice>,
    version: usize,
    uuid: [u8; 16],
}
impl Handle {
    fn current(&self, stream: cu::CUstream) -> Result<(), String> {
        let mut current = std::ptr::null_mut();
        unsafe { cu::lib().cuCtxGetCurrent(&mut current) }
            .result()
            .map_err(|e| format!("Lt current context: {e:?}"))?;
        if current != *self.device.cu_primary_ctx() {
            return Err("F32 GEMM current context differs from actual owner".into());
        }
        if !stream.is_null() {
            let mut context = std::ptr::null_mut();
            unsafe { cu::lib().cuStreamGetCtx(stream, &mut context) }
                .result()
                .map_err(|e| format!("Lt stream context: {e:?}"))?;
            if context != current {
                return Err("F32 GEMM foreign-context stream".into());
            }
        }
        Ok(())
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        // Prepared plans/graphs must outlive outstanding work. The persistent
        // model owner performs completion before releasing its final bank.
        if let Err(error) =
            unsafe { cu::lib().cuCtxPushCurrent_v2(*self.device.cu_primary_ctx()) }.result()
        {
            eprintln!("ERROR: Lt release could not enter owning CUDA context: {error:?}");
            return;
        }
        if let Err(error) = unsafe { lt::destroy_handle(self.raw) } {
            eprintln!("ERROR: Lt handle release failed: {error:?}");
        }
        let mut restored = std::ptr::null_mut();
        if let Err(error) = unsafe { cu::lib().cuCtxPopCurrent_v2(&mut restored) }.result() {
            eprintln!("ERROR: Lt release could not restore caller CUDA context: {error:?}");
        }
    }
}
struct Layout(sys::cublasLtMatrixLayout_t);
impl Layout {
    fn new(m: Matrix) -> Result<Self, String> {
        let result = Self(
            lt::create_matrix_layout(
                sys::cudaDataType::CUDA_R_32F,
                m.rows as u64,
                m.cols as u64,
                m.leading as i64,
            )
            .map_err(|e| e.to_string())?,
        );
        let batch = i32::try_from(m.batch).map_err(|e| e.to_string())?;
        let stride = i64::try_from(m.stride).map_err(|e| e.to_string())?;
        unsafe {
            lt::set_matrix_layout_attribute(
                result.0,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch as *const _ as _,
                size_of::<i32>(),
            )
            .map_err(|e| e.to_string())?;
            lt::set_matrix_layout_attribute(
                result.0,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &stride as *const _ as _,
                size_of::<i64>(),
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(result)
    }
}
impl Drop for Layout {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matrix_layout(self.0);
        }
    }
}
struct Desc(sys::cublasLtMatmulDesc_t);
impl Desc {
    fn new(shape: Shape) -> Result<Self, String> {
        let result = Self(
            lt::create_matmul_desc(
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                sys::cudaDataType::CUDA_R_32F,
            )
            .map_err(|e| e.to_string())?,
        );
        for (attribute, transposed) in [
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                shape.transpose_a,
            ),
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                shape.transpose_b,
            ),
        ] {
            let op = if transposed {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            };
            unsafe {
                lt::set_matmul_desc_attribute(
                    result.0,
                    attribute,
                    &op as *const _ as _,
                    size_of::<cublasOperation_t>(),
                )
            }
            .map_err(|e| e.to_string())?;
        }
        Ok(result)
    }
}
impl Drop for Desc {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matmul_desc(self.0);
        }
    }
}
struct Pref(sys::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matmul_pref(self.0);
        }
    }
}

/// A runtime-discovered configuration, tied to its actual CUDA/Lt owner.
/// No public constructor and no hardware/model algorithm ID constants.
pub(crate) struct Candidate {
    handle: Arc<Handle>,
    algorithm: sys::cublasLtMatmulAlgo_t,
    alignments: [usize; 4],
    pub(crate) discovery_workspace_bytes: usize,
}
pub(crate) struct Candidates {
    pub(crate) values: Vec<Candidate>,
}
impl Candidates {
    pub(crate) fn discover(
        device: Arc<CudaDevice>,
        shape: Shape,
        workspace_cap: usize,
        alignments: [usize; 4],
    ) -> Result<Self, String> {
        shape.validate()?;
        let mut current = std::ptr::null_mut();
        unsafe { cu::lib().cuCtxGetCurrent(&mut current) }
            .result()
            .map_err(|e| e.to_string())?;
        if current != *device.cu_primary_ctx() {
            return Err("Lt discovery requires the actual owning context to be current".into());
        }
        let mut uuid = unsafe { std::mem::zeroed::<cu::CUuuid>() };
        unsafe { cu::lib().cuDeviceGetUuid_v2(&mut uuid, *device.cu_device()) }
            .result()
            .map_err(|e| e.to_string())?;
        let handle = Arc::new(Handle {
            raw: lt::create_handle().map_err(|e| e.to_string())?,
            device,
            version: unsafe { sys::lib().cublasLtGetVersion() },
            uuid: uuid.bytes.map(|b| b as u8),
        });
        let desc = Desc::new(shape)?;
        let a = Layout::new(shape.a)?;
        let b = Layout::new(shape.b)?;
        let c = Layout::new(shape.c)?;
        let pref = Pref(lt::create_matmul_pref().map_err(|e| e.to_string())?);
        unsafe {
            lt::set_matmul_pref_attribute(
                pref.0,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_cap as *const _ as _,
                size_of::<usize>(),
            )
        }
        .map_err(|e| e.to_string())?;
        let attrs = [
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        ];
        for (attr, alignment) in attrs.into_iter().zip(alignments) {
            let value = u32::try_from(alignment).map_err(|_| "Lt alignment exceeds u32")?;
            if value < size_of::<f32>() as u32 || !value.is_power_of_two() {
                return Err("invalid actual F32 allocation alignment".into());
            }
            unsafe {
                lt::set_matmul_pref_attribute(
                    pref.0,
                    attr,
                    &value as *const _ as _,
                    size_of::<u32>(),
                )
            }
            .map_err(|e| e.to_string())?;
        }
        let mut capacity = 1i32;
        let mut values = Vec::new();
        loop {
            let mut list =
                vec![
                    unsafe { std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>() };
                    capacity as usize
                ];
            let mut count = 0;
            unsafe {
                sys::lib().cublasLtMatmulAlgoGetHeuristic(
                    handle.raw,
                    desc.0,
                    a.0,
                    b.0,
                    c.0,
                    c.0,
                    pref.0,
                    capacity,
                    list.as_mut_ptr(),
                    &mut count,
                )
            }
            .result()
            .map_err(|e| format!("Lt full candidate discovery: {e:?}"))?;
            if count < 0 || count > capacity {
                return Err("invalid Lt discovery count".into());
            }
            if count == capacity {
                capacity = capacity
                    .checked_mul(2)
                    .ok_or("Lt discovery count overflow")?;
                continue;
            }
            for item in list.into_iter().take(count as usize) {
                if item.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                    || item.workspaceSize > workspace_cap
                {
                    return Err(format!("invalid Lt heuristic result: {item:?}"));
                }
                let attrs = [
                    sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
                    sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
                    sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
                    sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
                ];
                let mut required = [0usize; 4];
                for (i, attr) in attrs.into_iter().enumerate() {
                    let mut v = 0u32;
                    let mut written = 0usize;
                    unsafe {
                        sys::lib().cublasLtMatmulAlgoCapGetAttribute(
                            &item.algo,
                            attr,
                            &mut v as *mut _ as _,
                            size_of::<u32>(),
                            &mut written,
                        )
                    }
                    .result()
                    .map_err(|e| format!("Lt alignment capability: {e:?}"))?;
                    if written != size_of::<u32>() || v == 0 || !v.is_power_of_two() {
                        return Err("invalid Lt alignment capability".into());
                    }
                    required[i] = v as usize;
                }
                values.push(Candidate {
                    handle: handle.clone(),
                    algorithm: item.algo,
                    alignments: required,
                    discovery_workspace_bytes: item.workspaceSize,
                });
            }
            break;
        }
        if values.is_empty() {
            return Err(
                "no admitted explicit F32 Lt candidate; refusing alternate arithmetic".into(),
            );
        }
        Ok(Self { values })
    }
}

pub(crate) enum Preparation {
    Ready(Prepared),
    Unsupported(String),
}
pub(crate) struct Prepared {
    // Descriptors are released before the final handle/device owner.
    desc: Desc,
    a: Layout,
    b: Layout,
    c: Layout,
    shape: Shape,
    algorithm: sys::cublasLtMatmulAlgo_t,
    alignments: [usize; 4],
    pub(crate) workspace_bytes: usize,
    pub(crate) identity: String,
    handle: Arc<Handle>,
}
impl Candidate {
    pub(crate) fn family_identity(&self) -> String {
        format!(
            "stable_lt_f32_family_v1:{:x}",
            Sha256::digest(
                format!(
                    "{:?}:{:?}:{}",
                    self.handle.uuid, self.algorithm.data, self.handle.version
                )
                .as_bytes()
            )
        )
    }
    pub(crate) fn configuration(&self) -> [u64; 8] {
        self.algorithm.data
    }
    pub(crate) fn prepare(&self, shape: Shape, workspace_cap: usize) -> Result<Prepared, String> {
        match self.admit(shape, workspace_cap)? {
            Preparation::Ready(plan) => Ok(plan),
            Preparation::Unsupported(reason) => Err(reason),
        }
    }
    pub(crate) fn admit(&self, shape: Shape, workspace_cap: usize) -> Result<Preparation, String> {
        shape.validate()?;
        self.handle.current(*self.handle.device.cu_stream())?;
        let desc = Desc::new(shape)?;
        let a = Layout::new(shape.a)?;
        let b = Layout::new(shape.b)?;
        let c = Layout::new(shape.c)?;
        let mut check = unsafe { std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>() };
        let status = unsafe {
            sys::lib().cublasLtMatmulAlgoCheck(
                self.handle.raw,
                desc.0,
                a.0,
                b.0,
                c.0,
                c.0,
                &self.algorithm,
                &mut check,
            )
        };
        if status == sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED {
            return Ok(Preparation::Unsupported(format!(
                "explicit Lt shape unsupported: {status:?}"
            )));
        }
        status
            .result()
            .map_err(|e| format!("Lt admission API failed: {e:?}"))?;
        if check.state == sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
            || check.workspaceSize > workspace_cap
        {
            return Ok(Preparation::Unsupported(format!(
                "explicit Lt shape/workspace not admitted: {check:?}, capacity={workspace_cap}"
            )));
        }
        check
            .state
            .result()
            .map_err(|e| format!("Lt candidate admission state failed: {e:?}"))?;
        let identity = format!(
            "stable_mla_f32_pedantic_v1:{:x}",
            Sha256::digest(
                format!(
                    "{:?}:{:?}:{}:{:?}",
                    self.handle.uuid, self.algorithm.data, self.handle.version, shape
                )
                .as_bytes()
            )
        );
        Ok(Preparation::Ready(Prepared {
            desc,
            a,
            b,
            c,
            shape,
            algorithm: self.algorithm,
            alignments: self.alignments,
            workspace_bytes: check.workspaceSize,
            identity,
            handle: self.handle.clone(),
        }))
    }
}
impl Prepared {
    /// No allocation, heuristic selection, synchronization, or Python crossing.
    /// Caller keeps the borrowed allocations and this plan alive through actual
    /// stream/graph completion even if Lt returns an error after enqueue.
    pub(crate) unsafe fn enqueue(
        &self,
        a: &Span<'_>,
        b: &Span<'_>,
        c: &Span<'_>,
        output: &Span<'_>,
        workspace: Option<&Span<'_>>,
        alpha: f32,
        beta: f32,
        stream: cu::CUstream,
    ) -> Result<(), String> {
        if !alpha.is_finite() || !beta.is_finite() {
            return Err("nonfinite Lt scale".into());
        }
        self.handle.current(stream)?;
        let layouts = [self.shape.a, self.shape.b, self.shape.c, self.shape.c];
        let spans = [a, b, c, output];
        let mut ranges = [(0, 0); 4];
        for i in 0..4 {
            if spans[i].element() != Element::F32 {
                return Err("F32 Lt operand dtype mismatch".into());
            }
            if spans[i].context() != *self.handle.device.cu_primary_ctx() {
                return Err("F32 Lt tensor from foreign owner context".into());
            }
            let n = product(layouts[i].elements(i == 3)?, size_of::<f32>())?;
            ranges[i] = spans[i].range(n)?;
            if spans[i].ptr % size_of::<f32>() as u64 != 0
                || spans[i].ptr % self.alignments[i] as u64 != 0
                || (layouts[i].batch > 1
                    && product(layouts[i].stride, size_of::<f32>())? % self.alignments[i] != 0)
            {
                return Err("F32 Lt allocation/batch alignment not admitted".into());
            }
        }
        if overlap(ranges[3], ranges[0])
            || overlap(ranges[3], ranges[1])
            || (overlap(ranges[3], ranges[2]) && ranges[3] != ranges[2])
        {
            return Err("F32 Lt output aliases a live operand or partial C view".into());
        }
        let wp = if self.workspace_bytes == 0 {
            0
        } else {
            let w = workspace.ok_or("required Lt workspace absent")?;
            if w.context() != *self.handle.device.cu_primary_ctx() {
                return Err("Lt workspace foreign owner".into());
            }
            let r = w.range(self.workspace_bytes)?;
            if ranges.iter().any(|&x| overlap(x, r)) {
                return Err("Lt workspace aliases matrix operand".into());
            }
            // CUDA/cuBLASLt workspace contract requires 256-byte alignment.
            if w.ptr % 256 != 0 {
                return Err("Lt workspace not CUDA-aligned".into());
            }
            w.ptr
        };
        sys::lib().cublasLtMatmul(self.handle.raw,self.desc.0,&alpha as *const _ as *const c_void,
            a.ptr as _,self.a.0,b.ptr as _,self.b.0,&beta as *const _ as *const c_void,c.ptr as _,self.c.0,
            output.ptr as _,self.c.0,&self.algorithm,wp as _,self.workspace_bytes,stream as _)
            .result().map_err(|e|format!("selected F32 Lt enqueue failed; current sequence requires error recovery: {e:?}"))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padded_columns_and_broadcast_are_physical_views() {
        let m = Matrix {
            rows: 3,
            cols: 2,
            leading: 5,
            batch: 4,
            stride: 0,
        };
        assert_eq!(m.elements(false).unwrap(), 8);
        assert!(m.elements(true).is_err());
        assert_eq!(Matrix { stride: 11, ..m }.elements(true).unwrap(), 41);
    }
    #[test]
    fn shape_overflow_and_incompatible_contractions_reject() {
        assert!(Matrix {
            rows: 1,
            cols: 2,
            leading: usize::MAX,
            batch: 2,
            stride: usize::MAX
        }
        .elements(false)
        .is_err());
        let m = Matrix {
            rows: 2,
            cols: 3,
            leading: 2,
            batch: 1,
            stride: 0,
        };
        assert!(Shape {
            a: m,
            b: m,
            c: m,
            transpose_a: false,
            transpose_b: false
        }
        .validate()
        .is_err());
    }
    #[test]
    fn empty_ranges_are_not_live_aliases() {
        assert!(!overlap((4, 4), (0, 8)));
        assert!(overlap((4, 8), (6, 9)));
    }
}
