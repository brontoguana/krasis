use crate::gpu_prefill::GpuBuf;
impl<T: DeviceAddress> DeviceAddress for Arc<T> {
    fn address(&self) -> u64 {
        (**self).address()
    }
}
trait DeviceAddress {
    fn address(&self) -> u64;
}
impl<T: cudarc::driver::DeviceRepr> DeviceAddress for CudaSlice<T> {
    fn address(&self) -> u64 {
        *self.device_ptr()
    }
}
impl<T: Copy> DeviceAddress for GpuBuf<T> {
    fn address(&self) -> u64 {
        *self.device_ptr()
    }
}
struct NativeStream {
    handle: sys::CUstream,
    owned: bool,
    prefill_owner: Option<Arc<crate::gpu_prefill::PrefillComputeStream>>,
    device: Arc<CudaDevice>,
}
// SAFETY: opaque immutable stream handle retains its actual device/stream owner.
// Driver use validates/binds the receiving thread; arena/family mutation is locked.
// Arc last-owner teardown cannot race a live borrowed stream invocation.
unsafe impl Send for NativeStream {}
unsafe impl Sync for NativeStream {}
impl NativeStream {
    fn from_prefill_owner(
        owner: Arc<crate::gpu_prefill::PrefillComputeStream>,
    ) -> Result<Arc<Self>, String> {
        let x = Arc::new(Self {
            handle: owner.handle(),
            owned: false,
            device: owner.device(),
            prefill_owner: Some(owner),
        });
        x.validate()?;
        Ok(x)
    }
    fn same_backing_owner(left: &Arc<Self>, right: &Arc<Self>) -> Result<bool, String> {
        left.validate()?;
        right.validate()?;
        if !Arc::ptr_eq(&left.device, &right.device) || left.handle != right.handle {
            return Ok(false);
        }
        if Arc::ptr_eq(left, right) {
            return Ok(true);
        }
        if left.owned || right.owned {
            return Ok(false);
        }
        Ok(match (&left.prefill_owner, &right.prefill_owner) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => left.handle == *left.device.cu_stream(),
            _ => false,
        })
    }
    fn validate(&self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let (mut current, mut stream_ctx) = (std::ptr::null_mut(), std::ptr::null_mut());
        let mut ordinal = 0;
        check(
            unsafe { sys::lib().cuCtxGetCurrent(&mut current) },
            "native currentcontext",
        )?;
        if self.handle.is_null() {
            if self.owned {
                return Err("owned stream cannot be legacy default".into());
            }
            // Legacy stream belongs to the bound primary context; no fictitious stream query.
            stream_ctx = current;
        } else {
            check(
                unsafe { sys::lib().cuStreamGetCtx(self.handle, &mut stream_ctx) },
                "native streamcontext",
            )?;
        }
        check(
            unsafe { sys::lib().cuCtxGetDevice(&mut ordinal) },
            "native contextdevice",
        )?;
        if current != *self.device.cu_primary_ctx()
            || stream_ctx != current
            || ordinal < 0
            || ordinal as usize != self.device.ordinal()
        {
            return Err("native stream/context/device mismatch".into());
        }
        Ok(())
    }
    fn borrowed_device(device: Arc<CudaDevice>) -> Result<Arc<Self>, String> {
        let x = Arc::new(Self {
            handle: *device.cu_stream(),
            owned: false,
            prefill_owner: None,
            device,
        });
        x.validate()?;
        Ok(x)
    }
    /// The caller must retain the nondefault stream owner for the entire borrowed lifetime.
    unsafe fn borrow_handle(
        device: Arc<CudaDevice>,
        handle: sys::CUstream,
    ) -> Result<Arc<Self>, String> {
        let x = Arc::new(Self {
            handle,
            owned: false,
            prefill_owner: None,
            device,
        });
        x.validate()?;
        Ok(x)
    }
    fn create(device: Arc<CudaDevice>) -> Result<Arc<Self>, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut handle = std::ptr::null_mut();
        check(
            unsafe {
                sys::lib().cuStreamCreate(
                    &mut handle,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "native explicit streamcreate",
        )?;
        let x = Arc::new(Self {
            handle,
            owned: true,
            prefill_owner: None,
            device,
        });
        x.validate()?;
        Ok(x)
    }
    fn drain(&self) -> Result<(), String> {
        self.validate()?;
        check(
            unsafe { sys::lib().cuStreamSynchronize(self.handle) },
            "native owned streamdrain",
        )
    }
}
impl Drop for NativeStream {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            if let Err(e) = self.drain() {
                crate::vram_monitor::fatal_cuda_context_error("native stream final completion", &e);
            }
            let rc = unsafe { sys::lib().cuStreamDestroy_v2(self.handle) };
            if rc != sys::CUresult::CUDA_SUCCESS {
                crate::vram_monitor::fatal_cuda_context_error(
                    "native stream release",
                    &format!("{rc:?}"),
                );
            }
        }
    }
}
#[derive(Serialize)]
struct NativePointerAdmission {
    ptr: u64,
    requested_bytes: usize,
    allocation_base: u64,
    allocation_span_bytes: usize,
    device_ordinal: i32,
    device_memory: bool,
    context_matches: bool,
    pointer_context: usize,
    allocation_domain: &'static str,
    pool_handle: usize,
    pool_access_flags: Option<u32>,
}
fn native_allocation_domain(
    pointer_context: usize,
    primary_context: usize,
    pool_handle: usize,
    pool_access: Option<u32>,
) -> Result<&'static str, String> {
    if primary_context == 0 {
        return Err("null executing allocation context".into());
    }
    if pointer_context != 0 {
        if pointer_context != primary_context {
            return Err("foreign synchronous allocation context".into());
        }
        return Ok("context_owned");
    }
    if pool_handle == 0
        || pool_access != Some(sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE as u32)
    {
        return Err(
            "context-independent allocation requires verified pool read/write access".into(),
        );
    }
    Ok("context_independent_pool")
}
fn native_pointer_admission(
    stream: &NativeStream,
    ptr: u64,
    bytes: usize,
) -> Result<NativePointerAdmission, String> {
    stream.validate()?;
    if ptr == 0 || bytes == 0 {
        return Err("zero native allocation/view".into());
    }
    let (mut base, mut span, mut ordinal, mut kind, mut ctx) = (
        0u64,
        0usize,
        -1i32,
        0i32,
        std::ptr::null_mut::<sys::CUctx_st>(),
    );
    check(
        unsafe { sys::lib().cuMemGetAddressRange_v2(&mut base, &mut span, ptr) },
        "native allocation extent",
    )?;
    check(
        unsafe {
            sys::lib().cuPointerGetAttribute(
                (&mut ctx as *mut sys::CUcontext).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                ptr,
            )
        },
        "native pointer context",
    )?;
    check(
        unsafe {
            sys::lib().cuPointerGetAttribute(
                (&mut ordinal as *mut i32).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                ptr,
            )
        },
        "native pointer ordinal",
    )?;
    check(
        unsafe {
            sys::lib().cuPointerGetAttribute(
                (&mut kind as *mut i32).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                ptr,
            )
        },
        "native pointer memorytype",
    )?;
    let mut pool = std::ptr::null_mut::<sys::CUmemPoolHandle_st>();
    let mut pool_access_flags = None;
    if ctx.is_null() {
        check(
            unsafe {
                sys::lib().cuPointerGetAttribute(
                    (&mut pool as *mut sys::CUmemoryPool).cast(),
                    sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                    ptr,
                )
            },
            "native pool allocation owner",
        )?;
        if pool.is_null() {
            return Err("null allocation context without owning pool".into());
        }
        let mut flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE;
        let mut location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: i32::try_from(stream.device.ordinal()).map_err(|_| "native pool device extent")?,
        };
        check(
            unsafe { sys::lib().cuMemPoolGetAccess(&mut flags, pool, &mut location) },
            "native allocation pool access",
        )?;
        pool_access_flags = Some(flags as u32);
    }
    let allocation_domain = native_allocation_domain(
        ctx as usize,
        *stream.device.cu_primary_ctx() as usize,
        pool as usize,
        pool_access_flags,
    )?;
    let end = ptr
        .checked_add(bytes as u64)
        .ok_or("native view endoverflow")?;
    let allocation_end = base
        .checked_add(span as u64)
        .ok_or("native allocation endoverflow")?;
    if ptr < base
        || end > allocation_end
        || ordinal < 0
        || ordinal as usize != stream.device.ordinal()
        || kind != sys::CUmemorytype::CU_MEMORYTYPE_DEVICE as i32
    {
        return Err("native view outside owning allocation/context/device".into());
    }
    Ok(NativePointerAdmission {
        ptr,
        requested_bytes: bytes,
        allocation_base: base,
        allocation_span_bytes: span,
        device_ordinal: ordinal,
        device_memory: true,
        context_matches: ctx == *stream.device.cu_primary_ctx(),
        pointer_context: ctx as usize,
        allocation_domain,
        pool_handle: pool as usize,
        pool_access_flags,
    })
}
struct NativeCudaF32Bf16Storage {
    owner: Arc<CudaSlice<f32>>,
    bf16_elements: usize,
}
enum NativeBf16Allocation {
    GpuBuf(Arc<GpuBuf<u16>>),
    CudaSlice(Arc<CudaSlice<u16>>),
    CudaF32Bf16Storage(NativeCudaF32Bf16Storage),
}
impl NativeBf16Allocation {
    fn cuda_f32_bf16_storage(owner: Arc<CudaSlice<f32>>) -> Result<Self, String> {
        let bf16_elements = owner
            .len()
            .checked_mul(std::mem::size_of::<f32>() / std::mem::size_of::<u16>())
            .ok_or("decode BF16 storage extent overflow")?;
        Ok(Self::CudaF32Bf16Storage(NativeCudaF32Bf16Storage {
            owner,
            bf16_elements,
        }))
    }
    fn address(&self) -> u64 {
        match self {
            Self::GpuBuf(x) => *x.device_ptr(),
            Self::CudaSlice(x) => *x.device_ptr(),
            Self::CudaF32Bf16Storage(x) => *x.owner.device_ptr(),
        }
    }
    fn elements(&self) -> usize {
        match self {
            Self::GpuBuf(x) => x.len(),
            Self::CudaSlice(x) => x.len(),
            Self::CudaF32Bf16Storage(x) => x.bf16_elements,
        }
    }
}
struct NativeBf16View {
    _allocation: NativeBf16Allocation,
    address: u64,
    elements: usize,
}
impl NativeBf16View {
    fn checked(
        allocation: NativeBf16Allocation,
        first: usize,
        elements: usize,
        stream: &NativeStream,
    ) -> Result<Self, String> {
        let end = first
            .checked_add(elements)
            .ok_or("owned BF16 view extent overflow")?;
        if elements == 0 || end > allocation.elements() {
            return Err("owned BF16 view outside actual typed source".into());
        }
        let offset = first
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or("owned BF16 view offset overflow")?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or("owned BF16 view bytes overflow")?;
        let address = allocation
            .address()
            .checked_add(u64::try_from(offset).map_err(|_| "owned view pointer offset")?)
            .ok_or("owned BF16 pointer overflow")?;
        native_pointer_admission(stream, address, bytes)?;
        Ok(Self {
            _allocation: allocation,
            address,
            elements,
        })
    }
}
enum NativeBuffer {
    Bf16(GpuBuf<u16>),
    ScalarF32(GpuBuf<f32>),
    Bf16View(NativeBf16View),
}
impl NativeBuffer {
    fn address(&self) -> u64 {
        match self {
            Self::Bf16(x) => *x.device_ptr(),
            Self::ScalarF32(x) => *x.device_ptr(),
            Self::Bf16View(x) => x.address,
        }
    }
    fn elements(&self) -> usize {
        match self {
            Self::Bf16(x) => x.len(),
            Self::ScalarF32(x) => x.len(),
            Self::Bf16View(x) => x.elements,
        }
    }
    fn dtype(&self) -> Dtype {
        match self {
            Self::Bf16(_) => Dtype::Bfloat16,
            Self::ScalarF32(_) => Dtype::Float32,
            Self::Bf16View(_) => Dtype::Bfloat16,
        }
    }
}
struct NativeRawOwner {
    device: Arc<CudaDevice>,
    schema: Schema,
    buffers: Vec<NativeBuffer>,
    admissions: Vec<NativePointerAdmission>,
    stream: Arc<NativeStream>,
}
impl Drop for NativeRawOwner {
    fn drop(&mut self) {
        if let Err(e) = self.stream.drain() {
            crate::vram_monitor::fatal_cuda_context_error("native raw owner final completion", &e);
        }
    }
}
impl NativeRawOwner {
    fn from_owned_buffers(
        stream: Arc<NativeStream>,
        schema: Schema,
        buffers: Vec<NativeBuffer>,
        reserve: usize,
    ) -> Result<Self, String> {
        schema.validate()?;
        stream.validate()?;
        if buffers.len() != schema.planes.len() {
            return Err("owned raw source plane count".into());
        }
        let mut admissions = Vec::with_capacity(buffers.len());
        let mut spans = Vec::new();
        for (plane, buffer) in schema.planes.iter().zip(&buffers) {
            if plane.dtype != buffer.dtype() || plane.elements != buffer.elements() {
                return Err("owned raw source typed schema extent".into());
            }
            let first = buffer.address();
            let end = first
                .checked_add(
                    u64::try_from(plane.storage_bytes).map_err(|_| "owned raw span extent")?,
                )
                .ok_or("owned raw span overflow")?;
            if spans
                .iter()
                .any(|&(previous_first, previous_end)| first < previous_end && previous_first < end)
            {
                return Err("owned raw source planes overlap".into());
            }
            spans.push((first, end));
            admissions.push(native_pointer_admission(
                &stream,
                first,
                plane.storage_bytes,
            )?);
        }
        if memory()?[0] < reserve {
            return Err("retained raw source observed safety floor".into());
        }
        // Source allocations already exist; wrapper introduces no GPU allocation or copy.
        Ok(Self {
            device: stream.device.clone(),
            schema,
            buffers,
            admissions,
            stream,
        })
    }
    fn allocate(stream: Arc<NativeStream>, schema: Schema, reserve: usize) -> Result<Self, String> {
        schema.validate()?;
        stream.validate()?;
        if memory()?[0]
            < schema
                .requested_bytes
                .checked_add(reserve)
                .ok_or("native rawreserve extent")?
        {
            return Err("actual measured native rawadmission".into());
        }
        let mut buffers = Vec::new();
        let mut admissions = Vec::new();
        let mut sum = 0usize;
        for p in &schema.planes {
            let b = match p.dtype {
                Dtype::Bfloat16 => NativeBuffer::Bf16(native_zeroed::<u16>(&stream, p.elements)?),
                Dtype::Float32 => {
                    NativeBuffer::ScalarF32(native_zeroed::<f32>(&stream, p.elements)?)
                }
            };
            if b.elements() != p.elements || b.dtype() != p.dtype {
                return Err("native allocated rawtype/extent".into());
            }
            admissions.push(native_pointer_admission(
                &stream,
                b.address(),
                p.storage_bytes,
            )?);
            sum = sum
                .checked_add(
                    b.elements()
                        .checked_mul(b.dtype().bytes())
                        .ok_or("native actual rawbytes")?,
                )
                .ok_or("native actual totals")?;
            buffers.push(b);
        }
        if sum != schema.requested_bytes || memory()?[0] < reserve {
            return Err("native rawbyte accounting/reserve".into());
        }
        Ok(Self {
            device: stream.device.clone(),
            schema,
            buffers,
            admissions,
            stream,
        })
    }
    fn inventory(&self) -> Vec<(Plane, u64)> {
        self.schema
            .planes
            .iter()
            .cloned()
            .zip(self.buffers.iter().map(NativeBuffer::address))
            .collect()
    }
    fn validate_phase_invalidation(&self, phase: &StatePhase<'_>) -> Result<(), String> {
        self.schema.validate()?;
        phase.stream.validate()?;
        if !std::ptr::eq(self, &phase.owner.pending_raw)
            || !Arc::ptr_eq(&self.device, &phase.owner.device)
            || phase.stream.handle != phase.owner.stream.handle
        {
            return Err(
                "pending raw invalidation requires actual owning state phase/context/stream".into(),
            );
        }
        if self.buffers.len() != self.schema.planes.len() {
            return Err("pending raw phase plane count".into());
        }
        for (plane, buffer) in self.schema.planes.iter().zip(&self.buffers) {
            if buffer.dtype() != plane.dtype || buffer.elements() != plane.elements {
                return Err("pending raw phase typed extent".into());
            }
            native_pointer_admission(phase.stream, buffer.address(), plane.storage_bytes)?;
        }
        Ok(())
    }
    fn invalidate_in_phase(&self, phase: &StatePhase<'_>) -> Result<(), String> {
        self.validate_phase_invalidation(phase)?;
        for (plane, buffer) in self.schema.planes.iter().zip(&self.buffers) {
            match plane.dtype {
                Dtype::Bfloat16 => check(
                    unsafe {
                        sys::lib().cuMemsetD16Async(
                            buffer.address(),
                            0x7fc1,
                            plane.elements,
                            phase.stream.handle,
                        )
                    },
                    "ordered pending raw BF16 invalidation",
                )?,
                Dtype::Float32 => check(
                    unsafe {
                        sys::lib().cuMemsetD32Async(
                            buffer.address(),
                            KDA_PENDING_INVALID_F32_BITS,
                            plane.elements,
                            phase.stream.handle,
                        )
                    },
                    "ordered pending raw F32 invalidation",
                )?,
            }
        }
        Ok(())
    }
    unsafe fn invalidate(&self, stream: sys::CUstream) -> Result<(), String> {
        if stream != self.stream.handle {
            return Err("native rawinvalidation wrong stream".into());
        }
        for (p, b) in self.schema.planes.iter().zip(&self.buffers) {
            match p.dtype {
                Dtype::Bfloat16 => check(
                    sys::lib().cuMemsetD16Async(b.address(), 0x7fc1, p.elements, stream),
                    "native rawBF16 invalidation",
                )?,
                Dtype::Float32 => check(
                    sys::lib().cuMemsetD32Async(
                        b.address(),
                        KDA_PENDING_INVALID_F32_BITS,
                        p.elements,
                        stream,
                    ),
                    "native rawF32 invalidation",
                )?,
            }
        }
        Ok(())
    }
    fn view(&self) -> Result<NativePreparedView<'_>, String> {
        self.schema.validate()?;
        self.stream.validate()?;
        Ok(NativePreparedView { owner: self })
    }
}
struct NativePreparedView<'a> {
    owner: &'a NativeRawOwner,
}
trait PreparedInput {
    fn schema(&self) -> &Schema;
    fn device(&self) -> &Arc<CudaDevice>;
    fn stream(&self) -> sys::CUstream;
    fn addresses(&self) -> Vec<u64>;
}
impl PreparedInput for RawOwner {
    fn schema(&self) -> &Schema {
        &self.schema
    }
    fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
    fn stream(&self) -> sys::CUstream {
        *self.device.cu_stream()
    }
    fn addresses(&self) -> Vec<u64> {
        self.buffers.iter().map(TypedBuffer::address).collect()
    }
}
impl PreparedInput for NativeRawOwner {
    fn schema(&self) -> &Schema {
        &self.schema
    }
    fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
    fn stream(&self) -> sys::CUstream {
        self.stream.handle
    }
    fn addresses(&self) -> Vec<u64> {
        self.buffers.iter().map(NativeBuffer::address).collect()
    }
}
impl PreparedInput for NativePreparedView<'_> {
    fn schema(&self) -> &Schema {
        &self.owner.schema
    }
    fn device(&self) -> &Arc<CudaDevice> {
        &self.owner.device
    }
    fn stream(&self) -> sys::CUstream {
        self.owner.stream.handle
    }
    fn addresses(&self) -> Vec<u64> {
        self.owner
            .buffers
            .iter()
            .map(NativeBuffer::address)
            .collect()
    }
}

// Allocation initialization belongs to the same admitted stream as every later consumer.
fn native_zeroed<T: Copy>(stream: &NativeStream, count: usize) -> Result<GpuBuf<T>, String> {
    stream.validate()?;
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or("native owned initialization byte overflow")?;
    let allocation_before = memory()?;
    let buffer = GpuBuf::<T>::alloc_uninit(count)?;
    let allocation_after = memory()?;
    let actual_admission = if bytes == 0 {
        None
    } else {
        Some(native_pointer_admission(stream, pointer(&buffer), bytes)?)
    };
    eprintln!(
        "[KDA-ALLOC-DATA] {}",
        serde_json::json!({"requested_bytes":bytes,"free_before":allocation_before,"free_after":allocation_after,"admission":actual_admission,"stream":stream.handle as usize,"synchronous_allocation_measurement":true})
    );
    if bytes != 0 {
        check(
            unsafe { sys::lib().cuMemsetD8Async(pointer(&buffer), 0, bytes, stream.handle) },
            "native owned zero before upload/consumer",
        )?;
    }
    Ok(buffer)
}
