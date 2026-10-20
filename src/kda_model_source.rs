// Cold source lease only. It neither selects nor runs a recurrence.
pub(crate) struct RetainedDecodeKdaSource {
    inner: Arc<NativeRawOwner>,
}
impl RetainedDecodeKdaSource {
    pub(crate) fn from_model_storage(
        device: Arc<CudaDevice>,
        raw: [Arc<CudaSlice<f32>>; 4],
        beta: Arc<CudaSlice<u16>>,
        heads: usize,
        dim: usize,
        reserve: usize,
    ) -> Result<Self, String> {
        Self::from_storage_on_stream(
            NativeStream::borrowed_device(device)?,
            raw,
            beta,
            heads,
            dim,
            reserve,
        )
    }
    fn from_storage_on_stream(
        stream: Arc<NativeStream>,
        raw: [Arc<CudaSlice<f32>>; 4],
        beta: Arc<CudaSlice<u16>>,
        heads: usize,
        dim: usize,
        reserve: usize,
    ) -> Result<Self, String> {
        let schema = Schema::checked(RawContract::KdaBf16VectorGate, 1, heads, dim, dim)?;
        let mut buffers = Vec::with_capacity(schema.planes.len());
        for (owner, plane) in raw.into_iter().zip(&schema.planes) {
            buffers.push(NativeBuffer::Bf16View(NativeBf16View::checked(
                NativeBf16Allocation::cuda_f32_bf16_storage(owner)?,
                0,
                plane.elements,
                &stream,
            )?));
        }
        let plane = schema.planes.last().ok_or("decode raw beta schema")?;
        buffers.push(NativeBuffer::Bf16View(NativeBf16View::checked(
            NativeBf16Allocation::CudaSlice(beta),
            0,
            plane.elements,
            &stream,
        )?));
        Ok(Self {
            inner: Arc::new(NativeRawOwner::from_owned_buffers(
                stream, schema, buffers, reserve,
            )?),
        })
    }
    pub(crate) fn addresses(&self) -> Vec<u64> {
        self.inner.addresses()
    }
}
