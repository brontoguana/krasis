pub(crate) struct RetainedPrefillKdaSource {
    inner: Arc<NativeRawOwner>,
}
impl RetainedPrefillKdaSource {
    pub(crate) fn from_model_storage(
        owner: Arc<crate::gpu_prefill::PrefillComputeStream>,
        raw: [Arc<GpuBuf<u16>>; 5],
        rows: usize,
        heads: usize,
        dim: usize,
        reserve: usize,
    ) -> Result<Self, String> {
        let stream = NativeStream::from_prefill_owner(owner)?;
        let schema = Schema::checked(RawContract::KdaBf16VectorGate, rows, heads, dim, dim)?;
        let mut buffers = Vec::with_capacity(schema.planes.len());
        for (source, plane) in raw.into_iter().zip(&schema.planes) {
            buffers.push(NativeBuffer::Bf16View(NativeBf16View::checked(
                NativeBf16Allocation::GpuBuf(source),
                0,
                plane.elements,
                &stream,
            )?));
        }
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
