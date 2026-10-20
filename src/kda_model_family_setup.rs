// Cold model-facing construction only. No serving dispatch is selected.
// Create one bank per actual device/model setup and reuse it across families.
pub(crate) struct KdaModelKernelBank {
    kernels: Arc<KdaRawKernels>,
    transport: String,
    math: String,
}
impl KdaModelKernelBank {
    pub(crate) fn load(
        device: Arc<CudaDevice>,
        transport: &str,
        math: &str,
        producer: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            kernels: KdaRawKernels::load(device, producer)?,
            transport: transport.into(),
            math: math.into(),
        })
    }
}
pub(crate) struct PreparedKdaModelFamily {
    prefill: PreparedOwner,
    decode: PreparedOwner,
    prefill_producer: KdaRawProducer,
    decode_producer: KdaRawProducer,
}
struct BorrowedKdaModelFamily<'a> {
    prefill: &'a PreparedOwner,
    decode: &'a PreparedOwner,
    prefill_producer: &'a KdaRawProducer,
    decode_producer: &'a KdaRawProducer,
}
impl<'a> BorrowedKdaModelFamily<'a> {
    fn checked(
        prefill: &'a PreparedOwner,
        decode: &'a PreparedOwner,
        prefill_producer: &'a KdaRawProducer,
        decode_producer: &'a KdaRawProducer,
    ) -> Result<Self, String> {
        if !Arc::ptr_eq(&prefill.state, &decode.state)
            || !Arc::ptr_eq(&prefill.device, &decode.device)
            || prefill.heads != decode.heads
            || prefill.dim != decode.dim
            || prefill.arithmetic_sha256 != decode.arithmetic_sha256
            || prefill_producer.arithmetic_sha256 != decode_producer.arithmetic_sha256
            || decode_producer.capacity != 1
            || decode.max_new_rows != 1
            || prefill.max_new_rows != prefill_producer.capacity
        {
            return Err("model KDA family state/geometry/arithmetic/capacity mismatch".into());
        }
        for (owner, producer) in [(prefill, prefill_producer), (decode, decode_producer)] {
            model_family_identity(&[ModelFamilyBinding {
                layer: 0,
                owner,
                producer,
            }])?;
        }
        let mut a = crate::session_cache::SequenceStateRegistry::default();
        let mut b = crate::session_cache::SequenceStateRegistry::default();
        prefill.register(&mut a, 0)?;
        decode.register(&mut b, 0)?;
        if a.allocations() != b.allocations() {
            return Err("model KDA prefill/decode persistent inventory mismatch".into());
        }
        Ok(Self {
            prefill,
            decode,
            prefill_producer,
            decode_producer,
        })
    }
    fn prefill_binding(&self, layer: usize) -> ModelFamilyBinding<'a> {
        ModelFamilyBinding {
            layer,
            owner: self.prefill,
            producer: self.prefill_producer,
        }
    }
    fn decode_binding(&self, layer: usize) -> ModelFamilyBinding<'a> {
        ModelFamilyBinding {
            layer,
            owner: self.decode,
            producer: self.decode_producer,
        }
    }
}
impl PreparedKdaModelFamily {
    /// Setup only. The caller retains the original loaded parameter owners and
    /// forbids mutation/unload until this copy returns. Source leases own their
    /// allocations/streams. Request execution needs a separate complete-model
    /// coordinator; this constructor does not admit a graph/weight epoch.
    pub(crate) unsafe fn from_model_storage(
        prefill: RetainedPrefillKdaSource,
        decode: RetainedDecodeKdaSource,
        heads: usize,
        dim: usize,
        measured_capacity: usize,
        reserve: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        bank: &KdaModelKernelBank,
        a_log_ptr: u64,
        a_log_elements: usize,
        bias_ptr: u64,
        bias_elements: usize,
    ) -> Result<Self, String> {
        let hd = heads
            .checked_mul(dim)
            .ok_or("model KDA loaded geometry overflow")?;
        if measured_capacity == 0 || a_log_elements != heads || bias_elements != hd {
            return Err("model KDA loaded parameter/capacity extent".into());
        }
        let rows = *prefill
            .inner
            .schema
            .planes
            .first()
            .and_then(|p| p.shape.first())
            .ok_or("model prefill row extent")?;
        if measured_capacity > rows {
            return Err("model KDA measured capacity exceeds retained input".into());
        }
        let prefill_schema =
            Schema::checked(RawContract::KdaBf16VectorGate, rows, heads, dim, dim)?;
        let decode_schema = Schema::checked(RawContract::KdaBf16VectorGate, 1, heads, dim, dim)?;
        if prefill.inner.schema != prefill_schema
            || decode.inner.schema != decode_schema
            || !Arc::ptr_eq(&prefill.inner.device, &decode.inner.device)
            || !Arc::ptr_eq(&prefill.inner.device, &bank.kernels.device)
        {
            return Err("model KDA retained source geometry/context mismatch".into());
        }
        let ps = prefill.inner.stream.clone();
        let ds = decode.inner.stream.clone();
        ps.validate()?;
        ds.validate()?;
        let a_bytes = a_log_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("model A-log bytes")?;
        let b_bytes = bias_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("model bias bytes")?;
        native_pointer_admission(&ps, a_log_ptr, a_bytes)?;
        native_pointer_admission(&ps, bias_ptr, b_bytes)?;
        // Setup has no live request phases; finish producers before cold reads.
        ps.drain()?;
        ds.drain()?;
        let mut a_log = vec![0u8; a_bytes];
        let mut bias = vec![0u8; b_bytes];
        check(
            unsafe { sys::lib().cuMemcpyDtoH_v2(a_log.as_mut_ptr().cast(), a_log_ptr, a_bytes) },
            "model loaded A-log cold copy",
        )?;
        check(
            unsafe { sys::lib().cuMemcpyDtoH_v2(bias.as_mut_ptr().cast(), bias_ptr, b_bytes) },
            "model loaded bias cold copy",
        )?;
        let kernels = bank.kernels.clone();
        let pp = KdaRawProducer::from_live_raw_owner(
            prefill.inner,
            heads,
            dim,
            measured_capacity,
            reserve,
            lower,
            scale,
            epsilon,
            kernels.clone(),
            &a_log,
            &bias,
        )?;
        let dp = KdaRawProducer::from_raw_owner(
            decode.inner,
            heads,
            dim,
            1,
            reserve,
            lower,
            scale,
            epsilon,
            kernels,
            &a_log,
            &bias,
        )?;
        let po = PreparedOwner::new_on_stream(
            ps.device.clone(),
            ps.clone(),
            heads,
            dim,
            measured_capacity,
            reserve,
            &bank.transport,
            &bank.math,
        )?;
        po.initialize_timeline(&[ps.clone(), ds.clone()])?;
        let decode_arena = PreparedOwner::new_on_stream(
            ds.device.clone(),
            ds.clone(),
            heads,
            dim,
            1,
            reserve,
            &bank.transport,
            &bank.math,
        )?;
        let do_ = PreparedOwner::bind_ordered(&decode_arena, &po)?;
        drop(decode_arena);
        let result = Self {
            prefill: po,
            decode: do_,
            prefill_producer: pp,
            decode_producer: dp,
        };
        result.borrowed()?;
        Ok(result)
    }
    fn borrowed(&self) -> Result<BorrowedKdaModelFamily<'_>, String> {
        BorrowedKdaModelFamily::checked(
            &self.prefill,
            &self.decode,
            &self.prefill_producer,
            &self.decode_producer,
        )
    }
    pub(crate) fn register(
        &self,
        registry: &mut crate::session_cache::SequenceStateRegistry,
        layer: usize,
    ) -> Result<(), String> {
        self.prefill.register(registry, layer)
    }
    pub(crate) fn identity(&self, layer: usize) -> Result<serde_json::Value, String> {
        let b = self.borrowed()?;
        Ok(
            serde_json::json!({"prefill":model_family_identity(&[b.prefill_binding(layer)])?,"decode":model_family_identity(&[b.decode_binding(layer)])?}),
        )
    }
}
