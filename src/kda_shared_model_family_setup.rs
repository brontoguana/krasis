// Cold explicit shared-arena constructor; no implicit allocation fallback.
impl PreparedKdaModelFamily {
    pub(crate) unsafe fn from_model_storage_shared_arenas(
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
        arena_family: &PreparedKdaModelFamily,
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
        let seed = arena_family.borrowed()?;
        let pref_seed = seed.prefill;
        let dec_seed = seed.decode;
        if pref_seed.heads != heads
            || pref_seed.dim != dim
            || pref_seed.max_new_rows != measured_capacity
            || dec_seed.max_new_rows != 1
            || !Arc::ptr_eq(&pref_seed.device, &prefill.inner.device)
            || pref_seed.stream.handle != prefill.inner.stream.handle
            || dec_seed.stream.handle != decode.inner.stream.handle
            || seed.prefill_producer.lower.to_bits() != lower.to_bits()
            || seed.prefill_producer.scale.to_bits() != scale.to_bits()
            || seed.prefill_producer.epsilon.to_bits() != epsilon.to_bits()
            || seed.prefill_producer.kernels.ptx_sha256 != bank.kernels.ptx_sha256
            || seed.prefill_producer.kernels.domain.cu_function()
                != bank.kernels.domain.cu_function()
            || seed.prefill_producer.kernels.prepare.cu_function()
                != bank.kernels.prepare.cu_function()
        {
            return Err("shared family arena/source/preparation contract mismatch".into());
        }
        let seed_slot = pref_seed
            .state
            .timeline
            .try_lock()
            .map_err(|_| "shared seed family concurrently leased/poisoned")?;
        let timeline = seed_slot.as_ref().ok_or("shared seed timeline missing")?;
        if timeline.poisoned {
            return Err("shared seed family poisoned".into());
        }
        for owner in [pref_seed, dec_seed] {
            for (ptx, names) in [(&bank.transport, TRANSPORT), (&bank.math, MATH)] {
                let module = format!(
                    "prepared-kda-vector-fixed16-{:x}",
                    Sha256::digest(ptx.as_bytes())
                );
                for &name in names {
                    let actual = owner
                        .device
                        .get_func(&module, name)
                        .ok_or("shared family bank module absent")?;
                    let retained = owner
                        .arena
                        .functions
                        .get(name)
                        .ok_or("shared family arena module incomplete")?;
                    if actual.cu_function() != retained.cu_function() {
                        return Err("shared family arena actual kernel mismatch".into());
                    }
                }
            }
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
        let po = PreparedOwner::from_shared(pref_seed, reserve)?;
        po.initialize_timeline(&[ps.clone(), ds.clone()])?;
        let do_ = PreparedOwner::bind_ordered(dec_seed, &po)?;
        drop(seed_slot);
        let result = Self {
            prefill: po,
            decode: do_,
            prefill_producer: pp,
            decode_producer: dp,
        };
        result.borrowed()?;
        Ok(result)
    }
}

impl PreparedKdaModelFamily {
    pub(crate) fn cold_shared_arena_key(
        prefill: &RetainedPrefillKdaSource,
        decode: &RetainedDecodeKdaSource,
        heads: usize,
        dim: usize,
        capacity: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        bank: &KdaModelKernelBank,
    ) -> Result<Vec<u8>, String> {
        prefill.inner.stream.validate()?;
        decode.inner.stream.validate()?;
        let rows = *prefill
            .inner
            .schema
            .planes
            .first()
            .and_then(|p| p.shape.first())
            .ok_or("cold group source extent")?;
        if capacity == 0
            || capacity > rows
            || !Arc::ptr_eq(&prefill.inner.device, &decode.inner.device)
            || !Arc::ptr_eq(&prefill.inner.device, &bank.kernels.device)
            || prefill.inner.schema
                != Schema::checked(RawContract::KdaBf16VectorGate, rows, heads, dim, dim)?
            || decode.inner.schema
                != Schema::checked(RawContract::KdaBf16VectorGate, 1, heads, dim, dim)?
        {
            return Err("cold shared group source geometry/context/capacity".into());
        }
        let layout = PreparedLayout::checked(heads, dim, capacity)?;
        serde_json::to_vec(&serde_json::json!({"context":*prefill.inner.device.cu_primary_ctx() as usize,"prefill_stream":prefill.inner.stream.handle as usize,"decode_stream":decode.inner.stream.handle as usize,"heads":heads,"dim":dim,"capacity":capacity,"input_rows":rows,"padded":layout.padded,"lower_bits":lower.to_bits(),"scale_bits":scale.to_bits(),"epsilon_bits":epsilon.to_bits(),"producer_ptx":bank.kernels.ptx_sha256,"transport_ptx":format!("{:x}",Sha256::digest(bank.transport.as_bytes())),"math_ptx":format!("{:x}",Sha256::digest(bank.math.as_bytes()))})).map_err(|e|e.to_string())
    }
}
