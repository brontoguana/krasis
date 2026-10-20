// Cold construction from actual loaded decode storage. Prefill has no source
// capability until scratch is prepared and its real owners are attached.
pub(crate) struct KdaColdLayer {
    pub(crate) layer: usize,
    pub(crate) decode: RetainedDecodeKdaSource,
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) lower: f32,
    pub(crate) scale: f32,
    pub(crate) epsilon: f32,
    pub(crate) a_log_ptr: u64,
    pub(crate) bias_ptr: u64,
}
impl KdaPrefillStorageLifetime {
    /// The loader retains actual parameter tensors throughout this cold call.
    /// Capacity describes GPU tiling, never prompt truncation or a memory budget.
    pub(crate) unsafe fn from_decode_storage(
        descriptors: Vec<KdaColdLayer>,
        prefill_stream: Arc<crate::gpu_prefill::PrefillComputeStream>,
        capacity: usize,
        reserve: usize,
        bank: &KdaModelKernelBank,
    ) -> Result<Self, String> {
        if descriptors.is_empty() || capacity == 0 {
            return Err("cold KDA empty model/capacity".into());
        }
        let ps = NativeStream::from_prefill_owner(prefill_stream)?;
        if !Arc::ptr_eq(&ps.device, &bank.kernels.device) {
            return Err("cold KDA kernel context mismatch".into());
        }
        let mut families: Vec<RetiredPrefillKdaFamily> = Vec::new();
        let mut layers = Vec::new();
        let mut arena_seeds = BTreeMap::new();
        for d in descriptors {
            if layers.last().is_some_and(|last| *last >= d.layer) {
                return Err("cold KDA duplicate/unordered layer inventory".into());
            }
            let ds = d.decode.inner.stream.clone();
            ds.validate()?;
            let hd = d
                .heads
                .checked_mul(d.dim)
                .ok_or("cold KDA loaded geometry")?;
            let expected =
                Schema::checked(RawContract::KdaBf16VectorGate, 1, d.heads, d.dim, d.dim)?;
            if d.decode.inner.schema != expected || !Arc::ptr_eq(&ds.device, &ps.device) {
                return Err("cold KDA actual decode geometry/context mismatch".into());
            }
            let a_bytes = d.heads.checked_mul(4).ok_or("cold A-log extent")?;
            let b_bytes = hd.checked_mul(4).ok_or("cold bias extent")?;
            native_pointer_admission(&ds, d.a_log_ptr, a_bytes)?;
            native_pointer_admission(&ds, d.bias_ptr, b_bytes)?;
            ps.drain()?;
            ds.drain()?;
            let mut a_log = vec![0u8; a_bytes];
            let mut bias = vec![0u8; b_bytes];
            check(
                sys::lib().cuMemcpyDtoH_v2(a_log.as_mut_ptr().cast(), d.a_log_ptr, a_bytes),
                "cold actual A-log read",
            )?;
            check(
                sys::lib().cuMemcpyDtoH_v2(bias.as_mut_ptr().cast(), d.bias_ptr, b_bytes),
                "cold actual bias read",
            )?;
            let dp = KdaRawProducer::from_raw_owner(
                d.decode.inner,
                d.heads,
                d.dim,
                1,
                reserve,
                d.lower,
                d.scale,
                d.epsilon,
                bank.kernels.clone(),
                &a_log,
                &bias,
            )?;
            let (po, do_) = if let Some(&seed_index) = arena_seeds.get(&(d.heads, d.dim)) {
                let seed: &RetiredPrefillKdaFamily = &families[seed_index];
                let po = PreparedOwner::from_shared(&seed.prefill, reserve)?;
                po.initialize_timeline(&[ps.clone(), ds.clone()])?;
                let do_ = PreparedOwner::bind_ordered(&seed.decode, &po)?;
                (po, do_)
            } else {
                let po = PreparedOwner::new_on_stream(
                    ps.device.clone(),
                    ps.clone(),
                    d.heads,
                    d.dim,
                    capacity,
                    reserve,
                    &bank.transport,
                    &bank.math,
                )?;
                po.initialize_timeline(&[ps.clone(), ds.clone()])?;
                // Workspace-only allocation: no discarded duplicate persistent state.
                let arena = PreparedArena::allocate_storage(
                    ds.device.clone(),
                    ds.clone(),
                    d.heads,
                    d.dim,
                    1,
                    reserve,
                    po.arena.functions.clone(),
                    po.arithmetic_sha256.clone(),
                )?;
                let do_ = PreparedOwner {
                    device: ds.device.clone(),
                    heads: d.heads,
                    dim: d.dim,
                    max_new_rows: 1,
                    padded: arena.padded,
                    arena,
                    state: po.state.clone(),
                    arithmetic_sha256: po.arithmetic_sha256.clone(),
                    persistent_bytes: po.persistent_bytes,
                    allocation_bytes: 0,
                    stream: ds.clone(),
                };
                do_.validate_lease(&do_.arena.lease()?)?;
                arena_seeds.insert((d.heads, d.dim), families.len());
                (po, do_)
            };
            let prepared_schema = Schema::checked(
                RawContract::KdaPreparedF32VectorGate,
                capacity,
                d.heads,
                d.dim,
                d.dim,
            )?;
            let owned_bytes = prepared_schema
                .requested_bytes
                .checked_add(a_bytes)
                .and_then(|v| v.checked_add(b_bytes))
                .ok_or("cold preparation accounting")?;
            if memory()?[0]
                < prepared_schema
                    .requested_bytes
                    .checked_add(reserve)
                    .ok_or("cold preparation admission overflow")?
            {
                return Err("cold preparation measured allocation admission".into());
            }
            let prepared = Arc::new(NativeRawOwner::allocate(
                ps.clone(),
                prepared_schema,
                reserve,
            )?);
            // Parameters are real immutable loaded values copied by dp; retain the
            // same typed allocation rather than uploading a second parameter bank.
            let parameter_admissions = vec![
                native_pointer_admission(&ps, pointer(&dp.a_log), a_bytes)?,
                native_pointer_admission(&ps, pointer(&dp.bias), b_bytes)?,
            ];
            let preparation = RetiredKdaPreparation {
                prepared,
                a_log: dp.a_log.clone(),
                bias: dp.bias.clone(),
                kernels: dp.kernels.clone(),
                heads: d.heads,
                dim: d.dim,
                capacity,
                source_epoch: 0,
                lower: d.lower,
                scale: d.scale,
                epsilon: d.epsilon,
                parameter_admissions,
                owned_bytes,
                arithmetic_sha256: dp.arithmetic_sha256.clone(),
            };
            families.push(RetiredPrefillKdaFamily {
                prefill: po,
                decode: do_,
                preparation,
                decode_producer: dp,
            });
            layers.push(d.layer);
        }
        let result = Self {
            layers,
            state: KdaPrefillStorageState::Retired(families),
        };
        result.decode_bindings()?;
        result.registry()?;
        result.storage_identity()?;
        Ok(result)
    }
}
