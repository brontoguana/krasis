// Explicit quiescent source replacement only; no serving selector or growth policy.
impl KdaRawProducer {
    fn validate_replacement(
        &self,
        raw: &NativeRawOwner,
        decode: bool,
    ) -> Result<(usize, usize, u64), String> {
        raw.schema.validate()?;
        raw.stream.validate()?;
        let rows = *raw
            .schema
            .planes
            .first()
            .and_then(|p| p.shape.first())
            .ok_or("replacement raw row extent")?;
        let schema = Schema::checked(
            RawContract::KdaBf16VectorGate,
            rows,
            self.heads,
            self.dim,
            self.dim,
        )?;
        if rows < self.capacity
            || (decode && rows != 1)
            || raw.schema != schema
            || !Arc::ptr_eq(&raw.device, &self.raw.device)
            || raw.stream.handle != self.raw.stream.handle
            || raw.buffers.len() != schema.planes.len()
            || raw.admissions.len() != schema.planes.len()
        {
            return Err("replacement raw geometry/context/stream/extent".into());
        }
        for ((plane, buffer), admitted) in
            schema.planes.iter().zip(&raw.buffers).zip(&raw.admissions)
        {
            if buffer.dtype() != plane.dtype
                || buffer.elements() != plane.elements
                || admitted.ptr != buffer.address()
                || admitted.requested_bytes != plane.storage_bytes
            {
                return Err("replacement raw typed span".into());
            }
            native_pointer_admission(&raw.stream, buffer.address(), plane.storage_bytes)?;
        }
        let requested = self
            .requested_bytes
            .checked_sub(self.raw.schema.requested_bytes)
            .and_then(|x| x.checked_add(schema.requested_bytes))
            .ok_or("replacement source accounting overflow")?;
        let epoch = self
            .source_epoch
            .checked_add(1)
            .ok_or("replacement source epoch overflow")?;
        Ok((rows, requested, epoch))
    }
}
impl PreparedKdaModelFamily {
    /// The exclusive borrow retires safe Rust graph/proof/domain borrows before
    /// replacement. The caller must also retire any external model graph epoch.
    /// Both leases are admitted before waiting or changing either producer.
    pub(crate) fn replace_raw_sources_quiescent(
        &mut self,
        prefill: RetainedPrefillKdaSource,
        decode: RetainedDecodeKdaSource,
        expected_epochs: [u64; 2],
    ) -> Result<[u64; 2], String> {
        self.borrowed()?;
        if expected_epochs
            != [
                self.prefill_producer.source_epoch,
                self.decode_producer.source_epoch,
            ]
        {
            return Err("stale source epoch replacement".into());
        }
        let p = self
            .prefill_producer
            .validate_replacement(&prefill.inner, false)?;
        let d = self
            .decode_producer
            .validate_replacement(&decode.inner, true)?;
        let mut slot = self
            .prefill
            .state
            .timeline
            .try_lock()
            .map_err(|_| "replacement family concurrently leased/poisoned")?;
        let t = slot.as_mut().ok_or("replacement family timeline missing")?;
        if t.poisoned {
            return Err("replacement family poisoned".into());
        }
        // Source replacement is a once-per-lifecycle operation, never per token.
        // Prior work must finish before stale source owners can be released.
        t.poisoned = true;
        if t.recorded {
            check(
                unsafe { sys::lib().cuEventSynchronize(t.event) },
                "replacement prior family completion",
            )?;
        }
        for stream in &t.streams {
            stream.drain()?;
        }
        self.prefill_producer.raw = prefill.inner;
        self.prefill_producer.input_capacity = p.0;
        self.prefill_producer.requested_bytes = p.1;
        self.prefill_producer.source_epoch = p.2;
        self.decode_producer.raw = decode.inner;
        self.decode_producer.input_capacity = d.0;
        self.decode_producer.requested_bytes = d.1;
        self.decode_producer.source_epoch = d.2;
        t.poisoned = false;
        Ok([p.2, d.2])
    }
}
