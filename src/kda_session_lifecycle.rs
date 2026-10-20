// Explicit retained diagnostic inventory; no serving selector or cache fallback.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct KdaSessionModelStamp {
    model_sha256: String,
    weights_sha256: String,
    checkpoint_sha256: String,
    config_sha256: String,
    source_sha256: String,
    binary_sha256: String,
    epoch: u64,
}
impl KdaSessionModelStamp {
    fn validate(&self) -> Result<(), String> {
        for id in [
            &self.model_sha256,
            &self.weights_sha256,
            &self.checkpoint_sha256,
            &self.config_sha256,
            &self.source_sha256,
            &self.binary_sha256,
        ] {
            if id.len() != 64
                || !id
                    .bytes()
                    .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
            {
                return Err("session model identity is not an explicit SHA256".into());
            }
        }
        Ok(())
    }
}
struct RetainedKdaSessionModel {
    families: Vec<PreparedKdaModelFamily>,
    layers: Vec<usize>,
    stamp: KdaSessionModelStamp,
    stable_compatibility: serde_json::Value,
    live_sequence_epoch: u64,
}
impl RetainedKdaSessionModel {
    fn registry(&self) -> Result<crate::session_cache::SequenceStateRegistry, String> {
        workspace_inventory_identity(&self.families, &self.layers)?;
        let mut registry = crate::session_cache::SequenceStateRegistry::default();
        for (f, &layer) in self.families.iter().zip(&self.layers) {
            f.register(&mut registry, layer)?;
        }
        Ok(registry)
    }
    fn physical_devices(&self) -> Result<Vec<String>, String> {
        let mut devices = std::collections::BTreeMap::new();
        for f in &self.families {
            let d = &f.prefill.device;
            if devices.contains_key(&d.ordinal()) {
                continue;
            }
            d.bind_to_thread().map_err(|e| e.to_string())?;
            let mut uuid = std::mem::MaybeUninit::<sys::CUuuid>::uninit();
            check(
                unsafe { sys::lib().cuDeviceGetUuid_v2(uuid.as_mut_ptr(), d.ordinal() as i32) },
                "session actual device UUID",
            )?;
            let uuid = unsafe { uuid.assume_init() };
            devices.insert(
                d.ordinal(),
                uuid.bytes
                    .iter()
                    .map(|x| format!("{:02x}", *x as u8))
                    .collect::<String>(),
            );
        }
        Ok(devices.into_values().collect())
    }
    fn compatibility(&self) -> Result<serde_json::Value, String> {
        self.stamp.validate()?;
        let registry = self.registry()?;
        Ok(
            serde_json::json!({"model":self.stamp,"physical_devices":self.physical_devices()?,"layout":registry.compatibility_layout_value(),
            "arithmetic":self.families.iter().map(|f|serde_json::json!({
                "recurrence":f.prefill.arithmetic_sha256,"prepare":f.prefill_producer.arithmetic_sha256
            })).collect::<Vec<_>>() }),
        )
    }
    fn new(
        families: Vec<PreparedKdaModelFamily>,
        layers: Vec<usize>,
        stamp: KdaSessionModelStamp,
    ) -> Result<Self, String> {
        let mut model = Self {
            families,
            layers,
            stamp,
            stable_compatibility: serde_json::Value::Null,
            live_sequence_epoch: 0,
        };
        model.stable_compatibility = model.compatibility()?;
        Ok(model)
    }
    fn validate_snapshot(
        &self,
        compatibility: &serde_json::Value,
        logical_tokens: usize,
        blobs: &[&crate::session_cache::SequenceStateBlob],
    ) -> Result<(), String> {
        if self.compatibility()? != self.stable_compatibility
            || compatibility != &self.stable_compatibility
        {
            return Err(
                "session incompatible retained model/weight/arithmetic/layout epoch".into(),
            );
        }
        let registry = self.registry()?;
        let allocations = registry
            .allocations()
            .iter()
            .filter(|a| a.used_bytes(logical_tokens) > 0)
            .collect::<Vec<_>>();
        if allocations.len() != blobs.len() {
            return Err("session incomplete full blob inventory".into());
        }
        validate_kda_snapshot_semantics(
            &allocations.into_iter().cloned().collect::<Vec<_>>(),
            logical_tokens,
            blobs,
        )?;
        Ok(())
    }
    fn snapshot_pinned(
        &self,
        current: &serde_json::Value,
        tokens: usize,
        stores: &mut [&mut crate::gpu_decode::GpuDecodeStore],
    ) -> Result<
        (
            Vec<crate::session_cache::SequenceStateBlob>,
            Vec<(usize, f64)>,
        ),
        String,
    > {
        let allocations = self.registry()?.allocations().to_vec();
        let mut retained = Vec::new();
        for f in &self.families {
            retained.extend(f.prefill.external_dma_regions()?);
        }

        self.session_transfer(current, stores, |stores| {
            let mut result = Vec::new();
            let mut readings = Vec::new();
            for store in stores {
                let (ordinal, context) = store.session_dma_device_identity_rust()?;
                let owners = retained
                    .iter()
                    .filter(|x| {
                        crate::sequence_external_owner::CudaRegion::context(*x) as usize == context
                    })
                    .map(|x| x as &dyn crate::sequence_external_owner::CudaRegion)
                    .collect::<Vec<_>>();
                let local = allocations
                    .iter()
                    .filter(|a| a.device_ordinal == ordinal)
                    .cloned()
                    .collect::<Vec<_>>();
                let (blobs, ms) =
                    store.snapshot_external_sequence_state_rust(&local, tokens, &owners)?;
                result.extend(blobs);
                readings.push((ordinal, ms));
            }
            result.sort_by(|a, b| {
                (a.device_ordinal, a.allocation_name.as_str())
                    .cmp(&(b.device_ordinal, b.allocation_name.as_str()))
            });
            self.validate_snapshot(
                &self.stable_compatibility,
                tokens,
                &result.iter().collect::<Vec<_>>(),
            )?;
            Ok((result, readings))
        })
    }
    fn restore_pinned(
        &self,
        current: &serde_json::Value,
        compatibility: &serde_json::Value,
        tokens: usize,
        blobs: &[&crate::session_cache::SequenceStateBlob],
        stores: &mut [&mut crate::gpu_decode::GpuDecodeStore],
    ) -> Result<Vec<(usize, f64)>, String> {
        // All model/cursor/blob metadata is rejected before any device transfer.
        self.validate_snapshot(compatibility, tokens, blobs)?;
        let allocations = self.registry()?.allocations().to_vec();
        let mut retained = Vec::new();
        for f in &self.families {
            retained.extend(f.prefill.external_dma_regions()?);
        }

        self.session_transfer(current, stores, |stores| {
            let mut readings = Vec::new();
            for store in stores {
                let (ordinal, context) = store.session_dma_device_identity_rust()?;
                let owners = retained
                    .iter()
                    .filter(|x| {
                        crate::sequence_external_owner::CudaRegion::context(*x) as usize == context
                    })
                    .map(|x| x as &dyn crate::sequence_external_owner::CudaRegion)
                    .collect::<Vec<_>>();
                let local = allocations
                    .iter()
                    .zip(blobs)
                    .filter(|(a, _)| a.device_ordinal == ordinal)
                    .collect::<Vec<_>>();
                let a = local.iter().map(|(a, _)| (*a).clone()).collect::<Vec<_>>();
                let b = local.iter().map(|(_, b)| **b).collect::<Vec<_>>();
                readings.push((
                    ordinal,
                    store.restore_external_sequence_state_rust(&a, tokens, &b, &owners)?,
                ));
            }
            Ok(readings)
        })
    }
    // Session lifecycle only. Every family/context/arena is admitted before GPU work.
    // The supplied stores retain existing bounded pinned DMA, not a replacement copy loop.
    fn session_transfer<T>(
        &self,
        current_inventory: &serde_json::Value,
        stores: &mut [&mut crate::gpu_decode::GpuDecodeStore],
        work: impl FnOnce(&mut [&mut crate::gpu_decode::GpuDecodeStore]) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.compatibility()? != self.stable_compatibility
            || &workspace_inventory_identity(&self.families, &self.layers)? != current_inventory
        {
            return Err("session stale model or current workspace proof".into());
        }
        let mut states = Vec::new();
        let mut arenas: Vec<Arc<PreparedArena>> = Vec::new();
        let mut devices: Vec<Arc<CudaDevice>> = Vec::new();
        for f in &self.families {
            if states.iter().any(|s| Arc::ptr_eq(s, &f.prefill.state)) {
                return Err("session duplicate persistent family".into());
            }
            states.push(f.prefill.state.clone());
            for owner in [&f.prefill, &f.decode] {
                owner.stream.validate()?;
                if !arenas.iter().any(|a| Arc::ptr_eq(a, &owner.arena)) {
                    arenas.push(owner.arena.clone());
                }
            }
            let d = &f.prefill.device;
            if !devices
                .iter()
                .any(|x| *x.cu_primary_ctx() == *d.cu_primary_ctx())
            {
                devices.push(d.clone());
            }
        }
        if stores.len() != devices.len() {
            return Err("session incomplete transfer device inventory".into());
        }
        for d in &devices {
            let mut matches = 0;
            for store in stores.iter() {
                if store
                    .session_dma_matches_device_rust(d.ordinal(), *d.cu_primary_ctx() as usize)?
                {
                    matches += 1;
                }
            }
            if matches != 1 {
                return Err("session transfer device missing/duplicate/foreign".into());
            }
        }
        let mut slots = states
            .iter()
            .map(|s| {
                s.timeline
                    .try_lock()
                    .map_err(|_| "session family concurrently leased/poisoned".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        for slot in &slots {
            let t = slot.as_ref().ok_or("session timeline missing")?;
            if t.poisoned {
                return Err("session family poisoned".into());
            }
            t.phases.checked_add(1).ok_or("session timeline overflow")?;
            for s in &t.streams {
                s.validate()?;
            }
        }
        let _leases = arenas
            .iter()
            .map(|a| a.lease())
            .collect::<Result<Vec<_>, _>>()?;
        for slot in &mut slots {
            slot.as_mut().ok_or("session timeline vanished")?.poisoned = true;
        }
        let result = (|| {
            for slot in &slots {
                let t = slot.as_ref().ok_or("session timeline vanished")?;
                t.device.bind_to_thread().map_err(|e| e.to_string())?;
                if t.recorded {
                    check(
                        unsafe { sys::lib().cuEventSynchronize(t.event) },
                        "session preceding completion",
                    )?;
                }
                for stream in &t.streams {
                    stream.drain()?;
                }
            }
            work(stores)
        })();
        // Complete any submitted pinned copy even when a later transfer errors.
        // No guard/state owner can retire while a failed write remains in flight.
        let mut completion_error = None;
        for d in &devices {
            let completed = d.bind_to_thread().map_err(|e| e.to_string()).and_then(|_| {
                check(
                    unsafe { sys::lib().cuCtxSynchronize() },
                    "session submitted transfer completion",
                )
            });
            if let Err(e) = completed {
                if completion_error.is_none() {
                    completion_error = Some(e);
                }
            }
        }
        if let Some(e) = completion_error {
            return Err(e);
        }
        let value = result?;
        for slot in &mut slots {
            let t = slot.as_mut().ok_or("session timeline vanished")?;
            // All admitted streams and DMA have completed; next request has no pending event.
            t.recorded = false;
            t.phases += 1;
            t.poisoned = false;
        }
        Ok(value)
    }
}
