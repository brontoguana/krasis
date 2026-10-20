// Actual serving operations on the selected prepared-state family. The Store
// serializes model requests; all raw input owners and arenas remain retained.
impl KdaPrefillExecutionPhase<'_, '_> {
    pub(crate) fn enqueue_recurrence(
        &self,
        layer: usize,
        absolute: usize,
        rows: usize,
        raw: [u64; 5],
        output: u64,
        invocation: Option<&crate::matched_prefill_capture::Invocation>,
    ) -> Result<(), String> {
        let binding = self
            .bindings
            .iter()
            .find(|b| b.layer == layer)
            .ok_or("selected prefill missing layer")?;
        let owner = binding.owner;
        let producer = binding.producer;
        self.validate_layer(layer, rows, self.stream.handle, owner.heads, owner.dim)?;
        if output == 0 || producer.raw.addresses() != raw {
            return Err("selected prefill raw/output binding changed".into());
        }
        if absolute == 0 {
            owner.reset_in_phase(&StatePhase {
                owner,
                stream: self.stream,
            })?;
        }
        let width = owner
            .heads
            .checked_mul(owner.dim)
            .ok_or("selected prefill width")?;
        let digest_capture=crate::matched_prefill_capture::target(invocation,"selected_prepared_window_sha256",layer,absolute.checked_add(rows).ok_or("selected digest height")?,None)?;
        let digest_limit=if digest_capture.is_some(){Some(crate::matched_probe::capture_maximum_bytes("selected_prepared_window_sha256",layer,Some(absolute+rows))?)}else{None};
        let lease = model_arena_lease(owner, self._leases)?;
        let mut local = 0usize;
        while local < rows {
            let count = (rows - local)
                .min(owner.max_new_rows)
                .min(producer.capacity);
            if count == 0 {
                return Err("selected prefill zero tile capacity".into());
            }
            let at = absolute
                .checked_add(local)
                .ok_or("selected prefill cursor overflow")?;
            let capture=crate::matched_prefill_capture::target(invocation,"selected_prepared_window",layer,
                absolute.checked_add(rows).ok_or("selected capture height")?,None)?;
            if let Some(target)=&capture {
                let start=at.max(target.row);let end=at.checked_add(count).ok_or("selected capture extent")?;
                if start<end && (start==target.row || at==absolute) {
                    // Actual incoming state before even the raw producer, while
                    // every real family/arena/source owner and lease remains held.
                    // Do not split computation to manufacture a requested cursor.
                    let state=crate::matched_probe::Target{row:target.row,directory:target.directory.clone()}.child(&format!("span-{start}-{end}"))?.child("incoming-state")?;
                    let mut registry=crate::session_cache::SequenceStateRegistry::default();
                    owner.register(&mut registry,layer)?;
                    let mut entries=Vec::new();
                    for (index,allocation) in registry.allocations().iter().enumerate() {
                        let file=format!("allocation-{index}.bin");
                        state.device(&file,allocation.ptr,allocation.storage_bytes,self.stream.handle)?;
                        entries.push(serde_json::json!({"index":index,"file":file,"name":allocation.name,
                            "dtype":allocation.dtype,"bytes":allocation.storage_bytes,"shape":allocation.shape,
                            "strides_bytes":allocation.strides_bytes,"device_ordinal":allocation.device_ordinal,
                            "actual_ptr":allocation.ptr}));
                    }
                    state.write("descriptor.json",serde_json::to_vec_pretty(&serde_json::json!({
                        "actual_state_before_producer":true,"tile_start":at,"selected_start":start,
                        "layer":layer,"heads":owner.heads,"dim":owner.dim,"stream":self.stream.handle as usize,
                        "arithmetic_sha256":owner.arithmetic_sha256,"entries":entries,
                        "diagnostic_gpu_allocation_bytes":0})).map_err(|e|e.to_string())?.as_slice())?;
                }
            }
            if let (Some(target),Some(limit))=(&digest_capture,digest_limit) {
                let end=at.checked_add(count).ok_or("selected digest end")?;
                if end>target.row {let span=target.digest_child(&format!("span-{at}-{end}"))?;capture_selected_state_sha256(owner,layer,at,"before",&span.child("before-state")?,limit,self.stream.handle)?;}
            }
            producer.enqueue(local, count, owner)?;
            if let Some(target) = capture {
                let start = at.max(target.row);
                let end = at.checked_add(count).ok_or("selected capture extent")?;
                if start < end {
                    let target = target.child(&format!("span-{start}-{end}"))?;
                    let lanes = width;
                    let raw = producer.raw.addresses();
                    let prepared = producer.prepared.addresses();
                    for (i, role) in ["q", "k", "v", "gate", "beta"].iter().enumerate() {
                        let lane = if i == 4 { owner.heads } else { lanes };
                        let elements = (end - start)
                            .checked_mul(lane)
                            .ok_or("selected captured elements")?;
                        let raw_offset = (start - absolute)
                            .checked_mul(lane)
                            .and_then(|x| x.checked_mul(2))
                            .ok_or("selected raw offset")?;
                        let ready_offset = (start - at)
                            .checked_mul(lane)
                            .and_then(|x| x.checked_mul(4))
                            .ok_or("selected prepared offset")?;
                        target.device(
                            &format!("raw-{role}.bf16"),
                            raw[i].checked_add(raw_offset as u64).ok_or("raw address")?,
                            elements.checked_mul(2).ok_or("raw bytes")?,
                            self.stream.handle,
                        )?;
                        target.device(
                            &format!("prepared-{role}.f32"),
                            prepared[i]
                                .checked_add(ready_offset as u64)
                                .ok_or("prepared address")?,
                            elements.checked_mul(4).ok_or("prepared bytes")?,
                            self.stream.handle,
                        )?;
                    }
                    target.write("descriptor.json",serde_json::to_vec_pretty(&serde_json::json!({"schema":1,"actual_selected_producer":true,"layer":layer,"absolute_start":start,"absolute_end":end,"invocation_start":absolute,"invocation_rows":rows,"tile_start":at,"tile_rows":count,"heads":owner.heads,"dim":owner.dim,"raw_dtype":"bf16","prepared_dtype":"f32","layout":"token_head_component","stream":self.stream.handle as usize,"producer_arithmetic_sha256":producer.arithmetic_sha256,"lower":producer.lower,"scale":producer.scale,"epsilon":producer.epsilon,"source_epoch":producer.source_epoch,"diagnostic_gpu_allocation_bytes":0})).map_err(|e|e.to_string())?.as_slice())?;
                }
            }
            if let (Some(target),Some(limit))=(&digest_capture,digest_limit) {
                let start=at.max(target.row);let end=at.checked_add(count).ok_or("selected digest rows")?;
                if start<end {
                    let span=target.digest_child(&format!("span-{at}-{end}"))?;let raw=producer.raw.addresses();let prepared=producer.prepared.addresses();let mut transfers=Vec::new();
                    for (i,role) in ["q","k","v","gate","beta"].iter().enumerate(){
                        let lanes=if i==4{owner.heads}else{width};
                        for (kind,ptr,size,offset_rows) in [("raw",raw[i],2usize,start-absolute),("prepared",prepared[i],4usize,start-at)] {
                            let row_bytes=lanes.checked_mul(size).ok_or("digest producer row bytes")?;
                            let bytes=(end-start).checked_mul(row_bytes).ok_or("digest producer bytes")?;
                            let offset=offset_rows.checked_mul(row_bytes).ok_or("digest producer offset")?;
                            let pointer=ptr.checked_add(u64::try_from(offset).map_err(|_|"digest producer offset exceeds u64")?).ok_or("digest producer address")?;
                            let file=format!("{kind}-{role}.sha256");let saved=span.device_row_sha256(&file,pointer,bytes,row_bytes,limit,self.stream.handle)?;
                            transfers.push(serde_json::json!({"file":file,"kind":kind,"role":role,"dtype":if size==2{"bf16"}else{"f32"},"width":lanes,"source_bytes_read":bytes,"retained_digest_bytes":saved}));
                        }
                    }
                    span.write("descriptor.json",&serde_json::to_vec_pretty(&serde_json::json!({"operation":"selected_prepared_window_sha256","actual_selected_producer":true,"layer":layer,"absolute_start":start,"absolute_end":end,"tile_start":at,"tile_rows":count,"invocation_start":absolute,"invocation_rows":rows,"heads":owner.heads,"dim":owner.dim,"encoding":"SHA256 per actual contiguous raw/prepared row","transfers":transfers,"producer_arithmetic_sha256":producer.arithmetic_sha256,"lower":producer.lower,"scale":producer.scale,"epsilon":producer.epsilon,"source_epoch":producer.source_epoch,"diagnostic_gpu_allocation_bytes":0})).map_err(|e|e.to_string())?)?;
                }
            }
            owner.prefill_in(lease, &producer.view(count)?, at as u64, count)?;
            if let (Some(target),Some(limit))=(&digest_capture,digest_limit){
                let start=at.max(target.row);let end=at.checked_add(count).ok_or("selected output digest extent")?;
                if start<end {let span=target.digest_child(&format!("span-{at}-{end}"))?;let row_bytes=width.checked_mul(2).ok_or("digest emitted row bytes")?;let offset=(start-at).checked_mul(row_bytes).ok_or("digest emitted offset")?;let bytes=(end-start).checked_mul(row_bytes).ok_or("digest emitted bytes")?;span.device_row_sha256("recurrence-output.sha256",pointer(&owner.arena.bf16_emitted).checked_add(offset as u64).ok_or("digest emitted pointer")?,bytes,row_bytes,limit,self.stream.handle)?;capture_selected_state_sha256(owner,layer,end,"after",&span.child("after-state")?,limit,self.stream.handle)?;}
            }
            if let Some(target)=crate::matched_prefill_capture::target(invocation,"selected_prepared_window",layer,
                absolute.checked_add(rows).ok_or("selected emitted height")?,None)? {
                let start=at.max(target.row);let end=at.checked_add(count).ok_or("selected emitted extent")?;
                if start<end {
                    let target=target.child(&format!("span-{start}-{end}"))?;
                    let offset=(start-at).checked_mul(width).and_then(|n|n.checked_mul(2)).ok_or("selected emitted offset")?;
                    let bytes=(end-start).checked_mul(width).and_then(|n|n.checked_mul(2)).ok_or("selected emitted bytes")?;
                    target.device("recurrence-output.bf16",pointer(&owner.arena.bf16_emitted).checked_add(offset as u64).ok_or("selected emitted pointer")?,bytes,self.stream.handle)?;
                }
            }
            let offset = local
                .checked_mul(width)
                .and_then(|v| v.checked_mul(2))
                .ok_or("selected output offset")?;
            let bytes = count
                .checked_mul(width)
                .and_then(|v| v.checked_mul(2))
                .ok_or("selected output extent")?;
            check(
                unsafe {
                    sys::lib().cuMemcpyDtoDAsync_v2(
                        output
                            .checked_add(offset as u64)
                            .ok_or("selected output address")?,
                        pointer(&owner.arena.bf16_emitted),
                        bytes,
                        self.stream.handle,
                    )
                },
                "selected prefill emitted copy",
            )?;
            local = local
                .checked_add(count)
                .ok_or("selected prefill local overflow")?;
        }
        Ok(())
    }
}
impl KdaEngineRequestStorage {
    /// This is called while recording the model's ordinary CUDA graph or its
    /// eager equivalent, after actual live convolution/gate writes. All pointer
    /// admission and buffer allocation happened at cold construction/attachment.
    pub(crate) fn enqueue_decode(
        &self,
        layer: usize,
        raw: [u64; 5],
        output: u64,
    ) -> Result<(), String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "selected decode storage concurrently leased/poisoned")?;
        if state.poisoned {
            return Err("selected decode storage poisoned".into());
        }
        let index = state
            .lifetime
            .layers
            .iter()
            .position(|&v| v == layer)
            .ok_or("selected decode missing layer")?;
        let (owner, producer) = match &state.lifetime.state {
            KdaPrefillStorageState::Bound(f) => (&f[index].decode, &f[index].decode_producer),
            KdaPrefillStorageState::Retired(f) => (&f[index].decode, &f[index].decode_producer),
            KdaPrefillStorageState::Poisoned => {
                return Err("selected decode lifetime poisoned".into())
            }
        };
        if output == 0 || producer.raw.addresses() != raw {
            return Err("selected decode raw/output binding changed".into());
        }
        // Acquire the real per-family and shared-arena leases before enqueue.
        // The model execution gate supplies ordering outside graph recording.
        // Capture changes no host phase counters or committed timeline state.

        let lease = owner.arena.lease()?;
        producer.enqueue(0, 1, owner)?;
        owner.decode_capture_in(&lease, &producer.view(1)?)?;
        let bytes = owner
            .heads
            .checked_mul(owner.dim)
            .and_then(|v| v.checked_mul(2))
            .ok_or("selected decode output extent")?;
        check(
            unsafe {
                sys::lib().cuMemcpyDtoDAsync_v2(
                    output,
                    pointer(&owner.arena.bf16_emitted),
                    bytes,
                    owner.stream.handle,
                )
            },
            "selected decode emitted copy",
        )
    }

    pub(crate) fn selected_latest_pointer(&self, layer: usize) -> Result<u64, String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "selected state held/poisoned")?;
        if state.poisoned {
            return Err("selected state poisoned".into());
        }
        let bindings = state.lifetime.decode_bindings()?;
        Ok(bindings
            .iter()
            .find(|b| b.layer == layer)
            .ok_or("selected latest layer absent")?
            .owner
            .p("latest"))
    }
    pub(crate) fn selected_registry(
        &self,
    ) -> Result<crate::session_cache::SequenceStateRegistry, String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "selected registry held/poisoned")?;
        if state.poisoned {
            return Err("selected registry poisoned".into());
        }
        state.lifetime.registry()
    }
}

impl KdaEngineRequestStorage {
    pub(crate) fn observe_actual_prepared(
        &self,
        layer: usize,
        observe: impl FnOnce([u64; 5], usize, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "selected readback storage held/poisoned")?;
        if state.poisoned {
            return Err("selected readback storage poisoned".into());
        }
        let index = state
            .lifetime
            .layers
            .iter()
            .position(|&x| x == layer)
            .ok_or("selected readback missing layer")?;
        let producer = match &state.lifetime.state {
            KdaPrefillStorageState::Bound(f) => &f[index].decode_producer,
            KdaPrefillStorageState::Retired(f) => &f[index].decode_producer,
            KdaPrefillStorageState::Poisoned => {
                return Err("selected readback lifetime poisoned".into())
            }
        };
        let view = producer.view(1)?;
        if view.schema.planes.len() != 5 {
            return Err("selected readback plane schema".into());
        }
        let addresses = producer
            .prepared
            .addresses()
            .try_into()
            .map_err(|_| "selected readback plane count")?;
        observe(addresses, producer.heads, producer.dim)
    }
}

include!("kda_prefix_digest.rs");
