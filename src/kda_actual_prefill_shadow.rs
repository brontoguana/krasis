// Explicit observed operator execution. Its outputs never enter generation.
#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct KdaActualPrefillShadowPlan {
    pub(crate) rows: usize,
    pub(crate) output: std::path::PathBuf,
}
fn shadow_save(
    dir: &std::path::Path,
    name: &str,
    bytes: &[u8],
) -> Result<serde_json::Value, String> {
    use sha2::Digest;
    use std::io::Write;
    let path = dir.join(name);
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("actual shadow evidence {}:{e}", path.display()))?;
    out.write_all(bytes)
        .and_then(|_| out.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(
        serde_json::json!({"path":path,"bytes":bytes.len(),"sha256":format!("{:x}",sha2::Sha256::digest(bytes))}),
    )
}
fn shadow_read(ptr: u64, bytes: usize) -> Result<Vec<u8>, String> {
    if ptr == 0 || bytes == 0 {
        return Err("actual shadow empty read".into());
    }
    let mut out = vec![0u8; bytes];
    check(
        unsafe { sys::lib().cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) },
        "actual shadow diagnostic read",
    )?;
    Ok(out)
}
impl KdaPrefillExecutionPhase<'_, '_> {
    pub(crate) fn observe_actual_recurrence(
        &self,
        layer: usize,
        absolute: usize,
        rows: usize,
        raw: [u64; 5],
    ) -> Result<(), String> {
        let Some(plan) = self.shadow else {
            return Ok(());
        };
        let binding = self
            .bindings
            .iter()
            .find(|b| b.layer == layer)
            .ok_or("actual shadow missing layer")?;
        let owner = binding.owner;
        let producer = binding.producer;
        self.validate_layer(layer, rows, self.stream.handle, owner.heads, owner.dim)?;
        let end = absolute
            .checked_add(rows)
            .ok_or("actual shadow end overflow")?;
        if end > plan.rows
            || producer
                .raw
                .buffers
                .iter()
                .map(|b| b.address())
                .collect::<Vec<_>>()
                != raw
        {
            return Err("actual shadow source pointers or absolute range changed".into());
        }
        let phase = StatePhase {
            owner,
            stream: self.stream,
        };
        if absolute == 0 {
            owner.reset_in_phase(&phase)?;
        }
        self.stream.drain()?;
        let dir = plan.output.join(format!("layer-{layer}-at-{absolute}"));
        std::fs::create_dir(&dir).map_err(|e| format!("actual shadow fresh chunk:{e}"))?;
        let hd = owner
            .heads
            .checked_mul(owner.dim)
            .ok_or("actual shadow geometry overflow")?;
        let mut files = Vec::new();
        let mut incoming = crate::session_cache::SequenceStateRegistry::default();
        owner.register(&mut incoming, layer)?;
        for allocation in incoming.allocations() {
            files.push(shadow_save(
                &dir,
                &format!("incoming-{}.bin", allocation.name),
                &shadow_read(allocation.ptr, allocation.storage_bytes)?,
            )?);
        }
        for (slot, name) in [
            "raw-q.bf16",
            "raw-k.bf16",
            "raw-v.bf16",
            "raw-gates.bf16",
            "raw-beta.bf16",
        ]
        .iter()
        .enumerate()
        {
            let row_elements = if slot == 4 { owner.heads } else { hd };
            let bytes = rows
                .checked_mul(row_elements)
                .and_then(|v| v.checked_mul(2))
                .ok_or("actual shadow raw bytes")?;
            files.push(shadow_save(&dir, name, &shadow_read(raw[slot], bytes)?)?);
        }
        files.push(shadow_save(
            &dir,
            "a-log.f32",
            &shadow_read(pointer(&producer.a_log), producer.a_log.len() * 4)?,
        )?);
        files.push(shadow_save(
            &dir,
            "dt-bias.f32",
            &shadow_read(pointer(&producer.bias), producer.bias.len() * 4)?,
        )?);
        let mut prepared_planes: Vec<Vec<u8>> = (0..5).map(|_| Vec::new()).collect();
        let mut emitted = Vec::new();
        let mut bf16 = Vec::new();
        let mut tiles = Vec::new();
        let mut local = 0usize;
        while local < rows {
            let count = (rows - local)
                .min(owner.max_new_rows)
                .min(producer.capacity);
            if count == 0 {
                return Err("actual shadow zero tile capacity".into());
            }
            let at = absolute
                .checked_add(local)
                .ok_or("actual shadow tile cursor")?;
            producer.enqueue(local, count, owner)?;
            self.stream.drain()?;
            for (slot, buffer) in producer.prepared.buffers.iter().enumerate() {
                let width = if slot == 4 { owner.heads } else { hd };
                let bytes = count
                    .checked_mul(width)
                    .and_then(|v| v.checked_mul(4))
                    .ok_or("actual prepared shadow extent")?;
                let live = shadow_read(buffer.address(), bytes)?;
                if live
                    .chunks_exact(4)
                    .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                {
                    return Err("actual prepared shadow nonfinite live operand".into());
                }
                prepared_planes[slot].extend_from_slice(&live);
            }
            owner.prefill_in(
                model_arena_lease(owner, self._leases)?,
                &producer.view(count)?,
                at as u64,
                count,
            )?;
            self.stream.drain()?;
            let error = shadow_read(pointer(&owner.error), 4)?;
            let pending = shadow_read(pointer(&owner.pending), 4)?;
            let cursor = shadow_read(pointer(&owner.absolute), 8)?;
            let cursor_value = u64::from_le_bytes(
                cursor
                    .as_slice()
                    .try_into()
                    .map_err(|_| "actual shadow cursor extent")?,
            );
            let pending_value = i32::from_le_bytes(
                pending
                    .as_slice()
                    .try_into()
                    .map_err(|_| "actual shadow pending extent")?,
            );
            let error_value = i32::from_le_bytes(
                error
                    .as_slice()
                    .try_into()
                    .map_err(|_| "actual shadow error extent")?,
            );
            if error_value != 0
                || cursor_value != (at + count) as u64
                || pending_value != ((at + count) % 16) as i32
            {
                return Err(format!("actual shadow cursor failure layer{layer} at{at}: error{error_value} absolute{cursor_value} pending{pending_value}"));
            }
            let f32 = shadow_read(
                owner.p("emitted"),
                count
                    .checked_mul(hd)
                    .and_then(|v| v.checked_mul(4))
                    .ok_or("actual shadow F32 extent")?,
            )?;
            if f32
                .chunks_exact(4)
                .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
            {
                return Err("actual shadow nonfinite F32 output".into());
            }
            emitted.extend_from_slice(&f32);
            let emitted_bf16 = shadow_read(
                pointer(&owner.arena.bf16_emitted),
                count
                    .checked_mul(hd)
                    .and_then(|v| v.checked_mul(2))
                    .ok_or("actual shadow BF16 extent")?,
            )?;
            for (f, b) in f32.chunks_exact(4).zip(emitted_bf16.chunks_exact(2)) {
                let bits = u32::from_le_bytes(f.try_into().unwrap());
                let rounded =
                    ((u64::from(bits) + 0x7fff + u64::from((bits >> 16) & 1)) >> 16) as u16;
                if rounded & 0x7f80 == 0x7f80
                    || u16::from_le_bytes(b.try_into().unwrap()) != rounded
                {
                    return Err("actual shadow BF16 RNE/finite mismatch".into());
                }
            }
            bf16.extend_from_slice(&emitted_bf16);
            tiles.push(serde_json::json!({"local_start":local,"absolute_start":at,"rows":count,"absolute_after":cursor_value,"pending_after":pending_value,"error":error_value}));
            local = local
                .checked_add(count)
                .ok_or("actual shadow local overflow")?;
        }
        for (slot, bytes) in prepared_planes.iter().enumerate() {
            files.push(shadow_save(&dir, &format!("prepared{slot}.f32"), bytes)?);
        }
        files.push(shadow_save(&dir, "prototype-output.f32", &emitted)?);
        files.push(shadow_save(&dir, "prototype-output.bf16", &bf16)?);
        let mut registry = crate::session_cache::SequenceStateRegistry::default();
        owner.register(&mut registry, layer)?;
        for allocation in registry.allocations() {
            files.push(shadow_save(
                &dir,
                &format!("state-{}.bin", allocation.name),
                &shadow_read(allocation.ptr, allocation.storage_bytes)?,
            )?);
        }
        let record = serde_json::json!({"layer":layer,"actual_chunk_absolute_start":absolute,"rows":rows,"complete_request_rows":plan.rows,"heads":owner.heads,"dim":owner.dim,"lower":producer.lower,"scale":producer.scale,"epsilon":producer.epsilon,"owner_arithmetic":owner.arithmetic_sha256,"producer_arithmetic":producer.arithmetic_sha256,"kernel_ptx":producer.kernels.ptx_sha256,"source_epoch":producer.source_epoch,"tiles":tiles,"state_inventory":registry.inventory(end),"files":files,"selected_for_generation":false,"ordinary_gate":"NOT_RUN"});
        let manifest = shadow_save(
            &dir,
            "REPORT.json",
            &serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?,
        )?;
        self.shadow_records.borrow_mut().push(manifest);
        Ok(())
    }
}
