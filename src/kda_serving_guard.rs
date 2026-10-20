struct ServingLogitTicket {
    ptr: u64,
    bytes: usize,
    buffer_id: u64,
    stream: usize,
}
fn serving_buffer_id(ptr: u64) -> Result<u64, String> {
    let mut id = 0u64;
    check(
        unsafe {
            sys::lib().cuPointerGetAttribute(
                (&mut id as *mut u64).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID,
                ptr,
            )
        },
        "selected logit allocation identity",
    )?;
    Ok(id)
}
impl KdaEngineRequestStorage {
    pub(crate) fn enable_serving_guard(storage: &mut Arc<Self>) -> Result<(), String> {
        let this =
            Arc::get_mut(storage).ok_or("selected guard must be constructed before publication")?;
        let state = this
            .inner
            .get_mut()
            .map_err(|_| "selected guard storage poisoned")?;
        this.serving_allocations = state.lifetime.registry()?.allocations().to_vec();
        let owners = current_sequence_owners(&state.lifetime)?;
        let mut groups: BTreeMap<String, Vec<&PreparedOwner>> = BTreeMap::new();
        for (p, _) in &owners {
            groups
                .entry(p.arithmetic_sha256.clone())
                .or_default()
                .push(p);
        }
        for owners in groups.values() {
            let first = owners[0];
            let timeline = first
                .state
                .timeline
                .lock()
                .map_err(|_| "selected guard timeline poisoned")?;
            let streams = timeline
                .as_ref()
                .ok_or("selected guard timeline absent")?
                .streams
                .clone();
            this.serving_guards
                .push(PreparedLogitGuard::new(owners, streams, this.reserve)?);
        }
        Ok(())
    }
    // Cold admission while the caller borrows its actual live logits allocation.
    // It is reissued after every scratch replacement, before graph capture.
    pub(crate) fn admit_serving_logits(
        &self,
        ptr: u64,
        bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        if self.serving_guards.is_empty() || bytes == 0 {
            return Err("selected logits guard not constructed/empty".into());
        }
        for guard in &self.serving_guards {
            let owner = guard
                .streams
                .iter()
                .find(|s| s.handle == stream)
                .ok_or("selected logits stream not admitted")?;
            native_pointer_admission(owner, ptr, bytes)?;
        }
        let ticket = ServingLogitTicket {
            ptr,
            bytes,
            buffer_id: serving_buffer_id(ptr)?,
            stream: stream as usize,
        };
        let mut tickets = self
            .serving_logit_tickets
            .lock()
            .map_err(|_| "selected logit tickets poisoned")?;
        // One current allocation per actual stream; stale scratch tickets retire.
        tickets.retain(|t| t.stream != ticket.stream);
        tickets.push(ticket);
        Ok(())
    }
    // Capture legal: identity query only, no pool/access/allocation API calls.
    // Actual caller retains the same typed logits owner through this launch.
    pub(crate) fn enqueue_serving_guard(
        &self,
        ptr: u64,
        bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let tickets = self
            .serving_logit_tickets
            .lock()
            .map_err(|_| "selected logit tickets poisoned")?;
        let ticket = tickets
            .iter()
            .find(|t| {
                t.stream == stream as usize
                    && ptr >= t.ptr
                    && ptr
                        .checked_add(bytes as u64)
                        .zip(t.ptr.checked_add(t.bytes as u64))
                        .is_some_and(|(end, limit)| end <= limit)
            })
            .ok_or("selected logit allocation not admitted")?;
        if serving_buffer_id(ptr)? != ticket.buffer_id {
            return Err("selected logit allocation remapped/replaced".into());
        }
        for guard in &self.serving_guards {
            invoke(
                &guard.device,
                stream,
                guard
                    .kernel_bank
                    .functions
                    .get("kda_prepared_logits_guard")
                    .ok_or("selected guard kernel absent")?,
                (1, 1, 1),
                1,
                0,
                &mut [
                    Parameter::Ptr(pointer(&guard.table)),
                    Parameter::Ptr(ptr),
                    Parameter::Int(
                        i32::try_from(guard.states.len())
                            .map_err(|_| "selected guard owner count")?,
                    ),
                ],
            )?;
        }
        Ok(())
    }
    pub(crate) fn validate_serving_snapshot(
        &self,
        tokens: usize,
        blobs: &[&crate::session_cache::SequenceStateBlob],
    ) -> Result<(), String> {
        let ordered = self
            .serving_allocations
            .iter()
            .map(|a| {
                let mut found = blobs.iter().filter(|b| b.allocation_name == a.name);
                let first = found
                    .next()
                    .copied()
                    .ok_or_else(|| format!("selected snapshot missing {}", a.name))?;
                if found.next().is_some() {
                    return Err(format!("selected snapshot duplicated {}", a.name));
                }
                Ok(first)
            })
            .collect::<Result<Vec<_>, String>>()?;
        validate_kda_snapshot_semantics(&self.serving_allocations, tokens, &ordered)
    }
}

impl KdaEngineRequestStorage {
    pub(crate) fn diagnostic_serving_layer(
        &self,
        layer: usize,
    ) -> Result<Vec<crate::session_cache::SequenceStateAllocation>, String> {
        let rows = self
            .serving_allocations
            .iter()
            .filter(|a| a.layer_idx == Some(layer))
            .cloned()
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return Err("selected observer layer absent".into());
        }
        Ok(rows)
    }
}
